"""Read bounded Advisor inputs using the existing InfluxDB aggregate contract."""

import asyncio
import csv
import io
import json
import math
from dataclasses import replace
from datetime import datetime

import httpx

from edgeops_ai.advisor import COUNTERS, DAY, DeviceSample, PipelineSample, check_time
from edgeops_ai.state_report import AdvisorWindow

MAX_RESPONSE_BYTES = 32 * 1024 * 1024

DEVICE_FIELDS = {
    ("bme680", "temp_c_count"): "sensor_samples",
    ("bme680", "pressure_hpa_mean"): "pressure_hpa",
    ("bme680", "iaq_score_mean"): "iaq_mean",
    ("bme680", "iaq_score_min"): "iaq_min",
    ("bme680", "warmed_up_last"): "warmed_up",
    ("device_status", "rssi_last"): "rssi",
    ("device_status", "free_mem_min"): "free_mem",
    ("device_status", "uptime_last"): "uptime",
    ("device_status", "uptime_resets"): "uptime_resets",
    **{("device_status", f"{name}_last"): name for name in COUNTERS},
    **{("device_status", f"{name}_resets"): f"{name}_resets" for name in COUNTERS},
}
CLIMATE_FIELDS = {
    ("bme680", "temp_c_mean"): "temperature_mean",
    ("bme680", "rel_hum_perc_mean"): "humidity_mean",
    ("bme680", "iaq_score_mean"): "iaq_mean",
    ("bme680", "gas_ohm_mean"): "gas_mean",
    ("bme680", "warmed_up_last"): "warmed_up",
    ("bme680", "time_valid_last"): "time_valid",
}
PIPELINE_FIELDS = {
    "dlq_messages_published_total",
    "ingest_incoming_invalid_json_total",
    "ingest_validate_raw_failed_total",
    "emqx_messages_dropped",
}


def records(body: bytes) -> list[dict[str, str]]:
    header = datatypes = None
    result = []
    for row in csv.reader(io.StringIO(body.decode("utf-8-sig"), newline="")):
        if row and row[0] == "#datatype":
            datatypes = row
            continue
        if not row or row[0].startswith("#"):
            continue
        if "error" in row and "reference" in row:
            raise ValueError("InfluxDB query returned a stream error")
        if {"_time", "_measurement", "_field", "_value"}.issubset(row):
            header = row
        elif header and datatypes and len(row) == len(header) == len(datatypes):
            result.append(
                {
                    **dict(zip(header, row, strict=True)),
                    "_value_type": datatypes[header.index("_value")],
                }
            )
        else:
            raise ValueError("Invalid annotated InfluxDB CSV")
    return result


def observation_time(row: dict[str, str], start: datetime, stop: datetime) -> datetime:
    at = datetime.fromisoformat(row["_time"].replace("Z", "+00:00"))
    check_time(at)
    if not start <= at < stop:
        raise ValueError("InfluxDB observation outside the requested interval")
    return at


class HttpInfluxReader:
    def __init__(
        self,
        url: str,
        org: str,
        token: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=10,
            transport=transport,
            params={"org": org},
            headers={"Authorization": f"Token {token}", "Accept": "application/csv"},
        )

    async def _query(self, query: str) -> list[dict[str, str]]:
        body = bytearray()
        async with self.client.stream(
            "POST",
            "/api/v2/query",
            json={
                "query": query,
                "dialect": {"annotations": ["datatype"]},
            },
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise ValueError("Advisor InfluxDB response exceeds the size limit")
                body.extend(chunk)
        return await asyncio.to_thread(records, bytes(body))

    async def read(self, start: datetime, stop: datetime) -> AdvisorWindow:
        check_time(start)
        check_time(stop)
        if not start < stop <= start + DAY:
            raise ValueError("Advisor reads require an increasing window of at most one day")
        fields = {**DEVICE_FIELDS, **CLIMATE_FIELDS}
        # Static comparisons let InfluxDB filter fields in storage; contains() cannot push down.
        field_filter = " or ".join(
            f"r._field == {json.dumps(field)}" for field in sorted({field for _, field in fields})
        )
        sensor_query = (
            'from(bucket: "sensors_1m") '
            f"|> range(start: {start.isoformat()}, stop: {stop.isoformat()}) "
            '|> filter(fn: (r) => r._measurement == "bme680" or r._measurement == "device_status") '
            f"|> filter(fn: (r) => {field_filter}) "
            '|> filter(fn: (r) => r.device_id != "stresstest" and r.device_id != "livingroom-01") '
            '|> keep(columns: ["_time", "_measurement", "_field", "_value", "device_id", "room"])'
        )
        devices: dict[tuple[str, datetime], DeviceSample] = {}
        sensor_minutes = set()
        for row in await self._query(sensor_query):
            name = fields.get((row["_measurement"], row["_field"]))
            if name is None or row.get("device_id") in {"stresstest", "livingroom-01"}:
                continue
            at = observation_time(row, start, stop)
            device = row.get("device_id", "")
            if not device or at.second or at.microsecond:
                raise ValueError("Sensor aggregates need identity and minute timestamps")
            if name in {"warmed_up", "time_valid"}:
                value: float | str = row["_value"]
                if value not in {"true", "false"}:
                    raise ValueError("Invalid sensor quality flag")
            else:
                if row["_value_type"] not in {"double", "long", "unsignedLong"}:
                    raise ValueError("Expected numeric sensor aggregate")
                value = float(row["_value"])
                if not math.isfinite(value) or (name != "rssi" and value < 0):
                    raise ValueError("Invalid sensor aggregate value")
                if name in {"humidity_mean", "iaq_mean", "iaq_min"} and value > 100:
                    raise ValueError("Sensor percentage/score exceeds its documented range")
            key = device, at
            sample = devices.setdefault(key, DeviceSample(at, device, {}))
            room = row.get("room", "")
            if room:
                if sample.room and sample.room != room:
                    raise ValueError("Conflicting historical rooms in one device minute")
                sample = devices[key] = replace(sample, room=room)
            if name in sample.values and sample.values[name] != value:
                raise ValueError("Conflicting sensor aggregates")
            sample.values[name] = value
            if row["_measurement"] == "bme680":
                sensor_minutes.add(key)
        for key in sensor_minutes:
            devices[key].values.setdefault("sensor_samples", 1)
        pipeline_filter = " or ".join(
            f"r._field == {json.dumps(field)}" for field in sorted(PIPELINE_FIELDS)
        )
        pipeline_query = (
            'from(bucket: "metrics") '
            f"|> range(start: {start.isoformat()}, stop: {stop.isoformat()}) "
            '|> filter(fn: (r) => r._measurement == "prometheus") '
            f"|> filter(fn: (r) => {pipeline_filter}) "
            '|> drop(columns: ["_start", "_stop"])'
        )
        pipeline = []
        for row in await self._query(pipeline_query):
            if row["_measurement"] != "prometheus" or row["_field"] not in PIPELINE_FIELDS:
                continue
            at = observation_time(row, start, stop)
            if row["_value_type"] not in {"double", "long", "unsignedLong"}:
                raise ValueError("Expected numeric pipeline counter")
            value = float(row["_value"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid pipeline counter")
            dimensions = {
                key: value
                for key, value in row.items()
                if value and key not in {"", "result", "table"} and not key.startswith("_")
            }
            entity = json.dumps({"measurement": row["_measurement"], **dimensions}, sort_keys=True)
            pipeline.append(PipelineSample(at, entity, row["_field"], value))
        return AdvisorWindow(start, stop, [devices[key] for key in sorted(devices)], pipeline)

    async def close(self) -> None:
        await self.client.aclose()
