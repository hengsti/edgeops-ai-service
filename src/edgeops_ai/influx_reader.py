"""Read bounded Advisor inputs using the existing InfluxDB aggregate contract."""

import asyncio
import csv
import io
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta

import httpx

from edgeops_ai.advisor import COUNTERS, DAY, DeviceSample, PipelineSample, check_time, day_start
from edgeops_ai.maintenance import MaintenanceDay, summarize_day
from edgeops_ai.resources import RESOURCE_FIELDS, STEP, ResourceSample
from edgeops_ai.schemas import SourceIdentity
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
MEMORY_FIELDS = {
    key: value
    for key, value in DEVICE_FIELDS.items()
    if value in {"free_mem", "uptime", "uptime_resets"}
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
        resource_history_hours: int = 24,
    ):
        self._history: dict[datetime, list[MaintenanceDay]] = {}
        self.resource_history_hours = resource_history_hours
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

    async def _devices(
        self,
        start: datetime,
        stop: datetime,
        fields: dict[tuple[str, str], str],
        marker: str = "",
    ) -> list[DeviceSample]:
        # Static comparisons let InfluxDB filter fields in storage; contains() cannot push down.
        field_filter = " or ".join(
            f"r._field == {json.dumps(field)}" for field in sorted({field for _, field in fields})
        )
        measurement_filter = " or ".join(
            f"r._measurement == {json.dumps(name)}" for name in sorted({name for name, _ in fields})
        )
        sensor_query = marker + (
            'from(bucket: "sensors_1m") '
            f"|> range(start: {start.isoformat()}, stop: {stop.isoformat()}) "
            f"|> filter(fn: (r) => {measurement_filter}) "
            f"|> filter(fn: (r) => {field_filter}) "
            '|> filter(fn: (r) => r.device_id != "stresstest" and r.device_id != "livingroom-01") '
            '|> keep(columns: ["_time", "_measurement", "_field", "_value", "device_id", "room"])'
        )
        devices: dict[tuple[str, datetime], DeviceSample] = {}
        sensor_minutes = set()
        # Large query normalization must let HTTP deadlines and cancellation run on this loop.
        for index, row in enumerate(await self._query(sensor_query)):
            if index % 128 == 0:
                await asyncio.sleep(0)
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
        return [devices[key] for key in sorted(devices)]

    async def read(self, start: datetime, stop: datetime) -> AdvisorWindow:
        check_time(start)
        check_time(stop)
        if not start < stop <= start + DAY:
            raise ValueError("Advisor reads require an increasing window of at most one day")
        devices = await self._devices(start, stop, {**DEVICE_FIELDS, **CLIMATE_FIELDS})
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
        for index, row in enumerate(await self._query(pipeline_query)):
            if index % 128 == 0:
                await asyncio.sleep(0)
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
            pipeline.append(
                PipelineSample(
                    at,
                    entity,
                    row["_field"],
                    value,
                    source=SourceIdentity(
                        kind="pipeline",
                        measurement=row["_measurement"],
                        host=dimensions.get("host"),
                        container_name=dimensions.get("container_name"),
                        dimensions=dimensions,
                    ),
                )
            )
        resource_start = (
            start
            if self.resource_history_hours == 24
            else stop - timedelta(hours=self.resource_history_hours)
        )
        resources = await self._resources(resource_start, stop)
        history_stop = day_start(stop)
        history_start = history_stop - 30 * DAY
        self._history = {
            day: rows for day, rows in self._history.items() if history_start <= day < history_stop
        }
        # shortcut: completed-day summaries are process-local; restart after historical backfills.
        for offset in range(30):
            day = history_start + offset * DAY
            if day not in self._history:
                memory = await self._devices(day, day + DAY, MEMORY_FIELDS, "// advisor-history\n")
                disk = await self._resources(day, day + DAY, history=True)
                self._history[day] = await asyncio.to_thread(summarize_day, memory, disk, day)
        return AdvisorWindow(
            start,
            stop,
            devices,
            pipeline,
            resources=tuple(resources),
            history=tuple(row for day in sorted(self._history) for row in self._history[day]),
            history_start=history_start,
        )

    async def _resources(
        self,
        start: datetime,
        stop: datetime,
        *,
        history: bool = False,
    ) -> list[ResourceSample]:
        stop = stop.replace(minute=stop.minute - stop.minute % 5, second=0, microsecond=0)
        if start >= stop:
            return []
        selected = {"disk": {"used_percent_mean"}} if history else RESOURCE_FIELDS
        filters = " or ".join(
            f"(r._measurement == {json.dumps(measurement)} and ("
            + " or ".join(f"r._field == {json.dumps(field)}" for field in sorted(fields))
            + "))"
            for measurement, fields in selected.items()
        )
        query = ("// advisor-history\n" if history else "") + (
            'from(bucket: "metrics_5m") '
            f"|> range(start: {start.isoformat()}, stop: {stop.isoformat()}) "
            f"|> filter(fn: (r) => {filters}) "
            '|> filter(fn: (r) => (r._measurement != "cpu" or r.cpu == "cpu-total") '
            'and (r._measurement != "disk" or r.path == "/")) '
            '|> drop(columns: ["_start", "_stop"])'
        )
        unique: dict[tuple[str, datetime, str, str], ResourceSample] = {}
        for index, row in enumerate(await self._query(query)):
            if index % 128 == 0:
                await asyncio.sleep(0)
            measurement, metric = row["_measurement"], row["_field"]
            if metric not in selected.get(measurement, set()):
                raise ValueError("Unexpected resource field")
            at = observation_time(row, start, stop)
            host = row.get("host", "")
            if not host or at.minute % 5 or at.second or at.microsecond or at + STEP > stop:
                raise ValueError("Resources need identity and completed five-minute timestamps")
            if measurement.startswith("docker_container_") and not row.get("container_name"):
                raise ValueError("Container resources need container_name")
            if (measurement == "cpu" and row.get("cpu") != "cpu-total") or (
                measurement == "disk" and row.get("path") != "/"
            ):
                raise ValueError("Unexpected host CPU or disk source")
            value: float | str
            if metric in {"health_status_last", "oomkilled_last"}:
                value = row["_value"]
                if metric == "oomkilled_last" and value not in {"true", "false"}:
                    raise ValueError("Invalid container OOM flag")
            else:
                if row["_value_type"] not in {"double", "long", "unsignedLong"}:
                    raise ValueError("Expected numeric resource aggregate")
                value = float(row["_value"])
                if (
                    not math.isfinite(value)
                    or value < 0
                    or (
                        measurement != "docker_container_cpu"
                        and "percent" in metric
                        and value > 100
                    )
                ):
                    raise ValueError("Invalid resource aggregate value")
            dimensions = {
                key: value
                for key, value in row.items()
                if value and key not in {"", "result", "table"} and not key.startswith("_")
            }
            entity = json.dumps(dimensions, sort_keys=True)
            key = entity, at, measurement, metric
            sample = ResourceSample(at, entity, measurement, metric, value, dimensions)
            if key in unique and unique[key] != sample:
                raise ValueError("Conflicting resource aggregates")
            unique[key] = sample
        return [unique[key] for key in sorted(unique)]

    async def close(self) -> None:
        await self.client.aclose()
