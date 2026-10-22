import asyncio
import csv
import io
import json
import re
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from edgeops_ai.influx_reader import (
    CLIMATE_FIELDS,
    DEVICE_FIELDS,
    PIPELINE_FIELDS,
    HttpInfluxReader,
)

START = datetime(2026, 10, 1, tzinfo=UTC)
STOP = START + timedelta(days=1)


def annotated(rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    columns = [
        "",
        "result",
        "table",
        "_time",
        "_measurement",
        "_field",
        "_value",
        "device_id",
        "room",
        "host",
        "cpu",
        "path",
        "device",
        "container_name",
        "container_version",
    ]
    for row in rows:
        datatypes = ["string"] * len(columns)
        datatypes[6] = row.get("_value_type", "double")
        writer.writerow(["#datatype", *datatypes[1:]])
        writer.writerow(columns)
        writer.writerow([row.get(column, "") for column in columns])
    return output.getvalue()


def sensor(field="rel_hum_perc_mean", value=65, **changes):
    return {
        "_time": START.isoformat(),
        "_measurement": "bme680",
        "_field": field,
        "_value": value,
        "device_id": "sensor",
        "room": "kitchen",
        **changes,
    }


def test_reader_preserves_fields_flags_rooms_and_pipeline_series():
    async def scenario():
        requests = []

        def handler(request):
            requests.append(request)
            assert request.headers["authorization"] == "Token private-token"
            assert request.url.params["org"] == "smarthome"
            query = json.loads(request.content)["query"]
            # contains() prevents InfluxDB 2.7 from pushing field selection into storage.
            assert "contains(" not in query
            selected_fields = set(re.findall(r'r\._field == "([^"]+)"', query))
            if "advisor-history" in query or 'bucket: "metrics_5m"' in query:
                return httpx.Response(200, text=annotated([]))
            if "sensors_1m" in query:
                assert selected_fields == {field for _, field in DEVICE_FIELDS | CLIMATE_FIELDS}
                body = annotated(
                    [
                        sensor(),
                        sensor("temp_c_count", 30),
                        sensor("warmed_up_last", "false", _value_type="boolean"),
                    ]
                )
            else:
                assert selected_fields == PIPELINE_FIELDS
                body = annotated(
                    [
                        {
                            "_time": START.isoformat(),
                            "_measurement": "prometheus",
                            "_field": "dlq_messages_published_total",
                            "_value": 3,
                            "host": "pi",
                        }
                    ]
                )
            return httpx.Response(200, text=body)

        reader = HttpInfluxReader(
            "http://influx", "smarthome", "private-token", transport=httpx.MockTransport(handler)
        )
        try:
            window = await reader.read(START, STOP)
            assert len(requests) == 63
            assert window.devices[0].values["sensor_samples"] == 30
            assert window.devices[0].values["humidity_mean"] == 65
            assert window.devices[0].values["warmed_up"] == "false"
            assert window.devices[0].room == "kitchen"
            assert window.pipeline[0].value == 3
            assert '"host": "pi"' in window.pipeline[0].entity_id
            assert window.history_start == STOP - timedelta(days=30)
            assert len(reader._history) == 30
            await reader.read(START, STOP)
            assert len(requests) == 66  # Only current detail/resource queries on a warm cache.
        finally:
            await reader.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "problem",
    [
        "nan",
        "outside",
        "cadence",
        "conflict",
        "room",
        "unannotated",
        "stream-error",
        "html",
        "http",
        "oversized",
    ],
)
def test_reader_rejects_invalid_or_failed_sources(problem, monkeypatch):
    from edgeops_ai import influx_reader

    rows = [sensor()]
    body = None
    if problem == "nan":
        rows[0]["_value"] = "nan"
    elif problem == "outside":
        rows[0]["_time"] = STOP.isoformat()
    elif problem == "cadence":
        rows[0]["_time"] = (START + timedelta(seconds=1)).isoformat()
    elif problem == "conflict":
        rows.append(sensor(value=90))
    elif problem == "room":
        rows.append(sensor("gas_ohm_mean", 30_000, room="office"))
    elif problem == "unannotated":
        body = annotated(rows).split("\n", 1)[1]
    elif problem == "stream-error":
        body = ",error,reference\n,query failed,123\n"
    elif problem == "html":
        body = "<html>proxy failure</html>"
    elif problem == "oversized":
        monkeypatch.setattr(influx_reader, "MAX_RESPONSE_BYTES", 10)

    async def scenario():
        reader = HttpInfluxReader(
            "http://influx",
            "smarthome",
            "token",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    503 if problem == "http" else 200,
                    text=body if body is not None else annotated(rows),
                )
            ),
        )
        try:
            with pytest.raises((ValueError, httpx.HTTPError)):
                await reader.read(START, STOP)
        finally:
            await reader.close()

    asyncio.run(scenario())


def resource(**changes):
    return {
        "_time": START.isoformat(),
        "_measurement": "disk",
        "_field": "used_percent_mean",
        "_value": 45,
        "host": "pi",
        "path": "/",
        "device": "mmcblk0p2",
        **changes,
    }


def test_resource_reader_preserves_tags_types_cadence_and_partial_windows():
    async def scenario():
        rows = [
            resource(),
            resource(_measurement="cpu", _field="usage_active_mean", cpu="cpu-total"),
            resource(
                _measurement="docker_container_health",
                _field="health_status_last",
                _value="unhealthy",
                _value_type="string",
                container_name="influx",
                container_version="v1",
            ),
        ]
        reader = HttpInfluxReader(
            "http://influx",
            "smarthome",
            "token",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=annotated(rows))
            ),
        )
        try:
            result = await reader._resources(START, START + timedelta(minutes=8))
            assert len(result) == 3
            disk = next(row for row in result if row.measurement == "disk")
            assert json.loads(disk.entity_id)["device"] == "mmcblk0p2"
            health = next(row for row in result if row.metric == "health_status_last")
            assert health.value == "unhealthy" and health.dimensions["container_version"] == "v1"
            assert await reader._resources(START, START + timedelta(minutes=1)) == []
        finally:
            await reader.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "problem",
    ["nan", "range", "host", "cadence", "future", "conflict", "container-name", "oom-type"],
)
def test_resource_reader_rejects_untrustworthy_evidence(problem):
    rows = [resource()]
    if problem == "nan":
        rows[0]["_value"] = "nan"
    elif problem == "range":
        rows[0]["_value"] = 101
    elif problem == "host":
        rows[0]["host"] = ""
    elif problem == "cadence":
        rows[0]["_time"] = (START + timedelta(minutes=1)).isoformat()
    elif problem == "future":
        rows[0]["_time"] = STOP.isoformat()
    elif problem == "conflict":
        rows.append(resource(_value=55))
    else:
        rows = [
            resource(
                _measurement="docker_container_status",
                _field="oomkilled_last",
                _value="wrong" if problem == "oom-type" else "true",
                _value_type="boolean",
                container_name="influx" if problem == "oom-type" else "",
            )
        ]

    async def scenario():
        reader = HttpInfluxReader(
            "http://influx",
            "smarthome",
            "token",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=annotated(rows))
            ),
        )
        try:
            with pytest.raises(ValueError):
                await reader._resources(START, STOP)
        finally:
            await reader.close()

    asyncio.run(scenario())


def test_history_cache_rolls_forward_and_resumes_after_failed_backfill():
    async def scenario():
        requests, fail = [], True

        def handler(request):
            nonlocal fail
            query = json.loads(request.content)["query"]
            if "advisor-history" in query:
                requests.append(query)
                if len(requests) == 5 and fail:
                    fail = False
                    return httpx.Response(503)
            return httpx.Response(200, text=annotated([]))

        reader = HttpInfluxReader(
            "http://influx", "smarthome", "token", transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(httpx.HTTPError):
                await reader.read(START, STOP)
            assert len(reader._history) == 2
            await reader.read(START, STOP)
            assert len(requests) == 61
            await reader.read(START + timedelta(days=1), STOP + timedelta(days=1))
            assert len(requests) == 63 and len(reader._history) == 30
            assert min(reader._history) == STOP - timedelta(days=29)
        finally:
            await reader.close()

    asyncio.run(scenario())
