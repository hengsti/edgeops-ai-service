import asyncio
import csv
import io
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from edgeops_ai.influx_reader import HttpInfluxReader

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
            if "sensors_1m" in request.content.decode():
                body = annotated(
                    [
                        sensor(),
                        sensor("temp_c_count", 30),
                        sensor("warmed_up_last", "false", _value_type="boolean"),
                    ]
                )
            else:
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
            assert len(requests) == 2
            assert window.devices[0].values["sensor_samples"] == 30
            assert window.devices[0].values["humidity_mean"] == 65
            assert window.devices[0].values["warmed_up"] == "false"
            assert window.devices[0].room == "kitchen"
            assert window.pipeline[0].value == 3
            assert '"host": "pi"' in window.pipeline[0].entity_id
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
