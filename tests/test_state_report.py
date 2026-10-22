import asyncio
import json
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient

from edgeops_ai.advisor import MINUTE, Advice, DeviceSample, PipelineSample
from edgeops_ai.main import create_app
from edgeops_ai.settings import Settings
from edgeops_ai.state_report import AdvisorWindow, StateReporter
from test_influx_reader import annotated


class FixtureReader:
    calls = 0

    async def read(self, start, stop):
        self.calls += 1
        devices = [
            DeviceSample(
                stop - (120 - index) * MINUTE,
                "sensor",
                {
                    "sensor_samples": 30,
                    "humidity_mean": 65,
                    "temperature_mean": 20,
                    "iaq_mean": 90,
                    "warmed_up": "true",
                    "sensor_read_failures_total": index,
                },
                "kitchen",
            )
            for index in range(120)
        ]
        pipeline = [
            PipelineSample(stop - 2 * MINUTE, "ingest", "dlq_messages_published_total", 0),
            PipelineSample(stop - MINUTE, "ingest", "dlq_messages_published_total", 2),
        ]
        return AdvisorWindow(start, stop, devices, pipeline)


def settings(**changes):
    return Settings(_env_file=None, collector_polling_enabled=False, **changes)


def test_unconfigured_advisor_is_unavailable_and_incident_api_stays_healthy():
    with TestClient(create_app(settings())) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 503
        assert client.get("/health").json()["status"] == "ok"


def test_state_report_contract_uses_real_rules_and_cache():
    reader = FixtureReader()
    with TestClient(create_app(settings(), advisor_reader=reader)) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["schema_version"] == "1.0"
        assert report["overall"] == "attention"
        assert report["status"] == "partial"
        assert report["coverage"]["maintenance"] == "no-data"
        assert report["maintenance_estimates"] == []
        assert {item["category"] for item in report["findings"]} == {
            "device",
            "pipeline",
            "climate",
        }
        assert len({item["id"] for item in report["findings"]}) == len(report["findings"])
        for item in report["findings"]:
            assert item["subject"] and item["recommendation"] and item["detector"]
            assert item["evidence"] and item["confidence"] is None
            assert item["window_start"] < item["window_stop"] <= report["observed_at"]
        assert client.get("/v1/state-report").json() == report
        assert reader.calls == 1
        schema = client.get("/openapi.json").json()
        assert "StateReport" in schema["components"]["schemas"]
        assert "503" in schema["paths"]["/v1/state-report"]["get"]["responses"]


@pytest.mark.parametrize("failure", [False, True])
def test_live_weather_is_shared_and_outage_does_not_stop_indoor_advice(failure):
    from datetime import UTC

    requests = []

    def handler(request):
        requests.append(str(request.url))
        if failure:
            return httpx.Response(500, text="private-weather-key")
        at = datetime.now(UTC).replace(second=0, microsecond=0) - 2 * MINUTE
        return httpx.Response(
            200,
            json={
                "dt": at.timestamp(),
                "main": {"temp": 5, "humidity": 90},
            },
        )

    class RoomsReader(FixtureReader):
        async def read(self, start, stop):
            window = await super().read(start, stop)
            return replace(
                window,
                devices=[
                    *window.devices,
                    *(
                        replace(row, device_id="office-sensor", room="office")
                        for row in window.devices
                    ),
                ],
            )

    app = create_app(
        settings(advisor_weather_postcode="1030,AT", advisor_weather_api_key="private-weather-key"),
        advisor_reader=RoomsReader(),
        weather_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200
        assert "private-weather-key" not in response.text
        climate = [item for item in response.json()["findings"] if item["category"] == "climate"]
        assert len(climate) == 2
        for item in climate:
            evidence = {entry["metric"]: entry["value"] for entry in item["evidence"]}
            assert evidence["outdoor_moisture_suitability"] == ("unknown" if failure else "drier")
        app.state.state_reporter._expires = 0
        assert client.get("/v1/state-report").status_code == 200
        assert len(requests) == 1


class SequenceDetector:
    category = "host"

    def __init__(self):
        self.present = True

    def detect(self, window):
        if not self.present:
            return []
        item = Advice(
            "host.fixture",
            "pi",
            window.start,
            window.stop,
            window.stop,
            "Host needs attention",
            "Inspect host resources",
            {"value": 90},
            severity="critical",
            detector="rule:fixture@1",
        )
        return [item, replace(item, stop=window.stop - MINUTE, observed_at=window.stop - MINUTE)]


def test_custom_detector_dedup_first_seen_hysteresis_and_single_refresh():
    async def scenario():
        reader, detector = FixtureReader(), SequenceDetector()
        reporter = StateReporter(reader, (detector,), refresh_seconds=0)
        first = await reporter.report()
        assert len(first.findings) == 1 and first.overall == "critical"
        second = await reporter.report()
        assert second.findings[0].id == first.findings[0].id
        assert second.findings[0].first_seen == first.findings[0].first_seen
        detector.present = False
        assert (await reporter.report()).findings
        assert (await reporter.report()).findings == []
        reporter.refresh_seconds = 60
        before = reader.calls
        await asyncio.gather(*(reporter.report() for _ in range(4)))
        assert reader.calls == before + 1

    asyncio.run(scenario())


def test_failed_refresh_does_not_serve_an_expired_report_or_leak_errors():
    class BrokenReader(FixtureReader):
        fail = False

        async def read(self, start, stop):
            if self.fail:
                raise ValueError("private-token-and-upstream-body")
            return await super().read(start, stop)

    reader = BrokenReader()
    app = create_app(settings(), advisor_reader=reader)
    with TestClient(app) as client:
        assert client.get("/v1/state-report").status_code == 200
        reader.fail = True
        app.state.state_reporter._expires = 0
        response = client.get("/v1/state-report")
        assert response.status_code == 503
        assert "private-token" not in response.text


def test_configured_reader_is_created_and_closed_with_application(monkeypatch):
    from edgeops_ai import influx_reader

    created = []

    class ManagedReader(FixtureReader):
        closed = False

        def __init__(self, url, org, token):
            created.append((url, org, token))

        async def close(self):
            self.closed = True

    monkeypatch.setattr(influx_reader, "HttpInfluxReader", ManagedReader)
    app = create_app(settings(advisor_influx_url="http://influx", advisor_influx_token="secret"))
    with TestClient(app) as client:
        assert client.get("/v1/state-report").status_code == 200
        reader = app.state.state_reporter.reader
    assert reader.closed
    assert created == [("http://influx/", "smarthome", "secret")]


def test_empty_data_does_not_return_an_overall_ok():
    class EmptyReader:
        async def read(self, start, stop):
            return AdvisorWindow(start, stop, [], [])

    with TestClient(create_app(settings(), advisor_reader=EmptyReader())) as client:
        report = client.get("/v1/state-report").json()
        assert report["findings"] == []
        assert report["overall"] == "attention"
        assert report["coverage"]["device"] == "no-data"


def test_failed_reads_are_not_retried_for_every_dashboard_request():
    class FailingReader:
        calls = 0

        async def read(self, start, stop):
            self.calls += 1
            raise ValueError("upstream failure")

    reader = FailingReader()
    with TestClient(create_app(settings(), advisor_reader=reader)) as client:
        assert all(client.get("/v1/state-report").status_code == 503 for _ in range(3))
    assert reader.calls == 1


def test_settings_require_paired_credentials_and_positive_bounded_refresh():
    with pytest.raises(ValueError):
        settings(advisor_influx_url="http://influx")
    with pytest.raises(ValueError):
        settings(advisor_influx_token="secret")
    with pytest.raises(ValueError):
        settings(advisor_refresh_seconds=0)
    configured = settings(advisor_influx_url="http://influx", advisor_influx_token="secret")
    assert "secret" not in repr(configured)


def test_enabled_advisor_does_not_import_training_or_ml_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import asyncio, sys\n"
                "from edgeops_ai.main import create_app\n"
                "from edgeops_ai.settings import Settings\n"
                "app = create_app(Settings(_env_file=None, collector_polling_enabled=False, "
                "prediction_backend='rules', advisor_influx_url='http://unused', "
                "advisor_influx_token='fixture-token'))\n"
                "async def check():\n"
                "    async with app.router.lifespan_context(app):\n"
                "        assert app.state.state_reporter is not None\n"
                "        from datetime import UTC, datetime, timedelta\n"
                "        from edgeops_ai.maintenance import forecast\n"
                "        base = datetime(2026, 10, 1, tzinfo=UTC)\n"
                "        points = [(base + timedelta(days=i), 60 + i) for i in range(7)]\n"
                "        assert forecast(points, 'disk_used_percent', "
                "points[-1][0])['advice_due']\n"
                "        blocked = {'joblib','numpy','scipy','sklearn','pyarrow'}\n"
                "        assert not blocked.intersection(sys.modules)\n"
                "asyncio.run(check())\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_annotated_influx_responses_reach_the_versioned_endpoint():
    from edgeops_ai.influx_reader import HttpInfluxReader

    requests = []

    def handler(request):
        query = json.loads(request.content)["query"]
        stop = datetime.fromisoformat(re.search(r"stop: ([^)]+)", query).group(1))
        requests.append(query)
        if "advisor-history" in query or 'bucket: "metrics_5m"' in query:
            return httpx.Response(200, text=annotated([]))
        if "sensors_1m" in query:
            rows = [
                {
                    "_time": (stop - (120 - index) * MINUTE).isoformat(),
                    "_measurement": "bme680",
                    "_field": "rel_hum_perc_mean",
                    "_value": 65,
                    "device_id": "sensor",
                    "room": "kitchen",
                }
                for index in range(120)
            ]
        else:
            rows = [
                {
                    "_time": (stop - (2 - index) * MINUTE).isoformat(),
                    "_measurement": "prometheus",
                    "_field": "dlq_messages_published_total",
                    "_value": index,
                    "host": "pi",
                }
                for index in range(2)
            ]
        return httpx.Response(200, text=annotated(rows))

    reader = HttpInfluxReader(
        "http://influx", "smarthome", "fixture-token", transport=httpx.MockTransport(handler)
    )
    try:
        with TestClient(create_app(settings(), advisor_reader=reader)) as client:
            response = client.get("/v1/state-report")
            assert response.status_code == 200, response.text
            report = response.json()
            assert {item["category"] for item in report["findings"]} == {"pipeline", "climate"}
            assert report["schema_version"] == "1.0"
            assert len(requests) == 63
    finally:
        asyncio.run(reader.close())


@pytest.mark.parametrize(
    "memory_threshold,horizon_days,due",
    [
        (65536, 30, True),
        (50000, 30, False),
        (65536, 10, False),
    ],
)
def test_resource_and_maintenance_endpoint_contract_includes_nondue_and_unknown_estimates(
    memory_threshold,
    horizon_days,
    due,
):
    from edgeops_ai.advisor import DAY, day_start
    from edgeops_ai.maintenance import MaintenanceDay
    from edgeops_ai.resources import STEP, ResourceSample

    class HistoryReader(FixtureReader):
        async def read(self, start, stop):
            window = await super().read(start, stop)
            base = day_start(stop) - 9 * DAY
            history = []
            for index in range(9):
                at = base + index * DAY
                first = DeviceSample(at, "sensor", {"uptime": index * 86400 + 1000}, "office")
                last = replace(
                    first, at=at + DAY - MINUTE, values={"uptime": (index + 1) * 86400 + 940}
                )
                history.extend(
                    [
                        MaintenanceDay(
                            at,
                            "free_mem_bytes",
                            "sensor",
                            100_000 - index * 1000,
                            100_000 - index * 1000,
                            720,
                            first=first,
                            last=last,
                        ),
                        MaintenanceDay(
                            at, "disk_used_percent", "pi/root", 40 + index / 10, 40, 144
                        ),
                    ]
                )
            history.append(
                MaintenanceDay(
                    base + 8 * DAY, "free_mem_bytes", "new-sensor", 200_000, 200_000, 720
                )
            )
            resource_stop = stop.replace(minute=stop.minute - stop.minute % 5)
            resources = [
                ResourceSample(
                    resource_stop - (3 - index) * STEP, "pi", "cpu", "usage_active_mean", 95
                )
                for index in range(3)
            ]
            resources.append(
                ResourceSample(
                    resource_stop - STEP,
                    "pi/influx",
                    "docker_container_health",
                    "health_status_last",
                    "unhealthy",
                )
            )
            return replace(
                window, history_start=base, history=tuple(history), resources=tuple(resources)
            )

    app = create_app(
        settings(
            advisor_memory_threshold=memory_threshold, advisor_maintenance_horizon_days=horizon_days
        ),
        advisor_reader=HistoryReader(),
    )
    with TestClient(app) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["schema_version"] == "1.0"
        assert report["coverage"]["host"] == report["coverage"]["maintenance"] == "available"
        categories = {"device", "pipeline", "climate", "host"}
        if due:
            categories.add("maintenance")
        assert {item["category"] for item in report["findings"]} == categories
        estimates = {item["entity_id"]: item for item in report["maintenance_estimates"]}
        assert estimates["sensor"]["advice_due"] is due
        assert estimates["pi/root"]["eta_days"] > 30
        assert estimates["pi/root"]["advice_due"] is False
        assert estimates["new-sensor"]["status"] == "insufficient-history"
        assert estimates["new-sensor"]["eta_days"] is None
        assert report["inputs"]["host-memory"] == "no-data"
        assert report["inputs"]["device-memory-eta"] == "insufficient-history"
        assert report["status"] == "partial"
        for finding in report["findings"]:
            assert report["window_start"] <= finding["window_start"]
            assert finding["window_stop"] <= report["observed_at"]
        first = {item["id"]: item["first_seen"] for item in report["findings"]}
        app.state.state_reporter._expires = 0
        refreshed = client.get("/v1/state-report")
        assert refreshed.status_code == 200
        assert {item["id"]: item["first_seen"] for item in refreshed.json()["findings"]} == first
        schema = client.get("/openapi.json").json()["components"]["schemas"]
        assert "MaintenanceEstimate" in schema


@pytest.mark.parametrize(
    "metric,value,expected",
    [
        ("health_status_last", "starting", "unsupported"),
        ("health_status_last", "healthy", "available"),
    ],
)
def test_container_health_does_not_infer_health_from_unknown_inputs(metric, value, expected):
    from edgeops_ai.resources import STEP, ResourceSample

    class ResourcesReader(FixtureReader):
        async def read(self, start, stop):
            window = await super().read(start, stop)
            at = stop.replace(minute=stop.minute - stop.minute % 5) - STEP
            return replace(
                window,
                resources=(
                    ResourceSample(at, "pi/influx", "docker_container_health", metric, value),
                ),
            )

    with TestClient(create_app(settings(), advisor_reader=ResourcesReader())) as client:
        report = client.get("/v1/state-report").json()
        assert report["inputs"]["container-health"] == expected
        assert report["overall"] == "attention"
        assert not any(item["category"] == "host" for item in report["findings"])


def test_silence_detection_keeps_first_detection_time_before_the_full_gap_ends():
    class SilentReader:
        async def read(self, start, stop):
            return AdvisorWindow(
                start, stop, [DeviceSample(stop - 20 * MINUTE, "sensor", {"sensor_samples": 1})], []
            )

    with TestClient(create_app(settings(), advisor_reader=SilentReader())) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200, response.text
        report = response.json()
        finding = next(
            item for item in report["findings"] if item["id"].startswith("device.all-silent")
        )
        assert finding["first_seen"] < finding["window_stop"]
        assert finding["last_seen"] == finding["window_stop"] == report["observed_at"]
        assert report["overall"] == "critical"
