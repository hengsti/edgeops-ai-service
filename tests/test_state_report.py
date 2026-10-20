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
        assert report["coverage"]["maintenance"] == "deferred"
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
            assert len(requests) == 2
    finally:
        asyncio.run(reader.close())
