import asyncio
import threading
from datetime import datetime, timedelta
from time import monotonic
from time import sleep as blocking_sleep

import httpx
import pytest

from edgeops_ai import api, influx_reader, state_report
from edgeops_ai.main import create_app
from edgeops_ai.state_report import StateReporter
from test_state_report import FixtureReader, SequenceDetector, settings


class GateReader(FixtureReader):
    def __init__(self):
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.fail = False
        self.failure = OSError("private-token-and-upstream-error")

    async def read(self, start, stop):
        window = await super().read(start, stop)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.fail:
            raise self.failure
        return window


def test_cold_http_budget_is_below_ui_timeout_and_cache_has_original_timestamps():
    async def scenario():
        reader = GateReader()
        app = create_app(settings(), advisor_reader=reader)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            before = monotonic()
            pending = await asyncio.wait_for(client.get("/v1/state-report"), timeout=2.5)
            assert monotonic() - before < 2.5  # Existing UI timeout is three seconds.
            assert pending.status_code == 202
            assert pending.json()["status"] == "pending"
            assert pending.headers["retry-after"] == "2"
            assert pending.headers["cache-control"] == "no-store"
            assert reader.calls == 1 and not reader.cancelled.is_set()
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/ready")).status_code == 200
            assert (await client.get("/v1/detections/latest")).status_code == 404
            reader.release.set()
            ready = await client.get("/v1/state-report")
            assert ready.status_code == 200
            cached = await client.get("/v1/state-report")
            assert cached.json() == ready.json()
            assert reader.calls == 1
            assert ready.headers["x-advisor-state"] == "current"
            assert ready.headers["x-advisor-refresh-seconds"] == "300"
            assert "retry-after" not in ready.headers
            assert ready.headers["x-advisor-expires-at"] == cached.headers["x-advisor-expires-at"]
            expiry = datetime.fromisoformat(ready.headers["x-advisor-expires-at"])
            assert expiry == datetime.fromisoformat(ready.json()["generated_at"]) + timedelta(
                seconds=300
            )
            schema = (await client.get("/openapi.json")).json()
            responses = schema["paths"]["/v1/state-report"]["get"]["responses"]
            assert {"200", "202", "503"} <= responses.keys()
            assert "Retry-After" in responses["202"]["headers"]

    asyncio.run(scenario())


def test_cancelled_request_and_simultaneous_retries_keep_one_refresh(monkeypatch):
    monkeypatch.setattr(api, "ADVISOR_RESPONSE_SECONDS", 0.01, raising=False)

    async def scenario():
        reader = GateReader()
        app = create_app(settings(), advisor_reader=reader)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            cancelled = asyncio.create_task(client.get("/v1/state-report"))
            await asyncio.wait_for(reader.started.wait(), 1)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            replies = await asyncio.wait_for(
                asyncio.gather(*(client.get("/v1/state-report") for _ in range(8))), 1
            )
            assert all(reply.status_code == 202 for reply in replies)
            assert reader.calls == 1 and not reader.cancelled.is_set()
            reader.release.set()
            assert (await client.get("/v1/state-report")).status_code == 200
            assert reader.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [OSError, TimeoutError])
def test_expired_snapshot_is_not_served_while_refreshing_or_after_failure(
    monkeypatch, failure_type
):
    monkeypatch.setattr(api, "ADVISOR_RESPONSE_SECONDS", 0.01, raising=False)

    async def scenario():
        reader = GateReader()
        reader.failure = failure_type("private-token-and-upstream-error")
        reader.release.set()
        app = create_app(settings(), advisor_reader=reader)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            await app.state.state_reporter.report()
            first = await client.get("/v1/state-report")
            assert first.status_code == 200
            reporter = app.state.state_reporter
            reporter._expires = 0
            reader.release.clear()
            pending = await asyncio.wait_for(client.get("/v1/state-report"), 1)
            assert pending.status_code == 202
            assert "findings" not in pending.json() and "overall" not in pending.json()
            reader.fail = True
            reader.release.set()
            failed = await client.get("/v1/state-report")
            assert failed.status_code == 503
            assert failed.json()["status"] == "failed"
            assert "private-token" not in failed.text
            backoff = await client.get("/v1/state-report")
            assert backoff.status_code == 503
            assert backoff.json()["status"] == "backoff"
            assert 1 <= int(backoff.headers["retry-after"]) <= 60
            assert reader.calls == 2
            reader.fail = False
            reporter._retry_after = 0
            await reporter.report()
            recovered = await client.get("/v1/state-report")
            assert recovered.status_code == 200
            assert reader.calls == 3
            assert recovered.json()["generated_at"] != first.json()["generated_at"]
            assert {row["id"]: row["first_seen"] for row in recovered.json()["findings"]} == {
                row["id"]: row["first_seen"] for row in first.json()["findings"]
            }

    asyncio.run(scenario())


def test_background_failure_without_waiters_is_observed_and_backs_off(monkeypatch):
    monkeypatch.setattr(api, "ADVISOR_RESPONSE_SECONDS", 0.01, raising=False)

    async def scenario():
        errors = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: errors.append(context)
        )
        reader = GateReader()
        reader.fail = True
        app = create_app(settings(), advisor_reader=reader)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            assert (await asyncio.wait_for(client.get("/v1/state-report"), 1)).status_code == 202
            reader.release.set()
            await app.state.state_reporter._refresh_task
            response = await client.get("/v1/state-report")
            assert response.status_code == 503 and response.json()["status"] == "backoff"
            assert reader.calls == 1
        assert errors == []

    asyncio.run(scenario())


def test_shutdown_cancels_read_before_the_owned_input_client_closes(monkeypatch):
    monkeypatch.setattr(api, "ADVISOR_RESPONSE_SECONDS", 0.01, raising=False)
    readers = []

    class OwnedReader(GateReader):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.closed = False
            readers.append(self)

        async def close(self):
            assert self.cancelled.is_set()
            self.closed = True

    monkeypatch.setattr(influx_reader, "HttpInfluxReader", OwnedReader)

    async def scenario():
        app = create_app(
            settings(advisor_influx_url="http://influx", advisor_influx_token="secret")
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            assert (await asyncio.wait_for(client.get("/v1/state-report"), 1)).status_code == 202
        assert readers[0].closed
        assert app.state.state_reporter._refresh_task.done()
        with pytest.raises(ValueError):
            await app.state.state_reporter.report()
        assert readers[0].calls == 1

    asyncio.run(scenario())


def test_shutdown_waits_for_detector_thread_before_closing_clients():
    async def scenario():
        entered, release = threading.Event(), threading.Event()

        class Detector:
            category = "host"

            def detect(self, window):
                entered.set()
                assert release.wait(5)
                return []

        reporter = StateReporter(FixtureReader(), (Detector(),))
        request = asyncio.create_task(reporter.report())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            closing = asyncio.create_task(reporter.close())
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
        await closing
        with pytest.raises(asyncio.CancelledError):
            await request
        assert reporter._cached is None

    asyncio.run(scenario())


def test_unconfigured_is_machine_readable_and_has_no_automatic_retry():
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings())) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 503
        assert response.json()["status"] == "unconfigured"
        assert response.json()["detail"] == "Advisor is not configured."
        assert "retry-after" not in response.headers
        assert response.headers["cache-control"] == "no-store"


def test_backoff_counts_down_and_one_refresh_restarts_at_the_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(state_report, "monotonic", lambda: clock[0])

    async def scenario():
        reader = GateReader()
        reader.release.set()
        reader.fail = True
        reporter = StateReporter(reader)
        with pytest.raises(ValueError):
            await reporter.report()
        assert reporter.retry_after_seconds == 60
        clock[0] = 159.25
        assert reporter.retry_after_seconds == 1
        with pytest.raises(ValueError):
            await reporter.report()
        assert reader.calls == 1
        clock[0] = 160
        reader.fail = False
        await asyncio.gather(*(reporter.report() for _ in range(8)))
        assert reader.calls == 2
        assert reporter.retry_after_seconds == 0
        await reporter.close()

    asyncio.run(scenario())


def test_failed_refresh_does_not_advance_finding_recovery():
    async def scenario():
        reader, detector = GateReader(), SequenceDetector()
        reader.release.set()
        reporter = StateReporter(reader, (detector,), refresh_seconds=0)
        first = await reporter.report()
        detector.present, reader.fail = False, True
        for _ in range(2):
            with pytest.raises(ValueError):
                await reporter.report()
        reader.fail = False
        assert (await reporter.report()).findings == first.findings
        assert (await reporter.report()).findings == []
        await reporter.close()

    asyncio.run(scenario())


def test_pi_measurement_phases_check_real_endpoint_and_shared_refresh(monkeypatch):
    from scripts import validate_advisor_lifecycle as validation

    monkeypatch.setattr(validation, "host_cpu", lambda: (100, 50))
    monkeypatch.setattr(validation, "peak_rss_mib", lambda: 12)

    async def scenario():
        reader = GateReader()
        reader.queries = 0
        reader.release.set()
        app = create_app(settings(), advisor_reader=reader)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            reporter = app.state.state_reporter
            cold, first = await validation.measure_phase("cold", client, reporter, reader)
            cached, second = await validation.measure_phase("cached", client, reporter, reader)
            warm, _ = await validation.measure_phase("warm-refresh", client, reporter, reader)
            simultaneous, _ = await validation.measure_phase(
                "simultaneous-refresh", client, reporter, reader, 8
            )
            assert first == second
            assert cold["reader_calls"] == warm["reader_calls"] == simultaneous["reader_calls"] == 1
            assert cached["reader_calls"] == cached["queries"] == 0
            assert cached["generated_at"] == cold["generated_at"]
            assert len(simultaneous["requests"]) == simultaneous["initial_callers"] == 8
            assert reader.calls == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_held_out_validation_waits_for_pending_and_keeps_errors(monkeypatch, failure):
    from fastapi.testclient import TestClient
    from scripts.validate_advisor_runtime import report_response

    monkeypatch.setattr(api, "ADVISOR_RESPONSE_SECONDS", 0.001)

    class SlowReader(FixtureReader):
        async def read(self, start, stop):
            await asyncio.sleep(0.02)
            if failure:
                raise ValueError("private-error")
            return await super().read(start, stop)

    reader = SlowReader()
    with TestClient(create_app(settings(), advisor_reader=reader)) as client:
        response = report_response(client)
        assert response.status_code == (503 if failure else 200)
        assert "private-error" not in response.text
        assert reader.calls == (0 if failure else 1)


@pytest.mark.parametrize("source", ["sensor", "pipeline", "resource"])
def test_real_reader_normalization_keeps_the_http_deadline_responsive(monkeypatch, source):
    original_time = influx_reader.observation_time
    slow = [True]
    processed = []

    def costly_observation(*args):
        if slow[0]:
            blocking_sleep(0.0015)  # Model a slower CPU processing a large query result.
        processed.append(1)
        return original_time(*args)

    monkeypatch.setattr(influx_reader, "observation_time", costly_observation)

    class RowsReader(influx_reader.HttpInfluxReader):
        async def _query(self, query):
            if "advisor-history" in query:
                return []
            kind = (
                "sensor"
                if '"sensors_1m"' in query
                else ("resource" if '"metrics_5m"' in query else "pipeline")
            )
            if kind != source:
                return []
            stop = datetime.fromisoformat(query.split("stop: ", 1)[1].split(")", 1)[0])
            at = stop.replace(minute=stop.minute - stop.minute % 5) - timedelta(minutes=5)
            return [
                {
                    "_time": at.isoformat(),
                    "_value_type": "double",
                    "_value": "1",
                    "_measurement": {
                        "sensor": "bme680",
                        "resource": "disk",
                        "pipeline": "prometheus",
                    }[source],
                    "_field": {
                        "sensor": "temp_c_count",
                        "resource": "used_percent_mean",
                        "pipeline": "dlq_messages_published_total",
                    }[source],
                    "device_id": f"sensor-{index}",
                    "host": f"host-{index}",
                    "path": "/",
                }
                for index in range(2200)
            ]

    async def scenario():
        reader = RowsReader("http://influx", "org", "fixture-token")
        app = create_app(settings(), advisor_reader=reader)
        try:
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client,
            ):
                try:
                    before = monotonic()
                    response = await asyncio.wait_for(client.get("/v1/state-report"), 2.5)
                    assert monotonic() - before < 2.5
                    assert response.status_code == 202
                    assert processed
                    assert (await asyncio.wait_for(client.get("/health"), 1)).status_code == 200
                finally:
                    slow[0] = False
        finally:
            await reader.close()

    asyncio.run(scenario())


def test_pi_measurement_rejects_missing_configured_models_before_reading(monkeypatch, tmp_path):
    from scripts import validate_advisor_lifecycle as validation

    configured = settings(
        advisor_influx_url="http://influx",
        advisor_influx_token="fixture-token",
        advisor_forecast_config=tmp_path / "missing.json",
    )
    monkeypatch.setattr(validation, "Settings", lambda **kwargs: configured)
    monkeypatch.setattr(validation, "host_cpu", lambda: (100, 50))
    monkeypatch.setattr(validation, "peak_rss_mib", lambda: 0)
    readers = []

    class Reader(GateReader):
        queries = 0

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.release.set()
            readers.append(self)

        async def close(self):
            pass

    monkeypatch.setattr(validation, "MeasuredReader", Reader)
    with pytest.raises(ValueError, match="Configured Advisor consumers failed to load"):
        asyncio.run(validation.main())
    assert readers[0].calls == 0
