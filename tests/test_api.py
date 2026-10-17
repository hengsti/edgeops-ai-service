import asyncio
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from edgeops_ai.collector_client import CollectorClient
from edgeops_ai.features import FeatureExtractor
from edgeops_ai.main import create_app
from edgeops_ai.ml import ML_MODEL_VERSION, MlPredictor, ModelArtifactError, load_ml_predictor
from edgeops_ai.prediction import PredictionBackendName
from edgeops_ai.schemas import FeatureVector, ObservationV1
from edgeops_ai.settings import Settings


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "collector_polling_enabled": False,
        "prediction_backend": PredictionBackendName.RULES,
    }
    values.update(overrides)

    return Settings(**values)


def observation() -> dict:
    return {
        "schema_version": "1.0",
        "metadata": {
            "captured_at": "2024-01-01T00:00:00Z",
            "collector_mode": "simulation",
            "simulated": True,
            "source_host": "local-dev",
            "scenario_id": "ingestion-backpressure",
            "simulation_run_id": "test-run-1",
            "simulation_seed": 42,
            "simulation_phase": "normal",
        },
        "metrics": {
            "messages_enqueued_total": 1000,
            "messages_processed_total": 1000,
            "queue_full_total": 0,
            "transform_success_total": 1000,
            "transform_failed_total": 0,
            "influx_lines_written_total": 1000,
            "pipeline_duration_seconds_sum": 2,
            "pipeline_duration_seconds_count": 1000,
        },
        "devices": [
            {
                "device_id": "esp32-example",
                "heartbeat_missing": False,
                "heartbeat_age_seconds": 2,
            }
        ],
    }


def enriched_observation() -> dict:
    payload = observation()
    payload["metrics"].update(
        {
            "influx_write_success_total": 998,
            "queue_depth": 20,
            "queue_capacity": 100,
            "sensor_messages_processed_total": 800,
            "status_messages_processed_total": 200,
            "mqtt_messages_received_total": 1010,
            "event_queue_full_total": 10,
            "decode_success_total": 1000,
            "incoming_oversized_total": 1,
            "incoming_non_utf8_total": 2,
            "incoming_invalid_json_total": 3,
            "raw_validation_success_total": 990,
            "raw_validation_ignored_total": 4,
            "raw_validation_failed_total": 6,
            "transform_attempt_total": 990,
            "transform_deserialize_failed_total": 2,
            "transform_ignored_total": 1,
            "business_validation_success_total": 985,
            "business_validation_failed_total": 3,
            "wal_queue_full_total": 5,
            "wal_queue_closed_total": 1,
            "durability_ack_failed_total": 1,
            "dlq_messages_published_total": 10,
            "dlq_publish_errors_total": 1,
            "wal_committed_total": 980,
            "wal_drop_total": 2,
            "wal_retry_total": 8,
            "wal_commit_retry_total": 1,
            "wal_retry_outage_active": 1,
            "decode_duration_seconds_sum": 5,
            "decode_duration_seconds_count": 1000,
            "transform_duration_seconds_sum": 10,
            "transform_duration_seconds_count": 990,
            "persist_duration_seconds_sum": 8,
            "persist_duration_seconds_count": 985,
            "influx_write_duration_seconds_sum": 20,
            "influx_write_duration_seconds_count": 1000,
        }
    )
    return payload


def test_health() -> None:
    with TestClient(create_app(make_settings())) as client:
        response = client.get("/health")

        assert response.status_code == 200


def test_documented_quickstart_starts_with_rules_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PREDICTION_BACKEND", raising=False)
    settings = Settings(_env_file=None, collector_polling_enabled=False)

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["prediction_backend"] == "rules"


def test_docker_image_bundles_only_loadable_artifacts() -> None:
    for line in Path("Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY") and ".joblib" in line:
            load_ml_predictor(Path(line.split()[1]))


def test_first_observation_returns_warming_up() -> None:
    with TestClient(create_app(make_settings())) as client:
        response = client.post("/v1/observations", json=observation())

        assert response.status_code == 200, response.json()
        assert response.json()["status"] == "warming-up"
        assert response.json()["prediction"] is None


def test_enriched_observation_is_accepted() -> None:
    with TestClient(create_app(make_settings())) as client:
        response = client.post("/v1/observations", json=enriched_observation())

    assert response.status_code == 200, response.json()
    assert response.json()["status"] == "warming-up"


def test_enriched_features_are_serialized() -> None:
    with TestClient(create_app(make_settings())) as client:
        first = enriched_observation()
        client.post("/v1/observations", json=first)

        second = deepcopy(first)
        second["metadata"]["captured_at"] = "2024-01-01T00:00:10Z"
        response = client.post("/v1/observations", json=second)

    assert response.status_code == 200, response.json()
    features = response.json()["features"]
    assert features["queue_utilization_ratio"] == 0.2
    assert features["queue_metrics_available"] is True
    assert features["decode_failure_ratio"] == 0
    assert features["decode_metrics_available"] is True
    assert features["wal_retry_outage_active"] == 1
    assert features["wal_metrics_available"] is True
    assert features["decode_duration_average_seconds"] == 0
    assert features["decode_duration_metrics_available"] is True


def test_backpressure_is_detected() -> None:
    with TestClient(create_app(make_settings())) as client:
        first = observation()
        client.post("/v1/observations", json=first)

        second = deepcopy(first)
        second["metadata"]["captured_at"] = "2024-01-01T00:00:10Z"
        second["metadata"]["simulation_phase"] = "overloaded"

        second["metrics"]["messages_enqueued_total"] = 1200
        second["metrics"]["messages_processed_total"] = 1050
        second["metrics"]["queue_full_total"] = 20

        second["metrics"]["transform_success_total"] = 1050
        second["metrics"]["influx_lines_written_total"] = 1050
        second["metrics"]["pipeline_duration_seconds_sum"] = 2.5
        second["metrics"]["pipeline_duration_seconds_count"] = 1050

        response = client.post("/v1/observations", json=second)

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "predicted"
        assert body["prediction"] == "ingestion-backpressure"
        assert body["features"]["messages_enqueued_rate"] == 20
        assert body["features"]["messages_processed_rate"] == 5
        assert body["features"]["queue_growth_rate"] == 15


def test_missing_heartbeat_is_detected() -> None:
    with TestClient(create_app(make_settings())) as client:
        first = observation()
        client.post("/v1/observations", json=first)

        second = deepcopy(first)
        second["metadata"]["captured_at"] = "2024-01-01T00:00:10Z"
        second["metadata"]["scenario_id"] = "missing-device-heartbeat"
        second["metadata"]["simulation_phase"] = "heartbeat-missing"

        second["metrics"]["messages_enqueued_total"] = 1100
        second["metrics"]["messages_processed_total"] = 1100
        second["metrics"]["transform_success_total"] = 1100
        second["metrics"]["influx_lines_written_total"] = 1100
        second["metrics"]["pipeline_duration_seconds_sum"] = 2.2
        second["metrics"]["pipeline_duration_seconds_count"] = 1100

        second["devices"][0]["heartbeat_missing"] = True
        second["devices"][0]["heartbeat_age_seconds"] = 120

        response = client.post("/v1/observations", json=second)

        assert response.status_code == 200
        assert response.json()["prediction"] == "missing-device-heartbeat"


def test_latest_detection() -> None:
    with TestClient(create_app(make_settings())) as client:
        assert client.get("/v1/detections/latest").status_code == 404

        client.post("/v1/observations", json=observation())

        response = client.get("/v1/detections/latest")

        assert response.status_code == 200
        assert response.json()["status"] == "warming-up"


def test_unknown_fields_are_rejected() -> None:
    with TestClient(create_app(make_settings())) as client:
        payload = observation()
        payload["unexpected"] = True

        response = client.post("/v1/observations", json=payload)

        assert response.status_code == 422


def test_ml_health_reports_artifact_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "edgeops_ai.ml.load_ml_predictor",
        lambda *_: MlPredictor(model=object()),
    )
    settings = Settings(
        collector_polling_enabled=False,
        prediction_backend="ml",
        model_artifact_path=Path("artifacts/model.joblib"),
        model_artifact_sha256="0" * 64,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["prediction_backend"] == "ml"
    assert response.json()["model_version"] == ML_MODEL_VERSION


def test_ml_warming_up_uses_ml_model_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "edgeops_ai.ml.load_ml_predictor",
        lambda *_: MlPredictor(model=object()),
    )
    settings = Settings(
        collector_polling_enabled=False,
        prediction_backend="ml",
        model_artifact_path=Path("artifacts/model.joblib"),
        model_artifact_sha256="0" * 64,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/observations",
            json=observation(),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "warming-up"
    assert response.json()["model_version"] == ML_MODEL_VERSION


def test_ml_startup_fails_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    settings = Settings(
        collector_polling_enabled=False,
        prediction_backend="ml",
        model_artifact_path=tmp_path / "missing.joblib",
        model_artifact_sha256="0" * 64,
    )

    with (
        pytest.raises(
            ModelArtifactError,
            match="does not exist",
        ),
        TestClient(create_app(settings)),
    ):
        pass


def test_ml_startup_requires_artifact_sha256() -> None:
    settings = Settings(
        collector_polling_enabled=False,
        prediction_backend="ml",
        model_artifact_path=Path("artifacts/model.joblib"),
    )

    with (
        pytest.raises(ModelArtifactError, match="MODEL_ARTIFACT_SHA256 is required"),
        TestClient(create_app(settings)),
    ):
        pass


def test_rules_application_does_not_import_the_ml_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, sys\n"
            "from edgeops_ai.main import create_app\n"
            "from edgeops_ai.settings import Settings\n"
            "app = create_app(Settings(_env_file=None, prediction_backend='rules', "
            "collector_polling_enabled=False))\n"
            "async def startup():\n"
            "    async with app.router.lifespan_context(app):\n"
            "        assert 'joblib' not in sys.modules\n"
            "        assert 'numpy' not in sys.modules\n"
            "asyncio.run(startup())\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_collector_client_closes_when_polling_task_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = False
    original_close = CollectorClient.close

    async def fail_fetch(self: CollectorClient):
        raise RuntimeError("unexpected prediction failure")

    async def record_close(self: CollectorClient) -> None:
        nonlocal closed
        await original_close(self)
        closed = True

    monkeypatch.setattr(CollectorClient, "fetch_observation", fail_fetch)
    monkeypatch.setattr(CollectorClient, "close", record_close)
    app = create_app(make_settings(collector_polling_enabled=True))

    async def run() -> None:
        with pytest.raises(RuntimeError, match="unexpected prediction failure"):
            async with app.router.lifespan_context(app):
                await asyncio.sleep(0)

    asyncio.run(run())
    assert closed is True


def test_push_analysis_runs_on_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_threads: list[int] = []
    original_update = FeatureExtractor.update

    def record_update(self: FeatureExtractor, observation: ObservationV1) -> FeatureVector | None:
        analysis_threads.append(threading.get_ident())
        return original_update(self, observation)

    monkeypatch.setattr(FeatureExtractor, "update", record_update)

    async def run() -> None:
        app = create_app(make_settings())
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://ai.test"
            ) as client,
        ):
            response = await client.post("/v1/observations", json=observation())
            assert response.status_code == 200
            assert response.json()["status"] == "warming-up"
            assert analysis_threads == [threading.get_ident()]

    asyncio.run(run())


def test_concurrent_push_and_poll_preserve_the_polling_window(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def run() -> None:
        failed = asyncio.Event()
        recover = asyncio.Event()
        recovered = asyncio.Event()
        fetch_count = 0
        first = ObservationV1.model_validate(observation())
        second = deepcopy(first)
        second.metadata.captured_at = first.metadata.captured_at.replace(second=5)
        second.metrics.messages_enqueued_total = 2000
        third = deepcopy(first)
        third.metadata.captured_at = first.metadata.captured_at.replace(second=10)
        third.metrics.messages_enqueued_total = 1100
        third.metrics.messages_processed_total = 1050

        async def fetch(self: CollectorClient) -> ObservationV1:
            nonlocal fetch_count
            fetch_count += 1
            if fetch_count == 1:
                return first
            if fetch_count == 2:
                raise httpx.ConnectError("collector offline")
            if fetch_count == 3:
                failed.set()
                await recover.wait()
                return third
            recovered.set()
            await asyncio.Event().wait()
            raise AssertionError("Polling should have been cancelled.")

        monkeypatch.setattr(CollectorClient, "fetch_observation", fetch)
        app = create_app(
            make_settings(collector_polling_enabled=True, collector_poll_interval_seconds=0.001)
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://ai.test"
            ) as client,
        ):
            await asyncio.wait_for(failed.wait(), timeout=5)
            latest = await client.get("/v1/detections/latest")
            assert latest.json()["captured_at"] == first.metadata.captured_at.isoformat().replace(
                "+00:00", "Z"
            )
            assert latest.json()["status"] == "warming-up"
            assert app.state.analyzer.collector_last_error == "collector offline"

            submission = asyncio.create_task(
                client.post("/v1/observations", json=second.model_dump(mode="json"))
            )
            recover.set()
            submitted = await submission
            assert submitted.status_code == 409

            await asyncio.wait_for(recovered.wait(), timeout=5)
            latest = await client.get("/v1/detections/latest")
            assert latest.json()["status"] == "predicted"
            assert latest.json()["features"]["elapsed_seconds"] == 10
            assert latest.json()["features"]["messages_enqueued_rate"] == 10
            assert latest.json()["features"]["messages_processed_rate"] == 5
            assert latest.json()["features"]["queue_growth_rate"] == 5
            assert app.state.analyzer.collector_last_error is None

    asyncio.run(run())
    assert "Collector polling failed: collector offline" in caplog.text
