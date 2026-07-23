from copy import deepcopy

from fastapi.testclient import TestClient

from edgeops_ai.main import create_app


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


def test_health() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "edgeops-ai-service",
        "version": "0.1.0",
        "prediction_backend": "rules",
        "model_version": "rules-0.1.0",
    }


def test_first_observation_returns_warming_up() -> None:
    client = TestClient(create_app())

    response = client.post("/v1/observations", json=observation())

    assert response.status_code == 200, response.json()
    assert response.json()["status"] == "warming-up"
    assert response.json()["prediction"] is None


def test_backpressure_is_detected() -> None:
    client = TestClient(create_app())

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
    client = TestClient(create_app())

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
    client = TestClient(create_app())

    assert client.get("/v1/detections/latest").status_code == 404

    client.post("/v1/observations", json=observation())

    response = client.get("/v1/detections/latest")

    assert response.status_code == 200
    assert response.json()["status"] == "warming-up"


def test_unknown_fields_are_rejected() -> None:
    client = TestClient(create_app())

    payload = observation()
    payload["unexpected"] = True

    response = client.post("/v1/observations", json=payload)

    assert response.status_code == 422
