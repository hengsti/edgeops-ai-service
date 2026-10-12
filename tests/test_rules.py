import pytest

from edgeops_ai.prediction import PredictionBackendName
from edgeops_ai.rules import RULES_MODEL_VERSION, RuleBasedPredictor

predictor = RuleBasedPredictor()


def test_exposes_backend_metadata() -> None:
    assert predictor.backend_name is PredictionBackendName.RULES
    assert predictor.model_version == RULES_MODEL_VERSION


def test_healthy_features_are_normal(make_features) -> None:
    result = predictor.predict(make_features())

    assert result.label == "normal"
    assert result.model_version == RULES_MODEL_VERSION


def test_missing_heartbeat_wins_and_names_devices(make_features) -> None:
    features = make_features(
        heartbeat_missing=True,
        missing_device_ids=["sensor-a", "sensor-b"],
    )

    result = predictor.predict(features)

    assert result.label == "missing-device-heartbeat"
    assert "sensor-a, sensor-b" in result.recommendation


def test_low_persistence_is_influxdb_write_failure(make_features) -> None:
    features = make_features(transform_success_rate=8.0, persistence_ratio=0.1)

    result = predictor.predict(features)

    assert result.label == "influxdb-write-failure"


def test_transform_failures_are_malformed_payload(make_features) -> None:
    features = make_features(transform_failure_ratio=0.05)

    result = predictor.predict(features)

    assert result.label == "malformed-sensor-payload"


def test_full_queue_is_ingestion_backpressure(make_features) -> None:
    features = make_features(queue_full_rate=0.5)

    result = predictor.predict(features)

    assert result.label == "ingestion-backpressure"


def test_growing_queue_with_slow_processing_is_backpressure(make_features) -> None:
    features = make_features(queue_growth_rate=20.0, processing_ratio=0.5)

    result = predictor.predict(features)

    assert result.label == "ingestion-backpressure"


def test_confirmed_broker_outage_wins_over_stale_heartbeats(make_features) -> None:
    features = make_features(
        broker_available=False,
        heartbeat_missing=True,
        missing_device_ids=["sensor-a", "sensor-b"],
    )

    result = predictor.predict(features)

    assert result.label == "mqtt-broker-unavailable"
    assert "EMQX" in result.recommendation


@pytest.mark.parametrize("broker_available", [True, None])
def test_missing_device_without_confirmed_outage_stays_device_issue(
    make_features, broker_available: bool | None
) -> None:
    features = make_features(
        broker_available=broker_available,
        heartbeat_missing=True,
        missing_device_ids=["sensor-a"],
    )

    assert predictor.predict(features).label == "missing-device-heartbeat"


@pytest.mark.parametrize("broker_available", [True, None])
def test_idle_traffic_is_not_a_broker_outage(make_features, broker_available: bool | None) -> None:
    features = make_features(
        broker_available=broker_available,
        messages_enqueued_rate=0.0,
        messages_processed_rate=0.0,
        transform_success_rate=0.0,
    )

    assert predictor.predict(features).label == "normal"
