import asyncio
from datetime import timedelta

import httpx
import pytest

from edgeops_ai.collector_client import CollectorClient
from edgeops_ai.features import FeatureExtractor
from edgeops_ai.rules import RuleBasedPredictor
from edgeops_ai.schemas import ObservationV1

EXPECTED_A2_MAPPING = {
    "mqtt_messages_received_total": "mqtt_messages_received_total",
    "ingest_event_queue_full_total": "event_queue_full_total",
    "ingest_decode_success_total": "decode_success_total",
    "ingest_incoming_oversized_total": "incoming_oversized_total",
    "ingest_incoming_non_utf8_total": "incoming_non_utf8_total",
    "ingest_incoming_invalid_json_total": "incoming_invalid_json_total",
    "ingest_validate_raw_success_total": "raw_validation_success_total",
    "ingest_validate_raw_ignored_total": "raw_validation_ignored_total",
    "ingest_validate_raw_failed_total": "raw_validation_failed_total",
    "ingest_transform_attempt_total": "transform_attempt_total",
    "ingest_transform_deserialize_failed_total": "transform_deserialize_failed_total",
    "ingest_transform_ignored_total": "transform_ignored_total",
    "ingest_validate_business_success_total": "business_validation_success_total",
    "ingest_validate_business_failed_total": "business_validation_failed_total",
    "ingest_wal_queue_full_total": "wal_queue_full_total",
    "ingest_queue_closed_total": "wal_queue_closed_total",
    "ingest_durability_ack_failed_total": "durability_ack_failed_total",
    "dlq_messages_published_total": "dlq_messages_published_total",
    "dlq_publish_errors_total": "dlq_publish_errors_total",
    "wal_forwarder_committed_total": "wal_committed_total",
    "wal_forwarder_drop_total": "wal_drop_total",
    "wal_forwarder_retry_total": "wal_retry_total",
    "wal_forwarder_commit_retry_total": "wal_commit_retry_total",
    "wal_forwarder_retry_outage_active": "wal_retry_outage_active",
    "ingest_decode_duration_seconds_sum": "decode_duration_seconds_sum",
    "ingest_decode_duration_seconds_count": "decode_duration_seconds_count",
    "ingest_transform_duration_seconds_sum": "transform_duration_seconds_sum",
    "ingest_transform_duration_seconds_count": "transform_duration_seconds_count",
    "ingest_persist_duration_seconds_sum": "persist_duration_seconds_sum",
    "ingest_persist_duration_seconds_count": "persist_duration_seconds_count",
    "influx_write_duration_seconds_sum": "influx_write_duration_seconds_sum",
    "influx_write_duration_seconds_count": "influx_write_duration_seconds_count",
}


def collector_metrics() -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "ingest_messages_enqueued_total": 100,
        "ingest_messages_processed_total": 90,
        "ingest_transform_success_total": 80,
        "ingest_transform_failed_total": 3,
        "influx_lines_written_total": 75,
        "influx_write_failed_total": 2,
        "influxdb_healthy": 0,
        "ingest_pipeline_duration_seconds_sum": 12,
        "ingest_pipeline_duration_seconds_count": 80,
        "influx_write_success_total": 73,
        "ingest_queue_depth": 20,
        "ingest_queue_capacity": 100,
        "ingest_sensor_messages_processed_total": 70,
        "ingest_status_messages_processed_total": 20,
    }
    metrics.update(
        {
            source_name: float(index)
            for index, source_name in enumerate(EXPECTED_A2_MAPPING, start=101)
        }
    )
    metrics["ingest_event_queue_full_total"] = 7
    metrics["ingest_wal_queue_full_total"] = 11
    metrics["wal_forwarder_retry_outage_active"] = 1
    return metrics


def envelope(data: object) -> dict[str, object]:
    return {
        "metadata": {
            "captured_at": "2026-08-01T08:00:00Z",
            "collector_mode": "production",
            "simulated": False,
            "source_host": "collector-test",
        },
        "data": data,
    }


def transport_for(
    metrics: dict[str, float | None],
    devices: dict[str, dict[str, object]] | None = None,
    emqx: object = None,
) -> httpx.MockTransport:
    device_responses = devices or {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-EdgeOps-Key"] == "test-key"
        if request.url.path == "/v1/metrics/ingestion":
            return httpx.Response(200, json=envelope(metrics))

        if request.url.path == "/v1/services/emqx" and emqx is not None:
            reply = emqx.pop(0) if isinstance(emqx, list) else emqx
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(200, json=envelope(reply))

        prefix = "/v1/devices/"
        if request.url.path.startswith(prefix):
            device_id = request.url.path.removeprefix(prefix)
            return httpx.Response(200, json=envelope(device_responses[device_id]))

        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def fetch_observation(
    metrics: dict[str, float | None],
    *,
    devices: dict[str, dict[str, object]] | None = None,
    device_ids: tuple[str, ...] = (),
    emqx: object = None,
) -> ObservationV1:
    client = CollectorClient(
        base_url="http://collector.test",
        api_key="test-key",
        timeout_seconds=1,
        device_ids=device_ids,
        transport=transport_for(metrics, devices, emqx),
    )
    try:
        return await client.fetch_observation()
    finally:
        await client.close()


def test_maps_optional_and_a2_metrics() -> None:
    source_metrics = collector_metrics()

    observation = asyncio.run(fetch_observation(source_metrics))

    assert observation.metrics.influx_write_success_total == 73
    assert observation.metrics.queue_depth == 20
    assert observation.metrics.queue_capacity == 100
    assert observation.metrics.sensor_messages_processed_total == 70
    assert observation.metrics.status_messages_processed_total == 20
    for source_name, target_name in EXPECTED_A2_MAPPING.items():
        assert getattr(observation.metrics, target_name) == source_metrics[source_name]

    assert observation.metrics.queue_full_total == 7
    assert observation.metrics.event_queue_full_total == 7
    assert observation.metrics.wal_queue_full_total == 11
    assert observation.metrics.wal_retry_outage_active == 1
    assert observation.metrics.influxdb_healthy == 0


@pytest.mark.parametrize("optional_metrics_present", [True, False])
def test_missing_or_null_optional_metrics_validate(optional_metrics_present: bool) -> None:
    metrics = collector_metrics()
    optional_names = (
        "influx_write_success_total",
        "ingest_queue_depth",
        "ingest_queue_capacity",
        "ingest_sensor_messages_processed_total",
        "ingest_status_messages_processed_total",
    )
    for name in optional_names:
        if optional_metrics_present:
            metrics[name] = None
        else:
            metrics.pop(name)

    observation = asyncio.run(fetch_observation(metrics))

    assert observation.metrics.influx_write_success_total is None
    assert observation.metrics.queue_depth is None
    assert observation.metrics.queue_capacity is None
    assert observation.metrics.sensor_messages_processed_total is None
    assert observation.metrics.status_messages_processed_total is None


def test_missing_required_metric_fails_visibly() -> None:
    metrics = collector_metrics()
    metrics.pop("ingest_messages_enqueued_total")

    with pytest.raises(KeyError, match="ingest_messages_enqueued_total"):
        asyncio.run(fetch_observation(metrics))


def test_masked_a2_group_remains_nullable_and_keeps_legacy_shape() -> None:
    metrics = collector_metrics()
    for source_name in EXPECTED_A2_MAPPING:
        metrics[source_name] = None

    observation = asyncio.run(fetch_observation(metrics))

    assert observation.metrics.queue_full_total == 0.0
    for target_name in EXPECTED_A2_MAPPING.values():
        assert getattr(observation.metrics, target_name) is None


def test_explicit_zero_failure_counters_are_preserved() -> None:
    metrics = collector_metrics()
    zero_failure_metrics = {
        "ingest_incoming_oversized_total": "incoming_oversized_total",
        "ingest_incoming_non_utf8_total": "incoming_non_utf8_total",
        "ingest_incoming_invalid_json_total": "incoming_invalid_json_total",
        "ingest_validate_raw_failed_total": "raw_validation_failed_total",
        "ingest_transform_deserialize_failed_total": "transform_deserialize_failed_total",
        "ingest_validate_business_failed_total": "business_validation_failed_total",
        "ingest_queue_closed_total": "wal_queue_closed_total",
        "ingest_durability_ack_failed_total": "durability_ack_failed_total",
        "dlq_publish_errors_total": "dlq_publish_errors_total",
        "wal_forwarder_drop_total": "wal_drop_total",
        "wal_forwarder_retry_total": "wal_retry_total",
        "wal_forwarder_commit_retry_total": "wal_commit_retry_total",
    }
    metrics.update(dict.fromkeys(zero_failure_metrics, 0.0))

    observation = asyncio.run(fetch_observation(metrics))

    for target_name in zero_failure_metrics.values():
        assert getattr(observation.metrics, target_name) == 0


@pytest.mark.parametrize(
    ("device_id", "heartbeat_age_seconds"),
    [("sensor-a", 12.5), ("sensor-b", None)],
)
def test_maps_populated_and_null_device_age(
    device_id: str,
    heartbeat_age_seconds: float | None,
) -> None:
    devices = {
        device_id: {
            "device_id": device_id,
            "available": heartbeat_age_seconds is not None,
            "heartbeat_age_seconds": heartbeat_age_seconds,
        }
    }

    observation = asyncio.run(
        fetch_observation(collector_metrics(), devices=devices, device_ids=(device_id,))
    )

    assert observation.devices[0].device_id == device_id
    assert observation.devices[0].heartbeat_age_seconds == heartbeat_age_seconds
    assert observation.devices[0].heartbeat_missing is (heartbeat_age_seconds is None)


def emqx_state(status: str, health: str | None) -> dict[str, object]:
    return {"service": "emqx", "status": status, "health": health, "restart_count": 0}


@pytest.mark.parametrize(
    ("emqx", "expected"),
    [
        (emqx_state("running", "healthy"), True),
        (emqx_state("exited", None), False),
        (emqx_state("restarting", None), False),
        (emqx_state("running", "unhealthy"), False),
        (emqx_state("unavailable", None), None),
        (emqx_state("running", "starting"), None),
        (emqx_state("running", None), None),
        (httpx.Response(503), None),
        (httpx.ReadTimeout("timed out"), None),
        (None, None),
    ],
)
def test_maps_emqx_service_to_broker_evidence(emqx: object, expected: bool | None) -> None:
    observation = asyncio.run(fetch_observation(collector_metrics(), emqx=emqx))

    assert observation.broker_available is expected


@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("timed out"), httpx.ConnectError("connection lost"), httpx.Response(503)],
)
@pytest.mark.parametrize(
    ("state", "expected"),
    [(emqx_state("exited", None), False), (emqx_state("running", "healthy"), True)],
)
def test_transient_broker_request_failure_is_retried(
    failure: object, state: dict[str, object], expected: bool
) -> None:
    replies = [failure, state]

    observation = asyncio.run(fetch_observation(collector_metrics(), emqx=replies))

    assert observation.broker_available is expected
    assert replies == []


def test_persistent_broker_request_failure_is_unknown_and_logged(caplog) -> None:
    replies = [httpx.ReadTimeout("timed out"), httpx.ReadTimeout("still timed out")]

    observation = asyncio.run(fetch_observation(collector_metrics(), emqx=replies))

    assert observation.broker_available is None
    assert replies == []
    assert "EMQX broker telemetry request failed after retry" in caplog.text
    assert "still timed out" in caplog.text


def test_broker_outage_and_recovery_from_collection_to_prediction() -> None:
    device_ids = ("sensor-a", "sensor-b")
    stale = {device_id: {"available": False} for device_id in device_ids}
    fresh = {
        device_id: {"available": True, "heartbeat_age_seconds": 1.0} for device_id in device_ids
    }
    extractor = FeatureExtractor()
    predictor = RuleBasedPredictor()

    def analyze(
        seconds: int, devices: dict[str, dict[str, object]], emqx: object, scale: float = 1.0
    ) -> str | None:
        metrics = {
            name: value * scale if value is not None and name.endswith("_total") else value
            for name, value in collector_metrics().items()
        }
        observation = asyncio.run(
            fetch_observation(metrics, devices=devices, device_ids=device_ids, emqx=emqx)
        )
        captured_at = observation.metadata.captured_at + timedelta(seconds=seconds)
        observation = observation.model_copy(
            update={
                "metadata": observation.metadata.model_copy(update={"captured_at": captured_at})
            }
        )
        features = extractor.update(observation)
        return predictor.predict(features).label if features else None

    down = emqx_state("exited", None)
    healthy = emqx_state("running", "healthy")
    assert analyze(0, stale, down) is None
    assert analyze(5, stale, down) == "mqtt-broker-unavailable"
    assert analyze(10, stale, httpx.ReadTimeout("timed out")) == "missing-device-heartbeat"
    # Restarted counters force one warmup interval before diagnosing again.
    assert analyze(15, fresh, healthy, scale=0.5) is None
    assert analyze(20, fresh, healthy, scale=0.5) == "normal"
