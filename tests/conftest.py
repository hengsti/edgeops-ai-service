from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from edgeops_ai.schemas import METRICS, FeatureVector, ObservationV1

HEALTHY_FEATURE_VALUES: dict[str, Any] = {
    "elapsed_seconds": 5.0,
    "messages_enqueued_rate": 10.0,
    "messages_processed_rate": 10.0,
    "transform_success_rate": 10.0,
    "queue_growth_rate": 0.0,
    "processing_ratio": 1.0,
    "queue_full_rate": 0.0,
    "queue_utilization_ratio": 0.0,
    "queue_metrics_available": True,
    "transform_failure_ratio": 0.0,
    "transform_deserialize_failure_ratio": 0.0,
    "transform_metrics_available": True,
    "decode_failure_ratio": 0.0,
    "decode_metrics_available": True,
    "raw_validation_failure_ratio": 0.0,
    "raw_validation_metrics_available": True,
    "business_validation_failure_ratio": 0.0,
    "business_validation_metrics_available": True,
    "event_queue_drop_ratio": 0.0,
    "input_queue_metrics_available": True,
    "wal_queue_full_ratio": 0.0,
    "wal_queue_metrics_available": True,
    "durability_ack_failure_rate": 0.0,
    "durability_ack_metrics_available": True,
    "dlq_publish_failure_ratio": 0.0,
    "dlq_metrics_available": True,
    "wal_retry_rate": 0.0,
    "wal_drop_ratio": 0.0,
    "wal_retry_outage_active": 0.0,
    "wal_metrics_available": True,
    "influx_write_failure_ratio": 0.0,
    "influx_write_metrics_available": True,
    "influxdb_healthy": 1.0,
    "persistence_ratio": 1.0,
    "pipeline_duration_average_seconds": 0.05,
    "decode_duration_average_seconds": 0.0,
    "decode_duration_metrics_available": True,
    "transform_duration_average_seconds": 0.0,
    "transform_duration_metrics_available": True,
    "persist_duration_average_seconds": 0.0,
    "persist_duration_metrics_available": True,
    "influx_write_duration_average_seconds": 0.0,
    "influx_write_duration_metrics_available": True,
    "heartbeat_missing": False,
    "observed_device_count": 0,
    "missing_device_count": 0,
}


@pytest.fixture
def make_features() -> Callable[..., FeatureVector]:
    """Build a healthy FeatureVector; keyword overrides inject faults."""

    def _make(**overrides: Any) -> FeatureVector:
        return FeatureVector(**{**HEALTHY_FEATURE_VALUES, **overrides})

    return _make


@pytest.fixture
def production_snapshots(tmp_path):
    """Synthetic production-format observations; not evidence from a live server."""
    path = tmp_path / "production-fixture.jsonl"
    rows = []
    for sample in range(31):
        metrics = {name: 0.0 for name, metric in METRICS.items() if metric.kind == "counter"}
        metrics.update(
            messages_enqueued_total=sample * 10,
            messages_processed_total=sample * 10,
            sensor_messages_processed_total=sample * 8,
            status_messages_processed_total=sample * 2,
            transform_success_total=sample * 10,
            influx_lines_written_total=sample * 10,
            pipeline_duration_seconds_sum=sample * 0.5,
            pipeline_duration_seconds_count=sample * 10,
            influx_write_success_total=sample,
            mqtt_messages_received_total=sample * 10,
            transform_attempt_total=sample * 10,
            raw_validation_success_total=sample * 10,
            business_validation_success_total=sample * 10,
            influxdb_healthy=1,
            queue_depth=0,
            queue_capacity=100,
            wal_retry_outage_active=0,
        )
        observation = ObservationV1.model_validate(
            {
                "schema_version": "1.0",
                "metadata": {
                    "captured_at": datetime(2026, 10, 18, tzinfo=UTC) + timedelta(seconds=sample),
                    "collector_mode": "production",
                    "simulated": False,
                    "source_host": "fixture",
                },
                "metrics": metrics,
                "broker_available": True,
            }
        )
        rows.append(observation.model_dump_json())
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
