from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from edgeops_ai.features import FeatureExtractor, safe_ratio
from edgeops_ai.schemas import FeatureVector, ObservationV1

BASE_METRICS: dict[str, float] = {
    "messages_enqueued_total": 100,
    "messages_processed_total": 100,
    "queue_full_total": 0,
    "transform_success_total": 100,
    "transform_failed_total": 0,
    "influx_lines_written_total": 100,
    "influx_write_failed_total": 0,
    "influxdb_healthy": 1,
    "pipeline_duration_seconds_sum": 10,
    "pipeline_duration_seconds_count": 100,
}


def observation(
    elapsed_seconds: int,
    *,
    metrics: dict[str, Any] | None = None,
    devices: list[dict[str, Any]] | None = None,
    collector_mode: str = "simulation",
    source_host: str = "feature-test",
    simulation_run_id: str = "run-1",
) -> ObservationV1:
    values = {**BASE_METRICS, **(metrics or {})}
    return ObservationV1.model_validate(
        {
            "schema_version": "1.0",
            "metadata": {
                "captured_at": datetime(2026, 8, 1, tzinfo=UTC)
                + timedelta(seconds=elapsed_seconds),
                "collector_mode": collector_mode,
                "simulated": collector_mode == "simulation",
                "source_host": source_host,
                "simulation_run_id": simulation_run_id,
            },
            "metrics": values,
            "devices": devices or [],
        }
    )


def extract_features(
    *,
    previous_metrics: dict[str, Any] | None = None,
    current_metrics: dict[str, Any] | None = None,
    previous_devices: list[dict[str, Any]] | None = None,
    current_devices: list[dict[str, Any]] | None = None,
    collector_mode: str = "simulation",
) -> FeatureVector:
    extractor = FeatureExtractor()
    assert (
        extractor.update(
            observation(
                0,
                metrics=previous_metrics,
                devices=previous_devices,
                collector_mode=collector_mode,
            )
        )
        is None
    )
    features = extractor.update(
        observation(
            10,
            metrics=current_metrics,
            devices=current_devices,
            collector_mode=collector_mode,
        )
    )
    assert features is not None
    return features


def test_safe_ratio_uses_default_for_empty_denominator() -> None:
    assert safe_ratio(3, 6) == 0.5
    assert safe_ratio(3, 0) == 0.0
    assert safe_ratio(3, 0, default=1.0) == 1.0


@pytest.mark.parametrize(
    ("current_metrics", "expected_ratio", "expected_available"),
    [
        ({"queue_depth": 20, "queue_capacity": 100}, 0.2, True),
        ({"queue_depth": 0, "queue_capacity": 100}, 0.0, True),
        ({"queue_depth": 120, "queue_capacity": 100}, 1.0, True),
        ({}, None, False),
    ],
)
def test_queue_utilization_distinguishes_zero_from_missing(
    current_metrics: dict[str, float],
    expected_ratio: float | None,
    expected_available: bool,
) -> None:
    features = extract_features(current_metrics=current_metrics)

    assert features.queue_utilization_ratio == expected_ratio
    assert features.queue_metrics_available is expected_available


@pytest.mark.parametrize(
    ("previous_metrics", "current_metrics", "expected_ratio", "expected_available"),
    [
        (
            {"influx_write_success_total": 100},
            {"influx_write_success_total": 105, "influx_write_failed_total": 1},
            pytest.approx(1 / 6),
            True,
        ),
        (
            {"influx_write_success_total": 100},
            {"influx_write_success_total": 100},
            0.0,
            True,
        ),
        ({}, {}, None, False),
    ],
)
def test_influx_write_ratio_distinguishes_zero_from_missing(
    previous_metrics: dict[str, float],
    current_metrics: dict[str, float],
    expected_ratio: float | None,
    expected_available: bool,
) -> None:
    features = extract_features(
        previous_metrics=previous_metrics,
        current_metrics=current_metrics,
    )

    assert features.influx_write_failure_ratio == expected_ratio
    assert features.influx_write_metrics_available is expected_available


def test_device_ages_are_aggregated() -> None:
    devices = [
        {"device_id": "sensor-a", "heartbeat_missing": False, "heartbeat_age_seconds": 10},
        {"device_id": "sensor-b", "heartbeat_missing": False, "heartbeat_age_seconds": 30},
    ]

    features = extract_features(current_devices=devices)

    assert features.heartbeat_age_mean_seconds == 20
    assert features.heartbeat_age_max_seconds == 30
    assert features.heartbeat_age_available is True
    assert features.observed_device_count == 2
    assert features.missing_device_count == 0
    assert features.missing_device_ratio == 0
    assert features.device_metrics_available is True


def test_no_devices_are_explicitly_unavailable() -> None:
    features = extract_features()

    assert features.observed_device_count == 0
    assert features.missing_device_count == 0
    assert features.missing_device_ratio == 0
    assert features.device_metrics_available is False
    assert features.heartbeat_age_available is False


def test_observation_without_broker_field_is_unknown() -> None:
    assert extract_features().broker_available is None


def test_missing_device_is_counted_and_named() -> None:
    devices = [
        {"device_id": "sensor-a", "heartbeat_missing": True, "heartbeat_age_seconds": None},
        {"device_id": "sensor-b", "heartbeat_missing": False, "heartbeat_age_seconds": 4},
    ]

    features = extract_features(current_devices=devices)

    assert features.heartbeat_missing is True
    assert features.missing_device_ids == ["sensor-a"]
    assert features.missing_device_count == 1
    assert features.missing_device_ratio == 0.5


def test_optional_counter_reset_restarts_warmup() -> None:
    extractor = FeatureExtractor()
    assert extractor.update(observation(0, metrics={"wal_retry_total": 10})) is None

    assert extractor.update(observation(10, metrics={"wal_retry_total": 5})) is None
    assert extractor.update(observation(20, metrics={"wal_retry_total": 6})) is not None


def test_gauges_can_decrease_without_resetting_the_stream() -> None:
    features = extract_features(
        previous_metrics={
            "queue_depth": 20,
            "queue_capacity": 100,
            "wal_retry_outage_active": 1,
        },
        current_metrics={
            "queue_depth": 5,
            "queue_capacity": 100,
            "wal_retry_outage_active": 0,
        },
    )

    assert features.queue_utilization_ratio == 0.05
    assert features.wal_retry_outage_active == 0


def test_pipeline_average_uses_window_deltas() -> None:
    features = extract_features(
        previous_metrics={
            "pipeline_duration_seconds_sum": 10,
            "pipeline_duration_seconds_count": 100,
        },
        current_metrics={
            "pipeline_duration_seconds_sum": 14,
            "pipeline_duration_seconds_count": 102,
        },
    )

    assert features.pipeline_duration_average_seconds == 2


def test_production_persistence_override_is_preserved() -> None:
    features = extract_features(
        previous_metrics={"transform_success_total": 100, "influx_lines_written_total": 100},
        current_metrics={"transform_success_total": 110, "influx_lines_written_total": 100},
        collector_mode="production",
    )

    assert features.persistence_ratio == 1


def test_stream_change_restarts_warmup() -> None:
    extractor = FeatureExtractor()
    assert extractor.update(observation(0)) is None

    assert extractor.update(observation(10, source_host="replacement-host")) is None


def test_event_queue_and_wal_queue_pressure_remain_separate() -> None:
    previous = {
        "mqtt_messages_received_total": 100,
        "event_queue_full_total": 0,
        "wal_queue_full_total": 0,
        "wal_queue_closed_total": 0,
    }
    event_pressure = extract_features(
        previous_metrics=previous,
        current_metrics={
            "messages_enqueued_total": 190,
            "mqtt_messages_received_total": 190,
            "event_queue_full_total": 10,
            "wal_queue_full_total": 0,
            "wal_queue_closed_total": 0,
        },
    )
    wal_pressure = extract_features(
        previous_metrics=previous,
        current_metrics={
            "messages_enqueued_total": 190,
            "mqtt_messages_received_total": 200,
            "event_queue_full_total": 0,
            "wal_queue_full_total": 10,
            "wal_queue_closed_total": 0,
        },
    )

    assert event_pressure.event_queue_drop_ratio == 0.1
    assert event_pressure.wal_queue_full_ratio == 0
    assert wal_pressure.event_queue_drop_ratio == 0
    assert wal_pressure.wal_queue_full_ratio == 0.1


def test_decode_failure_ratio_uses_all_decode_failures_and_input_drops() -> None:
    previous = {
        "mqtt_messages_received_total": 100,
        "event_queue_full_total": 0,
        "incoming_oversized_total": 0,
        "incoming_non_utf8_total": 0,
        "incoming_invalid_json_total": 0,
    }
    current = {
        "mqtt_messages_received_total": 190,
        "event_queue_full_total": 10,
        "incoming_oversized_total": 2,
        "incoming_non_utf8_total": 3,
        "incoming_invalid_json_total": 5,
    }

    features = extract_features(previous_metrics=previous, current_metrics=current)

    assert features.decode_failure_ratio == 0.1
    assert features.event_queue_drop_ratio == 0.1
    assert features.decode_metrics_available is True
    assert features.input_queue_metrics_available is True


def test_validation_ratios_and_ignored_transform_traffic() -> None:
    previous = {
        "raw_validation_success_total": 0,
        "raw_validation_ignored_total": 0,
        "raw_validation_failed_total": 0,
        "business_validation_success_total": 0,
        "business_validation_failed_total": 0,
        "transform_attempt_total": 0,
        "transform_deserialize_failed_total": 0,
        "transform_ignored_total": 0,
    }
    current = {
        "raw_validation_success_total": 7,
        "raw_validation_ignored_total": 1,
        "raw_validation_failed_total": 2,
        "business_validation_success_total": 7,
        "business_validation_failed_total": 3,
        "transform_attempt_total": 10,
        "transform_deserialize_failed_total": 0,
        "transform_ignored_total": 10,
    }

    features = extract_features(previous_metrics=previous, current_metrics=current)

    assert features.raw_validation_failure_ratio == 0.2
    assert features.raw_validation_ignored_ratio == 0.1
    assert features.raw_validation_metrics_available is True
    assert features.business_validation_failure_ratio == 0.3
    assert features.business_validation_metrics_available is True
    assert features.transform_deserialize_failure_ratio == 0
    assert features.transform_ignored_ratio == 1
    assert features.transform_failure_ratio == 0
    assert features.transform_metrics_available is True


def test_transform_deserialization_contributes_to_failure_ratio() -> None:
    features = extract_features(
        previous_metrics={
            "transform_attempt_total": 0,
            "transform_deserialize_failed_total": 0,
        },
        current_metrics={
            "transform_failed_total": 5,
            "transform_attempt_total": 100,
            "transform_deserialize_failed_total": 5,
        },
    )

    assert features.transform_deserialize_failure_ratio == 0.05
    assert features.transform_failure_ratio == 0.1


def test_wal_queue_and_durability_rates_use_the_interval() -> None:
    previous = {
        "wal_queue_full_total": 0,
        "wal_queue_closed_total": 0,
        "durability_ack_failed_total": 0,
    }
    current = {
        "messages_enqueued_total": 180,
        "wal_queue_full_total": 10,
        "wal_queue_closed_total": 10,
        "durability_ack_failed_total": 2,
    }

    features = extract_features(previous_metrics=previous, current_metrics=current)

    assert features.wal_queue_full_ratio == 0.1
    assert features.wal_queue_closed_rate == 1
    assert features.wal_queue_metrics_available is True
    assert features.durability_ack_failure_rate == 0.2
    assert features.durability_ack_metrics_available is True


def test_dlq_publish_failures_are_distinct_from_successes() -> None:
    features = extract_features(
        previous_metrics={"dlq_messages_published_total": 0, "dlq_publish_errors_total": 0},
        current_metrics={"dlq_messages_published_total": 9, "dlq_publish_errors_total": 1},
    )

    assert features.dlq_publish_failure_ratio == 0.1
    assert features.dlq_metrics_available is True


def test_wal_retries_outage_and_terminal_drops_remain_distinct() -> None:
    previous = {
        "wal_committed_total": 0,
        "wal_drop_total": 0,
        "wal_retry_total": 0,
        "wal_commit_retry_total": 0,
        "wal_retry_outage_active": 0,
    }
    current = {
        "wal_committed_total": 8,
        "wal_drop_total": 2,
        "wal_retry_total": 4,
        "wal_commit_retry_total": 3,
        "wal_retry_outage_active": 1,
    }

    features = extract_features(previous_metrics=previous, current_metrics=current)

    assert features.wal_retry_rate == 0.4
    assert features.wal_commit_retry_rate == 0.3
    assert features.wal_drop_ratio == 0.2
    assert features.wal_retry_outage_active == 1
    assert features.wal_metrics_available is True


def test_stage_duration_averages_use_sum_and_count_deltas() -> None:
    previous = {
        "decode_duration_seconds_sum": 10,
        "decode_duration_seconds_count": 100,
        "transform_duration_seconds_sum": 20,
        "transform_duration_seconds_count": 100,
        "persist_duration_seconds_sum": 30,
        "persist_duration_seconds_count": 100,
        "influx_write_duration_seconds_sum": 40,
        "influx_write_duration_seconds_count": 100,
    }
    current = {
        "decode_duration_seconds_sum": 14,
        "decode_duration_seconds_count": 102,
        "transform_duration_seconds_sum": 29,
        "transform_duration_seconds_count": 103,
        "persist_duration_seconds_sum": 46,
        "persist_duration_seconds_count": 104,
        "influx_write_duration_seconds_sum": 65,
        "influx_write_duration_seconds_count": 105,
    }

    features = extract_features(previous_metrics=previous, current_metrics=current)

    assert features.decode_duration_average_seconds == 2
    assert features.transform_duration_average_seconds == 3
    assert features.persist_duration_average_seconds == 4
    assert features.influx_write_duration_average_seconds == 5
    assert features.decode_duration_metrics_available is True
    assert features.transform_duration_metrics_available is True
    assert features.persist_duration_metrics_available is True
    assert features.influx_write_duration_metrics_available is True


def test_missing_a2_denominator_is_not_encoded_as_healthy_zero() -> None:
    missing = extract_features(
        previous_metrics={"raw_validation_success_total": 0, "raw_validation_failed_total": 0},
        current_metrics={"raw_validation_success_total": 0, "raw_validation_failed_total": 0},
    )
    present_zero = extract_features(
        previous_metrics={
            "raw_validation_success_total": 0,
            "raw_validation_ignored_total": 0,
            "raw_validation_failed_total": 0,
        },
        current_metrics={
            "raw_validation_success_total": 0,
            "raw_validation_ignored_total": 0,
            "raw_validation_failed_total": 0,
        },
    )

    assert missing.raw_validation_failure_ratio is None
    assert missing.raw_validation_metrics_available is False
    assert present_zero.raw_validation_failure_ratio == 0
    assert present_zero.raw_validation_metrics_available is True


def test_missing_stage_duration_count_is_unavailable() -> None:
    features = extract_features(
        previous_metrics={"decode_duration_seconds_sum": 1},
        current_metrics={"decode_duration_seconds_sum": 2},
    )

    assert features.decode_duration_average_seconds is None
    assert features.decode_duration_metrics_available is False
