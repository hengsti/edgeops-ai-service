from dataclasses import dataclass

from edgeops_ai.schemas import IngestionMetrics, ObservationV1


def safe_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    return default if denominator <= 0.0 else numerator / denominator


def optional_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return safe_ratio(numerator, denominator)


@dataclass(frozen=True)
class MetricWindow:
    previous: IngestionMetrics
    current: IngestionMetrics
    elapsed_seconds: float

    def delta(self, name: str) -> float:
        return float(getattr(self.current, name) - getattr(self.previous, name))

    def optional_delta(self, name: str) -> float | None:
        previous = getattr(self.previous, name)
        current = getattr(self.current, name)
        return None if previous is None or current is None else float(current - previous)

    def total(self, *names: str) -> float | None:
        values = [self.optional_delta(name) for name in names]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    def rate(self, name: str) -> float | None:
        value = self.optional_delta(name)
        return None if value is None else value / self.elapsed_seconds


@dataclass(frozen=True)
class QueueFeatures:
    messages_enqueued_rate: float
    messages_processed_rate: float
    queue_growth_rate: float
    processing_ratio: float
    queue_full_rate: float
    queue_depth: float | None
    queue_capacity: float | None
    queue_utilization_ratio: float | None
    event_queue_drop_ratio: float | None
    decode_failure_ratio: float | None


def queue_features(window: MetricWindow) -> QueueFeatures:
    enqueued = window.delta("messages_enqueued_total")
    processed = window.delta("messages_processed_total")
    utilization = optional_ratio(window.current.queue_depth, window.current.queue_capacity)
    input_attempts = window.total("mqtt_messages_received_total", "event_queue_full_total")
    return QueueFeatures(
        messages_enqueued_rate=max(enqueued / window.elapsed_seconds, 0.0),
        messages_processed_rate=max(processed / window.elapsed_seconds, 0.0),
        queue_growth_rate=enqueued / window.elapsed_seconds - processed / window.elapsed_seconds,
        processing_ratio=max(safe_ratio(processed, enqueued, default=1.0), 0.0),
        queue_full_rate=max(window.delta("queue_full_total") / window.elapsed_seconds, 0.0),
        queue_depth=window.current.queue_depth,
        queue_capacity=window.current.queue_capacity,
        queue_utilization_ratio=None if utilization is None else min(max(utilization, 0.0), 1.0),
        event_queue_drop_ratio=optional_ratio(
            window.optional_delta("event_queue_full_total"), input_attempts
        ),
        decode_failure_ratio=optional_ratio(
            window.total(
                "incoming_oversized_total", "incoming_non_utf8_total", "incoming_invalid_json_total"
            ),
            input_attempts,
        ),
    )


@dataclass(frozen=True)
class TransformFeatures:
    transform_success_rate: float
    transform_failure_ratio: float
    transform_deserialize_failure_ratio: float | None
    transform_ignored_ratio: float | None
    raw_validation_failure_ratio: float | None
    raw_validation_ignored_ratio: float | None
    business_validation_failure_ratio: float | None


def transform_features(window: MetricWindow) -> TransformFeatures:
    success = window.delta("transform_success_total")
    failed = window.delta("transform_failed_total")
    attempts = window.optional_delta("transform_attempt_total")
    deserialize_failed = window.optional_delta("transform_deserialize_failed_total")
    failure_ratio = (
        safe_ratio(failed + deserialize_failed, attempts)
        if attempts is not None and deserialize_failed is not None
        else safe_ratio(failed, success + failed)
    )
    raw_outcomes = window.total(
        "raw_validation_success_total",
        "raw_validation_ignored_total",
        "raw_validation_failed_total",
    )
    return TransformFeatures(
        transform_success_rate=max(success / window.elapsed_seconds, 0.0),
        transform_failure_ratio=max(failure_ratio, 0.0),
        transform_deserialize_failure_ratio=optional_ratio(deserialize_failed, attempts),
        transform_ignored_ratio=optional_ratio(
            window.optional_delta("transform_ignored_total"), attempts
        ),
        raw_validation_failure_ratio=optional_ratio(
            window.optional_delta("raw_validation_failed_total"), raw_outcomes
        ),
        raw_validation_ignored_ratio=optional_ratio(
            window.optional_delta("raw_validation_ignored_total"), raw_outcomes
        ),
        business_validation_failure_ratio=optional_ratio(
            window.optional_delta("business_validation_failed_total"),
            window.total("business_validation_success_total", "business_validation_failed_total"),
        ),
    )


@dataclass(frozen=True)
class InfluxFeatures:
    influx_write_failure_ratio: float | None
    influxdb_healthy: float
    persistence_ratio: float


def influx_features(window: MetricWindow) -> InfluxFeatures:
    write_failed = window.delta("influx_write_failed_total")
    persistence = safe_ratio(
        window.delta("influx_lines_written_total"),
        window.delta("transform_success_total"),
        default=1.0,
    )
    if window.current.influxdb_healthy >= 1.0 and write_failed == 0.0:
        persistence = 1.0
    return InfluxFeatures(
        influx_write_failure_ratio=optional_ratio(
            write_failed, window.total("influx_write_success_total", "influx_write_failed_total")
        ),
        influxdb_healthy=float(window.current.influxdb_healthy),
        persistence_ratio=min(max(persistence, 0.0), 1.0),
    )


@dataclass(frozen=True)
class DeviceFeatures:
    heartbeat_missing: bool
    heartbeat_age_max_seconds: float | None
    heartbeat_age_mean_seconds: float | None
    missing_device_ids: list[str]
    observed_device_count: int
    missing_device_count: int
    missing_device_ratio: float


def device_features(current: ObservationV1) -> DeviceFeatures:
    missing_ids = [device.device_id for device in current.devices if device.heartbeat_missing]
    ages = [
        float(device.heartbeat_age_seconds)
        for device in current.devices
        if device.heartbeat_age_seconds is not None
    ]
    return DeviceFeatures(
        heartbeat_missing=bool(missing_ids),
        heartbeat_age_max_seconds=max(ages) if ages else None,
        heartbeat_age_mean_seconds=sum(ages) / len(ages) if ages else None,
        missing_device_ids=missing_ids,
        observed_device_count=len(current.devices),
        missing_device_count=len(missing_ids),
        missing_device_ratio=safe_ratio(len(missing_ids), len(current.devices)),
    )


@dataclass(frozen=True)
class WalFeatures:
    wal_queue_full_ratio: float | None
    wal_queue_closed_rate: float | None
    durability_ack_failure_rate: float | None
    dlq_publish_failure_ratio: float | None
    wal_retry_rate: float | None
    wal_commit_retry_rate: float | None
    wal_drop_ratio: float | None
    wal_retry_outage_active: float | None


def wal_features(window: MetricWindow) -> WalFeatures:
    queue_attempts = window.total(
        "messages_enqueued_total", "wal_queue_full_total", "wal_queue_closed_total"
    )
    return WalFeatures(
        wal_queue_full_ratio=optional_ratio(
            window.optional_delta("wal_queue_full_total"), queue_attempts
        ),
        wal_queue_closed_rate=window.rate("wal_queue_closed_total")
        if queue_attempts is not None
        else None,
        durability_ack_failure_rate=window.rate("durability_ack_failed_total"),
        dlq_publish_failure_ratio=optional_ratio(
            window.optional_delta("dlq_publish_errors_total"),
            window.total("dlq_messages_published_total", "dlq_publish_errors_total"),
        ),
        wal_retry_rate=window.rate("wal_retry_total"),
        wal_commit_retry_rate=window.rate("wal_commit_retry_total"),
        wal_drop_ratio=optional_ratio(
            window.optional_delta("wal_drop_total"),
            window.total("wal_committed_total", "wal_drop_total"),
        ),
        wal_retry_outage_active=window.current.wal_retry_outage_active,
    )


@dataclass(frozen=True)
class DurationFeatures:
    pipeline_duration_average_seconds: float
    decode_duration_average_seconds: float | None
    transform_duration_average_seconds: float | None
    persist_duration_average_seconds: float | None
    influx_write_duration_average_seconds: float | None


def duration_features(window: MetricWindow) -> DurationFeatures:
    def average(stage: str) -> float | None:
        return optional_ratio(
            window.optional_delta(f"{stage}_duration_seconds_sum"),
            window.optional_delta(f"{stage}_duration_seconds_count"),
        )

    return DurationFeatures(
        pipeline_duration_average_seconds=max(
            safe_ratio(
                window.delta("pipeline_duration_seconds_sum"),
                window.delta("pipeline_duration_seconds_count"),
            ),
            0.0,
        ),
        decode_duration_average_seconds=average("decode"),
        transform_duration_average_seconds=average("transform"),
        persist_duration_average_seconds=average("persist"),
        influx_write_duration_average_seconds=average("influx_write"),
    )
