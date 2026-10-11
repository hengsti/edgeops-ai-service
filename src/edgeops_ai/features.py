from edgeops_ai.schemas import FeatureVector, IngestionMetrics, ObservationV1

REQUIRED_COUNTER_NAMES = (
    "messages_enqueued_total",
    "messages_processed_total",
    "queue_full_total",
    "transform_success_total",
    "transform_failed_total",
    "influx_lines_written_total",
    "influx_write_failed_total",
    "pipeline_duration_seconds_sum",
    "pipeline_duration_seconds_count",
)

OPTIONAL_COUNTER_NAMES = (
    "influx_write_success_total",
    "sensor_messages_processed_total",
    "status_messages_processed_total",
    "mqtt_messages_received_total",
    "event_queue_full_total",
    "decode_success_total",
    "incoming_oversized_total",
    "incoming_non_utf8_total",
    "incoming_invalid_json_total",
    "raw_validation_success_total",
    "raw_validation_ignored_total",
    "raw_validation_failed_total",
    "transform_attempt_total",
    "transform_deserialize_failed_total",
    "transform_ignored_total",
    "business_validation_success_total",
    "business_validation_failed_total",
    "wal_queue_full_total",
    "wal_queue_closed_total",
    "durability_ack_failed_total",
    "dlq_messages_published_total",
    "dlq_publish_errors_total",
    "wal_committed_total",
    "wal_drop_total",
    "wal_retry_total",
    "wal_commit_retry_total",
    "decode_duration_seconds_sum",
    "decode_duration_seconds_count",
    "transform_duration_seconds_sum",
    "transform_duration_seconds_count",
    "persist_duration_seconds_sum",
    "persist_duration_seconds_count",
    "influx_write_duration_seconds_sum",
    "influx_write_duration_seconds_count",
)


def safe_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    if denominator <= 0.0:
        return default
    return numerator / denominator


def optional_counter_delta(
    previous: IngestionMetrics,
    current: IngestionMetrics,
    name: str,
) -> float | None:
    previous_value = getattr(previous, name)
    current_value = getattr(current, name)
    if previous_value is None or current_value is None:
        return None
    return float(current_value - previous_value)


class FeatureExtractor:
    """
    Keeps exactly one previous observation in memory.

    The first observation cannot produce rates and therefore results in a warming-up response.
    """

    def __init__(self) -> None:
        self._previous: ObservationV1 | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, current: ObservationV1) -> FeatureVector | None:
        previous = self._previous

        if previous is None:
            self._previous = current
            return None

        if self._stream_changed(previous, current):
            self._previous = current
            return None

        elapsed_seconds = (
            current.metadata.captured_at - previous.metadata.captured_at
        ).total_seconds()

        if elapsed_seconds <= 0:
            return None

        if self._counter_reset(previous, current):
            self._previous = current
            return None

        features = self._calculate(previous, current, elapsed_seconds)
        self._previous = current

        return features

    @staticmethod
    def _stream_changed(previous: ObservationV1, current: ObservationV1) -> bool:
        return (
            previous.metadata.source_host != current.metadata.source_host
            or previous.metadata.simulation_run_id != current.metadata.simulation_run_id
        )

    @staticmethod
    def _counter_reset(previous: ObservationV1, current: ObservationV1) -> bool:
        previous_metrics = previous.metrics
        current_metrics = current.metrics

        if any(
            getattr(current_metrics, name) < getattr(previous_metrics, name)
            for name in REQUIRED_COUNTER_NAMES
        ):
            return True

        for name in OPTIONAL_COUNTER_NAMES:
            previous_value = getattr(previous_metrics, name)
            current_value = getattr(current_metrics, name)
            if (
                previous_value is not None
                and current_value is not None
                and current_value < previous_value
            ):
                return True

        return False

    @staticmethod
    def _calculate(
        previous: ObservationV1, current: ObservationV1, elapsed_seconds: float
    ) -> FeatureVector:
        previous_metrics = previous.metrics
        current_metrics = current.metrics

        def delta(name: str) -> float:
            return float(getattr(current_metrics, name) - getattr(previous_metrics, name))

        def optional_delta(name: str) -> float | None:
            return optional_counter_delta(previous_metrics, current_metrics, name)

        def histogram_average(sum_name: str, count_name: str) -> tuple[float | None, bool]:
            sum_delta = optional_delta(sum_name)
            count_delta = optional_delta(count_name)
            if sum_delta is None or count_delta is None:
                return None, False
            return safe_ratio(sum_delta, count_delta), True

        enqueued_delta = delta("messages_enqueued_total")
        processed_delta = delta("messages_processed_total")
        queue_full_delta = delta("queue_full_total")
        queue_depth = current_metrics.queue_depth
        queue_capacity = current_metrics.queue_capacity
        queue_metrics_available = queue_depth is not None and queue_capacity is not None
        queue_utilization_ratio = (
            min(max(safe_ratio(float(queue_depth), float(queue_capacity)), 0.0), 1.0)
            if queue_depth is not None and queue_capacity is not None
            else None
        )

        transform_success_delta = delta("transform_success_total")
        transform_failed_delta = delta("transform_failed_total")
        transform_attempt_delta = optional_delta("transform_attempt_total")
        transform_deserialize_failed_delta = optional_delta("transform_deserialize_failed_total")
        transform_ignored_delta = optional_delta("transform_ignored_total")
        transform_metrics_available = (
            transform_attempt_delta is not None and transform_deserialize_failed_delta is not None
        )
        if transform_attempt_delta is not None and transform_deserialize_failed_delta is not None:
            transform_deserialize_failure_ratio = safe_ratio(
                transform_deserialize_failed_delta,
                transform_attempt_delta,
            )
            transform_failure_ratio = safe_ratio(
                transform_failed_delta + transform_deserialize_failed_delta,
                transform_attempt_delta,
            )
        else:
            transform_total = transform_success_delta + transform_failed_delta
            transform_deserialize_failure_ratio = None
            transform_failure_ratio = safe_ratio(transform_failed_delta, transform_total)
        transform_ignored_ratio = (
            safe_ratio(transform_ignored_delta, transform_attempt_delta)
            if transform_ignored_delta is not None and transform_attempt_delta is not None
            else None
        )

        influx_lines_delta = delta("influx_lines_written_total")
        influx_write_failed_delta = delta("influx_write_failed_total")

        pipeline_duration_delta = delta("pipeline_duration_seconds_sum")
        pipeline_count_delta = delta("pipeline_duration_seconds_count")

        enqueued_rate = enqueued_delta / elapsed_seconds
        processed_rate = processed_delta / elapsed_seconds
        transform_success_rate = transform_success_delta / elapsed_seconds

        processing_ratio = safe_ratio(processed_delta, enqueued_delta, default=1.0)
        raw_persistence_ratio = safe_ratio(
            influx_lines_delta,
            transform_success_delta,
            default=1.0,
        )

        if (
            current.metadata.collector_mode == "production"
            and current_metrics.influxdb_healthy >= 1.0
            and influx_write_failed_delta == 0.0
        ):
            # Production writes are asynchronous and batched through the WAL.
            # No lines in this interval does not indicate a failure when the
            # sink is healthy and no write errors occurred.
            persistence_ratio = 1.0
        else:
            persistence_ratio = min(max(raw_persistence_ratio, 0.0), 1.0)

        pipeline_average = safe_ratio(pipeline_duration_delta, pipeline_count_delta)

        missing_device_ids = [
            device.device_id for device in current.devices if device.heartbeat_missing
        ]
        observed_device_count = len(current.devices)
        missing_device_count = len(missing_device_ids)
        device_metrics_available = observed_device_count > 0
        missing_device_ratio = safe_ratio(missing_device_count, observed_device_count)

        heartbeat_ages = [
            float(device.heartbeat_age_seconds)
            for device in current.devices
            if device.heartbeat_age_seconds is not None
        ]
        heartbeat_age_available = bool(heartbeat_ages)
        heartbeat_age_max_seconds = max(heartbeat_ages) if heartbeat_ages else None
        heartbeat_age_mean_seconds = (
            sum(heartbeat_ages) / len(heartbeat_ages) if heartbeat_ages else None
        )

        success_delta = optional_delta("influx_write_success_total")
        if success_delta is None:
            influx_write_failure_ratio = None
            influx_write_metrics_available = False
        else:
            write_attempts = success_delta + influx_write_failed_delta
            influx_write_failure_ratio = safe_ratio(influx_write_failed_delta, write_attempts)
            influx_write_metrics_available = True

        mqtt_received_delta = optional_delta("mqtt_messages_received_total")
        event_queue_full_delta = optional_delta("event_queue_full_total")
        oversized_delta = optional_delta("incoming_oversized_total")
        non_utf8_delta = optional_delta("incoming_non_utf8_total")
        invalid_json_delta = optional_delta("incoming_invalid_json_total")

        input_queue_metrics_available = (
            mqtt_received_delta is not None and event_queue_full_delta is not None
        )
        if mqtt_received_delta is not None and event_queue_full_delta is not None:
            decode_input_attempts = mqtt_received_delta + event_queue_full_delta
            event_queue_drop_ratio = safe_ratio(event_queue_full_delta, decode_input_attempts)
        else:
            decode_input_attempts = None
            event_queue_drop_ratio = None

        decode_metrics_available = input_queue_metrics_available and all(
            value is not None for value in (oversized_delta, non_utf8_delta, invalid_json_delta)
        )
        decode_failure_ratio = (
            safe_ratio(
                oversized_delta + non_utf8_delta + invalid_json_delta,
                decode_input_attempts,
            )
            if decode_input_attempts is not None
            and oversized_delta is not None
            and non_utf8_delta is not None
            and invalid_json_delta is not None
            else None
        )

        raw_success_delta = optional_delta("raw_validation_success_total")
        raw_ignored_delta = optional_delta("raw_validation_ignored_total")
        raw_failed_delta = optional_delta("raw_validation_failed_total")
        raw_validation_metrics_available = all(
            value is not None for value in (raw_success_delta, raw_ignored_delta, raw_failed_delta)
        )
        if (
            raw_success_delta is not None
            and raw_ignored_delta is not None
            and raw_failed_delta is not None
        ):
            raw_validation_outcomes = raw_success_delta + raw_ignored_delta + raw_failed_delta
            raw_validation_failure_ratio = safe_ratio(
                raw_failed_delta,
                raw_validation_outcomes,
            )
            raw_validation_ignored_ratio = safe_ratio(
                raw_ignored_delta,
                raw_validation_outcomes,
            )
        else:
            raw_validation_failure_ratio = None
            raw_validation_ignored_ratio = None

        business_success_delta = optional_delta("business_validation_success_total")
        business_failed_delta = optional_delta("business_validation_failed_total")
        business_validation_metrics_available = (
            business_success_delta is not None and business_failed_delta is not None
        )
        business_validation_failure_ratio = (
            safe_ratio(
                business_failed_delta,
                business_success_delta + business_failed_delta,
            )
            if business_success_delta is not None and business_failed_delta is not None
            else None
        )

        wal_queue_full_delta = optional_delta("wal_queue_full_total")
        wal_queue_closed_delta = optional_delta("wal_queue_closed_total")
        wal_queue_metrics_available = (
            wal_queue_full_delta is not None and wal_queue_closed_delta is not None
        )
        if wal_queue_full_delta is not None and wal_queue_closed_delta is not None:
            wal_enqueue_attempts = enqueued_delta + wal_queue_full_delta + wal_queue_closed_delta
            wal_queue_full_ratio = safe_ratio(wal_queue_full_delta, wal_enqueue_attempts)
            wal_queue_closed_rate = wal_queue_closed_delta / elapsed_seconds
        else:
            wal_queue_full_ratio = None
            wal_queue_closed_rate = None

        durability_ack_failed_delta = optional_delta("durability_ack_failed_total")
        durability_ack_metrics_available = durability_ack_failed_delta is not None
        durability_ack_failure_rate = (
            durability_ack_failed_delta / elapsed_seconds
            if durability_ack_failed_delta is not None
            else None
        )

        dlq_published_delta = optional_delta("dlq_messages_published_total")
        dlq_errors_delta = optional_delta("dlq_publish_errors_total")
        dlq_metrics_available = dlq_published_delta is not None and dlq_errors_delta is not None
        dlq_publish_failure_ratio = (
            safe_ratio(dlq_errors_delta, dlq_published_delta + dlq_errors_delta)
            if dlq_published_delta is not None and dlq_errors_delta is not None
            else None
        )

        wal_committed_delta = optional_delta("wal_committed_total")
        wal_drop_delta = optional_delta("wal_drop_total")
        wal_retry_delta = optional_delta("wal_retry_total")
        wal_commit_retry_delta = optional_delta("wal_commit_retry_total")
        wal_retry_outage_active = (
            float(current_metrics.wal_retry_outage_active)
            if current_metrics.wal_retry_outage_active is not None
            else None
        )
        wal_retry_rate = wal_retry_delta / elapsed_seconds if wal_retry_delta is not None else None
        wal_commit_retry_rate = (
            wal_commit_retry_delta / elapsed_seconds if wal_commit_retry_delta is not None else None
        )
        wal_drop_ratio = (
            safe_ratio(wal_drop_delta, wal_committed_delta + wal_drop_delta)
            if wal_committed_delta is not None and wal_drop_delta is not None
            else None
        )
        wal_metrics_available = all(
            value is not None
            for value in (
                wal_committed_delta,
                wal_drop_delta,
                wal_retry_delta,
                wal_retry_outage_active,
            )
        )

        decode_duration_average, decode_duration_metrics_available = histogram_average(
            "decode_duration_seconds_sum",
            "decode_duration_seconds_count",
        )
        transform_duration_average, transform_duration_metrics_available = histogram_average(
            "transform_duration_seconds_sum",
            "transform_duration_seconds_count",
        )
        persist_duration_average, persist_duration_metrics_available = histogram_average(
            "persist_duration_seconds_sum",
            "persist_duration_seconds_count",
        )
        influx_write_duration_average, influx_write_duration_metrics_available = histogram_average(
            "influx_write_duration_seconds_sum",
            "influx_write_duration_seconds_count",
        )

        return FeatureVector(
            elapsed_seconds=elapsed_seconds,
            messages_enqueued_rate=max(enqueued_rate, 0.0),
            messages_processed_rate=max(processed_rate, 0.0),
            transform_success_rate=max(transform_success_rate, 0.0),
            queue_growth_rate=enqueued_rate - processed_rate,
            processing_ratio=max(processing_ratio, 0.0),
            queue_full_rate=max(queue_full_delta / elapsed_seconds, 0.0),
            queue_depth=queue_depth,
            queue_capacity=queue_capacity,
            queue_utilization_ratio=queue_utilization_ratio,
            queue_metrics_available=queue_metrics_available,
            transform_failure_ratio=max(transform_failure_ratio, 0.0),
            transform_deserialize_failure_ratio=transform_deserialize_failure_ratio,
            transform_ignored_ratio=transform_ignored_ratio,
            transform_metrics_available=transform_metrics_available,
            decode_failure_ratio=decode_failure_ratio,
            decode_metrics_available=decode_metrics_available,
            raw_validation_failure_ratio=raw_validation_failure_ratio,
            raw_validation_ignored_ratio=raw_validation_ignored_ratio,
            raw_validation_metrics_available=raw_validation_metrics_available,
            business_validation_failure_ratio=business_validation_failure_ratio,
            business_validation_metrics_available=business_validation_metrics_available,
            event_queue_drop_ratio=event_queue_drop_ratio,
            input_queue_metrics_available=input_queue_metrics_available,
            wal_queue_full_ratio=wal_queue_full_ratio,
            wal_queue_closed_rate=wal_queue_closed_rate,
            wal_queue_metrics_available=wal_queue_metrics_available,
            durability_ack_failure_rate=durability_ack_failure_rate,
            durability_ack_metrics_available=durability_ack_metrics_available,
            dlq_publish_failure_ratio=dlq_publish_failure_ratio,
            dlq_metrics_available=dlq_metrics_available,
            wal_retry_rate=wal_retry_rate,
            wal_commit_retry_rate=wal_commit_retry_rate,
            wal_drop_ratio=wal_drop_ratio,
            wal_retry_outage_active=wal_retry_outage_active,
            wal_metrics_available=wal_metrics_available,
            influx_write_failure_ratio=influx_write_failure_ratio,
            influx_write_metrics_available=influx_write_metrics_available,
            influxdb_healthy=float(current_metrics.influxdb_healthy),
            persistence_ratio=max(persistence_ratio, 0.0),
            pipeline_duration_average_seconds=max(pipeline_average, 0.0),
            decode_duration_average_seconds=decode_duration_average,
            decode_duration_metrics_available=decode_duration_metrics_available,
            transform_duration_average_seconds=transform_duration_average,
            transform_duration_metrics_available=transform_duration_metrics_available,
            persist_duration_average_seconds=persist_duration_average,
            persist_duration_metrics_available=persist_duration_metrics_available,
            influx_write_duration_average_seconds=influx_write_duration_average,
            influx_write_duration_metrics_available=influx_write_duration_metrics_available,
            heartbeat_missing=bool(missing_device_ids),
            heartbeat_age_max_seconds=heartbeat_age_max_seconds,
            heartbeat_age_mean_seconds=heartbeat_age_mean_seconds,
            heartbeat_age_available=heartbeat_age_available,
            missing_device_ids=missing_device_ids,
            observed_device_count=observed_device_count,
            missing_device_count=missing_device_count,
            missing_device_ratio=missing_device_ratio,
            device_metrics_available=device_metrics_available,
        )
