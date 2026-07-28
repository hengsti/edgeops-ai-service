from edgeops_ai.schemas import FeatureVector, ObservationV1


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

        counter_names = (
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

        return any(
            getattr(current_metrics, name) < getattr(previous_metrics, name)
            for name in counter_names
        )

    @staticmethod
    def _calculate(
        previous: ObservationV1, current: ObservationV1, elapsed_seconds: float
    ) -> FeatureVector:
        previous_metrics = previous.metrics
        current_metrics = current.metrics

        def delta(name: str) -> float:
            return float(getattr(current_metrics, name) - getattr(previous_metrics, name))

        enqueued_delta = delta("messages_enqueued_total")
        processed_delta = delta("messages_processed_total")
        queue_full_delta = delta("queue_full_total")
        queue_depth = current_metrics.ingest_queue_depth
        queue_capacity = current_metrics.ingest_queue_capacity
        queue_metrics_available = queue_depth is not None and queue_capacity is not None
        queue_utilization_ratio = (
            min(max(float(queue_depth) / float(queue_capacity), 0.0), 1.0)
            if queue_metrics_available
            else None
        )

        transform_success_delta = delta("transform_success_total")
        transform_failed_delta = delta("transform_failed_total")

        influx_lines_delta = delta("influx_lines_written_total")
        influx_write_failed_delta = delta("influx_write_failed_total")

        pipeline_duration_delta = delta("pipeline_duration_seconds_sum")
        pipeline_count_delta = delta("pipeline_duration_seconds_count")

        enqueued_rate = enqueued_delta / elapsed_seconds
        processed_rate = processed_delta / elapsed_seconds
        transform_success_rate = transform_success_delta / elapsed_seconds

        processing_ratio = processed_delta / enqueued_delta if enqueued_delta > 0 else 1.0

        transform_total = transform_success_delta + transform_failed_delta
        transform_failure_ratio = (
            transform_failed_delta / transform_total if transform_total > 0 else 0.0
        )

        raw_persistence_ratio = (
            influx_lines_delta / transform_success_delta if transform_success_delta > 0 else 1.0
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
            persistence_ratio = min(
                max(raw_persistence_ratio, 0.0),
                1.0,
            )

        pipeline_average = (
            pipeline_duration_delta / pipeline_count_delta if pipeline_count_delta > 0 else 0.0
        )

        missing_device_ids = [
            device.device_id for device in current.devices if device.heartbeat_missing
        ]
        observed_device_count = len(current.devices)
        missing_device_count = len(missing_device_ids)
        device_metrics_available = observed_device_count > 0

        missing_device_ratio = (
            (missing_device_count / observed_device_count) if observed_device_count > 0 else 0.0
        )

        heartbeat_ages = [
            float(device.heartbeat_age_seconds)
            for device in current.devices
            if device.heartbeat_age_seconds is not None
        ]
        heartbeat_age_max_available = bool(heartbeat_ages)
        heartbeat_age_max_seconds = max(heartbeat_ages) if heartbeat_ages else None
        heartbeat_age_mean_seconds = (
            sum(heartbeat_ages) / len(heartbeat_ages) if heartbeat_ages else None
        )

        def optional_delta(name: str) -> float | None:
            if getattr(previous_metrics, name) is None or getattr(current_metrics, name) is None:
                return None
            return float(getattr(current_metrics, name) - getattr(previous_metrics, name))

        success_delta = optional_delta("influx_write_success_total")

        if success_delta is None:
            influx_write_failure_ratio = None
            influx_write_metrics_available = False
        else:
            write_attempts = success_delta + influx_write_failed_delta
            influx_write_failure_ratio = (
                influx_write_failed_delta / write_attempts if write_attempts > 0 else 0.0
            )
            influx_write_metrics_available = True

        return FeatureVector(
            elapsed_seconds=elapsed_seconds,
            messages_enqueued_rate=max(enqueued_rate, 0.0),
            messages_processed_rate=max(processed_rate, 0.0),
            transform_success_rate=max(transform_success_rate, 0.0),
            queue_growth_rate=enqueued_rate - processed_rate,
            processing_ratio=max(processing_ratio, 0.0),
            queue_full_rate=max(queue_full_delta / elapsed_seconds, 0.0),
            transform_failure_ratio=max(transform_failure_ratio, 0.0),
            persistence_ratio=max(persistence_ratio, 0.0),
            pipeline_duration_average_seconds=max(pipeline_average, 0.0),
            heartbeat_missing=bool(missing_device_ids),
            missing_device_ids=missing_device_ids,
            observed_device_count=observed_device_count,
            missing_device_count=missing_device_count,
            missing_device_ratio=missing_device_ratio,
            device_metrics_available=device_metrics_available,
            queue_depth=queue_depth,
            queue_capacity=queue_capacity,
            queue_utilization_ratio=queue_utilization_ratio,
            queue_metrics_available=queue_metrics_available,
            influx_write_failure_ratio=influx_write_failure_ratio,
            influx_write_metrics_available=influx_write_metrics_available,
            influxdb_healthy=float(current_metrics.influxdb_healthy),
            heartbeat_age_max_seconds=heartbeat_age_max_seconds,
            heartbeat_age_mean_seconds=heartbeat_age_mean_seconds,
            heartbeat_age_available=heartbeat_age_max_available,
        )
