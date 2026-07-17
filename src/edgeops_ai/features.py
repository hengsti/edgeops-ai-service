from edgeops_ai.schemas import FeatureVector, ObservationV1


class FeatureExtractor:
    """
    Keeps exactly on previous observation in memory.

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

        transform_success_delta = delta("transform_success_total")
        transform_failed_delta = delta("transform_failed_total")

        influx_lines_delta = delta("influx_lines_written_total")

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

        persistence_ratio = (
            influx_lines_delta / transform_success_delta if transform_success_delta > 0 else 1.0
        )

        pipeline_average = (
            pipeline_duration_delta / pipeline_count_delta if pipeline_count_delta > 0 else 0.0
        )

        missing_device_ids = [
            device.device_id for device in current.devices if device.heartbeat_missing
        ]

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
        )
