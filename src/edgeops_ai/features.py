from dataclasses import asdict

from edgeops_ai.feature_groups import (
    MetricWindow,
    device_features,
    duration_features,
    influx_features,
    queue_features,
    transform_features,
    wal_features,
)
from edgeops_ai.feature_groups import (
    safe_ratio as safe_ratio,
)
from edgeops_ai.schemas import (
    OPTIONAL_COUNTER_NAMES,
    REQUIRED_COUNTER_NAMES,
    FeatureVector,
    ObservationV1,
)


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
        window = MetricWindow(previous.metrics, current.metrics, elapsed_seconds)
        queue = queue_features(window)
        transform = transform_features(window)
        influx = influx_features(window)
        devices = device_features(current)
        wal = wal_features(window)
        durations = duration_features(window)

        values = {
            name: value
            for group in (queue, transform, influx, devices, wal, durations)
            for name, value in asdict(group).items()
        }
        return FeatureVector(
            **values,
            elapsed_seconds=elapsed_seconds,
            queue_metrics_available=queue.queue_utilization_ratio is not None,
            transform_metrics_available=transform.transform_deserialize_failure_ratio is not None,
            decode_metrics_available=queue.decode_failure_ratio is not None,
            raw_validation_metrics_available=transform.raw_validation_failure_ratio is not None,
            business_validation_metrics_available=transform.business_validation_failure_ratio
            is not None,
            input_queue_metrics_available=queue.event_queue_drop_ratio is not None,
            wal_queue_metrics_available=wal.wal_queue_full_ratio is not None,
            durability_ack_metrics_available=wal.durability_ack_failure_rate is not None,
            dlq_metrics_available=wal.dlq_publish_failure_ratio is not None,
            wal_metrics_available=all(
                value is not None
                for value in (
                    wal.wal_retry_rate,
                    wal.wal_drop_ratio,
                    wal.wal_retry_outage_active,
                )
            ),
            influx_write_metrics_available=influx.influx_write_failure_ratio is not None,
            decode_duration_metrics_available=durations.decode_duration_average_seconds is not None,
            transform_duration_metrics_available=durations.transform_duration_average_seconds
            is not None,
            persist_duration_metrics_available=durations.persist_duration_average_seconds
            is not None,
            influx_write_duration_metrics_available=durations.influx_write_duration_average_seconds
            is not None,
            heartbeat_age_available=devices.heartbeat_age_max_seconds is not None,
            device_metrics_available=devices.observed_device_count > 0,
            broker_available=current.broker_available,
        )
