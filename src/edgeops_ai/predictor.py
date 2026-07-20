from dataclasses import dataclass

from edgeops_ai.schemas import FeatureVector, PredictionLabel

MODEL_VERSION = "rules-0.1.0"


@dataclass(frozen=True)
class Prediction:
    label: PredictionLabel
    confidence: float
    recommendation: str


class RuleBasedPredictor:
    """
    Transparent baseline for evaluating the later ML Model.

    The thresholds are intentionally kept in one place and should initially be calibrated against the collector scenarios.
    """

    TRANSFORM_FAILURE_RATIO = 0.01
    MIN_PERSISTENCE_RATIO = 0.20

    QUEUE_GROWTH_RATE = 10.0
    MIN_PROCESSING_RATIO = 0.98
    QUEUE_FULL_RATE = 0.10

    def predict(self, features: FeatureVector) -> Prediction:
        if features.heartbeat_missing:
            devices = ", ".join(features.missing_device_ids)

            return Prediction(
                label="missing-device-heartbeat",
                confidence=1.0,
                recommendation=f"Check connectivity and power state of: {devices}.",
            )

        if (
            features.transform_success_rate > 1.0
            and features.persistence_ratio < self.MIN_PERSISTENCE_RATIO
        ):
            return Prediction(
                label="influxdb-write-failure",
                confidence=1.0,
                recommendation="Check InfluxDB health, connectivity and ingestion write logs.",
            )

        if features.transform_failure_ratio >= self.TRANSFORM_FAILURE_RATIO:
            return Prediction(
                label="malformed-sensor-payload",
                confidence=1.0,
                recommendation="Inspect DLQ messages and validate sensor payload schemas.",
            )

        backpressure_detected = features.queue_full_rate >= self.QUEUE_FULL_RATE or (
            features.queue_growth_rate > self.QUEUE_GROWTH_RATE
            and features.processing_ratio < self.MIN_PROCESSING_RATIO
        )

        if backpressure_detected:
            return Prediction(
                label="ingestion-backpressure",
                confidence=1.0,
                recommendation="Inspect ingestion queue pressure and processing throughput.",
            )

        return Prediction(
            label="normal",
            confidence=1.0,
            recommendation="No issues detected.",
        )
