from edgeops_ai.prediction import Prediction, PredictionBackendName, recommendation_for
from edgeops_ai.schemas import FeatureVector

RULES_MODEL_VERSION = "rules-0.1.0"


class RuleBasedPredictor:
    """
    Transparent baseline for evaluating the later ML Model.

    The thresholds are intentionally kept in one place and should initially be calibrated against
    the collector scenarios.
    """

    TRANSFORM_FAILURE_RATIO = 0.01
    MIN_PERSISTENCE_RATIO = 0.20

    QUEUE_GROWTH_RATE = 10.0
    MIN_PROCESSING_RATIO = 0.98
    QUEUE_FULL_RATE = 0.10

    @property
    def backend_name(self) -> PredictionBackendName:
        return PredictionBackendName.RULES

    @property
    def model_version(self) -> str:
        return RULES_MODEL_VERSION

    def predict(self, features: FeatureVector) -> Prediction:
        if features.heartbeat_missing:
            return Prediction(
                label="missing-device-heartbeat",
                confidence=1.0,
                recommendation=recommendation_for("missing-device-heartbeat", features),
                model_version=RULES_MODEL_VERSION,
            )

        if (
            features.transform_success_rate > 1.0
            and features.persistence_ratio < self.MIN_PERSISTENCE_RATIO
        ):
            return Prediction(
                label="influxdb-write-failure",
                confidence=1.0,
                recommendation=recommendation_for("influxdb-write-failure", features),
                model_version=RULES_MODEL_VERSION,
            )

        if features.transform_failure_ratio >= self.TRANSFORM_FAILURE_RATIO:
            return Prediction(
                label="malformed-sensor-payload",
                confidence=1.0,
                recommendation=recommendation_for("malformed-sensor-payload", features),
                model_version=RULES_MODEL_VERSION,
            )

        backpressure_detected = features.queue_full_rate >= self.QUEUE_FULL_RATE or (
            features.queue_growth_rate > self.QUEUE_GROWTH_RATE
            and features.processing_ratio < self.MIN_PROCESSING_RATIO
        )

        if backpressure_detected:
            return Prediction(
                label="ingestion-backpressure",
                confidence=1.0,
                recommendation=recommendation_for("ingestion-backpressure", features),
                model_version=RULES_MODEL_VERSION,
            )

        return Prediction(
            label="normal",
            confidence=1.0,
            recommendation=recommendation_for("normal", features),
            model_version=RULES_MODEL_VERSION,
        )
