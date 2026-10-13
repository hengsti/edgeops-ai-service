from edgeops_ai.prediction import Prediction, PredictionBackendName, recommendation_for
from edgeops_ai.schemas import FeatureVector, PredictionLabel

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
        label: PredictionLabel
        # A confirmed broker outage explains stale heartbeats, so it must win over them.
        if features.broker_available is False:
            label = "mqtt-broker-unavailable"
        elif features.heartbeat_missing:
            label = "missing-device-heartbeat"
        elif (
            features.transform_success_rate > 1.0
            and features.persistence_ratio < self.MIN_PERSISTENCE_RATIO
        ):
            label = "influxdb-write-failure"
        elif features.transform_failure_ratio >= self.TRANSFORM_FAILURE_RATIO:
            label = "malformed-sensor-payload"
        elif features.queue_full_rate >= self.QUEUE_FULL_RATE or (
            features.queue_growth_rate > self.QUEUE_GROWTH_RATE
            and features.processing_ratio < self.MIN_PROCESSING_RATIO
        ):
            label = "ingestion-backpressure"
        else:
            label = "normal"
        return Prediction(
            label=label,
            confidence=1.0,
            recommendation=recommendation_for(label, features),
            model_version=RULES_MODEL_VERSION,
        )
