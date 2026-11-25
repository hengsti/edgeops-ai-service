from edgeops_ai.prediction import Prediction, PredictionBackendName, recommendation_for
from edgeops_ai.schemas import FeatureVector, PredictionLabel

RULES_MODEL_VERSION = "rules-0.1.0"
INGEST_V3_RULES_VERSION = "rules-ingest-v3-0.1.0"


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
            and features.persistence_ratio is not None
            and features.persistence_ratio < self.MIN_PERSISTENCE_RATIO
        ):
            label = "influxdb-write-failure"
        elif (features.transform_failure_ratio or 0) >= self.TRANSFORM_FAILURE_RATIO:
            label = "malformed-sensor-payload"
        elif (features.queue_full_rate or 0) >= self.QUEUE_FULL_RATE or (
            (features.queue_growth_rate or 0) > self.QUEUE_GROWTH_RATE
            and features.processing_ratio is not None
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

    def predict_ingest_v3(self, features: FeatureVector) -> Prediction | None:
        ingest = features.ingest_v3
        assert ingest is not None
        label: PredictionLabel
        if features.broker_available is False:
            label = "mqtt-broker-unavailable"
        elif features.heartbeat_missing:
            label = "missing-device-heartbeat"
        elif ingest.sink_state in {"blocked", "retrying"} or features.wal_retry_outage_active == 1:
            label = "influxdb-write-failure"
        elif (ingest.input_rejection_rate or 0) > 0:
            label = "malformed-sensor-payload"
        elif (
            ingest.admission_pressure == 1
            or (features.queue_full_rate or 0) >= self.QUEUE_FULL_RATE
            or (features.wal_queue_full_ratio or 0) > 0
        ):
            label = "ingestion-backpressure"
        elif (
            ingest.received_rate == 0
            and ingest.forwarder_alive == 1
            and ingest.sink_state == "healthy"
            and ingest.admission_pressure == 0
        ):
            label = "normal"
        else:
            return None
        return Prediction(
            label=label,
            confidence=1.0,
            recommendation=(
                "Ingest is idle, the sink is healthy, and admission pressure is clear."
                if label == "normal"
                else recommendation_for(label, features)
            ),
            model_version=INGEST_V3_RULES_VERSION,
        )
