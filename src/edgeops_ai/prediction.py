from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from edgeops_ai.schemas import FeatureVector, PredictionLabel


class PredictionBackendName(StrEnum):
    RULES = "rules"
    ML = "ml"


@dataclass(frozen=True, slots=True)
class Prediction:
    label: PredictionLabel
    confidence: float
    recommendation: str
    model_version: str


class Predictor(Protocol):
    """Any prediction method the service can serve."""

    @property
    def backend_name(self) -> PredictionBackendName: ...

    @property
    def model_version(self) -> str: ...

    def predict(self, features: FeatureVector) -> Prediction: ...


RECOMMENDATIONS: dict[str, str] = {
    "normal": "No issues detected.",
    "influxdb-write-failure": "Check InfluxDB health, connectivity and ingestion write logs.",
    "ingestion-backpressure": "Inspect ingestion queue pressure and processing throughput.",
    "malformed-sensor-payload": "Inspect DLQ messages and validate sensor payload schemas.",
    "missing-device-heartbeat": "Check device connectivity and power state.",
    "mqtt-broker-unavailable": "Check the EMQX broker: container state, health and logs.",
}


def recommendation_for(label: str, features: FeatureVector) -> str:
    if label == "missing-device-heartbeat" and features.missing_device_ids:
        devices = ", ".join(features.missing_device_ids)
        return f"Check connectivity and power state of: {devices}."
    return RECOMMENDATIONS[label]
