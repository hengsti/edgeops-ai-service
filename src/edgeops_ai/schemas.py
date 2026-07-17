from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, NonNegativeFloat


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationMetadata(StrictModel):
    captured_at: AwareDatetime
    collector_mode: Literal["production", "simulation"]
    simulated: bool
    source_host: str = Field(min_length=1)

    scenario_id: str | None = None
    simulation_run_id: str | None = None
    simulation_seed: int | None = None
    simulation_phase: str | None = None


class IngestionMetrics(StrictModel):
    messages_enqueued_total: NonNegativeFloat
    messages_processed_total: NonNegativeFloat
    queue_full_total: NonNegativeFloat

    transform_success_total: NonNegativeFloat
    transform_failed_total: NonNegativeFloat

    influx_lines_written_total: NonNegativeFloat

    pipeline_duration_seconds_sum: NonNegativeFloat
    pipeline_duration_seconds_count: NonNegativeFloat


class DeviceObservation(StrictModel):
    device_id: str = Field(min_length=1)
    heartbeat_missing: bool
    heartbeat_age_seconds: NonNegativeFloat | None = None


class ObservationV1(StrictModel):
    schema_version: Literal["1.0"]
    metadata: ObservationMetadata
    metrics: IngestionMetrics
    devices: list[DeviceObservation] = Field(min_length=1)


class FeatureVector(StrictModel):
    elapsed_seconds: float = Field(gt=0)

    messages_enqueued_rate: NonNegativeFloat
    messages_processed_rate: NonNegativeFloat
    transform_success_rate: NonNegativeFloat

    queue_growth_rate: float
    processing_ratio: NonNegativeFloat
    queue_full_rate: NonNegativeFloat

    transform_failure_ratio: NonNegativeFloat
    persistence_ratio: NonNegativeFloat
    pipeline_duration_average_seconds: NonNegativeFloat

    heartbeat_missing: bool
    missing_device_ids: list[str] = Field(default_factory=list)


PredictionLabel = Literal[
    "normal",
    "influxdb-write-failure",
    "ingestion-backpressure",
    "malformed-sensor-payload",
    "missing-device-heartbeat",
]


class AnalysisResult(StrictModel):
    status: Literal["warming-up", "predicted"]
    captured_at: AwareDatetime

    prediction: PredictionLabel | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_version: str

    recommendation: str | None = None
    features: FeatureVector | None = None

    scenario_id: str | None = None
    simulation_run_id: str | None = None
    simulation_phase: str | None = None
