from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
)


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


@dataclass(frozen=True)
class Metric:
    source_name: str
    kind: Literal["counter", "gauge"] = "counter"
    optional_zero: bool = False


class IngestionMetrics(StrictModel):
    messages_enqueued_total: Annotated[NonNegativeFloat, Metric("ingest_messages_enqueued_total")]
    messages_processed_total: Annotated[NonNegativeFloat, Metric("ingest_messages_processed_total")]
    queue_full_total: Annotated[
        NonNegativeFloat, Metric("ingest_event_queue_full_total", optional_zero=True)
    ]

    transform_success_total: Annotated[NonNegativeFloat, Metric("ingest_transform_success_total")]
    transform_failed_total: Annotated[NonNegativeFloat, Metric("ingest_transform_failed_total")]

    influx_lines_written_total: NonNegativeFloat
    influx_write_failed_total: NonNegativeFloat = 0.0
    influxdb_healthy: Annotated[float, Metric("influxdb_healthy", kind="gauge")] = Field(
        default=1.0, ge=0, le=1
    )

    pipeline_duration_seconds_sum: Annotated[
        NonNegativeFloat, Metric("ingest_pipeline_duration_seconds_sum")
    ]
    pipeline_duration_seconds_count: Annotated[
        NonNegativeFloat, Metric("ingest_pipeline_duration_seconds_count")
    ]

    influx_write_success_total: NonNegativeFloat | None = None
    queue_depth: Annotated[NonNegativeFloat | None, Metric("ingest_queue_depth", kind="gauge")] = (
        None
    )
    queue_capacity: Annotated[
        NonNegativeFloat | None, Metric("ingest_queue_capacity", kind="gauge")
    ] = Field(default=None, gt=0)
    sensor_messages_processed_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_sensor_messages_processed_total")
    ] = None
    status_messages_processed_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_status_messages_processed_total")
    ] = None

    mqtt_messages_received_total: NonNegativeFloat | None = None
    event_queue_full_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_event_queue_full_total")
    ] = None
    decode_success_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_decode_success_total")
    ] = None
    incoming_oversized_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_incoming_oversized_total")
    ] = None
    incoming_non_utf8_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_incoming_non_utf8_total")
    ] = None
    incoming_invalid_json_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_incoming_invalid_json_total")
    ] = None
    raw_validation_success_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_validate_raw_success_total")
    ] = None
    raw_validation_ignored_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_validate_raw_ignored_total")
    ] = None
    raw_validation_failed_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_validate_raw_failed_total")
    ] = None
    transform_attempt_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_transform_attempt_total")
    ] = None
    transform_deserialize_failed_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_transform_deserialize_failed_total")
    ] = None
    transform_ignored_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_transform_ignored_total")
    ] = None
    business_validation_success_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_validate_business_success_total")
    ] = None
    business_validation_failed_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_validate_business_failed_total")
    ] = None
    wal_queue_full_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_wal_queue_full_total")
    ] = None
    wal_queue_closed_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_queue_closed_total")
    ] = None
    durability_ack_failed_total: Annotated[
        NonNegativeFloat | None, Metric("ingest_durability_ack_failed_total")
    ] = None
    dlq_messages_published_total: NonNegativeFloat | None = None
    dlq_publish_errors_total: NonNegativeFloat | None = None
    wal_committed_total: Annotated[
        NonNegativeFloat | None, Metric("wal_forwarder_committed_total")
    ] = None
    wal_drop_total: Annotated[NonNegativeFloat | None, Metric("wal_forwarder_drop_total")] = None
    wal_retry_total: Annotated[NonNegativeFloat | None, Metric("wal_forwarder_retry_total")] = None
    wal_commit_retry_total: Annotated[
        NonNegativeFloat | None, Metric("wal_forwarder_commit_retry_total")
    ] = None
    wal_retry_outage_active: Annotated[
        float | None, Metric("wal_forwarder_retry_outage_active", kind="gauge")
    ] = Field(default=None, ge=0, le=1)

    decode_duration_seconds_sum: Annotated[
        NonNegativeFloat | None, Metric("ingest_decode_duration_seconds_sum")
    ] = None
    decode_duration_seconds_count: Annotated[
        NonNegativeFloat | None, Metric("ingest_decode_duration_seconds_count")
    ] = None
    transform_duration_seconds_sum: Annotated[
        NonNegativeFloat | None, Metric("ingest_transform_duration_seconds_sum")
    ] = None
    transform_duration_seconds_count: Annotated[
        NonNegativeFloat | None, Metric("ingest_transform_duration_seconds_count")
    ] = None
    persist_duration_seconds_sum: Annotated[
        NonNegativeFloat | None, Metric("ingest_persist_duration_seconds_sum")
    ] = None
    persist_duration_seconds_count: Annotated[
        NonNegativeFloat | None, Metric("ingest_persist_duration_seconds_count")
    ] = None
    influx_write_duration_seconds_sum: NonNegativeFloat | None = None
    influx_write_duration_seconds_count: NonNegativeFloat | None = None


METRICS = {
    name: next((item for item in field.metadata if isinstance(item, Metric)), Metric(name))
    for name, field in IngestionMetrics.model_fields.items()
}
REQUIRED_METRIC_MAPPING = {
    metric.source_name: name
    for name, metric in METRICS.items()
    if IngestionMetrics.model_fields[name].default is not None and not metric.optional_zero
}
OPTIONAL_METRIC_MAPPING = {
    metric.source_name: name
    for name, metric in METRICS.items()
    if IngestionMetrics.model_fields[name].default is None
}
ZERO_DEFAULT_METRIC_MAPPING = {
    metric.source_name: name for name, metric in METRICS.items() if metric.optional_zero
}
REQUIRED_COUNTER_NAMES = tuple(
    name
    for name, metric in METRICS.items()
    if metric.kind == "counter" and IngestionMetrics.model_fields[name].default is not None
)
OPTIONAL_COUNTER_NAMES = tuple(
    name
    for name, metric in METRICS.items()
    if metric.kind == "counter" and IngestionMetrics.model_fields[name].default is None
)


class DeviceObservation(StrictModel):
    device_id: str = Field(min_length=1)
    heartbeat_missing: bool
    heartbeat_age_seconds: NonNegativeFloat | None = None


class ObservationV1(StrictModel):
    schema_version: Literal["1.0"]
    metadata: ObservationMetadata
    metrics: IngestionMetrics
    devices: list[DeviceObservation] = Field(default_factory=list)
    # EMQX evidence: True = running and healthy, False = confirmed down, None = unknown.
    broker_available: bool | None = None


class FeatureVector(StrictModel):
    elapsed_seconds: float = Field(gt=0)

    messages_enqueued_rate: NonNegativeFloat
    messages_processed_rate: NonNegativeFloat
    transform_success_rate: NonNegativeFloat

    queue_growth_rate: float
    processing_ratio: NonNegativeFloat
    queue_full_rate: NonNegativeFloat
    queue_depth: NonNegativeFloat | None = None
    queue_capacity: NonNegativeFloat | None = None
    queue_utilization_ratio: NonNegativeFloat | None = None
    queue_metrics_available: bool = Field(default=False)

    transform_failure_ratio: NonNegativeFloat
    transform_deserialize_failure_ratio: NonNegativeFloat | None = None
    transform_ignored_ratio: NonNegativeFloat | None = None
    transform_metrics_available: bool = Field(default=False)

    decode_failure_ratio: NonNegativeFloat | None = None
    decode_metrics_available: bool = Field(default=False)
    raw_validation_failure_ratio: NonNegativeFloat | None = None
    raw_validation_ignored_ratio: NonNegativeFloat | None = None
    raw_validation_metrics_available: bool = Field(default=False)
    business_validation_failure_ratio: NonNegativeFloat | None = None
    business_validation_metrics_available: bool = Field(default=False)

    event_queue_drop_ratio: NonNegativeFloat | None = None
    input_queue_metrics_available: bool = Field(default=False)
    wal_queue_full_ratio: NonNegativeFloat | None = None
    wal_queue_closed_rate: NonNegativeFloat | None = None
    wal_queue_metrics_available: bool = Field(default=False)
    durability_ack_failure_rate: NonNegativeFloat | None = None
    durability_ack_metrics_available: bool = Field(default=False)
    dlq_publish_failure_ratio: NonNegativeFloat | None = None
    dlq_metrics_available: bool = Field(default=False)

    wal_retry_rate: NonNegativeFloat | None = None
    wal_commit_retry_rate: NonNegativeFloat | None = None
    wal_drop_ratio: NonNegativeFloat | None = None
    wal_retry_outage_active: float | None = Field(default=None, ge=0, le=1)
    wal_metrics_available: bool = Field(default=False)

    influx_write_failure_ratio: NonNegativeFloat | None = None
    influx_write_metrics_available: bool = Field(default=False)
    influxdb_healthy: NonNegativeFloat
    persistence_ratio: NonNegativeFloat

    pipeline_duration_average_seconds: NonNegativeFloat
    decode_duration_average_seconds: NonNegativeFloat | None = None
    decode_duration_metrics_available: bool = Field(default=False)
    transform_duration_average_seconds: NonNegativeFloat | None = None
    transform_duration_metrics_available: bool = Field(default=False)
    persist_duration_average_seconds: NonNegativeFloat | None = None
    persist_duration_metrics_available: bool = Field(default=False)
    influx_write_duration_average_seconds: NonNegativeFloat | None = None
    influx_write_duration_metrics_available: bool = Field(default=False)

    heartbeat_missing: bool
    heartbeat_age_max_seconds: NonNegativeFloat | None = None
    heartbeat_age_mean_seconds: NonNegativeFloat | None = None
    heartbeat_age_available: bool = Field(default=False)

    missing_device_ids: list[str] = Field(default_factory=list)
    observed_device_count: NonNegativeInt
    missing_device_count: NonNegativeInt
    missing_device_ratio: float = Field(default=0.0, ge=0, le=1)
    device_metrics_available: bool = Field(default=False)

    broker_available: bool | None = None


PredictionLabel = Literal[
    "normal",
    "influxdb-write-failure",
    "ingestion-backpressure",
    "malformed-sensor-payload",
    "missing-device-heartbeat",
    "mqtt-broker-unavailable",
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


class LatestDetection(AnalysisResult):
    observed_at: AwareDatetime
    age_seconds: NonNegativeFloat
    stale: bool
