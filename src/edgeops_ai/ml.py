"""ML prediction backend: feature vectorization, artifact loading, and inference."""

import hashlib
from collections.abc import Mapping
from math import isfinite, log1p
from pathlib import Path
from typing import Any, cast

import joblib

from edgeops_ai.prediction import Prediction, PredictionBackendName, recommendation_for
from edgeops_ai.schemas import FeatureVector, PredictionLabel

ML_MODEL_VERSION = "lightweight-ml-0.5.0"
ARTIFACT_SCHEMA_VERSION = "2.0"

MODEL_FEATURE_COLUMNS = (
    "queue_growth_ratio",
    "processing_ratio",
    "queue_full_ratio",
    "transform_failure_ratio",
    "persistence_ratio",
    "heartbeat_missing",
    "queue_utilization_ratio",
    "queue_metrics_available",
    "influx_write_failure_ratio",
    "influx_write_metrics_available",
    "influxdb_healthy",
    "pipeline_duration_log_seconds",
    "heartbeat_age_max_log_seconds",
    "heartbeat_age_available",
    "missing_device_ratio",
    "device_metrics_available",
    "decode_failure_ratio",
    "decode_metrics_available",
    "raw_validation_failure_ratio",
    "raw_validation_metrics_available",
    "business_validation_failure_ratio",
    "business_validation_metrics_available",
    "transform_deserialize_failure_ratio",
    "transform_metrics_available",
    "event_queue_drop_ratio",
    "input_queue_metrics_available",
    "wal_queue_full_ratio",
    "wal_queue_metrics_available",
    "durability_ack_failure_log_rate",
    "durability_ack_metrics_available",
    "dlq_publish_failure_ratio",
    "dlq_metrics_available",
    "wal_retry_log_rate",
    "wal_drop_ratio",
    "wal_retry_outage_active",
    "wal_metrics_available",
    "decode_duration_log_seconds",
    "decode_duration_metrics_available",
    "transform_duration_log_seconds",
    "transform_duration_metrics_available",
    "persist_duration_log_seconds",
    "persist_duration_metrics_available",
    "influx_write_duration_log_seconds",
    "influx_write_duration_metrics_available",
    "broker_unavailable",
    "broker_metrics_available",
)

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "model_version",
        "feature_columns",
        "labels",
        "model",
    }
)


class ModelArtifactError(RuntimeError):
    """Raised when a model artifact is missing or incompatible."""


def _bounded(value: float) -> float:
    if not isfinite(value):
        raise ValueError("ML feature values must be finite.")

    return min(max(value, 0.0), 1.0)


def _non_negative(value: float) -> float:
    if not isfinite(value):
        raise ValueError("ML feature values must be finite.")

    return max(value, 0.0)


def _optional_bounded(value: float | None) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0

    return _bounded(value), 1.0


def _optional_log(value: float | None) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0

    return log1p(_non_negative(value)), 1.0


def _bounded_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0

    return _bounded(numerator / denominator)


def build_model_feature_row(
    *,
    messages_enqueued_rate: float,
    queue_growth_rate: float,
    processing_ratio: float,
    queue_full_rate: float,
    transform_failure_ratio: float,
    persistence_ratio: float,
    heartbeat_missing: bool | float,
    queue_utilization_ratio: float | None = None,
    influx_write_failure_ratio: float | None = None,
    influxdb_healthy: float = 1.0,
    pipeline_duration_average_seconds: float = 0.0,
    heartbeat_age_max_seconds: float | None = None,
    missing_device_ratio: float = 0.0,
    device_metrics_available: bool | float = False,
    decode_failure_ratio: float | None = None,
    raw_validation_failure_ratio: float | None = None,
    business_validation_failure_ratio: float | None = None,
    transform_deserialize_failure_ratio: float | None = None,
    event_queue_drop_ratio: float | None = None,
    wal_queue_full_ratio: float | None = None,
    durability_ack_failure_rate: float | None = None,
    dlq_publish_failure_ratio: float | None = None,
    wal_retry_rate: float | None = None,
    wal_drop_ratio: float | None = None,
    wal_retry_outage_active: float | None = None,
    decode_duration_average_seconds: float | None = None,
    transform_duration_average_seconds: float | None = None,
    persist_duration_average_seconds: float | None = None,
    influx_write_duration_average_seconds: float | None = None,
    broker_available: bool | float | None = None,
) -> list[float]:
    """
    Build the workload-independent feature vector used for training
    and deployed inference.

    The order must exactly match MODEL_FEATURE_COLUMNS.
    """

    queue_utilization, queue_available = _optional_bounded(queue_utilization_ratio)
    write_failure, write_available = _optional_bounded(influx_write_failure_ratio)
    heartbeat_age, heartbeat_age_available = _optional_log(heartbeat_age_max_seconds)

    decode_failure, decode_available = _optional_bounded(decode_failure_ratio)
    raw_validation_failure, raw_validation_available = _optional_bounded(
        raw_validation_failure_ratio
    )
    business_validation_failure, business_validation_available = _optional_bounded(
        business_validation_failure_ratio
    )
    transform_deserialize_failure, transform_available = _optional_bounded(
        transform_deserialize_failure_ratio
    )
    event_queue_drop, input_queue_available = _optional_bounded(event_queue_drop_ratio)
    wal_queue_full, wal_queue_available = _optional_bounded(wal_queue_full_ratio)
    durability_ack_failure, durability_ack_available = _optional_log(durability_ack_failure_rate)
    dlq_publish_failure, dlq_available = _optional_bounded(dlq_publish_failure_ratio)

    if (
        wal_retry_rate is not None
        and wal_drop_ratio is not None
        and wal_retry_outage_active is not None
    ):
        wal_available = True
        wal_retry = log1p(_non_negative(wal_retry_rate))
        bounded_wal_drop = _bounded(wal_drop_ratio)
        bounded_wal_outage = _bounded(wal_retry_outage_active)
    else:
        wal_available = False
        wal_retry = 0.0
        bounded_wal_drop = 0.0
        bounded_wal_outage = 0.0

    decode_duration, decode_duration_available = _optional_log(decode_duration_average_seconds)
    transform_duration, transform_duration_available = _optional_log(
        transform_duration_average_seconds
    )
    persist_duration, persist_duration_available = _optional_log(persist_duration_average_seconds)
    influx_write_duration, influx_write_duration_available = _optional_log(
        influx_write_duration_average_seconds
    )

    row = [
        _bounded_ratio(max(queue_growth_rate, 0.0), messages_enqueued_rate),
        _bounded(processing_ratio),
        _bounded_ratio(queue_full_rate, messages_enqueued_rate),
        _bounded(transform_failure_ratio),
        _bounded(persistence_ratio),
        float(bool(heartbeat_missing)),
        queue_utilization,
        queue_available,
        write_failure,
        write_available,
        _bounded(influxdb_healthy),
        log1p(_non_negative(pipeline_duration_average_seconds)),
        heartbeat_age,
        heartbeat_age_available,
        _bounded(missing_device_ratio),
        float(bool(device_metrics_available)),
        decode_failure,
        decode_available,
        raw_validation_failure,
        raw_validation_available,
        business_validation_failure,
        business_validation_available,
        transform_deserialize_failure,
        transform_available,
        event_queue_drop,
        input_queue_available,
        wal_queue_full,
        wal_queue_available,
        durability_ack_failure,
        durability_ack_available,
        dlq_publish_failure,
        dlq_available,
        wal_retry,
        bounded_wal_drop,
        bounded_wal_outage,
        float(wal_available),
        decode_duration,
        decode_duration_available,
        transform_duration,
        transform_duration_available,
        persist_duration,
        persist_duration_available,
        influx_write_duration,
        influx_write_duration_available,
        float(broker_available is not None and not broker_available),
        float(broker_available is not None),
    ]

    if len(row) != len(MODEL_FEATURE_COLUMNS):
        raise RuntimeError("ML feature row does not match MODEL_FEATURE_COLUMNS.")

    return row


def vectorize_model_features(features: FeatureVector) -> list[float]:
    return build_model_feature_row(
        messages_enqueued_rate=float(features.messages_enqueued_rate),
        queue_growth_rate=float(features.queue_growth_rate),
        processing_ratio=float(features.processing_ratio),
        queue_full_rate=float(features.queue_full_rate),
        transform_failure_ratio=float(features.transform_failure_ratio),
        persistence_ratio=float(features.persistence_ratio),
        heartbeat_missing=features.heartbeat_missing,
        queue_utilization_ratio=features.queue_utilization_ratio,
        influx_write_failure_ratio=features.influx_write_failure_ratio,
        influxdb_healthy=float(features.influxdb_healthy),
        pipeline_duration_average_seconds=float(features.pipeline_duration_average_seconds),
        heartbeat_age_max_seconds=features.heartbeat_age_max_seconds,
        missing_device_ratio=float(features.missing_device_ratio),
        device_metrics_available=features.device_metrics_available,
        decode_failure_ratio=features.decode_failure_ratio,
        raw_validation_failure_ratio=features.raw_validation_failure_ratio,
        business_validation_failure_ratio=features.business_validation_failure_ratio,
        transform_deserialize_failure_ratio=features.transform_deserialize_failure_ratio,
        event_queue_drop_ratio=features.event_queue_drop_ratio,
        wal_queue_full_ratio=features.wal_queue_full_ratio,
        durability_ack_failure_rate=features.durability_ack_failure_rate,
        dlq_publish_failure_ratio=features.dlq_publish_failure_ratio,
        wal_retry_rate=features.wal_retry_rate,
        wal_drop_ratio=features.wal_drop_ratio,
        wal_retry_outage_active=features.wal_retry_outage_active,
        decode_duration_average_seconds=features.decode_duration_average_seconds,
        transform_duration_average_seconds=features.transform_duration_average_seconds,
        persist_duration_average_seconds=features.persist_duration_average_seconds,
        influx_write_duration_average_seconds=features.influx_write_duration_average_seconds,
        broker_available=features.broker_available,
    )


class MlPredictor:
    """
    Serves predictions from a trained probability classifier.

    Works with any estimator exposing predict() and predict_proba(),
    validated once at startup by load_ml_predictor().
    """

    def __init__(self, *, model: Any, model_version: str = ML_MODEL_VERSION) -> None:
        self._model = model
        self._model_version = model_version

    @property
    def backend_name(self) -> PredictionBackendName:
        return PredictionBackendName.ML

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, features: FeatureVector) -> Prediction:
        matrix = [vectorize_model_features(features)]

        predicted_label = str(self._model.predict(matrix)[0])

        classes = [str(value) for value in self._model.classes_]
        probabilities = self._model.predict_proba(matrix)[0]
        confidence = float(probabilities[classes.index(predicted_label)])

        return Prediction(
            label=cast(PredictionLabel, predicted_label),
            confidence=confidence,
            recommendation=recommendation_for(predicted_label, features),
            model_version=self._model_version,
        )


def load_ml_predictor(artifact_path: Path, expected_sha256: str | None = None) -> MlPredictor:
    if not artifact_path.is_file():
        raise ModelArtifactError(f"Artifact file does not exist: {artifact_path}")

    # joblib.load unpickles and can execute code, so verify the file before loading it.
    if expected_sha256 is not None:
        with artifact_path.open("rb") as file:
            actual_sha256 = hashlib.file_digest(file, "sha256").hexdigest()

        if actual_sha256 != expected_sha256.strip().lower():
            raise ModelArtifactError(
                f"Artifact SHA-256 mismatch for {artifact_path}: {actual_sha256}."
            )

    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise ModelArtifactError(
            f"Failed to load ML model artifact from {artifact_path}: {exc}"
        ) from exc

    if not isinstance(artifact, Mapping):
        raise ModelArtifactError("ML model artifact must be a dictionary-like mapping.")

    missing_fields = REQUIRED_ARTIFACT_FIELDS.difference(artifact)

    if missing_fields:
        raise ModelArtifactError(
            "ML model artifact is missing required fields: " + ", ".join(sorted(missing_fields))
        )

    if artifact["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError(
            f"Unsupported artifact schema version: {artifact['schema_version']!r}. "
            f"Expected {ARTIFACT_SCHEMA_VERSION!r}."
        )

    if artifact["model_version"] != ML_MODEL_VERSION:
        raise ModelArtifactError(
            f"Unsupported ML model version: {artifact['model_version']!r}. "
            f"Expected {ML_MODEL_VERSION!r}."
        )

    if tuple(artifact["feature_columns"]) != MODEL_FEATURE_COLUMNS:
        raise ModelArtifactError(
            "Artifact feature_columns does not match the expected training order. "
            f"Expected {MODEL_FEATURE_COLUMNS!r}, received {tuple(artifact['feature_columns'])!r}."
        )

    model = artifact["model"]

    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ModelArtifactError("ML model must provide predict() and predict_proba().")

    return MlPredictor(model=model, model_version=artifact["model_version"])
