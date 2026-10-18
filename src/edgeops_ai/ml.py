"""ML prediction backend: feature vectorization, artifact loading, and inference."""

import hashlib
import json
from collections.abc import Callable, Mapping
from math import isfinite, log1p
from pathlib import Path
from typing import Any, cast, get_args

from edgeops_ai.prediction import Prediction, PredictionBackendName, recommendation_for
from edgeops_ai.schemas import FeatureVector, PredictionLabel

FEATURE_CONTRACT_VERSION = "1.0"
ARTIFACT_SCHEMA_VERSION = "2.0"

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "model_version",
        "feature_contract_version",
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


def _optional_bounded(value: float | None) -> float:
    return 0.0 if value is None else _bounded(value)


def _optional_log(value: float | None) -> float:
    return 0.0 if value is None else log1p(_non_negative(value))


def _bounded_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else _bounded(numerator / denominator)


def _wal_available(features: FeatureVector) -> bool:
    return all(
        value is not None
        for value in (
            features.wal_retry_rate,
            features.wal_drop_ratio,
            features.wal_retry_outage_active,
        )
    )


MODEL_FEATURES: dict[str, Callable[[FeatureVector], float]] = {
    "queue_growth_ratio": lambda f: _bounded_ratio(
        max(f.queue_growth_rate, 0.0), f.messages_enqueued_rate
    ),
    "processing_ratio": lambda f: _bounded(f.processing_ratio),
    "queue_full_ratio": lambda f: _bounded_ratio(f.queue_full_rate, f.messages_enqueued_rate),
    "transform_failure_ratio": lambda f: _bounded(f.transform_failure_ratio),
    "persistence_ratio": lambda f: _bounded(f.persistence_ratio),
    "heartbeat_missing": lambda f: float(bool(f.heartbeat_missing)),
    "queue_utilization_ratio": lambda f: _optional_bounded(f.queue_utilization_ratio),
    "queue_metrics_available": lambda f: float(f.queue_utilization_ratio is not None),
    "influx_write_failure_ratio": lambda f: _optional_bounded(f.influx_write_failure_ratio),
    "influx_write_metrics_available": lambda f: float(f.influx_write_failure_ratio is not None),
    "influxdb_healthy": lambda f: _bounded(f.influxdb_healthy),
    "pipeline_duration_log_seconds": lambda f: log1p(
        _non_negative(f.pipeline_duration_average_seconds)
    ),
    "heartbeat_age_max_log_seconds": lambda f: _optional_log(f.heartbeat_age_max_seconds),
    "heartbeat_age_available": lambda f: float(f.heartbeat_age_max_seconds is not None),
    "missing_device_ratio": lambda f: _bounded(f.missing_device_ratio),
    "device_metrics_available": lambda f: float(bool(f.device_metrics_available)),
    "decode_failure_ratio": lambda f: _optional_bounded(f.decode_failure_ratio),
    "decode_metrics_available": lambda f: float(f.decode_failure_ratio is not None),
    "raw_validation_failure_ratio": lambda f: _optional_bounded(f.raw_validation_failure_ratio),
    "raw_validation_metrics_available": lambda f: float(f.raw_validation_failure_ratio is not None),
    "business_validation_failure_ratio": lambda f: _optional_bounded(
        f.business_validation_failure_ratio
    ),
    "business_validation_metrics_available": lambda f: float(
        f.business_validation_failure_ratio is not None
    ),
    "transform_deserialize_failure_ratio": lambda f: _optional_bounded(
        f.transform_deserialize_failure_ratio
    ),
    "transform_metrics_available": lambda f: float(
        f.transform_deserialize_failure_ratio is not None
    ),
    "event_queue_drop_ratio": lambda f: _optional_bounded(f.event_queue_drop_ratio),
    "input_queue_metrics_available": lambda f: float(f.event_queue_drop_ratio is not None),
    "wal_queue_full_ratio": lambda f: _optional_bounded(f.wal_queue_full_ratio),
    "wal_queue_metrics_available": lambda f: float(f.wal_queue_full_ratio is not None),
    "durability_ack_failure_log_rate": lambda f: _optional_log(f.durability_ack_failure_rate),
    "durability_ack_metrics_available": lambda f: float(f.durability_ack_failure_rate is not None),
    "dlq_publish_failure_ratio": lambda f: _optional_bounded(f.dlq_publish_failure_ratio),
    "dlq_metrics_available": lambda f: float(f.dlq_publish_failure_ratio is not None),
    "wal_retry_log_rate": lambda f: _optional_log(f.wal_retry_rate) if _wal_available(f) else 0.0,
    "wal_drop_ratio": lambda f: _optional_bounded(f.wal_drop_ratio) if _wal_available(f) else 0.0,
    "wal_retry_outage_active": lambda f: (
        _optional_bounded(f.wal_retry_outage_active) if _wal_available(f) else 0.0
    ),
    "wal_metrics_available": lambda f: float(_wal_available(f)),
    "decode_duration_log_seconds": lambda f: _optional_log(f.decode_duration_average_seconds),
    "decode_duration_metrics_available": lambda f: float(
        f.decode_duration_average_seconds is not None
    ),
    "transform_duration_log_seconds": lambda f: _optional_log(f.transform_duration_average_seconds),
    "transform_duration_metrics_available": lambda f: float(
        f.transform_duration_average_seconds is not None
    ),
    "persist_duration_log_seconds": lambda f: _optional_log(f.persist_duration_average_seconds),
    "persist_duration_metrics_available": lambda f: float(
        f.persist_duration_average_seconds is not None
    ),
    "influx_write_duration_log_seconds": lambda f: _optional_log(
        f.influx_write_duration_average_seconds
    ),
    "influx_write_duration_metrics_available": lambda f: float(
        f.influx_write_duration_average_seconds is not None
    ),
    "broker_unavailable": lambda f: float(
        f.broker_available is not None and not f.broker_available
    ),
    "broker_metrics_available": lambda f: float(f.broker_available is not None),
}
MODEL_FEATURE_COLUMNS = tuple(MODEL_FEATURES)


def vectorize_model_features(features: FeatureVector) -> list[float]:
    return [extract(features) for extract in MODEL_FEATURES.values()]


class MlPredictor:
    """
    Serves predictions from a trained probability classifier.

    Works with any classifier exposing predict_proba(), classes_, and n_features_in_,
    validated once at startup by load_ml_predictor().
    """

    def __init__(self, *, model: Any, model_version: str) -> None:
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

        probabilities = self._model.predict_proba(matrix)[0]
        index, confidence = max(enumerate(probabilities), key=lambda item: item[1])
        predicted_label = str(self._model.classes_[index])

        return Prediction(
            label=cast(PredictionLabel, predicted_label),
            confidence=float(confidence),
            recommendation=recommendation_for(predicted_label, features),
            model_version=self._model_version,
        )


class _OnnxClassifier:
    def __init__(self, session: Any, labels: Any) -> None:
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 2:
            raise ModelArtifactError("ONNX input must be a matrix of float32 model features.")
        outputs = {output.name: output for output in session.get_outputs()}
        probabilities = outputs.get("probabilities")
        if (
            probabilities is None
            or probabilities.type != "tensor(float)"
            or len(probabilities.shape) != 2
            or not isinstance(labels, list)
            or probabilities.shape[1] != len(labels)
        ):
            raise ModelArtifactError("ONNX probability output must match the manifest labels.")
        self._session = session
        self._input_name = inputs[0].name
        self.n_features_in_ = inputs[0].shape[1]
        self.classes_ = labels

    def predict_proba(self, matrix: list[list[float]]) -> Any:
        import numpy as np

        return self._session.run(
            ["probabilities"], {self._input_name: np.asarray(matrix, dtype=np.float32)}
        )[0]


def load_ml_artifact(artifact_path: Path, expected_sha256: str | None = None) -> Mapping[str, Any]:
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
        if artifact_path.suffix.lower() == ".onnx":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                artifact_path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
            )
            artifact = json.loads(
                session.get_modelmeta().custom_metadata_map["edgeops_ai_manifest"]
            )
            if not isinstance(artifact, dict):
                raise ModelArtifactError("ONNX manifest must be a dictionary-like mapping.")
            artifact["model"] = _OnnxClassifier(session, artifact.get("labels"))
        else:
            import joblib

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

    if artifact["feature_contract_version"] != FEATURE_CONTRACT_VERSION:
        raise ModelArtifactError(
            f"Unsupported feature contract version: {artifact['feature_contract_version']!r}. "
            f"Expected {FEATURE_CONTRACT_VERSION!r}."
        )

    if not isinstance(artifact["model_version"], str) or not artifact["model_version"].strip():
        raise ModelArtifactError("Artifact model_version must be a non-empty string.")

    columns = artifact["feature_columns"]
    if not isinstance(columns, (list, tuple)) or tuple(columns) != MODEL_FEATURE_COLUMNS:
        raise ModelArtifactError(
            "Artifact feature_columns does not match the expected training order. "
            f"Expected {MODEL_FEATURE_COLUMNS!r}, received {columns!r}."
        )

    model = artifact["model"]

    if not callable(getattr(model, "predict_proba", None)):
        raise ModelArtifactError("ML model must provide predict_proba().")

    try:
        classes = set(model.classes_)
        class_count = len(model.classes_)
    except (AttributeError, TypeError) as error:
        raise ModelArtifactError("ML model must expose a sequence of classes_.") from error
    if not classes or len(classes) != class_count or not classes <= set(get_args(PredictionLabel)):
        raise ModelArtifactError("ML model classes_ must be unique, known PredictionLabels.")

    if getattr(model, "n_features_in_", None) != len(MODEL_FEATURE_COLUMNS):
        raise ModelArtifactError(f"ML model feature count must equal {len(MODEL_FEATURE_COLUMNS)}.")

    try:
        labels = set(artifact["labels"])
    except TypeError as error:
        raise ModelArtifactError("Artifact labels must match the model classes_.") from error
    if labels != classes:
        raise ModelArtifactError("Artifact labels must match the model classes_.")

    return artifact


def load_ml_predictor(artifact_path: Path, expected_sha256: str | None = None) -> MlPredictor:
    artifact = load_ml_artifact(artifact_path, expected_sha256)
    return MlPredictor(model=artifact["model"], model_version=artifact["model_version"])
