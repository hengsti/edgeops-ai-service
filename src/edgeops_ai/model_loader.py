from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import joblib

from edgeops_ai.prediction_backends import (
    ML_MODEL_VERSION,
    MlPredictionBackend,
    ProbabilityClassifier,
)
from edgeops_ai.schemas import FeatureVector

ARTIFACT_SCHEMA_VERSION = "1.0"

SUPPORTED_LABELS = frozenset(
    {
        "normal",
        "influxdb-write-failure",
        "ingestion-backpressure",
        "malformed-sensor-payload",
        "missing-device-heartbeat",
    }
)

EXPECTED_FEATURE_COLUMNS = (
    "messages_enqueued_rate",
    "messages_processed_rate",
    "transform_success_rate",
    "queue_growth_rate",
    "processing_ratio",
    "queue_full_rate",
    "transform_failure_ratio",
    "persistence_ratio",
    "pipeline_duration_average_seconds",
    "heartbeat_missing",
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


def recommendation_for_prediction(label: str, features: FeatureVector) -> str:
    if label == "normal":
        return "No issues detected."

    if label == "influxdb-write-failure":
        return "Check InfluxDB health, connectivity and ingestion write logs."

    if label == "ingestion-backpressure":
        return "Inspect ingestion queue pressure and processing throughput."

    if label == "malformed-sensor-payload":
        return "Inspect DLQ messages and validate sensor payload schemas."

    if label == "missing-device-heartbeat":
        devices = ", ".join(features.missing_device_ids)

        if devices:
            return f"Check connectivity and power state of: {devices}."

        return "Check device connectivity and power state."

    raise RuntimeError(f"ML model returned an unsupported label: {label!r}.")


def load_ml_backend(artifact_path: Path) -> MlPredictionBackend:
    artifact = _load_artifact(artifact_path)

    model = _validate_model(artifact["model"])
    feature_columns = _validate_feature_columns(artifact["feature_columns"])
    labels = _validate_labels(artifact["labels"])

    _validate_model_classes(model, labels)
    _validate_model_feature_count(model, feature_columns)

    def vectorize(features: FeatureVector) -> list[float]:
        return [getattr(features, col) for col in feature_columns]

    return MlPredictionBackend(
        model=model,
        vectorizer=vectorize,
        recommendation_resolver=recommendation_for_prediction,
        model_version=cast(str, artifact["model_version"]),
    )


def _load_artifact(artifact_path: Path) -> Mapping[str, Any]:
    if not artifact_path.is_file():
        raise ModelArtifactError(f"Artifact file does not exist: {artifact_path!r}")

    try:
        loaded = joblib.load(artifact_path)
    except Exception as exc:
        raise ModelArtifactError(
            f"Failed to load ML model artifact from {artifact_path!r}: {exc}"
        ) from exc

    if not isinstance(loaded, Mapping):
        raise ModelArtifactError("ML model artifact must contain a dictionary-like mapping.")

    missing_fields = REQUIRED_ARTIFACT_FIELDS.difference(loaded)

    if missing_fields:
        raise ModelArtifactError(
            "ML model artifact is missing required fields: "
            + ", ".join(sorted(missing_fields))
            + "."
        )

    if loaded["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError(
            "Unsupported artifact schema version: "
            f"{loaded['schema_version']!r}. "
            f"Expected {ARTIFACT_SCHEMA_VERSION!r}."
        )

    if loaded["model_version"] != ML_MODEL_VERSION:
        raise ModelArtifactError(
            "Unsupported ML model version: "
            f"{loaded['model_version']!r}. "
            f"Expected {ML_MODEL_VERSION!r}."
        )

    return loaded


def _validate_feature_columns(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(column, str) for column in value)
    ):
        raise ModelArtifactError("Artifact feature_columns must be a sequence of strings.")

    feature_columns = tuple(value)

    if len(feature_columns) != len(set(feature_columns)):
        raise ModelArtifactError("Artifact feature_columns contains duplicate columns.")

    if feature_columns != EXPECTED_FEATURE_COLUMNS:
        raise ModelArtifactError(
            "Artifact feature_columns does not match the expected training "
            f"order. Expected {EXPECTED_FEATURE_COLUMNS!r}, "
            f"received {feature_columns!r}."
        )

    return feature_columns


def _validate_labels(value: Any) -> frozenset[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(label, str) for label in value)
    ):
        raise ModelArtifactError("Artifact labels must be a sequence of strings.")

    labels = frozenset(value)

    if labels != SUPPORTED_LABELS:
        raise ModelArtifactError(
            "Artifact labels do not match the supported prediction labels. "
            f"Expected {sorted(SUPPORTED_LABELS)!r}, "
            f"received {sorted(labels)!r}."
        )

    return labels


def _validate_model(value: Any) -> ProbabilityClassifier:
    if not callable(getattr(value, "predict", None)):
        raise ModelArtifactError("ML model does not provide predict().")

    if not callable(getattr(value, "predict_proba", None)):
        raise ModelArtifactError("ML model does not provide predict_proba().")

    if not hasattr(value, "classes_"):
        raise ModelArtifactError("ML model does not provide classes_.")

    if not hasattr(value, "n_features_in_"):
        raise ModelArtifactError("ML model does not provide n_features_in_.")

    return cast(ProbabilityClassifier, value)


def _validate_model_classes(model: ProbabilityClassifier, artifact_labels: frozenset[str]) -> None:
    try:
        model_labels = frozenset(str(label) for label in model.classes_)
    except TypeError as exc:
        raise ModelArtifactError("ML model classes_ is not iterable.") from exc

    if model_labels != artifact_labels:
        raise ModelArtifactError(
            "ML model classes_ does not match artifact labels. "
            f"Artifact labels: {sorted(artifact_labels)!r}; "
            f"model classes: {sorted(model_labels)!r}."
        )


def _validate_model_feature_count(
    model: ProbabilityClassifier, feature_columns: tuple[str, ...]
) -> None:
    try:
        model_feature_count = int(model.n_features_in_)
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("ML model n_features_in_ is invalid.") from exc

    if model_feature_count != len(feature_columns):
        raise ModelArtifactError(
            "ML model feature count does not match feature_columns: "
            f"model expects {model_feature_count}, "
            f"artifact defines {len(feature_columns)}."
        )
