"""ML prediction backend: feature vectorization, artifact loading, and inference."""

from collections.abc import Mapping
from math import isfinite, log1p
from pathlib import Path
from typing import Any, cast

import joblib

from edgeops_ai.prediction import Prediction, PredictionBackendName, recommendation_for
from edgeops_ai.schemas import FeatureVector, PredictionLabel

ML_MODEL_VERSION = "lightweight-ml-0.3.0"
ARTIFACT_SCHEMA_VERSION = "2.0"

MODEL_FEATURE_COLUMNS = (
    "queue_growth_ratio",
    "processing_ratio",
    "queue_full_ratio",
    "transform_failure_ratio",
    "persistence_ratio",
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


def _bounded(value: float) -> float:
    if not isfinite(value):
        raise ValueError("ML feature values must be finite.")

    return min(max(value, 0.0), 1.0)


def _bounded_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0

    return _bounded(numerator / denominator)


def pipeline_duration_average_seconds(
    pipeline_duration_delta: float, pipeline_count_delta: float
) -> float:
    if not isfinite(pipeline_duration_delta) or not isfinite(pipeline_count_delta):
        raise ValueError("ML feature values must be finite.")

    return (
        log1p(pipeline_duration_delta / pipeline_count_delta) if pipeline_count_delta > 0 else 0.0
    )


def build_model_feature_row(
    *,
    messages_enqueued_rate: float,
    queue_growth_rate: float,
    processing_ratio: float,
    queue_full_rate: float,
    transform_failure_ratio: float,
    persistence_ratio: float,
    heartbeat_missing: bool | float,
) -> list[float]:
    """
    Build the workload-independent feature vector used for training
    and deployed inference.

    The order must exactly match MODEL_FEATURE_COLUMNS.
    """

    return [
        _bounded_ratio(max(queue_growth_rate, 0.0), messages_enqueued_rate),
        _bounded(processing_ratio),
        _bounded_ratio(queue_full_rate, messages_enqueued_rate),
        _bounded(transform_failure_ratio),
        _bounded(persistence_ratio),
        float(bool(heartbeat_missing)),
    ]


def vectorize_model_features(features: FeatureVector) -> list[float]:
    return build_model_feature_row(
        messages_enqueued_rate=float(features.messages_enqueued_rate),
        queue_growth_rate=float(features.queue_growth_rate),
        processing_ratio=float(features.processing_ratio),
        queue_full_rate=float(features.queue_full_rate),
        transform_failure_ratio=float(features.transform_failure_ratio),
        persistence_ratio=float(features.persistence_ratio),
        heartbeat_missing=features.heartbeat_missing,
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


def load_ml_predictor(artifact_path: Path) -> MlPredictor:
    if not artifact_path.is_file():
        raise ModelArtifactError(f"Artifact file does not exist: {artifact_path}")

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
