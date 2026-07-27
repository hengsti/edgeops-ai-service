from pathlib import Path
from typing import Any

import joblib
import pytest

from edgeops_ai.ml import (
    ML_MODEL_VERSION,
    MODEL_FEATURE_COLUMNS,
    MlPredictor,
    ModelArtifactError,
    build_model_feature_row,
    load_ml_predictor,
)
from edgeops_ai.prediction import PredictionBackendName


class FakeModel:
    classes_ = (
        "influxdb-write-failure",
        "ingestion-backpressure",
        "malformed-sensor-payload",
        "missing-device-heartbeat",
        "normal",
    )

    def predict(self, values: Any) -> list[str]:
        return ["ingestion-backpressure"]

    def predict_proba(self, values: Any) -> list[list[float]]:
        return [[0.02, 0.90, 0.03, 0.02, 0.03]]


def artifact() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "model_version": ML_MODEL_VERSION,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "labels": list(FakeModel.classes_),
        "model": FakeModel(),
    }


def save_artifact(path: Path, value: dict[str, Any] | None = None) -> Path:
    joblib.dump(value if value is not None else artifact(), path)
    return path


def test_build_model_feature_row_normalizes_and_bounds() -> None:
    row = build_model_feature_row(
        messages_enqueued_rate=10.0,
        queue_growth_rate=5.0,
        processing_ratio=1.7,
        queue_full_rate=2.0,
        transform_failure_ratio=0.25,
        persistence_ratio=0.5,
        heartbeat_missing=True,
    )

    assert row == [0.5, 1.0, 0.2, 0.25, 0.5, 1.0]
    assert len(row) == len(MODEL_FEATURE_COLUMNS)


def test_predictor_uses_probability_of_predicted_class(make_features) -> None:
    predictor = MlPredictor(model=FakeModel())

    result = predictor.predict(make_features())

    assert predictor.backend_name is PredictionBackendName.ML
    assert result.label == "ingestion-backpressure"
    assert result.confidence == pytest.approx(0.90)
    assert result.model_version == ML_MODEL_VERSION
    assert "queue pressure" in result.recommendation


def test_loads_valid_ml_artifact(tmp_path: Path) -> None:
    path = save_artifact(tmp_path / "model.joblib")

    predictor = load_ml_predictor(path)

    assert predictor.backend_name is PredictionBackendName.ML
    assert predictor.model_version == ML_MODEL_VERSION


def test_missing_artifact_fails(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="does not exist"):
        load_ml_predictor(tmp_path / "missing.joblib")


def test_rejects_missing_artifact_fields(tmp_path: Path) -> None:
    value = artifact()
    del value["labels"]
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="missing required fields"):
        load_ml_predictor(path)


def test_rejects_wrong_feature_order(tmp_path: Path) -> None:
    value = artifact()
    value["feature_columns"][0], value["feature_columns"][1] = (
        value["feature_columns"][1],
        value["feature_columns"][0],
    )
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="expected training order"):
        load_ml_predictor(path)


def test_rejects_model_without_probabilities(tmp_path: Path) -> None:
    value = artifact()
    value["model"] = object()
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="predict_proba"):
        load_ml_predictor(path)
