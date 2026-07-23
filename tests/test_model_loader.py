from pathlib import Path
from typing import Any

import joblib
import pytest

from edgeops_ai.model_loader import (
    EXPECTED_FEATURE_COLUMNS,
    ModelArtifactError,
    load_ml_backend,
)
from edgeops_ai.prediction_backends import PredictionBackendName


class FakeModel:
    classes_ = (
        "influxdb-write-failure",
        "ingestion-backpressure",
        "malformed-sensor-payload",
        "missing-device-heartbeat",
        "normal",
    )
    n_features_in_ = 10

    def predict(self, values: Any) -> list[str]:
        return ["normal"]

    def predict_proba(self, values: Any) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0, 1.0]]


def artifact() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "model_version": "lightweight-ml-0.1.0",
        "feature_columns": list(EXPECTED_FEATURE_COLUMNS),
        "labels": [
            "normal",
            "influxdb-write-failure",
            "ingestion-backpressure",
            "malformed-sensor-payload",
            "missing-device-heartbeat",
        ],
        "model": FakeModel(),
    }


def save_artifact(
    path: Path,
    value: dict[str, Any] | None = None,
) -> Path:
    joblib.dump(value or artifact(), path)
    return path


def test_loads_valid_ml_artifact(tmp_path: Path) -> None:
    path = save_artifact(tmp_path / "model.joblib")

    backend = load_ml_backend(path)

    assert backend.backend_name is PredictionBackendName.ML
    assert backend.model_version == "lightweight-ml-0.1.0"


def test_missing_artifact_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing.joblib"

    with pytest.raises(
        ModelArtifactError,
        match="does not exist",
    ):
        load_ml_backend(path)


def test_rejects_wrong_feature_order(tmp_path: Path) -> None:
    value = artifact()
    value["feature_columns"][0], value["feature_columns"][1] = (
        value["feature_columns"][1],
        value["feature_columns"][0],
    )
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(
        ModelArtifactError,
        match="expected training order",
    ):
        load_ml_backend(path)


def test_rejects_wrong_labels(tmp_path: Path) -> None:
    value = artifact()
    value["labels"] = ["normal"]
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(
        ModelArtifactError,
        match="supported prediction labels",
    ):
        load_ml_backend(path)


def test_rejects_wrong_model_feature_count(tmp_path: Path) -> None:
    value = artifact()
    value["model"].n_features_in_ = 9
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(
        ModelArtifactError,
        match="feature count",
    ):
        load_ml_backend(path)
