import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from scripts.export_onnx import export_onnx

from edgeops_ai.ml import (
    MODEL_FEATURE_COLUMNS,
    ModelArtifactError,
    load_ml_artifact,
    load_ml_predictor,
)


@pytest.fixture
def onnx_artifact(tmp_path: Path) -> Path:
    source = Path("artifacts/model.joblib")
    path = tmp_path / "model.onnx"
    export_onnx(source, path, hashlib.sha256(source.read_bytes()).hexdigest())
    return path


def test_onnx_matches_existing_model_without_training(onnx_artifact: Path, make_features) -> None:
    original = load_ml_predictor(Path("artifacts/model.joblib"))
    converted = load_ml_predictor(onnx_artifact)
    for overrides in (
        {},
        {"broker_available": False, "heartbeat_missing": True},
        {"influxdb_healthy": 0, "influx_write_failure_ratio": 0.9},
        {"processing_ratio": 0.1, "queue_growth_rate": 9, "queue_utilization_ratio": 0.9},
        {"decode_failure_ratio": 0.5, "transform_failure_ratio": 0.5},
        {"heartbeat_missing": True, "heartbeat_age_max_seconds": 30},
    ):
        features = make_features(**overrides)
        expected, actual = original.predict(features), converted.predict(features)
        assert actual.label == expected.label
        assert actual.confidence == pytest.approx(expected.confidence, abs=1e-6)
        assert actual.model_version == expected.model_version
        assert actual.recommendation == expected.recommendation


def test_onnx_probabilities_match_for_varied_vectors(onnx_artifact: Path) -> None:
    original = load_ml_artifact(Path("artifacts/model.joblib"))["model"]
    converted = load_ml_artifact(onnx_artifact)["model"]
    matrix = np.random.default_rng(14).uniform(0, 5, size=(1000, len(MODEL_FEATURE_COLUMNS)))
    expected = original.predict_proba(matrix)
    actual = converted.predict_proba(matrix)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_contract_version", "incompatible", "feature contract"),
        ("labels", ["unknown-anomaly"] * 6, "classes"),
        ("feature_columns", ["wrong"] * 46, "expected training order"),
    ],
)
def test_onnx_enforces_existing_manifest_checks(
    onnx_artifact: Path, field: str, value: object, message: str
) -> None:
    graph = onnx.load(onnx_artifact)
    manifest = json.loads(graph.metadata_props[0].value)
    manifest[field] = value
    onnx.helper.set_model_props(graph, {"edgeops_ai_manifest": json.dumps(manifest)})
    onnx.save(graph, onnx_artifact)
    with pytest.raises(ModelArtifactError, match=message):
        load_ml_predictor(onnx_artifact)


def test_onnx_rejects_wrong_input_feature_count(onnx_artifact: Path) -> None:
    graph = onnx.load(onnx_artifact)
    graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 47
    onnx.save(graph, onnx_artifact)
    with pytest.raises(ModelArtifactError, match="feature count"):
        load_ml_predictor(onnx_artifact)


def test_onnx_checks_hash_before_opening_session(onnx_artifact: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: pytest.fail("Unverified ONNX loaded")
    )
    with pytest.raises(ModelArtifactError, match="SHA-256 mismatch"):
        load_ml_predictor(onnx_artifact, "0" * 64)


def test_onnx_serving_does_not_import_pickle_stack(onnx_artifact: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from edgeops_ai.ml import load_ml_predictor; "
            "load_ml_predictor(Path(sys.argv[1])); "
            "assert 'joblib' not in sys.modules; assert 'sklearn' not in sys.modules",
            str(onnx_artifact),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_export_does_not_fit_or_change_source(tmp_path: Path, monkeypatch) -> None:
    from sklearn.tree import DecisionTreeClassifier

    source = Path("artifacts/model.joblib")
    before = source.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    monkeypatch.setattr(
        DecisionTreeClassifier, "fit", lambda *a, **kw: pytest.fail("Model retrained")
    )
    output = tmp_path / "exported.onnx"
    export_onnx(source, output, digest)
    assert source.read_bytes() == before
    assert load_ml_predictor(output).model_version == load_ml_predictor(source).model_version
