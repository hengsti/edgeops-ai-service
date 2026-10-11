import hashlib
from math import isfinite, log1p
from pathlib import Path
from typing import Any

import joblib
import pytest

from edgeops_ai.ml import (
    ARTIFACT_SCHEMA_VERSION,
    ML_MODEL_VERSION,
    MODEL_FEATURE_COLUMNS,
    MlPredictor,
    ModelArtifactError,
    build_model_feature_row,
    load_ml_predictor,
    vectorize_model_features,
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
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": ML_MODEL_VERSION,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "labels": list(FakeModel.classes_),
        "model": FakeModel(),
    }


def save_artifact(path: Path, value: dict[str, Any] | None = None) -> Path:
    joblib.dump(value if value is not None else artifact(), path)
    return path


def test_model_feature_columns_match_release_a_contract() -> None:
    assert MODEL_FEATURE_COLUMNS == (
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
    )


def test_build_model_feature_row_normalizes_and_bounds() -> None:
    row = build_model_feature_row(
        messages_enqueued_rate=10.0,
        queue_growth_rate=5.0,
        processing_ratio=1.7,
        queue_full_rate=2.0,
        transform_failure_ratio=0.25,
        persistence_ratio=0.5,
        heartbeat_missing=True,
        queue_utilization_ratio=1.2,
        influx_write_failure_ratio=0.25,
        influxdb_healthy=1.5,
        pipeline_duration_average_seconds=3.0,
        heartbeat_age_max_seconds=3.0,
        missing_device_ratio=1.2,
        device_metrics_available=True,
        decode_failure_ratio=0.1,
        raw_validation_failure_ratio=0.2,
        business_validation_failure_ratio=0.3,
        transform_deserialize_failure_ratio=0.4,
        event_queue_drop_ratio=0.5,
        wal_queue_full_ratio=0.6,
        durability_ack_failure_rate=3.0,
        dlq_publish_failure_ratio=0.7,
        wal_retry_rate=3.0,
        wal_drop_ratio=0.8,
        wal_retry_outage_active=1.0,
        decode_duration_average_seconds=3.0,
        transform_duration_average_seconds=3.0,
        persist_duration_average_seconds=3.0,
        influx_write_duration_average_seconds=3.0,
    )

    log_four = log1p(3.0)
    assert row == pytest.approx(
        [
            0.5,
            1.0,
            0.2,
            0.25,
            0.5,
            1.0,
            1.0,
            1.0,
            0.25,
            1.0,
            1.0,
            log_four,
            log_four,
            1.0,
            1.0,
            1.0,
            0.1,
            1.0,
            0.2,
            1.0,
            0.3,
            1.0,
            0.4,
            1.0,
            0.5,
            1.0,
            0.6,
            1.0,
            log_four,
            1.0,
            0.7,
            1.0,
            log_four,
            0.8,
            1.0,
            1.0,
            log_four,
            1.0,
            log_four,
            1.0,
            log_four,
            1.0,
            log_four,
            1.0,
        ]
    )
    assert len(row) == len(MODEL_FEATURE_COLUMNS)
    assert all(isfinite(value) for value in row)


def base_model_row(**overrides: Any) -> list[float]:
    values: dict[str, Any] = {
        "messages_enqueued_rate": 10.0,
        "queue_growth_rate": 0.0,
        "processing_ratio": 1.0,
        "queue_full_rate": 0.0,
        "transform_failure_ratio": 0.0,
        "persistence_ratio": 1.0,
        "heartbeat_missing": False,
    }
    values.update(overrides)
    return build_model_feature_row(**values)


def named_row(row: list[float]) -> dict[str, float]:
    return dict(zip(MODEL_FEATURE_COLUMNS, row, strict=True))


def test_optional_a1_values_distinguish_missing_from_real_zero() -> None:
    missing = named_row(base_model_row())
    present = named_row(
        base_model_row(
            queue_utilization_ratio=0.0,
            influx_write_failure_ratio=0.0,
            heartbeat_age_max_seconds=0.0,
        )
    )

    for value_column, availability_column in (
        ("queue_utilization_ratio", "queue_metrics_available"),
        ("influx_write_failure_ratio", "influx_write_metrics_available"),
        ("heartbeat_age_max_log_seconds", "heartbeat_age_available"),
    ):
        assert missing[value_column] == present[value_column] == 0.0
        assert missing[availability_column] == 0.0
        assert present[availability_column] == 1.0


@pytest.mark.parametrize(
    ("source_name", "model_name", "availability_name"),
    [
        ("decode_failure_ratio", "decode_failure_ratio", "decode_metrics_available"),
        (
            "raw_validation_failure_ratio",
            "raw_validation_failure_ratio",
            "raw_validation_metrics_available",
        ),
        (
            "business_validation_failure_ratio",
            "business_validation_failure_ratio",
            "business_validation_metrics_available",
        ),
        (
            "transform_deserialize_failure_ratio",
            "transform_deserialize_failure_ratio",
            "transform_metrics_available",
        ),
        ("event_queue_drop_ratio", "event_queue_drop_ratio", "input_queue_metrics_available"),
        ("wal_queue_full_ratio", "wal_queue_full_ratio", "wal_queue_metrics_available"),
        (
            "durability_ack_failure_rate",
            "durability_ack_failure_log_rate",
            "durability_ack_metrics_available",
        ),
        ("dlq_publish_failure_ratio", "dlq_publish_failure_ratio", "dlq_metrics_available"),
        (
            "decode_duration_average_seconds",
            "decode_duration_log_seconds",
            "decode_duration_metrics_available",
        ),
        (
            "transform_duration_average_seconds",
            "transform_duration_log_seconds",
            "transform_duration_metrics_available",
        ),
        (
            "persist_duration_average_seconds",
            "persist_duration_log_seconds",
            "persist_duration_metrics_available",
        ),
        (
            "influx_write_duration_average_seconds",
            "influx_write_duration_log_seconds",
            "influx_write_duration_metrics_available",
        ),
    ],
)
def test_a2_candidate_values_distinguish_missing_from_real_zero(
    source_name: str,
    model_name: str,
    availability_name: str,
) -> None:
    missing = named_row(base_model_row())
    present = named_row(base_model_row(**{source_name: 0.0}))

    assert missing[model_name] == present[model_name] == 0.0
    assert missing[availability_name] == 0.0
    assert present[availability_name] == 1.0


def test_wal_candidate_group_requires_every_source_value() -> None:
    missing = named_row(base_model_row())
    partial = named_row(base_model_row(wal_retry_rate=2.0))
    present = named_row(
        base_model_row(
            wal_retry_rate=0.0,
            wal_drop_ratio=0.0,
            wal_retry_outage_active=0.0,
        )
    )

    for row in (missing, partial):
        assert row["wal_retry_log_rate"] == 0.0
        assert row["wal_drop_ratio"] == 0.0
        assert row["wal_retry_outage_active"] == 0.0
        assert row["wal_metrics_available"] == 0.0
    assert present["wal_metrics_available"] == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"pipeline_duration_average_seconds": float("inf")},
        {"heartbeat_age_max_seconds": float("nan")},
        {"durability_ack_failure_rate": float("inf")},
        {"decode_duration_average_seconds": float("nan")},
    ],
)
def test_non_finite_latency_is_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        base_model_row(**overrides)


def test_vectorize_model_features_passes_every_source(make_features) -> None:
    features = make_features(
        queue_utilization_ratio=0.1,
        influx_write_failure_ratio=0.2,
        heartbeat_age_max_seconds=2.0,
        missing_device_ratio=0.3,
        device_metrics_available=True,
        decode_failure_ratio=0.4,
        raw_validation_failure_ratio=0.5,
        business_validation_failure_ratio=0.6,
        transform_deserialize_failure_ratio=0.7,
        event_queue_drop_ratio=0.8,
        wal_queue_full_ratio=0.9,
        durability_ack_failure_rate=1.0,
        dlq_publish_failure_ratio=0.1,
        wal_retry_rate=2.0,
        wal_drop_ratio=0.2,
        wal_retry_outage_active=1.0,
        decode_duration_average_seconds=0.1,
        transform_duration_average_seconds=0.2,
        persist_duration_average_seconds=0.3,
        influx_write_duration_average_seconds=0.4,
    )

    row = named_row(vectorize_model_features(features))

    assert row["queue_utilization_ratio"] == 0.1
    assert row["influx_write_failure_ratio"] == 0.2
    assert row["heartbeat_age_max_log_seconds"] == pytest.approx(log1p(2.0))
    assert row["decode_failure_ratio"] == 0.4
    assert row["wal_retry_log_rate"] == pytest.approx(log1p(2.0))
    assert row["influx_write_duration_log_seconds"] == pytest.approx(log1p(0.4))
    assert len(row) == len(MODEL_FEATURE_COLUMNS)
    assert all(isfinite(value) for value in row.values())


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


def test_loads_artifact_with_matching_sha256(tmp_path: Path) -> None:
    path = save_artifact(tmp_path / "model.joblib")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    assert load_ml_predictor(path, sha256.upper()).model_version == ML_MODEL_VERSION


def test_rejects_artifact_with_wrong_sha256(tmp_path: Path, monkeypatch) -> None:
    path = save_artifact(tmp_path / "model.joblib")
    monkeypatch.setattr(joblib, "load", lambda _: pytest.fail("joblib.load must not run"))

    with pytest.raises(ModelArtifactError, match="SHA-256 mismatch"):
        load_ml_predictor(path, "0" * 64)


def test_missing_artifact_fails(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="does not exist"):
        load_ml_predictor(tmp_path / "missing.joblib")


def test_rejects_missing_artifact_fields(tmp_path: Path) -> None:
    value = artifact()
    del value["labels"]
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="missing required fields"):
        load_ml_predictor(path)


def test_rejects_old_artifact_schema(tmp_path: Path) -> None:
    value = artifact()
    value["schema_version"] = "1.0"
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="Unsupported artifact schema version"):
        load_ml_predictor(path)


@pytest.mark.parametrize("version", ["lightweight-ml-0.2.0", "lightweight-ml-0.3.0"])
def test_rejects_old_model_version(tmp_path: Path, version: str) -> None:
    value = artifact()
    value["model_version"] = version
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="Unsupported ML model version"):
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


def test_rejects_previous_six_column_feature_order(tmp_path: Path) -> None:
    value = artifact()
    value["feature_columns"] = [
        "queue_growth_ratio",
        "processing_ratio",
        "queue_full_ratio",
        "transform_failure_ratio",
        "persistence_ratio",
        "heartbeat_missing",
    ]
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="expected training order"):
        load_ml_predictor(path)


def test_rejects_model_without_probabilities(tmp_path: Path) -> None:
    value = artifact()
    value["model"] = object()
    path = save_artifact(tmp_path / "model.joblib", value)

    with pytest.raises(ModelArtifactError, match="predict_proba"):
        load_ml_predictor(path)
