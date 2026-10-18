import csv
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import pytest

from edgeops_ai.ml import (
    MODEL_FEATURE_COLUMNS,
    load_ml_predictor,
    vectorize_model_features,
)
from edgeops_ai.training.common import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    ScriptError,
)
from edgeops_ai.training.dataset import build_dataset_row, write_dataset
from edgeops_ai.training.deployed import (
    DATASET_COLUMNS as DEPLOYED_DATASET_COLUMNS,
)
from edgeops_ai.training.deployed import (
    build_dataset_row as build_deployed_dataset_row,
)
from edgeops_ai.training.deployed import (
    parse_args as parse_holdout_args,
)
from edgeops_ai.training.evaluate import read_dataset
from edgeops_ai.training.manifest import empty_manifest, upsert_manifest_run, write_manifest_atomic
from edgeops_ai.training.pipeline import (
    dataset_is_complete,
)
from edgeops_ai.training.pipeline import (
    parse_args as parse_pipeline_args,
)
from edgeops_ai.training.train import (
    DatasetError,
    load_training_dataset,
    train_model,
)
from edgeops_ai.training.train import (
    parse_args as parse_training_args,
)

TEST_MODEL_VERSION = "test-model-v1"

OPTIONAL_SOURCE_COLUMNS = (
    "queue_utilization_ratio",
    "influx_write_failure_ratio",
    "heartbeat_age_max_seconds",
    "decode_failure_ratio",
    "raw_validation_failure_ratio",
    "business_validation_failure_ratio",
    "transform_deserialize_failure_ratio",
    "event_queue_drop_ratio",
    "wal_queue_full_ratio",
    "durability_ack_failure_rate",
    "dlq_publish_failure_ratio",
    "wal_retry_rate",
    "wal_drop_ratio",
    "wal_retry_outage_active",
    "decode_duration_average_seconds",
    "transform_duration_average_seconds",
    "persist_duration_average_seconds",
    "influx_write_duration_average_seconds",
    "broker_available",
)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def reusable_dataset_rows(
    *,
    scenario: str = "ingestion-backpressure",
    seed: int = 42,
    dataset_schema_version: str = DATASET_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    base = dict.fromkeys(DATASET_COLUMNS, "0")
    base.update(
        {
            "dataset_schema_version": dataset_schema_version,
            "scenario_id": scenario,
            "simulation_seed": str(seed),
        }
    )
    return [
        {
            **base,
            "simulation_phase": "normal",
            "expected_label": "normal",
        },
        {
            **base,
            "simulation_phase": "overloaded",
            "expected_label": scenario,
        },
        {
            **base,
            "simulation_phase": "recovery",
            "expected_label": "normal",
        },
    ]


def training_row(
    *,
    label: str,
    seed: int,
    phase: str,
    run_suffix: str,
) -> dict[str, Any]:
    row = dict.fromkeys(DATASET_COLUMNS, "0")
    scenario = label if label != "normal" else "ingestion-backpressure"
    row.update(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "captured_at": "2026-08-01T08:00:00+00:00",
            "source_host": "training-test",
            "scenario_id": scenario,
            "simulation_run_id": f"run-{seed}-{run_suffix}",
            "simulation_seed": str(seed),
            "simulation_phase": phase,
            "expected_label": label,
            "heartbeat_missing": "0",
            "device_metrics_available": "0",
            "influxdb_healthy": "1",
            "processing_ratio": "1",
            "persistence_ratio": "1",
        }
    )
    return row


def training_rows(seed: int) -> list[dict[str, Any]]:
    labels = (
        "normal",
        "ingestion-backpressure",
        "missing-device-heartbeat",
        "malformed-sensor-payload",
        "influxdb-write-failure",
        "mqtt-broker-unavailable",
    )
    rows = [
        training_row(
            label=label,
            seed=seed,
            phase="normal" if label == "normal" else "overloaded",
            run_suffix=str(index),
        )
        for index, label in enumerate(labels)
    ]
    rows.append(
        training_row(
            label="normal",
            seed=seed,
            phase="recovery",
            run_suffix="recovery",
        )
    )
    return rows


def write_training_matrix(path: Path, *, blank_optional: bool = False) -> None:
    for seed in (42, 43):
        rows = training_rows(seed)
        if blank_optional:
            for row in rows:
                row.update(dict.fromkeys(OPTIONAL_SOURCE_COLUMNS, ""))
        write_csv(
            path / f"matrix-seed-{seed}.csv",
            list(DATASET_COLUMNS),
            rows,
        )


def test_generated_dataset_row_contains_current_schema_version(
    make_features,
) -> None:
    metadata = SimpleNamespace(
        captured_at=datetime(2026, 7, 28, tzinfo=UTC),
        source_host="test-host",
        scenario_id="ingestion-backpressure",
        simulation_run_id="run-42",
        simulation_seed=42,
        simulation_phase="overloaded",
    )
    observation = SimpleNamespace(metadata=metadata)

    row = build_dataset_row(
        observation,
        make_features(),
        "ingestion-backpressure",
        "ingestion-backpressure",
    )

    assert row["dataset_schema_version"] == DATASET_SCHEMA_VERSION
    assert set(row) == set(DATASET_COLUMNS)


def test_generated_recovery_row_preserves_phase_and_provenance(make_features) -> None:
    metadata = SimpleNamespace(
        captured_at=datetime(2026, 8, 1, tzinfo=UTC),
        source_host="test-host",
        scenario_id="ingestion-backpressure",
        simulation_run_id="run-recovery",
        simulation_seed=42,
        simulation_phase="recovery",
    )

    row = build_dataset_row(
        SimpleNamespace(metadata=metadata),
        make_features(),
        "normal",
        "normal",
    )

    assert row["simulation_phase"] == "recovery"
    assert row["simulation_run_id"] == "run-recovery"
    assert row["simulation_seed"] == 42


def test_deployed_holdout_row_contains_version_and_every_feature(make_features) -> None:
    metadata = SimpleNamespace(
        captured_at=datetime(2026, 8, 1, tzinfo=UTC),
        source_host="test-host",
        scenario_id="ingestion-backpressure",
        simulation_run_id="holdout-recovery",
        simulation_seed=62,
        simulation_phase="recovery",
    )
    features = make_features().model_dump(mode="json")

    row = build_deployed_dataset_row(
        SimpleNamespace(metadata=metadata),
        {
            "features": features,
            "prediction": "normal",
            "confidence": 0.9,
            "model_version": TEST_MODEL_VERSION,
        },
        "normal",
    )

    assert set(row) == set(DEPLOYED_DATASET_COLUMNS)
    assert row["dataset_schema_version"] == DATASET_SCHEMA_VERSION
    assert row["simulation_phase"] == "recovery"
    assert all(column in row for column in FEATURE_COLUMNS)


def test_deployed_holdout_rejects_missing_enriched_feature(make_features) -> None:
    metadata = SimpleNamespace(
        captured_at=datetime(2026, 8, 1, tzinfo=UTC),
        source_host="test-host",
        scenario_id="ingestion-backpressure",
        simulation_run_id="holdout-run",
        simulation_seed=62,
        simulation_phase="overloaded",
    )
    features = make_features().model_dump(mode="json")
    missing_column = FEATURE_COLUMNS[-1]
    features.pop(missing_column)

    with pytest.raises(ScriptError, match=missing_column):
        build_deployed_dataset_row(
            SimpleNamespace(metadata=metadata),
            {
                "features": features,
                "prediction": "ingestion-backpressure",
                "confidence": 0.9,
                "model_version": TEST_MODEL_VERSION,
            },
            "ingestion-backpressure",
        )


def test_current_dataset_is_reusable(tmp_path: Path) -> None:
    path = tmp_path / "ingestion-backpressure-seed-42.csv"
    write_csv(path, list(DATASET_COLUMNS), reusable_dataset_rows())
    manifest = upsert_manifest_run(
        empty_manifest(),
        output_dir=tmp_path,
        run_id="0",
        entry={
            "scenario_id": "ingestion-backpressure",
            "seed": 42,
            "variant_id": "mild",
            "speed": 1.0,
            "resolved_effects": {},
            "workload_scale": 1.0,
            "unavailable_metric_groups": [],
            "collector_version": "0.1.0",
            "csv_path": path.name,
        },
    )
    write_manifest_atomic(tmp_path, manifest)

    assert dataset_is_complete(
        path,
        scenario="ingestion-backpressure",
        seed=42,
        variant_id="mild",
        manifest=manifest,
        output_dir=tmp_path,
    )


@pytest.mark.parametrize(
    ("fieldnames", "version"),
    [
        (
            [column for column in DATASET_COLUMNS if column != "dataset_schema_version"],
            DATASET_SCHEMA_VERSION,
        ),
        (list(DATASET_COLUMNS), "1.0"),
        (
            [column for column in DATASET_COLUMNS if column != FEATURE_COLUMNS[0]],
            DATASET_SCHEMA_VERSION,
        ),
        (
            [column for column in DATASET_COLUMNS if column != "model_version"],
            DATASET_SCHEMA_VERSION,
        ),
    ],
)
def test_stale_or_incomplete_dataset_is_not_reusable(
    tmp_path: Path,
    fieldnames: list[str],
    version: str,
) -> None:
    path = tmp_path / "ingestion-backpressure-seed-42.csv"
    write_csv(
        path,
        fieldnames,
        reusable_dataset_rows(dataset_schema_version=version),
    )

    assert not dataset_is_complete(
        path,
        scenario="ingestion-backpressure",
        seed=42,
        variant_id="mild",
        manifest=None,
        output_dir=tmp_path,
    )


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_direct_training_rejects_old_dataset_version(tmp_path: Path, version: str) -> None:
    row = dict.fromkeys(DATASET_COLUMNS, "0")
    row.update(
        {
            "dataset_schema_version": version,
            "simulation_run_id": "run-42",
            "simulation_seed": "42",
            "simulation_phase": "normal",
            "expected_label": "normal",
        }
    )
    write_csv(
        tmp_path / "ingestion-backpressure-seed-42.csv",
        list(DATASET_COLUMNS),
        [row],
    )

    with pytest.raises(DatasetError, match="unsupported dataset_schema_version"):
        load_training_dataset(tmp_path)


def test_dataset_evaluation_rejects_old_dataset_version(tmp_path: Path) -> None:
    rows = reusable_dataset_rows(dataset_schema_version="1.0")
    path = tmp_path / "ingestion-backpressure-seed-42.csv"
    write_csv(path, list(DATASET_COLUMNS), rows)

    with pytest.raises(ValueError, match="unsupported dataset_schema_version"):
        read_dataset(path)


def test_valid_release_a_dataset_loads_without_preprocessing(tmp_path: Path) -> None:
    write_training_matrix(tmp_path)

    dataset = load_training_dataset(tmp_path)

    assert dataset.row_count == 12
    assert dataset.excluded_recovery_rows == 2
    assert dataset.features.shape == (12, len(MODEL_FEATURE_COLUMNS))
    assert set(dataset.labels) == {
        "normal",
        "ingestion-backpressure",
        "missing-device-heartbeat",
        "malformed-sensor-payload",
        "influxdb-write-failure",
        "mqtt-broker-unavailable",
    }
    assert set(dataset.seeds) == {42, 43}


def test_valid_release_a_dataset_trains_candidate_artifact(tmp_path: Path) -> None:
    input_dir = tmp_path / "datasets"
    input_dir.mkdir()
    write_training_matrix(input_dir)
    artifact_path = tmp_path / "model-candidate.joblib"
    report_path = tmp_path / "training-report.json"

    report = train_model(
        input_dir=input_dir,
        artifact_path=artifact_path,
        report_path=report_path,
        min_accuracy=0,
        min_label_recall=0,
    )

    artifact = joblib.load(artifact_path)
    assert report["quality_gate"]["passed"]
    assert report["dataset"]["stable_rows"] == 12
    assert report["dataset"]["excluded_recovery_rows"] == 2
    assert artifact["dataset_schema_version"] == DATASET_SCHEMA_VERSION
    assert tuple(artifact["feature_columns"]) == MODEL_FEATURE_COLUMNS
    assert artifact["training_rows"] == 12
    assert artifact["simulation_seeds"] == [42, 43]
    assert artifact["feature_contract_version"] == "1.0"


@pytest.mark.parametrize("version", ["", "  "])
def test_training_rejects_empty_model_version(tmp_path: Path, version: str) -> None:
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"previous model")
    with pytest.raises(DatasetError, match="model version"):
        train_model(
            input_dir=tmp_path,
            artifact_path=artifact_path,
            report_path=tmp_path / "report.json",
            model_version=version,
        )
    assert artifact_path.read_bytes() == b"previous model"


def test_training_defaults_to_a_generated_loadable_version(tmp_path: Path) -> None:
    write_training_matrix(tmp_path)
    artifact_path = tmp_path / "candidate.joblib"
    train_model(
        input_dir=tmp_path,
        artifact_path=artifact_path,
        report_path=tmp_path / "report.json",
        min_accuracy=0,
        min_label_recall=0,
    )
    assert load_ml_predictor(artifact_path).model_version.startswith("lightweight-ml-")
    assert parse_training_args([]).model_version is None


def test_training_accepts_a_custom_model_version_without_code_changes(tmp_path: Path) -> None:
    write_training_matrix(tmp_path)
    artifact_path = tmp_path / "candidate.joblib"
    report = train_model(
        input_dir=tmp_path,
        artifact_path=artifact_path,
        report_path=tmp_path / "report.json",
        model_version="my-retrained-model-42",
        min_accuracy=0,
        min_label_recall=0,
    )
    assert report["model_version"] == "my-retrained-model-42"
    assert load_ml_predictor(artifact_path).model_version == "my-retrained-model-42"


def test_retrained_model_accepts_healthy_production_heartbeat_age(
    tmp_path: Path, make_features
) -> None:
    faults = {
        "normal": {},
        "missing-device-heartbeat": {"heartbeat_age_max_seconds": 30, "heartbeat_missing": True},
        "ingestion-backpressure": {"processing_ratio": 0.5, "queue_growth_rate": 5},
        "malformed-sensor-payload": {"transform_failure_ratio": 0.5},
        "influxdb-write-failure": {"influx_write_failure_ratio": 0.5, "influxdb_healthy": 0},
        "mqtt-broker-unavailable": {
            "heartbeat_age_max_seconds": 30,
            "heartbeat_missing": True,
            "broker_available": False,
        },
    }
    for seed in (42, 43):
        rows = []
        for label, overrides in faults.items():
            for sample in range(20):
                features = make_features(
                    **{
                        "heartbeat_age_max_seconds": (sample + 0.5) / 10,
                        # Cover healthy and unknown broker telemetry for every other label.
                        "broker_available": True if sample % 2 else None,
                        **overrides,
                    }
                )
                row = training_row(
                    label=label,
                    seed=seed,
                    phase="normal" if label == "normal" else label,
                    run_suffix=label,
                )
                row.update(features.model_dump())
                rows.append(row)
        write_csv(tmp_path / f"matrix-seed-{seed}.csv", list(DATASET_COLUMNS), rows)

    artifact_path = tmp_path / "candidate.joblib"
    report = train_model(
        input_dir=tmp_path,
        artifact_path=artifact_path,
        report_path=tmp_path / "report.json",
    )
    assert report["quality_gate"]["passed"]
    predictor = load_ml_predictor(artifact_path)
    # Production incident report: all four devices present, oldest reading 1.887 seconds old.
    healthy = make_features(
        elapsed_seconds=5.115,
        messages_enqueued_rate=2.541,
        messages_processed_rate=2.541,
        transform_success_rate=2.541,
        queue_utilization_ratio=None,
        queue_metrics_available=False,
        pipeline_duration_average_seconds=0.006,
        persist_duration_average_seconds=0.005,
        influx_write_duration_average_seconds=0.009,
        heartbeat_age_max_seconds=1.887,
        heartbeat_age_mean_seconds=1.106,
        heartbeat_age_available=True,
        observed_device_count=4,
        device_metrics_available=True,
    )
    assert predictor.predict(healthy).label == "normal"
    assert (
        predictor.predict(
            healthy.model_copy(update={"heartbeat_missing": True, "heartbeat_age_max_seconds": 30})
        ).label
        == "missing-device-heartbeat"
    )
    stale = {"heartbeat_missing": True, "heartbeat_age_max_seconds": 30}
    assert (
        predictor.predict(healthy.model_copy(update={**stale, "broker_available": True})).label
        == "missing-device-heartbeat"
    )
    assert (
        predictor.predict(healthy.model_copy(update={**stale, "broker_available": False})).label
        == "mqtt-broker-unavailable"
    )


@pytest.mark.parametrize("broker_available", [None, True, False])
def test_training_and_runtime_use_the_same_vector(
    tmp_path: Path,
    make_features,
    broker_available: bool | None,
) -> None:
    features = make_features(
        messages_enqueued_rate=10,
        queue_growth_rate=2,
        processing_ratio=0.8,
        queue_full_rate=1,
        transform_failure_ratio=0.15,
        persistence_ratio=0.5,
        heartbeat_missing=True,
        queue_utilization_ratio=0.25,
        influx_write_failure_ratio=0.2,
        influxdb_healthy=0,
        pipeline_duration_average_seconds=2,
        heartbeat_age_max_seconds=3,
        missing_device_ratio=0.5,
        device_metrics_available=True,
        decode_failure_ratio=0.1,
        raw_validation_failure_ratio=0.2,
        business_validation_failure_ratio=0.3,
        transform_deserialize_failure_ratio=0.05,
        event_queue_drop_ratio=0.1,
        wal_queue_full_ratio=0.1,
        durability_ack_failure_rate=3,
        dlq_publish_failure_ratio=0.1,
        wal_retry_rate=2,
        wal_drop_ratio=0.2,
        wal_retry_outage_active=1,
        decode_duration_average_seconds=2,
        transform_duration_average_seconds=3,
        persist_duration_average_seconds=4,
        influx_write_duration_average_seconds=5,
        broker_available=broker_available,
    )
    for seed in (42, 43):
        rows = training_rows(seed)
        rows[0].update(features.model_dump())
        write_csv(tmp_path / f"matrix-seed-{seed}.csv", list(DATASET_COLUMNS), rows)

    dataset = load_training_dataset(tmp_path)

    assert dataset.features[0].tolist() == pytest.approx(vectorize_model_features(features))


def test_blank_optional_values_remain_unavailable(tmp_path: Path) -> None:
    write_training_matrix(tmp_path, blank_optional=True)

    dataset = load_training_dataset(tmp_path)
    first = dataset.features[0]

    for availability_column in (
        "queue_metrics_available",
        "influx_write_metrics_available",
        "heartbeat_age_available",
        "decode_metrics_available",
        "raw_validation_metrics_available",
        "business_validation_metrics_available",
        "transform_metrics_available",
        "input_queue_metrics_available",
        "wal_queue_metrics_available",
        "durability_ack_metrics_available",
        "dlq_metrics_available",
        "wal_metrics_available",
        "decode_duration_metrics_available",
        "transform_duration_metrics_available",
        "persist_duration_metrics_available",
        "influx_write_duration_metrics_available",
        "broker_metrics_available",
    ):
        assert first[MODEL_FEATURE_COLUMNS.index(availability_column)] == 0


@pytest.mark.parametrize("invalid_value", ["not-a-number", "nan", "inf"])
def test_invalid_optional_numeric_value_reports_location(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    rows = training_rows(42)
    rows[0]["queue_utilization_ratio"] = invalid_value
    write_csv(tmp_path / "matrix-seed-42.csv", list(DATASET_COLUMNS), rows)
    write_csv(tmp_path / "matrix-seed-43.csv", list(DATASET_COLUMNS), training_rows(43))

    with pytest.raises(
        DatasetError,
        match=r"matrix-seed-42\.csv:2: column 'queue_utilization_ratio'",
    ):
        load_training_dataset(tmp_path)


def test_missing_new_source_column_is_rejected(tmp_path: Path) -> None:
    fieldnames = [column for column in DATASET_COLUMNS if column != "queue_utilization_ratio"]
    write_csv(tmp_path / "matrix-seed-42.csv", fieldnames, training_rows(42))

    with pytest.raises(DatasetError, match="queue_utilization_ratio"):
        load_training_dataset(tmp_path)


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("captured_at", "captured_at is empty"),
        ("source_host", "source_host is empty"),
        ("scenario_id", "unsupported scenario_id"),
        ("simulation_run_id", "simulation_run_id is empty"),
        ("simulation_seed", "simulation_seed is not an integer"),
        ("simulation_phase", "simulation_phase is empty"),
    ],
)
def test_missing_provenance_is_rejected(
    tmp_path: Path,
    column: str,
    message: str,
) -> None:
    rows = training_rows(42)
    rows[0][column] = ""
    write_csv(tmp_path / "matrix-seed-42.csv", list(DATASET_COLUMNS), rows)
    write_csv(tmp_path / "matrix-seed-43.csv", list(DATASET_COLUMNS), training_rows(43))

    with pytest.raises(DatasetError, match=message):
        load_training_dataset(tmp_path)


def test_recovery_rows_are_validated_then_excluded(tmp_path: Path) -> None:
    write_training_matrix(tmp_path)

    dataset = load_training_dataset(tmp_path)

    assert dataset.excluded_recovery_rows == 2
    assert dataset.row_count == 12
    assert all("recovery" not in run_id for run_id in dataset.run_ids)


def test_invalid_recovery_provenance_is_not_silently_skipped(tmp_path: Path) -> None:
    rows = training_rows(42)
    rows[-1]["simulation_run_id"] = ""
    write_csv(tmp_path / "matrix-seed-42.csv", list(DATASET_COLUMNS), rows)
    write_csv(tmp_path / "matrix-seed-43.csv", list(DATASET_COLUMNS), training_rows(43))

    with pytest.raises(DatasetError, match="simulation_run_id is empty"):
        load_training_dataset(tmp_path)


def test_missing_labels_are_rejected(tmp_path: Path) -> None:
    for seed in (42, 43):
        write_csv(
            tmp_path / f"matrix-seed-{seed}.csv",
            list(DATASET_COLUMNS),
            [
                training_row(
                    label="normal",
                    seed=seed,
                    phase="normal",
                    run_suffix="normal",
                )
            ],
        )

    with pytest.raises(DatasetError, match="missing labels"):
        load_training_dataset(tmp_path)


def test_fewer_than_two_seeds_are_rejected(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "matrix-seed-42.csv",
        list(DATASET_COLUMNS),
        training_rows(42),
    )

    with pytest.raises(DatasetError, match="At least two simulation seeds"):
        load_training_dataset(tmp_path)


def test_training_defaults_to_candidate_artifact() -> None:
    training = parse_training_args([]).artifact
    assert training == parse_pipeline_args([]).model_artifact
    assert training.name == f"model-1.0-{datetime.now(UTC):%Y%m%d}.joblib"
    assert training != Path("artifacts/model.onnx")


def test_holdout_defaults_to_independent_simulation_collector() -> None:
    args = parse_holdout_args(["--expected-model-version", TEST_MODEL_VERSION])

    assert args.simulation_base_url == "http://localhost:8097"
    assert args.simulation_read_key == "local-dev-key"
    assert args.simulation_admin_key == "local-dev-admin-key"
    assert args.simulation_device_id == "esp32-simulated-01"


def test_append_rejects_legacy_dataset_header(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    write_csv(
        path,
        ["scenario_id", "expected_label"],
        [
            {
                "scenario_id": "ingestion-backpressure",
                "expected_label": "normal",
            }
        ],
    )

    with pytest.raises(ValueError, match="incompatible dataset schema"):
        write_dataset(
            reusable_dataset_rows(),
            path,
            append=True,
        )
