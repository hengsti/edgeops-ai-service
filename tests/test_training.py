import csv
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts.collect_rules_dataset import build_dataset_row, write_dataset
from scripts.common import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    FEATURE_COLUMNS,
)
from scripts.run_pipeline import (
    dataset_is_complete,
)
from scripts.run_pipeline import (
    parse_args as parse_pipeline_args,
)
from scripts.train_model import (
    DatasetError,
    load_training_dataset,
)
from scripts.train_model import (
    parse_args as parse_training_args,
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


def test_current_dataset_is_reusable(tmp_path: Path) -> None:
    path = tmp_path / "ingestion-backpressure-seed-42.csv"
    write_csv(path, list(DATASET_COLUMNS), reusable_dataset_rows())

    assert dataset_is_complete(
        path,
        scenario="ingestion-backpressure",
        seed=42,
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
    )


def test_direct_training_rejects_old_dataset_version(tmp_path: Path) -> None:
    row = dict.fromkeys(DATASET_COLUMNS, "0")
    row.update(
        {
            "dataset_schema_version": "1.0",
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


def test_training_defaults_to_candidate_artifact() -> None:
    assert parse_training_args([]).artifact == Path("artifacts/model-candidate.joblib")
    assert parse_pipeline_args([]).model_artifact == Path("artifacts/model-candidate.joblib")


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
