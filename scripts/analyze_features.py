"""Inspect Phase 6 feature distributions, availability, and scenario behavior."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.common import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    SCENARIOS,
    normalize_label_value,
)
from scripts.simulation_manifest import (
    OPTIONAL_METRIC_GROUPS,
    SimulationManifestError,
    validate_manifest_datasets,
)
from scripts.train_model import DatasetError, load_training_dataset

ANALYSIS_SCHEMA_VERSION = "1.0"

NUMERIC_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column != "missing_device_ids"
)

AVAILABILITY_COLUMNS = tuple(
    column for column in NUMERIC_FEATURE_COLUMNS if column.endswith("_available")
)

BOUNDED_SOURCE_COLUMNS = (
    "processing_ratio",
    "transform_failure_ratio",
    "persistence_ratio",
    "queue_utilization_ratio",
    "influx_write_failure_ratio",
    "influxdb_healthy",
    "missing_device_ratio",
    "decode_failure_ratio",
    "raw_validation_failure_ratio",
    "business_validation_failure_ratio",
    "transform_deserialize_failure_ratio",
    "event_queue_drop_ratio",
    "wal_queue_full_ratio",
    "dlq_publish_failure_ratio",
    "wal_drop_ratio",
    "wal_retry_outage_active",
)

A2_SOURCE_COLUMNS = (
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
)

STAGE_LATENCY_COLUMNS = (
    "decode_duration_average_seconds",
    "transform_duration_average_seconds",
    "persist_duration_average_seconds",
    "influx_write_duration_average_seconds",
)

PRESSURE_COLUMNS = (
    "event_queue_drop_ratio",
    "wal_queue_full_ratio",
)


class FeatureAnalysisError(ValueError):
    """Raised when a dataset cannot be analyzed safely."""


def _parse_numeric(
    value: str | None,
    *,
    column: str,
    location: str,
) -> float | None:
    text = (value or "").strip().lower()

    if not text:
        return None

    if text == "true":
        return 1.0

    if text == "false":
        return 0.0

    try:
        parsed = float(text)
    except ValueError as error:
        raise FeatureAnalysisError(
            f"{location}: column {column!r} is not numeric or boolean: {value!r}"
        ) from error

    if not math.isfinite(parsed):
        raise FeatureAnalysisError(f"{location}: column {column!r} must be finite, got {value!r}")

    return parsed


def load_feature_rows(
    input_dir: Path,
    *,
    pattern: str = "*-seed-*.csv",
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Load and validate every generated row, including recovery rows."""

    paths = tuple(sorted(input_dir.glob(pattern)))

    if not paths:
        raise FeatureAnalysisError(f"No {pattern} files found in {input_dir}")

    _, provenance = validate_manifest_datasets(input_dir, pattern=pattern)
    rows: list[dict[str, Any]] = []

    for path in paths:
        run_id, manifest_entry = provenance[path.name]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            actual_columns = set(reader.fieldnames or ())
            missing_columns = sorted(set(DATASET_COLUMNS) - actual_columns)

            if missing_columns:
                raise FeatureAnalysisError(
                    f"{path}: missing required columns: {', '.join(missing_columns)}"
                )

            file_row_count = 0

            for line_number, source_row in enumerate(reader, start=2):
                file_row_count += 1
                location = f"{path}:{line_number}"
                dataset_version = (source_row.get("dataset_schema_version") or "").strip()

                if dataset_version != DATASET_SCHEMA_VERSION:
                    raise FeatureAnalysisError(
                        f"{location}: unsupported dataset_schema_version "
                        f"{dataset_version!r}; expected {DATASET_SCHEMA_VERSION!r}"
                    )

                scenario = normalize_label_value(source_row.get("scenario_id") or "")
                phase = normalize_label_value(source_row.get("simulation_phase") or "")

                if scenario not in SCENARIOS:
                    raise FeatureAnalysisError(f"{location}: unsupported scenario_id {scenario!r}")

                if not phase:
                    raise FeatureAnalysisError(f"{location}: simulation_phase is empty")

                parsed_row: dict[str, Any] = {
                    "scenario_id": scenario,
                    "simulation_phase": phase,
                    "simulation_run_id": run_id,
                    "simulation_seed": int(manifest_entry["seed"]),
                    "variant_id": str(manifest_entry["variant_id"]),
                    "unavailable_metric_groups": tuple(manifest_entry["unavailable_metric_groups"]),
                }

                for column in NUMERIC_FEATURE_COLUMNS:
                    parsed_row[column] = _parse_numeric(
                        source_row.get(column),
                        column=column,
                        location=location,
                    )

                rows.append(parsed_row)

            if file_row_count == 0:
                raise FeatureAnalysisError(f"{path}: dataset file is empty")

    return rows, tuple(path.name for path in paths)


def summarize_numeric(
    rows: Sequence[Mapping[str, Any]],
    column: str,
) -> dict[str, int | float | None]:
    values = [float(row[column]) for row in rows if row.get(column) is not None]

    summary: dict[str, int | float | None] = {
        "rows": len(rows),
        "present": len(values),
        "missing": len(rows) - len(values),
        "missing_ratio": (len(rows) - len(values)) / len(rows) if rows else 0.0,
        "min": None,
        "median": None,
        "max": None,
        "nonzero": 0,
        "unique": 0,
    }

    if values:
        summary.update(
            {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
                "nonzero": sum(value != 0.0 for value in values),
                "unique": len(set(values)),
            }
        )

    return summary


def _correlation(
    rows: Sequence[Mapping[str, Any]],
    left_column: str,
    right_column: str,
) -> dict[str, int | float | None]:
    pairs = [
        (float(row[left_column]), float(row[right_column]))
        for row in rows
        if row.get(left_column) is not None and row.get(right_column) is not None
    ]

    if len(pairs) < 2:
        return {"paired_rows": len(pairs), "pearson_correlation": None}

    left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)

    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])

    return {
        "paired_rows": len(pairs),
        "pearson_correlation": correlation,
    }


def build_feature_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_files: Sequence[str],
    stable_model_matrix_is_finite: bool,
) -> dict[str, Any]:
    scenarios = sorted({str(row["scenario_id"]) for row in rows})
    phases = sorted({str(row["simulation_phase"]) for row in rows})
    scenario_rows = {
        scenario: [row for row in rows if row["scenario_id"] == scenario] for scenario in scenarios
    }
    phase_rows = {
        phase: [row for row in rows if row["simulation_phase"] == phase] for phase in phases
    }
    variants = sorted({str(row["variant_id"]) for row in rows})
    variant_rows = {
        variant: [row for row in rows if row["variant_id"] == variant] for variant in variants
    }
    scenario_variant_rows = {
        scenario: {
            variant: [row for row in selected_rows if row["variant_id"] == variant]
            for variant in variants
        }
        for scenario, selected_rows in scenario_rows.items()
    }
    unique_runs = {str(row["simulation_run_id"]): row for row in rows}

    def optional_availability(selected_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total_cells = len(selected_rows) * len(OPTIONAL_METRIC_GROUPS)
        by_group = {
            group: (
                sum(group not in row["unavailable_metric_groups"] for row in selected_rows)
                / len(selected_rows)
                if selected_rows
                else None
            )
            for group in OPTIONAL_METRIC_GROUPS
        }
        available_cells = sum(
            group not in row["unavailable_metric_groups"]
            for row in selected_rows
            for group in OPTIONAL_METRIC_GROUPS
        )
        return {
            "rows": len(selected_rows),
            "available_cells": available_cells,
            "total_cells": total_cells,
            "available_ratio": available_cells / total_cells if total_cells else None,
            "by_group": by_group,
        }

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset": {
            "input_files": list(input_files),
            "rows": len(rows),
            "scenarios": scenarios,
            "phases": phases,
            "stable_model_matrix_is_finite": stable_model_matrix_is_finite,
            "simulation_runs": len(unique_runs),
            "variants": variants,
            "run_counts_by_scenario_variant": {
                scenario: {
                    variant: len(
                        {
                            row["simulation_run_id"]
                            for row in selected_rows
                            if row["variant_id"] == variant
                        }
                    )
                    for variant in variants
                }
                for scenario, selected_rows in scenario_rows.items()
            },
            "row_counts_by_scenario_variant": {
                scenario: {
                    variant: len(selected_rows) for variant, selected_rows in variants_rows.items()
                }
                for scenario, variants_rows in scenario_variant_rows.items()
            },
        },
        "feature_distributions": {
            column: summarize_numeric(rows, column) for column in NUMERIC_FEATURE_COLUMNS
        },
        "availability": {
            column: {
                **summarize_numeric(rows, column),
                "available_ratio": (
                    sum(float(row[column]) != 0.0 for row in rows if row[column] is not None)
                    / len(rows)
                    if rows
                    else 0.0
                ),
            }
            for column in AVAILABILITY_COLUMNS
        },
        "optional_group_availability": {
            "overall": optional_availability(rows),
            "by_scenario": {
                scenario: optional_availability(selected_rows)
                for scenario, selected_rows in scenario_rows.items()
            },
            "by_variant": {
                variant: optional_availability(selected_rows)
                for variant, selected_rows in variant_rows.items()
            },
        },
        "unavailable_group_combinations": dict(
            sorted(
                {
                    ",".join(row["unavailable_metric_groups"]) or "none": sum(
                        candidate["unavailable_metric_groups"] == row["unavailable_metric_groups"]
                        for candidate in unique_runs.values()
                    )
                    for row in unique_runs.values()
                }.items()
            )
        ),
        "a2_by_scenario": {
            scenario: {
                column: summarize_numeric(selected_rows, column) for column in A2_SOURCE_COLUMNS
            }
            for scenario, selected_rows in scenario_rows.items()
        },
        "a2_by_variant": {
            variant: {
                column: summarize_numeric(selected_rows, column) for column in A2_SOURCE_COLUMNS
            }
            for variant, selected_rows in variant_rows.items()
        },
        "failure_signals_by_phase": {
            phase: {
                column: summarize_numeric(selected_rows, column) for column in A2_SOURCE_COLUMNS
            }
            for phase, selected_rows in phase_rows.items()
        },
        "constant_features_by_scenario": {
            scenario: [
                column
                for column in NUMERIC_FEATURE_COLUMNS
                if summarize_numeric(selected_rows, column)["unique"] <= 1
            ]
            for scenario, selected_rows in scenario_rows.items()
        },
        "constant_feature_counts_by_scenario": {
            scenario: sum(
                summarize_numeric(selected_rows, column)["unique"] <= 1
                for column in NUMERIC_FEATURE_COLUMNS
            )
            for scenario, selected_rows in scenario_rows.items()
        },
        "constant_features_by_scenario_variant": {
            scenario: {
                variant: [
                    column
                    for column in NUMERIC_FEATURE_COLUMNS
                    if summarize_numeric(selected_rows, column)["unique"] <= 1
                ]
                for variant, selected_rows in variants_rows.items()
            }
            for scenario, variants_rows in scenario_variant_rows.items()
        },
        "bounded_transform_clipping": {
            column: {
                "below_zero": sum(
                    row[column] is not None and float(row[column]) < 0.0 for row in rows
                ),
                "above_one": sum(
                    row[column] is not None and float(row[column]) > 1.0 for row in rows
                ),
            }
            for column in BOUNDED_SOURCE_COLUMNS
        },
        "event_wal_pressure_by_scenario": {
            scenario: {
                column: summarize_numeric(selected_rows, column) for column in PRESSURE_COLUMNS
            }
            for scenario, selected_rows in scenario_rows.items()
        },
        "stage_latency_vs_pipeline": {
            column: _correlation(rows, column, "pipeline_duration_average_seconds")
            for column in STAGE_LATENCY_COLUMNS
        },
    }


def analyze_features(
    *,
    input_dir: Path,
    output: Path,
) -> dict[str, Any]:
    rows, input_files = load_feature_rows(input_dir)
    training_dataset = load_training_dataset(input_dir)
    report = build_feature_report(
        rows,
        input_files=input_files,
        stable_model_matrix_is_finite=bool(np.isfinite(training_dataset.features).all()),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the fresh Phase 6 *-seed-*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: <input-dir>/feature-audit.json).",
    )
    args = parser.parse_args(argv)

    if args.output is None:
        args.output = args.input_dir / "feature-audit.json"

    return args


def print_report(report: Mapping[str, Any], output: Path) -> None:
    dataset = report["dataset"]
    print(
        "Feature audit: "
        f"{dataset['rows']} rows across {len(dataset['input_files'])} files; "
        f"stable model matrix finite={dataset['stable_model_matrix_is_finite']}"
    )
    print("Availability:")

    for column, summary in report["availability"].items():
        print(
            f"  {column:42} "
            f"available={summary['available_ratio']:.3f} "
            f"missing={summary['missing_ratio']:.3f}"
        )

    print("Constant features by scenario:")

    for scenario, columns in report["constant_features_by_scenario"].items():
        print(f"  {scenario:31} {len(columns)}")

    print(f"Feature audit written to {output}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        report = analyze_features(
            input_dir=args.input_dir,
            output=args.output,
        )
    except (
        DatasetError,
        FeatureAnalysisError,
        OSError,
        SimulationManifestError,
    ) as error:
        print(f"Feature analysis failed: {error}")
        return 1

    print_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
