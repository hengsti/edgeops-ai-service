"""Compare Phase 6 feature groups using leave-one-simulation-seed-out folds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from edgeops_ai.ml import MODEL_FEATURE_COLUMNS
from scripts.common import DATASET_SCHEMA_VERSION
from scripts.simulation_manifest import (
    OPTIONAL_METRIC_GROUPS,
    SIMULATION_VARIANTS,
    SimulationManifestError,
    validate_manifest_datasets,
)
from scripts.train_model import (
    CANDIDATE_MODELS,
    DatasetError,
    TrainingDataset,
    cross_validated_predictions,
    load_training_dataset,
    metrics_for_predictions,
)

ABLATION_SCHEMA_VERSION = "1.0"

A1_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS[:16]

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "decode-validation": (
        "decode_failure_ratio",
        "decode_metrics_available",
        "raw_validation_failure_ratio",
        "raw_validation_metrics_available",
        "business_validation_failure_ratio",
        "business_validation_metrics_available",
        "transform_deserialize_failure_ratio",
        "transform_metrics_available",
    ),
    "input-wal-pressure": (
        "event_queue_drop_ratio",
        "input_queue_metrics_available",
        "wal_queue_full_ratio",
        "wal_queue_metrics_available",
        "durability_ack_failure_log_rate",
        "durability_ack_metrics_available",
    ),
    "dlq": (
        "dlq_publish_failure_ratio",
        "dlq_metrics_available",
    ),
    "wal-influx": (
        "wal_retry_log_rate",
        "wal_drop_ratio",
        "wal_retry_outage_active",
        "wal_metrics_available",
    ),
    "stage-latency": (
        "decode_duration_log_seconds",
        "decode_duration_metrics_available",
        "transform_duration_log_seconds",
        "transform_duration_metrics_available",
        "persist_duration_log_seconds",
        "persist_duration_metrics_available",
        "influx_write_duration_log_seconds",
        "influx_write_duration_metrics_available",
    ),
    "broker": (
        "broker_unavailable",
        "broker_metrics_available",
    ),
}


def feature_sets() -> dict[str, tuple[str, ...]]:
    """Return ordered A1, enriched, additive, and subtractive feature sets."""

    a1 = set(A1_FEATURE_COLUMNS)
    all_features = set(MODEL_FEATURE_COLUMNS)
    selected: dict[str, set[str]] = {
        "a1": a1,
        "full": all_features,
    }

    for group_name, group_columns in FEATURE_GROUPS.items():
        group = set(group_columns)
        selected[f"a1-plus-{group_name}"] = a1 | group
        selected[f"full-minus-{group_name}"] = all_features - group

    return {
        name: tuple(column for column in MODEL_FEATURE_COLUMNS if column in columns)
        for name, columns in selected.items()
    }


def subset_dataset(
    dataset: TrainingDataset,
    columns: Sequence[str],
) -> TrainingDataset:
    indices = [MODEL_FEATURE_COLUMNS.index(column) for column in columns]
    return replace(dataset, features=dataset.features[:, indices])


def _metric_delta(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"] - baseline["accuracy"]),
        "macro_f1": float(metrics["macro_f1"] - baseline["macro_f1"]),
        "worst_seed_accuracy": float(
            min(fold["accuracy"] for fold in metrics["per_seed"])
            - min(fold["accuracy"] for fold in baseline["per_seed"])
        ),
    }


def _slice_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    rows = int(mask.sum())
    if rows == 0:
        return {"rows": 0, "accuracy": None, "macro_f1": None}
    return {"rows": rows, **metrics_for_predictions(expected[mask], predicted[mask])}


def _candidate_evaluation(
    dataset: TrainingDataset,
    candidate: Any,
    provenance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predicted, per_seed = cross_validated_predictions(dataset, candidate)
    variants = np.asarray([entry["variant_id"] for entry in provenance], dtype=object)
    unavailable = [tuple(entry["unavailable_metric_groups"]) for entry in provenance]
    unavailable_counts = np.asarray([len(groups) for groups in unavailable], dtype=np.int64)
    metrics = {
        "strategy": "leave-one-simulation-seed-out",
        **metrics_for_predictions(dataset.labels, predicted),
        "per_seed": per_seed,
    }
    metrics["slices"] = {
        "by_variant": {
            variant: _slice_metrics(dataset.labels, predicted, variants == variant)
            for variant in SIMULATION_VARIANTS
        },
        "by_unavailable_group_count": {
            str(count): _slice_metrics(
                dataset.labels,
                predicted,
                unavailable_counts == count,
            )
            for count in range(4)
        },
        "by_unavailable_group": {
            group: _slice_metrics(
                dataset.labels,
                predicted,
                np.asarray([group in groups for groups in unavailable], dtype=bool),
            )
            for group in OPTIONAL_METRIC_GROUPS
        },
    }
    return metrics


def build_ablation_report(
    dataset: TrainingDataset,
    provenance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(provenance) != dataset.row_count:
        raise SimulationManifestError("Ablation provenance does not align with stable rows")
    evaluated_sets: dict[str, dict[str, Any]] = {}

    for set_name, columns in feature_sets().items():
        subset = subset_dataset(dataset, columns)
        evaluated_sets[set_name] = {
            "feature_columns": list(columns),
            "feature_count": len(columns),
            "candidates": {
                candidate_name: _candidate_evaluation(subset, candidate, provenance)
                for candidate_name, candidate in CANDIDATE_MODELS.items()
            },
        }

    comparisons: dict[str, dict[str, dict[str, float]]] = {}

    for group_name in FEATURE_GROUPS:
        additive = evaluated_sets[f"a1-plus-{group_name}"]["candidates"]
        comparisons[group_name] = {
            candidate_name: _metric_delta(
                additive[candidate_name],
                evaluated_sets["a1"]["candidates"][candidate_name],
            )
            for candidate_name in CANDIDATE_MODELS
        }

    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "evaluation_strategy": "leave-one-simulation-seed-out",
        "dataset": {
            "input_files": list(dataset.input_files),
            "stable_rows": dataset.row_count,
            "excluded_recovery_rows": dataset.excluded_recovery_rows,
            "simulation_seeds": sorted({int(seed) for seed in dataset.seeds}),
        },
        "feature_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
        "feature_sets": evaluated_sets,
        "a1_additive_deltas": comparisons,
    }


def evaluate_feature_ablation(
    *,
    input_dir: Path,
    output: Path,
) -> dict[str, Any]:
    manifest, _ = validate_manifest_datasets(input_dir)
    dataset = load_training_dataset(input_dir)
    try:
        provenance = [manifest["runs"][str(run_id)] for run_id in dataset.run_ids]
    except KeyError as error:
        raise SimulationManifestError(
            f"Stable row references unknown simulation run ID: {error.args[0]}"
        ) from error
    report = build_ablation_report(dataset, provenance)
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
        help="JSON report path (default: <input-dir>/ablation-report.json).",
    )
    args = parser.parse_args(argv)

    if args.output is None:
        args.output = args.input_dir / "ablation-report.json"

    return args


def print_report(report: Mapping[str, Any], output: Path) -> None:
    print(
        "Feature ablation: "
        f"{report['dataset']['stable_rows']} stable rows; "
        f"{len(report['dataset']['simulation_seeds'])} grouped seeds"
    )
    print(
        f"{'feature set':34} {'candidate':22} "
        f"{'features':>8} {'accuracy':>9} {'macro-F1':>9} {'worst seed':>11}"
    )

    for set_name, feature_set in report["feature_sets"].items():
        for candidate_name, metrics in feature_set["candidates"].items():
            worst_seed = min(fold["accuracy"] for fold in metrics["per_seed"])
            print(
                f"{set_name:34} {candidate_name:22} "
                f"{feature_set['feature_count']:8d} "
                f"{metrics['accuracy']:9.3f} "
                f"{metrics['macro_f1']:9.3f} "
                f"{worst_seed:11.3f}"
            )

    print(f"Feature ablation report written to {output}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        report = evaluate_feature_ablation(
            input_dir=args.input_dir,
            output=args.output,
        )
    except (DatasetError, OSError, SimulationManifestError, ValueError) as error:
        print(f"Feature ablation failed: {error}")
        return 1

    print_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
