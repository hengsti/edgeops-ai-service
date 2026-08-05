"""Validate promotion-readiness gates for an end-to-end candidate holdout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edgeops_ai.ml import ML_MODEL_VERSION
from scripts.common import DATASET_SCHEMA_VERSION, LABELS, SCENARIOS
from scripts.simulation_manifest import (
    SIMULATION_CONTRACT_VERSION,
    SIMULATION_VARIANTS,
    validate_manifest_datasets,
)
from scripts.train_model import load_training_dataset

HOLDOUT_VALIDATION_SCHEMA_VERSION = "1.0"


class HoldoutValidationError(ValueError):
    """Raised when holdout evidence cannot be validated."""


def _gate(name: str, observed: Any, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "passed": bool(passed),
    }


def build_holdout_validation(
    *,
    manifest: Mapping[str, Any],
    evaluation_report: Mapping[str, Any],
    model_versions: set[str],
    stable_matrix_finite: bool,
    training_seeds: set[int],
    min_accuracy: float = 0.95,
    min_macro_f1: float = 0.95,
    min_label_recall: float = 0.90,
) -> dict[str, Any]:
    runs = manifest.get("runs", {})
    holdout_seeds = {int(entry["seed"]) for entry in runs.values()}
    scenario_variant_counts = {
        scenario: Counter(
            entry["variant_id"] for entry in runs.values() if entry["scenario_id"] == scenario
        )
        for scenario in SCENARIOS
    }
    all_variants = all(
        scenario_variant_counts[scenario].get(variant, 0) > 0
        for scenario in SCENARIOS
        for variant in SIMULATION_VARIANTS
    )

    metrics = evaluation_report.get("primary_stable_phase_metrics", {})
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    recalls = {label: metrics.get("per_label", {}).get(label, {}).get("recall") for label in LABELS}
    detection_runs = evaluation_report.get("detection_delay", {}).get("runs", [])
    recovery_runs = evaluation_report.get("recovery", {}).get("runs", [])

    gates = [
        _gate(
            "contract-versions",
            {
                "simulation": manifest.get("simulation_contract_version"),
                "dataset": manifest.get("dataset_schema_version"),
            },
            {
                "simulation": SIMULATION_CONTRACT_VERSION,
                "dataset": DATASET_SCHEMA_VERSION,
            },
            manifest.get("simulation_contract_version") == SIMULATION_CONTRACT_VERSION
            and manifest.get("dataset_schema_version") == DATASET_SCHEMA_VERSION,
        ),
        _gate(
            "all-scenarios-and-variants-represented",
            {
                scenario: dict(sorted(counts.items()))
                for scenario, counts in scenario_variant_counts.items()
            },
            {"minimum_runs_per_scenario_variant": 1},
            all_variants,
        ),
        _gate(
            "holdout-seeds-unseen",
            {
                "holdout_seeds": sorted(holdout_seeds),
                "overlap": sorted(holdout_seeds & training_seeds),
            },
            {"training_seed_overlap": []},
            bool(holdout_seeds) and not holdout_seeds & training_seeds,
        ),
        _gate(
            "stable-model-matrix-finite",
            stable_matrix_finite,
            True,
            stable_matrix_finite,
        ),
        _gate(
            "candidate-model-version",
            sorted(model_versions),
            [ML_MODEL_VERSION],
            model_versions == {ML_MODEL_VERSION},
        ),
        _gate(
            "stable-accuracy-and-macro-f1",
            {"accuracy": accuracy, "macro_f1": macro_f1},
            {"minimum_accuracy": min_accuracy, "minimum_macro_f1": min_macro_f1},
            isinstance(accuracy, (int, float))
            and isinstance(macro_f1, (int, float))
            and float(accuracy) >= min_accuracy
            and float(macro_f1) >= min_macro_f1,
        ),
        _gate(
            "stable-per-label-recall",
            recalls,
            {"minimum": min_label_recall},
            all(
                isinstance(recall, (int, float)) and float(recall) >= min_label_recall
                for recall in recalls.values()
            ),
        ),
        _gate(
            "all-incidents-detected",
            {
                "runs": len(detection_runs),
                "detected": sum(bool(run.get("detected")) for run in detection_runs),
            },
            {"all_runs_detected": True},
            bool(detection_runs) and all(run.get("detected") is True for run in detection_runs),
        ),
        _gate(
            "all-recovery-runs-clear",
            {
                "runs": len(recovery_runs),
                "cleared": sum(
                    bool(run.get("cleared_during_observation")) for run in recovery_runs
                ),
                "maximum_sampled_clearance_delay_seconds": evaluation_report.get(
                    "recovery", {}
                ).get("max_sampled_clearance_delay_seconds"),
            },
            {"all_runs_clear": True},
            bool(recovery_runs)
            and all(run.get("cleared_during_observation") is True for run in recovery_runs),
        ),
    ]
    return {
        "schema_version": HOLDOUT_VALIDATION_SCHEMA_VERSION,
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HoldoutValidationError(f"Unable to read JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise HoldoutValidationError(f"JSON file must contain an object: {path}")
    return payload


def _model_versions(input_dir: Path) -> set[str]:
    versions: set[str] = set()
    for path in sorted(input_dir.glob("*-seed-*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                version = (row.get("model_version") or "").strip()
                if not version:
                    raise HoldoutValidationError(f"{path}: model_version is empty")
                versions.add(version)
    return versions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-seed-range", nargs=2, type=int, default=(42, 61))
    args = parser.parse_args(argv)
    args.evaluation_report = args.evaluation_report or args.input_dir / "candidate-report.json"
    args.output = args.output or args.input_dir / "promotion-readiness.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest, _ = validate_manifest_datasets(args.input_dir)
        dataset = load_training_dataset(args.input_dir)
        first, last = args.training_seed_range
        if first > last:
            raise HoldoutValidationError("training seed range must be ordered")
        report = build_holdout_validation(
            manifest=manifest,
            evaluation_report=_read_json(args.evaluation_report),
            model_versions=_model_versions(args.input_dir),
            stable_matrix_finite=bool(np.isfinite(dataset.features).all()),
            training_seeds=set(range(first, last + 1)),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (HoldoutValidationError, OSError, ValueError) as error:
        print(f"Candidate holdout validation failed to run: {error}")
        return 2

    for gate in report["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {gate['name']}")
    print(f"Promotion-readiness report written to {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
