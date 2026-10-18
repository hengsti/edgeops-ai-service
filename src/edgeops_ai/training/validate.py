"""Validate promotion-readiness gates for an end-to-end candidate holdout."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edgeops_ai.training.common import DATASET_SCHEMA_VERSION, LABELS, SCENARIOS
from edgeops_ai.training.dataset import load_training_dataset
from edgeops_ai.training.manifest import (
    SIMULATION_CONTRACT_VERSION,
    SIMULATION_VARIANTS,
    load_manifest,
    validate_manifest_datasets,
)
from edgeops_ai.training.reports import read_report, write_report

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
    expected_model_version: str,
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
            [expected_model_version],
            model_versions == {expected_model_version},
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
    parser.add_argument("--expected-model-version", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-seed-range", nargs=2, type=int, default=(42, 61))
    parser.add_argument(
        "--training-dir",
        type=Path,
        help="Also check training, feature audit and ablation evidence.",
    )
    args = parser.parse_args(argv)
    args.evaluation_report = args.evaluation_report or args.input_dir / "candidate-report.json"
    args.output = args.output or args.input_dir / "release-validation.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        first, last = args.training_seed_range
        if first > last:
            raise HoldoutValidationError("training seed range must be ordered")
        report = validate_release(
            input_dir=args.input_dir,
            evaluation_report=args.evaluation_report,
            expected_model_version=args.expected_model_version,
            training_seeds=set(range(first, last + 1)),
            training_dir=args.training_dir,
        )
        write_report(report, args.output)
    except (HoldoutValidationError, OSError, ValueError) as error:
        print(f"Candidate holdout validation failed to run: {error}")
        return 2

    for gate in report["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {gate['name']}")
    print(f"Promotion-readiness report written to {args.output}")
    return 0 if report["passed"] else 1


ACCEPTANCE_SCHEMA_VERSION = "1.0"


class AcceptanceError(ValueError):
    """Raised when release evidence is malformed."""


def _recovery_gate(report: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    recovery = report.get("recovery", {})
    runs = recovery.get("runs", []) if isinstance(recovery, dict) else []
    cleared = sum(bool(run.get("cleared_during_observation")) for run in runs)
    observed = {"runs": len(runs), "cleared_runs": cleared}
    return observed, bool(runs) and cleared == len(runs)


def build_acceptance_report(
    *,
    manifest: Mapping[str, Any],
    rule_report: Mapping[str, Any],
    training_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
    ablation_report: Mapping[str, Any],
    min_optional_availability: float = 0.70,
    max_optional_availability: float = 0.95,
    max_constant_features: int = 29,
    min_a2_macro_f1_delta: float = 0.02,
    max_worst_seed_accuracy_loss: float = 0.02,
    min_holdout_accuracy: float = 0.95,
    min_holdout_label_recall: float = 0.90,
) -> dict[str, Any]:
    runs = manifest.get("runs", {})
    counts = {
        scenario: Counter(
            entry["variant_id"] for entry in runs.values() if entry["scenario_id"] == scenario
        )
        for scenario in SCENARIOS
    }
    represented = all(
        counts[scenario].get(variant, 0) > 0
        for scenario in SCENARIOS
        for variant in SIMULATION_VARIANTS
    )
    gates = [
        _gate(
            "all-scenarios-and-variants-represented",
            observed={scenario: dict(sorted(count.items())) for scenario, count in counts.items()},
            threshold={"minimum_runs_per_scenario_variant": 1},
            passed=represented,
        )
    ]

    finite = bool(feature_audit.get("dataset", {}).get("stable_model_matrix_is_finite"))
    gates.append(
        _gate(
            "stable-model-matrices-finite",
            observed=finite,
            threshold=True,
            passed=finite,
        )
    )

    availability = (
        feature_audit.get("optional_group_availability", {})
        .get("overall", {})
        .get("available_ratio")
    )
    availability_ok = isinstance(availability, (int, float)) and (
        min_optional_availability <= float(availability) <= max_optional_availability
    )
    gates.append(
        _gate(
            "optional-group-availability",
            observed=availability,
            threshold={"minimum": min_optional_availability, "maximum": max_optional_availability},
            passed=availability_ok,
        )
    )

    constants = feature_audit.get("constant_feature_counts_by_scenario", {})
    constants_ok = set(constants) == set(SCENARIOS) and all(
        isinstance(count, int) and count <= max_constant_features for count in constants.values()
    )
    gates.append(
        _gate(
            "constant-features-per-scenario",
            observed=constants,
            threshold={"maximum": max_constant_features},
            passed=constants_ok,
        )
    )

    feature_sets = ablation_report.get("feature_sets", {})
    universally_perfect: list[str] = []
    candidates = set(feature_sets.get("a1", {}).get("candidates", {})) & set(
        feature_sets.get("full", {}).get("candidates", {})
    )
    for set_name, feature_set in feature_sets.items():
        for candidate_name, metrics in feature_set.get("candidates", {}).items():
            folds = metrics.get("per_seed", [])
            if folds and all(
                fold.get("accuracy") == 1.0 and fold.get("macro_f1") == 1.0 for fold in folds
            ):
                universally_perfect.append(f"{set_name}/{candidate_name}")
    gates.append(
        _gate(
            "no-universally-perfect-feature-set-candidate",
            observed=universally_perfect,
            threshold={"maximum_offending_combinations": 0},
            passed=not universally_perfect,
        )
    )

    improvements: dict[str, dict[str, float | bool]] = {}
    qualifying: list[str] = []
    for candidate_name in sorted(candidates):
        a1 = feature_sets["a1"]["candidates"][candidate_name]
        full = feature_sets["full"]["candidates"][candidate_name]
        macro_delta = float(full["macro_f1"] - a1["macro_f1"])
        a1_worst = min(float(fold["accuracy"]) for fold in a1["per_seed"])
        full_worst = min(float(fold["accuracy"]) for fold in full["per_seed"])
        worst_delta = full_worst - a1_worst
        candidate_passed = (
            macro_delta >= min_a2_macro_f1_delta and worst_delta >= -max_worst_seed_accuracy_loss
        )
        improvements[candidate_name] = {
            "a1_macro_f1": float(a1["macro_f1"]),
            "full_macro_f1": float(full["macro_f1"]),
            "macro_f1_delta": macro_delta,
            "a1_worst_seed_accuracy": a1_worst,
            "full_worst_seed_accuracy": full_worst,
            "worst_seed_accuracy_delta": worst_delta,
            "passed": candidate_passed,
        }
        if candidate_passed:
            qualifying.append(candidate_name)
    gates.append(
        _gate(
            "a2-improves-a1-with-stable-worst-seed",
            observed={"candidates": improvements, "qualifying": qualifying},
            threshold={
                "minimum_macro_f1_delta": min_a2_macro_f1_delta,
                "minimum_worst_seed_accuracy_delta": -max_worst_seed_accuracy_loss,
            },
            passed=bool(qualifying),
        )
    )

    rule_recovery, rule_recovery_ok = _recovery_gate(rule_report)
    gates.append(
        _gate(
            "rule-recovery-clears",
            observed=rule_recovery,
            threshold={"all_runs_clear": True},
            passed=rule_recovery_ok,
        )
    )

    training_quality = bool(training_report.get("quality_gate", {}).get("passed"))
    gates.append(
        _gate(
            "training-quality-gate",
            observed=training_report.get("quality_gate"),
            threshold={"passed": True},
            passed=training_quality,
        )
    )

    holdout_metrics = holdout_report.get("primary_stable_phase_metrics", {})
    holdout_accuracy = holdout_metrics.get("accuracy")
    recalls = {
        label: metrics.get("recall")
        for label, metrics in holdout_metrics.get("per_label", {}).items()
    }
    holdout_recovery, holdout_recovery_ok = _recovery_gate(holdout_report)
    holdout_ok = (
        isinstance(holdout_accuracy, (int, float))
        and float(holdout_accuracy) >= min_holdout_accuracy
        and recalls
        and all(
            isinstance(recall, (int, float)) and float(recall) >= min_holdout_label_recall
            for recall in recalls.values()
        )
        and holdout_recovery_ok
    )
    gates.append(
        _gate(
            "deployed-holdout-gates",
            observed={
                "accuracy": holdout_accuracy,
                "per_label_recall": recalls,
                "recovery": holdout_recovery,
            },
            threshold={
                "minimum_accuracy": min_holdout_accuracy,
                "minimum_label_recall": min_holdout_label_recall,
                "all_recovery_runs_clear": True,
            },
            passed=bool(holdout_ok),
        )
    )

    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "simulation_contract_version": manifest.get("simulation_contract_version"),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
    }


def validate_release(
    *,
    input_dir: Path,
    evaluation_report: Path,
    expected_model_version: str,
    training_seeds: set[int],
    training_dir: Path | None = None,
) -> dict[str, Any]:
    """Check holdout gates and, when supplied, the complete training evidence."""
    manifest, _ = validate_manifest_datasets(input_dir)
    dataset = load_training_dataset(input_dir)
    holdout = read_report(evaluation_report)
    training = read_report(training_dir / "training-report.json") if training_dir else None
    if training is not None:
        training_seeds = set(training["dataset"]["simulation_seeds"])
    report = build_holdout_validation(
        manifest=manifest,
        evaluation_report=holdout,
        model_versions=_model_versions(input_dir),
        expected_model_version=expected_model_version,
        stable_matrix_finite=bool(np.isfinite(dataset.features).all()),
        training_seeds=training_seeds,
    )
    if training_dir is not None and training is not None:
        acceptance = build_acceptance_report(
            manifest=load_manifest(training_dir),
            rule_report=read_report(training_dir / "rule-report.json"),
            training_report=training,
            holdout_report=holdout,
            feature_audit=read_report(training_dir / "feature-audit.json"),
            ablation_report=read_report(training_dir / "ablation-report.json"),
        )
        report["gates"].extend(
            {**gate, "name": "training-" + gate["name"]} for gate in acceptance["gates"]
        )
        report["passed"] = all(gate["passed"] for gate in report["gates"])
    return report


def parse_acceptance_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rule-report", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--holdout-report", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path)
    parser.add_argument("--ablation-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.rule_report = args.rule_report or args.input_dir / "rule-report.json"
    args.training_report = args.training_report or args.input_dir / "training-report.json"
    args.feature_audit = args.feature_audit or args.input_dir / "feature-audit.json"
    args.ablation_report = args.ablation_report or args.input_dir / "ablation-report.json"
    args.output = args.output or args.input_dir / "acceptance-report.json"
    return args


def acceptance_main(argv: Sequence[str] | None = None) -> int:
    args = parse_acceptance_args(argv)
    try:
        report = build_acceptance_report(
            manifest=load_manifest(args.input_dir),
            rule_report=read_report(args.rule_report),
            training_report=read_report(args.training_report),
            holdout_report=read_report(args.holdout_report),
            feature_audit=read_report(args.feature_audit),
            ablation_report=read_report(args.ablation_report),
        )
        write_report(report, args.output)
    except (AcceptanceError, OSError, ValueError) as error:
        print(f"release acceptance failed to run: {error}")
        return 2

    for gate in report["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {gate['name']}")
    print(f"Acceptance report written to {args.output}")
    return 0 if report["passed"] else 1
