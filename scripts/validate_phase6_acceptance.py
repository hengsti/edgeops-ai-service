"""Validate every realistic Phase 6 acceptance gate and emit JSON evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.common import SCENARIOS
from scripts.simulation_manifest import SIMULATION_VARIANTS, load_manifest

ACCEPTANCE_SCHEMA_VERSION = "1.0"


class AcceptanceError(ValueError):
    """Raised when acceptance inputs are missing or malformed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"Unable to read JSON report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AcceptanceError(f"JSON report must contain an object: {path}")
    return payload


def _gate(
    name: str,
    *,
    observed: Any,
    threshold: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "passed": bool(passed),
    }


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
            passed=holdout_ok,
        )
    )

    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "simulation_contract_version": manifest.get("simulation_contract_version"),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_acceptance_report(
            manifest=load_manifest(args.input_dir),
            rule_report=_read_json(args.rule_report),
            training_report=_read_json(args.training_report),
            holdout_report=_read_json(args.holdout_report),
            feature_audit=_read_json(args.feature_audit),
            ablation_report=_read_json(args.ablation_report),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (AcceptanceError, OSError, ValueError) as error:
        print(f"Phase 6 acceptance failed to run: {error}")
        return 2

    for gate in report["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {gate['name']}")
    print(f"Acceptance report written to {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
