import numpy as np
from scripts.common import LABELS, SCENARIOS
from scripts.evaluate_feature_ablation import _candidate_evaluation
from scripts.simulation_manifest import SIMULATION_VARIANTS, empty_manifest
from scripts.train_model import TrainingDataset
from scripts.validate_candidate_holdout import build_holdout_validation
from scripts.validate_phase6_acceptance import build_acceptance_report
from sklearn.tree import DecisionTreeClassifier

from edgeops_ai.ml import MODEL_FEATURE_COLUMNS


def test_ablation_slices_include_all_variants_and_empty_missingness_buckets() -> None:
    labels = np.asarray(list(LABELS) * 2, dtype=object)
    rows = len(labels)
    dataset = TrainingDataset(
        features=np.arange(rows * len(MODEL_FEATURE_COLUMNS), dtype=float).reshape(rows, -1),
        labels=labels,
        seeds=np.asarray([42] * len(LABELS) + [43] * len(LABELS)),
        run_ids=np.asarray(["run-42"] * len(LABELS) + ["run-43"] * len(LABELS)),
        input_files=("one.csv", "two.csv"),
        excluded_recovery_rows=2,
    )
    provenance = [
        {
            "variant_id": SIMULATION_VARIANTS[index % 3],
            "unavailable_metric_groups": [] if index % 2 else ["DLQ"],
        }
        for index in range(rows)
    ]

    report = _candidate_evaluation(
        dataset,
        DecisionTreeClassifier(max_depth=2, random_state=7),
        provenance,
    )

    assert set(report["slices"]["by_variant"]) == set(SIMULATION_VARIANTS)
    assert report["slices"]["by_unavailable_group_count"]["3"] == {
        "rows": 0,
        "accuracy": None,
        "macro_f1": None,
    }
    assert all("macro_f1" in fold for fold in report["per_seed"])


def test_acceptance_report_lists_every_gate_and_can_pass() -> None:
    manifest = empty_manifest()
    run_number = 0
    for scenario in SCENARIOS:
        for variant in SIMULATION_VARIANTS:
            run_number += 1
            manifest["runs"][f"run-{run_number}"] = {
                "scenario_id": scenario,
                "variant_id": variant,
            }

    folds = [
        {"held_out_seed": 42, "accuracy": 0.90, "macro_f1": 0.88},
        {"held_out_seed": 43, "accuracy": 0.92, "macro_f1": 0.89},
    ]
    ablation = {
        "feature_sets": {
            "a1": {
                "candidates": {
                    "decision-tree": {"macro_f1": 0.80, "per_seed": folds},
                }
            },
            "full": {
                "candidates": {
                    "decision-tree": {
                        "macro_f1": 0.84,
                        "per_seed": [
                            {"held_out_seed": 42, "accuracy": 0.89, "macro_f1": 0.87},
                            {"held_out_seed": 43, "accuracy": 0.93, "macro_f1": 0.90},
                        ],
                    }
                }
            },
        }
    }
    recovery = {"runs": [{"cleared_during_observation": True}]}
    holdout = {
        "primary_stable_phase_metrics": {
            "accuracy": 0.98,
            "per_label": {label: {"recall": 0.95} for label in LABELS},
        },
        "recovery": recovery,
    }

    report = build_acceptance_report(
        manifest=manifest,
        rule_report={"recovery": recovery},
        training_report={"quality_gate": {"passed": True}},
        holdout_report=holdout,
        feature_audit={
            "dataset": {"stable_model_matrix_is_finite": True},
            "optional_group_availability": {"overall": {"available_ratio": 0.85}},
            "constant_feature_counts_by_scenario": dict.fromkeys(SCENARIOS, 12),
        },
        ablation_report=ablation,
    )

    assert report["passed"] is True
    assert len(report["gates"]) == 9


def test_acceptance_detects_universally_perfect_folds() -> None:
    perfect = [{"held_out_seed": 42, "accuracy": 1.0, "macro_f1": 1.0}]
    report = build_acceptance_report(
        manifest={"runs": {}},
        rule_report={"recovery": {"runs": []}},
        training_report={"quality_gate": {"passed": False}},
        holdout_report={},
        feature_audit={
            "dataset": {"stable_model_matrix_is_finite": False},
            "optional_group_availability": {"overall": {"available_ratio": 1.0}},
            "constant_feature_counts_by_scenario": {},
        },
        ablation_report={
            "feature_sets": {
                "a1": {"candidates": {"tree": {"macro_f1": 1.0, "per_seed": perfect}}},
                "full": {"candidates": {"tree": {"macro_f1": 1.0, "per_seed": perfect}}},
            }
        },
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "no-universally-perfect-feature-set-candidate"
    )
    assert gate["passed"] is False
    assert gate["observed"] == ["a1/tree", "full/tree"]


def test_candidate_holdout_validation_passes_unseen_balanced_evidence() -> None:
    manifest = empty_manifest()
    for scenario in SCENARIOS:
        for index, variant in enumerate(SIMULATION_VARIANTS):
            manifest["runs"][f"{scenario}-{variant}"] = {
                "scenario_id": scenario,
                "seed": 62 + index,
                "variant_id": variant,
            }
    evaluation = {
        "primary_stable_phase_metrics": {
            "accuracy": 0.98,
            "macro_f1": 0.97,
            "per_label": {label: {"recall": 0.96} for label in LABELS},
        },
        "detection_delay": {"runs": [{"detected": True}] * 12},
        "recovery": {
            "runs": [{"cleared_during_observation": True}] * 12,
            "max_sampled_clearance_delay_seconds": 2.0,
        },
    }

    report = build_holdout_validation(
        manifest=manifest,
        evaluation_report=evaluation,
        model_versions={"lightweight-ml-0.3.0"},
        stable_matrix_finite=True,
        training_seeds=set(range(42, 62)),
    )

    assert report["passed"] is True
    assert len(report["gates"]) == 9


def test_candidate_holdout_validation_rejects_training_seed_overlap() -> None:
    report = build_holdout_validation(
        manifest={
            "simulation_contract_version": 2,
            "dataset_schema_version": "2.0",
            "runs": {
                "overlap": {
                    "scenario_id": "ingestion-backpressure",
                    "seed": 61,
                    "variant_id": "mild",
                }
            },
        },
        evaluation_report={},
        model_versions=set(),
        stable_matrix_finite=False,
        training_seeds=set(range(42, 62)),
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "holdout-seeds-unseen")
    assert gate["passed"] is False
    assert gate["observed"]["overlap"] == [61]
