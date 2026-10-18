"""Train and evaluate the lightweight EdgeOps fault classifier offline."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from edgeops_ai.ml import (
    ARTIFACT_SCHEMA_VERSION,
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
)
from edgeops_ai.training.common import (
    DATASET_SCHEMA_VERSION,
    LABELS,
    add_input_dir,
    default_artifact_path,
)
from edgeops_ai.training.dataset import DatasetError, TrainingDataset, load_training_dataset
from edgeops_ai.training.metrics import classification_metrics
from edgeops_ai.training.reports import write_report

FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS

# Adding a new sklearn-compatible estimator is a single entry here.
# Candidates must be deterministic and are cloned before every fit.
CANDIDATE_MODELS: dict[str, Any] = {
    "logistic-regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "decision-tree": DecisionTreeClassifier(
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    ),
}


def cross_validated_predictions(
    dataset: TrainingDataset,
    candidate: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return leave-one-seed-out predictions and fold metrics."""

    predicted = np.empty(dataset.labels.shape, dtype=object)
    per_seed: list[dict[str, Any]] = []

    splitter = LeaveOneGroupOut()

    for train_indices, test_indices in splitter.split(
        dataset.features,
        dataset.labels,
        groups=dataset.seeds,
    ):
        train_seeds = sorted({int(value) for value in dataset.seeds[train_indices]})
        test_seeds = sorted({int(value) for value in dataset.seeds[test_indices]})

        if set(train_seeds) & set(test_seeds):
            raise RuntimeError("Grouped split leaked a simulation seed into train and test")

        classifier = clone(candidate)
        classifier.fit(
            dataset.features[train_indices],
            dataset.labels[train_indices],
        )

        fold_predictions = classifier.predict(dataset.features[test_indices])

        predicted[test_indices] = fold_predictions

        fold_metrics = metrics_for_predictions(dataset.labels[test_indices], fold_predictions)
        per_seed.append(
            {
                "held_out_seed": test_seeds[0],
                "training_seeds": train_seeds,
                "rows": int(test_indices.size),
                "accuracy": fold_metrics["accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
            }
        )

    return predicted, sorted(per_seed, key=lambda item: item["held_out_seed"])


def metrics_for_predictions(
    expected: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """Calculate the shared multiclass metrics for existing predictions."""

    metrics = classification_metrics(expected, predicted, include_unsupported=True)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": [
            [metrics["confusion_matrix"][expected][predicted] for predicted in LABELS]
            for expected in LABELS
        ],
        "per_label": metrics["per_label"],
    }


def evaluate_candidate(
    dataset: TrainingDataset,
    candidate: Any,
) -> dict[str, Any]:
    """Evaluate one candidate using leave-one-seed-out folds."""

    predicted, per_seed = cross_validated_predictions(dataset, candidate)

    return {
        "strategy": "leave-one-simulation-seed-out",
        **metrics_for_predictions(dataset.labels, predicted),
        "per_seed": per_seed,
    }


def evaluate_candidates(
    dataset: TrainingDataset,
) -> tuple[str, dict[str, dict[str, Any]]]:
    results = {
        name: evaluate_candidate(dataset, candidate) for name, candidate in CANDIDATE_MODELS.items()
    }

    # Accuracy is the primary selection criterion.
    # Macro-F1 breaks metric ties.
    # The shallow tree wins exact ties because its decisions are easier
    # to inspect and compare to the existing rule baseline.
    selected_name = max(
        results,
        key=lambda name: (
            results[name]["accuracy"],
            results[name]["macro_f1"],
            name == "decision-tree",
        ),
    )

    return selected_name, results


def train_model(
    *,
    input_dir: Path,
    artifact_path: Path,
    report_path: Path,
    model_version: str | None = None,
    min_accuracy: float = 0.95,
    min_label_recall: float = 0.90,
) -> dict[str, Any]:
    if model_version is not None and not model_version.strip():
        raise DatasetError("The model version must be a non-empty string.")
    model_version = model_version or datetime.now(UTC).strftime("lightweight-ml-%Y%m%dT%H%M%S.%fZ")

    dataset = load_training_dataset(input_dir)

    selected_model, candidates = evaluate_candidates(dataset)
    cross_validation = candidates[selected_model]

    per_label_recall = {
        label: float(values["recall"]) for label, values in cross_validation["per_label"].items()
    }

    failed_recall_labels = [
        label for label, recall in per_label_recall.items() if recall < min_label_recall
    ]

    quality_gate_passed = cross_validation["accuracy"] >= min_accuracy and not failed_recall_labels

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "model_version": model_version,
        "artifact_path": str(artifact_path),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_configuration": {name: repr(model) for name, model in CANDIDATE_MODELS.items()},
        "dataset": {
            "input_files": list(dataset.input_files),
            "stable_rows": dataset.row_count,
            "excluded_recovery_rows": (dataset.excluded_recovery_rows),
            "simulation_runs": len(set(dataset.run_ids.tolist())),
            "simulation_seeds": sorted({int(value) for value in dataset.seeds}),
            "class_distribution": dict(sorted(Counter(dataset.labels.tolist()).items())),
            "feature_columns": list(FEATURE_COLUMNS),
        },
        "candidates": candidates,
        "selected_model": selected_model,
        "cross_validation": cross_validation,
        "quality_gate": {
            "min_accuracy": min_accuracy,
            "min_label_recall": min_label_recall,
            "failed_recall_labels": failed_recall_labels,
            "passed": quality_gate_passed,
        },
    }

    write_report(report, report_path)

    if not quality_gate_passed:
        return report

    classifier = clone(CANDIDATE_MODELS[selected_model])
    classifier.fit(dataset.features, dataset.labels)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "model_version": model_version,
        "model_name": selected_model,
        "trained_at": report["generated_at"],
        "feature_columns": list(FEATURE_COLUMNS),
        "labels": list(LABELS),
        "training_rows": dataset.row_count,
        "simulation_seeds": (report["dataset"]["simulation_seeds"]),
        "model": classifier,
    }

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(artifact, artifact_path)
    manifest = {key: value for key, value in artifact.items() if key != "model"}
    manifest.update(
        artifact=artifact_path.name, sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )
    write_report(manifest, artifact_path.parent / "manifest.json")

    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    add_input_dir(parser)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=default_artifact_path(),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-version",
        help="Artifact model version (default: a generated UTC timestamp version).",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--min-label-recall",
        type=float,
        default=0.90,
    )

    args = parser.parse_args(argv)

    for name in (
        "min_accuracy",
        "min_label_recall",
    ):
        value = getattr(args, name)

        if not 0.0 <= value <= 1.0:
            argument_name = name.replace("_", "-")
            parser.error(f"--{argument_name} must be between 0 and 1")

    args.report = args.report or args.input_dir / "training-report.json"
    return args


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    try:
        report = train_model(
            input_dir=args.input_dir,
            artifact_path=args.artifact,
            report_path=args.report,
            model_version=args.model_version,
            min_accuracy=args.min_accuracy,
            min_label_recall=args.min_label_recall,
        )
    except DatasetError as error:
        print(f"Training dataset validation failed: {error}")
        return 2

    cross_validation = report["cross_validation"]

    print(f"Selected candidate: {report['selected_model']}")

    for candidate_name, metrics in report["candidates"].items():
        print(
            f"{candidate_name:32} "
            f"accuracy={metrics['accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )

    print(
        "Selected leave-one-seed-out accuracy: "
        f"{cross_validation['accuracy']:.3f} "
        f"({report['dataset']['stable_rows']} stable rows)"
    )

    for label in LABELS:
        metrics = cross_validation["per_label"][label]

        print(
            f"{label:32} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"support={metrics['support']}"
        )

    if not report["quality_gate"]["passed"]:
        print(f"Quality gate failed; report written to {args.report}; artifact not updated.")
        return 1

    print(f"Quality gate passed; model written to {args.artifact}")
    print(f"Training report written to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
