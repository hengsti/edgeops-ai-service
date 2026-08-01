"""Train and evaluate the lightweight EdgeOps fault classifier offline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from edgeops_ai.ml import (
    ARTIFACT_SCHEMA_VERSION,
    ML_MODEL_VERSION,
    MODEL_FEATURE_COLUMNS,
    build_model_feature_row,
)
from scripts.common import (
    DATASET_SCHEMA_VERSION,
    LABELS,
    RECOVERY_PHASES,
    SCENARIOS,
    normalize_label_value,
)
from scripts.common import (
    FEATURE_COLUMNS as DATASET_FEATURE_COLUMNS,
)

FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS

SOURCE_FEATURE_COLUMNS = (
    "messages_enqueued_rate",
    "queue_growth_rate",
    "processing_ratio",
    "queue_full_rate",
    "transform_failure_ratio",
    "persistence_ratio",
    "heartbeat_missing",
    "queue_utilization_ratio",
    "influx_write_failure_ratio",
    "influxdb_healthy",
    "pipeline_duration_average_seconds",
    "heartbeat_age_max_seconds",
    "missing_device_ratio",
    "device_metrics_available",
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

METADATA_COLUMNS = (
    "dataset_schema_version",
    "captured_at",
    "source_host",
    "scenario_id",
    "simulation_run_id",
    "simulation_seed",
    "simulation_phase",
    "expected_label",
)

DEFAULT_MODEL_VERSION = ML_MODEL_VERSION


class DatasetError(ValueError):
    """Raised when generated training data violates the CSV contract."""


@dataclass(frozen=True)
class TrainingDataset:
    features: np.ndarray
    labels: np.ndarray
    seeds: np.ndarray
    run_ids: np.ndarray
    input_files: tuple[str, ...]
    excluded_recovery_rows: int

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])


def _parse_bool(
    value: str | None,
    *,
    column: str,
    path: Path,
    line_number: int,
) -> float:
    normalized = (value or "").strip().lower()

    if normalized in {"true", "1"}:
        return 1.0

    if normalized in {"false", "0"}:
        return 0.0

    raise DatasetError(
        f"{path}:{line_number}: column {column!r} must be true/false or 1/0, got {value!r}"
    )


def _parse_float(
    value: str | None,
    *,
    column: str,
    path: Path,
    line_number: int,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(
            f"{path}:{line_number}: column {column!r} is not numeric: {value!r}"
        ) from error

    if not math.isfinite(parsed):
        raise DatasetError(f"{path}:{line_number}: column {column!r} must be finite, got {value!r}")

    return parsed


def _parse_optional_float(
    value: str | None,
    *,
    column: str,
    path: Path,
    line_number: int,
) -> float | None:
    if value is None or not value.strip():
        return None

    return _parse_float(
        value,
        column=column,
        path=path,
        line_number=line_number,
    )


def _parse_seed(
    value: str | None,
    *,
    path: Path,
    line_number: int,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(
            f"{path}:{line_number}: simulation_seed is not an integer: {value!r}"
        ) from error


def _build_model_row(
    row: Mapping[str, str | None],
    *,
    path: Path,
    line_number: int,
) -> list[float]:
    def required_float(column: str) -> float:
        return _parse_float(
            row.get(column),
            column=column,
            path=path,
            line_number=line_number,
        )

    def optional_float(column: str) -> float | None:
        return _parse_optional_float(
            row.get(column),
            column=column,
            path=path,
            line_number=line_number,
        )

    def required_bool(column: str) -> float:
        return _parse_bool(
            row.get(column),
            column=column,
            path=path,
            line_number=line_number,
        )

    return build_model_feature_row(
        messages_enqueued_rate=required_float("messages_enqueued_rate"),
        queue_growth_rate=required_float("queue_growth_rate"),
        processing_ratio=required_float("processing_ratio"),
        queue_full_rate=required_float("queue_full_rate"),
        transform_failure_ratio=required_float("transform_failure_ratio"),
        persistence_ratio=required_float("persistence_ratio"),
        heartbeat_missing=required_bool("heartbeat_missing"),
        queue_utilization_ratio=optional_float("queue_utilization_ratio"),
        influx_write_failure_ratio=optional_float("influx_write_failure_ratio"),
        influxdb_healthy=required_float("influxdb_healthy"),
        pipeline_duration_average_seconds=required_float("pipeline_duration_average_seconds"),
        heartbeat_age_max_seconds=optional_float("heartbeat_age_max_seconds"),
        missing_device_ratio=required_float("missing_device_ratio"),
        device_metrics_available=required_bool("device_metrics_available"),
        decode_failure_ratio=optional_float("decode_failure_ratio"),
        raw_validation_failure_ratio=optional_float("raw_validation_failure_ratio"),
        business_validation_failure_ratio=optional_float("business_validation_failure_ratio"),
        transform_deserialize_failure_ratio=optional_float("transform_deserialize_failure_ratio"),
        event_queue_drop_ratio=optional_float("event_queue_drop_ratio"),
        wal_queue_full_ratio=optional_float("wal_queue_full_ratio"),
        durability_ack_failure_rate=optional_float("durability_ack_failure_rate"),
        dlq_publish_failure_ratio=optional_float("dlq_publish_failure_ratio"),
        wal_retry_rate=optional_float("wal_retry_rate"),
        wal_drop_ratio=optional_float("wal_drop_ratio"),
        wal_retry_outage_active=optional_float("wal_retry_outage_active"),
        decode_duration_average_seconds=optional_float("decode_duration_average_seconds"),
        transform_duration_average_seconds=optional_float("transform_duration_average_seconds"),
        persist_duration_average_seconds=optional_float("persist_duration_average_seconds"),
        influx_write_duration_average_seconds=optional_float(
            "influx_write_duration_average_seconds"
        ),
    )


def load_training_dataset(input_dir: Path) -> TrainingDataset:
    """Load stable-phase rows from generated scenario/seed CSV files."""

    paths = tuple(sorted(input_dir.glob("*-seed-*.csv")))

    if not paths:
        raise DatasetError(f"No *-seed-*.csv files found in {input_dir}")

    required_columns = (
        set(METADATA_COLUMNS) | set(SOURCE_FEATURE_COLUMNS) | set(DATASET_FEATURE_COLUMNS)
    )

    feature_rows: list[list[float]] = []
    labels: list[str] = []
    seeds: list[int] = []
    run_ids: list[str] = []
    excluded_recovery_rows = 0

    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            actual_columns = set(reader.fieldnames or ())
            missing_columns = sorted(required_columns - actual_columns)

            if missing_columns:
                raise DatasetError(
                    f"{path}: missing required columns: {', '.join(missing_columns)}"
                )

            file_row_count = 0

            for line_number, row in enumerate(reader, start=2):
                file_row_count += 1

                dataset_version = (row.get("dataset_schema_version") or "").strip()

                if dataset_version != DATASET_SCHEMA_VERSION:
                    raise DatasetError(
                        f"{path}:{line_number}: unsupported dataset_schema_version "
                        f"{dataset_version!r}; expected {DATASET_SCHEMA_VERSION!r}"
                    )

                captured_at = (row.get("captured_at") or "").strip()
                source_host = (row.get("source_host") or "").strip()
                scenario = normalize_label_value(row.get("scenario_id") or "")
                run_id = (row.get("simulation_run_id") or "").strip()
                phase = normalize_label_value(row.get("simulation_phase") or "")
                label = normalize_label_value(row.get("expected_label") or "")

                if not captured_at:
                    raise DatasetError(f"{path}:{line_number}: captured_at is empty")

                if not source_host:
                    raise DatasetError(f"{path}:{line_number}: source_host is empty")

                if scenario not in SCENARIOS:
                    raise DatasetError(
                        f"{path}:{line_number}: unsupported scenario_id {scenario!r}"
                    )

                if not run_id:
                    raise DatasetError(f"{path}:{line_number}: simulation_run_id is empty")

                if not phase:
                    raise DatasetError(f"{path}:{line_number}: simulation_phase is empty")

                if label not in LABELS:
                    raise DatasetError(f"{path}:{line_number}: unknown expected_label {label!r}")

                seed = _parse_seed(
                    row.get("simulation_seed"),
                    path=path,
                    line_number=line_number,
                )

                if phase in RECOVERY_PHASES:
                    excluded_recovery_rows += 1
                    continue

                parsed_features = _build_model_row(
                    row,
                    path=path,
                    line_number=line_number,
                )

                feature_rows.append(parsed_features)
                labels.append(label)
                seeds.append(seed)
                run_ids.append(run_id)

            if file_row_count == 0:
                raise DatasetError(f"{path}: dataset file is empty")

    if not feature_rows:
        raise DatasetError("No stable-phase rows remain after excluding recovery rows")

    missing_labels = sorted(set(LABELS) - set(labels))

    if missing_labels:
        raise DatasetError(f"Training data is missing labels: {', '.join(missing_labels)}")

    unique_seeds = set(seeds)

    if len(unique_seeds) < 2:
        raise DatasetError("At least two simulation seeds are required for grouped evaluation")

    return TrainingDataset(
        features=np.asarray(feature_rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=object),
        seeds=np.asarray(seeds, dtype=np.int64),
        run_ids=np.asarray(run_ids, dtype=object),
        input_files=tuple(path.name for path in paths),
        excluded_recovery_rows=excluded_recovery_rows,
    )


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
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    ),
}


def _classification_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(LABELS),
        zero_division=0,
    )

    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }


def evaluate_candidate(
    dataset: TrainingDataset,
    candidate: Any,
) -> dict[str, Any]:
    """Evaluate one candidate using leave-one-seed-out folds."""

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

        per_seed.append(
            {
                "held_out_seed": test_seeds[0],
                "training_seeds": train_seeds,
                "rows": int(test_indices.size),
                "accuracy": float(
                    accuracy_score(
                        dataset.labels[test_indices],
                        fold_predictions,
                    )
                ),
            }
        )

    per_label = _classification_metrics(
        dataset.labels,
        predicted,
    )

    macro_f1 = sum(values["f1"] for values in per_label.values()) / len(LABELS)

    return {
        "strategy": "leave-one-simulation-seed-out",
        "accuracy": float(accuracy_score(dataset.labels, predicted)),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(
            dataset.labels,
            predicted,
            labels=list(LABELS),
        ).tolist(),
        "per_label": per_label,
        "per_seed": sorted(
            per_seed,
            key=lambda item: item["held_out_seed"],
        ),
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
    model_version: str = DEFAULT_MODEL_VERSION,
    min_accuracy: float = 0.95,
    min_label_recall: float = 0.90,
) -> dict[str, Any]:
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

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    if not quality_gate_passed:
        return report

    classifier = clone(CANDIDATE_MODELS[selected_model])
    classifier.fit(dataset.features, dataset.labels)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
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

    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/generated"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/model-candidate.joblib"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/generated/training-report.json"),
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
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
