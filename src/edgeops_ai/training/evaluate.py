"""Validate and evaluate collected rule-baseline datasets offline.

The primary metrics exclude recovery observations because recovery contains
expected transitional effects. Recovery behavior is retained and reported
separately, while all-phase metrics remain available for completeness.

Example:
    uv run edgeops-train evaluate \
        --input-dir data/generated \
        --output data/generated/rule-report.json
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from edgeops_ai.training.common import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    DEFAULT_SEEDS,
    LABELS,
    RECOVERY_PHASES,
    SCENARIOS,
    add_input_dir,
    normalize_label_value,
)
from edgeops_ai.training.metrics import classification_metrics as prediction_metrics
from edgeops_ai.training.reports import write_report


@dataclass(frozen=True)
class DatasetRow:
    source_file: str
    source_row: int
    captured_at: datetime
    scenario_id: str
    simulation_run_id: str
    simulation_seed: int
    simulation_phase: str
    expected_label: str
    predicted_label: str
    prediction_correct: bool

    @property
    def is_recovery(self) -> bool:
        phase = normalize_label_value(self.simulation_phase)
        return phase in RECOVERY_PHASES


def parse_timestamp(
    value: str,
    location: str,
) -> datetime:
    normalized = value.strip().replace(
        "Z",
        "+00:00",
    )

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{location}: invalid captured_at value {value!r}.") from error

    if parsed.tzinfo is None:
        raise ValueError(f"{location}: captured_at must include a timezone.")

    return parsed


def parse_boolean(
    value: str,
    location: str,
) -> bool:
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"{location}: prediction_correct must be true or false.")


def read_dataset(
    path: Path,
) -> list[DatasetRow]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as dataset_file:
        reader = csv.DictReader(dataset_file)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is missing.")

        missing_columns = sorted(set(DATASET_COLUMNS) - set(reader.fieldnames))

        if missing_columns:
            raise ValueError(f"{path}: missing columns: {', '.join(missing_columns)}.")

        rows: list[DatasetRow] = []

        for source_row, raw in enumerate(
            reader,
            start=2,
        ):
            location = f"{path}:{source_row}"

            dataset_version = (raw.get("dataset_schema_version") or "").strip()

            if dataset_version != DATASET_SCHEMA_VERSION:
                raise ValueError(
                    f"{location}: unsupported dataset_schema_version "
                    f"{dataset_version!r}; expected {DATASET_SCHEMA_VERSION!r}."
                )

            scenario_id = normalize_label_value(raw["scenario_id"])

            run_id = raw["simulation_run_id"].strip()

            phase = raw["simulation_phase"].strip()

            expected = normalize_label_value(raw["expected_label"])

            predicted = normalize_label_value(raw["predicted_label"])

            if scenario_id not in SCENARIOS:
                raise ValueError(f"{location}: unsupported scenario_id {scenario_id!r}.")

            if not run_id:
                raise ValueError(f"{location}: simulation_run_id is empty.")

            if not phase:
                raise ValueError(f"{location}: simulation_phase is empty.")

            if expected not in LABELS:
                raise ValueError(f"{location}: unsupported expected_label {expected!r}.")

            if predicted not in LABELS:
                raise ValueError(f"{location}: unsupported predicted_label {predicted!r}.")

            if expected not in {
                "normal",
                scenario_id,
            }:
                raise ValueError(
                    f"{location}: expected_label "
                    f"{expected!r} does not match "
                    f"scenario {scenario_id!r}."
                )

            try:
                seed = int(raw["simulation_seed"])
            except ValueError as error:
                raise ValueError(
                    f"{location}: invalid simulation_seed {raw['simulation_seed']!r}."
                ) from error

            prediction_correct = parse_boolean(
                raw["prediction_correct"],
                location,
            )

            labels_equal = expected == predicted

            if prediction_correct != labels_equal:
                raise ValueError(f"{location}: prediction_correct conflicts with the two labels.")

            rows.append(
                DatasetRow(
                    source_file=path.name,
                    source_row=source_row,
                    captured_at=parse_timestamp(
                        raw["captured_at"],
                        location,
                    ),
                    scenario_id=scenario_id,
                    simulation_run_id=run_id,
                    simulation_seed=seed,
                    simulation_phase=phase,
                    expected_label=expected,
                    predicted_label=predicted,
                    prediction_correct=(prediction_correct),
                )
            )

    if not rows:
        raise ValueError(f"{path}: dataset contains no rows.")

    return rows


def validate_run_file(
    rows: Sequence[DatasetRow],
) -> tuple[str, int, str]:
    source_file = rows[0].source_file

    scenarios = {row.scenario_id for row in rows}

    seeds = {row.simulation_seed for row in rows}

    run_ids = {row.simulation_run_id for row in rows}

    if len(scenarios) != 1:
        raise ValueError(f"{source_file}: expected exactly one scenario_id.")

    if len(seeds) != 1:
        raise ValueError(f"{source_file}: expected exactly one simulation_seed.")

    if len(run_ids) != 1:
        raise ValueError(f"{source_file}: expected exactly one simulation_run_id.")

    scenario = next(iter(scenarios))
    seed = next(iter(seeds))
    run_id = next(iter(run_ids))

    expected_labels = {row.expected_label for row in rows}

    if "normal" not in expected_labels:
        raise ValueError(f"{source_file}: normal observations are missing.")

    if scenario not in expected_labels:
        raise ValueError(f"{source_file}: fault observations for {scenario!r} are missing.")

    if not any(row.is_recovery for row in rows):
        raise ValueError(f"{source_file}: recovery observations are missing.")

    return scenario, seed, run_id


def load_datasets(
    paths: Sequence[Path],
    expected_scenarios: Sequence[str] = (SCENARIOS),
    expected_seeds: Sequence[int] = (DEFAULT_SEEDS),
) -> tuple[
    list[DatasetRow],
    list[dict[str, Any]],
]:
    if not paths:
        raise ValueError("No dataset files matched the input pattern.")

    all_rows: list[DatasetRow] = []
    runs: list[dict[str, Any]] = []

    observed_pairs: dict[
        tuple[str, int],
        str,
    ] = {}

    for path in sorted(paths):
        rows = read_dataset(path)

        scenario, seed, run_id = validate_run_file(rows)

        pair = (scenario, seed)

        if pair in observed_pairs:
            raise ValueError(
                "Duplicate run for scenario "
                f"{scenario!r}, seed {seed}: "
                f"{observed_pairs[pair]} and "
                f"{path.name}."
            )

        observed_pairs[pair] = path.name
        all_rows.extend(rows)

        runs.append(
            {
                "source_file": path.name,
                "scenario_id": scenario,
                "simulation_seed": seed,
                "simulation_run_id": (run_id),
                "rows": len(rows),
            }
        )

    required_pairs = {
        (
            normalize_label_value(scenario),
            seed,
        )
        for scenario in expected_scenarios
        for seed in expected_seeds
    }

    observed = set(observed_pairs)

    missing = sorted(required_pairs - observed)

    unexpected = sorted(observed - required_pairs)

    if missing:
        formatted = ", ".join(f"{scenario}/seed-{seed}" for scenario, seed in missing)

        raise ValueError(f"Required scenario runs are missing: {formatted}.")

    if unexpected:
        formatted = ", ".join(f"{scenario}/seed-{seed}" for scenario, seed in unexpected)

        raise ValueError(f"Unexpected scenario runs were selected: {formatted}.")

    sorted_runs = sorted(
        runs,
        key=lambda run: (
            run["scenario_id"],
            run["simulation_seed"],
        ),
    )

    return all_rows, sorted_runs


def classification_metrics(
    rows: Iterable[DatasetRow],
) -> dict[str, Any]:
    materialized = list(rows)
    return prediction_metrics(
        (row.expected_label for row in materialized),
        (row.predicted_label for row in materialized),
    )


def grouped_metrics(
    rows: Sequence[DatasetRow],
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[
        str,
        list[DatasetRow],
    ] = defaultdict(list)

    for row in rows:
        value = getattr(row, key)
        groups[str(value)].append(row)

    return {group: classification_metrics(groups[group]) for group in sorted(groups)}


def detection_delay(
    rows: Sequence[DatasetRow],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, int, str],
        list[DatasetRow],
    ] = defaultdict(list)

    for row in rows:
        key = (
            row.scenario_id,
            row.simulation_seed,
            row.simulation_run_id,
        )

        grouped[key].append(row)

    results: list[dict[str, Any]] = []

    for (
        scenario,
        seed,
        run_id,
    ), run_rows in sorted(grouped.items()):
        fault_rows = sorted(
            (row for row in run_rows if row.expected_label == scenario),
            key=lambda row: row.captured_at,
        )

        start = fault_rows[0].captured_at

        detected = next(
            (row for row in fault_rows if row.predicted_label == scenario),
            None,
        )

        results.append(
            {
                "scenario_id": scenario,
                "simulation_seed": seed,
                "simulation_run_id": (run_id),
                "fault_phase_first_sample_at": (start.isoformat()),
                "first_detection_at": (detected.captured_at.isoformat() if detected else None),
                "sampled_detection_delay_seconds": (
                    (detected.captured_at - start).total_seconds() if detected else None
                ),
                "detected": (detected is not None),
            }
        )

    return results


def recovery_analysis(
    rows: Sequence[DatasetRow],
) -> dict[str, Any]:
    grouped: dict[
        tuple[str, int, str],
        list[DatasetRow],
    ] = defaultdict(list)

    for row in rows:
        if row.is_recovery:
            key = (
                row.scenario_id,
                row.simulation_seed,
                row.simulation_run_id,
            )

            grouped[key].append(row)

    runs: list[dict[str, Any]] = []

    for (
        scenario,
        seed,
        run_id,
    ), recovery_rows in sorted(grouped.items()):
        ordered = sorted(
            recovery_rows,
            key=lambda row: row.captured_at,
        )

        alert_rows = [row for row in ordered if row.predicted_label != "normal"]

        leading_alert_rows = 0

        for row in ordered:
            if row.predicted_label == "normal":
                break

            leading_alert_rows += 1

        first_normal = next(
            (row for row in ordered if row.predicted_label == "normal"),
            None,
        )

        last_alert = alert_rows[-1] if alert_rows else None

        runs.append(
            {
                "scenario_id": scenario,
                "simulation_seed": seed,
                "simulation_run_id": (run_id),
                "recovery_rows": (len(ordered)),
                "alert_rows": (len(alert_rows)),
                "alert_rate": (len(alert_rows) / len(ordered)),
                "leading_alert_rows": (leading_alert_rows),
                "cleared_during_observation": (first_normal is not None),
                "sampled_clearance_delay_seconds": (
                    (first_normal.captured_at - ordered[0].captured_at).total_seconds()
                    if first_normal
                    else None
                ),
                "last_alert_offset_seconds": (
                    (last_alert.captured_at - ordered[0].captured_at).total_seconds()
                    if last_alert
                    else None
                ),
            }
        )

    clearance_delays = [
        run["sampled_clearance_delay_seconds"]
        for run in runs
        if run["sampled_clearance_delay_seconds"] is not None
    ]

    return {
        "rows": sum(run["recovery_rows"] for run in runs),
        "alert_rows": sum(run["alert_rows"] for run in runs),
        "runs_with_alerts": sum(run["alert_rows"] > 0 for run in runs),
        "runs": runs,
        "mean_sampled_clearance_delay_seconds": (
            sum(clearance_delays) / len(clearance_delays) if clearance_delays else None
        ),
        "max_sampled_clearance_delay_seconds": (
            max(
                clearance_delays,
                default=None,
            )
        ),
    }


def build_report(
    rows: Sequence[DatasetRow],
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    stable_rows = [row for row in rows if not row.is_recovery]

    detection = detection_delay(rows)

    delays = [
        item["sampled_detection_delay_seconds"]
        for item in detection
        if item["sampled_detection_delay_seconds"] is not None
    ]

    return {
        "schema_version": "1.0",
        "dataset": {
            "files": len(runs),
            "runs": len(runs),
            "rows": len(rows),
            "scenarios": sorted({row.scenario_id for row in rows}),
            "seeds": sorted({row.simulation_seed for row in rows}),
            "validated_runs": list(runs),
        },
        "primary_stable_phase_metrics": (classification_metrics(stable_rows)),
        "all_phase_metrics": (classification_metrics(rows)),
        "stable_metrics_by_scenario": (
            grouped_metrics(
                stable_rows,
                "scenario_id",
            )
        ),
        "stable_metrics_by_seed": (
            grouped_metrics(
                stable_rows,
                "simulation_seed",
            )
        ),
        "detection_delay": {
            "runs": detection,
            "detected_runs": sum(item["detected"] for item in detection),
            "mean_seconds": (sum(delays) / len(delays) if delays else None),
            "max_seconds": max(
                delays,
                default=None,
            ),
        },
        "recovery": (recovery_analysis(rows)),
    }


def format_seconds(
    value: float | None,
) -> str:
    if value is None or math.isnan(value):
        return "n/a"

    return f"{value:.3f}"


def print_report(
    report: dict[str, Any],
    output: Path,
) -> None:
    dataset = report["dataset"]

    stable = report["primary_stable_phase_metrics"]

    all_phases = report["all_phase_metrics"]

    recovery = report["recovery"]

    detection = report["detection_delay"]

    print(f"Validated datasets: {dataset['files']} files, {dataset['rows']} rows")

    print(
        "Primary stable-phase accuracy: "
        f"{stable['accuracy']:.3f} "
        f"({stable['correct']}/"
        f"{stable['rows']})"
    )

    print(
        "All-phase accuracy: "
        f"{all_phases['accuracy']:.3f} "
        f"({all_phases['correct']}/"
        f"{all_phases['rows']})"
    )

    print("\nStable-phase per-label metrics:")

    print(f"{'label':31} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}")

    for label, metrics in stable["per_label"].items():
        if metrics["support"] == 0:
            continue

        print(
            f"{label:31} "
            f"{metrics['precision']:9.3f} "
            f"{metrics['recall']:9.3f} "
            f"{metrics['f1']:9.3f} "
            f"{metrics['support']:9d}"
        )

    print("\nStable-phase accuracy by scenario:")

    for scenario, metrics in report["stable_metrics_by_scenario"].items():
        print(f"{scenario:31} {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['rows']})")

    print(
        "\nDetection delay: "
        f"{detection['detected_runs']}/"
        f"{dataset['runs']} runs detected, "
        "mean="
        f"{format_seconds(detection['mean_seconds'])}s, "
        "max="
        f"{format_seconds(detection['max_seconds'])}s"
    )

    print(
        "Recovery alerts: "
        f"{recovery['alert_rows']}/"
        f"{recovery['rows']} rows across "
        f"{recovery['runs_with_alerts']}/"
        f"{dataset['runs']} runs"
    )

    print(f"Report: {output}")


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Validate and evaluate collected rule-baseline datasets.")
    )

    add_input_dir(parser)

    parser.add_argument(
        "--pattern",
        default="*-seed-*.csv",
        help=("Dataset filename pattern (default: *-seed-*.csv)."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path.",
    )

    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIOS,
        help=("Expected scenario; repeat to evaluate a subset (default: all scenarios)."),
    )

    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help=("Expected seed; repeat to evaluate a subset (default: 42, 43, 44)."),
    )

    args = parser.parse_args(argv)
    args.output = args.output or args.input_dir / "rule-report.json"
    return args


def evaluate_dataset(
    *,
    input_dir: Path,
    output: Path,
    scenarios: Sequence[str] = SCENARIOS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    pattern: str = "*-seed-*.csv",
) -> dict[str, Any]:
    rows, runs = load_datasets(
        list(input_dir.glob(pattern)),
        expected_scenarios=scenarios,
        expected_seeds=seeds,
    )
    report = build_report(rows, runs)
    write_report(report, output)
    return report


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    scenarios = tuple(args.scenario) if args.scenario else SCENARIOS

    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS

    try:
        report = evaluate_dataset(
            input_dir=args.input_dir,
            output=args.output,
            scenarios=scenarios,
            seeds=seeds,
            pattern=args.pattern,
        )

    except (
        OSError,
        ValueError,
    ) as error:
        print(f"Dataset evaluation failed: {error}")

        return 1

    print_report(
        report,
        args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
