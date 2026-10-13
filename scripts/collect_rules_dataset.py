"""Create a labeled feature dataset and evaluate the rule-based baseline.

The script polls the EdgeOps collector, reuses the production feature extractor
and predictor, and writes one flat dataset row for every observation that
produces a feature vector. The first observation of a stream and observations
after a stream change or counter reset are warming-up samples and are skipped.

Examples:
    uv run python -m scripts.collect_rules_dataset --samples 120
    uv run python -m scripts.collect_rules_dataset --scenario-id ingestion-backpressure \
        --simulation-seed 42 --max-samples 200
    uv run python -m scripts.collect_rules_dataset --samples 120 \
        --output data/generated/rules.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from edgeops_ai.collector_client import CollectorClient
from edgeops_ai.features import FeatureExtractor
from edgeops_ai.rules import RULES_MODEL_VERSION, RuleBasedPredictor
from edgeops_ai.schemas import FeatureVector, ObservationV1
from edgeops_ai.settings import Settings
from scripts.common import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    LABELS,
    build_dataset_metadata,
    infer_expected_label,
    non_negative_float,
    normalize_label_value,
    positive_int,
)


@dataclass(frozen=True)
class EvaluationSummary:
    collected_observations: int
    dataset_rows: int
    warming_up_observations: int
    correct_predictions: int
    expected_counts: Counter[str]
    predicted_counts: Counter[str]
    true_positive_counts: Counter[str]

    @property
    def accuracy(self) -> float:
        if self.dataset_rows == 0:
            return 0.0

        return self.correct_predictions / self.dataset_rows


@dataclass(frozen=True)
class CollectionResult:
    rows: list[dict[str, Any]]
    collected_observations: int
    warming_up_observations: int
    stop_reason: str
    simulation_run_id: str | None = None


def build_dataset_row(
    observation: ObservationV1,
    features: FeatureVector,
    expected_label: str,
    predicted_label: str,
) -> dict[str, Any]:
    row = build_dataset_metadata(observation)

    row.update(features.model_dump(mode="json"))

    row.update(
        {
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "prediction_correct": (expected_label == predicted_label),
            "model_version": RULES_MODEL_VERSION,
        }
    )

    return row


def write_dataset(
    rows: Sequence[dict[str, Any]],
    output: Path,
    append: bool,
) -> None:
    if not rows:
        return

    suffix = output.suffix.lower()

    if suffix not in {".csv", ".jsonl"}:
        raise ValueError("Dataset output must use the .csv or .jsonl extension.")

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if suffix == ".csv":
        file_exists_with_content = output.exists() and output.stat().st_size > 0

        if append and file_exists_with_content:
            with output.open(
                encoding="utf-8",
                newline="",
            ) as existing_file:
                existing_header = next(csv.reader(existing_file), None)

            if tuple(existing_header or ()) != DATASET_COLUMNS:
                raise ValueError("Cannot append to a CSV with an incompatible dataset schema.")

        mode = "a" if append else "w"

        with output.open(
            mode,
            encoding="utf-8",
            newline="",
        ) as dataset_file:
            writer = csv.DictWriter(
                dataset_file,
                fieldnames=DATASET_COLUMNS,
            )

            if not append or not file_exists_with_content:
                writer.writeheader()

            writer.writerows(rows)

        return

    file_exists_with_content = output.exists() and output.stat().st_size > 0

    if append and file_exists_with_content:
        with output.open(encoding="utf-8") as existing_file:
            first_line = next(
                (line for line in existing_file if line.strip()),
                "",
            )

        try:
            first_row = json.loads(first_line)
        except json.JSONDecodeError as error:
            raise ValueError("Cannot append to JSONL with invalid existing content.") from error

        if (
            not isinstance(first_row, dict)
            or first_row.get("dataset_schema_version") != DATASET_SCHEMA_VERSION
            or not set(DATASET_COLUMNS).issubset(first_row)
        ):
            raise ValueError("Cannot append to JSONL with an incompatible dataset schema.")

    mode = "a" if append else "w"

    with output.open(
        mode,
        encoding="utf-8",
    ) as dataset_file:
        for row in rows:
            dataset_file.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            dataset_file.write("\n")


def summarize(
    rows: Iterable[dict[str, Any]],
    collected_observations: int,
    warming_up_observations: int,
) -> EvaluationSummary:
    materialized_rows = list(rows)

    expected_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    true_positive_counts: Counter[str] = Counter()

    for row in materialized_rows:
        expected = str(row["expected_label"])
        predicted = str(row["predicted_label"])

        expected_counts[expected] += 1
        predicted_counts[predicted] += 1

        if expected == predicted:
            true_positive_counts[expected] += 1

    return EvaluationSummary(
        collected_observations=collected_observations,
        dataset_rows=len(materialized_rows),
        warming_up_observations=warming_up_observations,
        correct_predictions=sum(true_positive_counts.values()),
        expected_counts=expected_counts,
        predicted_counts=predicted_counts,
        true_positive_counts=true_positive_counts,
    )


def print_summary(
    summary: EvaluationSummary,
    output: Path,
) -> None:
    print(f"Dataset: {output}")
    print(f"Collected observations: {summary.collected_observations}")
    print(f"Dataset rows: {summary.dataset_rows}")
    print(f"Warming-up observations skipped: {summary.warming_up_observations}")
    print(f"Accuracy: {summary.accuracy:.3f}")

    evaluated_labels = sorted(summary.expected_counts.keys() | summary.predicted_counts.keys())

    if not evaluated_labels:
        return

    print("\nPer-label metrics:")

    print(f"{'label':31} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}")

    for label in evaluated_labels:
        true_positives = summary.true_positive_counts[label]
        predicted = summary.predicted_counts[label]
        expected = summary.expected_counts[label]

        precision = true_positives / predicted if predicted else 0.0

        recall = true_positives / expected if expected else 0.0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        print(f"{label:31} {precision:9.3f} {recall:9.3f} {f1:9.3f} {expected:9d}")


def scenario_is_ready(
    observation: ObservationV1,
    scenario_id: str,
    simulation_seed: int | None,
    start_phase: str,
) -> bool:
    metadata = observation.metadata

    return (
        metadata.scenario_id is not None
        and normalize_label_value(metadata.scenario_id) == normalize_label_value(scenario_id)
        and (simulation_seed is None or metadata.simulation_seed == simulation_seed)
        and metadata.simulation_run_id is not None
        and metadata.simulation_phase is not None
        and normalize_label_value(metadata.simulation_phase) == normalize_label_value(start_phase)
    )


def observation_belongs_to_run(
    observation: ObservationV1,
    scenario_id: str,
    simulation_run_id: str,
) -> bool:
    metadata = observation.metadata

    return (
        metadata.scenario_id is not None
        and normalize_label_value(metadata.scenario_id) == normalize_label_value(scenario_id)
        and metadata.simulation_run_id == simulation_run_id
    )


async def collect_dataset(
    settings: Settings,
    max_samples: int,
    interval_seconds: float,
    fault_label: str | None,
    scenario_id: str | None = None,
    simulation_seed: int | None = None,
    start_phase: str = "normal",
    wait_timeout_seconds: float = 300.0,
) -> CollectionResult:
    extractor = FeatureExtractor()
    predictor = RuleBasedPredictor()

    rows: list[dict[str, Any]] = []

    warming_up_observations = 0
    collected_observations = 0

    active_run_id: str | None = None

    stop_reason = "max-samples-reached"

    wait_deadline = time.monotonic() + wait_timeout_seconds

    client = CollectorClient(
        base_url=settings.collector_base_url,
        api_key=settings.collector_api_key,
        timeout_seconds=(settings.collector_timeout_seconds),
        device_ids=settings.device_ids,
    )

    try:
        while collected_observations < max_samples:
            observation = await client.fetch_observation()

            if scenario_id is not None:
                if active_run_id is None:
                    if not scenario_is_ready(
                        observation=observation,
                        scenario_id=scenario_id,
                        simulation_seed=(simulation_seed),
                        start_phase=start_phase,
                    ):
                        if time.monotonic() >= wait_deadline:
                            seed_description = (
                                f" and seed {simulation_seed}"
                                if simulation_seed is not None
                                else ""
                            )

                            raise TimeoutError(
                                f"Scenario "
                                f"{scenario_id!r}"
                                f"{seed_description} "
                                f"did not reach phase "
                                f"{start_phase!r} "
                                f"within "
                                f"{wait_timeout_seconds:g} "
                                f"seconds."
                            )

                        await asyncio.sleep(interval_seconds)

                        continue

                    active_run_id = observation.metadata.simulation_run_id

                    extractor.reset()

                    print(
                        f"Scenario started: "
                        f"{scenario_id} "
                        f"(run_id="
                        f"{active_run_id}, "
                        f"seed="
                        f"{observation.metadata.simulation_seed})"
                    )

                elif not observation_belongs_to_run(
                    observation=observation,
                    scenario_id=scenario_id,
                    simulation_run_id=active_run_id,
                ):
                    stop_reason = "scenario-run-ended"

                    break

            expected_label = infer_expected_label(
                observation,
                fault_label,
            )

            features = extractor.update(observation)

            collected_observations += 1

            if features is None:
                warming_up_observations += 1
            else:
                prediction = predictor.predict(features)

                rows.append(
                    build_dataset_row(
                        observation=observation,
                        features=features,
                        expected_label=(expected_label),
                        predicted_label=(prediction.label),
                    )
                )

            if collected_observations < max_samples:
                await asyncio.sleep(interval_seconds)

    finally:
        await client.close()

    return CollectionResult(
        rows=rows,
        collected_observations=(collected_observations),
        warming_up_observations=(warming_up_observations),
        stop_reason=stop_reason,
        simulation_run_id=active_run_id,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Collect labeled simulation features and evaluate the rule baseline.")
    )

    parser.add_argument(
        "--max-samples",
        "--samples",
        dest="max_samples",
        type=positive_int,
        default=120,
        help=(
            "Maximum observations to collect. "
            "--samples remains available as an alias "
            "(default: 120)."
        ),
    )

    parser.add_argument(
        "--interval",
        type=non_negative_float,
        default=None,
        help=("Seconds between observations (default: COLLECTOR_POLL_INTERVAL_SECONDS)."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/rule_evaluation.csv"),
        help="Output .csv or .jsonl file.",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help=("Append to an existing dataset instead of replacing it."),
    )

    parser.add_argument(
        "--fault-label",
        choices=tuple(label for label in LABELS if label != "normal"),
        help=("Ground-truth label for every non-normal phase; use for ambiguous phase names."),
    )

    parser.add_argument(
        "--scenario-id",
        help=("Wait for this scenario and stop automatically when its run ends."),
    )

    parser.add_argument(
        "--simulation-seed",
        type=int,
        help=("Required simulation seed when waiting for a scenario."),
    )

    parser.add_argument(
        "--start-phase",
        default="normal",
        help=("Phase at which scenario collection may start (default: normal)."),
    )

    parser.add_argument(
        "--wait-timeout",
        type=non_negative_float,
        default=300.0,
        help=("Maximum seconds to wait for the requested scenario (default: 300)."),
    )

    args = parser.parse_args(argv)

    if args.simulation_seed is not None and args.scenario_id is None:
        parser.error("--simulation-seed requires --scenario-id")

    return args


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    settings = Settings()

    interval_seconds = (
        settings.collector_poll_interval_seconds if args.interval is None else args.interval
    )

    try:
        result = asyncio.run(
            collect_dataset(
                settings=settings,
                max_samples=args.max_samples,
                interval_seconds=(interval_seconds),
                fault_label=args.fault_label,
                scenario_id=args.scenario_id,
                simulation_seed=(args.simulation_seed),
                start_phase=args.start_phase,
                wait_timeout_seconds=(args.wait_timeout),
            )
        )

        write_dataset(
            result.rows,
            args.output,
            append=args.append,
        )

    except (
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as error:
        print(f"Dataset generation failed: {error}")

        return 1

    summary = summarize(
        result.rows,
        collected_observations=(result.collected_observations),
        warming_up_observations=(result.warming_up_observations),
    )

    print_summary(
        summary,
        args.output,
    )

    if result.simulation_run_id is not None:
        print(f"Simulation run ID: {result.simulation_run_id}")

    print(f"Collection stopped: {result.stop_reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
