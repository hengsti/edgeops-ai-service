"""Evaluate the deployed FastAPI ML backend with unseen simulation seeds.

This script starts every requested collector simulation, sends the resulting
observations through the real ``POST /v1/observations`` endpoint, stores the
returned ML predictions, and delegates aggregate reporting to
``evaluate_dataset.py``.

The model is never loaded or called directly here. This makes the evaluation an
end-to-end test of the deployed inference path:

    collector simulation -> CollectorClient -> FastAPI -> ML backend

Example (PowerShell):
    uv run edgeops-train evaluate-deployed `
        --expected-model-version <version-from-training-report> `
        --seed-range 62 66 `
        --speed 1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from edgeops_ai.collector_client import CollectorClient
from edgeops_ai.schemas import ObservationV1
from edgeops_ai.settings import Settings
from edgeops_ai.training.common import (
    FEATURE_COLUMNS,
    LABELS,
    METADATA_COLUMNS,
    SCENARIOS,
    ScriptError,
    add_seed_options,
    dataset_path,
    infer_expected_label,
    non_negative_float,
    normalize_label_value,
    positive_float,
    positive_int,
    selected_seeds,
    start_simulation,
)
from edgeops_ai.training.dataset import build_prediction_row, write_dataset
from edgeops_ai.training.evaluate import evaluate_dataset, print_report
from edgeops_ai.training.manifest import (
    upsert_manifest_run,
    write_manifest_atomic,
)
from edgeops_ai.training.pipeline import (
    DEFAULT_SIMULATION_ADMIN_KEY,
    DEFAULT_SIMULATION_BASE_URL,
    DEFAULT_SIMULATION_READ_KEY,
    assigned_variants,
    fetch_collector_version,
    simulation_runs_url,
    validate_started_run,
)

DEFAULT_HOLDOUT_SEEDS = tuple(range(62, 67))
DEFAULT_AI_BASE_URL = "http://localhost:8096"

RESULT_COLUMNS = (
    "expected_label",
    "predicted_label",
    "prediction_correct",
    "confidence",
    "model_version",
)

DATASET_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + RESULT_COLUMNS


def observation_belongs_to_run(
    observation: ObservationV1,
    *,
    scenario: str,
    seed: int,
    run_id: str,
) -> bool:
    metadata = observation.metadata

    return (
        metadata.scenario_id is not None
        and normalize_label_value(metadata.scenario_id) == normalize_label_value(scenario)
        and metadata.simulation_seed == seed
        and metadata.simulation_run_id == run_id
    )


def run_is_ready(
    observation: ObservationV1,
    *,
    scenario: str,
    seed: int,
    run_id: str,
    start_phase: str,
) -> bool:
    return (
        observation_belongs_to_run(
            observation,
            scenario=scenario,
            seed=seed,
            run_id=run_id,
        )
        and observation.metadata.simulation_phase is not None
        and normalize_label_value(observation.metadata.simulation_phase)
        == normalize_label_value(start_phase)
    )


def build_dataset_row(
    observation: ObservationV1,
    response: dict[str, Any],
    expected_label: str,
) -> dict[str, Any]:
    features = response.get("features")

    if not isinstance(features, dict):
        raise ScriptError("Predicted API response does not contain a feature object.")

    missing_features = [name for name in FEATURE_COLUMNS if name not in features]

    if missing_features:
        raise ScriptError(
            "Predicted API response is missing features: " + ", ".join(missing_features)
        )

    predicted_label = response.get("prediction")

    if predicted_label not in LABELS:
        raise ScriptError(f"API returned unsupported prediction label: {predicted_label!r}.")

    confidence = response.get("confidence")

    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ScriptError(f"API returned invalid confidence value: {confidence!r}.")

    if not 0.0 <= float(confidence) <= 1.0:
        raise ScriptError(f"API confidence must be between 0 and 1, got {confidence!r}.")

    return build_prediction_row(
        observation,
        features,
        expected_label,
        predicted_label,
        response["model_version"],
        confidence=float(confidence),
    )


async def require_ml_backend(
    client: httpx.AsyncClient,
    ai_base_url: str,
    expected_model_version: str,
) -> None:
    response = await client.get(f"{ai_base_url.rstrip('/')}/health")

    response.raise_for_status()
    health = response.json()

    if not isinstance(health, dict):
        raise ScriptError("AI health response must be a JSON object.")

    if health.get("prediction_backend") != "ml":
        raise ScriptError(
            "AI service is not using the ML backend: "
            f"prediction_backend="
            f"{health.get('prediction_backend')!r}."
        )

    if health.get("model_version") != expected_model_version:
        raise ScriptError(
            "AI service exposes the wrong model version: "
            f"expected {expected_model_version!r}, "
            f"got {health.get('model_version')!r}."
        )

    print(
        f"Verified deployed backend: prediction_backend=ml, model_version={expected_model_version}"
    )


async def post_observation(
    client: httpx.AsyncClient,
    *,
    ai_base_url: str,
    observation: ObservationV1,
    expected_model_version: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{ai_base_url.rstrip('/')}/v1/observations",
        json=observation.model_dump(mode="json"),
    )

    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ScriptError("AI observation response must be a JSON object.")

    status = payload.get("status")

    if status not in {"warming-up", "predicted"}:
        raise ScriptError(f"AI service returned unexpected status: {status!r}.")

    model_version = payload.get("model_version")

    if model_version != expected_model_version:
        raise ScriptError(
            "Observation was not evaluated by the expected "
            "deployed model: "
            f"expected {expected_model_version!r}, "
            f"got {model_version!r}."
        )

    return payload


async def collect_run(
    *,
    collector: CollectorClient,
    api_client: httpx.AsyncClient,
    scenario: str,
    seed: int,
    variant_id: str,
    speed: float,
    simulation_base_url: str,
    simulation_admin_key: str,
    ai_base_url: str,
    interval_seconds: float,
    wait_timeout_seconds: float,
    max_samples: int,
    api_timeout_seconds: float,
    start_phase: str,
    expected_model_version: str,
) -> tuple[list[dict[str, Any]], int, int, dict[str, Any]]:
    start_response = start_simulation(
        url=simulation_runs_url(simulation_base_url),
        api_key=simulation_admin_key,
        scenario=scenario,
        seed=seed,
        speed=speed,
        timeout_seconds=api_timeout_seconds,
        variant_id=variant_id,
    )
    run_id, run_state = validate_started_run(
        start_response,
        scenario=scenario,
        seed=seed,
        variant_id=variant_id,
        speed=speed,
    )

    print(
        "\nStarted candidate evaluation: "
        f"scenario={scenario}, seed={seed}, variant={variant_id}, run_id={run_id}"
    )

    rows: list[dict[str, Any]] = []

    collected_observations = 0
    warming_up_observations = 0

    run_started = False

    wait_deadline = time.monotonic() + wait_timeout_seconds

    while collected_observations < max_samples:
        observation = await collector.fetch_observation()

        if not run_started:
            if not run_is_ready(
                observation,
                scenario=scenario,
                seed=seed,
                run_id=run_id,
                start_phase=start_phase,
            ):
                if time.monotonic() >= wait_deadline:
                    raise ScriptError(
                        f"{scenario}/seed-{seed} did not reach "
                        f"{start_phase!r} within "
                        f"{wait_timeout_seconds:g} seconds."
                    )

                await asyncio.sleep(interval_seconds)
                continue

            run_started = True

            print(
                f"Run ready: "
                f"scenario={scenario}, "
                f"seed={seed}, "
                f"phase="
                f"{observation.metadata.simulation_phase}"
            )

        elif not observation_belongs_to_run(
            observation,
            scenario=scenario,
            seed=seed,
            run_id=run_id,
        ):
            break

        expected_label = infer_expected_label(observation)

        api_response = await post_observation(
            api_client,
            ai_base_url=ai_base_url,
            observation=observation,
            expected_model_version=expected_model_version,
        )

        collected_observations += 1

        if api_response["status"] == "warming-up":
            warming_up_observations += 1
        else:
            rows.append(
                build_dataset_row(
                    observation=observation,
                    response=api_response,
                    expected_label=expected_label,
                )
            )

        await asyncio.sleep(interval_seconds)

    else:
        raise ScriptError(
            f"{scenario}/seed-{seed} reached the "
            "--max-samples safety limit before the "
            "simulation run ended."
        )

    print(
        f"Completed run: "
        f"{len(rows)} predicted rows, "
        f"{warming_up_observations} warming-up observations"
    )

    return (
        rows,
        collected_observations,
        warming_up_observations,
        run_state,
    )


async def evaluate(
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    settings = Settings()

    interval_seconds = (
        settings.collector_poll_interval_seconds if args.interval is None else args.interval
    )

    collector = CollectorClient(
        base_url=args.simulation_base_url,
        api_key=args.simulation_read_key,
        timeout_seconds=args.api_timeout,
        device_ids=(args.simulation_device_id,),
    )

    api_client = httpx.AsyncClient(timeout=args.api_timeout)

    total_rows = 0
    total_observations = 0
    total_warming_up = 0
    manifest: dict[str, Any] | None = None
    variants = assigned_variants(args.seeds)

    try:
        await require_ml_backend(
            api_client,
            args.ai_base_url,
            args.expected_model_version,
        )

        for seed in args.seeds:
            for scenario in args.scenarios:
                (
                    rows,
                    observations,
                    warming_up,
                    run_state,
                ) = await collect_run(
                    collector=collector,
                    api_client=api_client,
                    scenario=scenario,
                    seed=seed,
                    variant_id=variants[seed],
                    speed=args.speed,
                    simulation_base_url=args.simulation_base_url,
                    simulation_admin_key=args.simulation_admin_key,
                    ai_base_url=args.ai_base_url,
                    interval_seconds=interval_seconds,
                    wait_timeout_seconds=args.wait_timeout,
                    max_samples=args.max_samples,
                    api_timeout_seconds=args.api_timeout,
                    start_phase=args.start_phase,
                    expected_model_version=args.expected_model_version,
                )

                output = dataset_path(
                    args.output_dir,
                    scenario,
                    seed,
                )

                if not rows:
                    raise ScriptError(f"No predicted rows were collected for {output.name}.")
                write_dataset(
                    rows,
                    output,
                    columns=DATASET_COLUMNS,
                )
                run_id = str(run_state["run_id"])
                manifest = upsert_manifest_run(
                    manifest,
                    output_dir=args.output_dir,
                    run_id=run_id,
                    entry={
                        "scenario_id": scenario,
                        "seed": seed,
                        "variant_id": variants[seed],
                        "speed": float(run_state["speed"]),
                        "resolved_effects": run_state["resolved_effects"],
                        "workload_scale": float(run_state["workload_scale"]),
                        "unavailable_metric_groups": sorted(
                            str(group) for group in run_state["unavailable_metric_groups"]
                        ),
                        "collector_version": args.collector_version,
                        "csv_path": output.resolve()
                        .relative_to(args.output_dir.resolve())
                        .as_posix(),
                    },
                )
                write_manifest_atomic(args.output_dir, manifest)

                print(f"Dataset: {output}")

                total_rows += len(rows)
                total_observations += observations
                total_warming_up += warming_up

    finally:
        await api_client.aclose()
        await collector.close()

    return (
        total_rows,
        total_observations,
        total_warming_up,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--expected-model-version",
        required=True,
        help="Candidate artifact version expected from the deployed ML backend.",
    )
    add_seed_options(parser)

    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIOS,
        help=("Scenario to evaluate; repeat for a subset (default: all scenarios)."),
    )

    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help=("Simulation speed sent to the collector (default: 1)."),
    )

    parser.add_argument(
        "--simulation-base-url",
        default=os.getenv("SIMULATION_COLLECTOR_BASE_URL", DEFAULT_SIMULATION_BASE_URL),
        help=f"Independent simulation collector URL (default: {DEFAULT_SIMULATION_BASE_URL}).",
    )
    parser.add_argument(
        "--simulation-read-key",
        default=os.getenv("SIMULATION_COLLECTOR_API_KEY", DEFAULT_SIMULATION_READ_KEY),
        help="Simulation collector read key.",
    )
    parser.add_argument(
        "--simulation-admin-key",
        default=os.getenv("SIMULATION_COLLECTOR_ADMIN_API_KEY", DEFAULT_SIMULATION_ADMIN_KEY),
        help="Simulation collector admin key.",
    )
    parser.add_argument(
        "--simulation-device-id",
        default=os.getenv("SIMULATION_COLLECTOR_DEVICE_ID", "esp32-simulated-01"),
        help="Simulation device ID used for heartbeat evaluation.",
    )

    parser.add_argument(
        "--ai-base-url",
        default=DEFAULT_AI_BASE_URL,
        help=(f"Deployed AI service URL (default: {DEFAULT_AI_BASE_URL})."),
    )

    parser.add_argument(
        "--api-timeout",
        type=positive_float,
        default=10.0,
        help=("HTTP request timeout in seconds (default: 10)."),
    )

    parser.add_argument(
        "--max-samples",
        type=positive_int,
        default=1_000,
        help=("Per-run safety limit (default: 1000)."),
    )

    parser.add_argument(
        "--interval",
        type=non_negative_float,
        default=None,
        help=("Polling interval; otherwise use the service setting."),
    )

    parser.add_argument(
        "--wait-timeout",
        type=non_negative_float,
        default=300.0,
        help=("Maximum wait for a started run (default: 300 seconds)."),
    )

    parser.add_argument(
        "--start-phase",
        default="normal",
        help=("Phase at which collection may start (default: normal)."),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/holdout"),
        help=("Holdout CSV directory (default: data/holdout)."),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Aggregate evaluation report path.",
    )

    args = parser.parse_args(argv)

    args.report = args.report or args.output_dir / "candidate-report.json"
    args.seeds = selected_seeds(args, default_seeds=DEFAULT_HOLDOUT_SEEDS)

    args.scenarios = tuple(args.scenario) if args.scenario else SCENARIOS

    return args


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    try:
        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.collector_version = fetch_collector_version(
            base_url=args.simulation_base_url,
            api_key=args.simulation_read_key,
            timeout_seconds=args.api_timeout,
        )

        print(
            f"Holdout matrix: "
            f"{len(args.scenarios)} scenarios x "
            f"{len(args.seeds)} unseen seeds = "
            f"{len(args.scenarios) * len(args.seeds)} runs"
        )

        print(f"Seeds: {', '.join(map(str, args.seeds))}")

        (
            total_rows,
            total_observations,
            total_warming_up,
        ) = asyncio.run(evaluate(args))

        report = evaluate_dataset(
            input_dir=args.output_dir,
            output=args.report,
            scenarios=args.scenarios,
            seeds=args.seeds,
        )
        print_report(report, args.report)

    except (
        ScriptError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Deployed model evaluation failed: {error}",
            file=sys.stderr,
        )

        return 1

    print("\nDeployed ML evaluation completed successfully.")

    print(f"Observations submitted: {total_observations}")

    print(f"Warming-up observations skipped: {total_warming_up}")

    print(f"Predicted rows: {total_rows}")
    print(f"Report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
