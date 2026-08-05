"""Orchestrate EdgeOps simulation runs, dataset generation, and evaluation.

For every missing scenario/seed pair, this script:

1. Starts a simulation through the simulation API.
2. Invokes collect_rules_dataset.py to collect the run.
3. Continues sequentially until all requested datasets exist.
4. Invokes evaluate_dataset.py for the complete dataset matrix.
5. Optionally invokes train_model.py when --train is set.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from scripts.common import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    SCENARIOS,
    ScriptError,
    dataset_path,
    extract_run_id,
    non_negative_float,
    positive_float,
    positive_int,
    resolve_project_path,
    selected_seeds,
    start_simulation,
)
from scripts.simulation_manifest import (
    SIMULATION_CONTRACT_VERSION,
    SIMULATION_VARIANTS,
    SimulationManifestError,
    entry_for_csv,
    try_load_manifest,
    upsert_manifest_run,
    write_manifest_atomic,
)

DEFAULT_SIMULATION_BASE_URL = "http://localhost:8097"
DEFAULT_SIMULATION_READ_KEY = "local-dev-key"
DEFAULT_SIMULATION_ADMIN_KEY = "local-dev-admin-key"


def assigned_variants(seeds: Sequence[int]) -> dict[int, str]:
    """Assign mild/degraded/severe in deterministic ordered-seed rotation."""

    ordered = tuple(dict.fromkeys(seeds))
    return {
        seed: SIMULATION_VARIANTS[index % len(SIMULATION_VARIANTS)]
        for index, seed in enumerate(ordered)
    }


def dataset_is_complete(
    path: Path,
    *,
    scenario: str,
    seed: int,
    variant_id: str,
    manifest: dict[str, object] | None,
    output_dir: Path,
) -> bool:
    """Check whether an existing CSV represents a complete run.

    A reusable dataset must:

    - exist and contain data;
    - use the current dataset schema version;
    - contain the required metadata columns;
    - contain every current feature column;
    - contain only the requested scenario and seed;
    - contain normal and fault labels;
    - contain a recovery phase.

    Empty, truncated, or mismatched files are regenerated.
    """

    if not path.is_file() or manifest is None:
        return False

    manifest_match = entry_for_csv(
        manifest,
        output_dir=output_dir,
        csv_path=path,
    )
    if manifest_match is None:
        return False
    manifest_run_id, manifest_entry = manifest_match
    if (
        manifest_entry["scenario_id"] != scenario
        or manifest_entry["seed"] != seed
        or manifest_entry["variant_id"] != variant_id
    ):
        return False

    try:
        if path.stat().st_size == 0:
            return False

        with path.open(
            encoding="utf-8",
            newline="",
        ) as dataset_file:
            reader = csv.DictReader(dataset_file)

            required_columns = set(DATASET_COLUMNS)

            if reader.fieldnames is None:
                return False

            if not required_columns.issubset(reader.fieldnames):
                return False

            row_count = 0
            expected_labels: set[str] = set()
            phases: set[str] = set()
            run_ids: set[str] = set()

            for row in reader:
                row_count += 1

                if (row.get("dataset_schema_version") or "").strip() != DATASET_SCHEMA_VERSION:
                    return False

                row_scenario = (row.get("scenario_id") or "").strip().lower()

                if row_scenario != scenario:
                    return False

                try:
                    row_seed = int(row.get("simulation_seed") or "")
                except (TypeError, ValueError):
                    return False

                if row_seed != seed:
                    return False

                expected_labels.add((row.get("expected_label") or "").strip().lower())
                phases.add((row.get("simulation_phase") or "").strip().lower())
                run_id = (row.get("simulation_run_id") or "").strip()
                if not run_id:
                    return False
                run_ids.add(run_id)

    except (OSError, csv.Error):
        return False

    has_required_labels = {
        "normal",
        scenario,
    }.issubset(expected_labels)

    has_recovery_phase = bool({"recovery", "recovered"} & phases)

    return (
        row_count > 0
        and has_required_labels
        and has_recovery_phase
        and run_ids == {manifest_run_id}
    )


def partition_runs(
    *,
    output_dir: Path,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    overwrite: bool,
) -> tuple[
    list[tuple[str, int, str, Path]],
    list[tuple[str, int, str, Path]],
]:
    """Check the complete matrix before starting simulations.

    Returns:
        A tuple containing reusable runs and pending runs.
    """

    reusable: list[tuple[str, int, str, Path]] = []
    pending: list[tuple[str, int, str, Path]] = []
    manifest = try_load_manifest(output_dir)
    variants = assigned_variants(seeds)

    # Seed-first ordering ensures all scenarios for one seed
    # are completed before moving to the next seed.
    for seed in seeds:
        for scenario in scenarios:
            output = dataset_path(
                output_dir,
                scenario,
                seed,
            )
            variant_id = variants[seed]
            run = (scenario, seed, variant_id, output)

            if overwrite:
                pending.append(run)
                continue

            if dataset_is_complete(
                output,
                scenario=scenario,
                seed=seed,
                variant_id=variant_id,
                manifest=manifest,
                output_dir=output_dir,
            ):
                reusable.append(run)
            else:
                pending.append(run)

    return reusable, pending


def run_command(
    command: Sequence[str],
    *,
    project_root: Path,
    stage: str,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    """Run a child script with the project root on PYTHONPATH."""

    print(f"\n--- {stage} ---", flush=True)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")

    environment["PYTHONPATH"] = (
        str(project_root)
        if not existing_pythonpath
        else str(project_root) + os.pathsep + existing_pythonpath
    )
    if environment_overrides:
        environment.update(environment_overrides)

    try:
        result = subprocess.run(
            list(command),
            cwd=project_root,
            env=environment,
            check=False,
        )
    except OSError as error:
        raise ScriptError(f"Could not start {stage}: {error}") from error

    if result.returncode != 0:
        raise ScriptError(f"{stage} failed with exit code {result.returncode}.")


def simulation_runs_url(base_url: str) -> str:
    """Return the contract-v2 run endpoint for an independent collector."""

    return f"{base_url.rstrip('/')}/v1/simulation/runs"


def fetch_collector_version(*, base_url: str, api_key: str, timeout_seconds: float) -> str:
    """Verify the target is a simulation collector and return its version."""

    response = httpx.get(
        f"{base_url.rstrip('/')}/v1/meta",
        headers={"X-EdgeOps-Key": api_key},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("mode") != "simulation":
        raise ScriptError("Target collector /v1/meta does not report simulation mode.")
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ScriptError("Simulation collector /v1/meta did not return a version.")
    return version.strip()


def validate_started_run(
    response: dict[str, object] | None,
    *,
    scenario: str,
    seed: int,
    variant_id: str,
    speed: float,
) -> tuple[str, dict[str, object]]:
    """Validate the simulation contract-v2 start response."""

    if response is None:
        raise ScriptError("Simulation start returned no response body.")
    if response.get("simulation_contract_version") != SIMULATION_CONTRACT_VERSION:
        raise ScriptError("Simulation start response is not contract version 2.")
    if (
        response.get("scenario_id") != scenario
        or response.get("seed") != seed
        or response.get("variant_id") != variant_id
    ):
        raise ScriptError("Simulation start response does not match the requested run.")
    response_speed = response.get("speed")
    if not isinstance(response_speed, (int, float)) or float(response_speed) != speed:
        raise ScriptError("Simulation start response has an unexpected speed.")
    run_id = extract_run_id(response)
    if run_id is None:
        raise ScriptError("Simulation start response has no run_id.")
    if not isinstance(response.get("resolved_effects"), dict):
        raise ScriptError("Simulation start response has no resolved_effects object.")
    workload_scale = response.get("workload_scale")
    if (
        not isinstance(workload_scale, (int, float))
        or isinstance(workload_scale, bool)
        or not 0.65 <= float(workload_scale) <= 1.50
    ):
        raise ScriptError("Simulation start response has an invalid workload_scale.")
    unavailable = response.get("unavailable_metric_groups")
    if not isinstance(unavailable, list):
        raise ScriptError("Simulation start response has invalid unavailable_metric_groups.")
    return run_id, response


def validate_collected_csv(
    path: Path,
    *,
    scenario: str,
    seed: int,
    run_id: str,
) -> None:
    """Validate identity and minimum lifecycle coverage before publishing a CSV."""

    try:
        with path.open(encoding="utf-8", newline="") as dataset_file:
            reader = csv.DictReader(dataset_file)
            if reader.fieldnames is None or not set(DATASET_COLUMNS).issubset(reader.fieldnames):
                raise ScriptError(f"Generated dataset has a stale schema: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise ScriptError(f"Unable to validate generated dataset {path}: {error}") from error

    labels = {(row.get("expected_label") or "").strip().lower() for row in rows}
    phases = {(row.get("simulation_phase") or "").strip().lower() for row in rows}
    if not rows or not {"normal", scenario}.issubset(labels):
        raise ScriptError(f"Generated dataset lacks normal/fault coverage: {path}")
    if not ({"recovery", "recovered"} & phases):
        raise ScriptError(f"Generated dataset lacks a recovery phase: {path}")
    for row in rows:
        try:
            row_seed = int(row.get("simulation_seed") or "")
        except (TypeError, ValueError) as error:
            raise ScriptError(f"Generated dataset has an invalid seed: {path}") from error
        if (
            (row.get("dataset_schema_version") or "").strip() != DATASET_SCHEMA_VERSION
            or (row.get("scenario_id") or "").strip().lower() != scenario
            or row_seed != seed
            or (row.get("simulation_run_id") or "").strip() != run_id
        ):
            raise ScriptError(f"Generated dataset metadata does not match its run: {path}")


def script_paths(
    script_dir: Path,
) -> tuple[Path, Path, Path]:
    """Resolve and validate the child scripts."""

    collection_script = script_dir / "collect_rules_dataset.py"
    evaluation_script = script_dir / "evaluate_dataset.py"
    training_script = script_dir / "train_model.py"

    required_scripts = (
        collection_script,
        evaluation_script,
    )

    missing = [str(path) for path in required_scripts if not path.is_file()]

    if missing:
        raise ScriptError(f"Required script not found: {', '.join(missing)}")

    return (
        collection_script,
        evaluation_script,
        training_script,
    )


def generate_dataset(
    *,
    args: argparse.Namespace,
    project_root: Path,
    collection_script: Path,
    scenario: str,
    seed: int,
    variant_id: str,
    output: Path,
) -> None:
    """Start and collect one complete simulation run."""

    print(
        "\nStarting simulation: "
        f"scenario={scenario}, seed={seed}, variant={variant_id}, speed={args.speed}"
    )

    collector_version = fetch_collector_version(
        base_url=args.simulation_base_url,
        api_key=args.simulation_read_key,
        timeout_seconds=args.api_timeout,
    )

    response = start_simulation(
        url=simulation_runs_url(args.simulation_base_url),
        api_key=args.simulation_admin_key,
        scenario=scenario,
        seed=seed,
        speed=args.speed,
        timeout_seconds=args.api_timeout,
        variant_id=variant_id,
    )
    run_id, run_state = validate_started_run(
        response,
        scenario=scenario,
        seed=seed,
        variant_id=variant_id,
        speed=args.speed,
    )
    print(f"Simulation API run ID: {run_id}")

    temporary_output = output.with_suffix(".tmp.csv")

    command = [
        sys.executable,
        str(collection_script),
        "--scenario-id",
        scenario,
        "--simulation-seed",
        str(seed),
        "--max-samples",
        str(args.max_samples),
        "--wait-timeout",
        str(args.wait_timeout),
        "--output",
        str(temporary_output),
    ]

    if args.interval is not None:
        command.extend(
            (
                "--interval",
                str(args.interval),
            )
        )

    run_command(
        command,
        project_root=project_root,
        stage=f"Collect {scenario}/seed-{seed}",
        environment_overrides={
            "COLLECTOR_BASE_URL": args.simulation_base_url,
            "COLLECTOR_API_KEY": args.simulation_read_key,
            "COLLECTOR_DEVICE_IDS": args.simulation_device_id,
        },
    )

    if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
        raise ScriptError(f"Dataset collector did not create a non-empty file: {temporary_output}")
    validate_collected_csv(
        temporary_output,
        scenario=scenario,
        seed=seed,
        run_id=run_id,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_output, output)
    current_manifest = try_load_manifest(args.output_dir)
    manifest = upsert_manifest_run(
        current_manifest,
        output_dir=args.output_dir,
        run_id=run_id,
        entry={
            "scenario_id": scenario,
            "seed": seed,
            "variant_id": variant_id,
            "speed": float(run_state["speed"]),
            "resolved_effects": run_state["resolved_effects"],
            "workload_scale": float(run_state["workload_scale"]),
            "unavailable_metric_groups": sorted(
                str(group) for group in run_state["unavailable_metric_groups"]
            ),
            "collector_version": collector_version,
            "csv_path": output.resolve().relative_to(args.output_dir.resolve()).as_posix(),
        },
    )
    write_manifest_atomic(args.output_dir, manifest)


def evaluate_datasets(
    *,
    args: argparse.Namespace,
    project_root: Path,
    evaluation_script: Path,
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> None:
    """Evaluate the complete requested dataset matrix."""

    command = [
        sys.executable,
        str(evaluation_script),
        "--input-dir",
        str(args.output_dir),
        "--output",
        str(args.evaluation_report),
    ]

    for scenario in scenarios:
        command.extend(
            (
                "--scenario",
                scenario,
            )
        )

    for seed in seeds:
        command.extend(
            (
                "--seed",
                str(seed),
            )
        )

    run_command(
        command,
        project_root=project_root,
        stage="Evaluate complete dataset",
    )


def train_model(
    *,
    args: argparse.Namespace,
    project_root: Path,
    training_script: Path,
) -> None:
    """Run the lightweight model training script."""

    if not training_script.is_file():
        raise ScriptError(f"Training script not found: {training_script}")

    command = [
        sys.executable,
        str(training_script),
        "--input-dir",
        str(args.output_dir),
        "--artifact",
        str(args.model_artifact),
        "--report",
        str(args.training_report),
    ]

    run_command(
        command,
        project_root=project_root,
        stage="Train lightweight models",
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    seed_group = parser.add_mutually_exclusive_group()

    seed_group.add_argument(
        "--seed",
        action="append",
        type=int,
        help=("Simulation seed. Repeat for multiple seeds. Default: 42, 43, 44."),
    )

    seed_group.add_argument(
        "--seed-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help=("Inclusive simulation seed range. Example: --seed-range 42 61."),
    )

    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIOS,
        help=("Scenario to generate. Repeat for a subset. Default: all four."),
    )

    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help=("Simulation speed sent to the API. Default: 1."),
    )

    parser.add_argument(
        "--simulation-base-url",
        default=os.getenv("SIMULATION_COLLECTOR_BASE_URL", DEFAULT_SIMULATION_BASE_URL),
        help=(f"Independent simulation collector URL. Default: {DEFAULT_SIMULATION_BASE_URL}."),
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
        help="Device ID queried while collecting simulated heartbeat state.",
    )

    parser.add_argument(
        "--api-timeout",
        type=positive_float,
        default=10.0,
        help=("Simulation start request timeout in seconds. Default: 10."),
    )

    parser.add_argument(
        "--max-samples",
        type=positive_int,
        default=1_000,
        help=("Safety limit passed to the collector for each run. Default: 1000."),
    )

    parser.add_argument(
        "--interval",
        type=non_negative_float,
        default=None,
        help=("Collector polling interval. If omitted, the collector default is used."),
    )

    parser.add_argument(
        "--wait-timeout",
        type=non_negative_float,
        default=300.0,
        help=("Seconds the collector may wait for a started run. Default: 300."),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help=("Dataset output directory. Default: data/generated."),
    )

    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("data/generated/rule-baseline-report.json"),
        help="Rule evaluation report path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Regenerate every selected dataset, including complete existing datasets."),
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help=("Run train_model.py after successful dataset evaluation."),
    )

    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=Path("artifacts/model-contract-v2-experimental.joblib"),
        help="Candidate model artifact path used with --train.",
    )

    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("data/generated/ml-baseline-report.json"),
        help="Training report path used with --train.",
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    try:
        (
            collection_script,
            evaluation_script,
            training_script,
        ) = script_paths(script_dir)

        scenarios = tuple(args.scenario) if args.scenario else SCENARIOS
        seeds = selected_seeds(args)

        if args.train and set(scenarios) != set(SCENARIOS):
            raise ScriptError("--train requires all four scenarios so every label is present.")

        # Always interpret relative paths from the project
        # root, regardless of the current terminal directory.
        args.output_dir = resolve_project_path(
            args.output_dir,
            project_root,
        )
        args.evaluation_report = resolve_project_path(
            args.evaluation_report,
            project_root,
        )
        args.model_artifact = resolve_project_path(
            args.model_artifact,
            project_root,
        )
        args.training_report = resolve_project_path(
            args.training_report,
            project_root,
        )

        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_runs = len(scenarios) * len(seeds)

        # Scan the entire matrix before contacting the
        # simulation API.
        reusable_runs, pending_runs = partition_runs(
            output_dir=args.output_dir,
            scenarios=scenarios,
            seeds=seeds,
            overwrite=args.overwrite,
        )

        reusable_pairs = {(scenario, seed) for scenario, seed, _, _ in reusable_runs}

        complete_seeds = [
            seed
            for seed in seeds
            if all((scenario, seed) in reusable_pairs for scenario in scenarios)
        ]

        pending_seeds = list(dict.fromkeys(seed for _, seed, _, _ in pending_runs))

        print(
            f"Pipeline matrix: {len(scenarios)} scenarios x {len(seeds)} seeds = {total_runs} runs"
        )

        if args.overwrite:
            print("Overwrite enabled: all requested runs will be generated again.")
        else:
            print(
                "Dataset pre-check: "
                f"{len(reusable_runs)} complete runs "
                f"found, {len(pending_runs)} runs "
                "missing or incomplete."
            )
            print(
                "Complete seeds: "
                + (", ".join(map(str, complete_seeds)) if complete_seeds else "none")
            )

        print(
            "Seeds to generate: "
            + (", ".join(map(str, pending_seeds)) if pending_seeds else "none")
        )

        # Only missing, incomplete, or explicitly
        # overwritten datasets are generated.
        for scenario, seed, variant_id, output in pending_runs:
            generate_dataset(
                args=args,
                project_root=project_root,
                collection_script=collection_script,
                scenario=scenario,
                seed=seed,
                variant_id=variant_id,
                output=output,
            )

        print(
            "\nDataset generation complete: "
            f"{len(pending_runs)} generated, "
            f"{len(reusable_runs)} reused."
        )

        # Evaluate the entire requested matrix, including
        # both existing and newly generated datasets.
        evaluate_datasets(
            args=args,
            project_root=project_root,
            evaluation_script=evaluation_script,
            scenarios=scenarios,
            seeds=seeds,
        )

        if args.train:
            train_model(
                args=args,
                project_root=project_root,
                training_script=training_script,
            )

    except (httpx.HTTPError, OSError, ScriptError, SimulationManifestError) as error:
        print(
            f"Pipeline failed: {error}",
            file=sys.stderr,
        )
        return 1

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
