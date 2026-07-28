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
from collections.abc import Sequence
from pathlib import Path

from scripts.common import (
    DATASET_SCHEMA_VERSION,
    FEATURE_COLUMNS,
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

DEFAULT_SIMULATION_URL = "http://localhost:8095/v1/simulation/runs"
DEFAULT_API_KEY = "local-dev-admin-key"


def dataset_is_complete(
    path: Path,
    *,
    scenario: str,
    seed: int,
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

    if not path.is_file():
        return False

    try:
        if path.stat().st_size == 0:
            return False

        with path.open(
            encoding="utf-8",
            newline="",
        ) as dataset_file:
            reader = csv.DictReader(dataset_file)

            required_columns = {
                "dataset_schema_version",
                "scenario_id",
                "simulation_seed",
                "simulation_phase",
                "expected_label",
            } | set(FEATURE_COLUMNS)

            if reader.fieldnames is None:
                return False

            if not required_columns.issubset(reader.fieldnames):
                return False

            row_count = 0
            expected_labels: set[str] = set()
            phases: set[str] = set()

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

    except (OSError, csv.Error):
        return False

    has_required_labels = {
        "normal",
        scenario,
    }.issubset(expected_labels)

    has_recovery_phase = bool({"recovery", "recovered"} & phases)

    return row_count > 0 and has_required_labels and has_recovery_phase


def partition_runs(
    *,
    output_dir: Path,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    overwrite: bool,
) -> tuple[
    list[tuple[str, int, Path]],
    list[tuple[str, int, Path]],
]:
    """Check the complete matrix before starting simulations.

    Returns:
        A tuple containing reusable runs and pending runs.
    """

    reusable: list[tuple[str, int, Path]] = []
    pending: list[tuple[str, int, Path]] = []

    # Seed-first ordering ensures all scenarios for one seed
    # are completed before moving to the next seed.
    for seed in seeds:
        for scenario in scenarios:
            output = dataset_path(
                output_dir,
                scenario,
                seed,
            )
            run = (scenario, seed, output)

            if overwrite:
                pending.append(run)
                continue

            if dataset_is_complete(
                output,
                scenario=scenario,
                seed=seed,
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
    output: Path,
) -> None:
    """Start and collect one complete simulation run."""

    print(f"\nStarting simulation: scenario={scenario}, seed={seed}, speed={args.speed}")

    response = start_simulation(
        url=args.simulation_url,
        api_key=args.api_key,
        scenario=scenario,
        seed=seed,
        speed=args.speed,
        timeout_seconds=args.api_timeout,
    )

    if response:
        run_id = extract_run_id(response)

        if run_id is not None:
            print(f"Simulation API run ID: {run_id}")

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
        str(output),
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
    )

    if not output.is_file():
        raise ScriptError(f"Dataset collector did not create the expected file: {output}")

    if output.stat().st_size == 0:
        raise ScriptError(f"Dataset collector created an empty file: {output}")

    if not dataset_is_complete(
        output,
        scenario=scenario,
        seed=seed,
    ):
        raise ScriptError(
            f"Generated dataset is incomplete or contains mismatched metadata: {output}"
        )


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
        "--simulation-url",
        default=DEFAULT_SIMULATION_URL,
        help=(f"Simulation start endpoint. Default: {DEFAULT_SIMULATION_URL}."),
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "EDGEOPS_SIMULATION_API_KEY",
            DEFAULT_API_KEY,
        ),
        help=("Simulation API key. Uses EDGEOPS_SIMULATION_API_KEY when set."),
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
        default=Path("artifacts/model-candidate.joblib"),
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

        reusable_pairs = {(scenario, seed) for scenario, seed, _ in reusable_runs}

        complete_seeds = [
            seed
            for seed in seeds
            if all((scenario, seed) in reusable_pairs for scenario in scenarios)
        ]

        pending_seeds = list(dict.fromkeys(seed for _, seed, _ in pending_runs))

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
        for scenario, seed, output in pending_runs:
            generate_dataset(
                args=args,
                project_root=project_root,
                collection_script=collection_script,
                scenario=scenario,
                seed=seed,
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

    except (OSError, ScriptError) as error:
        print(
            f"Pipeline failed: {error}",
            file=sys.stderr,
        )
        return 1

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
