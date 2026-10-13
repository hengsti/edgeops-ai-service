"""Shared constants and helpers for the EdgeOps AI scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

import httpx

from edgeops_ai.schemas import FeatureVector, ObservationV1, PredictionLabel

SCENARIOS = (
    "ingestion-backpressure",
    "missing-device-heartbeat",
    "malformed-sensor-payload",
    "influxdb-write-failure",
    "mqtt-broker-unavailable",
)

DEFAULT_SEEDS = (42, 43, 44)

LABELS: tuple[str, ...] = get_args(PredictionLabel)

DATASET_SCHEMA_VERSION = "2.2"

NORMAL_PHASES = {
    "baseline",
    "healthy",
    "normal",
    "recovered",
    "recovery",
}

RECOVERY_PHASES = {
    "recovered",
    "recovery",
}

PHASE_LABEL_ALIASES = {
    "backpressure": "ingestion-backpressure",
    "heartbeat-missing": "missing-device-heartbeat",
    "influx-failure": "influxdb-write-failure",
    "influxdb-failure": "influxdb-write-failure",
    "malformed-payload": "malformed-sensor-payload",
    "malformed-payloads": "malformed-sensor-payload",
    "overloaded": "ingestion-backpressure",
    "transform-failure": "malformed-sensor-payload",
    "write-failure": "influxdb-write-failure",
}

METADATA_COLUMNS = (
    "dataset_schema_version",
    "captured_at",
    "source_host",
    "scenario_id",
    "simulation_run_id",
    "simulation_seed",
    "simulation_phase",
)

FEATURE_COLUMNS = tuple(FeatureVector.model_fields)

RESULT_COLUMNS = (
    "expected_label",
    "predicted_label",
    "prediction_correct",
    "model_version",
)

DATASET_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + RESULT_COLUMNS


class ScriptError(RuntimeError):
    """Raised when a script stage cannot be completed."""


def build_dataset_metadata(observation: ObservationV1) -> dict[str, Any]:
    metadata = observation.metadata
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "captured_at": metadata.captured_at.isoformat(),
        "source_host": metadata.source_host,
        "scenario_id": metadata.scenario_id,
        "simulation_run_id": metadata.simulation_run_id,
        "simulation_seed": metadata.simulation_seed,
        "simulation_phase": metadata.simulation_phase,
    }


def normalize_label_value(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def positive_int(value: str) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)

    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")

    return parsed


def selected_seeds(
    args: argparse.Namespace,
    default_seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> tuple[int, ...]:
    """Return the selected seeds without duplicates."""

    if args.seed_range is not None:
        first, last = args.seed_range

        if first > last:
            raise ScriptError("The first --seed-range value must not exceed the second.")

        seeds: Sequence[int] = range(first, last + 1)

    elif args.seed:
        seeds = args.seed

    else:
        seeds = default_seeds

    return tuple(dict.fromkeys(seeds))


def resolve_project_path(path: Path, project_root: Path) -> Path:
    """Resolve relative paths from the project root."""

    if path.is_absolute():
        return path

    return project_root / path


def dataset_path(output_dir: Path, scenario: str, seed: int) -> Path:
    """Return the expected path for one dataset."""

    return output_dir / f"{scenario}-seed-{seed}.csv"


def infer_expected_label(observation: Any, fault_label: str | None = None) -> str:
    """Infer ground truth without using metadata as predictor input.

    A normal/recovery phase always maps to ``normal``. For any other phase, an
    explicit ``fault_label`` wins, followed by a known phase alias and finally
    the simulation scenario ID. Ambiguous metadata is rejected.
    """

    metadata = observation.metadata

    if not metadata.simulated or metadata.collector_mode != "simulation":
        raise ValueError("Dataset generation requires collector simulation observations.")

    if not metadata.simulation_phase:
        if metadata.scenario_id is None:
            return "normal"

        raise ValueError(
            "Simulation observation has a scenario_id but is missing metadata.simulation_phase."
        )

    phase = normalize_label_value(metadata.simulation_phase)

    if phase in NORMAL_PHASES:
        return "normal"

    if fault_label is not None:
        normalized_fault_label = normalize_label_value(fault_label)

        if normalized_fault_label not in LABELS or normalized_fault_label == "normal":
            raise ValueError(f"Unsupported fault label: {fault_label!r}.")

        return normalized_fault_label

    if phase in LABELS:
        return phase

    if phase in PHASE_LABEL_ALIASES:
        return PHASE_LABEL_ALIASES[phase]

    if metadata.scenario_id:
        scenario_id = normalize_label_value(metadata.scenario_id)

        if scenario_id in LABELS and scenario_id != "normal":
            return scenario_id

    raise ValueError(
        "Cannot infer an expected label from "
        f"simulation_phase={metadata.simulation_phase!r} and "
        f"scenario_id={metadata.scenario_id!r}. "
        "Pass --fault-label explicitly."
    )


def start_simulation(
    *,
    url: str,
    api_key: str,
    scenario: str,
    seed: int,
    speed: float,
    timeout_seconds: float,
    variant_id: str | None = None,
) -> dict[str, Any] | None:
    """Start one simulation through the simulation API."""

    try:
        request_body: dict[str, Any] = {
            "scenario_id": scenario,
            "seed": seed,
            "speed": speed,
        }
        if variant_id is not None:
            request_body["variant_id"] = variant_id

        response = httpx.post(
            url,
            headers={"X-EdgeOps-Key": api_key},
            json=request_body,
            timeout=timeout_seconds,
        )
        response.raise_for_status()

    except httpx.HTTPStatusError as error:
        details = error.response.text.strip()
        suffix = f": {details}" if details else ""

        raise ScriptError(
            f"Simulation API rejected {scenario}/seed-{seed} "
            f"with HTTP {error.response.status_code}{suffix}"
        ) from error

    except httpx.TimeoutException as error:
        raise ScriptError(
            f"Simulation API request timed out after {timeout_seconds:g} seconds."
        ) from error

    except httpx.HTTPError as error:
        raise ScriptError(f"Could not reach the simulation API at {url}: {error}") from error

    if not response.content:
        return None

    try:
        decoded = response.json()
    except ValueError as error:
        raise ScriptError("Simulation API returned invalid JSON.") from error

    if not isinstance(decoded, dict):
        raise ScriptError("Simulation API response must be a JSON object.")

    return decoded


def extract_run_id(payload: dict[str, Any] | None) -> str | None:
    """Extract the simulation run ID from a simulation start response."""

    if not payload:
        return None

    run_id = payload.get("simulation_run_id") or payload.get("run_id") or payload.get("id")

    if isinstance(run_id, str) and run_id:
        return run_id

    return None
