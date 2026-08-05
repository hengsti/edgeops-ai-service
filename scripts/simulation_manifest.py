"""Validation and atomic persistence for simulation contract-v2 manifests."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.common import DATASET_SCHEMA_VERSION, SCENARIOS

SIMULATION_CONTRACT_VERSION = 2
SIMULATION_MANIFEST_NAME = "simulation-manifest.json"
SIMULATION_VARIANTS = ("mild", "degraded", "severe")
OPTIONAL_METRIC_GROUPS = (
    "decode/validation",
    "input/WAL pressure",
    "DLQ",
    "WAL/Influx",
    "stage latency",
)

REQUIRED_RUN_FIELDS = frozenset(
    {
        "scenario_id",
        "seed",
        "variant_id",
        "speed",
        "resolved_effects",
        "workload_scale",
        "unavailable_metric_groups",
        "collector_version",
        "csv_path",
    }
)


class SimulationManifestError(ValueError):
    """A simulation manifest is missing, stale, or unsafe."""


def manifest_path(output_dir: Path) -> Path:
    return output_dir / SIMULATION_MANIFEST_NAME


def empty_manifest() -> dict[str, Any]:
    return {
        "simulation_contract_version": SIMULATION_CONTRACT_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "runs": {},
    }


def _require_finite_number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SimulationManifestError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SimulationManifestError(f"{name} must be finite")
    return parsed


def _safe_csv_path(output_dir: Path, value: Any, *, run_id: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise SimulationManifestError(f"manifest run {run_id!r} has an invalid csv_path")
    relative = Path(value)
    if relative.is_absolute():
        raise SimulationManifestError(f"manifest run {run_id!r} csv_path must be relative")
    root = output_dir.resolve()
    resolved = (output_dir / relative).resolve()
    if not resolved.is_relative_to(root):
        raise SimulationManifestError(f"manifest run {run_id!r} csv_path escapes output directory")
    return relative.as_posix(), resolved


def validate_manifest(
    payload: Any,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SimulationManifestError("simulation manifest must be a JSON object")
    if payload.get("simulation_contract_version") != SIMULATION_CONTRACT_VERSION:
        raise SimulationManifestError("simulation manifest contract version must be integer 2")
    if payload.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise SimulationManifestError(
            f"simulation manifest dataset schema must be {DATASET_SCHEMA_VERSION!r}"
        )
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        raise SimulationManifestError("simulation manifest runs must be a JSON object")

    validated_runs: dict[str, dict[str, Any]] = {}
    csv_owners: dict[str, str] = {}
    for run_id, raw_entry in runs.items():
        if not isinstance(run_id, str) or not run_id.strip():
            raise SimulationManifestError("simulation manifest run IDs must be non-empty strings")
        if not isinstance(raw_entry, dict):
            raise SimulationManifestError(f"manifest run {run_id!r} must be a JSON object")
        missing = sorted(REQUIRED_RUN_FIELDS - set(raw_entry))
        if missing:
            raise SimulationManifestError(
                f"manifest run {run_id!r} is missing fields: {', '.join(missing)}"
            )

        scenario = raw_entry["scenario_id"]
        if scenario not in SCENARIOS:
            raise SimulationManifestError(f"manifest run {run_id!r} has invalid scenario_id")
        seed = raw_entry["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SimulationManifestError(f"manifest run {run_id!r} seed must be an integer")
        variant = raw_entry["variant_id"]
        if variant not in SIMULATION_VARIANTS:
            raise SimulationManifestError(f"manifest run {run_id!r} has invalid variant_id")
        speed = _require_finite_number(raw_entry["speed"], name=f"run {run_id} speed")
        if speed <= 0:
            raise SimulationManifestError(f"manifest run {run_id!r} speed must be positive")
        workload_scale = _require_finite_number(
            raw_entry["workload_scale"], name=f"run {run_id} workload_scale"
        )
        if not 0.65 <= workload_scale <= 1.50:
            raise SimulationManifestError(
                f"manifest run {run_id!r} workload_scale must be in [0.65, 1.50]"
            )
        if not isinstance(raw_entry["resolved_effects"], dict):
            raise SimulationManifestError(
                f"manifest run {run_id!r} resolved_effects must be an object"
            )
        unavailable = raw_entry["unavailable_metric_groups"]
        if (
            not isinstance(unavailable, list)
            or unavailable != sorted(set(unavailable))
            or len(unavailable) > 3
            or any(group not in OPTIONAL_METRIC_GROUPS for group in unavailable)
        ):
            raise SimulationManifestError(
                f"manifest run {run_id!r} has invalid unavailable_metric_groups"
            )
        collector_version = raw_entry["collector_version"]
        if not isinstance(collector_version, str) or not collector_version.strip():
            raise SimulationManifestError(f"manifest run {run_id!r} collector_version is invalid")
        relative_csv, _ = _safe_csv_path(output_dir, raw_entry["csv_path"], run_id=run_id)
        if relative_csv in csv_owners:
            raise SimulationManifestError(
                f"manifest CSV {relative_csv!r} is referenced by multiple run IDs"
            )
        csv_owners[relative_csv] = run_id

        entry = dict(raw_entry)
        entry.update(
            {
                "scenario_id": scenario,
                "seed": seed,
                "variant_id": variant,
                "speed": speed,
                "workload_scale": workload_scale,
                "unavailable_metric_groups": list(unavailable),
                "csv_path": relative_csv,
            }
        )
        validated_runs[run_id] = entry

    return {
        "simulation_contract_version": SIMULATION_CONTRACT_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "runs": validated_runs,
    }


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = manifest_path(output_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SimulationManifestError(f"Unable to read simulation manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise SimulationManifestError(f"Invalid JSON simulation manifest: {path}") from error
    return validate_manifest(payload, output_dir=output_dir)


def try_load_manifest(output_dir: Path) -> dict[str, Any] | None:
    try:
        return load_manifest(output_dir)
    except SimulationManifestError:
        return None


def entry_for_csv(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    csv_path: Path,
) -> tuple[str, dict[str, Any]] | None:
    relative = csv_path.resolve().relative_to(output_dir.resolve()).as_posix()
    matches = [
        (run_id, entry)
        for run_id, entry in manifest["runs"].items()
        if entry["csv_path"] == relative
    ]
    return matches[0] if len(matches) == 1 else None


def write_manifest_atomic(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    validated = validate_manifest(dict(manifest), output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = manifest_path(output_dir)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def upsert_manifest_run(
    manifest: Mapping[str, Any] | None,
    *,
    output_dir: Path,
    run_id: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    updated = empty_manifest() if manifest is None else json.loads(json.dumps(manifest))
    runs = updated["runs"]
    stale_ids = [
        existing_run_id
        for existing_run_id, existing in runs.items()
        if existing.get("csv_path") == entry.get("csv_path") and existing_run_id != run_id
    ]
    for stale_id in stale_ids:
        del runs[stale_id]
    runs[run_id] = dict(entry)
    return validate_manifest(updated, output_dir=output_dir)


def validate_manifest_datasets(
    output_dir: Path,
    *,
    pattern: str = "*-seed-*.csv",
) -> tuple[dict[str, Any], dict[str, tuple[str, dict[str, Any]]]]:
    """Join manifest entries to CSVs and validate row-level provenance."""

    manifest = load_manifest(output_dir)
    paths = tuple(sorted(output_dir.glob(pattern)))
    if not paths:
        raise SimulationManifestError(f"No {pattern} files found in {output_dir}")

    joined: dict[str, tuple[str, dict[str, Any]]] = {}
    seen_run_ids: set[str] = set()
    for path in paths:
        match = entry_for_csv(manifest, output_dir=output_dir, csv_path=path)
        if match is None:
            raise SimulationManifestError(
                f"CSV {path.name!r} has no unique simulation manifest entry"
            )
        run_id, entry = match
        if run_id in seen_run_ids:
            raise SimulationManifestError(f"run ID {run_id!r} is used by multiple CSV files")
        seen_run_ids.add(run_id)

        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                row_count = 0
                for line_number, row in enumerate(reader, start=2):
                    row_count += 1
                    row_run_id = (row.get("simulation_run_id") or "").strip()
                    row_scenario = (row.get("scenario_id") or "").strip().lower()
                    try:
                        row_seed = int(row.get("simulation_seed") or "")
                    except (TypeError, ValueError) as error:
                        raise SimulationManifestError(
                            f"{path}:{line_number}: simulation_seed is invalid"
                        ) from error
                    if row_run_id != run_id:
                        raise SimulationManifestError(
                            f"{path}:{line_number}: unknown or mismatched simulation_run_id"
                        )
                    if row_scenario != entry["scenario_id"] or row_seed != entry["seed"]:
                        raise SimulationManifestError(
                            f"{path}:{line_number}: scenario or seed mismatches manifest"
                        )
        except OSError as error:
            raise SimulationManifestError(f"Unable to read manifest CSV {path}") from error
        if row_count == 0:
            raise SimulationManifestError(f"Manifest CSV is empty: {path}")
        joined[path.name] = (run_id, entry)

    referenced = {
        entry["csv_path"]
        for entry in manifest["runs"].values()
        if Path(entry["csv_path"]).match(pattern)
    }
    selected = {path.resolve().relative_to(output_dir.resolve()).as_posix() for path in paths}
    if referenced != selected:
        raise SimulationManifestError(
            "simulation manifest contains missing or unselected dataset CSV entries"
        )
    return manifest, joined
