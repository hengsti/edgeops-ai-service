import json
from pathlib import Path

import pytest
from scripts.common import ScriptError
from scripts.run_pipeline import (
    DEFAULT_SIMULATION_BASE_URL,
    assigned_variants,
    parse_args,
    simulation_runs_url,
    validate_started_run,
)
from scripts.simulation_manifest import (
    SimulationManifestError,
    empty_manifest,
    load_manifest,
    manifest_path,
    upsert_manifest_run,
    write_manifest_atomic,
)

from edgeops_ai.settings import Settings


def run_entry(csv_path: str = "ingestion-backpressure-seed-42.csv") -> dict[str, object]:
    return {
        "scenario_id": "ingestion-backpressure",
        "seed": 42,
        "variant_id": "mild",
        "speed": 100.0,
        "resolved_effects": {"queue": {"high_watermark": 0.8}},
        "workload_scale": 1.1,
        "unavailable_metric_groups": ["DLQ"],
        "collector_version": "0.1.0",
        "csv_path": csv_path,
    }


def test_simulation_defaults_are_isolated_from_production() -> None:
    args = parse_args([])

    assert DEFAULT_SIMULATION_BASE_URL == "http://localhost:8097"
    assert args.simulation_base_url == "http://localhost:8097"
    assert simulation_runs_url(args.simulation_base_url).endswith("/v1/simulation/runs")
    assert Settings().collector_base_url == "http://localhost:8095"


def test_variants_rotate_by_ordered_unique_seed() -> None:
    assert assigned_variants([51, 12, 51, 99, 7]) == {
        51: "mild",
        12: "degraded",
        99: "severe",
        7: "mild",
    }


def test_manifest_round_trip_is_atomic_and_contract_v2(tmp_path: Path) -> None:
    manifest = upsert_manifest_run(
        empty_manifest(),
        output_dir=tmp_path,
        run_id="run-42",
        entry=run_entry(),
    )

    destination = write_manifest_atomic(tmp_path, manifest)

    assert destination == manifest_path(tmp_path)
    assert not destination.with_suffix(".json.tmp").exists()
    assert load_manifest(tmp_path) == manifest


@pytest.mark.parametrize(
    "mutation",
    [
        {"simulation_contract_version": 1},
        {"dataset_schema_version": "1.0"},
        {"runs": []},
    ],
)
def test_manifest_rejects_stale_or_malformed_top_level(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    payload = empty_manifest()
    payload.update(mutation)
    manifest_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SimulationManifestError):
        load_manifest(tmp_path)


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(SimulationManifestError, match="escapes"):
        upsert_manifest_run(
            empty_manifest(),
            output_dir=tmp_path,
            run_id="run-42",
            entry=run_entry("../outside.csv"),
        )


def test_started_run_requires_contract_v2_and_exact_identity() -> None:
    response = {
        "run_id": "run-42",
        "scenario_id": "ingestion-backpressure",
        "seed": 42,
        "speed": 100.0,
        "variant_id": "mild",
        "simulation_contract_version": 2,
        "resolved_effects": {},
        "workload_scale": 1.0,
        "unavailable_metric_groups": [],
    }

    assert (
        validate_started_run(
            response,
            scenario="ingestion-backpressure",
            seed=42,
            variant_id="mild",
            speed=100.0,
        )[0]
        == "run-42"
    )

    response["simulation_contract_version"] = 1
    with pytest.raises(ScriptError, match="contract version 2"):
        validate_started_run(
            response,
            scenario="ingestion-backpressure",
            seed=42,
            variant_id="mild",
            speed=100.0,
        )
