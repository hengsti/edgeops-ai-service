import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from edgeops_ai.ml import load_ml_predictor
from edgeops_ai.training.cli import COMMANDS, main
from edgeops_ai.training.common import DATASET_SCHEMA_VERSION, SCENARIOS
from edgeops_ai.training.dataset import write_dataset
from edgeops_ai.training.manifest import empty_manifest, upsert_manifest_run, write_manifest_atomic


@pytest.fixture
def training_data(tmp_path, make_features):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    faults = {
        "ingestion-backpressure": {"processing_ratio": 0.5, "queue_growth_rate": 5},
        "missing-device-heartbeat": {"heartbeat_missing": True},
        "malformed-sensor-payload": {"transform_failure_ratio": 0.5},
        "influxdb-write-failure": {"influx_write_failure_ratio": 0.5, "influxdb_healthy": 0},
        "mqtt-broker-unavailable": {"broker_available": False, "heartbeat_missing": True},
    }
    manifest = empty_manifest()
    for seed, variant in ((42, "mild"), (43, "degraded"), (44, "severe")):
        for scenario in SCENARIOS:
            run_id = f"{scenario}-{seed}"
            output = dataset_dir / f"{scenario}-seed-{seed}.csv"
            rows = []
            for sample in range(10):
                fault = 4 <= sample < 8
                label = scenario if fault else "normal"
                features = make_features(
                    **{"broker_available": True, **(faults[scenario] if fault else {})}
                )
                rows.append(
                    {
                        **features.model_dump(mode="json"),
                        "dataset_schema_version": DATASET_SCHEMA_VERSION,
                        "captured_at": (
                            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=sample)
                        ).isoformat(),
                        "source_host": "fixture",
                        "scenario_id": scenario,
                        "simulation_seed": seed,
                        "simulation_run_id": run_id,
                        "simulation_phase": "fault"
                        if fault
                        else ("recovery" if sample >= 8 else "normal"),
                        "expected_label": label,
                        "predicted_label": label,
                        "prediction_correct": True,
                        "model_version": "fixture-rules",
                    }
                )
            write_dataset(rows, output, append=False)
            manifest = upsert_manifest_run(
                manifest,
                output_dir=dataset_dir,
                run_id=run_id,
                entry={
                    "scenario_id": scenario,
                    "seed": seed,
                    "variant_id": variant,
                    "speed": 1.0,
                    "resolved_effects": {},
                    "workload_scale": 1.0,
                    "unavailable_metric_groups": [],
                    "collector_version": "fixture",
                    "csv_path": output.name,
                },
            )
    write_manifest_atomic(dataset_dir, manifest)
    return dataset_dir


def test_training_pipeline_cli_smoke(tmp_path, make_features, training_data):
    """Exercise real CSV validation, evaluation, training and artifact loading offline."""
    dataset_dir = training_data
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "edgeops_ai.training",
            "pipeline",
            "--output-dir",
            str(dataset_dir),
            "--train",
            "--model-version",
            "cli-smoke-model",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    artifact = next((tmp_path / "artifacts").rglob("model-*.joblib"))
    metadata = json.loads((artifact.parent / "manifest.json").read_text())
    assert metadata["model_version"] == "cli-smoke-model"
    assert metadata["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert (
        load_ml_predictor(artifact).predict(make_features(broker_available=True)).label == "normal"
    )
    assert json.loads((dataset_dir / "training-report.json").read_text())["quality_gate"]["passed"]
    assert json.loads((dataset_dir / "rule-report.json").read_text())["dataset"]["runs"] == 15


@pytest.mark.parametrize("command", COMMANDS)
def test_training_subcommand_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_pipeline_propagates_training_gate_failure(training_data, tmp_path, make_features):
    # Identical features for every label cannot meet the real training quality gate.
    for path in training_data.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.update(make_features().model_dump(mode="json"))
        write_dataset(rows, path)
    artifact = tmp_path / "candidate.joblib"
    artifact.write_bytes(b"previous candidate")
    assert (
        main(
            [
                "pipeline",
                "--output-dir",
                str(training_data),
                "--train",
                "--model-artifact",
                str(artifact),
            ]
        )
        == 1
    )
    assert artifact.read_bytes() == b"previous candidate"
    assert not (artifact.parent / "manifest.json").exists()
    assert not json.loads((training_data / "training-report.json").read_text())["quality_gate"][
        "passed"
    ]


def test_pipeline_generates_in_process_with_explicit_simulation_settings(
    training_data, monkeypatch
):
    from edgeops_ai.training import pipeline
    from edgeops_ai.training.dataset import CollectionResult
    from edgeops_ai.training.manifest import load_manifest

    output = training_data / "ingestion-backpressure-seed-42.csv"
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = load_manifest(training_data)
    run_id = rows[0]["simulation_run_id"]
    run = manifest["runs"][run_id]
    monkeypatch.setattr(pipeline, "fetch_collector_version", lambda **kwargs: "fixture")
    monkeypatch.setattr(
        pipeline,
        "start_simulation",
        lambda **kwargs: {
            **run,
            "run_id": run_id,
            "simulation_contract_version": 2,
        },
    )

    async def collect(**kwargs):
        settings = kwargs["settings"]
        assert settings.collector_base_url == "http://isolated-simulation:8097"
        assert settings.collector_api_key == "simulation-key"
        assert settings.device_ids == ("fixture-device",)
        assert kwargs["simulation_seed"] == 42
        return CollectionResult(rows, 11, 1, "scenario-run-ended", run_id)

    monkeypatch.setattr(pipeline, "collect_dataset", collect)
    args = pipeline.parse_args(
        [
            "--output-dir",
            str(training_data),
            "--simulation-base-url",
            "http://isolated-simulation:8097",
            "--simulation-read-key",
            "simulation-key",
            "--simulation-device-id",
            "fixture-device",
        ]
    )
    pipeline.generate_dataset(
        args=args, scenario="ingestion-backpressure", seed=42, variant_id="mild", output=output
    )
    assert pipeline.dataset_is_complete(
        output,
        scenario="ingestion-backpressure",
        seed=42,
        variant_id="mild",
        manifest=load_manifest(training_data),
        output_dir=training_data,
    )
    assert not output.with_suffix(".tmp.csv").exists()


def test_release_cli_combines_holdout_and_training_gates(training_data, tmp_path):
    from edgeops_ai.training.evaluate import evaluate_dataset
    from edgeops_ai.training.manifest import load_manifest
    from edgeops_ai.training.reports import write_report

    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    manifest = empty_manifest()
    for run_id, entry in load_manifest(training_data)["runs"].items():
        seed = entry["seed"] + 20
        output = holdout_dir / f"{entry['scenario_id']}-seed-{seed}.csv"
        with (training_data / entry["csv_path"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.update(
                simulation_seed=seed,
                simulation_run_id=f"holdout-{run_id}",
                model_version="holdout-model",
            )
        write_dataset(rows, output)
        manifest = upsert_manifest_run(
            manifest,
            output_dir=holdout_dir,
            run_id=f"holdout-{run_id}",
            entry={**entry, "seed": seed, "csv_path": output.name},
        )
    write_manifest_atomic(holdout_dir, manifest)
    evaluate_dataset(
        input_dir=holdout_dir, output=holdout_dir / "candidate-report.json", seeds=(62, 63, 64)
    )
    write_report(
        {"quality_gate": {"passed": True}, "dataset": {"simulation_seeds": [42, 43, 44]}},
        training_data / "training-report.json",
    )
    write_report(
        {"recovery": {"runs": [{"cleared_during_observation": True}]}},
        training_data / "rule-report.json",
    )
    write_report(
        {
            "dataset": {"stable_model_matrix_is_finite": True},
            "optional_group_availability": {"overall": {"available_ratio": 0.85}},
            "constant_feature_counts_by_scenario": dict.fromkeys(SCENARIOS, 12),
        },
        training_data / "feature-audit.json",
    )
    ablation = {
        "feature_sets": {
            "a1": {
                "candidates": {
                    "tree": {"macro_f1": 0.8, "per_seed": [{"accuracy": 0.90, "macro_f1": 0.88}]}
                }
            },
            "full": {
                "candidates": {
                    "tree": {"macro_f1": 0.84, "per_seed": [{"accuracy": 0.92, "macro_f1": 0.90}]}
                }
            },
        }
    }
    write_report(ablation, training_data / "ablation-report.json")
    argv = [
        "validate",
        "--input-dir",
        str(holdout_dir),
        "--training-dir",
        str(training_data),
        "--expected-model-version",
        "holdout-model",
    ]
    assert main(argv) == 0
    result = json.loads((holdout_dir / "release-validation.json").read_text())
    assert len(result["gates"]) == 18
    assert result["passed"]

    ablation["feature_sets"]["full"]["candidates"]["tree"]["per_seed"] = [
        {"accuracy": 1.0, "macro_f1": 1.0}
    ]
    write_report(ablation, training_data / "ablation-report.json")
    assert main(argv) == 1
    result = json.loads((holdout_dir / "release-validation.json").read_text())
    assert not result["passed"]
    assert any(
        not gate["passed"] and gate["name"].startswith("training-") for gate in result["gates"]
    )
