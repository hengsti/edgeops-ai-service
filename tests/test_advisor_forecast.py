import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.training.advisor_dataset import SCHEMA
from edgeops_ai.training.cli import main

START = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def history(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    rows = []
    for minute in range(144):
        for field, value, text in (
            ("temp_c_mean", 20 + minute / 100, None),
            ("rel_hum_perc_mean", 50 + np.sin(minute / 10), None),
            ("warmed_up_last", None, "true"),
        ):
            rows.append(
                {
                    "time": START + timedelta(minutes=minute),
                    "entity_id": "device-a",
                    "measurement": "bme680",
                    "field": field,
                    "value": value,
                    "text": text,
                    "dimensions": '{"room":"kitchen"}',
                    "interval_minutes": 1,
                    "split": ("train", "validation", "test")[minute // 48],
                    "weak_labels": [],
                    "manual_label": None,
                }
            )
    path = root / "device.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card = {
        "schema_version": "advisor-1",
        "start": START.isoformat(),
        "stop": (START + timedelta(minutes=144)).isoformat(),
        "split": {
            "train_stop": (START + timedelta(minutes=48)).isoformat(),
            "validation_stop": (START + timedelta(minutes=96)).isoformat(),
            "shuffle": False,
        },
        "tables": {
            "device": {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        },
    }
    (root / "data-card.json").write_text(json.dumps(card))
    return root


def test_forecast_windows_stay_inside_splits_and_use_shared_preprocessing(history):
    from edgeops_ai.forecasting import forecast_contract, prepare_input
    from edgeops_ai.training.advisor_forecast import load_windows

    contract = forecast_contract("climate", 16, 8)
    windows, sources = load_windows(history, contract)
    assert (
        sources["device.parquet"]
        == hashlib.sha256((history / "device.parquet").read_bytes()).hexdigest()
    )
    for index, split in enumerate(("train", "validation", "test")):
        assert len(windows[split]) == 4
        for window in windows[split]:
            assert window["times"][0] >= START + timedelta(minutes=index * 48)
            assert window["target_times"][-1] < START + timedelta(minutes=(index + 1) * 48)
            assert np.array_equal(
                window["past"], prepare_input(window["past"][0], window["times"], contract)
            )
            assert window["room"] == "kitchen"


@pytest.mark.parametrize(
    "problem", ["gap", "warmup", "room", "split", "duplicate", "nan", "overflow", "hash"]
)
def test_forecast_rejects_or_excludes_bad_history(history, problem):
    from edgeops_ai.forecasting import forecast_contract
    from edgeops_ai.training.advisor_forecast import load_windows

    path = history / "device.parquet"
    rows = pq.read_table(path).to_pylist()
    if problem == "gap":
        rows = [row for row in rows if row["time"] != START + timedelta(minutes=24)]
    elif problem == "warmup":
        for row in rows:
            if row["field"] == "warmed_up_last" and row["time"] == START + timedelta(minutes=24):
                row["text"] = "false"
    elif problem == "room":
        for row in rows:
            if row["time"] >= START + timedelta(minutes=24):
                row["dimensions"] = '{"room":"hall"}'
    elif problem == "split":
        rows[0]["split"] = "test"
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "nan":
        rows[0]["value"] = float("nan")
    elif problem == "overflow":
        rows[-3]["value"] = 1e300
    else:
        rows[0]["value"] = 123
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card_path = history / "data-card.json"
    card = json.loads(card_path.read_text())
    if problem != "hash":
        card["tables"]["device"] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    card_path.write_text(json.dumps(card))
    if problem in {"split", "duplicate", "nan", "overflow", "hash"}:
        with pytest.raises(ValueError):
            load_windows(history, forecast_contract("climate", 16, 8))
    else:
        windows, _ = load_windows(history, forecast_contract("climate", 16, 8))
        assert len(windows["train"]) < 4
        assert all(
            not (w["times"][0] < START + timedelta(minutes=24) < w["target_times"][-1])
            for w in windows["train"]
        )


def test_forecast_contract_rejects_missing_values_and_irregular_times():
    from edgeops_ai.forecasting import forecast_contract, prepare_input

    contract = forecast_contract("climate", 16, 8)
    times = [START + timedelta(minutes=m) for m in range(16)]
    values = np.ones((16, 2), dtype=np.float32)
    assert prepare_input(values, times, contract).shape == (1, 16, 2)
    for bad_values, bad_times in (
        (values[:15], times),
        (values * np.nan, times),
        (values, times[::-1]),
        (values, [t.replace(tzinfo=None) for t in times]),
    ):
        with pytest.raises(ValueError):
            prepare_input(bad_values, bad_times, contract)


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tsfm_public")
    from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction

    torch.manual_seed(7)
    model = TinyTimeMixerForPrediction(
        TinyTimeMixerConfig(
            context_length=16,
            prediction_length=8,
            num_input_channels=2,
            patch_length=4,
            patch_stride=4,
            d_model=8,
            num_layers=1,
            decoder_num_layers=1,
            decoder_d_model=8,
            dropout=0,
            head_dropout=0,
        )
    )
    path = tmp_path / "checkpoint"
    model.save_pretrained(path)
    return path


def test_native_training_export_only_and_onnx_inference_smoke(history, checkpoint, tmp_path):
    from edgeops_ai.forecasting import ForecastModel

    digest = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    common = [
        "advisor-forecast",
        "--input-dir",
        str(history),
        "--checkpoint",
        str(checkpoint),
        "--sha256",
        digest,
        "--profile",
        "climate",
        "--model-version",
        "fixture-ttm",
    ]
    output = tmp_path / "candidate"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "edgeops_ai.training",
            *common,
            "--output-dir",
            str(output),
            "--epochs",
            "1",
        ],
        env={
            **os.environ,
            "PYTHONIOENCODING": "ascii",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "validation-report.json").read_text())
    assert report["passed"] is True
    assert report["deployment_approved"] is False
    assert report["splits"]["test"]["windows"] == 4
    assert report["training"]["epochs"] == 1
    assert (output / "checkpoint" / "model.safetensors").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    predictor = ForecastModel(output / "model.onnx", manifest["sha256"])
    times = [START + timedelta(minutes=m) for m in range(16)]
    assert predictor.predict(np.ones((16, 2)), times).shape == (1, 8, 2)
    exported = tmp_path / "export-only"
    assert main([*common, "--output-dir", str(exported), "--export-only"]) == 0
    assert {p.name: p.read_bytes() for p in checkpoint.iterdir()} == before
    assert not (exported / "checkpoint").exists()
    assert main([*common, "--output-dir", str(exported), "--export-only"]) == 1
    with pytest.raises(ValueError, match="SHA-256"):
        ForecastModel(output / "model.onnx", "0" * 64)
    resumed = tmp_path / "resumed"
    retained = output / "checkpoint"
    assert (
        main(
            [
                *common,
                "--checkpoint",
                str(retained),
                "--sha256",
                hashlib.sha256((retained / "model.safetensors").read_bytes()).hexdigest(),
                "--output-dir",
                str(resumed),
                "--epochs",
                "1",
            ]
        )
        == 0
    )
    assert (resumed / "checkpoint" / "model.safetensors").exists()
    import onnx

    from edgeops_ai.forecasting import MANIFEST_KEY

    graph = onnx.load(output / "model.onnx")
    embedded = json.loads(next(p.value for p in graph.metadata_props if p.key == MANIFEST_KEY))
    embedded["contract"]["input"]["name"] = "incompatible-input"
    onnx.helper.set_model_props(graph, {MANIFEST_KEY: json.dumps(embedded)})
    bad = tmp_path / "incompatible.onnx"
    onnx.save(graph, bad)
    with pytest.raises(ValueError, match="contract"):
        ForecastModel(bad, hashlib.sha256(bad.read_bytes()).hexdigest())


def test_host_cpu_windows_preserve_five_minute_cadence(history):
    from edgeops_ai.forecasting import forecast_contract
    from edgeops_ai.training.advisor_forecast import load_windows

    rows = pq.read_table(history / "device.parquet").to_pylist()
    rows = [
        {
            **row,
            "time": START + 5 * (row["time"] - START),
            "field": "usage_active_mean",
            "measurement": "cpu",
            "interval_minutes": 5,
            "dimensions": '{"cpu":"cpu-total"}',
        }
        for row in rows
        if row["field"] == "temp_c_mean"
    ]
    path = history / "host.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card_path = history / "data-card.json"
    card = json.loads(card_path.read_text())
    card["stop"] = (START + timedelta(minutes=720)).isoformat()
    card["split"].update(
        train_stop=(START + timedelta(minutes=240)).isoformat(),
        validation_stop=(START + timedelta(minutes=480)).isoformat(),
    )
    card["tables"]["host"] = {
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    card_path.write_text(json.dumps(card))
    windows, _ = load_windows(history, forecast_contract("host-cpu", 16, 8))
    assert [len(windows[split]) for split in windows] == [4, 4, 4]
    assert windows["test"][0]["past"].shape == (1, 16, 1)
    assert windows["test"][0]["target_times"][0] - windows["test"][0]["times"][-1] == timedelta(
        minutes=5
    )


def test_parity_failure_returns_nonzero_without_publishing(
    history, checkpoint, tmp_path, monkeypatch
):
    from edgeops_ai.forecasting import ForecastModel

    original = ForecastModel.predict
    monkeypatch.setattr(ForecastModel, "predict", lambda self, *args: original(self, *args) + 1)
    output = tmp_path / "bad-parity"
    assert (
        main(
            [
                "advisor-forecast",
                "--input-dir",
                str(history),
                "--checkpoint",
                str(checkpoint),
                "--sha256",
                hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest(),
                "--profile",
                "climate",
                "--model-version",
                "bad",
                "--output-dir",
                str(output),
                "--export-only",
            ]
        )
        == 1
    )
    assert not output.exists()
