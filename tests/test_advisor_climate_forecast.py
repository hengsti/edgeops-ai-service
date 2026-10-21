import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import onnx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.forecasting import ATOL, MANIFEST_KEY, RTOL, forecast_contract
from edgeops_ai.training.advisor_climate import replay
from edgeops_ai.training.advisor_dataset import SCHEMA
from edgeops_ai.training.cli import main

START = datetime(2026, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def rewrite(history, rows):
    path = history / "device.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card_path = history / "data-card.json"
    card = json.loads(card_path.read_text())
    card["tables"]["device"] = {
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    card_path.write_text(json.dumps(card))


@pytest.fixture
def history(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    rows = []
    for minute in range(3 * 1440):
        for field, value, text in (
            ("temp_c_mean", 20 + (minute % 96) / 100, None),
            ("rel_hum_perc_mean", 65 if 20 <= minute % 96 < 40 else 50, None),
            ("warmed_up_last", None, "true"),
        ):
            rows.append(
                {
                    "time": START + minute * MINUTE,
                    "entity_id": "device-a",
                    "measurement": "bme680",
                    "field": field,
                    "value": value,
                    "text": text,
                    "dimensions": '{"room":"kitchen"}',
                    "interval_minutes": 1,
                    "split": ("train", "validation", "test")[minute // 1440],
                    "weak_labels": [],
                    "manual_label": None,
                }
            )
    (root / "data-card.json").write_text(
        json.dumps(
            {
                "schema_version": "advisor-1",
                "start": START.isoformat(),
                "stop": (START + 4320 * MINUTE).isoformat(),
                "split": {
                    "train_stop": (START + 1440 * MINUTE).isoformat(),
                    "validation_stop": (START + 2880 * MINUTE).isoformat(),
                    "shuffle": False,
                },
                "tables": {},
            }
        )
    )
    rewrite(root, rows)
    return root


@pytest.fixture
def candidate(history, tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    manifest = {
        "schema_version": "advisor-onnx-1",
        "model_version": "climate-fixture",
        "contract": forecast_contract("climate", 96, 96),
        "dataset": {
            name: hashlib.sha256((history / name).read_bytes()).hexdigest()
            for name in ("data-card.json", "device.parquet")
        },
    }
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["past_values"], ["forecast"])],
        "climate-fixture",
        [onnx.helper.make_tensor_value_info("past_values", onnx.TensorProto.FLOAT, [1, 96, 2])],
        [onnx.helper.make_tensor_value_info("forecast", onnx.TensorProto.FLOAT, [1, 96, 2])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.helper.set_model_props(model, {MANIFEST_KEY: json.dumps(manifest)})
    onnx.save(model, root / "model.onnx")
    digest = hashlib.sha256((root / "model.onnx").read_bytes()).hexdigest()
    (root / "model.onnx.sha256").write_text(digest + "\n")
    (root / "manifest.json").write_text(json.dumps({**manifest, "sha256": digest}))
    (root / "validation-report.json").write_text(
        json.dumps(
            {
                "schema_version": "advisor-validation-1",
                "model_version": "climate-fixture",
                "sources": manifest["dataset"],
                "passed": True,
                "deployment_approved": False,
                "tolerances": {"atol": ATOL, "rtol": RTOL},
            }
        )
    )
    return root


def test_replay_aligns_targets_and_reports_identical_held_out_baselines(
    history, candidate, tmp_path
):
    output = tmp_path / "climate"
    before = {path: path.read_bytes() for path in (*history.iterdir(), *candidate.iterdir())}
    report = replay(history, output, forecast_candidate=candidate)
    frames = json.loads((output / "forecasts.json").read_text())["forecasts"]
    first = frames[0]
    assert first["room"] == "kitchen"
    assert first["model_version"] == "climate-fixture"
    assert first["forecast_origin"] == (START + 96 * MINUTE).isoformat()
    assert first["last_input"] == (START + 95 * MINUTE).isoformat()
    assert first["context_start"] == START.isoformat()
    assert first["target_times"][0] == (START + 96 * MINUTE).isoformat()
    assert first["target_times"][30] == (START + 126 * MINUTE).isoformat()
    assert first["target_times"][60] == (START + 156 * MINUTE).isoformat()
    assert first["temperature_c"][30] == pytest.approx(20.3)
    assert first["humidity_percent"][30] == 65
    advice = json.loads((output / "advice.json").read_text())["findings"]
    assert advice[0]["evidence"]["minutes_to_crossing"] == 20
    assert advice[0]["evidence"]["crossing_at"] == (START + 116 * MINUTE).isoformat()
    assert advice[0]["observed_at"] == first["forecast_origin"]
    assert "cause" not in advice[0]["summary"].lower()
    assert report["forecast_available"] is True
    assert report["forecast"]["deployment_approved"] is False
    for split in ("validation", "test"):
        evaluation = report["forecast"]["evaluation"][split]
        assert evaluation["windows"] == 14
        assert evaluation["compared_windows"] == 14
        for name in ("ttm", "last_value", "seasonal"):
            assert evaluation["models"][name]["windows"] == 14
        assert evaluation["models"]["ttm"]["mae_by_horizon_channel"][30] == [0, 0]
        assert evaluation["models"]["seasonal"]["mae_by_horizon_channel"][60] == [0, 0]
        assert evaluation["models"]["ttm"]["humidity_crossings"]["true_positive"] == 14
        assert evaluation["models"]["ttm"]["humidity_crossings"]["timing_mae_minutes"] == 0
        assert evaluation["models"]["last_value"]["humidity_crossings"]["false_negative"] == 14
    assert all(path.read_bytes() == content for path, content in before.items())


def test_forecast_candidate_cli_is_optional_and_supported(history, candidate, tmp_path):
    assert (
        main(
            [
                "advisor-climate",
                "--input-dir",
                str(history),
                "--output-dir",
                str(tmp_path / "cli"),
                "--forecast-candidate",
                str(candidate),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("problem", ["gap", "missing", "warmup", "clock", "room", "reset"])
def test_forecast_context_does_not_cross_quality_room_or_split_boundaries(history, problem):
    from edgeops_ai.training.advisor_forecast import load_windows

    at = START + 300 * MINUTE
    rows = pq.read_table(history / "device.parquet").to_pylist()
    if problem == "gap":
        rows = [row for row in rows if row["time"] != at]
    elif problem == "missing":
        rows = [row for row in rows if not (row["time"] == at and row["field"] == "temp_c_mean")]
    elif problem == "warmup":
        for row in rows:
            if row["time"] == at and row["field"] == "warmed_up_last":
                row["text"] = "false"
    elif problem == "room":
        for row in rows:
            if row["time"] >= at:
                row["dimensions"] = '{"room":"office"}'
    else:
        row = next(row.copy() for row in rows if row["time"] == at)
        row.update(
            field="time_valid_last" if problem == "clock" else "uptime_resets",
            value=None if problem == "clock" else 1,
            text="false" if problem == "clock" else None,
            measurement="bme680" if problem == "clock" else "device_status",
        )
        rows.append(row)
    rewrite(history, rows)
    windows, _ = load_windows(history, forecast_contract("climate", 96, 96))
    assert len(windows["train"]) < 14
    for index, split in enumerate(("train", "validation", "test")):
        for window in windows[split]:
            assert window["times"][0] >= START + index * 1440 * MINUTE
            assert window["target_times"][-1] < START + (index + 1) * 1440 * MINUTE
            assert not window["times"][0] < at <= window["target_times"][-1]


def test_device_status_clock_does_not_duplicate_the_sensor_clock(history):
    from edgeops_ai.training.advisor_forecast import load_windows

    rows = pq.read_table(history / "device.parquet").to_pylist()
    for measurement in ("bme680", "device_status"):
        row = rows[0].copy()
        row.update(measurement=measurement, field="time_valid_last", value=None, text="true")
        rows.append(row)
    rewrite(history, rows)
    windows, _ = load_windows(history, forecast_contract("climate", 96, 96))
    assert len(windows["train"]) == 14


def test_daily_seasonal_fit_is_unchanged_by_held_out_observations(history, candidate):
    from edgeops_ai.forecasting import ForecastModel
    from edgeops_ai.training.advisor_climate_forecast import score_windows
    from edgeops_ai.training.advisor_forecast import load_windows

    model = ForecastModel(
        candidate / "model.onnx", (candidate / "model.onnx.sha256").read_text().strip()
    )
    windows, _ = load_windows(history, model.contract)
    _, _, before = score_windows(model, windows)
    changed = deepcopy(windows)
    for split in ("validation", "test"):
        for window in changed[split]:
            window["future"] += 2
            window["past"] += 2
    _, _, after = score_windows(model, changed)
    assert before["seasonal_profile_sha256"] == after["seasonal_profile_sha256"]
    assert before["seasonal_training_samples"] == after["seasonal_training_samples"] == 1440
    assert after["evaluation"]["test"]["models"]["seasonal"]["mae_by_horizon_channel"][30] == [2, 2]


def test_crossing_evaluation_counts_false_alarms_misses_and_timing():
    import numpy as np

    from edgeops_ai.training.advisor_climate_forecast import metrics

    low = np.full((61, 2), [20, 50.0])
    observed = low.copy()
    observed[20:, 1] = 65
    predicted = low.copy()
    predicted[25:, 1] = 65
    result = metrics(
        [
            (observed, predicted, 50),
            (low, predicted, 50),
            (observed, low, 50),
            (low, low, 50),
            (observed, predicted, 65),
        ]
    )
    assert result["humidity_crossings"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "eligible_windows": 4,
        "timing_mae_minutes": 5,
    }


@pytest.mark.parametrize("problem", ["parity", "manifest", "sha256", "dataset", "short-horizon"])
def test_unvalidated_candidate_never_publishes_forecast_advice(
    history, candidate, tmp_path, problem
):
    if problem == "parity":
        path = candidate / "validation-report.json"
        data = json.loads(path.read_text())
        data["passed"] = False
        path.write_text(json.dumps(data))
    elif problem == "manifest":
        path = candidate / "manifest.json"
        data = json.loads(path.read_text())
        data["model_version"] = "other"
        path.write_text(json.dumps(data))
    elif problem == "sha256":
        (candidate / "model.onnx.sha256").write_text("0" * 64)
    elif problem == "dataset":
        path = history / "data-card.json"
        path.write_text(path.read_text() + " ")
    else:
        # A graph/manifest mismatch is rejected even if the external digest was updated.
        path = candidate / "model.onnx"
        model = onnx.load(path)
        manifest = json.loads(model.metadata_props[0].value)
        manifest["contract"] = forecast_contract("climate", 96, 30)
        onnx.helper.set_model_props(model, {MANIFEST_KEY: json.dumps(manifest)})
        onnx.save(model, path)
        (candidate / "model.onnx.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        replay(history, output, forecast_candidate=candidate)
    assert not output.exists()


def test_nonphysical_forecast_is_not_published(history, candidate, tmp_path):
    path = candidate / "model.onnx"
    model = onnx.load(path)
    model.graph.node[0].output[0] = "unchanged"
    model.graph.initializer.append(
        onnx.helper.make_tensor("bias", onnx.TensorProto.FLOAT, [2], [0, 100])
    )
    model.graph.node.append(onnx.helper.make_node("Add", ["unchanged", "bias"], ["forecast"]))
    onnx.save(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (candidate / "model.onnx.sha256").write_text(digest)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "nonphysical"
    with pytest.raises(ValueError, match="humidity"):
        replay(history, output, forecast_candidate=candidate)
    assert not output.exists()
