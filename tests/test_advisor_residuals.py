import csv
import hashlib
import json
from datetime import timedelta

import onnx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.forecasting import ATOL, MANIFEST_KEY, RTOL, forecast_contract
from edgeops_ai.training.advisor_dataset import MANUAL_COLUMNS, SCHEMA
from edgeops_ai.training.advisor_health import replay
from edgeops_ai.training.cli import main
from test_advisor_forecast import START

pytest_plugins = ["test_advisor_forecast"]


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
def candidate(history, tmp_path):
    rows = pq.read_table(history / "device.parquet").to_pylist()
    for row in rows:
        if row["field"] == "temp_c_mean":
            row["value"] = (
                25
                if START + timedelta(minutes=64) <= row["time"] < START + timedelta(minutes=80)
                else 20
            )
        elif row["field"] == "rel_hum_perc_mean":
            row["value"] = 50
    rewrite(history, rows)
    root = tmp_path / "candidate"
    root.mkdir()
    manifest = {
        "schema_version": "advisor-onnx-1",
        "model_version": "residual-fixture",
        "contract": forecast_contract("climate", 16, 16),
        "dataset": {
            name: hashlib.sha256((history / name).read_bytes()).hexdigest()
            for name in ("data-card.json", "device.parquet")
        },
    }
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["past_values"], ["forecast"])],
        "residual-fixture",
        [onnx.helper.make_tensor_value_info("past_values", onnx.TensorProto.FLOAT, [1, 16, 2])],
        [onnx.helper.make_tensor_value_info("forecast", onnx.TensorProto.FLOAT, [1, 16, 2])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.helper.set_model_props(model, {MANIFEST_KEY: json.dumps(manifest)})
    onnx.save(model, root / "model.onnx")
    digest = hashlib.sha256((root / "model.onnx").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({**manifest, "sha256": digest}))
    (root / "model.onnx.sha256").write_text(digest + "\n")
    (root / "validation-report.json").write_text(
        json.dumps(
            {
                "schema_version": "advisor-validation-1",
                "model_version": manifest["model_version"],
                "sources": manifest["dataset"],
                "passed": True,
                "deployment_approved": False,
                "tolerances": {"atol": ATOL, "rtol": RTOL},
            }
        )
    )
    return root


def test_residuals_align_with_future_values_and_require_sustained_channel_evidence(
    history, candidate
):
    from edgeops_ai.training.advisor_residuals import residual_findings

    findings, scores, metadata = residual_findings(history, candidate)
    assert metadata["model_version"] == "residual-fixture"
    assert metadata["windows"] == {"train": 2, "validation": 2, "test": 2}
    assert metadata["devices"]["device-a:kitchen"]["training_windows"] == 2
    validation = [row for row in scores if row["split"] == "validation"]
    first = validation[0]
    assert first["time"] == (START + timedelta(minutes=64)).isoformat()
    assert first["last_input"] == (START + timedelta(minutes=63)).isoformat()
    assert first["horizon_step"] == 1
    assert first["observed"] == 25
    assert first["forecast"] == 20
    assert first["residual"] == 5
    assert first["score"] > 1
    assert findings
    assert {f.evidence["channel"] for f in findings} == {"temp_c_mean"}
    assert findings[0].start == START + timedelta(minutes=64)
    assert findings[0].evidence["sustained_minutes"] == 32
    assert findings[0].stop == START + timedelta(minutes=96)
    assert findings[0].evidence["model_version"] == "residual-fixture"


def test_calibration_is_frozen_when_validation_and_test_change(history, candidate):
    from edgeops_ai.training.advisor_residuals import residual_findings

    _, _, before = residual_findings(history, candidate)
    rows = pq.read_table(history / "device.parquet").to_pylist()
    for row in rows:
        if row["split"] != "train" and row["value"] is not None:
            row["value"] += 100
    rewrite(history, rows)
    # A different dataset must first be explicitly bound to a validated candidate.
    with pytest.raises(ValueError, match="dataset"):
        residual_findings(history, candidate)
    manifest = json.loads((candidate / "manifest.json").read_text())
    from edgeops_ai.forecasting import ForecastModel
    from edgeops_ai.training.advisor_forecast import load_windows
    from edgeops_ai.training.advisor_residuals import score_windows

    windows, _ = load_windows(history, manifest["contract"])
    model = ForecastModel(candidate / "model.onnx", manifest["sha256"])
    _, _, after = score_windows(model, windows)
    assert before["devices"] == after["devices"]


@pytest.mark.parametrize("problem", ["failed-parity", "wrong-manifest", "hash", "profile"])
def test_incompatible_candidates_fail_without_publishing(history, candidate, tmp_path, problem):
    if problem == "failed-parity":
        path = candidate / "validation-report.json"
        report = json.loads(path.read_text())
        report["passed"] = False
        path.write_text(json.dumps(report))
    elif problem == "wrong-manifest":
        path = candidate / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["model_version"] = "changed"
        path.write_text(json.dumps(manifest))
    elif problem == "hash":
        (candidate / "model.onnx.sha256").write_text("0" * 64)
    else:
        path = candidate / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["contract"] = forecast_contract("host-cpu", 16, 16)
        path.write_text(json.dumps(manifest))
    output = tmp_path / "bad"
    assert (
        main(
            [
                "advisor-health",
                "--input-dir",
                str(history),
                "--forecast-candidate",
                str(candidate),
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()


def test_replay_keeps_rules_adds_comparison_and_requires_fresh_reviews(
    history, candidate, tmp_path
):
    # #26 exports may contain no manual-label file; there is no implied human review.
    output = tmp_path / "residual-replay"
    result = replay(history, output, forecast_candidate=candidate)
    assert not result["acceptance"]["passed"]
    assert result["false_positives"]["unreviewed"] > 0
    assert result["forecast_residuals"]["comparison"]["ttm"]["unreviewed"] > 0
    assert (output / "forecast-residuals.csv").exists()
    with (output / "advice-review.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(not row["verdict"] for row in rows if row["detector"].startswith("model:ttm"))
    with (history / "manual-labels.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(MANUAL_COLUMNS)
    baseline = tmp_path / "baseline"
    replay(history, baseline)
    residual = tmp_path / "with-labels"
    replay(history, residual, forecast_candidate=candidate)
    old = json.loads((baseline / "advice.json").read_text())["findings"]
    new = json.loads((residual / "advice.json").read_text())["findings"]
    assert old == [r for r in new if not r["detector"].startswith("model:ttm")]


@pytest.mark.parametrize("problem", ["gap", "warmup", "room", "reset"])
def test_sustained_runs_stop_at_incompatible_segments(history, candidate, problem):
    from edgeops_ai.forecasting import ForecastModel
    from edgeops_ai.training.advisor_forecast import load_windows
    from edgeops_ai.training.advisor_residuals import score_windows

    rows = pq.read_table(history / "device.parquet").to_pylist()
    at = START + timedelta(minutes=80)
    if problem == "gap":
        rows = [r for r in rows if r["time"] != at]
    elif problem == "warmup":
        for row in rows:
            if row["time"] == at and row["field"] == "warmed_up_last":
                row["text"] = "false"
    elif problem == "room":
        for row in rows:
            if at <= row["time"] < START + timedelta(minutes=96):
                row["dimensions"] = '{"room":"hall"}'
    else:
        rows.append(
            {**rows[0], "time": at, "field": "uptime_resets", "value": 1, "split": "validation"}
        )
    rewrite(history, rows)
    model = ForecastModel(
        candidate / "model.onnx", (candidate / "model.onnx.sha256").read_text().strip()
    )
    windows, _ = load_windows(history, model.contract)
    findings, _, _ = score_windows(model, windows)
    assert findings[0].stop == at
    assert findings[0].evidence["sustained_minutes"] == 16


def test_brief_residual_spikes_do_not_produce_advice(history, candidate):
    from edgeops_ai.forecasting import ForecastModel
    from edgeops_ai.training.advisor_forecast import load_windows
    from edgeops_ai.training.advisor_residuals import score_windows

    model = ForecastModel(
        candidate / "model.onnx", (candidate / "model.onnx.sha256").read_text().strip()
    )
    windows, _ = load_windows(history, model.contract)
    for items in windows.values():
        for window in items:
            window["future"] = model.predict(window["past"][0], window["times"]).copy()
    windows["validation"][0]["future"][0, :14, 0] += 5
    assert score_windows(model, windows)[0] == []
    windows["validation"][0]["future"][0, 14, 0] += 5
    assert len(score_windows(model, windows)[0]) == 1
    windows["validation"][0]["entity_id"] = "untrained-device"
    with pytest.raises(ValueError, match="every split"):
        score_windows(model, windows)


def test_changed_model_evidence_cannot_inherit_old_advice_reviews(history, candidate, tmp_path):
    output = tmp_path / "original"
    replay(history, output, forecast_candidate=candidate)
    reviews = output / "advice-review.csv"
    with reviews.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(verdict="true-positive", note="Explicit fixture review")
    with reviews.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    model = onnx.load(candidate / "model.onnx")
    manifest = json.loads((candidate / "manifest.json").read_text())
    manifest.pop("sha256")
    manifest["model_version"] = "changed-model-evidence"
    onnx.helper.set_model_props(model, {MANIFEST_KEY: json.dumps(manifest)})
    onnx.save(model, candidate / "model.onnx")
    digest = hashlib.sha256((candidate / "model.onnx").read_bytes()).hexdigest()
    (candidate / "model.onnx.sha256").write_text(digest)
    (candidate / "manifest.json").write_text(json.dumps({**manifest, "sha256": digest}))
    report_path = candidate / "validation-report.json"
    report = json.loads(report_path.read_text())
    report["model_version"] = manifest["model_version"]
    report_path.write_text(json.dumps(report))
    changed = tmp_path / "changed"
    with pytest.raises(ValueError, match="Unknown advice review id"):
        replay(history, changed, forecast_candidate=candidate, reviews_path=reviews)
    assert not changed.exists()
