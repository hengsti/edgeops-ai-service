import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import onnx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.advisor import Advice
from edgeops_ai.forecasting import ATOL, MANIFEST_KEY, RTOL, forecast_contract
from edgeops_ai.training.advisor_dataset import MANUAL_COLUMNS, SCHEMA
from edgeops_ai.training.advisor_forecast import load_windows
from edgeops_ai.training.advisor_maintenance import replay
from edgeops_ai.training.advisor_resources import review_findings

START = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=5)


def rewrite(root, kind, rows):
    path = root / f"{kind}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card_path = root / "data-card.json"
    card = json.loads(card_path.read_text())
    card["tables"][kind] = {
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    card_path.write_text(json.dumps(card))


@pytest.fixture
def history(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    card = {
        "schema_version": "advisor-1",
        "start": START.isoformat(),
        "stop": (START + 144 * STEP).isoformat(),
        "split": {
            "train_stop": (START + 48 * STEP).isoformat(),
            "validation_stop": (START + 96 * STEP).isoformat(),
            "shuffle": False,
        },
        "tables": {},
    }
    (root / "data-card.json").write_text(json.dumps(card))
    tables = {"host": [], "container": [], "device": []}
    for index in range(144):
        base = {
            "time": START + index * STEP,
            "entity_id": "pi",
            "value": None,
            "text": None,
            "interval_minutes": 5,
            "split": ("train", "validation", "test")[index // 48],
            "weak_labels": [],
            "manual_label": None,
        }
        tables["host"].extend(
            [
                {
                    **base,
                    "measurement": "cpu",
                    "field": "usage_active_mean",
                    "value": 20.0,
                    "dimensions": '{"cpu":"cpu-total"}',
                },
                {
                    **base,
                    "measurement": "mem",
                    "field": "used_percent_mean",
                    "value": 50.0,
                    "dimensions": "{}",
                },
            ]
        )
        for field, value, text in (
            ("health_status_last", None, "healthy"),
            ("restart_count_last", 0.0, None),
            ("uptime_ns_last", (index + 1) * 300e9, None),
        ):
            tables["container"].append(
                {
                    **base,
                    "entity_id": "pi/service",
                    "field": field,
                    "measurement": "docker_container_health"
                    if field == "health_status_last"
                    else "docker_container_status",
                    "value": value,
                    "text": text,
                    "dimensions": '{"container_version":"1"}',
                }
            )
    for kind, rows in tables.items():
        rewrite(root, kind, rows)
    return root


def change_load(history, *, cpu=95.0, unhealthy=False, begin=68, end=74):
    for kind in ("host", "container"):
        rows = pq.read_table(history / f"{kind}.parquet").to_pylist()
        for row in rows:
            if START + begin * STEP <= row["time"] < START + end * STEP:
                if row["field"] == "usage_active_mean":
                    row["value"] = cpu
                elif row["field"] == "health_status_last" and unhealthy:
                    row["text"] = "unhealthy"
        rewrite(history, kind, rows)


@pytest.fixture
def candidate(history, tmp_path):
    change_load(history)
    root = tmp_path / "candidate"
    root.mkdir()
    manifest = {
        "schema_version": "advisor-onnx-1",
        "model_version": "cpu-fixture",
        "contract": forecast_contract("host-cpu", 16, 4),
        "dataset": {
            name: hashlib.sha256((history / name).read_bytes()).hexdigest()
            for name in ("data-card.json", "host.parquet")
        },
    }
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Constant",
                [],
                ["forecast"],
                value=onnx.numpy_helper.from_array(np.full((1, 4, 1), 20, dtype=np.float32)),
            )
        ],
        "cpu-fixture",
        [onnx.helper.make_tensor_value_info("past_values", onnx.TensorProto.FLOAT, [1, 16, 1])],
        [onnx.helper.make_tensor_value_info("forecast", onnx.TensorProto.FLOAT, [1, 4, 1])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.helper.set_model_props(model, {MANIFEST_KEY: json.dumps(manifest)})
    onnx.save(model, root / "model.onnx")
    digest = hashlib.sha256((root / "model.onnx").read_bytes()).hexdigest()
    (root / "model.onnx.sha256").write_text(digest)
    (root / "manifest.json").write_text(json.dumps({**manifest, "sha256": digest}))
    (root / "validation-report.json").write_text(
        json.dumps(
            {
                "schema_version": "advisor-validation-1",
                "model_version": "cpu-fixture",
                "sources": manifest["dataset"],
                "passed": True,
                "tolerances": {"atol": ATOL, "rtol": RTOL},
                "deployment_approved": False,
            }
        )
    )
    return root


def test_one_step_windows_do_not_wait_for_the_full_forecast_horizon(history):
    windows, _ = load_windows(history, forecast_contract("host-cpu", 16, 4), one_step=True)
    assert {split: len(items) for split, items in windows.items()} == {
        "train": 32,
        "validation": 32,
        "test": 32,
    }
    first = windows["validation"][0]
    assert first["times"][-1] == START + 63 * STEP
    assert first["target_times"] == [START + 64 * STEP]
    assert first["future"].shape == (1, 1, 1)


@pytest.mark.parametrize("activity", [None, "build"])
def test_healthy_busy_activity_is_informational_and_preserves_baseline(
    history, candidate, tmp_path, activity
):
    output = tmp_path / "assessment"
    events = event_file(tmp_path, [(activity, 68, 74)]) if activity else None
    replay(history, output, forecast_candidate=candidate, workload_events=events)
    state = json.loads((output / "state-report.json").read_text())
    assert state["forecasts"] == []
    advice = json.loads((output / "advice.json").read_text())["findings"]
    assert advice and all(item["severity"] == "info" for item in advice)
    assert advice[0]["evidence"]["first_detected_at"] == (START + 71 * STEP).isoformat()
    assert "build" not in advice[0]["summary"].lower()
    assert advice[0]["evidence"]["activity"] == (activity or "unknown")
    assert "cause" in advice[0]["evidence"] and advice[0]["evidence"]["cause"] == "unknown"
    assert state["resources"][0]["state"] == "expected-workload"
    assert state["resources"][0]["confidence"] is None
    metadata = json.loads((output / "report.json").read_text())["resources"]
    assert metadata["lead_minutes"] == 5
    assert metadata["acceptance"]["passed"] is False


def test_unusual_load_with_unhealthy_service_warns_within_fifteen_minutes(
    history, candidate, tmp_path
):
    # Context health is separate from model inputs, which stay unchanged.
    rows = pq.read_table(history / "container.parquet").to_pylist()
    for row in rows:
        if (
            row["field"] == "health_status_last"
            and START + 68 * STEP <= row["time"] < START + 74 * STEP
        ):
            row["text"] = "unhealthy"
    rewrite(history, "container", rows)
    # Bind the fixture to its new data-card using an equally valid fixture candidate.
    rebind(candidate, history)
    output = tmp_path / "pressure"
    replay(history, output, forecast_candidate=candidate)
    findings = json.loads((output / "advice.json").read_text())["findings"]
    warning = next(item for item in findings if item["severity"] == "warning")
    assert warning["evidence"]["first_detected_at"] == (START + 71 * STEP).isoformat()
    assert warning["evidence"]["affected_services"] == "pi/service"
    assert warning["evidence"]["model_version"] == "cpu-fixture"
    assert warning["evidence"]["expectation_lead_minutes"] == 5


def rebind(candidate, history):
    embedded = json.loads(onnx.load(candidate / "model.onnx").metadata_props[0].value)
    embedded["dataset"] = {
        name: hashlib.sha256((history / name).read_bytes()).hexdigest()
        for name in ("data-card.json", "host.parquet")
    }
    model = onnx.load(candidate / "model.onnx")
    onnx.helper.set_model_props(model, {MANIFEST_KEY: json.dumps(embedded)})
    onnx.save(model, candidate / "model.onnx")
    digest = hashlib.sha256((candidate / "model.onnx").read_bytes()).hexdigest()
    (candidate / "model.onnx.sha256").write_text(digest)
    (candidate / "manifest.json").write_text(json.dumps({**embedded, "sha256": digest}))
    validation = json.loads((candidate / "validation-report.json").read_text())
    validation["sources"] = embedded["dataset"]
    (candidate / "validation-report.json").write_text(json.dumps(validation))


@pytest.mark.parametrize("activity", ["reset", "deploy"])
def test_known_host_reset_breaks_context_and_persistence(history, candidate, tmp_path, activity):
    events = tmp_path / "events.csv"
    with events.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "entity_type": "host",
                "entity_id": "pi",
                "label": activity,
                "start": (START + 69 * STEP).isoformat(),
                "stop": (START + 70 * STEP).isoformat(),
                "note": f"Reviewed host {activity}",
            }
        )
    output = tmp_path / "reset"
    replay(history, output, forecast_candidate=candidate, workload_events=events)
    with (output / "resource-observations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    reset = next(row for row in rows if row["time"] == (START + 71 * STEP).isoformat())
    assert reset["expected_cpu_percent"] == ""
    assert reset["model_status"] == "insufficient-context"


def test_one_step_windows_break_at_explicit_boundaries(history):
    windows, _ = load_windows(
        history,
        forecast_contract("host-cpu", 16, 4),
        one_step=True,
        boundaries={("pi", START + 68 * STEP)},
    )
    assert all(
        not (w["times"][0] < START + 68 * STEP <= w["target_times"][0])
        for w in windows["validation"]
    )


def event_file(tmp_path, events):
    path = tmp_path / "workload-events.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_COLUMNS)
        writer.writeheader()
        for label, begin, end in events:
            writer.writerow(
                {
                    "entity_type": "host",
                    "entity_id": "pi",
                    "label": label,
                    "start": (START + begin * STEP).isoformat(),
                    "stop": (START + end * STEP).isoformat(),
                    "note": "Explicitly reviewed fixture activity",
                }
            )
    return path


def test_advice_requires_review_and_normal_and_incident_cases(history, candidate, tmp_path):
    change_load(history, cpu=60, unhealthy=True)
    rebind(candidate, history)
    events = event_file(tmp_path, [("normal-activity", 64, 68), ("resource-pressure", 68, 74)])
    first = tmp_path / "unreviewed"
    before = replay(history, first, forecast_candidate=candidate, workload_events=events)
    assert before["resources"]["acceptance"]["passed"] is False
    with (first / "resource-advice-review.csv").open(newline="") as handle:
        reviews = list(csv.DictReader(handle))
    assert reviews
    for row in reviews:
        row["verdict"], row["note"] = "true-positive", "Reviewed fixture finding is useful"
    review_file = tmp_path / "reviews.csv"
    with review_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    report = replay(
        history,
        tmp_path / "reviewed",
        forecast_candidate=candidate,
        workload_events=events,
        resource_reviews=review_file,
    )
    comparison = report["resources"]["comparison"]
    assert comparison["ttm"]["missed_incidents"] == 0
    assert comparison["rules"]["missed_incidents"] == 1
    assert comparison["ttm"]["incident_cases"][0]["detection_delay_minutes"] == 15
    assert report["resources"]["acceptance"]["passed"] is True
    assert report["resources"]["deployment_approved"] is False


def test_calibration_does_not_fit_held_out_cpu_or_health(history, candidate, tmp_path):
    first = replay(history, tmp_path / "first", forecast_candidate=candidate)["resources"]
    change_load(history, cpu=99, unhealthy=True)
    rebind(candidate, history)
    after = replay(history, tmp_path / "after", forecast_candidate=candidate)["resources"]
    assert first["calibration"] == after["calibration"]
    assert first["calibration_sha256"] == after["calibration_sha256"]


def test_rejected_warning_does_not_count_as_detecting_an_incident(tmp_path):
    begin = START + timedelta(days=1, hours=1)
    stop = begin + timedelta(minutes=15)
    warning = Advice(
        "host.resource-pressure",
        "pi",
        begin,
        stop,
        stop,
        "Pressure",
        "Inspect",
        {"first_detected_at": stop.isoformat()},
    )
    findings = {"ttm": [warning], "rules": [], "last_value": []}
    metadata = {"available": True, "sources": {}, "candidate_sha256": {}}
    cases = [
        {"entity_id": "pi", "start": begin, "stop": stop, "label": "resource-pressure"},
        {
            "entity_id": "pi",
            "start": START + timedelta(days=1),
            "stop": begin,
            "label": "normal-activity",
        },
    ]
    cuts = [START + timedelta(days=1), START + timedelta(days=3)]
    rows = review_findings(findings, metadata, cases, cuts, None, 15)
    reviews = tmp_path / "rejected.csv"
    with reviews.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "verdict", "note"])
        writer.writeheader()
        writer.writerow({"id": rows[0]["id"], "verdict": "false-positive", "note": "Not useful"})
    review_findings(findings, metadata, cases, cuts, reviews, 15)
    comparison = metadata["comparison"]["ttm"]
    assert comparison["false_positive_warnings_per_day"] == 0.5
    assert comparison["missed_incidents"] == 1
    assert metadata["acceptance"]["passed"] is False


@pytest.mark.parametrize("missing_at", [0, 1])
def test_missing_current_or_previous_container_uptime_is_safe(
    history, candidate, tmp_path, missing_at
):
    rows = pq.read_table(history / "container.parquet").to_pylist()
    for row in rows:
        if row["field"] == "uptime_ns_last" and row["time"] == START + missing_at * STEP:
            row["value"] = None
    rewrite(history, "container", rows)
    rebind(candidate, history)
    output = tmp_path / "missing-uptime"
    replay(history, output, forecast_candidate=candidate)
    findings = json.loads((output / "advice.json").read_text())["findings"]
    assert findings and all(item["severity"] == "info" for item in findings)


def test_two_sample_spike_never_becomes_advice(history, candidate, tmp_path):
    change_load(history, cpu=20)
    change_load(history, cpu=99, unhealthy=True, begin=68, end=70)
    rebind(candidate, history)
    output = tmp_path / "spike"
    replay(history, output, forecast_candidate=candidate)
    assert json.loads((output / "advice.json").read_text())["findings"] == []


def test_missing_health_is_explicit_and_never_corroborates_a_warning(history, candidate, tmp_path):
    rewrite(history, "container", [])
    rebind(candidate, history)
    output = tmp_path / "no-health"
    replay(history, output, forecast_candidate=candidate)
    with (output / "resource-observations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["health_status"] == "unavailable" for row in rows)
    assert all(
        f["severity"] == "info"
        for f in json.loads((output / "advice.json").read_text())["findings"]
    )


@pytest.mark.parametrize("problem", ["gap", "missing", "invalid", "reset"])
def test_bad_cpu_history_breaks_one_step_context(history, problem):
    rows = pq.read_table(history / "host.parquet").to_pylist()
    point = next(
        row
        for row in rows
        if row["field"] == "usage_active_mean" and row["time"] == START + 68 * STEP
    )
    if problem == "gap":
        rows.remove(point)
    elif problem == "missing":
        point["value"] = None
    elif problem == "invalid":
        point["value"] = 101.0
    else:
        rows.append({**point, "field": "uptime_resets", "value": 1.0})
    rewrite(history, "host", rows)
    windows, _ = load_windows(history, forecast_contract("host-cpu", 16, 4), one_step=True)
    assert all(
        w["times"][0] >= START + (68 if problem == "reset" else 69) * STEP
        or w["target_times"][0] < START + 68 * STEP
        for w in windows["validation"]
    )


def test_changed_context_cannot_inherit_advice_review(history, candidate, tmp_path):
    first = tmp_path / "before"
    replay(history, first, forecast_candidate=candidate)
    reviews = first / "resource-advice-review.csv"
    rows = pq.read_table(history / "container.parquet").to_pylist()
    rows[0]["text"] = "unhealthy"
    rewrite(history, "container", rows)
    rebind(candidate, history)
    with pytest.raises(ValueError, match="review"):
        replay(
            history, tmp_path / "changed", forecast_candidate=candidate, resource_reviews=reviews
        )
    assert not (tmp_path / "changed").exists()


def test_advice_pairs_the_observed_and_expected_values_at_the_same_target(
    history, candidate, tmp_path
):
    rows = pq.read_table(history / "host.parquet").to_pylist()
    for row in rows:
        if row["field"] == "usage_active_mean" and row["time"] == START + 73 * STEP:
            row["value"] = 99.0
    rewrite(history, "host", rows)
    rebind(candidate, history)
    output = tmp_path / "aligned-evidence"
    replay(history, output, forecast_candidate=candidate)
    evidence = json.loads((output / "advice.json").read_text())["findings"][0]["evidence"]
    assert evidence["expectation_target"] == (START + 68 * STEP).isoformat()
    assert evidence["cpu_percent"] == 95.0
    assert evidence["maximum_cpu_percent"] == 99.0
    assert evidence["memory_percent"] == 50.0


def test_uncalibrated_model_is_unavailable_even_with_valid_held_out_context(
    history, candidate, tmp_path
):
    rows = [
        row
        for row in pq.read_table(history / "host.parquet").to_pylist()
        if row["split"] != "train"
        or row["field"] != "usage_active_mean"
        or row["time"] < START + 8 * STEP
    ]
    rewrite(history, "host", rows)
    rebind(candidate, history)
    output = tmp_path / "uncalibrated"
    report = replay(history, output, forecast_candidate=candidate)
    assert report["resources"]["available"] is False
    assert report["resources"]["acceptance"]["passed"] is False
    with (output / "resource-observations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if row["time"] == (START + 64 * STEP).isoformat())
    assert row["model_status"] == "uncalibrated"
    assert row["state"] == "insufficient-evidence"


def test_container_restart_breaks_health_persistence_without_erasing_host_history(
    history, candidate, tmp_path
):
    change_load(history, unhealthy=True)
    rows = pq.read_table(history / "container.parquet").to_pylist()
    for row in rows:
        if row["field"] == "restart_count_last" and row["time"] >= START + 70 * STEP:
            row["value"] = 1.0
    rewrite(history, "container", rows)
    rebind(candidate, history)
    output = tmp_path / "container-restart"
    replay(history, output, forecast_candidate=candidate)
    warning = next(
        item
        for item in json.loads((output / "advice.json").read_text())["findings"]
        if item["severity"] == "warning"
    )
    assert warning["evidence"]["first_detected_at"] == (START + 74 * STEP).isoformat()


def test_corrupt_artifact_does_not_publish_a_report(history, candidate, tmp_path):
    (candidate / "model.onnx.sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        replay(history, tmp_path / "corrupt", forecast_candidate=candidate)
    assert not (tmp_path / "corrupt").exists()


def test_latest_missing_interval_is_explicitly_stale(history, candidate, tmp_path):
    for kind in ("host", "container"):
        rows = [
            row
            for row in pq.read_table(history / f"{kind}.parquet").to_pylist()
            if row["time"] < START + 143 * STEP
        ]
        rewrite(history, kind, rows)
    rebind(candidate, history)
    output = tmp_path / "stale"
    replay(history, output, forecast_candidate=candidate)
    state = json.loads((output / "state-report.json").read_text())["resources"][0]
    assert state["model_status"] == "stale"
    assert state["state"] == "insufficient-evidence"


def test_replay_preserves_sources_and_refuses_output_overwrite(history, candidate, tmp_path):
    sources = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*history.iterdir(), *candidate.iterdir()]
    }
    output = tmp_path / "preserved"
    replay(history, output, forecast_candidate=candidate)
    original = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        replay(history, output, forecast_candidate=candidate)
    assert (output / "report.json").read_bytes() == original
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in sources.items()
    )
