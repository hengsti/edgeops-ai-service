import csv
import hashlib
import json
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.training.advisor_dataset import MANUAL_COLUMNS, SCHEMA
from edgeops_ai.training.advisor_health import (
    baseline_findings,
    load_devices,
    load_pipeline,
    reference_case,
    replay,
)
from edgeops_ai.training.cli import COMMANDS, main
from test_advisor_health import START, samples


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    rows = []
    mapping = {
        "rssi": "rssi_last",
        "free_mem": "free_mem_min",
        "uptime": "uptime_last",
        "sensor_read_failures_total": "sensor_read_failures_total_last",
        "samples_failed_total": "samples_failed_total_last",
        "mqtt_connects_total": "mqtt_connects_total_last",
    }
    for sample in samples(minutes=3 * 1440, rssi=-80):
        split = (
            "train"
            if sample.at < START + timedelta(days=1)
            else ("validation" if sample.at < START + timedelta(days=2) else "test")
        )
        for field, value in {
            **{name: sample.values[key] for key, name in mapping.items()},
            "temp_c_count": 30,
        }.items():
            rows.append(
                {
                    "time": sample.at,
                    "entity_id": sample.device_id,
                    "measurement": "bme680" if field == "temp_c_count" else "device_status",
                    "field": field,
                    "value": value,
                    "text": None,
                    "dimensions": "{}",
                    "interval_minutes": 1,
                    "split": split,
                    "weak_labels": [],
                    "manual_label": None,
                }
            )
    path = root / "device.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card = {
        "schema_version": "advisor-1",
        "start": START.isoformat(),
        "stop": (START + timedelta(days=3)).isoformat(),
        "split": {
            "train_stop": (START + timedelta(days=1)).isoformat(),
            "validation_stop": (START + timedelta(days=2)).isoformat(),
            "shuffle": False,
        },
        "tables": {
            "device": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
        },
    }
    (root / "data-card.json").write_text(json.dumps(card), encoding="utf-8")
    with (root / "manual-labels.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(MANUAL_COLUMNS)
    return root


def test_health_command_and_real_replay_publish_reviewable_evidence(exported, tmp_path):
    assert COMMANDS.get("advisor-health") == "advisor_health"
    source_hash = hashlib.sha256((exported / "device.parquet").read_bytes()).hexdigest()
    output = tmp_path / "health"
    assert main(["advisor-health", "--input-dir", str(exported), "--output-dir", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["false_positives"]["unreviewed"] == 1
    assert report["false_positives"]["per_day"] is None
    assert not report["acceptance"]["passed"]
    assert report["baseline"]["devices"]["a"]["training_windows"] == 24
    assert (
        report["baseline"]["devices"]["a"]["training_stop"]
        == (START + timedelta(days=1)).isoformat()
    )
    assert (output / "advice-review.csv").exists()
    assert (output / "anomaly-scores.csv").exists()
    assert hashlib.sha256((exported / "device.parquet").read_bytes()).hexdigest() == source_hash


def test_reviewed_false_positive_rate_is_counted_per_validation_calendar_day(exported, tmp_path):
    first = tmp_path / "first"
    replay(exported, first)
    with (first / "advice-review.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(verdict="false-positive", note="Reviewed signal; no action was useful")
    reviews = tmp_path / "reviews.csv"
    with reviews.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = replay(exported, tmp_path / "reviewed", reviews_path=reviews)
    assert report["false_positives"]["unreviewed"] == 0
    assert report["false_positives"]["per_day"] == 1
    assert report["false_positives"]["passed"] is False  # target is strictly under one
    assert report["historical_reproduction"]["weak-wifi"] is False


def test_hash_mismatch_and_existing_output_fail_without_overwriting(exported, tmp_path):
    output = tmp_path / "health"
    output.mkdir()
    with pytest.raises(ValueError, match="exists"):
        replay(exported, output)
    card_path = exported / "data-card.json"
    card = json.loads(card_path.read_text())
    card["tables"]["device"]["sha256"] = "wrong"
    card_path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="hash"):
        replay(exported, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_reader_keeps_field_meanings(exported):
    loaded = load_devices(exported)
    assert loaded[0].values["sensor_samples"] == 30
    assert loaded[0].values["free_mem"] == 470000
    assert loaded[0].values["mqtt_connects_total"] == 1


def test_climate_rows_before_the_sample_count_do_not_create_a_conflicting_count(exported):
    path = exported / "device.parquet"
    table = pq.read_table(path)
    row = table.slice(0, 1).to_pylist()[0]
    row.update(measurement="bme680", field="iaq_score_mean", value=40)
    pq.write_table(pa.concat_tables([pa.Table.from_pylist([row], schema=SCHEMA), table]), path)
    assert load_devices(exported)[0].values["sensor_samples"] == 30


def test_pipeline_reader_uses_all_source_tags_and_rejects_bad_values(tmp_path):
    path = tmp_path / "pipeline.csv"
    path.write_text(
        "#datatype,string,string,double,string,string\n"
        "_time,_measurement,_value,_field,host,url\n"
        "2026-09-01T00:00:00Z,prometheus,100,ingest_incoming_invalid_json_total,pi,ingest\n"
        "2026-09-01T00:01:00Z,prometheus,102,ingest_incoming_invalid_json_total,pi,ingest\n"
        "2026-09-01T00:01:00Z,prometheus,200,ingest_incoming_invalid_json_total,other,ingest\n",
        encoding="utf-8",
    )
    loaded = load_pipeline(path, START, START + timedelta(days=1))
    assert len(loaded) == 3
    assert loaded[1].entity_id != loaded[2].entity_id
    path.write_text(path.read_text().replace(",102,", ",nan,"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline(path, START, START + timedelta(days=1))


def test_iforest_training_does_not_change_when_future_values_change():
    rows = samples(minutes=3 * 1440)
    for i, row in enumerate(rows):
        row.values.update(rssi=-50 + (i // 60) % 11, free_mem=470000 - (i // 60) % 7 * 100)
    train_stop, validation_stop, stop = [START + timedelta(days=i) for i in (1, 2, 3)]
    _, _, before = baseline_findings(rows, START, train_stop, validation_stop, stop, [])
    for row in rows[1440:]:
        row.values.update(rssi=-100, free_mem=100)
    findings, scores, after = baseline_findings(rows, START, train_stop, validation_stop, stop, [])
    assert before["devices"] == after["devices"]
    assert findings and any(row["split"] == "validation" and row["score"] < 0 for row in scores)


def test_baseline_does_not_fill_missing_channels_or_cross_split_boundaries():
    rows = samples(minutes=3 * 1440)
    for row in rows:
        row.values.pop("free_mem")
    findings, scores, metadata = baseline_findings(
        rows,
        START,
        START + timedelta(days=1, minutes=5),
        START + timedelta(days=2),
        START + timedelta(days=3),
        [],
    )
    assert findings == [] and scores == []
    assert metadata["devices"]["a"]["training_windows"] == 0


def test_baseline_uses_dense_health_channels_without_inventing_sparse_counters():
    rows = samples(minutes=3 * 1440)
    for row in rows:
        for name in ("sensor_read_failures_total", "samples_failed_total", "mqtt_connects_total"):
            row.values.pop(name)
    _, scores, metadata = baseline_findings(
        rows,
        START,
        START + timedelta(days=1),
        START + timedelta(days=2),
        START + timedelta(days=3),
        [],
    )
    assert metadata["devices"]["a"]["training_windows"] == 24
    assert scores and metadata["features"] == ("median_rssi_dbm", "median_free_mem_bytes")


def test_unknown_review_ids_are_rejected_atomically(exported, tmp_path):
    path = tmp_path / "reviews.csv"
    path.write_text("id,verdict,note\nunknown,true-positive,checked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        replay(exported, tmp_path / "bad", reviews_path=path)
    assert not (tmp_path / "bad").exists()


def test_reviews_cannot_be_reused_with_changed_source_identity(exported, tmp_path):
    first = tmp_path / "first"
    replay(exported, first)
    card_path = exported / "data-card.json"
    card = json.loads(card_path.read_text())
    card["changed_source_metadata"] = True
    card_path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="Unknown"):
        replay(exported, tmp_path / "changed", reviews_path=first / "advice-review.csv")


def test_reviewing_another_device_does_not_verify_the_named_historical_finding(exported, tmp_path):
    first = tmp_path / "first"
    replay(exported, first)
    with (first / "advice-review.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(verdict="true-positive", note="Reviewed this fixture device")
    path = tmp_path / "reviews.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = replay(exported, tmp_path / "reviewed", reviews_path=path)
    assert report["false_positives"]["per_day"] == 0
    assert report["signal_coverage"]["weak-wifi"] is True
    assert report["historical_reproduction"]["weak-wifi"] is False


def test_staleness_first_detected_at_train_stop_is_validation(exported, tmp_path):
    path = exported / "device.parquet"
    rows = pq.read_table(path).to_pylist()
    rows = [
        row
        for row in rows
        if not (
            row["measurement"] == "bme680" and START < row["time"] < START + timedelta(minutes=20)
        )
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card_path = exported / "data-card.json"
    card = json.loads(card_path.read_text())
    card["split"]["train_stop"] = (START + timedelta(minutes=11)).isoformat()
    card["tables"]["device"].update(
        rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    card_path.write_text(json.dumps(card))
    output = tmp_path / "boundary"
    replay(exported, output)
    findings = json.loads((output / "advice.json").read_text())["findings"]
    outage = next(f for f in findings if f["code"] == "device.all-silent")
    assert outage["split"] == "validation"


def test_training_only_anomaly_hours_do_not_become_validation_advice(monkeypatch):
    class Forest:
        def __init__(self, **kwargs):
            pass

        def fit(self, matrix):
            return self

        def decision_function(self, matrix):
            return [-0.1 if 26 <= i <= 28 else 0.1 for i in range(len(matrix))]

    monkeypatch.setattr("edgeops_ai.training.advisor_health.IsolationForest", Forest)
    cut = START + timedelta(days=1, hours=12)
    findings, _, _ = baseline_findings(
        samples(minutes=3 * 1440),
        START,
        cut,
        START + timedelta(days=2),
        START + timedelta(days=3),
        [],
    )
    assert len(findings) == 1 and findings[0].stop == cut


def test_reference_outage_requires_actual_overlap_with_the_known_incident():
    assert (
        reference_case(
            "unnoticed-outage",
            {
                "entity_id": "*",
                "start": "2026-09-10T01:00:00+00:00",
                "stop": "2026-09-11T00:00:00+00:00",
            },
        )
        is False
    )
