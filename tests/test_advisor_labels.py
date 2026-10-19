import csv
import hashlib
import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.training.advisor_dataset import EVENT_COLUMNS, KINDS, MANUAL_COLUMNS, SCHEMA
from edgeops_ai.training.cli import COMMANDS, main


def test_advisor_labels_is_a_training_command():
    assert COMMANDS.get("advisor-labels") == "advisor_labels"


def write_csv(path, rows, columns=MANUAL_COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def review(entity, begin, end, label, kind="device"):
    return {
        "entity_type": kind,
        "entity_id": entity,
        "start": f"2026-09-11T00:{begin}:00Z",
        "stop": f"2026-09-11T00:{end}:00Z",
        "label": label,
        "note": "Confirmed by the user",
    }


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    card = {
        "schema_version": "advisor-1",
        "start": "2026-09-11T00:00:00Z",
        "stop": "2026-09-11T01:00:00Z",
        "source_values_sha256": "original-input",
        "split": {
            "train_stop": "2026-09-11T00:20:00Z",
            "validation_stop": "2026-09-11T00:40:00Z",
            "shuffle": False,
        },
        "manual_labels": {"reviewed_intervals": 0},
        "tables": {},
    }
    for kind in KINDS:
        rows = [
            {
                "time": datetime(2026, 9, 11, 0, minute, tzinfo=UTC),
                "entity_id": "sensor-a" if kind == "device" else kind,
                "measurement": "bme680",
                "field": "temp_c_mean",
                "value": 20.0,
                "text": None,
                "dimensions": "{}",
                "interval_minutes": 5 if kind in {"container", "host"} else 1,
                "split": "train" if minute < 20 else "validation" if minute < 40 else "test",
                "weak_labels": ["counter-reset"] if minute == 5 else [],
                "manual_label": None,
            }
            for minute in (0, 5, 20, 40)
        ]
        path = root / f"{kind}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        card["tables"][kind] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "data-card.json").write_text(json.dumps(card), encoding="utf-8")
    (root / "data-card.md").write_text("Original data card", encoding="utf-8")
    write_csv(root / "manual-labels.csv", [], EVENT_COLUMNS)
    events = [
        {**review(entity, "01", "20", label), "source": "aggregate-rule"}
        for entity, label in [("sensor-a", "sensor-telemetry-gap"), ("*", "all-devices-silent")]
    ]
    write_csv(root / "weak-labels.csv", events, EVENT_COLUMNS)
    return root


def apply(exported, labels, output):
    return main(
        [
            "advisor-labels",
            "--input-dir",
            str(exported),
            "--labels",
            str(labels),
            "--output-dir",
            str(output),
        ]
    )


def test_offline_review_preserves_data_and_weak_labels_and_records_gap_only_reviews(
    exported, tmp_path
):
    labels, output = tmp_path / "reviews.csv", tmp_path / "reviewed"
    write_csv(
        labels,
        [
            review("sensor-a", "20", "21", "device-unresponsive"),
            review("sensor-a", "01", "20", "server-outage"),
            review("*", "01", "20", "server-outage"),
        ],
    )
    original = {p.name: p.read_bytes() for p in exported.iterdir()}
    assert apply(exported, labels, output) == 0
    for kind in KINDS:
        before, after = (pq.read_table(root / f"{kind}.parquet") for root in (exported, output))
        assert before.drop(["manual_label"]).equals(after.drop(["manual_label"]))
        if kind == "device":
            assert after["manual_label"].to_pylist() == [
                None,
                "server-outage",
                "device-unresponsive",
                None,
            ]
            assert after["weak_labels"].to_pylist()[1] == ["counter-reset"]
        else:
            assert after["manual_label"].null_count == len(after)
    assert original == {p.name: p.read_bytes() for p in exported.iterdir()}
    assert (output / "weak-labels.csv").read_bytes() == original["weak-labels.csv"]
    card = json.loads((output / "data-card.json").read_text())
    assert card["source_values_sha256"] == "original-input"
    assert card["manual_labels"]["reviewed_intervals"] == 3
    assert card["review"]["by_split"] == {"train": 2, "validation": 1}
    assert (output / "source-data-card.json").read_bytes() == original["data-card.json"]


def test_gap_review_needs_no_synthetic_measurement_rows(exported, tmp_path):
    labels, output = tmp_path / "reviews.csv", tmp_path / "reviewed"
    write_csv(labels, [review("sensor-a", "06", "20", "server-outage")])
    assert apply(exported, labels, output) == 0
    assert pq.read_table(output / "device.parquet")["manual_label"].null_count == 4
    card = json.loads((output / "data-card.json").read_text())
    assert card["review"]["gap_only_intervals"] == 1
    assert card["manual_labels"]["reviewed_intervals"] == 1


@pytest.mark.parametrize(
    "row",
    [
        review("missing-device", "20", "21", "normal"),
        review("sensor-a", "10", "30", "server-outage"),
        review("container", "21", "26", "normal", "container"),
    ],
)
def test_invalid_review_does_not_publish_output(exported, tmp_path, row):
    labels, output = tmp_path / "reviews.csv", tmp_path / "reviewed"
    write_csv(labels, [row])
    assert apply(exported, labels, output) == 1
    assert not output.exists()


def test_conflicting_global_and_device_reviews_fail(exported, tmp_path):
    labels, output = tmp_path / "reviews.csv", tmp_path / "reviewed"
    write_csv(
        labels, [review("*", "01", "20", "server-outage"), review("sensor-a", "05", "06", "normal")]
    )
    assert apply(exported, labels, output) == 1
    assert not output.exists()


def test_subsequent_import_retains_previous_reviews(exported, tmp_path):
    labels = tmp_path / "reviews.csv"
    first, second = tmp_path / "first", tmp_path / "second"
    write_csv(labels, [review("sensor-a", "20", "21", "device-unresponsive")])
    assert apply(exported, labels, first) == 0
    write_csv(labels, [review("host", "20", "25", "normal", "host")])
    assert apply(first, labels, second) == 0
    assert (
        pq.read_table(second / "device.parquet")["manual_label"].to_pylist()[2]
        == "device-unresponsive"
    )
    assert (
        json.loads((second / "data-card.json").read_text())["manual_labels"]["reviewed_intervals"]
        == 2
    )


def test_input_hash_mismatch_or_existing_output_is_rejected(exported, tmp_path):
    labels, output = tmp_path / "reviews.csv", tmp_path / "reviewed"
    write_csv(labels, [review("sensor-a", "20", "21", "normal")])
    path = exported / "device.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    assert apply(exported, labels, output) == 1
    assert not output.exists()
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    assert apply(exported, labels, output) == 1
    assert (output / "keep.txt").read_text() == "keep"
