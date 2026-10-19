import csv
import io
import json
import re
from datetime import UTC, datetime

import httpx
import pyarrow.parquet as pq
import pytest

from edgeops_ai.training.cli import COMMANDS, main


def test_advisor_dataset_is_a_training_command():
    assert COMMANDS.get("advisor-dataset") == "advisor_dataset"


def csv_response(rows):
    output = io.StringIO(newline="")
    fields = [
        "",
        "result",
        "table",
        "_time",
        "_measurement",
        "_field",
        "_value",
        "device_id",
        "room",
        "host",
        "container_name",
        "container_status",
        "path",
        "ip",
        "fw_version",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    for row in rows:
        datatype = row.get("_datatype", "double")
        if row["_value"] in {"unhealthy", "true", "false"}:
            datatype = "string"
        annotations = ["string"] * len(fields)
        annotations[0], annotations[fields.index("_value")] = "#datatype", datatype
        csv.writer(output).writerow(annotations)
        writer.writeheader()
        writer.writerow(
            {
                "result": "_result",
                "table": "0",
                **{k: v for k, v in row.items() if k != "_datatype"},
            }
        )
    return output.getvalue()


def point(minute, field="temp_c_mean", value="20", **tags):
    return {
        "_time": f"2026-09-11T{minute}:00Z",
        "_measurement": "bme680",
        "_field": field,
        "_value": value,
        "device_id": "sensor-a",
        "room": "office",
        **tags,
    }


@pytest.fixture
def source_rows():
    return {
        "sensors_1m": [
            point("00:00", ip="old"),
            point("00:00", ip="new"),
            point("00:00", field="temp_c_count", value="10"),
            point("00:00", value="30", device_id="sensor-b"),
            point("00:00", field="temp_c_count", value="30", device_id="sensor-b"),
            point("00:00", device_id="stresstest"),
            point("00:00", device_id="livingroom-01"),
            point(
                "00:00",
                _measurement="device_status",
                room="",
                field="sensor_read_failures_total_last",
                value="4",
            ),
            point(
                "00:01",
                _measurement="device_status",
                room="",
                field="sensor_read_failures_total_last",
                value="5",
            ),
            point(
                "00:02",
                _measurement="device_status",
                room="",
                field="sensor_read_failures_total_last",
                value="6",
            ),
            point(
                "00:02",
                _measurement="device_status",
                room="",
                field="sensor_read_failures_total_resets",
                value="1",
            ),
            point("00:20"),
            point("00:20", field="temp_c_count", value="10"),
            point("00:40"),
            point("00:40", field="temp_c_count", value="10"),
        ],
        "metrics_5m": [
            point(
                "00:00",
                _measurement="cpu",
                device_id="",
                room="",
                host="pi",
                field="usage_active_mean",
                value="15",
            ),
            point(
                "00:20",
                _measurement="disk",
                device_id="",
                room="",
                host="pi",
                path="/",
                field="used_percent_mean",
                value="40",
            ),
            point(
                "00:20",
                _measurement="docker_container_cpu",
                device_id="",
                room="",
                host="pi",
                container_name="ingest",
                field="usage_percent_mean",
                value="25",
            ),
            point(
                "00:20",
                _measurement="docker_container_mem",
                device_id="",
                room="",
                host="pi",
                container_name="ingest",
                field="usage_mean",
                value="0",
            ),
            point(
                "00:40",
                _measurement="docker_container_health",
                device_id="",
                room="",
                host="pi",
                container_name="ingest",
                field="health_status_last",
                value="unhealthy",
            ),
        ],
    }


def run_build(tmp_path, monkeypatch, source_rows, extra=()):
    from edgeops_ai.training import advisor_dataset

    def query(request):
        payload = json.loads(request.content)
        assert request.headers["Authorization"] == "Token secret-token"
        assert request.url.params["org"] == "test-org"
        bucket = "sensors_1m" if '"sensors_1m"' in payload["query"] else "metrics_5m"
        begin, end = re.search(r"range\(start: (.*), stop: (.*?)\)", payload["query"]).groups()
        begin, end = datetime.fromisoformat(begin), datetime.fromisoformat(end)
        rows = [r for r in source_rows[bucket] if begin <= datetime.fromisoformat(r["_time"]) < end]
        return httpx.Response(200, text=csv_response(rows))

    real_client = httpx.Client
    monkeypatch.setattr(
        advisor_dataset.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(query), **kwargs),
    )
    monkeypatch.setenv("INFLUX_URL", "http://influx.test")
    monkeypatch.setenv("INFLUX_ORG", "test-org")
    monkeypatch.setenv("INFLUX_TOKEN", "secret-token")
    return main(
        [
            "advisor-dataset",
            "--start",
            "2026-09-11T00:00:00Z",
            "--train-stop",
            "2026-09-11T00:20:00Z",
            "--validation-stop",
            "2026-09-11T00:40:00Z",
            "--stop",
            "2026-09-11T01:00:00Z",
            "--output-dir",
            str(tmp_path / "dataset"),
            *extra,
        ]
    )


def test_export_preserves_values_gaps_entity_identity_and_shared_time_splits(
    tmp_path, monkeypatch, source_rows
):
    assert run_build(tmp_path, monkeypatch, source_rows) == 0
    output = tmp_path / "dataset"
    devices = pq.read_table(output / "device.parquet").to_pylist()
    assert {r["entity_id"] for r in devices} == {"sensor-a", "sensor-b"}
    assert len([r for r in devices if r["field"] == "temp_c_mean"]) == 4
    assert {r["split"] for r in devices} == {"train", "validation", "test"}
    assert all(r["time"].tzinfo is not None for r in devices)
    assert all(r["manual_label"] is None for r in devices)
    assert "secret-token" not in (output / "data-card.json").read_text()
    rooms = pq.read_table(output / "room.parquet").to_pylist()
    assert (
        next(r["value"] for r in rooms if r["time"].minute == 0 and r["field"] == "temp_c_mean")
        == 27.5
    )
    hosts = pq.read_table(output / "host.parquet").to_pylist()
    assert {r["interval_minutes"] for r in hosts} == {5}
    containers = pq.read_table(output / "container.parquet").to_pylist()
    assert {r["measurement"] for r in containers} == {
        "docker_container_cpu",
        "docker_container_health",
    }
    assert any("unhealthy" in r["weak_labels"] for r in containers)
    assert any("sensor-read-failure" in r["weak_labels"] for r in devices)
    assert not any(
        "sensor-read-failure" in r["weak_labels"] for r in devices if r["time"].minute == 2
    )
    with (output / "weak-labels.csv").open(newline="") as handle:
        events = list(csv.DictReader(handle))
    assert any(
        r["label"] == "all-devices-silent"
        and r["start"].endswith("00:01:00+00:00")
        and r["stop"].endswith("00:20:00+00:00")
        for r in events
    )
    card = json.loads((output / "data-card.json").read_text())
    assert card["manual_labels"]["reviewed_intervals"] == 0
    assert card["tables"]["device"]["entities"]["sensor-a"]["missing_windows"] == 36
    assert "time-based" in (output / "data-card.md").read_text()
    with (output / "manual-labels-template.csv").open(newline="") as handle:
        review = list(csv.DictReader(handle))
    assert review and all(not r["label"] for r in review)
    assert all("00:20" <= r["start"][11:16] < "00:40" for r in review)


def test_manual_review_is_separate_from_weak_labels(tmp_path, monkeypatch, source_rows):
    labels = tmp_path / "review.csv"
    labels.write_text(
        "entity_type,entity_id,start,stop,label,note\n"
        "device,sensor-a,2026-09-11T00:20:00Z,2026-09-11T00:21:00Z,"
        "normal,Compared with raw observations\n",
        encoding="utf-8",
    )
    assert run_build(tmp_path, monkeypatch, source_rows, ["--manual-labels", str(labels)]) == 0
    rows = pq.read_table(tmp_path / "dataset/device.parquet").to_pylist()
    assert {r["manual_label"] for r in rows if r["time"].minute == 20} == {"normal"}
    assert all(r["manual_label"] is None for r in rows if r["time"].minute != 20)
    card = json.loads((tmp_path / "dataset/data-card.json").read_text())
    assert card["manual_labels"]["reviewed_intervals"] == 1


@pytest.mark.parametrize("bad_time", ["2026-09-11T00:00:00", "2026-09-11T00:01:00Z"])
def test_rejects_ambiguous_or_unaligned_boundaries(tmp_path, bad_time):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "advisor-dataset",
                "--start",
                bad_time,
                "--stop",
                "2026-09-12T00:00:00Z",
                "--train-stop",
                "2026-09-11T12:00:00Z",
                "--validation-stop",
                "2026-09-11T18:00:00Z",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert error.value.code == 2


def test_existing_dataset_is_never_overwritten(tmp_path, monkeypatch, source_rows):
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "keep.txt").write_text("previous export")
    assert run_build(tmp_path, monkeypatch, source_rows) == 1
    assert (output / "keep.txt").read_text() == "previous export"


def test_influx_stream_error_does_not_publish_a_dataset(tmp_path, monkeypatch):
    from edgeops_ai.training.advisor_dataset import read_records

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=",error,reference\r\n,query failed,123\r\n")
        ),
        base_url="http://influx.test",
    )
    with client, pytest.raises(ValueError, match="InfluxDB"):
        list(
            read_records(
                client,
                "sensors_1m",
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
            )
        )


@pytest.mark.parametrize("value", ["NaN", "-inf", "1e309"])
def test_rejects_nonfinite_measurements(tmp_path, monkeypatch, source_rows, value):
    source_rows["sensors_1m"][0]["_value"] = value
    assert run_build(tmp_path, monkeypatch, source_rows) == 1
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".advisor-*"))


def test_counter_and_gap_state_survive_query_boundaries(tmp_path, monkeypatch, source_rows):
    source_rows["sensors_1m"] += [
        point(
            "05:59",
            _measurement="device_status",
            field="sensor_read_failures_total_last",
            value="7",
            room="",
        ),
        point(
            "06:00",
            _measurement="device_status",
            field="sensor_read_failures_total_last",
            value="8",
            room="",
        ),
        point("06:20"),
    ]
    assert run_build(tmp_path, monkeypatch, source_rows, ["--stop", "2026-09-11T07:00:00Z"]) == 0
    rows = pq.read_table(tmp_path / "dataset/device.parquet").to_pylist()
    assert any("sensor-read-failure" in r["weak_labels"] for r in rows if r["time"].hour == 6)
    with (tmp_path / "dataset/weak-labels.csv").open(newline="") as handle:
        events = list(csv.DictReader(handle))
    assert any(
        r["start"].endswith("00:41:00+00:00") and r["stop"].endswith("06:20:00+00:00")
        for r in events
    )


def test_conflicting_tag_churn_values_fail_without_publishing(tmp_path, monkeypatch, source_rows):
    source_rows["sensors_1m"][1]["_value"] = "21"
    assert run_build(tmp_path, monkeypatch, source_rows) == 1
    assert not (tmp_path / "dataset").exists()


def test_manual_gap_labels_are_retained_without_fabricating_measurements(
    tmp_path, monkeypatch, source_rows
):
    labels = tmp_path / "review.csv"
    labels.write_text(
        "entity_type,entity_id,start,stop,label,note\n"
        "device,*,2026-09-11T00:21:00Z,2026-09-11T00:40:00Z,"
        "confirmed-outage,Checked incident log\n",
        encoding="utf-8",
    )
    assert run_build(tmp_path, monkeypatch, source_rows, ["--manual-labels", str(labels)]) == 0
    card = json.loads((tmp_path / "dataset/data-card.json").read_text())
    assert card["manual_labels"]["reviewed_intervals"] == 1
    rows = pq.read_table(tmp_path / "dataset/device.parquet").to_pylist()
    assert not any(20 < r["time"].minute < 40 for r in rows)


def test_numeric_looking_string_metadata_keeps_its_source_type(tmp_path, monkeypatch, source_rows):
    source_rows["sensors_1m"].append(
        point(
            "00:20",
            _measurement="device_status",
            room="",
            field="firmware_version_last",
            value="1.10",
            _datatype="string",
        )
    )
    assert run_build(tmp_path, monkeypatch, source_rows) == 0
    rows = pq.read_table(tmp_path / "dataset/device.parquet").to_pylist()
    firmware = next(row for row in rows if row["field"] == "firmware_version_last")
    assert firmware["text"] == "1.10"
    assert firmware["value"] is None


def test_manual_reviews_must_stay_inside_validation(tmp_path, monkeypatch, source_rows):
    labels = tmp_path / "review.csv"
    labels.write_text(
        "entity_type,entity_id,start,stop,label,note\n"
        "device,sensor-a,2026-09-11T00:00:00Z,2026-09-11T00:01:00Z,"
        "normal,Reviewed training observation\n",
        encoding="utf-8",
    )
    assert run_build(tmp_path, monkeypatch, source_rows, ["--manual-labels", str(labels)]) == 1
    assert not (tmp_path / "dataset").exists()


def test_container_state_changes_preserve_both_aggregate_series(tmp_path, monkeypatch, source_rows):
    for status, count in [("restarting", "1"), ("running", "29")]:
        source_rows["metrics_5m"].append(
            point(
                "00:20",
                _measurement="docker_container_status",
                device_id="",
                room="",
                host="pi",
                container_name="ingest",
                container_status=status,
                field="exitcode_count",
                value=count,
            )
        )
    assert run_build(tmp_path, monkeypatch, source_rows) == 0
    rows = pq.read_table(tmp_path / "dataset/container.parquet").to_pylist()
    statuses = [r for r in rows if r["field"] == "exitcode_count"]
    assert {r["value"] for r in statuses} == {1.0, 29.0}
    assert {json.loads(r["dimensions"])["container_status"] for r in statuses} == {
        "running",
        "restarting",
    }
    assert {r["entity_id"] for r in statuses} == {"pi/ingest"}
