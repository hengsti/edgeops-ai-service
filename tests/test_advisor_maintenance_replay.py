import csv
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.training.advisor_dataset import SCHEMA
from edgeops_ai.training.advisor_maintenance import load_series, replay
from edgeops_ai.training.cli import COMMANDS, main
from test_advisor_maintenance import DAY, START


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    tables = {"device": [], "host": []}
    for day in range(30):
        for kind, count, cadence in (("device", 720, 1), ("host", 144, 5)):
            for minute in range(count):
                fields = (
                    {
                        "free_mem_min": 100_000 - day * 1000,
                        "uptime_last": 1000 + day * 86400 + minute * 60,
                    }
                    if kind == "device"
                    else {"used_percent_mean": 60 + day}
                )
                for field, value in fields.items():
                    tables[kind].append(
                        {
                            "time": START + day * DAY + DAY * (minute * cadence / 1440),
                            "entity_id": "sensor" if kind == "device" else "pi",
                            "measurement": "device_status" if kind == "device" else "disk",
                            "field": field,
                            "value": value,
                            "text": None,
                            "dimensions": "{}"
                            if kind == "device"
                            else '{"path":"/","device":"sd"}',
                            "interval_minutes": cadence,
                            "split": "train" if day < 10 else "validation" if day < 20 else "test",
                            "weak_labels": [],
                            "manual_label": None,
                        }
                    )
    card = {
        "schema_version": "advisor-1",
        "start": START.isoformat(),
        "stop": (START + 30 * DAY).isoformat(),
        "split": {
            "train_stop": (START + 10 * DAY).isoformat(),
            "validation_stop": (START + 20 * DAY).isoformat(),
            "shuffle": False,
        },
        "tables": {},
    }
    for kind, rows in tables.items():
        path = root / f"{kind}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        card["tables"][kind] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows),
        }
    (root / "data-card.json").write_text(json.dumps(card), encoding="utf-8")
    return root


def test_maintenance_command_writes_disk_memory_state_and_only_due_advice(exported, tmp_path):
    assert COMMANDS.get("advisor-maintenance") == "advisor_maintenance"
    output = tmp_path / "maintenance"
    assert (
        main(["advisor-maintenance", "--input-dir", str(exported), "--output-dir", str(output)])
        == 0
    )
    state = json.loads((output / "state-report.json").read_text())
    assert {row["metric"] for row in state["forecasts"]} == {"free_mem_bytes", "disk_used_percent"}
    assert all(row["observed_days"] == 10 and row["eta_days"] > 0 for row in state["forecasts"])
    advice = json.loads((output / "advice.json").read_text())["findings"]
    assert len(advice) == 2
    assert all(row["recommendation"] and row["evidence"]["eta_days"] <= 30 for row in advice)
    report = json.loads((output / "report.json").read_text())
    assert report["battery_backtest"]["passed"] is None
    assert report["seasonal_model_available"] is False


def test_battery_csv_is_optional_validated_and_hashed(exported, tmp_path):
    battery = tmp_path / "battery.csv"
    with battery.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "entity_id", "battery_percent"])
        for day in range(30):
            writer.writerow([(START + day * DAY).isoformat(), "hue/vorzimmer", 80 - day])
    output = tmp_path / "maintenance"
    report = replay(exported, output, battery_path=battery)
    assert (
        report["sources"]["battery_csv_sha256"] == hashlib.sha256(battery.read_bytes()).hexdigest()
    )
    state = json.loads((output / "state-report.json").read_text())
    (row,) = [row for row in state["forecasts"] if row["metric"] == "battery_percent"]
    assert row["eta_days"] == pytest.approx(41)
    assert row["advice_due"] is False
    battery.write_text("time,entity_id,battery_percent\n2026-09-01T00:00:00Z,hue,101\n")
    with pytest.raises(ValueError):
        replay(exported, tmp_path / "invalid", battery_path=battery)


def test_memory_reboot_day_is_excluded_and_does_not_bridge_history(exported):
    path = exported / "device.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if row["time"] >= START + 25 * DAY and row["field"] == "uptime_last":
            row["value"] -= 25 * 86400
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card = json.loads((exported / "data-card.json").read_text())
    series = load_series(exported, card)
    memory = series[("free_mem_bytes", "sensor", "test")]
    assert START + 26 * DAY not in {at for at, _ in memory}
    from edgeops_ai.training.advisor_maintenance import forecast

    assert forecast(memory, "free_mem_bytes", START + 30 * DAY)["status"] == "insufficient-history"


@pytest.mark.parametrize("problem", ["hash", "rows", "shuffle", "bounds", "schema", "horizon"])
def test_invalid_input_does_not_publish(exported, tmp_path, problem):
    path = exported / "data-card.json"
    card = json.loads(path.read_text())
    options = {}
    if problem == "hash":
        card["tables"]["host"]["sha256"] = "wrong"
    elif problem == "rows":
        card["tables"]["device"]["rows"] += 1
    elif problem == "shuffle":
        card["split"]["shuffle"] = True
    elif problem == "bounds":
        card["split"]["validation_stop"] = card["start"]
    elif problem == "schema":
        card["schema_version"] = "unsupported"
    else:
        options["horizon_days"] = 0
    path.write_text(json.dumps(card), encoding="utf-8")
    with pytest.raises(ValueError):
        replay(exported, tmp_path / "bad", **options)
    assert not (tmp_path / "bad").exists()


def test_output_and_sources_are_preserved(exported, tmp_path):
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in exported.iterdir()
    }
    output = tmp_path / "maintenance"
    replay(exported, output)
    report = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        replay(exported, output)
    assert (output / "report.json").read_bytes() == report
    assert before == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in exported.iterdir()
    }


@pytest.mark.parametrize("problem", ["identity", "cadence", "time"])
def test_invalid_disk_contract_is_rejected(exported, problem):
    path = exported / "host.parquet"
    rows = pq.read_table(path).to_pylist()
    if problem == "identity":
        rows[0]["entity_id"] = ""
    elif problem == "cadence":
        rows[0]["interval_minutes"] = 1
    else:
        rows[0]["time"] += DAY / 1440
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card = json.loads((exported / "data-card.json").read_text())
    with pytest.raises(ValueError):
        load_series(exported, card)


def test_source_changes_during_replay_do_not_publish(exported, tmp_path, monkeypatch):
    from edgeops_ai.training import advisor_maintenance

    original = advisor_maintenance.load_series

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with (exported / "data-card.json").open("a") as handle:
            handle.write(" ")
        return result

    monkeypatch.setattr(advisor_maintenance, "load_series", changed)
    with pytest.raises(ValueError, match="changed"):
        replay(exported, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
