import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from edgeops_ai.training.advisor_climate import CLIMATE_FIELDS, replay
from edgeops_ai.training.advisor_dataset import SCHEMA
from edgeops_ai.training.advisor_health import load_devices
from edgeops_ai.training.cli import COMMANDS, main
from test_advisor_climate import MINUTE, START, samples


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    rows = []
    for sample in samples(240, iaq_mean=40):
        for field, name in {
            "rel_hum_perc_mean": "humidity_mean",
            "iaq_score_mean": "iaq_mean",
            "gas_ohm_mean": "gas_mean",
            "warmed_up_last": "warmed_up",
        }.items():
            if name not in sample.values:
                continue
            value = sample.values[name]
            rows.append(
                {
                    "time": sample.at,
                    "entity_id": sample.device_id,
                    "measurement": "bme680",
                    "field": field,
                    "value": value if isinstance(value, (float, int)) else None,
                    "text": value if isinstance(value, str) else None,
                    "dimensions": json.dumps({"room": sample.room}),
                    "interval_minutes": 1,
                    "split": "train"
                    if sample.at < START + 60 * MINUTE
                    else ("validation" if sample.at < START + 180 * MINUTE else "test"),
                    "weak_labels": [],
                    "manual_label": None,
                }
            )
    path = root / "device.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    card = {
        "schema_version": "advisor-1",
        "start": START.isoformat(),
        "stop": (START + 240 * MINUTE).isoformat(),
        "split": {
            "train_stop": (START + 60 * MINUTE).isoformat(),
            "validation_stop": (START + 180 * MINUTE).isoformat(),
            "shuffle": False,
        },
        "tables": {
            "device": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
        },
    }
    (root / "data-card.json").write_text(json.dumps(card), encoding="utf-8")
    return root


def test_climate_command_replays_real_parquet_and_preserves_inputs(exported, tmp_path):
    assert COMMANDS.get("advisor-climate") == "advisor_climate"
    path = exported / "device.parquet"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "climate"
    assert main(["advisor-climate", "--input-dir", str(exported), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    advice = json.loads((output / "advice.json").read_text())["findings"]
    assert report["advice_counts"] == {"climate.air-quality": 1, "climate.mould-risk": 1}
    assert report["sources"]["device_parquet_sha256"] == before
    assert {item["split"] for item in advice} == {"train", "validation"}
    assert all(item["room"] == "kitchen" for item in advice)
    assert report["forecast_available"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_loader_selects_climate_fields_without_changing_health_defaults(exported):
    climate = load_devices(exported, fields=CLIMATE_FIELDS)
    assert climate[0].values["humidity_mean"] == 65
    assert climate[0].values["gas_mean"] == 50_000
    assert climate[0].room == "kitchen"
    assert "humidity_mean" not in load_devices(exported)[0].values


@pytest.mark.parametrize("problem", ["hash", "rows", "shuffle", "bounds", "schema", "cadence"])
def test_invalid_dataset_does_not_publish_a_report(exported, tmp_path, problem):
    card_path = exported / "data-card.json"
    card = json.loads(card_path.read_text())
    if problem == "hash":
        card["tables"]["device"]["sha256"] = "wrong"
    elif problem == "rows":
        card["tables"]["device"]["rows"] += 1
    elif problem == "shuffle":
        card["split"]["shuffle"] = True
    elif problem == "bounds":
        card["split"]["validation_stop"] = card["start"]
    elif problem == "schema":
        card["schema_version"] = "unsupported"
    else:
        path = exported / "device.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["interval_minutes"] = 5
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        card["tables"]["device"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    card_path.write_text(json.dumps(card), encoding="utf-8")
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        replay(exported, output)
    assert not output.exists()


def test_existing_report_and_invalid_duration_are_not_overwritten(exported, tmp_path):
    output = tmp_path / "climate"
    replay(exported, output)
    before = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        replay(exported, output)
    assert (output / "report.json").read_bytes() == before
    assert (
        main(
            [
                "advisor-climate",
                "--input-dir",
                str(exported),
                "--output-dir",
                str(tmp_path / "bad"),
                "--humidity-minutes",
                "0",
            ]
        )
        == 1
    )
    assert not (tmp_path / "bad").exists()


def test_input_change_during_replay_does_not_publish(exported, tmp_path, monkeypatch):
    from edgeops_ai.training import advisor_climate

    original = advisor_climate.climate_findings

    def changing_input(*args, **kwargs):
        result = original(*args, **kwargs)
        with (exported / "data-card.json").open("a") as handle:
            handle.write(" ")
        return result

    monkeypatch.setattr(advisor_climate, "climate_findings", changing_input)
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="changed"):
        replay(exported, output)
    assert not output.exists()
