from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from edgeops_ai.advisor import DeviceSample
from edgeops_ai.climate import climate_findings

START = datetime(2026, 9, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def samples(minutes=120, device="a", room="kitchen", **changes):
    return [
        DeviceSample(
            START + i * MINUTE,
            device,
            {
                "humidity_mean": 65,
                "iaq_mean": 90,
                "gas_mean": 50_000,
                "warmed_up": "true",
                **changes,
            },
            room,
        )
        for i in range(minutes)
    ]


def findings(rows, minutes=120, **options):
    return climate_findings(rows, START, START + minutes * MINUTE, **options)


def test_sustained_humidity_has_room_causal_window_evidence_and_action():
    rows = samples()
    assert findings(rows[:-1]) == []
    advice = findings(rows)
    assert len(advice) == 1
    item = advice[0]
    assert item.code == "climate.mould-risk"
    assert item.entity_id == "kitchen"
    assert (item.start, item.stop, item.observed_at) == (
        START,
        START + 120 * MINUTE,
        START + 120 * MINUTE,
    )
    assert item.evidence["device_id"] == "a"
    assert item.evidence["humidity_min_percent"] == 65
    assert "kitchen" in item.recommendation and "10 minutes" in item.recommendation
    assert "drier" in item.recommendation
    assert "risk" in item.summary.lower()


@pytest.mark.parametrize("change", ["gap", "missing", "threshold", "clock", "room"])
def test_missing_minutes_fields_and_room_changes_break_humidity_duration(change):
    rows = samples()
    if change == "gap":
        del rows[60]
    elif change == "missing":
        rows[60].values.pop("humidity_mean")
    elif change == "threshold":
        rows[60].values["humidity_mean"] = 60
    elif change == "clock":
        rows[60].values["time_valid"] = "false"
    else:
        rows[60:] = [replace(row, room="office") for row in rows[60:]]
    assert findings(rows) == []


def test_humidity_hysteresis_does_not_flap_and_rearms_after_recovery():
    rows = samples(400)
    for row in rows[120:180]:
        row.values["humidity_mean"] = 59
    assert len(findings(rows, 400)) == 1
    rows[180].values["humidity_mean"] = 54
    assert [f.observed_at for f in findings(rows, 400)] == [
        START + 120 * MINUTE,
        START + 301 * MINUTE,
    ]


def test_poor_iaq_uses_higher_is_better_scale_and_reports_both_trends():
    rows = samples(15, humidity_mean=45, iaq_mean=50)
    for i, row in enumerate(rows):
        row.values.update(iaq_mean=55 - i, gas_mean=20_000 - 500 * i)
    (item,) = findings(rows, 15)
    assert item.code == "climate.air-quality"
    assert (item.start, item.stop) == (START, START + 15 * MINUTE)
    assert item.evidence["iaq_change"] == -14
    assert item.evidence["gas_change_ohm"] == -7000
    assert item.evidence["iaq_max_score"] == 55
    assert "source" in item.recommendation.lower()
    assert findings(samples(60, humidity_mean=45), 60) == []


@pytest.mark.parametrize("warmup", ["false", None, "unknown"])
def test_air_advice_requires_explicit_warmup_and_a_full_new_window(warmup):
    rows = samples(29, humidity_mean=45, iaq_mean=40)
    rows[14].values.pop("warmed_up")
    if warmup is not None:
        rows[14].values["warmed_up"] = warmup
    assert findings(rows, 29) == []


def test_poor_iaq_without_gas_does_not_invent_a_gas_trend():
    rows = samples(15, iaq_mean=50)
    for row in rows:
        row.values.pop("gas_mean")
    (item,) = findings(rows, 15)
    assert "gas_change_ohm" not in item.evidence


def test_air_quality_hysteresis_and_room_cooldown_cover_all_devices_and_rules():
    rows = samples(180, iaq_mean=40)
    for row in rows[15:20]:
        row.values["iaq_mean"] = 65
    for row in rows[30:45]:
        row.values["iaq_mean"] = 90
    rows += samples(180, device="b", iaq_mean=40)
    rows += samples(180, device="c", room="office", iaq_mean=40)
    advice = findings(rows, 180)
    kitchen = [item for item in advice if item.entity_id == "kitchen"]
    assert kitchen[0].observed_at == START + 15 * MINUTE
    assert all(
        right.observed_at - left.observed_at >= 60 * MINUTE for left, right in pairwise(kitchen)
    )
    assert any(item.code == "climate.mould-risk" for item in kitchen)
    assert any(item.entity_id == "office" for item in advice)
    assert len([item for item in kitchen if item.code == "climate.air-quality"]) == 2


def test_windows_do_not_cross_splits_and_cooldown_survives_a_split():
    rows = samples(135, iaq_mean=40)
    advice = findings(rows, 135, split_boundaries=(START + 125 * MINUTE,))
    assert all(not item.start < START + 125 * MINUTE < item.stop for item in advice)
    assert [item.observed_at for item in advice] == [START + 15 * MINUTE, START + 120 * MINUTE]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 101, "bad"])
def test_invalid_humidity_is_rejected(value):
    with pytest.raises(ValueError):
        findings(samples(humidity_mean=value))


def test_bounds_duplicates_unknown_rooms_and_duration_are_validated():
    with pytest.raises(ValueError, match="Duplicate"):
        findings(samples() + samples())
    with pytest.raises(ValueError):
        findings(samples(), minutes=0)
    with pytest.raises(ValueError):
        findings(samples(), humidity_minutes=0)
    with pytest.raises(ValueError):
        findings([replace(samples(1)[0], at=START.replace(tzinfo=None))])
    assert findings(samples(room="")) == []
