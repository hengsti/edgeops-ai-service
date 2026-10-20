from datetime import UTC, datetime, timedelta

import pytest

from edgeops_ai.training.advisor_maintenance import battery_backtest, forecast

START = datetime(2026, 9, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def points(start=30, slope=-1, days=15):
    return [(START + i * DAY, start + slope * i) for i in range(days)]


def test_battery_eta_and_bounds_follow_decreasing_percentage():
    result = forecast(points(), "battery_percent", START + 14 * DAY)
    assert result["eta_days"] == pytest.approx(6)
    assert result["eta_range_days"] == pytest.approx([6, 6])
    assert result["slope_per_day"] == -1
    assert result["advice_due"] is True
    assert result["threshold"] == 10


def test_disk_and_memory_have_correct_units_and_threshold_direction():
    disk = forecast(points(60, 1), "disk_used_percent", START + 14 * DAY)
    memory = forecast(points(100_000, -1000), "free_mem_bytes", START + 14 * DAY)
    assert disk["eta_days"] == pytest.approx(16)
    assert memory["eta_days"] == pytest.approx((86_000 - 65536) / 1000)
    assert memory["unit"] == "bytes"


def test_robust_slope_ignores_a_build_cache_spike():
    rows = points(60, 1)
    rows[7] = rows[7][0], 99
    result = forecast(rows, "disk_used_percent", START + 14 * DAY)
    assert result["slope_per_day"] == pytest.approx(1)
    assert result["eta_days"] == pytest.approx(16)


def test_long_eta_is_reported_without_advice_and_horizon_is_configurable():
    result = forecast(points(90, -1), "battery_percent", START + 14 * DAY)
    assert result["eta_days"] == 66
    assert result["advice_due"] is False
    assert forecast(points(90, -1), "battery_percent", START + 14 * DAY, horizon_days=70)[
        "advice_due"
    ]


@pytest.mark.parametrize("slope", [0, 1])
def test_flat_and_recovering_batteries_have_no_exhaustion_eta(slope):
    result = forecast(points(30, slope), "battery_percent", START + 14 * DAY)
    assert result["eta_days"] is None
    assert result["advice_due"] is False


def test_confidence_interval_crossing_zero_does_not_raise_advice():
    rows = [(START + i * DAY, value) for i, value in enumerate([20, 18, 21, 17, 22, 16, 19])]
    result = forecast(rows, "battery_percent", START + 6 * DAY)
    assert result["advice_due"] is False
    assert result["eta_range_days"][1] is None


@pytest.mark.parametrize("kind", ["gap", "battery-replacement", "disk-cleanup"])
def test_interventions_and_missing_days_restart_history(kind):
    metric = "disk_used_percent" if kind == "disk-cleanup" else "battery_percent"
    rows = points(60, 1) if kind == "disk-cleanup" else points()
    if kind == "gap":
        del rows[10]
    else:
        rows[10:] = [
            (at, value + (30 if kind == "battery-replacement" else -15)) for at, value in rows[10:]
        ]
    result = forecast(rows, metric, START + 14 * DAY)
    assert result["status"] == "insufficient-history"
    assert result["eta_days"] is None


def test_stale_and_sparse_history_are_explicit():
    assert forecast(points(), "battery_percent", START + 17 * DAY)["status"] == "stale"
    assert forecast(points(), "battery_percent", START + 15 * DAY)["status"] == "stale"
    assert (
        forecast(points(days=3), "battery_percent", START + 2 * DAY)["status"]
        == "insufficient-history"
    )


def test_already_reached_threshold_is_zero_without_fabricating_a_trend():
    result = forecast(points(5, 0, 1), "battery_percent", START)
    assert result["status"] == "threshold-reached"
    assert result["eta_days"] == 0
    assert result["slope_per_day"] is None


def test_backtest_requires_an_actual_future_crossing():
    assert battery_backtest(points())["passed"] is None
    result = battery_backtest(points(days=25))
    assert result["passed"] is True
    assert result["cases"] and all(case["relative_error"] <= 0.2 for case in result["cases"])
    assert all(case["origin"] < case["actual_crossing"] for case in result["cases"])


def test_failed_backtest_is_not_reported_as_success():
    rows = points(days=15) + [(START + i * DAY, max(0, 16 - (i - 14) * 3)) for i in range(15, 23)]
    assert battery_backtest(rows)["passed"] is False


@pytest.mark.parametrize("problem", ["nan", "duplicate", "naive", "future", "range", "horizon"])
def test_invalid_forecast_inputs_are_rejected(problem):
    rows = points()
    options = {}
    if problem == "nan":
        rows[0] = rows[0][0], float("nan")
    elif problem == "duplicate":
        rows.append(rows[-1])
    elif problem == "naive":
        rows[0] = START.replace(tzinfo=None), 30
    elif problem == "future":
        rows.append((START + 20 * DAY, 20))
    elif problem == "range":
        rows[0] = START, 120
    else:
        options["horizon_days"] = 0
    with pytest.raises(ValueError):
        forecast(rows, "battery_percent", START + 14 * DAY, **options)
