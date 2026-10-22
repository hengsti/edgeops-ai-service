import random
from datetime import UTC, datetime

import pytest
from scipy.stats import theilslopes

from edgeops_ai import maintenance
from edgeops_ai.advisor import DAY, MINUTE, DeviceSample

START = datetime(2026, 9, 1, tzinfo=UTC)


def test_runtime_forecast_preserves_offline_contract_and_scipy_bounds():
    from edgeops_ai.training.advisor_maintenance import forecast

    assert forecast is maintenance.forecast
    rng = random.Random(31)
    for days in (7, 15, 30):
        for values in (
            [70.0] * days,
            [60 + index for index in range(days)],
            [70 + rng.choice((-2, -1, 0, 1, 2)) for _ in range(days)],
            [60 + index / 10 + rng.random() for index in range(days)],
        ):
            points = [(START + index * DAY, value) for index, value in enumerate(values)]
            expected = theilslopes(values, list(range(days)), alpha=0.95)
            actual = maintenance.forecast(points, "disk_used_percent", points[-1][0])
            assert actual["slope_per_day"] == pytest.approx(expected.slope)
            assert actual["slope_95_percent_interval"] == pytest.approx(
                [expected.low_slope, expected.high_slope]
            )


def memory_samples(days=9, *, reset_day=None, missing_day=None):
    for day in range(days):
        if day == missing_day:
            continue
        for minute in range(720):
            offset = day * 1440 + minute
            uptime = offset * 60 + 2000
            if reset_day is not None and day >= reset_day:
                uptime -= reset_day * 86400
            yield DeviceSample(
                START + offset * MINUTE,
                "sensor",
                {"free_mem": 100_000 - 1000 * day, "uptime": uptime, "uptime_resets": 0},
                "office",
            )


def test_daily_summaries_preserve_memory_rule_and_eta_without_minute_history():
    rows = list(memory_samples())
    days = [
        item
        for day in range(9)
        for item in maintenance.summarize_day(
            [row for row in rows if START + day * DAY <= row.at < START + (day + 1) * DAY],
            [],
            START + day * DAY,
        )
    ]
    estimates, findings = maintenance.assess(days, START + 9 * DAY)
    estimate = estimates[0]
    assert estimate["entity_id"] == "sensor"
    assert estimate["observed_days"] == 9
    assert estimate["eta_days"] == pytest.approx((92_000 - 65536) / 1000)
    assert estimate["advice_due"] is True
    assert {item.code for item in findings} == {
        "device.memory-decline",
        "maintenance.free_mem_bytes",
    }


@pytest.mark.parametrize("problem", ["reset", "gap", "room", "missing-uptime"])
def test_memory_daily_history_breaks_at_discontinuities(problem):
    rows = list(
        memory_samples(
            reset_day=4 if problem == "reset" else None, missing_day=4 if problem == "gap" else None
        )
    )
    if problem in {"room", "missing-uptime"}:
        from dataclasses import replace

        rows = [
            replace(row, room="kitchen")
            if problem == "room" and row.at >= START + 4 * DAY
            else replace(row, values={"free_mem": row.values["free_mem"]})
            if problem == "missing-uptime" and START + 4 * DAY <= row.at < START + 5 * DAY
            else row
            for row in rows
        ]
    days = [
        item
        for day in range(9)
        for item in maintenance.summarize_day(
            [row for row in rows if START + day * DAY <= row.at < START + (day + 1) * DAY],
            [],
            START + day * DAY,
        )
    ]
    estimates, findings = maintenance.assess(days, START + 9 * DAY)
    assert estimates[0]["status"] == "insufficient-history"
    assert estimates[0]["advice_due"] is False
    assert findings == []


def test_resource_adapters_require_adjacent_fresh_evidence_and_preserve_identities():
    from edgeops_ai.resources import STEP, ResourceSample, resource_findings

    rows = [
        ResourceSample(START + index * STEP, "pi", "cpu", "usage_active_mean", 95)
        for index in range(3)
    ]
    findings = resource_findings(rows, START + 3 * STEP)
    assert [item.code for item in findings] == ["host.cpu-activity"]
    assert findings[0].severity == "info"
    assert resource_findings(rows[::2], START + 3 * STEP) == []
    assert resource_findings(rows, START + DAY) == []
    unhealthy = ResourceSample(
        START, "pi/influx", "docker_container_health", "health_status_last", "unhealthy"
    )
    findings = resource_findings([unhealthy], START + STEP)
    assert len(findings) == 1 and findings[0].entity_id == "pi/influx"
    assert (
        resource_findings(
            [ResourceSample(START, "pi", "docker_container_cpu", "usage_percent_mean", 350)],
            START + STEP,
        )
        == []
    )


def test_disk_daily_history_reports_nondue_and_unknown_estimates():
    from edgeops_ai.resources import STEP, ResourceSample

    days = []
    for day in range(9):
        resources = [
            ResourceSample(
                START + day * DAY + index * STEP,
                '{"host":"pi","path":"/"}',
                "disk",
                "used_percent_mean",
                40 + day / 10,
            )
            for index in range(144)
        ]
        days.extend(maintenance.summarize_day([], resources, START + day * DAY))
    estimates, findings = maintenance.assess(days, START + 9 * DAY)
    assert estimates[0]["eta_days"] > 30
    assert estimates[0]["advice_due"] is False
    assert findings == []
    assert maintenance.assess(days[-2:], START + 9 * DAY)[0][0]["status"] == "insufficient-history"
    assert maintenance.assess(days, START + 10 * DAY)[0][0]["status"] == "stale"


def test_reset_flag_is_honored_even_when_that_minute_has_no_uptime():
    rows = list(memory_samples(days=1))
    rows[100].values.pop("uptime")
    rows[100].values["uptime_resets"] = 1
    assert maintenance.summarize_day(rows, [], START)[0].valid is False


def test_cross_day_reboot_excludes_the_entire_reboot_day():
    rows = list(memory_samples(days=11, reset_day=4))
    summaries = [
        item
        for day in range(11)
        for item in maintenance.summarize_day(
            [row for row in rows if START + day * DAY <= row.at < START + (day + 1) * DAY],
            [],
            START + day * DAY,
        )
    ]
    estimates, findings = maintenance.assess(summaries, START + 11 * DAY)
    assert estimates[0]["observed_days"] == 6
    assert estimates[0]["status"] == "insufficient-history"
    assert findings == []


def test_room_change_is_honored_even_when_that_minute_has_no_uptime():
    from dataclasses import replace

    rows = list(memory_samples(days=1))
    rows[100] = replace(rows[100], room="kitchen", values={"free_mem": 100_000})
    assert maintenance.summarize_day(rows, [], START)[0].valid is False
