"""Shared, dependency-free maintenance estimates for runtime and offline replay."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from statistics import NormalDist, median
from typing import Any

from edgeops_ai.advisor import DAY, Advice, DeviceSample, check_time, day_start, memory_decline
from edgeops_ai.resources import ResourceSample


def slope_bounds(values: list[float]) -> tuple[float, float, float]:
    """Theil-Sen slope and 95% rank bounds for consecutive daily observations."""
    n = len(values)
    slopes = sorted(
        (right - left) / (j - i)
        for i, left in enumerate(values)
        for j, right in enumerate(values)
        if j > i
    )
    ties = sum(k * (k - 1) * (2 * k + 5) for k in Counter(values).values())
    variance = (n * (n - 1) * (2 * n + 5) - ties) / 18
    width = NormalDist().inv_cdf(0.975) * math.sqrt(variance)
    low = max(round((len(slopes) - width) / 2) - 1, 0)
    high = min(round((len(slopes) + width) / 2), len(slopes) - 1)
    return median(slopes), slopes[low], slopes[high]


POLICIES = {
    "battery_percent": (10, -1, "%", "Replace the battery before the estimated threshold."),
    "disk_used_percent": (
        90,
        1,
        "%",
        "Review disk usage and remove unneeded build caches or expand storage.",
    ),
    "free_mem_bytes": (
        65536,
        -1,
        "bytes",
        "Inspect the device memory trend and firmware allocations.",
    ),
}


def forecast(
    points: list[tuple[datetime, float]],
    metric: str,
    as_of: datetime,
    *,
    threshold: float | None = None,
    horizon_days: float = 30,
) -> dict[str, Any]:
    check_time(as_of)
    default, direction, unit, _ = POLICIES[metric]
    threshold = default if threshold is None else threshold
    if (
        not math.isfinite(horizon_days)
        or horizon_days <= 0
        or not math.isfinite(threshold)
        or threshold < 0
        or (unit == "%" and threshold > 100)
    ):
        raise ValueError("Invalid maintenance horizon or threshold")
    seen = set()
    for at, value in points:
        check_time(at)
        if at > as_of or at != day_start(at) or at in seen:
            raise ValueError("Expected unique completed UTC daily observations before the forecast")
        if not math.isfinite(value) or value < 0 or (unit == "%" and value > 100):
            raise ValueError("Invalid maintenance value")
        seen.add(at)
    history: list[tuple[datetime, float]] = []
    for at, value in sorted(points):
        if history:
            previous = median(v for _, v in history[-3:])
            if (
                at - history[-1][0] != DAY
                or (metric == "battery_percent" and value - previous >= 10)
                or (metric == "disk_used_percent" and previous - value >= 5)
            ):
                history.clear()
        history.append((at, value))
        history = history[-30:]
    result: dict[str, Any] = {
        "metric": metric,
        "unit": unit,
        "as_of": as_of.isoformat(),
        "start": (history[0][0] - DAY).isoformat() if history else None,
        "stop": history[-1][0].isoformat() if history else None,
        "observed_days": len(history),
        "threshold": threshold,
        "current": history[-1][1] if history else None,
        "status": "insufficient-history",
        "slope_per_day": None,
        "slope_95_percent_interval": [None, None],
        "eta_days": None,
        "eta_range_days": [None, None],
        "advice_due": False,
    }
    if not history:
        return result
    if as_of - history[-1][0] >= DAY:
        result["status"] = "stale"
        return result
    distance = direction * (threshold - history[-1][1])
    if distance <= 0:
        result.update(
            status="threshold-reached", eta_days=0, eta_range_days=[0, 0], advice_due=True
        )
        return result
    if len(history) < 7:
        return result
    slope, low, high = slope_bounds([value for _, value in history])
    result.update(slope_per_day=slope, slope_95_percent_interval=[low, high])
    speed = direction * slope
    if speed <= 0:
        result["status"] = "not-approaching"
        return result
    slow, fast = sorted((direction * low, direction * high))
    eta = distance / speed
    # shortcut: ETA bounds propagate slope uncertainty only; calibrate with observed crossings.
    lower = distance / fast if fast > 0 else None
    upper = distance / slow if slow > 0 else None
    if not math.isfinite(eta) or any(
        v is not None and not math.isfinite(v) for v in (lower, upper)
    ):
        result["status"] = "uncertain-trend"
        return result
    result.update(
        status="forecast" if upper is not None else "uncertain-trend",
        eta_days=eta,
        eta_range_days=[lower, upper],
        advice_due=upper is not None and upper <= horizon_days,
    )
    return result


@dataclass(frozen=True)
class MaintenanceDay:
    at: datetime
    metric: str
    entity_id: str
    value: float | None
    minimum: float | None
    count: int
    valid: bool = True
    first: DeviceSample | None = None
    last: DeviceSample | None = None


def same_boot(before: DeviceSample, after: DeviceSample) -> bool:
    return (
        before.room == after.room
        and not after.values.get("uptime_resets", 0)
        and float(after.values["uptime"]) >= float(before.values["uptime"])
        and abs(
            float(after.values["uptime"])
            - float(before.values["uptime"])
            - (after.at - before.at).total_seconds()
        )
        <= 120
    )


def summarize_day(
    devices: list[DeviceSample],
    resources: list[ResourceSample],
    at: datetime,
) -> list[MaintenanceDay]:
    result = []
    groups: dict[str, list[DeviceSample]] = defaultdict(list)
    for row in devices:
        groups[row.device_id].append(row)
    for device, samples in sorted(groups.items()):
        rows = sorted((row for row in samples if "uptime" in row.values), key=lambda row: row.at)
        values = [float(row.values["free_mem"]) for row in rows if "free_mem" in row.values]
        valid = bool(rows) and not any(row.values.get("uptime_resets", 0) for row in samples)
        valid = valid and len({row.room for row in samples}) <= 1
        valid = valid and all(same_boot(a, b) for a, b in pairwise(rows))
        result.append(
            MaintenanceDay(
                at,
                "free_mem_bytes",
                device,
                median(values) if values else None,
                min(values) if values else None,
                len(values),
                valid,
                rows[0] if rows else None,
                rows[-1] if rows else None,
            )
        )
    disks: dict[str, list[float]] = defaultdict(list)
    for resource in resources:
        if resource.measurement == "disk" and resource.metric == "used_percent_mean":
            disks[resource.entity_id].append(float(resource.value))
    result.extend(
        MaintenanceDay(at, "disk_used_percent", entity, median(values), min(values), len(values))
        for entity, values in sorted(disks.items())
    )
    return result


def assess(
    days: list[MaintenanceDay],
    as_of: datetime,
    *,
    disk_threshold: float = 90,
    memory_threshold: float = 65536,
    horizon_days: float = 30,
) -> tuple[list[dict[str, Any]], list[Advice]]:
    series: dict[tuple[str, str], list[MaintenanceDay]] = defaultdict(list)
    for row in days:
        series[row.metric, row.entity_id].append(row)
    estimates, findings = [], []
    for (metric, entity), rows in sorted(series.items()):
        rows.sort(key=lambda row: row.at)
        history: list[MaintenanceDay] = []
        decline_history: list[tuple[datetime, float]] = []
        previous = None
        for row in rows:
            valid = row.valid
            if (
                previous is not None
                and previous.last is not None
                and row.first is not None
                and not same_boot(previous.last, row.first)
            ):
                valid = False
            continuous = previous is None or (
                row.at - previous.at == DAY
                and (
                    metric != "free_mem_bytes"
                    or (
                        previous.last is not None
                        and row.first is not None
                        and same_boot(previous.last, row.first)
                    )
                )
            )
            if not continuous or not valid:
                history.clear()
                decline_history.clear()
            previous = row
            if valid and row.minimum is not None and row.count >= 60:
                decline_history.append((row.at, row.minimum))
            else:
                decline_history.clear()
            if (
                valid
                and row.value is not None
                and row.count >= (720 if metric == "free_mem_bytes" else 144)
            ):
                history.append(row)
            else:
                history.clear()
        threshold = memory_threshold if metric == "free_mem_bytes" else disk_threshold
        item = {
            **forecast(
                [(row.at + DAY, row.value) for row in history if row.value is not None],
                metric,
                as_of,
                threshold=threshold,
                horizon_days=horizon_days,
            ),
            "entity_id": entity,
            "recommendation": POLICIES[metric][3],
        }
        estimates.append(item)
        if metric == "free_mem_bytes" and as_of - rows[-1].at < 2 * DAY:
            finding = memory_decline(entity, decline_history, rows[-1].at, rows[-1].at + DAY)
            if finding is not None:
                findings.append(finding)
        if item["advice_due"]:
            assert item["start"] is not None and item["stop"] is not None
            findings.append(
                Advice(
                    f"maintenance.{metric}",
                    entity,
                    datetime.fromisoformat(item["start"]),
                    datetime.fromisoformat(item["stop"]),
                    as_of,
                    f"{entity}: {metric} is estimated to reach {threshold:g} {item['unit']} "
                    f"in {item['eta_days']:.1f} days.",
                    POLICIES[metric][3],
                    {
                        "metric": metric,
                        "current": item["current"],
                        "threshold": threshold,
                        "eta_days": item["eta_days"],
                        "eta_lower_days": item["eta_range_days"][0],
                        "eta_upper_days": item["eta_range_days"][1],
                    },
                    detector="trend:theil-sen@1",
                )
            )
    return estimates, findings
