"""Device and pipeline health facts, independent of HTTP and model training."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import linear_regression, median

MINUTE = timedelta(minutes=1)
DAY = timedelta(days=1)
COUNTERS = ("sensor_read_failures_total", "samples_failed_total", "mqtt_connects_total")


@dataclass(frozen=True)
class DeviceSample:
    at: datetime
    device_id: str
    values: dict[str, float | str]
    room: str = ""


@dataclass(frozen=True)
class PipelineSample:
    at: datetime
    entity_id: str
    metric: str
    value: float


@dataclass(frozen=True)
class Advice:
    code: str
    entity_id: str
    start: datetime
    stop: datetime
    observed_at: datetime
    summary: str
    recommendation: str
    evidence: dict[str, float | str] = field(default_factory=dict)
    severity: str = "warning"
    detector: str = "rule:device-health@1"

    @property
    def id(self) -> str:
        return f"{self.code}:{self.entity_id}:{self.start.isoformat()}"


def check_time(at: datetime) -> None:
    if at.utcoffset() != timedelta(0):
        raise ValueError("Health samples require explicit UTC timestamps")


def day_start(at: datetime) -> datetime:
    return at.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def day_window(
    at: datetime,
    start: datetime,
    stop: datetime,
    cuts: tuple[datetime, ...] = (),
) -> tuple[datetime, datetime]:
    begin, end = max(start, day_start(at)), min(stop, day_start(at) + DAY)
    return max([begin, *(cut for cut in cuts if cut <= at)]), min(
        [end, *(cut for cut in cuts if cut > at)]
    )


def counter_increases(previous: DeviceSample, current: DeviceSample) -> dict[str, float]:
    if current.at - previous.at != MINUTE:
        return {}
    before, after = previous.values, current.values
    if after.get("uptime_resets", 0) or (
        "uptime" in before
        and "uptime" in after
        and float(after["uptime"]) < float(before["uptime"])
    ):
        return {}
    return {
        name: float(after[name]) - float(before[name])
        for name in COUNTERS
        if name in before
        and name in after
        and not after.get(f"{name}_resets", 0)
        and float(after[name]) >= float(before[name])
    }


def _gaps(times: list[datetime], stop: datetime) -> list[tuple[datetime, datetime]]:
    return [
        (left + MINUTE, right)
        for left, right in zip(times, [*times[1:], stop], strict=True)
        if right - left - MINUTE >= 10 * MINUTE
    ]


def memory_decline(
    device: str,
    history: list[tuple[datetime, float]],
    begin: datetime,
    end: datetime,
) -> Advice | None:
    if len({day_start(day) for day, _ in history}) < 7:
        return None
    ages = [(day - history[0][0]) / DAY for day, _ in history]
    slope, _ = linear_regression(ages, [value for _, value in history])
    if slope > -256 or slope * ages[-1] > -4096:
        return None
    return Advice(
        "device.memory-decline",
        device,
        begin,
        end,
        end,
        f"{device}: minimum free memory declined; a firmware leak is suspected.",
        "Investigate allocations and logs; the trend does not prove a leak.",
        {"bytes_per_day": slope, "days": len(history), "fitted_decline_bytes": -slope * ages[-1]},
    )


def device_findings(
    samples: list[DeviceSample],
    start: datetime,
    stop: datetime,
    split_boundaries: tuple[datetime, ...] = (),
) -> list[Advice]:
    check_time(start)
    check_time(stop)
    if start >= stop:
        raise ValueError("Health replay needs increasing time boundaries")
    devices: dict[str, list[DeviceSample]] = defaultdict(list)
    seen: set[tuple[str, datetime]] = set()
    for row in sorted(samples, key=lambda row: (row.device_id, row.at)):
        check_time(row.at)
        if not row.device_id or not start <= row.at < stop:
            raise ValueError("Device sample outside the replay interval or missing identity")
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values.values()):
            raise ValueError("Health metrics must be finite")
        key = row.device_id, row.at
        if key in seen:
            raise ValueError("Duplicate device sample")
        seen.add(key)
        devices[row.device_id].append(row)

    findings = []
    presence = {
        device: [row.at for row in rows if row.values.get("sensor_samples", 0)]
        for device, rows in devices.items()
    }
    presence = {device: times for device, times in presence.items() if times}
    global_gaps = []
    if presence:
        union = sorted({at for times in presence.values() for at in times})
        global_gaps = _gaps(union, stop)
        for begin, end in global_gaps:
            findings.append(
                Advice(
                    "device.all-silent",
                    "*",
                    begin,
                    end,
                    begin + 10 * MINUTE,
                    "No sensor data from any observed device; the cause is unknown.",
                    "Check the server, MQTT broker, ingest service and Wi-Fi AP.",
                    {
                        "missing_minutes": (end - begin).total_seconds() / 60,
                        "devices": sum(times[0] < begin for times in presence.values()),
                    },
                    "critical",
                )
            )
        for device, times in presence.items():
            for begin, end in _gaps(times, stop):
                remaining = [(begin, end)]
                for global_begin, global_end in global_gaps:
                    remaining = [
                        part
                        for a, b in remaining
                        for part in ((a, min(b, global_begin)), (max(a, global_end), b))
                        if part[1] - part[0] >= 10 * MINUTE
                    ]
                for a, b in remaining:
                    findings.append(
                        Advice(
                            "device.telemetry-stale",
                            device,
                            a,
                            b,
                            a + 10 * MINUTE,
                            f"No sensor data from {device} since {a.isoformat()}.",
                            "Check device power, firmware responsiveness, Wi-Fi and the data path.",
                            {"missing_minutes": (b - a).total_seconds() / 60},
                        )
                    )

    pressure: dict[datetime, dict[str, float]] = defaultdict(dict)
    for device, rows in devices.items():
        for row in rows:
            if "pressure_hpa" in row.values:
                pressure[row.at][device] = float(row.values["pressure_hpa"])
    offsets: dict[tuple[str, datetime], list[float]] = defaultdict(list)
    for at, values in pressure.items():
        if len(values) >= 4:
            for device, value in values.items():
                reference = median(v for other, v in values.items() if other != device)
                period, _ = day_window(at, start, stop, split_boundaries)
                offsets[(device, period)].append(value - reference)

    for device, rows in devices.items():
        daily: dict[datetime, list[DeviceSample]] = defaultdict(list)
        increases: dict[datetime, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        memory: dict[tuple[int, datetime], list[float]] = defaultdict(list)
        segment = 0
        previous = None
        last_uptime = None
        for row in rows:
            day, _ = day_window(row.at, start, stop, split_boundaries)
            daily[day].append(row)
            if previous is not None:
                for name, value in counter_increases(previous, row).items():
                    increases[day][name] += value
            if "uptime" in row.values:
                if last_uptime is not None:
                    delta = float(row.values["uptime"]) - float(last_uptime.values["uptime"])
                    elapsed = (row.at - last_uptime.at).total_seconds()
                    if (
                        row.values.get("uptime_resets", 0)
                        or delta < 0
                        or abs(delta - elapsed) > 120
                    ):
                        segment += 1
                last_uptime = row
            if "free_mem" in row.values and "uptime" in row.values:
                memory[(segment, day)].append(float(row.values["free_mem"]))
            previous = row

        for day, points in sorted(daily.items()):
            begin, end = day_window(day, start, stop, split_boundaries)
            subject = f"{points[-1].room} ({device})" if points[-1].room else device

            def emit(
                code: str,
                summary: str,
                recommendation: str,
                evidence: dict[str, float | str],
                severity: str = "warning",
                device: str = device,
                begin: datetime = begin,
                end: datetime = end,
                room: str = points[-1].room,
            ) -> None:
                findings.append(
                    Advice(
                        f"device.{code}",
                        device,
                        begin,
                        end,
                        end,
                        summary,
                        recommendation,
                        {**evidence, "room": room},
                        severity,
                    )
                )

            for counter, code, threshold, action in (
                (
                    "sensor_read_failures_total",
                    "sensor-read-failures",
                    1,
                    f"Check the BME680 wiring and I2C on {subject}.",
                ),
                (
                    "samples_failed_total",
                    "delivery-failures",
                    1,
                    f"Inspect the device queue, Wi-Fi and MQTT delivery on {subject}.",
                ),
                (
                    "mqtt_connects_total",
                    "reconnects",
                    2,
                    f"Check Wi-Fi and broker logs for repeated MQTT reconnects on {subject}.",
                ),
            ):
                delta = increases[day].get(counter, 0)
                if delta >= threshold:
                    emit(
                        code,
                        f"{subject}: {delta:g} observed {counter} increments today.",
                        action,
                        {"increase": delta, "counter": counter},
                    )
            rssi = [float(p.values["rssi"]) for p in points if "rssi" in p.values]
            if len(rssi) >= 60 and sum(v <= -75 for v in rssi) / len(rssi) >= 0.8:
                emit(
                    "weak-wifi",
                    f"{subject}: persistently weak Wi-Fi signal.",
                    "Review AP/device placement and interference; weak signal can cause gaps.",
                    {"median_dbm": median(rssi), "observed_minutes": len(rssi)},
                )
            quality = [
                p
                for p in points
                if p.values.get("warmed_up") == "false"
                or (
                    float(p.values.get("iaq_min", 100)) <= 59.8
                    and (
                        ("warmed_up" not in p.values and float(p.values.get("uptime", 60)) < 60)
                        or float(p.values.get("iaq_mean", 0)) >= 70
                    )
                )
            ]
            if quality:
                scores = [float(p.values["iaq_min"]) for p in quality if "iaq_min" in p.values]
                quality_evidence: dict[str, float | str] = {"flagged_minutes": len(quality)}
                if scores:
                    quality_evidence["min_iaq"] = min(scores)
                emit(
                    "iaq-quality",
                    f"{subject}: IAQ quality needs review during warm-up or a transient.",
                    "Flag these IAQ windows; check warm-up before interpreting air quality.",
                    quality_evidence,
                    "info",
                )
            residuals = offsets.get((device, day), [])
            if len(residuals) >= 60:
                offset = median(residuals)
                deviation = median(abs(v - offset) for v in residuals)
                if abs(offset) >= 0.8 and deviation <= 0.25:
                    emit(
                        "pressure-offset",
                        f"{subject}: persistent pressure offset versus peer sensors.",
                        "Compare sensor elevation and placement before calibrating an offset.",
                        {
                            "offset_hpa": offset,
                            "median_absolute_deviation_hpa": deviation,
                            "aligned_minutes": len(residuals),
                        },
                    )

        for (boot, day), free_values in sorted(memory.items()):
            if len(free_values) < 60:
                continue
            history = sorted(
                (d, min(v))
                for (b, d), v in memory.items()
                if b == boot and day - 29 * DAY <= d <= day and len(v) >= 60
            )
            begin, end = day_window(day, start, stop, split_boundaries)
            finding = memory_decline(device, history, begin, end)
            if finding is not None:
                findings.append(finding)
    return sorted(findings, key=lambda f: (f.observed_at, f.id))


def pipeline_findings(
    samples: list[PipelineSample],
    split_boundaries: tuple[datetime, ...] = (),
) -> list[Advice]:
    previous: dict[tuple[str, str], PipelineSample] = {}
    daily: dict[tuple[str, datetime], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in sorted(samples, key=lambda row: (row.at, row.entity_id, row.metric)):
        check_time(row.at)
        if not row.entity_id or not row.metric or not math.isfinite(row.value) or row.value < 0:
            raise ValueError("Pipeline samples need identity and finite nonnegative counters")
        key = row.entity_id, row.metric
        before = previous.get(key)
        if before is not None:
            elapsed = row.at - before.at
            if elapsed == timedelta(0):
                raise ValueError("Duplicate pipeline counter sample")
            if elapsed <= 2 * MINUTE and row.value >= before.value:
                period = max(
                    [day_start(row.at), *(cut for cut in split_boundaries if cut <= row.at)]
                )
                daily[(row.entity_id, period)][row.metric] += row.value - before.value
        previous[key] = row
    return [
        Advice(
            "pipeline.errors",
            entity,
            day,
            min([day_start(day) + DAY, *(cut for cut in split_boundaries if cut > day)]),
            min([day_start(day) + DAY, *(cut for cut in split_boundaries if cut > day)]),
            f"{entity}: pipeline error counters increased.",
            "Inspect the DLQ, ingest errors and broker drops; identify the source first.",
            {name: value for name, value in deltas.items() if value > 0},
            detector="rule:pipeline-counter-increase@1",
        )
        for (entity, day), deltas in sorted(daily.items())
        if any(v > 0 for v in deltas.values())
    ]
