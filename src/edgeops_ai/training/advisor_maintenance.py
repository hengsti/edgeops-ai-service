"""Robust time-to-threshold estimates from historical maintenance metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import theilslopes

from edgeops_ai.advisor import DAY, Advice, check_time, day_start
from edgeops_ai.training.advisor_dataset import SCHEMA, timestamp
from edgeops_ai.training.advisor_health import load_devices, split_at
from edgeops_ai.training.advisor_labels import file_hash
from edgeops_ai.training.reports import read_report, write_report

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
    fit = theilslopes(
        [value for _, value in history],
        [(at - history[0][0]) / DAY for at, _ in history],
        alpha=0.95,
    )
    slope, low, high = float(fit.slope), float(fit.low_slope), float(fit.high_slope)
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


def battery_backtest(
    points: list[tuple[datetime, float]],
    *,
    threshold: float = 10,
) -> dict[str, Any]:
    points = sorted(points)
    cases: list[dict[str, Any]] = []
    for index in range(6, len(points) - 1):
        origin = points[index][0]
        estimate = forecast(points[: index + 1], "battery_percent", origin, threshold=threshold)
        if estimate["status"] != "forecast":
            continue
        previous_at, previous_value = points[index]
        for at, value in points[index + 1 :]:
            if at - previous_at != DAY or value - previous_value >= 10:
                break
            if value <= threshold:
                actual = (at - origin) / DAY
                cases.append(
                    {
                        "origin": origin.isoformat(),
                        "actual_crossing": at.isoformat(),
                        "predicted_days": estimate["eta_days"],
                        "actual_days": actual,
                        "relative_error": abs(estimate["eta_days"] - actual) / actual,
                    }
                )
                break
            previous_at, previous_value = at, value
    return {
        "cases": cases,
        "passed": all(case["relative_error"] <= 0.2 for case in cases) if cases else None,
        "criterion": "Each evaluated battery ETA is within 20% of its observed threshold time",
    }


def load_series(
    input_dir: Path,
    card: dict[str, Any],
    battery_path: Path | None = None,
) -> dict[tuple[str, str, str], list[tuple[datetime, float]]]:
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    train_stop = timestamp(card["split"]["train_stop"])
    validation_stop = timestamp(card["split"]["validation_stop"])
    daily: dict[tuple[str, str, datetime], list[float]] = defaultdict(list)
    seen: set[tuple[str, str, datetime]] = set()
    reset_days: set[tuple[str, str, datetime]] = set()

    def add(metric: str, entity: str, at: datetime, value: float) -> None:
        check_time(at)
        if not entity or not start <= at < stop or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid maintenance identity, time or value")
        if POLICIES[metric][2] == "%" and value > 100:
            raise ValueError("Maintenance percentages must be between 0 and 100")
        key = metric, entity, at
        if key in seen:
            raise ValueError("Duplicate maintenance observation")
        seen.add(key)
        daily[(metric, entity, day_start(at))].append(value)

    host = pq.ParquetFile(input_dir / "host.parquet")
    if host.schema_arrow != SCHEMA:
        raise ValueError("Expected the advisor-1 Parquet schema")
    for batch in host.iter_batches():
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.equal(table["field"], "used_percent_mean"))
        for row in table.to_pylist():
            if row["measurement"] != "disk":
                continue
            dimensions = json.loads(row["dimensions"])
            if dimensions.get("path") != "/":
                continue
            if not row["entity_id"]:
                raise ValueError("Root disk aggregates need a host identity")
            at = row["time"]
            if row["interval_minutes"] != 5 or at.minute % 5 or at.second or at.microsecond:
                raise ValueError("Disk aggregates must use aligned five-minute windows")
            entity = json.dumps({"host": row["entity_id"], **dimensions}, sort_keys=True)
            add("disk_used_percent", entity, at, row["value"])

    previous: dict[str, tuple[datetime, float]] = {}
    for sample in load_devices(
        input_dir,
        fields={
            ("device_status", "free_mem_min"): "free_mem",
            ("device_status", "uptime_last"): "uptime",
            ("device_status", "uptime_resets"): "uptime_resets",
        },
    ):
        if "uptime" not in sample.values:
            continue
        uptime = float(sample.values["uptime"])
        resets = float(sample.values.get("uptime_resets", 0))
        if not math.isfinite(uptime) or uptime < 0 or not math.isfinite(resets) or resets < 0:
            raise ValueError("Invalid device uptime/reset evidence")
        before = previous.get(sample.device_id)
        if resets or (
            before
            and (
                uptime < before[1]
                or abs(uptime - before[1] - (sample.at - before[0]).total_seconds()) > 120
            )
        ):
            reset_days.add(("free_mem_bytes", sample.device_id, day_start(sample.at)))
        previous[sample.device_id] = sample.at, uptime
        if "free_mem" in sample.values:
            add("free_mem_bytes", sample.device_id, sample.at, float(sample.values["free_mem"]))

    if battery_path is not None:
        with battery_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"time", "entity_id", "battery_percent"}.issubset(reader.fieldnames or []):
                raise ValueError("Battery CSV needs time, entity_id, battery_percent columns")
            for row in reader:
                add(
                    "battery_percent",
                    row["entity_id"],
                    timestamp(row["time"]),
                    float(row["battery_percent"]),
                )
    series: dict[tuple[str, str, str], list[tuple[datetime, float]]] = defaultdict(list)
    for (metric, entity, day), values in sorted(daily.items()):
        minimum = {"disk_used_percent": 144, "free_mem_bytes": 720, "battery_percent": 1}[metric]
        if (
            len(values) < minimum
            or (metric, entity, day) in reset_days
            or day < start
            or day + DAY > stop
            or any(day < cut < day + DAY for cut in (train_stop, validation_stop))
        ):
            continue
        split = split_at(day, train_stop, validation_stop)
        series[(metric, entity, split)].append((day + DAY, median(values)))
    return dict(series)


def replay(
    input_dir: Path,
    output: Path,
    *,
    battery_path: Path | None = None,
    horizon_days: float = 30,
    battery_threshold: float = 10,
    disk_threshold: float = 90,
    memory_threshold: float = 65536,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Output already exists; choose a new report directory")
    paths = {
        "data_card_sha256": input_dir / "data-card.json",
        "device_parquet_sha256": input_dir / "device.parquet",
        "host_parquet_sha256": input_dir / "host.parquet",
    }
    if battery_path is not None:
        paths["battery_csv_sha256"] = battery_path
    hashes = {name: file_hash(path) for name, path in paths.items()}
    card = read_report(paths["data_card_sha256"])
    if card.get("schema_version") != "advisor-1":
        raise ValueError("Expected an advisor-1 dataset")
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    train_stop = timestamp(card["split"]["train_stop"])
    validation_stop = timestamp(card["split"]["validation_stop"])
    if not start < train_stop < validation_stop < stop or card["split"].get("shuffle") is not False:
        raise ValueError("Expected chronological, non-shuffled dataset boundaries")
    for kind in ("device", "host"):
        metadata = card["tables"][kind]
        if hashes[f"{kind}_parquet_sha256"] != metadata["sha256"]:
            raise ValueError(f"Source {kind} Parquet hash mismatch")
        if pq.ParquetFile(input_dir / f"{kind}.parquet").metadata.num_rows != metadata["rows"]:
            raise ValueError(f"Source {kind} Parquet row count mismatch")
    thresholds = {
        "battery_percent": battery_threshold,
        "disk_used_percent": disk_threshold,
        "free_mem_bytes": memory_threshold,
    }
    for metric, threshold in thresholds.items():
        forecast([], metric, stop, threshold=threshold, horizon_days=horizon_days)
    series = load_series(input_dir, card, battery_path)
    latest: dict[tuple[str, str], tuple[str, list[tuple[datetime, float]]]] = {}
    cases: list[dict[str, Any]] = []
    for (metric, entity, split), points in series.items():
        key = metric, entity
        if key not in latest or points[-1][0] > latest[key][1][-1][0]:
            latest[key] = split, points
        if metric == "battery_percent" and split != "train":
            cases.extend(
                {**case, "entity_id": entity, "split": split}
                for case in battery_backtest(points, threshold=battery_threshold)["cases"]
            )
    forecasts, advice = [], []
    for (metric, entity), (split, points) in sorted(latest.items()):
        item = {
            **forecast(
                points, metric, stop, threshold=thresholds[metric], horizon_days=horizon_days
            ),
            "entity_id": entity,
            "split": split,
        }
        forecasts.append(item)
        if item["advice_due"]:
            finding = Advice(
                f"maintenance.{metric}",
                entity,
                timestamp(item["start"]),
                timestamp(item["stop"]),
                stop,
                f"{entity}: {metric} is estimated to reach "
                f"{item['threshold']:g} {item['unit']} in {item['eta_days']:.1f} days.",
                POLICIES[metric][3],
                {
                    "metric": metric,
                    "current": item["current"],
                    "threshold": item["threshold"],
                    "eta_days": item["eta_days"],
                    "eta_lower_days": item["eta_range_days"][0],
                    "eta_upper_days": item["eta_range_days"][1],
                },
                detector="trend:theil-sen@1",
            )
            advice.append(
                {
                    **asdict(finding),
                    "start": finding.start.isoformat(),
                    "stop": finding.stop.isoformat(),
                    "observed_at": finding.observed_at.isoformat(),
                }
            )
    report = {
        "schema_version": "advisor-maintenance-1",
        "start": start.isoformat(),
        "stop": stop.isoformat(),
        "sources": hashes,
        "thresholds": thresholds,
        "horizon_days": horizon_days,
        "forecast_count": len(forecasts),
        "advice_count": len(advice),
        "battery_backtest": {
            "cases": cases,
            "passed": all(case["relative_error"] <= 0.2 for case in cases) if cases else None,
        },
        "seasonal_model_available": False,
        "eta_range_method": (
            "95% Theil-Sen slope bounds propagated from the last daily median; "
            "not a calibrated crossing-time prediction interval"
        ),
    }
    if any(file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Source files changed during replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".advisor-maintenance-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        write_report(report, staging / "report.json")
        write_report(
            {"as_of": stop.isoformat(), "forecasts": forecasts}, staging / "state-report.json"
        )
        write_report({"findings": advice}, staging / "advice.json")
        staging.rename(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--battery-csv", type=Path)
    parser.add_argument("--horizon-days", type=float, default=30)
    parser.add_argument("--battery-threshold", type=float, default=10)
    parser.add_argument("--disk-threshold", type=float, default=90)
    parser.add_argument("--memory-threshold", type=float, default=65536)
    args = parser.parse_args(argv)
    try:
        report = replay(
            args.input_dir,
            args.output_dir,
            battery_path=args.battery_csv,
            horizon_days=args.horizon_days,
            battery_threshold=args.battery_threshold,
            disk_threshold=args.disk_threshold,
            memory_threshold=args.memory_threshold,
        )
        print(
            f"Report: {args.output_dir}; forecasts: {report['forecast_count']}; "
            f"advice: {report['advice_count']}"
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
        print(f"Advisor maintenance replay failed: {error}")
        return 1
