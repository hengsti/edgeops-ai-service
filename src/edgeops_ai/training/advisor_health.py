"""Replay device/pipeline health detectors and a training-only IsolationForest baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import IsolationForest

from edgeops_ai.advisor import (
    DAY,
    Advice,
    DeviceSample,
    PipelineSample,
    day_window,
    device_findings,
    pipeline_findings,
)
from edgeops_ai.influx_reader import DEVICE_FIELDS, PIPELINE_FIELDS
from edgeops_ai.training.advisor_dataset import SCHEMA, load_manual_labels, timestamp
from edgeops_ai.training.advisor_labels import file_hash
from edgeops_ai.training.reports import read_report, write_report

BASELINE_FEATURES = ("median_rssi_dbm", "median_free_mem_bytes")
SCORE_EPSILON = 1e-12
EXPECTED = {
    "unnoticed-outage": {"device.all-silent"},
    "sensor-fault": {"device.sensor-read-failures"},
    "weak-wifi": {"device.weak-wifi"},
    "possible-memory-leak": {"device.memory-decline"},
    "reconnects-and-failures": {"device.reconnects", "device.delivery-failures"},
    "iaq-glitches": {"device.iaq-quality"},
    "calibration-offset": {"device.pressure-offset"},
    "pipeline-errors": {"pipeline.errors"},
}


def load_devices(
    input_dir: Path, *, fields: Mapping[tuple[str, str], str] = DEVICE_FIELDS
) -> list[DeviceSample]:
    parquet = pq.ParquetFile(input_dir / "device.parquet")
    if parquet.schema_arrow != SCHEMA:
        raise ValueError("Expected the advisor-1 Parquet schema")
    samples: dict[tuple[str, datetime], DeviceSample] = {}
    sensor_minutes: set[tuple[str, datetime]] = set()
    selected = pa.array(sorted({field for _, field in fields}))
    for batch in parquet.iter_batches(
        columns=[
            "time",
            "entity_id",
            "measurement",
            "field",
            "value",
            "text",
            "dimensions",
            "interval_minutes",
        ]
    ):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["field"], value_set=selected))
        for row in table.to_pylist():
            name = fields.get((row["measurement"], row["field"]))
            if name is None:
                continue
            if row["interval_minutes"] != 1:
                raise ValueError("Device aggregates must use one-minute windows")
            key = row["entity_id"], row["time"]
            room = json.loads(row["dimensions"]).get("room", "")
            sample = samples.setdefault(key, DeviceSample(row["time"], row["entity_id"], {}))
            if room:
                if sample.room and sample.room != room:
                    raise ValueError("Conflicting room tags in one device minute")
                sample = samples[key] = replace(sample, room=room)
            value = row["value"] if row["value"] is not None else row["text"]
            if value is None or (name in sample.values and sample.values[name] != value):
                raise ValueError("Missing or conflicting health aggregate values")
            sample.values[name] = value
            if row["measurement"] == "bme680":
                sensor_minutes.add(key)
    for key in sensor_minutes:
        samples[key].values.setdefault("sensor_samples", 1)
    return [samples[key] for key in sorted(samples)]


def load_pipeline(path: Path, start: datetime, stop: datetime) -> list[PipelineSample]:
    """Read the annotated CSV exported by an InfluxDB raw-metrics query."""
    samples = []
    header = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for values in csv.reader(handle):
            if not values or values[0].startswith("#"):
                continue
            if {"_time", "_measurement", "_field", "_value"}.issubset(values):
                header = values
                continue
            if header is None or len(header) != len(values):
                raise ValueError("Invalid InfluxDB pipeline CSV")
            row = dict(zip(header, values, strict=True))
            if row["_field"] not in PIPELINE_FIELDS:
                continue
            at, value = timestamp(row["_time"]), float(row["_value"])
            if not start <= at < stop or not math.isfinite(value) or value < 0:
                raise ValueError("Pipeline counter outside replay bounds or invalid")
            dimensions = {
                k: v
                for k, v in row.items()
                if v and k not in {"", "result", "table"} and not k.startswith("_")
            }
            entity = json.dumps({"measurement": row["_measurement"], **dimensions}, sort_keys=True)
            samples.append(PipelineSample(at, entity, row["_field"], value))
    if not samples:
        raise ValueError("Pipeline CSV contains no supported counters")
    return samples


def split_at(at: datetime, train_stop: datetime, validation_stop: datetime) -> str:
    return "train" if at < train_stop else "validation" if at < validation_stop else "test"


def reference_case(name: str, row: dict[str, Any]) -> bool:
    if name == "unnoticed-outage":
        return (
            row["entity_id"] == "*"
            and timestamp(row["start"]) < timestamp("2026-09-15T00:00:00Z")
            and timestamp(row["stop"]) > timestamp("2026-09-11T00:00:00Z")
        )
    if name == "sensor-fault":
        return row["entity_id"].endswith("ad89")
    if name == "weak-wifi":
        return row["entity_id"].endswith(("84be", "57ad"))
    if name == "calibration-offset":
        return row["evidence"].get("room") == "kitchen"
    return True


def baseline_findings(
    samples: list[DeviceSample],
    start: datetime,
    train_stop: datetime,
    validation_stop: datetime,
    stop: datetime,
    labels: list[dict[str, Any]],
) -> tuple[list[Advice], list[dict[str, Any]], dict[str, Any]]:
    hour = timedelta(hours=1)
    grouped: dict[tuple[str, datetime], list[DeviceSample]] = defaultdict(list)
    identities = sorted({row.device_id for row in samples})
    for row in sorted(samples, key=lambda row: (row.device_id, row.at)):
        if all(name in row.values for name in ("rssi", "free_mem")):
            grouped[(row.device_id, row.at.replace(minute=0, second=0, microsecond=0))].append(row)
    windows: dict[str, list[tuple[datetime, list[float]]]] = defaultdict(list)
    for (device, begin), points in sorted(grouped.items()):
        end = begin + hour
        if (
            len(points) < 45
            or begin < start
            or end > stop
            or any(begin < cut < end for cut in (train_stop, validation_stop))
        ):
            continue
        vector = [
            median(float(row.values[name]) for row in points) for name in ("rssi", "free_mem")
        ]
        windows[device].append((begin, vector))
    metadata: dict[str, Any] = {
        "model": "IsolationForest",
        "sklearn_version": sklearn_version,
        "features": BASELINE_FEATURES,
        "n_estimators": 64,
        "max_samples": 256,
        "contamination": "auto",
        "random_state": 42,
        "n_jobs": 1,
        "decision_threshold": -SCORE_EPSILON,
        "devices": {},
    }
    findings, scores = [], []
    for device in identities:
        device_windows = windows[device]
        training = [
            vector
            for begin, vector in device_windows
            if begin + hour <= train_stop
            and not any(
                label["entity_type"] == "device"
                and label["entity_id"] in {device, "*"}
                and label["label"] != "normal"
                and label["start"] < begin + hour
                and begin < label["stop"]
                for label in labels
            )
        ]
        metadata["devices"][device] = {
            "training_windows": len(training),
            "training_stop": train_stop.isoformat(),
            "feature_sha256": hashlib.sha256(json.dumps(training).encode()).hexdigest(),
            "status": "trained" if len(training) >= 24 else "insufficient-training-data",
        }
        if len(training) < 24:
            continue
        model = IsolationForest(
            n_estimators=64,
            max_samples=min(256, len(training)),
            contamination="auto",
            random_state=42,
            n_jobs=1,
        ).fit(training)
        decision = model.decision_function([vector for _, vector in device_windows])
        daily: dict[datetime, list[float]] = defaultdict(list)
        run, prior, prior_split = 0, None, None
        for (begin, _), raw_score in zip(device_windows, decision, strict=True):
            score = float(raw_score)
            split = split_at(begin, train_stop, validation_stop)
            scores.append(
                {
                    "entity_id": device,
                    "start": begin.isoformat(),
                    "stop": (begin + hour).isoformat(),
                    "split": split,
                    "score": score,
                }
            )
            outlier = score < -SCORE_EPSILON
            run = (
                run + 1
                if prior == begin - hour and prior_split == split and outlier
                else int(outlier)
            )
            if run >= 3:
                period, _ = day_window(begin, start, stop, (train_stop, validation_stop))
                daily[period].append(score)
            prior, prior_split = begin, split
        for day, outliers in sorted(daily.items()):
            begin, end = day_window(day, start, stop, (train_stop, validation_stop))
            findings.append(
                Advice(
                    "device.multivariate-anomaly",
                    device,
                    begin,
                    end,
                    end,
                    f"{device}: unusual metrics persisted for at least three adjacent hours.",
                    "Inspect RSSI, free memory and counters; the score does not identify a cause.",
                    {
                        "minimum_decision_score": min(outliers),
                        "sustained_outlier_hours": len(outliers),
                    },
                    detector="model:isolation-forest@1",
                )
            )
    return findings, scores, metadata


def replay(
    input_dir: Path,
    output: Path,
    *,
    pipeline_path: Path | None = None,
    reviews_path: Path | None = None,
    forecast_candidate: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Output already exists; choose a new report directory")
    card_path, device_path = input_dir / "data-card.json", input_dir / "device.parquet"
    card = read_report(card_path)
    if card.get("schema_version") != "advisor-1":
        raise ValueError("Expected an advisor-1 dataset")
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    train_stop = timestamp(card["split"]["train_stop"])
    validation_stop = timestamp(card["split"]["validation_stop"])
    if not start < train_stop < validation_stop < stop or card["split"].get("shuffle") is not False:
        raise ValueError("Expected chronological, non-shuffled dataset boundaries")
    if file_hash(device_path) != card["tables"]["device"]["sha256"]:
        raise ValueError("Source device Parquet hash mismatch")
    if pq.ParquetFile(device_path).metadata.num_rows != card["tables"]["device"]["rows"]:
        raise ValueError("Source device Parquet row count mismatch")
    labels_path = input_dir / "manual-labels.csv"
    source_paths = {
        "data_card_sha256": card_path,
        "device_parquet_sha256": device_path,
    }
    if labels_path.exists() or forecast_candidate is None:
        source_paths["manual_labels_sha256"] = labels_path
    if pipeline_path is not None:
        source_paths["pipeline_csv_sha256"] = pipeline_path
    source_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    labels = load_manual_labels(labels_path if labels_path.exists() else None, start, stop)
    if any(
        label["start"] < cut < label["stop"]
        for label in labels
        for cut in (train_stop, validation_stop)
    ):
        raise ValueError("Reviewed labels must stay inside one split")
    samples = load_devices(input_dir)
    findings = device_findings(samples, start, stop, (train_stop, validation_stop))
    print(f"Replayed {len(samples)} device minutes", flush=True)
    baseline, scores, metadata = baseline_findings(
        samples, start, train_stop, validation_stop, stop, labels
    )
    findings.extend(baseline)
    residual_scores: list[dict[str, Any]] = []
    residual_metadata = None
    if forecast_candidate is not None:
        from edgeops_ai.training.advisor_residuals import residual_findings

        residuals, residual_scores, residual_metadata = residual_findings(
            input_dir, forecast_candidate
        )
        if {row.device_id for row in samples} != {row["entity_id"] for row in residual_scores}:
            raise ValueError("Forecast inputs do not cover every replay device")
        findings.extend(residuals)
    pipeline = load_pipeline(pipeline_path, start, stop) if pipeline_path else []
    findings.extend(
        replace(
            f,
            start=max(f.start, start),
            stop=min(f.stop, stop),
            observed_at=min(f.observed_at, stop),
        )
        for f in pipeline_findings(pipeline, (train_stop, validation_stop))
    )
    findings.sort(key=lambda f: (f.observed_at, f.id))
    # Keep existing review IDs independent of additive subject metadata.
    review_ids = {
        f.id: hashlib.sha256(
            json.dumps(
                {
                    "finding": {key: value for key, value in asdict(f).items() if key != "source"},
                    "sources": source_hashes,
                },
                sort_keys=True,
                default=str,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        for f in findings
    }
    reviews = {}
    if reviews_path is not None:
        with reviews_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"id", "verdict", "note"}.issubset(reader.fieldnames or []):
                raise ValueError("Advice reviews require id, verdict and note columns")
            for review_row in reader:
                if review_row["id"] in reviews:
                    raise ValueError("Duplicate advice review id")
                verdict = (review_row["verdict"] or "").strip()
                note = (review_row["note"] or "").strip()
                if verdict not in {"", "true-positive", "false-positive"} or (verdict and not note):
                    raise ValueError(
                        "Advice verdict needs true-positive/false-positive and a review note"
                    )
                reviews[review_row["id"]] = (verdict, note)
        if set(reviews) - set(review_ids.values()):
            raise ValueError("Unknown advice review id; replay the same dataset and pipeline input")
    rows = []
    for finding in findings:
        row: dict[str, Any] = {**asdict(finding), "id": review_ids[finding.id]}
        for key in ("start", "stop", "observed_at"):
            row[key] = row[key].isoformat()
        event_time = finding.observed_at
        if finding.code not in {"device.all-silent", "device.telemetry-stale"}:
            event_time -= timedelta(microseconds=1)
        row["split"] = split_at(event_time, train_stop, validation_stop)
        row["verdict"], row["note"] = reviews.get(row["id"], ("", ""))
        if not row["verdict"] and finding.code in {"device.all-silent", "device.telemetry-stale"}:
            matched = next(
                (
                    label
                    for label in labels
                    if label["entity_type"] == "device"
                    and label["entity_id"] in {finding.entity_id, "*"}
                    and label["label"]
                    in {
                        "all-devices-silent",
                        "server-outage",
                        "device-interruption",
                        "device-unresponsive",
                    }
                    and label["start"] <= finding.start < finding.stop <= label["stop"]
                ),
                None,
            )
            if matched is not None:
                row["verdict"], row["note"] = "true-positive", matched["note"]
        rows.append(row)
    validation = [row for row in rows if row["split"] == "validation"]
    unknown = sum(not row["verdict"] for row in validation)
    false = sum(row["verdict"] == "false-positive" for row in validation)
    days = (validation_stop - train_stop).total_seconds() / DAY.total_seconds()
    rate = false / days if not unknown else None
    codes = {finding.code for finding in findings}
    coverage = {name: expected.issubset(codes) for name, expected in EXPECTED.items()}
    reproduced = {
        name: expected.issubset(
            {
                row["code"]
                for row in rows
                if row["verdict"] == "true-positive" and reference_case(name, row)
            }
        )
        for name, expected in EXPECTED.items()
    }
    trained = bool(metadata["devices"]) and all(
        device["status"] == "trained" for device in metadata["devices"].values()
    )
    report = {
        "schema_version": "advisor-health-1",
        "start": start.isoformat(),
        "stop": stop.isoformat(),
        "sources": {
            **source_hashes,
            "advice_reviews_sha256": file_hash(reviews_path) if reviews_path else None,
        },
        "device_minutes": len(samples),
        "pipeline_samples": len(pipeline),
        "pipeline_metrics": sorted({sample.metric for sample in pipeline}),
        "baseline": metadata,
        "historical_reproduction": reproduced,
        "signal_coverage": coverage,
        "advice_counts": {code: sum(f.code == code for f in findings) for code in sorted(codes)},
        "false_positives": {
            "validation_days": days,
            "advice": len(validation),
            "unreviewed": unknown,
            "confirmed_false_positives": false,
            "per_day": rate,
            "passed": rate is not None and rate < 1,
        },
        "acceptance": {
            "passed": all(reproduced.values()) and trained and rate is not None and rate < 1
        },
    }
    if residual_metadata is not None:
        residual_metadata["comparison"] = {}
        for name, detector in (
            ("isolation_forest", "model:isolation-forest@1"),
            ("ttm", "model:ttm-residual@1"),
        ):
            items = [row for row in validation if row["detector"] == detector]
            unreviewed = sum(not row["verdict"] for row in items)
            false_positives = sum(row["verdict"] == "false-positive" for row in items)
            residual_metadata["comparison"][name] = {
                "validation_start": train_stop.isoformat(),
                "validation_stop": validation_stop.isoformat(),
                "advice": len(items),
                "unreviewed": unreviewed,
                "confirmed_false_positives": false_positives,
                "per_day": None if unreviewed else false_positives / days,
            }
        report["forecast_residuals"] = residual_metadata
    if any(file_hash(path) != source_hashes[name] for name, path in source_paths.items()):
        raise ValueError("Source files changed during replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".advisor-health-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        write_report(report, staging / "report.json")
        write_report({"findings": rows}, staging / "advice.json")
        if residual_scores:
            with (staging / "forecast-residuals.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(residual_scores[0]))
                writer.writeheader()
                writer.writerows(residual_scores)
        with (staging / "advice-review.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "id",
                    "split",
                    "code",
                    "entity_id",
                    "start",
                    "stop",
                    "observed_at",
                    "severity",
                    "summary",
                    "recommendation",
                    "evidence",
                    "detector",
                    "verdict",
                    "note",
                ),
            )
            writer.writeheader()
            writer.writerows(
                {
                    **{key: value for key, value in row.items() if key != "source"},
                    "evidence": json.dumps(row["evidence"], sort_keys=True),
                }
                for row in rows
            )
        with (staging / "anomaly-scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("entity_id", "start", "stop", "split", "score")
            )
            writer.writeheader()
            writer.writerows(scores)
        staging.rename(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pipeline-csv", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument(
        "--forecast-candidate", type=Path, help="Validated #26 climate candidate directory"
    )
    args = parser.parse_args(argv)
    try:
        report = replay(
            args.input_dir,
            args.output_dir,
            pipeline_path=args.pipeline_csv,
            reviews_path=args.reviews,
            forecast_candidate=args.forecast_candidate,
        )
        print(f"Report: {args.output_dir}; acceptance passed: {report['acceptance']['passed']}")
        return 0 if report["acceptance"]["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
        print(f"Advisor health replay failed: {error}")
        return 1
