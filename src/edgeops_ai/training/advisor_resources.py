"""Offline host workload assessment from observed pressure and one-step TTM evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from edgeops_ai.forecast_advice import apply_container_context, classify, findings_for
from edgeops_ai.forecasting import ATOL
from edgeops_ai.training.advisor_dataset import SCHEMA, load_manual_labels, timestamp
from edgeops_ai.training.advisor_forecast import load_candidate
from edgeops_ai.training.advisor_health import split_at
from edgeops_ai.training.advisor_labels import file_hash
from edgeops_ai.training.reports import read_report

STEP = timedelta(minutes=5)
CONTEXT_FIELDS = {
    "usage_active_mean",
    "used_percent_mean",
    "health_status_last",
    "oomkilled_last",
    "restart_count_last",
    "uptime_ns_last",
    "container_id_last",
    "started_at_last",
}
EVENTS = {"build", "normal-activity", "deploy", "reset", "resource-pressure"}


def load_context(input_dir: Path, card: dict) -> tuple[dict, dict[str, str]]:
    points: dict[tuple[str, datetime], dict] = defaultdict(dict)
    containers: dict[tuple[str, datetime], dict] = defaultdict(dict)
    hashes = {}
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    cuts = [timestamp(card["split"][name]) for name in ("train_stop", "validation_stop")]
    for kind in ("host", "container"):
        if kind not in card["tables"]:
            continue
        path = input_dir / f"{kind}.parquet"
        hashes[path.name] = file_hash(path)
        source = pq.ParquetFile(path)
        if (
            source.schema_arrow != SCHEMA
            or source.metadata.num_rows != card["tables"][kind]["rows"]
            or hashes[path.name] != card["tables"][kind]["sha256"]
        ):
            raise ValueError(f"Invalid resource {kind} schema, row count or SHA-256")
        for batch in source.iter_batches():
            table = pa.Table.from_batches([batch])
            selected = table.filter(
                pc.is_in(table["field"], value_set=pa.array(sorted(CONTEXT_FIELDS)))
            )
            for row in selected.to_pylist():
                entity, at = row["entity_id"], row["time"]
                if (
                    not entity
                    or not start <= at < stop
                    or at.second
                    or at.microsecond
                    or at.minute % 5
                    or row["interval_minutes"] != 5
                    or row["split"] != split_at(at, *cuts)
                ):
                    raise ValueError("Invalid resource identity, timestamp, cadence or split")
                value = row["value"] if row["value"] is not None else row["text"]
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError("Resource observations must be finite")
                dimensions = json.loads(row["dimensions"])
                field, measurement = row["field"], row["measurement"]
                if kind == "host":
                    if (
                        measurement == "cpu"
                        and field == "usage_active_mean"
                        and dimensions.get("cpu") == "cpu-total"
                    ):
                        points[entity, at]["cpu_percent"] = value
                    elif measurement == "mem" and field == "used_percent_mean":
                        points[entity, at]["memory_percent"] = (
                            value if isinstance(value, (int, float)) and 0 <= value <= 100 else None
                        )
                    continue
                if (
                    measurement not in {"docker_container_health", "docker_container_status"}
                    or "/" not in entity
                ):
                    continue
                point = containers[entity, at]
                if field in point and point[field] != value:
                    point["conflict"] = True
                point[field] = value
                version = tuple(
                    dimensions.get(k, "")
                    for k in (
                        "container_version",
                        "container_image",
                        "org.opencontainers.image.revision",
                    )
                )
                point.setdefault("versions", set()).add(version)
    apply_container_context(points, containers)
    return dict(points), hashes


def review_findings(findings, metadata, labels, cuts, reviews_path, minutes):
    rows = []
    for method, items in findings.items():
        for item in items:
            row = asdict(item)
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "finding": row,
                        "method": method,
                        "sources": metadata["sources"],
                        "candidate": metadata["candidate_sha256"],
                    },
                    sort_keys=True,
                    default=str,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            for key in ("start", "stop", "observed_at"):
                row[key] = row[key].isoformat()
            row.update(
                id=digest,
                method=method,
                split=split_at(item.observed_at - timedelta(microseconds=1), *cuts),
                confidence=None,
                verdict="",
                note="",
            )
            rows.append(row)
    reviews = {}
    if reviews_path:
        with reviews_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"id", "verdict", "note"}.issubset(reader.fieldnames or []):
                raise ValueError("Resource reviews require id, verdict and note columns")
            for row in reader:
                verdict, note = (row["verdict"] or "").strip(), (row["note"] or "").strip()
                if (
                    row["id"] in reviews
                    or verdict not in {"", "true-positive", "false-positive"}
                    or (verdict and not note)
                ):
                    raise ValueError("Invalid or duplicate resource advice review")
                reviews[row["id"]] = verdict, note
        if set(reviews) - {row["id"] for row in rows}:
            raise ValueError("Unknown resource review id; source, model or policy evidence changed")
    for row in rows:
        row["verdict"], row["note"] = reviews.get(row["id"], ("", ""))
    days = (cuts[1] - cuts[0]).total_seconds() / 86400
    cases = [label for label in labels if cuts[0] <= label["start"] < label["stop"] <= cuts[1]]
    comparison = {}
    for method in findings:
        validation = [
            row for row in rows if row["method"] == method and row["split"] == "validation"
        ]
        warnings = [row for row in validation if row["severity"] == "warning"]
        unknown = sum(not row["verdict"] for row in validation)
        false = sum(row["verdict"] == "false-positive" for row in warnings)
        incidents, normal = [], []
        for case in cases:
            matched = [
                row
                for row in warnings
                if row["entity_id"] == case["entity_id"]
                and timestamp(row["start"]) < case["stop"]
                and timestamp(row["stop"]) > case["start"]
                and timestamp(row["evidence"]["first_detected_at"]) <= case["stop"]
            ]
            if case["label"] == "resource-pressure":
                delay = min(
                    (
                        max(
                            0,
                            (
                                timestamp(row["evidence"]["first_detected_at"]) - case["start"]
                            ).total_seconds()
                            / 60,
                        )
                        for row in matched
                        if row["verdict"] == "true-positive"
                    ),
                    default=None,
                )
                incidents.append(
                    {
                        "entity_id": case["entity_id"],
                        "start": case["start"].isoformat(),
                        "stop": case["stop"].isoformat(),
                        "detection_delay_minutes": delay,
                    }
                )
            elif case["label"] == "normal-activity":
                normal.append(
                    {
                        "entity_id": case["entity_id"],
                        "start": case["start"].isoformat(),
                        "unexpected_warnings": len(matched),
                    }
                )
        comparison[method] = {
            "validation_days": days,
            "advice": len(validation),
            "warnings": len(warnings),
            "unreviewed": unknown,
            "confirmed_false_positive_warnings": false,
            "false_positive_warnings_per_day": false / days if not unknown else None,
            "incident_cases": incidents,
            "missed_incidents": sum(c["detection_delay_minutes"] is None for c in incidents),
            "normal_cases": normal,
            "incidents_within_limit": bool(incidents)
            and all(
                c["detection_delay_minutes"] is not None and c["detection_delay_minutes"] <= minutes
                for c in incidents
            ),
        }
    ttm = comparison["ttm"]
    accepted = (
        metadata["available"]
        and all(item["unreviewed"] == 0 for item in comparison.values())
        and ttm["false_positive_warnings_per_day"] is not None
        and ttm["false_positive_warnings_per_day"] < 1
        and ttm["incidents_within_limit"]
        and bool(ttm["normal_cases"])
        and all(case["unexpected_warnings"] == 0 for case in ttm["normal_cases"])
    )
    metadata["comparison"] = comparison
    metadata["acceptance"] = {
        "passed": accepted,
        "warning_rate_limit_per_day_exclusive": 1,
        "detection_delay_limit_minutes": minutes,
        "requires_reviewed_normal_and_incident_cases": True,
    }
    return rows


def assess_resources(
    input_dir: Path,
    candidate: Path,
    *,
    workload_events: Path | None = None,
    cpu_threshold: float = 90,
    workload_minutes: int = 15,
    resource_reviews: Path | None = None,
) -> dict[str, Any]:
    if (
        not math.isfinite(cpu_threshold)
        or not 0 < cpu_threshold <= 100
        or workload_minutes <= 0
        or workload_minutes % 5
    ):
        raise ValueError(
            "CPU threshold must be in (0,100]; persistence must use whole five-minute samples"
        )
    card_path = input_dir / "data-card.json"
    card = read_report(card_path)
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    labels = load_manual_labels(workload_events, start, stop)
    review_hash = file_hash(resource_reviews) if resource_reviews else None
    if any(label["entity_type"] != "host" or label["label"] not in EVENTS for label in labels):
        raise ValueError(
            "Host workload labels: build, normal-activity, deploy, reset, resource-pressure"
        )
    boundaries = set()
    for label in labels:
        if label["label"] in {"deploy", "reset"}:
            at = label["start"]
            while at <= label["stop"]:
                boundaries.add((label["entity_id"], at))
                at += STEP
    model, windows, candidate_hashes = load_candidate(
        input_dir, candidate, one_step=True, boundaries=boundaries
    )
    if model.contract["profile"] != "host-cpu":
        raise ValueError("Resource assessment requires the host-cpu artifact contract")
    points, sources = load_context(input_dir, card)
    sources["data-card.json"] = file_hash(card_path)
    if workload_events:
        sources["workload_events"] = file_hash(workload_events)
    if any(label["entity_id"] not in {entity for entity, _ in points} for label in labels):
        raise ValueError("Workload event references an unknown host")
    predictions: dict[tuple[str, datetime], tuple[float, float, datetime]] = {}
    training: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for split, items in windows.items():
        for window in items:
            at, entity = window["target_times"][0], window["entity_id"]
            predicted = float(model.predict(window["past"][0], window["times"])[0, 0, 0])
            if not 0 <= predicted <= 100:
                raise ValueError("CPU expectation must remain between 0 and 100 percent")
            actual = float(window["future"][0, 0, 0])
            prior_value = float(window["past"][0, -1, 0])
            predictions[entity, at] = (predicted, prior_value, window["times"][0])
            if split == "train":
                training[entity]["ttm"].append(abs(actual - predicted))
                training[entity]["last_value"].append(abs(actual - prior_value))
    calibration = {
        entity: {
            method: {
                "scale": max(float(np.quantile(values, 0.99)), ATOL),
                "training_samples": len(values),
                "residual_sha256": hashlib.sha256(
                    np.asarray(values, dtype=np.float64).tobytes()
                ).hexdigest(),
            }
            for method, values in methods.items()
        }
        for entity, methods in training.items()
    }
    calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    cuts = [timestamp(card["split"][name]) for name in ("train_stop", "validation_stop")]
    observations = []
    for (entity, at), context in sorted(points.items()):
        expected, last, begin = predictions.get((entity, at), (None, None, None))
        event = next(
            (
                label
                for label in labels
                if label["entity_id"] == entity
                and label["label"] != "resource-pressure"
                and label["start"] <= at < label["stop"]
            ),
            None,
        )
        row = {
            "entity_id": entity,
            "time": at.isoformat(),
            "split": split_at(at, *cuts),
            "cpu_percent": context.get("cpu_percent"),
            "memory_percent": context.get("memory_percent"),
            "expected_cpu_percent": expected,
            "last_value_cpu_percent": last,
            "context_start": begin.isoformat() if begin else None,
            "expectation_origin": at.isoformat() if expected is not None else None,
            "last_input_start": (at - STEP).isoformat() if expected is not None else None,
            "model_status": "available"
            if expected is not None and entity in calibration
            else "uncalibrated"
            if expected is not None
            else "insufficient-context",
            "affected_services": ",".join(sorted(context.get("affected_services", []))),
            "health_observed": ",".join(sorted(context.get("health_observed", []))),
            "health_status": "unhealthy-observed"
            if context.get("affected_services")
            else "no-unhealthy-observed"
            if context.get("health_observed")
            else "unavailable",
            "container_boundary": context.get("container_boundary", False),
            "host_boundary": (entity, at) in boundaries,
            "activity": event["label"] if event else "unknown",
            "confidence": None,
            "model_version": model.manifest["model_version"],
            "model_sha256": candidate_hashes["model.onnx"],
            "calibration_sha256": calibration_hash,
        }
        for method, baseline in (("ttm", expected), ("last_value", last)):
            row[f"{method}_score"] = (
                abs(row["cpu_percent"] - baseline) / calibration[entity][method]["scale"]
                if baseline is not None and entity in calibration
                else None
            )
        row["state"] = classify(row, "ttm", cpu_threshold)
        observations.append(row)
    findings = {
        method: findings_for(observations, method, cpu_threshold, workload_minutes)
        for method in ("ttm", "rules", "last_value")
    }
    latest = {row["entity_id"]: row for row in observations}
    for row in latest.values():
        if timestamp(row["time"]) + STEP < stop:
            row["state"], row["model_status"] = "insufficient-evidence", "stale"
    metadata = {
        "available": bool(calibration) and {entity for entity, _ in points}.issubset(calibration),
        "model_version": model.manifest["model_version"],
        "contract": model.contract,
        "candidate_sha256": candidate_hashes,
        "sources": sources,
        "lead_minutes": 5,
        "cpu_threshold_percent": cpu_threshold,
        "sustained_minutes": workload_minutes,
        "calibration": calibration,
        "calibration_sha256": calibration_hash,
        "calibration_method": (
            "99th percentile absolute one-step training residual per host; no held-out fitting"
        ),
        "missing_host_boundary_evidence": not any(
            label["label"] in {"deploy", "reset"} for label in labels
        ),
        "deployment_approved": False,
        "coverage": {
            split: {
                "observations": sum(row["split"] == split for row in observations),
                "scored_observations": sum(
                    row["split"] == split and row["ttm_score"] is not None for row in observations
                ),
            }
            for split in ("train", "validation", "test")
        },
        "review_sha256": review_hash,
    }
    review_rows = review_findings(
        findings, metadata, labels, cuts, resource_reviews, workload_minutes
    )
    serialized = [row for row in review_rows if row["method"] == "ttm"]
    paths = {name: input_dir / name for name in sources if name != "workload_events"}
    if workload_events:
        paths["workload_events"] = workload_events
    if (
        any(file_hash(path) != sources[name] for name, path in paths.items())
        or any(file_hash(candidate / name) != digest for name, digest in candidate_hashes.items())
        or (resource_reviews is not None and file_hash(resource_reviews) != review_hash)
    ):
        raise ValueError("Resource source or candidate changed during replay")
    return {
        "findings": serialized,
        "observations": observations,
        "states": list(latest.values()),
        "report": metadata,
        "reviews": review_rows,
    }
