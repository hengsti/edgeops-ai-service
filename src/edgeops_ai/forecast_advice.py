"""Shared forecast advice policy for offline evaluation and optional runtime detectors."""

from datetime import datetime, timedelta
from itertools import groupby
from typing import Any

from edgeops_ai.advisor import Advice, check_time
from edgeops_ai.resources import STEP
from edgeops_ai.schemas import SourceIdentity

SUSTAINED_MINUTES = 15


def climate_projection(window: dict, prediction, model_version: str) -> tuple[dict, Advice | None]:
    current = float(window["past"][0, -1, 1])
    origin = window["times"][-1] + timedelta(minutes=1)
    targets = window["target_times"][:61]
    frame = {
        "device_id": window["entity_id"],
        "room": window["room"],
        "split": window["split"],
        "model_version": model_version,
        "context_start": window["times"][0].isoformat(),
        "context_stop": origin.isoformat(),
        "forecast_origin": origin.isoformat(),
        "last_input": window["times"][-1].isoformat(),
        "target_times": [at.isoformat() for at in targets],
        "temperature_c": prediction[:, 0].tolist(),
        "humidity_percent": prediction[:, 1].tolist(),
    }
    lead = next((i for i, value in enumerate(prediction[:, 1]) if value > 60), None)
    advice = None
    if current <= 60 and lead is not None and lead > 0:
        advice = Advice(
            "climate.humidity-forecast",
            window["room"],
            window["times"][0],
            origin,
            origin,
            f"{window['room']}: humidity is forecast to exceed 60% in about {lead} minutes.",
            f"Check humidity in {window['room']}; ventilate if outdoor air is drier, "
            "or use a dehumidifier.",
            {
                "device_id": window["entity_id"],
                "threshold_percent": 60,
                "minutes_to_crossing": lead,
                "crossing_at": targets[lead].isoformat(),
                "forecast_humidity_percent": float(prediction[lead, 1]),
                "current_humidity_percent": current,
                "model_version": model_version,
                "forecast_origin": origin.isoformat(),
            },
            detector="model:ttm-climate@1",
            source=SourceIdentity(
                kind="room", device_id=window["entity_id"], room=window["room"] or None
            ),
        )
    return frame, advice


def timestamp(value: str) -> datetime:
    at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    check_time(at)
    return at


def residual_advice(scores: list[dict], model_version: str, calibration_hash: str) -> list[Advice]:
    findings = []
    minute = timedelta(minutes=1)

    # Grouping and adjacency restart persistence at split, room and missing-data boundaries.
    def key(row):
        return row["entity_id"], row["room"], row["split"], row["channel"]

    for (device, room, _, channel), series in groupby(sorted(scores, key=key), key=key):
        run: list[dict[str, Any]] = []
        for row in [*series, None]:
            if run and (
                row is None
                or row["score"] <= 1
                or row["time"] != (run[-1]["at"] + minute).isoformat()
            ):
                if len(run) >= SUSTAINED_MINUTES:
                    begin, end = run[0]["at"], run[-1]["at"] + minute
                    findings.append(
                        Advice(
                            f"device.forecast-residual.{channel}",
                            device,
                            begin,
                            end,
                            end,
                            f"{device}: {channel} differed from its forecast "
                            f"for {len(run)} minutes.",
                            "Inspect the channel history, sensor and room conditions; "
                            "the residual score does not identify a cause.",
                            {
                                "room": room,
                                "channel": channel,
                                "maximum_residual_score": max(r["score"] for r in run),
                                "sustained_minutes": len(run),
                                "model_version": model_version,
                                "calibration_sha256": calibration_hash,
                            },
                            detector="model:ttm-residual@1",
                            source=SourceIdentity(
                                kind="device", device_id=device, room=room or None
                            ),
                        )
                    )
                run = []
            if row is not None and row["score"] > 1:
                run.append({**row, "at": datetime.fromisoformat(row["time"])})
    return findings


def classify(row: dict, method: str, threshold: float) -> str:
    cpu = row.get("cpu_percent")
    if cpu is None or not isinstance(cpu, (int, float)) or not 0 <= cpu <= 100:
        return "insufficient-evidence"
    score = row.get(f"{method}_score")
    busy = cpu >= threshold or (score is not None and score > 1)
    if busy and row["affected_services"] and not row["container_boundary"]:
        return "resource-pressure"
    if busy:
        return "unusual-activity"
    return (
        "expected-workload" if score is not None or method == "rules" else "insufficient-evidence"
    )


def findings_for(rows: list[dict], method: str, threshold: float, minutes: int) -> list[Advice]:
    findings: list[Advice] = []
    run: list[dict] = []

    def finish():
        if len(run) * 5 < minutes:
            return
        first, last = run[0], run[-1]
        pressure = first["state"] == "resource-pressure"
        expected = (
            first.get("last_value_cpu_percent")
            if method == "last_value"
            else first["expected_cpu_percent"]
            if method == "ttm"
            else None
        )
        affected = sorted(
            {service for item in run for service in item["affected_services"].split(",") if service}
        )
        condition = (
            "resource pressure with observed service impact" if pressure else "unusual CPU activity"
        )
        evidence = {
            "metric": "usage_active_mean",
            "cpu_percent": first["cpu_percent"],
            "maximum_cpu_percent": max(r["cpu_percent"] for r in run),
            "memory_percent": first["memory_percent"]
            if first["memory_percent"] is not None
            else "unavailable",
            "expected_cpu_percent": expected if expected is not None else "unavailable",
            "deviation_score": first.get(f"{method}_score")
            if first.get(f"{method}_score") is not None
            else "unavailable",
            "maximum_deviation_score": max(
                (r[f"{method}_score"] for r in run if r.get(f"{method}_score") is not None),
                default="unavailable",
            ),
            "first_detected_at": (
                timestamp(first["time"]) + timedelta(minutes=minutes)
            ).isoformat(),
            "expectation_origin": first["expectation_origin"] or "unavailable",
            "expectation_target": first["time"],
            "expectation_lead_minutes": 5,
            "context_start": first["context_start"] or "unavailable",
            "affected_services": ",".join(affected),
            "cause": "unknown",
            "activity": first["activity"],
            "sustained_minutes": len(run) * 5,
            "cpu_threshold_percent": threshold,
            "required_minutes": minutes,
            "model_version": first["model_version"],
            "model_sha256": first["model_sha256"],
            "calibration_sha256": first["calibration_sha256"],
        }
        findings.append(
            Advice(
                "host.resource-pressure" if pressure else "host.unusual-workload",
                first["entity_id"],
                timestamp(first["time"]),
                timestamp(last["time"]) + STEP,
                timestamp(last["time"]) + STEP,
                f"{first['entity_id']}: {condition} for {len(run) * 5} minutes.",
                "Inspect affected services and workload; check memory and recent changes."
                if pressure
                else "Monitor the workload; CPU activity alone does not establish a fault.",
                evidence,
                severity="warning" if pressure else "info",
                detector=f"model:{method}-workload@1"
                if method != "rules" and first.get(f"{method}_score") is not None
                else "rule:host-workload@1",
                source=SourceIdentity(kind="host", host=first.get("host") or None),
            )
        )

    previous = None
    for source in rows:
        row = {**source, "state": classify(source, method, threshold)}
        key = row["entity_id"], row["split"], row["state"], row["activity"]
        at = timestamp(row["time"])
        if (
            previous is None
            or key != previous[0]
            or at - previous[1] != STEP
            or row["container_boundary"]
            or row["host_boundary"]
        ):
            finish()
            run = []
        if row["state"] in {"unusual-activity", "resource-pressure"} and not row["host_boundary"]:
            run.append(row)
        previous = key, at
    finish()
    return findings


def apply_container_context(points: dict, containers: dict) -> None:
    previous: dict[str, dict] = {}
    for (entity, at), point in sorted(containers.items()):
        host = entity.split("/", 1)[0]
        context = points[host, at]
        before = previous.get(entity)
        boundary = bool(point.get("conflict") or len(point["versions"]) > 1)
        if before:
            boundary = boundary or point["versions"] != before["versions"]
            for field in ("container_id_last", "started_at_last", "restart_count_last"):
                if field in point and field in before and point[field] != before[field]:
                    boundary = True
            uptime, prior_uptime = point.get("uptime_ns_last"), before.get("uptime_ns_last")
            if (
                isinstance(uptime, (int, float))
                and isinstance(prior_uptime, (int, float))
                and 0 <= uptime < prior_uptime
            ):
                boundary = True
        previous[entity] = point
        if boundary:
            context["container_boundary"] = True
            continue
        status = point.get("health_status_last")
        if status in {"healthy", "unhealthy"}:
            context.setdefault("health_observed", []).append(entity)
        if status == "unhealthy" or point.get("oomkilled_last") in {"true", 1.0}:
            context.setdefault("affected_services", []).append(entity)
