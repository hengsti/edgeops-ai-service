"""Deterministic host/container facts from five-minute resource aggregates."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from edgeops_ai.advisor import MINUTE, Advice

STEP = 5 * MINUTE
RESOURCE_FIELDS = {
    "cpu": {"usage_active_mean"},
    "mem": {"used_percent_mean"},
    "disk": {"used_percent_mean"},
    "docker_container_cpu": {"usage_percent_mean"},
    "docker_container_mem": {"usage_percent_mean"},
    "docker_container_status": {"restart_count_last", "uptime_ns_last", "oomkilled_last"},
    "docker_container_health": {"health_status_last", "failing_streak_max"},
}


@dataclass(frozen=True)
class ResourceSample:
    at: datetime
    entity_id: str
    measurement: str
    metric: str
    value: float | str
    dimensions: dict[str, str] = field(default_factory=dict)


def resource_findings(samples: list[ResourceSample], stop: datetime) -> list[Advice]:
    series: dict[tuple[str, str, str], list[ResourceSample]] = defaultdict(list)
    for row in samples:
        series[row.entity_id, row.measurement, row.metric].append(row)
    findings = []
    for (_, measurement, metric), rows in sorted(series.items()):
        rows.sort(key=lambda row: row.at)
        last = rows[-1]
        if stop - last.at > 3 * STEP:
            continue
        code, action = "", ""
        if metric in {"usage_active_mean", "used_percent_mean", "usage_percent_mean"}:
            # Container CPU percentages can exceed 100% across cores; do not apply host limits.
            if measurement == "docker_container_cpu":
                continue
            tail = rows[-3:]
            if len(tail) != 3 or any(
                float(row.value) < 90 or row.at != last.at - (2 - index) * STEP
                for index, row in enumerate(tail)
            ):
                continue
            code = "cpu-activity" if measurement == "cpu" else f"{measurement}-pressure"
            action = "Review resource usage and the affected processes before changing capacity."
        elif (metric == "health_status_last" and last.value == "unhealthy") or (
            metric == "failing_streak_max" and float(last.value) > 0
        ):
            code, action = (
                "container-health",
                "Inspect the container health check and service logs.",
            )
        elif metric == "oomkilled_last" and last.value in {"true", 1.0}:
            code, action = (
                "container-oom",
                "Inspect container memory limits and allocation history.",
            )
        elif metric == "restart_count_last" and len(rows) >= 2:
            before = rows[-2]
            if last.at - before.at == STEP and float(last.value) > float(before.value):
                code, action = "container-restarts", "Review restart/deployment history and logs."
        if code:
            findings.append(
                Advice(
                    f"host.{code}",
                    last.entity_id,
                    rows[-min(3, len(rows))].at,
                    last.at + STEP,
                    stop,
                    f"{last.entity_id}: observed {measurement} {metric} needs review.",
                    action,
                    {"metric": metric, "value": last.value, "cadence_seconds": 300},
                    severity="info" if measurement == "cpu" else "warning",
                    detector="rule:resources@1",
                )
            )
    return findings
