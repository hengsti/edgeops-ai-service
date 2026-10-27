"""Versioned Advisor state-report contract and runtime."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from math import ceil
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import AwareDatetime, Field, FiniteFloat

from edgeops_ai.advisor import (
    DAY,
    MINUTE,
    Advice,
    DeviceSample,
    PipelineSample,
    check_time,
    device_findings,
    pipeline_findings,
)
from edgeops_ai.climate import climate_findings
from edgeops_ai.maintenance import MaintenanceDay, assess
from edgeops_ai.resources import STEP, ResourceSample, resource_findings
from edgeops_ai.schemas import SourceIdentity, StrictModel
from edgeops_ai.weather import WeatherClient, WeatherObservation

if TYPE_CHECKING:
    from edgeops_ai.forecast_runtime import ForecastDetector

Category = Literal["device", "pipeline", "climate", "host", "maintenance"]
Coverage = Literal["available", "no-data", "deferred"]
Severity = Literal["info", "warning", "critical"]
Overall = Literal["ok", "attention", "critical"]
logger = logging.getLogger(__name__)


class Evidence(StrictModel):
    metric: str = Field(min_length=1)
    window: str
    value: FiniteFloat | str
    baseline: FiniteFloat | str | None = None


class Finding(StrictModel):
    id: str
    category: Category
    severity: Severity
    subject: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence: list[Evidence]
    recommendation: str = Field(min_length=1)
    confidence: FiniteFloat | None = Field(default=None, ge=0, le=1)
    first_seen: AwareDatetime
    last_seen: AwareDatetime
    window_start: AwareDatetime
    window_stop: AwareDatetime
    detector: str = Field(min_length=1)
    code: str | None = Field(
        default=None, min_length=1, description="Original detector finding code"
    )
    source: SourceIdentity | None = Field(default=None, description="Observed subject identity")


class MaintenanceEstimate(StrictModel):
    metric: Literal["disk_used_percent", "free_mem_bytes"]
    entity_id: str
    unit: str
    as_of: AwareDatetime
    start: AwareDatetime | None
    stop: AwareDatetime | None
    observed_days: int = Field(ge=0, le=30)
    threshold: FiniteFloat
    current: FiniteFloat | None
    status: Literal[
        "insufficient-history",
        "stale",
        "threshold-reached",
        "not-approaching",
        "forecast",
        "uncertain-trend",
    ]
    slope_per_day: FiniteFloat | None
    slope_95_percent_interval: tuple[FiniteFloat | None, FiniteFloat | None]
    eta_days: FiniteFloat | None
    eta_range_days: tuple[FiniteFloat | None, FiniteFloat | None]
    advice_due: bool
    recommendation: str
    source: SourceIdentity | None = Field(
        default=None, description="Same identity as related findings"
    )


class ModelAvailability(StrictModel):
    status: Literal[
        "disabled", "available", "no-data", "stale", "insufficient-history", "uncalibrated", "error"
    ]
    model_version: str | None = None
    model_sha256: str | None = None
    calibration_sha256: str | None = None
    deployment_approved: bool = False
    entities: dict[str, str] = Field(default_factory=dict)


class ClimateForecast(StrictModel):
    device_id: str
    room: str
    model_version: str
    model_sha256: str
    context_start: AwareDatetime
    context_stop: AwareDatetime
    forecast_origin: AwareDatetime
    last_input: AwareDatetime
    target_times: list[AwareDatetime]
    temperature_c: list[FiniteFloat]
    humidity_percent: list[FiniteFloat]
    uncertainty: Literal["uncalibrated"] = "uncalibrated"


class StateReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: AwareDatetime
    observed_at: AwareDatetime
    window_start: AwareDatetime
    status: Literal["complete", "partial"]
    overall: Overall
    coverage: dict[Category, Coverage]
    findings: list[Finding]
    maintenance_estimates: list[MaintenanceEstimate] = Field(default_factory=list)
    models: dict[str, ModelAvailability] = Field(
        default_factory=lambda: {
            name: ModelAvailability(status="disabled")
            for name in ("health", "climate", "resources")
        }
    )
    forecasts: list[ClimateForecast] = Field(default_factory=list)
    inputs: dict[
        str, Literal["available", "no-data", "stale", "insufficient-history", "unsupported"]
    ] = Field(default_factory=dict)


class AdvisorStatus(StrictModel):
    status: Literal["pending", "unconfigured", "failed", "backoff"]
    detail: str
    retry_after_seconds: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class AdvisorWindow:
    start: datetime
    stop: datetime
    devices: list[DeviceSample]
    pipeline: list[PipelineSample]
    outdoor_weather: tuple[WeatherObservation, ...] = ()
    resources: tuple[ResourceSample, ...] = ()
    history: tuple[MaintenanceDay, ...] = ()
    history_start: datetime | None = None
    maintenance_findings: tuple[Advice, ...] = ()
    forecast_advice: tuple[Advice, ...] = ()


class InfluxReader(Protocol):
    async def read(self, start: datetime, stop: datetime) -> AdvisorWindow: ...


class Detector(Protocol):
    @property
    def category(self) -> Category: ...

    def detect(self, window: AdvisorWindow) -> list[Advice]: ...


class DeviceHealthDetector:
    category: Category = "device"

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        return [
            *device_findings(window.devices, window.start, window.stop),
            *(item for item in window.maintenance_findings if item.code.startswith("device.")),
        ]


class ClimateDetector:
    category: Category = "climate"

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        return climate_findings(
            window.devices,
            window.start,
            window.stop,
            outdoor_weather=window.outdoor_weather,
            forecast_advice=window.forecast_advice,
        )


class PipelineHealthDetector:
    category: Category = "pipeline"

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        return [
            replace(
                item,
                start=max(item.start, window.start),
                stop=min(item.stop, window.stop),
                observed_at=min(item.observed_at, window.stop),
            )
            for item in pipeline_findings(window.pipeline)
        ]


class ResourceDetector:
    category: Category = "host"

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        return resource_findings(list(window.resources), window.stop)


class MaintenanceDetector:
    category: Category = "maintenance"

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        return [
            item for item in window.maintenance_findings if item.code.startswith("maintenance.")
        ]


DEFAULT_DETECTORS: tuple[Detector, ...] = (
    DeviceHealthDetector(),
    PipelineHealthDetector(),
    ClimateDetector(),
    ResourceDetector(),
    MaintenanceDetector(),
)


class StateReporter:
    def __init__(
        self,
        reader: InfluxReader,
        detectors: tuple[Detector, ...] = DEFAULT_DETECTORS,
        *,
        refresh_seconds: float = 300,
        weather_client: WeatherClient | None = None,
        disk_threshold: float = 90,
        memory_threshold: float = 65536,
        horizon_days: float = 30,
        forecast_detectors: tuple["ForecastDetector", ...] = (),
    ):
        self.reader = reader
        self.detectors = detectors
        self.refresh_seconds = refresh_seconds
        self.weather_client = weather_client
        self.disk_threshold = disk_threshold
        self.memory_threshold = memory_threshold
        self.horizon_days = horizon_days
        self.forecast_detectors = forecast_detectors
        self._climate_advice: tuple[Advice, ...] = ()
        # shortcut: state is per process; persist it before adding durable notifications.
        self._refresh_task: asyncio.Task[StateReport | None] | None = None
        self._closed = False
        self._cached: StateReport | None = None
        self._expires = 0.0
        self.expires_at: datetime | None = None
        self._retry_after = 0.0
        self._missing: dict[str, int] = {}

    def _findings(
        self,
        window: AdvisorWindow,
    ) -> tuple[
        list[Finding],
        dict[Category, Coverage],
        list[MaintenanceEstimate],
        dict[str, ModelAvailability],
        list[ClimateForecast],
    ]:
        estimates, maintenance_findings = assess(
            list(window.history),
            window.stop,
            disk_threshold=self.disk_threshold,
            memory_threshold=self.memory_threshold,
            horizon_days=self.horizon_days,
        )
        model_items: dict[Category, list[Advice]] = {"device": [], "host": [], "climate": []}
        models = {
            name: ModelAvailability(status="disabled")
            for name in ("health", "climate", "resources")
        }
        frames: list[ClimateForecast] = []
        for consumer in self.forecast_detectors:
            model_items[consumer.category].extend(consumer.detect(window))
            models[consumer.name] = consumer.availability
            frames.extend(
                ClimateForecast.model_validate(
                    {name: frame[name] for name in ClimateForecast.model_fields}
                )
                for frame in consumer.frames
            )
        window = replace(
            window,
            maintenance_findings=tuple(maintenance_findings),
            forecast_advice=tuple(
                item
                for item in (*self._climate_advice, *model_items["climate"])
                if window.start <= item.start < item.observed_at <= window.stop
            ),
        )
        coverage: dict[Category, Coverage] = dict.fromkeys(
            ("device", "pipeline", "climate", "host", "maintenance"), "deferred"
        )
        available = {
            "device": bool(window.devices),
            "pipeline": bool(window.pipeline),
            "climate": any(
                "humidity_mean" in row.values or "iaq_mean" in row.values for row in window.devices
            ),
            "host": any(window.stop - row.at <= 3 * STEP for row in window.resources),
            "maintenance": any(
                item["status"] not in {"stale", "insufficient-history"} for item in estimates
            ),
        }
        latest: dict[str, Finding] = {}
        climate_advice = []
        for detector in self.detectors:
            items = detector.detect(window)
            if detector.category == "climate":
                # The climate rule applies the shared room cooldown to forecast advice as well.
                climate_advice = [item for item in items if item.detector == "model:ttm-climate@1"]
            else:
                items.extend(model_items.get(detector.category, []))
            coverage[detector.category] = (
                "available" if items or available.get(detector.category, False) else "no-data"
            )
            for item in items:
                for at in (item.start, item.stop, item.observed_at):
                    check_time(at)
                if not (
                    (window.history_start or window.start) <= item.start < item.stop <= window.stop
                    and item.start <= item.observed_at <= window.stop
                ):
                    raise ValueError("Finding evidence is outside the Advisor observation window")
                if item.code.partition(".")[0] != detector.category:
                    raise ValueError("Finding category does not match its detector")
                identity = hashlib.sha256(
                    f"{item.entity_id}\n{item.detector}".encode()
                ).hexdigest()[:16]
                room = item.evidence.get("room")
                finding = Finding(
                    id=f"{item.code}.{identity}",
                    category=detector.category,
                    severity=cast(Severity, item.severity),
                    subject=f"{room} ({item.entity_id})" if room else item.entity_id,
                    summary=item.summary,
                    recommendation=item.recommendation,
                    evidence=[
                        Evidence(
                            metric=name,
                            window=f"{(item.stop - item.start).total_seconds():g}s",
                            value=value,
                        )
                        for name, value in item.evidence.items()
                    ],
                    first_seen=item.observed_at,
                    # Silence rules first detect a gap before its full observation interval ends.
                    last_seen=max(item.observed_at, item.stop),
                    window_start=item.start,
                    window_stop=item.stop,
                    detector=item.detector,
                    code=item.code,
                    source=item.source,
                )
                before = latest.get(finding.id)
                if before is None or finding.last_seen > before.last_seen:
                    latest[finding.id] = finding
        self._climate_advice = tuple(climate_advice)
        return (
            list(latest.values()),
            coverage,
            [MaintenanceEstimate.model_validate(item) for item in estimates],
            models,
            frames,
        )

    def _inputs(self, window: AdvisorWindow, estimates: list[MaintenanceEstimate]) -> dict:
        inputs = {}
        for name, measurement, metric in (
            ("host-cpu", "cpu", "usage_active_mean"),
            ("host-memory", "mem", "used_percent_mean"),
            ("root-disk", "disk", "used_percent_mean"),
            ("container-cpu", "docker_container_cpu", "usage_percent_mean"),
            ("container-memory", "docker_container_mem", "usage_percent_mean"),
            ("container-health", "docker_container_health", "health_status_last"),
        ):
            rows = [
                row
                for row in window.resources
                if row.measurement == measurement and row.metric == metric
            ]
            fresh = [row for row in rows if window.stop - row.at <= 3 * STEP]
            inputs[name] = "available" if fresh else "stale" if rows else "no-data"
            if (
                name == "container-health"
                and fresh
                and not any(row.value in {"healthy", "unhealthy"} for row in fresh)
            ):
                inputs[name] = "unsupported"
        for name, metric in (
            ("root-disk-eta", "disk_used_percent"),
            ("device-memory-eta", "free_mem_bytes"),
        ):
            items = [item for item in estimates if item.metric == metric]
            inputs[name] = (
                "no-data"
                if not items
                else "stale"
                if all(item.status == "stale" for item in items)
                else "insufficient-history"
                if any(item.status in {"insufficient-history", "stale"} for item in items)
                else "available"
            )
        return inputs

    @property
    def retry_after_seconds(self) -> int:
        return max(0, ceil(self._retry_after - monotonic()))

    async def report(self) -> StateReport:
        if self._closed:
            raise ValueError("Advisor is stopping")
        if self._cached is not None and monotonic() < self._expires:
            return self._cached
        if self.retry_after_seconds:
            raise ValueError("Advisor refresh is temporarily unavailable")
        # No await between checking and creating the task: callers on this loop share it.
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh(), name="advisor_refresh")
        report = await asyncio.shield(self._refresh_task)
        if report is None:
            raise ValueError("Advisor report is unavailable")
        return report

    async def close(self) -> None:
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)

    async def _refresh(self) -> StateReport | None:
        try:
            now = datetime.now(UTC)
            stop = now.replace(second=0, microsecond=0) - MINUTE
            start = stop - DAY
            window = await self.reader.read(start, stop)
            if window.start != start or window.stop != stop:
                raise ValueError("Reader returned a different Advisor window")
            if self.weather_client is not None:
                observation = await self.weather_client.get(now)
                window = replace(
                    window, outdoor_weather=(observation,) if observation is not None else ()
                )
            analysis = asyncio.create_task(asyncio.to_thread(self._findings, window))
            try:
                findings, coverage, estimates, models, frames = await asyncio.shield(analysis)
            except asyncio.CancelledError:
                # Cancelling to_thread cannot stop its thread; drain it before closing inputs.
                await asyncio.gather(analysis, return_exceptions=True)
                raise
            inputs = self._inputs(window, estimates)
            previous = {item.id: item for item in self._cached.findings} if self._cached else {}
            current = {item.id: item for item in findings}
            missing = {}
            for identity, item in current.items():
                if identity in previous:
                    item.first_seen = min(item.first_seen, previous[identity].first_seen)
            for identity, item in previous.items():
                if identity not in current:
                    count = self._missing.get(identity, 0) + 1
                    if count < 2:
                        current[identity], missing[identity] = item, count
            partial = any(value != "available" for value in (*coverage.values(), *inputs.values()))
            partial = partial or any(
                item.status not in {"disabled", "available"} for item in models.values()
            )
            overall: Overall = (
                "critical"
                if any(item.severity == "critical" for item in current.values())
                else (
                    "attention"
                    if partial or any(item.severity == "warning" for item in current.values())
                    else "ok"
                )
            )
            report = StateReport(
                generated_at=datetime.now(UTC),
                observed_at=stop,
                window_start=window.history_start or start,
                status="partial" if partial else "complete",
                overall=overall,
                coverage=coverage,
                findings=sorted(current.values(), key=lambda item: item.id),
                maintenance_estimates=estimates,
                inputs=inputs,
                models=models,
                forecasts=frames,
            )
            self._cached, self._missing = report, missing
            self._expires = monotonic() + self.refresh_seconds
            self.expires_at = report.generated_at + timedelta(seconds=self.refresh_seconds)
            return report
        except Exception as error:
            # A timed-out/disconnected caller may no longer be awaiting this task's result.
            self._retry_after = monotonic() + min(60, self.refresh_seconds)
            logger.warning("Advisor report refresh failed (%s)", type(error).__name__)
            return None
