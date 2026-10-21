"""Versioned Advisor state-report contract and runtime."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import monotonic
from typing import Literal, Protocol, cast

import httpx
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
from edgeops_ai.schemas import StrictModel
from edgeops_ai.weather import WeatherClient, WeatherObservation

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


class StateReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: AwareDatetime
    observed_at: AwareDatetime
    window_start: AwareDatetime
    status: Literal["complete", "partial"]
    overall: Overall
    coverage: dict[Category, Coverage]
    findings: list[Finding]


@dataclass(frozen=True)
class AdvisorWindow:
    start: datetime
    stop: datetime
    devices: list[DeviceSample]
    pipeline: list[PipelineSample]
    outdoor_weather: tuple[WeatherObservation, ...] = ()


class InfluxReader(Protocol):
    async def read(self, start: datetime, stop: datetime) -> AdvisorWindow: ...


class Detector(Protocol):
    @property
    def category(self) -> Category: ...

    def detect(self, window: AdvisorWindow) -> list[Advice]: ...


class DeviceHealthDetector:
    category: Category = "device"

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        return device_findings(window.devices, window.start, window.stop)


class ClimateDetector:
    category: Category = "climate"

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        return climate_findings(
            window.devices, window.start, window.stop, outdoor_weather=window.outdoor_weather
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


DEFAULT_DETECTORS: tuple[Detector, ...] = (
    DeviceHealthDetector(),
    PipelineHealthDetector(),
    ClimateDetector(),
)


class StateReporter:
    def __init__(
        self,
        reader: InfluxReader,
        detectors: tuple[Detector, ...] = DEFAULT_DETECTORS,
        *,
        refresh_seconds: float = 300,
        weather_client: WeatherClient | None = None,
    ):
        self.reader = reader
        self.detectors = detectors
        self.refresh_seconds = refresh_seconds
        self.weather_client = weather_client
        # shortcut: state is per process; persist it before adding durable notifications.
        self._lock = asyncio.Lock()
        self._cached: StateReport | None = None
        self._expires = 0.0
        self._retry_after = 0.0
        self._missing: dict[str, int] = {}

    def _findings(self, window: AdvisorWindow) -> tuple[list[Finding], dict[Category, Coverage]]:
        coverage: dict[Category, Coverage] = dict.fromkeys(
            ("device", "pipeline", "climate", "host", "maintenance"), "deferred"
        )
        available = {
            "device": bool(window.devices),
            "pipeline": bool(window.pipeline),
            "climate": any(
                "humidity_mean" in row.values or "iaq_mean" in row.values for row in window.devices
            ),
        }
        latest: dict[str, Finding] = {}
        for detector in self.detectors:
            items = detector.detect(window)
            coverage[detector.category] = (
                "available" if items or available.get(detector.category, False) else "no-data"
            )
            for item in items:
                for at in (item.start, item.stop, item.observed_at):
                    check_time(at)
                if not window.start <= item.start < item.stop <= item.observed_at <= window.stop:
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
                    last_seen=item.observed_at,
                    window_start=item.start,
                    window_stop=item.stop,
                    detector=item.detector,
                )
                before = latest.get(finding.id)
                if before is None or finding.last_seen > before.last_seen:
                    latest[finding.id] = finding
        return list(latest.values()), coverage

    async def report(self) -> StateReport:
        async with self._lock:
            if self._cached is not None and monotonic() < self._expires:
                return self._cached
            if monotonic() < self._retry_after:
                raise ValueError("Advisor refresh is temporarily unavailable")
            now = datetime.now(UTC)
            stop = now.replace(second=0, microsecond=0) - MINUTE
            # shortcut: one day of live history; #31 adds bounded multi-day/resource inputs.
            start = stop - DAY
            try:
                window = await self.reader.read(start, stop)
                if window.start != start or window.stop != stop:
                    raise ValueError("Reader returned a different Advisor window")
                if self.weather_client is not None:
                    observation = await self.weather_client.get(now)
                    window = replace(
                        window, outdoor_weather=(observation,) if observation is not None else ()
                    )
                findings, coverage = await asyncio.to_thread(self._findings, window)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                self._retry_after = monotonic() + min(60, self.refresh_seconds)
                logger.warning("Advisor report refresh failed (%s)", type(error).__name__)
                raise
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
            partial = any(value != "available" for value in coverage.values())
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
                window_start=start,
                status="partial" if partial else "complete",
                overall=overall,
                coverage=coverage,
                findings=sorted(current.values(), key=lambda item: item.id),
            )
            self._cached, self._missing = report, missing
            self._expires = monotonic() + self.refresh_seconds
            return report
