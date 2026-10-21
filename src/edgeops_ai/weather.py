"""One cached OpenWeatherMap observation and moisture-based ventilation evidence."""

import asyncio
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Literal

import httpx
from pydantic import AwareDatetime, Field, FiniteFloat, field_validator

from edgeops_ai.advisor import Advice, DeviceSample, check_time
from edgeops_ai.schemas import StrictModel

MAX_WEATHER_AGE = timedelta(minutes=30)


class WeatherObservation(StrictModel):
    provider: Literal["openweathermap"] = "openweathermap"
    location: str = Field(min_length=1)
    observed_at: AwareDatetime
    temperature_c: FiniteFloat = Field(ge=-100, le=70, strict=True)
    humidity_percent: FiniteFloat = Field(ge=0, le=100, strict=True)

    @field_validator("observed_at")
    @classmethod
    def utc_time(cls, value: datetime) -> datetime:
        check_time(value)
        return value

    def fresh_at(self, at: datetime) -> bool:
        return timedelta(0) <= at - self.observed_at <= MAX_WEATHER_AGE


class _KeyTransport(httpx.AsyncBaseTransport):
    def __init__(self, key: str, transport: httpx.AsyncBaseTransport):
        self.key, self.transport = key, transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # HTTPX logs URLs after transport returns; keep query credentials inside this boundary.
        public_url = request.url
        request.url = public_url.copy_add_param("appid", self.key)
        try:
            return await self.transport.handle_async_request(request)
        finally:
            request.url = public_url

    async def aclose(self) -> None:
        await self.transport.aclose()


class WeatherClient:
    def __init__(
        self,
        postcode: str,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.postcode = postcode
        self.client = httpx.AsyncClient(
            base_url="https://api.openweathermap.org",
            params={"zip": postcode, "units": "metric"},
            timeout=3,
            transport=_KeyTransport(api_key, transport or httpx.AsyncHTTPTransport()),
        )
        self._lock = asyncio.Lock()
        # shortcut: cache is per process; coordinate it before adding multiple API workers.
        self._cached: WeatherObservation | None = None
        self._next_fetch = 0.0

    async def get(self, now: datetime | None = None) -> WeatherObservation | None:
        now = now or datetime.now(UTC)
        async with self._lock:
            if monotonic() >= self._next_fetch:
                self._next_fetch = monotonic() + 600
                self._cached = None
                try:
                    async with asyncio.timeout(3):
                        response = await self.client.get("/data/2.5/weather")
                        response.raise_for_status()
                        data = response.json()
                        self._cached = WeatherObservation(
                            location=self.postcode,
                            observed_at=datetime.fromtimestamp(data["dt"], UTC),
                            temperature_c=data["main"]["temp"],
                            humidity_percent=data["main"]["humidity"],
                        )
                except (
                    httpx.HTTPError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OverflowError,
                    OSError,
                ):
                    return None
            observation = self._cached
            return observation if observation is not None and observation.fresh_at(now) else None

    async def close(self) -> None:
        await self.client.aclose()


def absolute_humidity(temperature_c: float, humidity_percent: float) -> float:
    if (
        not math.isfinite(temperature_c)
        or not -100 <= temperature_c <= 70
        or not math.isfinite(humidity_percent)
        or not 0 <= humidity_percent <= 100
    ):
        raise ValueError("Invalid moisture comparison metrics")
    vapour_pressure = 6.112 * math.exp(17.67 * temperature_c / (temperature_c + 243.5))
    return 216.7 * vapour_pressure * humidity_percent / 100 / (temperature_c + 273.15)


def ventilation_advice(
    item: Advice, row: DeviceSample, observation: WeatherObservation | None
) -> Advice:
    evidence: dict[str, float | str] = {"outdoor_moisture_suitability": "unknown"}
    if observation is not None and observation.fresh_at(item.observed_at):
        evidence.update(
            weather_provider=observation.provider,
            weather_location=observation.location,
            weather_observed_at=observation.observed_at.isoformat(),
            outdoor_temperature_c=observation.temperature_c,
            outdoor_humidity_percent=observation.humidity_percent,
        )
        try:
            indoor = absolute_humidity(
                float(row.values["temperature_mean"]), float(row.values["humidity_mean"])
            )
            outdoor = absolute_humidity(observation.temperature_c, observation.humidity_percent)
            evidence.update(
                indoor_temperature_c=float(row.values["temperature_mean"]),
                indoor_humidity_percent=float(row.values["humidity_mean"]),
                indoor_absolute_humidity_g_m3=indoor,
                outdoor_absolute_humidity_g_m3=outdoor,
                absolute_humidity_difference_g_m3=indoor - outdoor,
                outdoor_moisture_suitability="drier" if outdoor < indoor else "not-drier",
            )
        except (KeyError, TypeError, ValueError):
            pass
    suitability = evidence["outdoor_moisture_suitability"]
    action = item.recommendation
    if suitability == "unknown":
        action += " Outdoor moisture suitability is unknown."
    elif item.code == "climate.air-quality":
        if suitability == "not-drier":
            action += " Outdoor air is not drier; ventilation is not expected to reduce moisture."
    elif suitability == "drier":
        action = action.replace(
            "if outdoor air is drier", "to reduce moisture when outdoor air quality permits"
        )
    else:
        action = (
            f"Outdoor air is not drier; use a dehumidifier in {row.room} "
            "and check damp or cold surfaces."
        )
    return replace(item, recommendation=action, evidence={**item.evidence, **evidence})
