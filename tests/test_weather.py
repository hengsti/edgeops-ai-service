import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest

AT = datetime(2026, 10, 21, 10, tzinfo=UTC)


def payload(at=AT, temperature=5, humidity=90):
    return {"dt": at.timestamp(), "main": {"temp": temperature, "humidity": humidity}}


def test_weather_client_caches_concurrent_requests_and_hides_key(caplog, monkeypatch):
    from edgeops_ai import weather

    caplog.set_level(logging.DEBUG)
    clock = [0.0]
    monkeypatch.setattr(weather, "monotonic", lambda: clock[0])
    requests = []

    def handler(request):
        requests.append(str(request.url))
        assert request.url.params["zip"] == "1030,AT"
        assert request.url.params["units"] == "metric"
        assert request.url.params["appid"] == "private-weather-key"
        assert request.extensions["timeout"]["read"] == 3
        return httpx.Response(200, json=payload())

    async def scenario():
        client = weather.WeatherClient(
            "1030,AT", "private-weather-key", transport=httpx.MockTransport(handler)
        )
        try:
            observations = await asyncio.gather(*(client.get(AT) for _ in range(4)))
            assert len(requests) == 1
            assert all(item == observations[0] for item in observations)
            assert observations[0].temperature_c == 5
            assert observations[0].humidity_percent == 90
            assert observations[0].observed_at == AT
            clock[0] = 599
            assert await client.get(AT + timedelta(minutes=9)) is not None
            assert len(requests) == 1
            clock[0] = 600
            assert await client.get(AT + timedelta(minutes=10)) is not None
            assert len(requests) == 2
        finally:
            await client.close()

    asyncio.run(scenario())
    assert "private-weather-key" not in caplog.text


@pytest.mark.parametrize("problem", ["timeout", "401", "429", "500", "json", "missing"])
def test_weather_failures_are_unavailable_cached_and_do_not_log_secrets(problem, caplog):
    from edgeops_ai.weather import WeatherClient

    caplog.set_level(logging.DEBUG)
    calls = []

    def handler(request):
        calls.append(request)
        if problem == "timeout":
            raise httpx.ReadTimeout("private-weather-key", request=request)
        if problem == "json":
            return httpx.Response(200, text="private-weather-key")
        if problem == "missing":
            return httpx.Response(200, json={"main": {"humidity": 50}})
        return httpx.Response(int(problem), text="private-weather-key")

    async def scenario():
        client = WeatherClient(
            "1030,AT", "private-weather-key", transport=httpx.MockTransport(handler)
        )
        try:
            assert await client.get(AT) is None
            assert await client.get(AT) is None
            assert len(calls) == 1
        finally:
            await client.close()

    asyncio.run(scenario())
    assert "private-weather-key" not in caplog.text


@pytest.mark.parametrize("age", [-1, 1801])
def test_future_and_stale_provider_observations_are_unavailable(age):
    from edgeops_ai.weather import WeatherClient

    async def scenario():
        client = WeatherClient(
            "1030,AT",
            "key",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload(at=AT - timedelta(seconds=age)))
            ),
        )
        try:
            assert await client.get(AT) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_expired_cached_observation_and_failed_refresh_are_not_reused(monkeypatch):
    from edgeops_ai import weather

    clock = [0]
    monkeypatch.setattr(weather, "monotonic", lambda: clock[0])
    replies = iter([httpx.Response(200, json=payload()), httpx.Response(500)])

    async def scenario():
        client = weather.WeatherClient(
            "1030,AT", "key", transport=httpx.MockTransport(lambda request: next(replies))
        )
        try:
            assert await client.get(AT) is not None
            assert await client.get(AT + timedelta(minutes=31)) is None
            clock[0] = 600
            assert await client.get(AT) is None
            assert await client.get(AT) is None
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("temperature,humidity", [(float("nan"), 50), (20, 101), (True, 50)])
def test_observation_validates_provider_metrics(temperature, humidity):
    from edgeops_ai.weather import WeatherObservation

    with pytest.raises(ValueError):
        WeatherObservation(
            observed_at=AT,
            temperature_c=temperature,
            humidity_percent=humidity,
            location="1030,AT",
        )
