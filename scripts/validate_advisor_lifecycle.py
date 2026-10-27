"""Read-only Pi endpoint measurements; run with the existing image and new source."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time

import httpx

from edgeops_ai.influx_reader import HttpInfluxReader
from edgeops_ai.main import create_app
from edgeops_ai.settings import Settings


class MeasuredReader(HttpInfluxReader):
    calls = 0
    queries = 0

    async def read(self, start, stop):
        self.calls += 1
        return await super().read(start, stop)

    async def _query(self, query):
        self.queries += 1
        return await super()._query(query)


def host_cpu() -> tuple[int, int]:
    values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3] + values[4]


def peak_rss_mib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


async def measure_phase(name, client, reporter, reader, callers=1):
    if name in {"warm-refresh", "simultaneous-refresh"}:
        reporter._expires = 0
    reads, queries = reader.calls, reader.queries
    cpu_before = host_cpu()
    started_at = datetime.now(UTC).isoformat()
    wall, cpu = perf_counter(), process_time()
    requests = []

    async def fetch():
        before = perf_counter()
        response = await asyncio.wait_for(client.get("/v1/state-report"), timeout=2.5)
        requests.append({"status": response.status_code, "seconds": perf_counter() - before})
        assert response.status_code in {200, 202}, f"Advisor HTTP {response.status_code}"
        assert response.headers["cache-control"] == "no-store"
        return response

    async with asyncio.timeout(900):
        initial = await asyncio.gather(*(fetch() for _ in range(callers)))
        response = initial[0]
        while response.status_code == 202:
            assert response.json()["status"] == "pending"
            await asyncio.sleep(int(response.headers["retry-after"]))
            response = await fetch()
    report = response.json()
    assert report["schema_version"] == "1.0"
    assert reader.calls - reads == (0 if name == "cached" else 1), "Duplicate/missing refresh"
    if name == "cached":
        assert reader.queries == queries, "Cache hit queried InfluxDB"
    cpu_after = host_cpu()
    elapsed, used = perf_counter() - wall, process_time() - cpu
    total, idle = (after - before for before, after in zip(cpu_before, cpu_after, strict=True))
    return {
        "phase": name,
        "started_at": started_at,
        "wall_seconds": elapsed,
        "initial_callers": callers,
        "requests": requests,
        "reader_calls": reader.calls - reads,
        "queries": reader.queries - queries,
        "process_cpu_seconds": used,
        "process_cpu_percent": 100 * used / elapsed,
        "host_cpu_percent": 100 * (total - idle) / total if total else None,
        "process_peak_rss_mib": peak_rss_mib(),
        "generated_at": report["generated_at"],
        "observed_at": report["observed_at"],
        "expires_at": response.headers["x-advisor-expires-at"],
    }, report


async def main():
    settings = Settings(
        _env_file=None,
        collector_polling_enabled=False,
        prediction_backend="rules",
        advisor_weather_api_key=None,
    )
    assert settings.advisor_influx_url and settings.advisor_influx_token, "Missing Advisor config"
    reader = MeasuredReader(
        str(settings.advisor_influx_url),
        settings.advisor_influx_org,
        settings.advisor_influx_token.get_secret_value(),
        resource_history_hours=48,
    )
    try:
        app = create_app(settings, advisor_reader=reader)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
        ):
            reporter = app.state.state_reporter
            failed = [
                item.name
                for item in reporter.forecast_detectors
                if item.availability.status == "error"
            ]
            if failed:
                raise ValueError(
                    f"Configured Advisor consumers failed to load: {', '.join(failed)}; "
                    "mount the configuration, evidence and model artifacts read-only."
                )
            enabled = [item.name for item in reporter.forecast_detectors if item.model is not None]
            reader.resource_history_hours = 48 if "resources" in enabled else 24
            print(
                json.dumps(
                    {
                        "validation": "advisor-lifecycle-1",
                        "deployment_approved": False,
                        "runtime_mode": "model-enabled" if enabled else "deterministic-only",
                        "enabled_consumers": enabled,
                        "configured_consumers": {
                            item.name: item.availability.status
                            for item in reporter.forecast_detectors
                        },
                        "scope": (
                            "In-process HTTP on target Pi with real InfluxDB; "
                            "no deployment or network RTT"
                        ),
                    }
                ),
                flush=True,
            )
            first = None
            for name, callers in (
                ("cold", 1),
                ("cached", 1),
                ("warm-refresh", 1),
                ("simultaneous-refresh", 8),
            ):
                measurements, report = await measure_phase(name, client, reporter, reader, callers)
                if name == "cold":
                    first = report
                if name == "cached":
                    assert report == first, "Cache changed snapshot timestamps/content"
                print(json.dumps(measurements), flush=True)
    finally:
        await reader.close()


if __name__ == "__main__":
    asyncio.run(main())
