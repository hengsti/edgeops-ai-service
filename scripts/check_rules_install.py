"""Smoke-test the base installation without development or ML dependencies."""

import asyncio
from importlib.util import find_spec

import httpx

from edgeops_ai.main import create_app
from edgeops_ai.settings import Settings


async def main() -> None:
    for module in ("joblib", "sklearn", "numpy", "onnxruntime"):
        assert find_spec(module) is None, f"Rules-only installation contains {module}"

    app = create_app(
        Settings(_env_file=None, collector_polling_enabled=False, prediction_backend="rules")
    )
    payload = {
        "schema_version": "1.0",
        "metadata": {
            "captured_at": "2026-01-01T00:00:00Z",
            "collector_mode": "production",
            "simulated": False,
            "source_host": "installation-smoke",
        },
        "metrics": {
            "messages_enqueued_total": 0,
            "messages_processed_total": 0,
            "queue_full_total": 0,
            "transform_success_total": 0,
            "transform_failed_total": 0,
            "influx_lines_written_total": 0,
            "pipeline_duration_seconds_sum": 0,
            "pipeline_duration_seconds_count": 0,
        },
    }
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["prediction_backend"] == "rules"
        first = await client.post("/v1/observations", json=payload)
        assert first.status_code == 200
        assert first.json()["status"] == "warming-up"
        payload["metadata"]["captured_at"] = "2026-01-01T00:00:10Z"
        second = await client.post("/v1/observations", json=payload)
        assert second.status_code == 200
        assert second.json()["status"] == "predicted"
        assert second.json()["prediction"] == "normal"
    print("Rules-only startup and prediction passed without the ML stack")


if __name__ == "__main__":
    asyncio.run(main())
