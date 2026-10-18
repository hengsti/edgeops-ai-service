import asyncio

import httpx
import pytest
from scripts.common import ScriptError
from scripts.evaluate_deployed_model import post_observation, require_ml_backend

from edgeops_ai.schemas import ObservationV1


@pytest.mark.parametrize("actual_version", ["my-retrained-model-42", "other-model"])
def test_health_checks_the_requested_candidate_version(actual_version: str) -> None:
    async def run() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"prediction_backend": "ml", "model_version": actual_version}
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            if actual_version == "my-retrained-model-42":
                await require_ml_backend(client, "http://ai", "my-retrained-model-42")
            else:
                with pytest.raises(ScriptError, match="wrong model version"):
                    await require_ml_backend(client, "http://ai", "my-retrained-model-42")

    asyncio.run(run())


@pytest.mark.parametrize("actual_version", ["my-retrained-model-42", "other-model"])
def test_holdout_rejects_a_model_change_between_observations(actual_version: str) -> None:
    observation = ObservationV1.model_validate(
        {
            "schema_version": "1.0",
            "metadata": {
                "captured_at": "2026-10-18T10:00:00Z",
                "source_host": "holdout-test",
                "collector_mode": "simulation",
                "simulated": True,
            },
            "metrics": dict.fromkeys(
                (
                    "messages_enqueued_total",
                    "messages_processed_total",
                    "queue_full_total",
                    "transform_success_total",
                    "transform_failed_total",
                    "influx_lines_written_total",
                    "pipeline_duration_seconds_sum",
                    "pipeline_duration_seconds_count",
                ),
                0,
            ),
        }
    )

    async def run() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"status": "warming-up", "model_version": actual_version}
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            if actual_version == "my-retrained-model-42":
                result = await post_observation(
                    client,
                    ai_base_url="http://ai",
                    observation=observation,
                    expected_model_version="my-retrained-model-42",
                )
                assert result["model_version"] == "my-retrained-model-42"
            else:
                with pytest.raises(ScriptError, match="expected deployed model"):
                    await post_observation(
                        client,
                        ai_base_url="http://ai",
                        observation=observation,
                        expected_model_version="my-retrained-model-42",
                    )

    asyncio.run(run())
