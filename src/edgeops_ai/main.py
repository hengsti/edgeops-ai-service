import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI

from edgeops_ai import __version__
from edgeops_ai.analysis import Analyzer
from edgeops_ai.api import router
from edgeops_ai.collector_client import CollectorClient
from edgeops_ai.polling import poll_collector, stop_polling
from edgeops_ai.prediction import PredictionBackendName, Predictor
from edgeops_ai.rules import RuleBasedPredictor
from edgeops_ai.settings import Settings


def create_app(
    settings: Settings | None = None,
    collector_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime_settings.prediction_backend is PredictionBackendName.RULES:
            predictor: Predictor = RuleBasedPredictor()
        else:
            from edgeops_ai.ml import ModelArtifactError, load_ml_predictor

            if not runtime_settings.model_artifact_sha256:
                raise ModelArtifactError("MODEL_ARTIFACT_SHA256 is required for the ml backend.")

            predictor = load_ml_predictor(
                runtime_settings.model_artifact_path, runtime_settings.model_artifact_sha256
            )

        app.state.analyzer = Analyzer(predictor)
        app.state.collector_polling_enabled = runtime_settings.collector_polling_enabled
        app.state.collector_poll_interval_seconds = runtime_settings.collector_poll_interval_seconds
        async with AsyncExitStack() as resources:
            if runtime_settings.collector_polling_enabled:
                collector_client = CollectorClient(
                    base_url=runtime_settings.collector_base_url,
                    api_key=runtime_settings.collector_api_key,
                    timeout_seconds=runtime_settings.collector_timeout_seconds,
                    device_ids=runtime_settings.device_ids,
                    transport=collector_transport,
                )
                resources.push_async_callback(collector_client.close)
                polling_task = asyncio.create_task(
                    poll_collector(
                        app.state.analyzer,
                        collector_client,
                        runtime_settings.collector_poll_interval_seconds,
                    ),
                    name="collector_polling_task",
                )
                resources.push_async_callback(stop_polling, polling_task)
            yield

    app = FastAPI(
        title="EdgeOps AI Service",
        version=__version__,
        lifespan=lifespan,
    )

    app.include_router(router)
    return app
