import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

import httpx2
from fastapi import FastAPI, HTTPException, Request, status

from edgeops_ai import __version__
from edgeops_ai.collector_client import CollectorClient
from edgeops_ai.features import FeatureExtractor
from edgeops_ai.predictor import MODEL_VERSION, RuleBasedPredictor
from edgeops_ai.schemas import AnalysisResult, ObservationV1
from edgeops_ai.settings import Settings

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, collector_transport: (httpx2.AsyncBaseTransport | None) = None
) -> FastAPI:
    runtime_settings = settings or Settings()

    def analyze_observation(observation: ObservationV1, app: FastAPI) -> AnalysisResult:
        extractor: FeatureExtractor = app.state.feature_extractor
        predictor: RuleBasedPredictor = app.state.predictor

        features = extractor.update(observation)

        if features is None:
            result = AnalysisResult(
                status="warming-up",
                captured_at=observation.metadata.captured_at,
                model_version=MODEL_VERSION,
                scenario_id=observation.metadata.scenario_id,
                simulation_run_id=observation.metadata.simulation_run_id,
                simulation_phase=observation.metadata.simulation_phase,
            )
        else:
            prediction = predictor.predict(features)

            result = AnalysisResult(
                status="predicted",
                captured_at=observation.metadata.captured_at,
                prediction=prediction.label,
                confidence=prediction.confidence,
                model_version=MODEL_VERSION,
                recommendation=prediction.recommendation,
                features=features,
                scenario_id=observation.metadata.scenario_id,
                simulation_run_id=observation.metadata.simulation_run_id,
                simulation_phase=observation.metadata.simulation_phase,
            )

        app.state.latest_detection = result

        return result

    async def poll_collector(app: FastAPI, client: CollectorClient) -> None:
        while True:
            try:
                observation = await client.fetch_observation()

                analyze_observation(observation, app)

                app.state.collector_last_error = None
            except asyncio.CancelledError:
                raise

            except (
                httpx2.HTTPError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                app.state.collector_last_error = str(error)

                logger.warning("Collector polling failed: %s", error)

            await asyncio.sleep(runtime_settings.collector_poll_interval_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        polling_task: asyncio.Task[None] | None = None
        collector_client: CollectorClient | None = None

        if runtime_settings.collector_polling_enabled:
            collector_client = CollectorClient(
                base_url=runtime_settings.collector_base_url,
                api_key=runtime_settings.collector_api_key,
                timeout_seconds=runtime_settings.collector_timeout_seconds,
                device_ids=runtime_settings.device_ids,
                transport=collector_transport,
            )

            polling_task = asyncio.create_task(
                poll_collector(app, collector_client), name="collector_polling_task"
            )
        try:
            yield
        finally:
            if polling_task:
                polling_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await polling_task

            if collector_client:
                await collector_client.close()

    app = FastAPI(
        title="EdgeOps AI Service",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.feature_extractor = FeatureExtractor()
    app.state.predictor = RuleBasedPredictor()
    app.state.latest_detection = None
    app.state.collector_last_error = None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "edgeops-ai-service", "version": __version__}

    @app.post("/v1/observations", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
    def create_observation(observation: ObservationV1, request: Request) -> AnalysisResult:
        return analyze_observation(observation, request.app)

    @app.get("/v1/detections/latest", response_model=AnalysisResult)
    def get_latest_detection(request: Request) -> AnalysisResult:
        latest: AnalysisResult | None = request.app.state.latest_detection

        if latest is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No detection has been analyzed yet.",
            )

        return latest

    return app
