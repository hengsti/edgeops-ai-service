import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, status

from edgeops_ai import __version__
from edgeops_ai.collector_client import CollectorClient
from edgeops_ai.features import FeatureExtractor
from edgeops_ai.ml import ModelArtifactError, load_ml_predictor
from edgeops_ai.prediction import PredictionBackendName, Predictor
from edgeops_ai.rules import RuleBasedPredictor
from edgeops_ai.schemas import AnalysisResult, ObservationV1
from edgeops_ai.settings import Settings

logger = logging.getLogger(__name__)


def analyze_observation(
    observation: ObservationV1,
    extractor: FeatureExtractor,
    predictor: Predictor,
) -> AnalysisResult:
    features = extractor.update(observation)

    if features is None:
        return AnalysisResult(
            status="warming-up",
            captured_at=observation.metadata.captured_at,
            model_version=predictor.model_version,
            scenario_id=observation.metadata.scenario_id,
            simulation_run_id=observation.metadata.simulation_run_id,
            simulation_phase=observation.metadata.simulation_phase,
        )

    prediction = predictor.predict(features)

    return AnalysisResult(
        status="predicted",
        captured_at=observation.metadata.captured_at,
        prediction=prediction.label,
        confidence=prediction.confidence,
        model_version=prediction.model_version,
        recommendation=prediction.recommendation,
        features=features,
        scenario_id=observation.metadata.scenario_id,
        simulation_run_id=observation.metadata.simulation_run_id,
        simulation_phase=observation.metadata.simulation_phase,
    )


async def poll_collector(
    app: FastAPI,
    client: CollectorClient,
    interval_seconds: float,
) -> None:
    while True:
        try:
            observation = await client.fetch_observation()

            app.state.latest_detection = analyze_observation(
                observation, app.state.feature_extractor, app.state.predictor
            )
            app.state.collector_last_error = None

        except asyncio.CancelledError:
            raise

        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            app.state.collector_last_error = str(error)

            logger.warning("Collector polling failed: %s", error)

        await asyncio.sleep(interval_seconds)


def create_app(
    settings: Settings | None = None,
    collector_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        polling_task: asyncio.Task[None] | None = None
        collector_client: CollectorClient | None = None

        if runtime_settings.prediction_backend is PredictionBackendName.RULES:
            app.state.predictor = RuleBasedPredictor()
        else:
            if not runtime_settings.model_artifact_sha256:
                raise ModelArtifactError("MODEL_ARTIFACT_SHA256 is required for the ml backend.")

            app.state.predictor = load_ml_predictor(
                runtime_settings.model_artifact_path, runtime_settings.model_artifact_sha256
            )

        if runtime_settings.collector_polling_enabled:
            collector_client = CollectorClient(
                base_url=runtime_settings.collector_base_url,
                api_key=runtime_settings.collector_api_key,
                timeout_seconds=runtime_settings.collector_timeout_seconds,
                device_ids=runtime_settings.device_ids,
                transport=collector_transport,
            )

            polling_task = asyncio.create_task(
                poll_collector(
                    app, collector_client, runtime_settings.collector_poll_interval_seconds
                ),
                name="collector_polling_task",
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
    app.state.latest_detection = None
    app.state.collector_last_error = None

    @app.get("/health")
    def health() -> dict[str, str]:
        predictor: Predictor = app.state.predictor

        return {
            "status": "ok",
            "service": "edgeops-ai-service",
            "version": __version__,
            "prediction_backend": predictor.backend_name.value,
            "model_version": predictor.model_version,
        }

    @app.post("/v1/observations", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
    def create_observation(observation: ObservationV1, request: Request) -> AnalysisResult:
        state = request.app.state
        state.latest_detection = analyze_observation(
            observation, state.feature_extractor, state.predictor
        )

        return state.latest_detection

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
