from datetime import datetime
from time import time
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse

from edgeops_ai import __version__
from edgeops_ai.analysis import Analyzer
from edgeops_ai.schemas import AnalysisResult, LatestDetection, ObservationV1

router = APIRouter()


def get_analyzer(request: Request) -> Analyzer:
    return cast(Analyzer, request.app.state.analyzer)


Analysis = Annotated[Analyzer, Depends(get_analyzer)]


@router.get("/health")
async def health(analyzer: Analysis) -> dict[str, str | int | datetime | None]:
    predictor = analyzer.predictor
    return {
        "status": "degraded" if analyzer.collector_consecutive_failures else "ok",
        "service": "edgeops-ai-service",
        "version": __version__,
        "prediction_backend": predictor.backend_name.value,
        "model_version": predictor.model_version,
        "last_poll_success_at": analyzer.collector_last_success_at,
        "last_error": analyzer.collector_last_error,
        "consecutive_failures": analyzer.collector_consecutive_failures,
    }


@router.get("/ready", responses={503: {"description": "Collector is not ready"}})
async def readiness(request: Request, response: Response, analyzer: Analysis) -> dict[str, str]:
    if request.app.state.collector_polling_enabled and (
        analyzer.collector_last_success_at is None or analyzer.collector_consecutive_failures > 0
    ):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not-ready"}
    return {"status": "ready"}


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(analyzer: Analysis) -> PlainTextResponse:
    return PlainTextResponse(
        "# HELP edgeops_poll_failures_total Total failed collector polls since startup.\n"
        "# TYPE edgeops_poll_failures_total counter\n"
        f"edgeops_poll_failures_total {analyzer.collector_poll_failures_total}\n"
        "# HELP edgeops_last_prediction_timestamp Unix time of the last predicted observation.\n"
        "# TYPE edgeops_last_prediction_timestamp gauge\n"
        f"edgeops_last_prediction_timestamp {analyzer.last_prediction_timestamp}\n",
        media_type="text/plain; version=0.0.4",
    )


@router.post("/v1/observations", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
async def create_observation(
    observation: ObservationV1, request: Request, analyzer: Analysis
) -> AnalysisResult:
    if request.app.state.collector_polling_enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Disable collector polling before submitting observations.",
        )
    return await analyzer.analyze(observation)


@router.get("/v1/detections/latest", response_model=LatestDetection)
async def get_latest_detection(request: Request, analyzer: Analysis) -> LatestDetection:
    detection = analyzer.latest_detection
    if detection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No detection has been analyzed yet.",
        )
    age_seconds = max(0.0, time() - detection.captured_at.timestamp())
    return LatestDetection(
        **detection.model_dump(),
        observed_at=detection.captured_at,
        age_seconds=age_seconds,
        stale=(
            analyzer.collector_consecutive_failures > 0
            or age_seconds > 2 * request.app.state.collector_poll_interval_seconds
        ),
    )
