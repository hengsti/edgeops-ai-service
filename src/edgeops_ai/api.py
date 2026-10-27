import asyncio
from datetime import datetime
from time import time
from typing import Annotated, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from edgeops_ai import __version__
from edgeops_ai.analysis import Analyzer
from edgeops_ai.schemas import AnalysisResult, LatestDetection, ObservationV1
from edgeops_ai.state_report import AdvisorStatus, StateReport, StateReporter

router = APIRouter()
ADVISOR_RESPONSE_SECONDS = 1


@router.get(
    "/v1/state-report",
    response_model=StateReport,
    responses={
        200: {
            "description": "Current schema-1.0 snapshot; timestamps stay unchanged on cache hits",
            "headers": {
                "X-Advisor-State": {"schema": {"type": "string", "enum": ["current"]}},
                "X-Advisor-Refresh-Seconds": {"schema": {"type": "number"}},
                "X-Advisor-Expires-At": {"schema": {"type": "string", "format": "date-time"}},
            },
        },
        202: {
            "model": AdvisorStatus,
            "description": "Refresh pending after a one-second generation wait budget; retry here",
            "headers": {"Retry-After": {"schema": {"type": "integer"}}},
        },
        503: {
            "model": AdvisorStatus,
            "description": "Advisor unconfigured, refresh failed or retry backoff active",
            "headers": {"Retry-After": {"schema": {"type": "integer"}}},
        },
    },
)
async def get_state_report(request: Request, response: Response) -> StateReport | JSONResponse:
    headers = {"Cache-Control": "no-store"}
    reporter = cast(StateReporter | None, request.app.state.state_reporter)
    if reporter is None:
        return JSONResponse(
            status_code=503,
            headers=headers,
            content=AdvisorStatus(
                status="unconfigured", detail="Advisor is not configured."
            ).model_dump(),
        )
    headers["X-Advisor-Refresh-Seconds"] = f"{reporter.refresh_seconds:g}"
    if retry := reporter.retry_after_seconds:
        headers["Retry-After"] = str(retry)
        return JSONResponse(
            status_code=503,
            headers=headers,
            content=AdvisorStatus(
                status="backoff",
                detail="Advisor refresh is temporarily unavailable.",
                retry_after_seconds=retry,
            ).model_dump(),
        )
    try:
        report = await asyncio.wait_for(reporter.report(), timeout=ADVISOR_RESPONSE_SECONDS)
    except TimeoutError:
        headers["Retry-After"] = "2"
        return JSONResponse(
            status_code=202,
            headers=headers,
            content=AdvisorStatus(
                status="pending", detail="Advisor refresh is in progress.", retry_after_seconds=2
            ).model_dump(),
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        retry = max(1, reporter.retry_after_seconds)
        headers["Retry-After"] = str(retry)
        return JSONResponse(
            status_code=503,
            headers=headers,
            content=AdvisorStatus(
                status="failed", detail="Advisor report is unavailable.", retry_after_seconds=retry
            ).model_dump(),
        )
    response.headers.update(headers)
    response.headers["X-Advisor-State"] = "current"
    assert reporter.expires_at is not None
    response.headers["X-Advisor-Expires-At"] = reporter.expires_at.isoformat()
    return report


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
