from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from edgeops_ai import __version__
from edgeops_ai.analysis import Analyzer
from edgeops_ai.schemas import AnalysisResult, ObservationV1

router = APIRouter()


def get_analyzer(request: Request) -> Analyzer:
    return cast(Analyzer, request.app.state.analyzer)


Analysis = Annotated[Analyzer, Depends(get_analyzer)]


@router.get("/health")
def health(analyzer: Analysis) -> dict[str, str]:
    predictor = analyzer.predictor
    return {
        "status": "ok",
        "service": "edgeops-ai-service",
        "version": __version__,
        "prediction_backend": predictor.backend_name.value,
        "model_version": predictor.model_version,
    }


@router.post("/v1/observations", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
def create_observation(observation: ObservationV1, analyzer: Analysis) -> AnalysisResult:
    return analyzer.analyze(observation)


@router.get("/v1/detections/latest", response_model=AnalysisResult)
def get_latest_detection(analyzer: Analysis) -> AnalysisResult:
    if analyzer.latest_detection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No detection has been analyzed yet.",
        )
    return analyzer.latest_detection
