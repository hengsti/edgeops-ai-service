from fastapi import FastAPI, HTTPException, Request, status

from edgeops_ai import __version__
from edgeops_ai.features import FeatureExtractor
from edgeops_ai.predictor import MODEL_VERSION, RuleBasedPredictor
from edgeops_ai.schemas import AnalysisResult, ObservationV1


def create_app() -> FastAPI:
    app = FastAPI(title="EdgeOps AI Service", version=__version__)

    app.state.feature_extractor = FeatureExtractor()
    app.state.predictor = RuleBasedPredictor()
    app.state.latest_detection = None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "edgeops-ai-service", "version": __version__}

    @app.post("/v1/observations", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
    def create_observation(observation: ObservationV1, request: Request) -> AnalysisResult:
        extractor: FeatureExtractor = request.app.state.feature_extractor
        predictor: RuleBasedPredictor = request.app.state.predictor

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

        request.app.state.latest_detection = result
        return result

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


if __name__ == "__main__":
    app = create_app()
