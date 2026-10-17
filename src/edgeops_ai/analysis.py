import asyncio
from datetime import datetime

from edgeops_ai.features import FeatureExtractor
from edgeops_ai.prediction import Predictor
from edgeops_ai.schemas import AnalysisResult, ObservationV1


class Analyzer:
    """Own one observation window and the latest result, independently of HTTP delivery."""

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self._extractor = FeatureExtractor()
        self._lock = asyncio.Lock()
        self.latest_detection: AnalysisResult | None = None
        self.collector_last_error: str | None = None
        self.collector_last_success_at: datetime | None = None
        self.collector_consecutive_failures = 0
        self.collector_poll_failures_total = 0
        self.last_prediction_timestamp = 0.0

    async def analyze(self, observation: ObservationV1) -> AnalysisResult:
        async with self._lock:
            features = self._extractor.update(observation)
            prediction = self.predictor.predict(features) if features is not None else None
            self.latest_detection = AnalysisResult(
                status="predicted" if prediction is not None else "warming-up",
                captured_at=observation.metadata.captured_at,
                prediction=prediction.label if prediction is not None else None,
                confidence=prediction.confidence if prediction is not None else None,
                model_version=(
                    prediction.model_version
                    if prediction is not None
                    else self.predictor.model_version
                ),
                recommendation=prediction.recommendation if prediction is not None else None,
                features=features,
                scenario_id=observation.metadata.scenario_id,
                simulation_run_id=observation.metadata.simulation_run_id,
                simulation_phase=observation.metadata.simulation_phase,
            )
            if prediction is not None:
                self.last_prediction_timestamp = observation.metadata.captured_at.timestamp()
            return self.latest_detection
