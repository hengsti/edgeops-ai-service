from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Callable, Protocol, Sequence, TypeAlias

from src.edgeops_ai.predictor import MODEL_VERSION as RULES_MODEL_VERSION
from src.edgeops_ai.predictor import RuleBasedPredictor
from src.edgeops_ai.schemas import FeatureVector

ML_MODEL_VERSION = "lightweight-ml-0.1.0"


class PredictionBackendName(StrEnum):
    RULES = "rules"
    ML = "ml"


@dataclass(frozen=True, slots=True)
class BackendPrediction:
    label: str
    confidence: float
    recommendation: str
    model_version: str


class PredictionBackend(Protocol):
    """Common interface for all prediction implementations."""

    @property
    def backend_name(self) -> PredictionBackendName: ...

    @property
    def model_version(self) -> str: ...

    def predict(self, features: FeatureVector) -> BackendPrediction: ...


class RulePredictionResult(Protocol):
    label: str
    confidence: float
    recommendation: str


class RulePredictor(Protocol):
    def predict(self, features: FeatureVector) -> RulePredictionResult: ...


class ProbabilityClassifier(Protocol):
    """Minimal interface required from the persisted scikit-learn model."""

    classes: Any

    def predict(self, values: Any) -> Any: ...

    def predict_proba(self, values: Any) -> Any: ...


FeatureVectorizer: TypeAlias = Callable[[FeatureVector], Sequence[float]]

RecommendationResolver: TypeAlias = Callable[[str, FeatureVector], str]


class RulesPredictionBackend:
    """Adapter around the existing rule-based predictor."""

    def __init__(self, predictor: RulePredictor | None = None) -> None:
        self._predictor = predictor or RuleBasedPredictor()

    @property
    def backend_name(self) -> PredictionBackendName:
        return PredictionBackendName.RULES

    @property
    def model_version(self) -> str:
        return RULES_MODEL_VERSION

    def predict(self, features: FeatureVector) -> BackendPrediction:
        result = self._predictor.predict(features)

        return BackendPrediction(
            label=result.label,
            confidence=float(result.confidence),
            recommendation=result.recommendation,
            model_version=self.model_version,
        )


class MlPredictionBackend:
    """
    Prediction backend for a preloaded probability classifier.

    Loading the joblib artifact is deliberately not handled here. The model
    will later be loaded and validated once in the FastAPI lifespan.
    """

    def __init__(
        self,
        *,
        model: ProbabilityClassifier,
        vectorizer: FeatureVectorizer,
        recommendation_resolver: RecommendationResolver,
        model_version: str = ML_MODEL_VERSION,
    ) -> None:
        self._model = model
        self._vectorizer = vectorizer
        self._recommendation_resolver = recommendation_resolver
        self._model_version = model_version

    @property
    def backend_name(self) -> PredictionBackendName:
        return PredictionBackendName.ML

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, features: FeatureVector) -> BackendPrediction:
        row = [float(value) for value in self._vectorizer(features)]

        if not row:
            raise ValueError("ML feature vector must not be empty.")

        if not all(isfinite(value) for value in row):
            raise ValueError("ML feature vector contains a non-finite value.")

        matrix = [row]

        raw_predictions = self._model.predict(matrix)

        if len(raw_predictions) != 1:
            raise RuntimeError(
                f"ML model returned an unexpected number of predictions: {len(raw_predictions)}"
            )

        predicted_label = str(raw_predictions[0])

        classes = [str(value) for value in self._model.classes_]

        if predicted_label not in classes:
            raise RuntimeError(
                f"ML model predicted a class that is not present in classes_: {predicted_label!r}"
            )

        raw_probabilities = self._model.predict_proba(matrix)

        if len(raw_probabilities) != 1:
            raise RuntimeError(
                "ML model returned an unexpected number of probability rows: "
                f"{len(raw_probabilities)}."
            )

        probabilities = [float(value) for value in raw_probabilities[0]]

        if len(probabilities) != len(classes):
            raise RuntimeError(
                "ML probability count does not match the model class count: "
                f"{len(probabilities)} for probabilities for "
                f"{len(classes)} classes."
            )

        class_index = classes.index(predicted_label)
        confidence = probabilities[class_index]

        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RuntimeError(
                f"ML model returned an invalid probability for {predicted_label!r}: {confidence!r}"
            )

        return BackendPrediction(
            label=predicted_label,
            confidence=confidence,
            recommendation=self._recommendation_resolver(predicted_label, features),
            model_version=self._model_version,
        )


def create_prediction_backend(
    backend_name: PredictionBackendName | str,
    *,
    rule_predictor: RulePredictor | None = None,
    ml_model: ProbabilityClassifier | None = None,
    ml_vectorizer: FeatureVectorizer | None = None,
    recommendation_resolver: RecommendationResolver | None = None,
) -> PredictionBackend:
    """
    Construct an explicit prediction backend.

    ML dependencies are injected so artifact loading remains an application
    startup responsibility.
    """

    try:
        selected_backend = PredictionBackendName(backend_name)
    except ValueError as exc:
        allowed = ",".join(backend.value for backend in PredictionBackendName)
        raise ValueError(
            f"Unknown prediction backend: {backend_name!r}. Expected one of: {allowed}"
        ) from exc

    if selected_backend is PredictionBackendName.RULES:
        return RulesPredictionBackend(predictor=rule_predictor)

    missing_dependencies: list[str] = []

    if ml_model is None:
        missing_dependencies.append("ml_model")

    if ml_vectorizer is None:
        missing_dependencies.append("ml_vectorizer")

    if recommendation_resolver is None:
        missing_dependencies.append("recommendation_resolver")

    if missing_dependencies:
        raise RuntimeError(
            "Cannot create ML prediction backend. Missing dependencies: "
            + ", ".join(missing_dependencies)
            + "."
        )

    return MlPredictionBackend(
        model=ml_model,
        vectorizer=ml_vectorizer,
        recommendation_resolver=recommendation_resolver,
    )
