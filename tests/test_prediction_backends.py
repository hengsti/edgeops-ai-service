from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from edgeops_ai.features import FeatureVector
from edgeops_ai.prediction_backends import (
    ML_MODEL_VERSION,
    BackendPrediction,
    MlPredictionBackend,
    PredictionBackendName,
    RulesPredictionBackend,
    create_prediction_backend,
)


@dataclass(frozen=True)
class FakeRuleResult:
    label: str
    confidence: float
    recommendation: str


class FakeRulePredictor:
    def predict(self, features: FeatureVector) -> FakeRuleResult:
        del features

        return FakeRuleResult(
            label="ingestion-backpressure",
            confidence=0.97,
            recommendation="Check ingestion throughput.",
        )


class FakeMlModel:
    classes_ = (
        "normal",
        "ingestion-backpressure",
        "malformed-sensor-payload",
    )

    def predict(
        self,
        values: list[list[float]],
    ) -> list[str]:
        assert values == [[1.5, 0.8, 0.0]]
        return ["ingestion-backpressure"]

    def predict_probability(
        self,
        values: list[list[float]],
    ) -> list[list[float]]:
        assert values == [[1.5, 0.8, 0.0]]
        return [[0.05, 0.90, 0.05]]


def fake_features() -> FeatureVector:
    # The backend tests do not access FeatureVector attributes because
    # vectorization and rule prediction are replaced by test doubles.
    return cast(FeatureVector, object())


def test_rules_backend_preserves_existing_prediction() -> None:
    backend = RulesPredictionBackend(
        predictor=FakeRulePredictor(),
    )

    result = backend.predict(fake_features())

    assert backend.backend_name is PredictionBackendName.RULES
    assert result == BackendPrediction(
        label="ingestion-backpressure",
        confidence=0.97,
        recommendation="Check ingestion throughput.",
        model_version="rules-0.1.0",
    )


def test_ml_backend_uses_probability_of_predicted_class() -> None:
    backend = MlPredictionBackend(
        model=FakeMlModel(),
        vectorizer=lambda features: (1.5, 0.8, 0.0),
        recommendation_resolver=lambda label, features: f"Recommendation for {label}.",
    )

    result = backend.predict(fake_features())

    assert backend.backend_name is PredictionBackendName.ML
    assert result.label == "ingestion-backpressure"
    assert result.confidence == pytest.approx(0.90)
    assert result.recommendation == ("Recommendation for ingestion-backpressure.")
    assert result.model_version == ML_MODEL_VERSION


def test_ml_backend_rejects_empty_feature_vector() -> None:
    backend = MlPredictionBackend(
        model=FakeMlModel(),
        vectorizer=lambda features: (),
        recommendation_resolver=lambda label, features: "unused",
    )

    with pytest.raises(
        ValueError,
        match="feature vector must not be empty",
    ):
        backend.predict(fake_features())


def test_ml_factory_requires_dependencies() -> None:
    with pytest.raises(
        RuntimeError,
        match="Missing dependencies",
    ):
        create_prediction_backend(
            PredictionBackendName.ML,
        )


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown prediction backend",
    ):
        create_prediction_backend("unsupported")


def test_factory_creates_rules_backend() -> None:
    backend = create_prediction_backend(
        PredictionBackendName.RULES,
        rule_predictor=FakeRulePredictor(),
    )

    assert isinstance(backend, RulesPredictionBackend)
