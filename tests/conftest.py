from collections.abc import Callable
from typing import Any

import pytest

from edgeops_ai.schemas import FeatureVector

HEALTHY_FEATURE_VALUES: dict[str, Any] = {
    "elapsed_seconds": 5.0,
    "messages_enqueued_rate": 10.0,
    "messages_processed_rate": 10.0,
    "transform_success_rate": 10.0,
    "queue_growth_rate": 0.0,
    "processing_ratio": 1.0,
    "queue_full_rate": 0.0,
    "transform_failure_ratio": 0.0,
    "persistence_ratio": 1.0,
    "pipeline_duration_average_seconds": 0.05,
    "heartbeat_missing": False,
}


@pytest.fixture
def make_features() -> Callable[..., FeatureVector]:
    """Build a healthy FeatureVector; keyword overrides inject faults."""

    def _make(**overrides: Any) -> FeatureVector:
        return FeatureVector(**{**HEALTHY_FEATURE_VALUES, **overrides})

    return _make
