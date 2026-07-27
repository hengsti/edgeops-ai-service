from math import isfinite

from edgeops_ai.schemas import FeatureVector

MODEL_FEATURE_COLUMNS = (
    "queue_growth_ratio",
    "processing_ratio",
    "queue_full_ratio",
    "transform_failure_ratio",
    "persistence_ratio",
    "heartbeat_missing",
)


def _bounded(value: float) -> float:
    if not isfinite(value):
        raise ValueError("ML feature values must be finite.")

    return min(max(value, 0.0), 1.0)


def _bounded_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0

    return _bounded(numerator / denominator)


def build_model_feature_row(
    *,
    messages_enqueued_rate: float,
    queue_growth_rate: float,
    processing_ratio: float,
    queue_full_rate: float,
    transform_failure_ratio: float,
    persistence_ratio: float,
    heartbeat_missing: bool | float,
) -> list[float]:
    """
    Build the workload-independent feature vector used for training
    and deployed inference.

    The order must exactly match MODEL_FEATURE_COLUMNS.
    """

    return [
        _bounded_ratio(
            max(queue_growth_rate, 0.0),
            messages_enqueued_rate,
        ),
        _bounded(processing_ratio),
        _bounded_ratio(
            queue_full_rate,
            messages_enqueued_rate,
        ),
        _bounded(transform_failure_ratio),
        _bounded(persistence_ratio),
        float(bool(heartbeat_missing)),
    ]


def vectorize_model_features(
    features: FeatureVector,
) -> list[float]:
    return build_model_feature_row(
        messages_enqueued_rate=float(features.messages_enqueued_rate),
        queue_growth_rate=float(features.queue_growth_rate),
        processing_ratio=float(features.processing_ratio),
        queue_full_rate=float(features.queue_full_rate),
        transform_failure_ratio=float(features.transform_failure_ratio),
        persistence_ratio=float(features.persistence_ratio),
        heartbeat_missing=features.heartbeat_missing,
    )
