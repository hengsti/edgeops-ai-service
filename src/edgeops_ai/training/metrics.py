"""Classification metrics shared by collection, evaluation and model training."""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from edgeops_ai.training.common import LABELS


def label_scores(true_positive: int, predicted: int, expected: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": expected}


def classification_metrics(
    expected: Iterable[str], predicted: Iterable[str], *, include_unsupported: bool = False
) -> dict[str, Any]:
    pairs = list(zip(expected, predicted, strict=True))
    confusion = {label: dict.fromkeys(LABELS, 0) for label in LABELS}
    expected_counts = Counter(label for label, _ in pairs)
    predicted_counts = Counter(label for _, label in pairs)
    for expected_label, predicted_label in pairs:
        confusion[expected_label][predicted_label] += 1
    per_label = {
        label: label_scores(
            confusion[label][label], predicted_counts[label], expected_counts[label]
        )
        for label in LABELS
    }
    f1s = [
        metrics["f1"] for metrics in per_label.values() if include_unsupported or metrics["support"]
    ]
    correct = sum(expected_label == predicted_label for expected_label, predicted_label in pairs)
    return {
        "rows": len(pairs),
        "correct": correct,
        "accuracy": correct / len(pairs) if pairs else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "labels": list(LABELS),
        "confusion_matrix": confusion,
        "per_label": per_label,
    }
