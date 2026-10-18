"""Compare normal simulation features with saved production observations."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from edgeops_ai.features import FeatureExtractor
from edgeops_ai.ml import FEATURE_CONTRACT_VERSION, MODEL_FEATURE_COLUMNS, vectorize_model_features
from edgeops_ai.rules import RuleBasedPredictor
from edgeops_ai.schemas import FeatureVector, ObservationV1
from edgeops_ai.training.common import RECOVERY_PHASES, add_input_dir
from edgeops_ai.training.dataset import load_training_dataset
from edgeops_ai.training.manifest import validate_manifest_datasets
from edgeops_ai.training.reports import write_report

DISTRIBUTION_COLUMNS = (*MODEL_FEATURE_COLUMNS, "raw_processing_ratio")
MIN_WINDOWS = 20


def simulation_digest(input_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(input_dir.glob("*-seed-*.csv")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes())
    return digest.hexdigest()


def build_distribution_report(
    simulation: np.ndarray,
    production: np.ndarray,
    *,
    max_ks_distance: float = 0.3,
    max_outside_fraction: float = 0.1,
) -> dict[str, Any]:
    for threshold in (max_ks_distance, max_outside_fraction):
        if not 0 <= threshold < 1:
            raise ValueError(
                "Distribution thresholds must be finite and between 0 and 1 (exclusive)."
            )
    for matrix in (simulation, production):
        if matrix.ndim != 2 or matrix.shape[1] != len(DISTRIBUTION_COLUMNS):
            raise ValueError("Distribution matrix does not match the feature contract.")
        if len(matrix) < MIN_WINDOWS or not np.isfinite(matrix).all():
            raise ValueError(
                f"At least {MIN_WINDOWS} finite feature windows are required per source."
            )
    features = {}
    for index, name in enumerate(DISTRIBUTION_COLUMNS):
        reference, observed = simulation[:, index], production[:, index]
        lower, upper = float(reference.min()), float(reference.max())
        distance = float(ks_2samp(reference, observed, method="asymp").statistic)
        outside = float(np.mean((observed < lower) | (observed > upper)))
        features[name] = {
            "simulation_range": [lower, upper],
            "production_range": [float(observed.min()), float(observed.max())],
            "ks_distance": distance,
            "outside_fraction": outside,
            "passed": distance <= max_ks_distance and outside <= max_outside_fraction,
        }
    return {
        "schema_version": "1.0",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "baseline": "stable-normal-simulation",
        "simulation_rows": len(simulation),
        "production_rows": len(production),
        "thresholds": {
            "max_ks_distance": max_ks_distance,
            "max_outside_fraction": max_outside_fraction,
        },
        "features": features,
        "passed": all(item["passed"] for item in features.values()),
    }


def distribution_passed(report: Mapping[str, Any]) -> bool:
    """Fail closed on missing, incompatible or internally inconsistent evidence."""
    try:
        if (
            report["schema_version"] != "1.0"
            or report["feature_contract_version"] != FEATURE_CONTRACT_VERSION
            or report["baseline"] != "stable-normal-simulation"
            or any(
                type(report[name]) is not int or report[name] < MIN_WINDOWS
                for name in ("simulation_rows", "production_rows")
            )
            or set(report["features"]) != set(DISTRIBUTION_COLUMNS)
        ):
            return False
        thresholds = report["thresholds"]
        limits = (thresholds["max_ks_distance"], thresholds["max_outside_fraction"])
        if any(not 0 <= value < 1 for value in limits):
            return False
        for feature in report["features"].values():
            for key, limit in zip(("ks_distance", "outside_fraction"), limits, strict=True):
                value = feature[key]
                if (
                    not isinstance(value, (int, float))
                    or not isfinite(value)
                    or not 0 <= value <= limit
                ):
                    return False
        return report["passed"] is True
    except (KeyError, TypeError, AttributeError):
        return False


def load_production_features(path: Path) -> tuple[list[FeatureVector], dict[str, Any]]:
    content = path.read_bytes()
    extractor = FeatureExtractor()
    features = []
    observations = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            observation = ObservationV1.model_validate(json.loads(line))
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid production observation: {error}"
            ) from error
        if observation.metadata.simulated or observation.metadata.collector_mode != "production":
            raise ValueError(f"{path}:{line_number}: expected a real production observation.")
        observations.append(observation)
        result = extractor.update(observation)
        if result is not None:
            features.append(result)
    return features, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "observations": len(observations),
        "feature_windows": len(features),
        "source_hosts": sorted({item.metadata.source_host for item in observations}),
        "captured_from": min(
            (item.metadata.captured_at.isoformat() for item in observations), default=None
        ),
        "captured_until": max(
            (item.metadata.captured_at.isoformat() for item in observations), default=None
        ),
    }


def check_distributions(
    *,
    input_dir: Path,
    production_observations: Path,
    output: Path,
    max_ks_distance: float = 0.3,
    max_outside_fraction: float = 0.1,
) -> dict[str, Any]:
    validate_manifest_datasets(input_dir)
    dataset = load_training_dataset(input_dir)
    normal = dataset.features[dataset.labels == "normal"]
    ratios = []
    for name in dataset.input_files:
        with (input_dir / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["expected_label"].strip().lower() == "normal"
                    and row["simulation_phase"].strip().lower() not in RECOVERY_PHASES
                ):
                    ratios.append(float(row["processing_ratio"]))
    production, source = load_production_features(production_observations)
    observed = np.asarray(
        [[*vectorize_model_features(row), row.processing_ratio] for row in production]
    )
    if not production:
        observed = np.empty((0, len(DISTRIBUTION_COLUMNS)))
    report = build_distribution_report(
        np.column_stack((normal, ratios)),
        observed,
        max_ks_distance=max_ks_distance,
        max_outside_fraction=max_outside_fraction,
    )
    report["simulation"] = {
        "path": str(input_dir.resolve()),
        "sha256": simulation_digest(input_dir),
    }
    report["production"] = source
    predictor = RuleBasedPredictor()
    report["production_rules_predictions"] = dict(
        Counter(predictor.predict(row).label for row in production)
    )
    report["production_queue_growth_range"] = [
        min(row.queue_growth_rate for row in production),
        max(row.queue_growth_rate for row in production),
    ]
    write_report(report, output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_dir(parser, required=True)
    parser.add_argument(
        "--production-observations",
        type=Path,
        required=True,
        help="JSONL ObservationV1 snapshots from a representative normal production period.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-ks-distance", type=float, default=0.3)
    parser.add_argument("--max-outside-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    output = args.output or args.input_dir / "distribution-report.json"
    try:
        report = check_distributions(
            input_dir=args.input_dir,
            production_observations=args.production_observations,
            output=output,
            max_ks_distance=args.max_ks_distance,
            max_outside_fraction=args.max_outside_fraction,
        )
    except (OSError, ValueError) as error:
        print(f"Production distribution check failed: {error}")
        return 2
    failed = [name for name, item in report["features"].items() if not item["passed"]]
    print(f"Distribution report: {output}; failed features: {', '.join(failed) or 'none'}")
    return 0 if report["passed"] else 1
