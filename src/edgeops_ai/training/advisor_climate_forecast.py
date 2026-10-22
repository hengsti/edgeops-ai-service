"""Offline room forecasts and held-out climate comparisons using the #26 artifact."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from edgeops_ai.advisor import Advice
from edgeops_ai.forecast_advice import climate_projection
from edgeops_ai.forecasting import ForecastModel
from edgeops_ai.training.advisor_forecast import load_candidate
from edgeops_ai.training.advisor_labels import file_hash


def crossing(humidity: np.ndarray, current: float) -> int | None:
    above = np.flatnonzero(humidity > 60)
    return int(above[0]) if current <= 60 and len(above) else None


def metrics(items: list) -> dict[str, Any]:
    counts = dict.fromkeys(
        ("true_positive", "false_positive", "false_negative", "true_negative"), 0
    )
    timing_errors = []
    for actual, predicted, current in items:
        if current > 60:
            continue
        observed, forecast = crossing(actual[:, 1], current), crossing(predicted[:, 1], current)
        if observed is not None and forecast is not None:
            counts["true_positive"] += 1
            timing_errors.append(abs(forecast - observed))
        elif forecast is not None:
            counts["false_positive"] += 1
        elif observed is not None:
            counts["false_negative"] += 1
        else:
            counts["true_negative"] += 1
    error = np.asarray([predicted - actual for actual, predicted, _ in items], dtype=np.float64)
    return {
        "windows": len(items),
        "mae_by_horizon_channel": np.abs(error).mean(axis=0).tolist() if items else [],
        "rmse_by_horizon_channel": np.sqrt((error**2).mean(axis=0)).tolist() if items else [],
        "mae_30_60_by_channel": np.abs(error[:, 30:61]).mean(axis=(0, 1)).tolist() if items else [],
        "humidity_crossings": {
            **counts,
            "eligible_windows": sum(counts.values()),
            "timing_mae_minutes": float(np.mean(timing_errors)) if timing_errors else None,
        },
    }


def score_windows(
    model: ForecastModel, windows: dict[str, list]
) -> tuple[list[Advice], list[dict[str, Any]], dict[str, Any]]:
    if model.contract["profile"] != "climate" or model.contract["prediction_length"] < 61:
        raise ValueError("Climate advice needs the climate contract with a full 60-minute lead")
    # The origin is the completed context; target zero is the next aggregate's start.
    training: dict[tuple[str, str], dict] = defaultdict(dict)
    for window in windows["train"]:
        training[window["entity_id"], window["room"]].update(
            zip(
                window["times"] + window["target_times"],
                np.concatenate((window["past"][0], window["future"][0])),
                strict=True,
            )
        )
    buckets: dict[tuple[str, str, int], list] = defaultdict(list)
    for (device, room), points in sorted(training.items()):
        for at, values in sorted(points.items()):
            buckets[device, room, at.hour * 60 + at.minute].append(values)
    seasonal = {key: np.mean(values, axis=0, dtype=np.float64) for key, values in buckets.items()}
    profile_hash = hashlib.sha256(
        json.dumps(
            [(key, values.tolist()) for key, values in sorted(seasonal.items())], allow_nan=False
        ).encode()
    ).hexdigest()
    findings, forecasts = [], []
    compared: dict[str, dict[str, list]] = {
        split: {name: [] for name in ("ttm", "last_value", "seasonal")}
        for split in ("validation", "test")
    }
    for split, items in windows.items():
        for window in items:
            current = float(window["past"][0, -1, 1])
            prediction = model.predict(window["past"][0], window["times"])[0, :61]
            if np.any((prediction[:, 1] < 0) | (prediction[:, 1] > 100)):
                raise ValueError("Forecast humidity must be between 0 and 100 percent")
            targets = window["target_times"][:61]
            frame, advice = climate_projection(
                {**window, "split": split}, prediction, model.manifest["model_version"]
            )
            forecasts.append(frame)
            if advice is not None:
                findings.append(advice)
            if split not in compared:
                continue
            keys = [
                (window["entity_id"], window["room"], at.hour * 60 + at.minute) for at in targets
            ]
            if any(key not in seasonal for key in keys):
                continue
            actual = window["future"][0, :61].astype(np.float64)
            for name, predicted in (
                ("ttm", prediction.astype(np.float64)),
                ("last_value", np.broadcast_to(window["past"][0, -1], actual.shape)),
                ("seasonal", np.asarray([seasonal[key] for key in keys])),
            ):
                compared[split][name].append((actual, predicted, current))
    metadata = {
        "model_version": model.manifest["model_version"],
        "contract": model.contract,
        "windows": {split: len(items) for split, items in windows.items()},
        "lead_minutes": list(range(61)),
        "humidity_threshold_percent": 60,
        "seasonal_baseline": "per-device/room UTC minute-of-day mean of unique training samples",
        "seasonal_profile_sha256": profile_hash,
        "seasonal_training_samples": sum(len(points) for points in training.values()),
        "evaluation": {
            split: {
                "windows": len(windows[split]),
                "compared_windows": len(models["ttm"]),
                "seasonal_unavailable_windows": len(windows[split]) - len(models["ttm"]),
                "models": {name: metrics(items) for name, items in models.items()},
            }
            for split, models in compared.items()
        },
        "deployment_approved": False,
    }
    return findings, forecasts, metadata


def forecast_findings(
    input_dir: Path, candidate: Path
) -> tuple[list[Advice], list[dict[str, Any]], dict[str, Any]]:
    model, windows, hashes = load_candidate(input_dir, candidate)
    findings, forecasts, metadata = score_windows(model, windows)
    for finding in findings:
        finding.evidence["model_sha256"] = hashes["model.onnx"]
    for frame in forecasts:
        frame["model_sha256"] = hashes["model.onnx"]
    metadata["candidate_sha256"] = hashes
    metadata["dataset"] = model.manifest["dataset"]
    if any(file_hash(candidate / name) != digest for name, digest in hashes.items()):
        raise ValueError("Forecast candidate changed during replay")
    return findings, forecasts, metadata
