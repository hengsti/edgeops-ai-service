"""Training-only calibration and offline health advice from validated TTM forecasts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from edgeops_ai.advisor import Advice
from edgeops_ai.forecast_advice import residual_advice
from edgeops_ai.forecasting import ATOL, ForecastModel
from edgeops_ai.training.advisor_forecast import load_candidate
from edgeops_ai.training.advisor_labels import file_hash

SUSTAINED_MINUTES = 15


def score_windows(
    model: ForecastModel, windows: dict[str, list]
) -> tuple[list[Advice], list[dict[str, Any]], dict[str, Any]]:
    contract = model.contract
    if contract["profile"] != "climate":
        raise ValueError("Device health requires the climate forecast contract")
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for split, items in windows.items():
        for window in items:
            grouped[window["entity_id"], window["room"]].append((split, window))
    metadata: dict[str, Any] = {
        "model_version": model.manifest["model_version"],
        "contract": contract,
        "windows": {split: len(items) for split, items in windows.items()},
        "calibration": "per-device/room/horizon/channel training absolute-residual 99th percentile",
        "scale_floor": ATOL,
        "score_threshold": 1,
        "sustained_minutes": SUSTAINED_MINUTES,
        "devices": {},
    }
    scores: list[dict[str, Any]] = []
    scales = {}
    for (device, room), items in sorted(grouped.items()):
        training = [window for split, window in items if split == "train"]
        if {split for split, _ in items} != {"train", "validation", "test"}:
            raise ValueError(f"Device/room {device}:{room} needs complete windows in every split")
        errors = np.asarray(
            [
                window["future"][0].astype(np.float64)
                - model.predict(window["past"][0], window["times"])[0].astype(np.float64)
                for window in training
            ]
        )
        scale = np.maximum(np.quantile(np.abs(errors), 0.99, axis=0), ATOL)
        scales[device, room] = scale
        metadata["devices"][f"{device}:{room}"] = {
            "training_windows": len(training),
            "training_residual_sha256": hashlib.sha256(errors.tobytes()).hexdigest(),
            "scale_by_horizon_channel": scale.tolist(),
        }
    for (device, room), items in sorted(grouped.items()):
        scale = scales[device, room]
        for split, window in items:
            forecast = model.predict(window["past"][0], window["times"])[0]
            residual = window["future"][0].astype(np.float64) - forecast.astype(np.float64)
            for step, at in enumerate(window["target_times"]):
                for channel, name in enumerate(contract["channels"]):
                    scores.append(
                        {
                            "entity_id": device,
                            "room": room,
                            "split": split,
                            "channel": name,
                            "time": at.isoformat(),
                            "last_input": window["times"][-1].isoformat(),
                            "horizon_step": step + 1,
                            "observed": float(window["future"][0, step, channel]),
                            "forecast": float(forecast[step, channel]),
                            "residual": float(residual[step, channel]),
                            "scale": float(scale[step, channel]),
                            "score": float(abs(residual[step, channel]) / scale[step, channel]),
                        }
                    )
    calibration_hash = hashlib.sha256(
        json.dumps(metadata["devices"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    findings = residual_advice(scores, model.manifest["model_version"], calibration_hash)
    return findings, scores, metadata


def residual_findings(
    input_dir: Path, candidate: Path
) -> tuple[list[Advice], list[dict[str, Any]], dict[str, Any]]:
    model, windows, hashes = load_candidate(input_dir, candidate)
    digest = hashes["model.onnx"]
    findings, scores, metadata = score_windows(model, windows)
    for finding in findings:
        finding.evidence["model_sha256"] = digest
    metadata["candidate_sha256"] = hashes
    metadata["dataset"] = model.manifest["dataset"]
    metadata["deployment_approved"] = False
    if any(file_hash(candidate / name) != digest for name, digest in hashes.items()):
        raise ValueError("Forecast candidate changed during replay")
    return findings, scores, metadata
