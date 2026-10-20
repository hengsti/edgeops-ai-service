"""Training-only calibration and offline health advice from validated TTM forecasts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np

from edgeops_ai.advisor import Advice
from edgeops_ai.forecasting import ATOL, RTOL, ForecastModel
from edgeops_ai.training.advisor_forecast import load_windows
from edgeops_ai.training.advisor_labels import file_hash
from edgeops_ai.training.reports import read_report

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
    findings = []
    minute = timedelta(minutes=1)

    # Grouping and adjacency restart persistence at split, room and missing-data boundaries.
    def key(row):
        return row["entity_id"], row["room"], row["split"], row["channel"]

    for (device, room, _, channel), series in groupby(sorted(scores, key=key), key=key):
        run: list[dict[str, Any]] = []
        for row in [*series, None]:
            if run and (
                row is None
                or row["score"] <= 1
                or row["time"] != (run[-1]["at"] + minute).isoformat()
            ):
                if len(run) >= SUSTAINED_MINUTES:
                    begin, end = run[0]["at"], run[-1]["at"] + minute
                    findings.append(
                        Advice(
                            f"device.forecast-residual.{channel}",
                            device,
                            begin,
                            end,
                            end,
                            f"{device}: {channel} differed from its forecast "
                            f"for {len(run)} minutes.",
                            "Inspect the channel history, sensor and room conditions; "
                            "the residual score does not identify a cause.",
                            {
                                "room": room,
                                "channel": channel,
                                "maximum_residual_score": max(r["score"] for r in run),
                                "sustained_minutes": len(run),
                                "model_version": model.manifest["model_version"],
                                "calibration_sha256": calibration_hash,
                            },
                            detector="model:ttm-residual@1",
                        )
                    )
                run = []
            if row is not None and row["score"] > 1:
                run.append({**row, "at": datetime.fromisoformat(row["time"])})
    return findings, scores, metadata


def residual_findings(
    input_dir: Path, candidate: Path
) -> tuple[list[Advice], list[dict[str, Any]], dict[str, Any]]:
    paths = {
        name: candidate / name
        for name in ("model.onnx", "model.onnx.sha256", "manifest.json", "validation-report.json")
    }
    hashes = {name: file_hash(path) for name, path in paths.items()}
    digest = paths["model.onnx.sha256"].read_text(encoding="ascii").strip()
    model = ForecastModel(paths["model.onnx"], digest)
    manifest = read_report(paths["manifest.json"])
    validation = read_report(paths["validation-report.json"])
    if manifest != {**model.manifest, "sha256": digest} or (
        validation.get("schema_version") != "advisor-validation-1"
        or validation.get("passed") is not True
        or validation.get("model_version") != model.manifest["model_version"]
        or validation.get("sources") != model.manifest["dataset"]
        or validation.get("tolerances") != {"atol": ATOL, "rtol": RTOL}
    ):
        raise ValueError("Expected a compatible validated #26 candidate and parity report")
    windows, sources = load_windows(input_dir, model.contract)
    if sources != model.manifest["dataset"]:
        raise ValueError("Forecast dataset must match the candidate's chronological dataset")
    findings, scores, metadata = score_windows(model, windows)
    for finding in findings:
        finding.evidence["model_sha256"] = digest
    metadata["candidate_sha256"] = hashes
    metadata["dataset"] = sources
    metadata["deployment_approved"] = False
    if any(file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Forecast candidate changed during replay")
    return findings, scores, metadata
