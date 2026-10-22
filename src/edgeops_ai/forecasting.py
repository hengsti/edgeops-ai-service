"""Advisor TTM contract and ONNX inference; no training framework is imported here."""

import hashlib
import json
from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

PROFILES: dict[str, dict[str, Any]] = {
    "climate": {
        "entity_type": "device",
        "cadence_minutes": 1,
        "channels": ["temp_c_mean", "rel_hum_perc_mean"],
        "units": ["degC", "%"],
    },
    "host-cpu": {
        "entity_type": "host",
        "cadence_minutes": 5,
        "channels": ["usage_active_mean"],
        "units": ["%"],
    },
}
MANIFEST_KEY = "edgeops_advisor_manifest"
ATOL = 1e-4
RTOL = 1e-4


def forecast_contract(profile: str, context: int, horizon: int) -> dict[str, Any]:
    if profile not in PROFILES or type(context) is not int or type(horizon) is not int:
        raise ValueError("Unsupported Advisor forecast profile or tensor dimensions")
    if context <= 0 or horizon <= 0:
        raise ValueError("Context and horizon must be positive")
    return {
        "schema_version": "advisor-forecast-1",
        "profile": profile,
        **PROFILES[profile],
        "context_length": context,
        "prediction_length": horizon,
        "input": {
            "name": "past_values",
            "shape": [1, context, len(PROFILES[profile]["channels"])],
            "type": "tensor(float)",
        },
        "output": {
            "name": "forecast",
            "shape": [1, horizon, len(PROFILES[profile]["channels"])],
            "type": "tensor(float)",
        },
        "preprocessing": "raw physical units; TTM per-context std scaling inside the graph",
        "missing_data": "reject incomplete/nonfinite/irregular context; no imputation",
        "output_meaning": (
            "point forecasts in channel units; first target is one cadence after last input"
        ),
    }


def validate_contract(contract: dict[str, Any]) -> None:
    expected = forecast_contract(
        contract["profile"], contract["context_length"], contract["prediction_length"]
    )
    if contract != expected:
        raise ValueError("Incompatible Advisor forecast contract")


def prepare_input(values: Any, times: Sequence[datetime], contract: dict[str, Any]) -> np.ndarray:
    validate_contract(contract)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != tuple(contract["input"]["shape"][1:]) or not np.isfinite(matrix).all():
        raise ValueError("Expected complete finite context in contract channel order")
    cadence = timedelta(minutes=contract["cadence_minutes"])
    if (
        len(times) != len(matrix)
        or any(
            at.utcoffset() != timedelta(0)
            or at.second
            or at.microsecond
            or at.minute % contract["cadence_minutes"]
            for at in times
        )
        or any(b - a != cadence for a, b in pairwise(times))
    ):
        raise ValueError("Expected regular completed UTC aggregate timestamps")
    return matrix[np.newaxis, ...]


def load_artifact(candidate: Path, expected_sha256: str | None = None) -> tuple[Any, dict]:
    """Validate the same artifact/manifest/parity evidence offline and at runtime."""
    names = ("model.onnx", "model.onnx.sha256", "manifest.json", "validation-report.json")
    content = {name: (candidate / name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
    digest = content["model.onnx.sha256"].decode("ascii").strip()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Advisor artifact does not match the pinned model")
    model = ForecastModel(candidate / "model.onnx", digest)
    manifest = json.loads(content["manifest.json"])
    validation = json.loads(content["validation-report.json"])
    if manifest != {**model.manifest, "sha256": digest} or (
        validation.get("schema_version") != "advisor-validation-1"
        or validation.get("passed") is not True
        or validation.get("model_version") != model.manifest["model_version"]
        or validation.get("sources") != model.manifest["dataset"]
        or validation.get("tolerances") != {"atol": ATOL, "rtol": RTOL}
    ):
        raise ValueError("Expected a compatible validated #26 candidate and parity report")
    if any(
        hashlib.sha256((candidate / name).read_bytes()).hexdigest() != value
        for name, value in hashes.items()
    ):
        raise ValueError("Forecast candidate changed while loading")
    return model, hashes


def build_windows(
    points: dict,
    contract: dict,
    *,
    cuts: tuple = (),
    one_step: bool = False,
    boundaries: set | None = None,
    partial: bool = False,
    context_only: bool = False,
) -> dict[str, list]:
    """Keep offline and runtime segmentation, cadence and target alignment identical."""
    validate_contract(contract)
    kind, channels = contract["entity_type"], contract["channels"]
    context = contract["context_length"]
    horizon = 1 if one_step else contract["prediction_length"]
    windows: dict[str, list] = {
        key: [] for key in (("train", "validation", "test") if cuts else ("runtime",))
    }
    segment: list = []
    previous = None

    def emit(count: int) -> None:
        selected = segment[-context - count :]
        times = [at for at, _ in selected[:context]]
        windows[identity[2]].append(
            {
                "entity_id": identity[0],
                "room": identity[1],
                "split": identity[2],
                "times": times,
                "target_times": [at for at, _ in selected[context:]],
                "past": prepare_input([v for _, v in selected[:context]], times, contract),
                "future": np.asarray([v for _, v in selected[context:]], dtype=np.float32)[
                    None, ...
                ],
            }
        )

    def finish() -> None:
        if context_only and len(segment) >= context:
            emit(0)
            return
        remainder = (len(segment) - context) % horizon
        if partial and len(segment) > context and remainder:
            emit(remainder)

    identity = ("", "", "runtime")
    for (entity, at), point in sorted(points.items()):
        split = ("train", "validation", "test")[bisect_right(cuts, at)] if cuts else "runtime"
        valid = all(type(point.get(c)) in (float, int) and np.isfinite(point[c]) for c in channels)
        if one_step and kind == "host":
            valid = valid and all(0 <= point[c] <= 100 for c in channels)
        if kind == "device":
            valid = (
                valid
                and point.get("warmed_up_last") == "true"
                and point.get("time_valid_last") != "false"
                and bool(point["room"])
            )
        next_identity = entity, point["room"], split
        reset = (
            bool(boundaries and (entity, at) in boundaries)
            or point.get("uptime_resets", 0)
            or (
                previous is not None
                and point.get("uptime_last", float("inf"))
                < previous[2].get("uptime_last", -float("inf"))
            )
        )
        if (
            not valid
            or previous is None
            or previous[0] != next_identity
            or (at - previous[1]).total_seconds() != contract["cadence_minutes"] * 60
            or point.get("source_identity") != previous[2].get("source_identity")
            or reset
        ):
            finish()
            segment = []
        identity = next_identity
        previous = identity, at, point
        if not valid:
            continue
        segment.append((at, [point[c] for c in channels]))
        if (
            not context_only
            and len(segment) >= context + horizon
            and (len(segment) - context - horizon) % horizon == 0
        ):
            emit(horizon)
    finish()
    return windows


class ForecastModel:
    def __init__(self, path: Path, expected_sha256: str):
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256.lower():
            raise ValueError("Advisor ONNX SHA-256 mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                content, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.manifest = json.loads(
                self.session.get_modelmeta().custom_metadata_map[MANIFEST_KEY]
            )
            if (
                self.manifest["schema_version"] != "advisor-onnx-1"
                or not self.manifest["model_version"]
            ):
                raise ValueError("Incompatible Advisor manifest")
            self.contract = self.manifest["contract"]
            validate_contract(self.contract)
            for nodes, key in (
                (self.session.get_inputs(), "input"),
                (self.session.get_outputs(), "output"),
            ):
                expected = self.contract[key]
                if len(nodes) != 1 or (nodes[0].name, nodes[0].shape, nodes[0].type) != (
                    expected["name"],
                    expected["shape"],
                    expected["type"],
                ):
                    raise ValueError("ONNX graph does not match the Advisor tensor contract")
        except Exception as error:
            raise ValueError(f"Invalid Advisor ONNX artifact: {error}") from error

    def predict(self, values: Any, times: Sequence[datetime]) -> np.ndarray:
        matrix = prepare_input(values, times, self.contract)
        result = self.session.run(["forecast"], {"past_values": matrix})[0]
        if result.shape != tuple(self.contract["output"]["shape"]) or not np.isfinite(result).all():
            raise ValueError("Invalid Advisor forecast output")
        return result
