"""Advisor TTM contract and ONNX inference; no training framework is imported here."""

import hashlib
import json
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
