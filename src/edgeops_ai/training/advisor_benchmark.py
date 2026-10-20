"""Measure Advisor ONNX inference on held-out inputs without training dependencies."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from edgeops_ai.forecasting import ATOL, RTOL, ForecastModel


def benchmark(
    model_path: Path, sha256: str, inputs: Path, *, iterations: int = 50
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("Benchmark iterations must be positive")
    started = perf_counter()
    model = ForecastModel(model_path, sha256)
    load_ms = (perf_counter() - started) * 1000
    with np.load(inputs, allow_pickle=False) as data:
        values, times, expected = data["values"], data["times"], data["expected"]
    if (
        values.ndim != 3
        or not len(values)
        or values.shape[1:] != tuple(model.contract["input"]["shape"][1:])
        or times.shape != values.shape[:2]
        or expected.shape != (len(values), *model.contract["output"]["shape"][1:])
    ):
        raise ValueError("Benchmark inputs do not match the forecast contract")
    timestamps = [[datetime.fromisoformat(str(at)) for at in row] for row in times]
    drift = 0.0
    for index in range(len(values)):
        actual = model.predict(values[index], timestamps[index])[0]
        if not np.allclose(actual, expected[index], atol=ATOL, rtol=RTOL):
            raise ValueError("Benchmark native/ONNX parity failed")
        drift = max(drift, float(np.max(np.abs(actual - expected[index]))))
    for _ in range(5):
        model.predict(values[0], timestamps[0])
    durations = []
    for index in range(iterations):
        sample = index % len(values)
        started = perf_counter()
        model.predict(values[sample], timestamps[sample])
        durations.append((perf_counter() - started) * 1000)
    peak_bytes = None
    if sys.platform == "linux":
        import resource

        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    p95 = float(np.percentile(durations, 95))
    return {
        "schema_version": "advisor-benchmark-1",
        "model_version": model.manifest["model_version"],
        "model_sha256": sha256,
        "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": version("onnxruntime"),
        "contract": model.contract,
        "provider": "CPUExecutionProvider",
        "threads": 1,
        "samples": len(values),
        "iterations": iterations,
        "warmup_iterations": 5,
        "parity_passed": True,
        "max_absolute_drift": drift,
        "latency_ms": {
            "median": float(np.median(durations)),
            "p95": p95,
            "maximum": max(durations),
        },
        "load_ms": load_ms,
        "peak_process_rss_bytes": peak_bytes,
        "memory_measurement": "Linux process high-water RSS; unavailable on other platforms",
        "resource_targets": {
            "p95_latency_ms_below": 50,
            "peak_process_rss_bytes_below": 100_000_000,
        },
        "resource_targets_met": peak_bytes is not None and peak_bytes < 100_000_000 and p95 < 50,
        "training_packages_absent": {
            name: importlib.util.find_spec(name) is None
            for name in ("torch", "tsfm_public", "sklearn", "pyarrow", "onnx", "onnxscript")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="NPZ with values, UTC times and native expected outputs",
    )
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(benchmark(args.model, args.sha256, args.inputs, iterations=args.iterations))
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Advisor benchmark failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
