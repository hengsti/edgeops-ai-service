import hashlib
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import onnx
import pytest

from edgeops_ai.forecasting import MANIFEST_KEY, forecast_contract
from edgeops_ai.training.cli import main


@pytest.fixture
def benchmark_files(tmp_path):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["past_values"], ["forecast"])],
        "benchmark-fixture",
        [onnx.helper.make_tensor_value_info("past_values", onnx.TensorProto.FLOAT, [1, 16, 2])],
        [onnx.helper.make_tensor_value_info("forecast", onnx.TensorProto.FLOAT, [1, 16, 2])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.helper.set_model_props(
        model,
        {
            MANIFEST_KEY: json.dumps(
                {
                    "schema_version": "advisor-onnx-1",
                    "model_version": "benchmark-fixture",
                    "contract": forecast_contract("climate", 16, 16),
                }
            )
        },
    )
    artifact = tmp_path / "model.onnx"
    onnx.save(model, artifact)
    values = np.arange(64, dtype=np.float32).reshape(2, 16, 2)
    times = np.array(
        [[str(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i)) for i in range(16)]] * 2
    )
    samples = tmp_path / "samples.npz"
    np.savez_compressed(samples, values=values, times=times, expected=values.copy())
    return artifact, samples, values, times


def test_benchmark_uses_heldout_values_and_rejects_parity_failure(benchmark_files, capsys):
    artifact, samples, values, times = benchmark_files
    args = [
        "advisor-benchmark",
        "--model",
        str(artifact),
        "--sha256",
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "--inputs",
        str(samples),
        "--iterations",
        "3",
    ]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["parity_passed"] is True
    assert report["samples"] == 2
    assert report["iterations"] == 3
    assert report["latency_ms"]["p95"] >= 0
    np.savez_compressed(samples, values=values, times=times, expected=values + 1)
    assert main(args) == 1
    assert "parity" in capsys.readouterr().out


def test_benchmark_rejects_nonpositive_iteration_count(benchmark_files):
    from edgeops_ai.training.advisor_benchmark import benchmark

    artifact, samples, _, _ = benchmark_files
    with pytest.raises(ValueError, match="iterations"):
        benchmark(
            artifact, hashlib.sha256(artifact.read_bytes()).hexdigest(), samples, iterations=0
        )
