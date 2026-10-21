"""Fine-tune a trusted local TTM checkpoint, export ONNX and validate chronological holdouts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from edgeops_ai.forecasting import (
    ATOL,
    MANIFEST_KEY,
    PROFILES,
    RTOL,
    ForecastModel,
    forecast_contract,
    prepare_input,
    validate_contract,
)
from edgeops_ai.training.advisor_dataset import SCHEMA, timestamp
from edgeops_ai.training.advisor_health import split_at
from edgeops_ai.training.advisor_labels import file_hash
from edgeops_ai.training.common import positive_int
from edgeops_ai.training.reports import read_report, write_report


def load_windows(
    input_dir: Path,
    contract: dict[str, Any],
    *,
    one_step: bool = False,
    boundaries: set[tuple[str, datetime]] | None = None,
) -> tuple[dict[str, list], dict[str, str]]:
    validate_contract(contract)
    kind = contract["entity_type"]
    paths = {
        "data-card.json": input_dir / "data-card.json",
        f"{kind}.parquet": input_dir / f"{kind}.parquet",
    }
    hashes = {name: file_hash(path) for name, path in paths.items()}
    card = read_report(paths["data-card.json"])
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    train_stop, val_stop = (
        timestamp(card["split"]["train_stop"]),
        timestamp(card["split"]["validation_stop"]),
    )
    if (
        card["schema_version"] != "advisor-1"
        or not start < train_stop < val_stop < stop
        or card["split"]["shuffle"] is not False
    ):
        raise ValueError("Expected an advisor-1 dataset with chronological splits")
    parquet = pq.ParquetFile(paths[f"{kind}.parquet"])
    if (
        parquet.schema_arrow != SCHEMA
        or parquet.metadata.num_rows != card["tables"][kind]["rows"]
        or hashes[f"{kind}.parquet"] != card["tables"][kind]["sha256"]
    ):
        raise ValueError("Advisor source schema, row count or SHA-256 mismatch")
    channels = contract["channels"]
    fields = pa.array(
        [*channels, "warmed_up_last", "time_valid_last", "uptime_last", "uptime_resets"]
    )
    points: dict[tuple[str, Any], dict[str, Any]] = {}
    for batch in parquet.iter_batches():
        table = pa.Table.from_batches([batch])
        for row in table.filter(pc.is_in(table["field"], value_set=fields)).to_pylist():
            if row["measurement"] not in (
                {"bme680", "device_status"} if kind == "device" else {"cpu"}
            ):
                continue
            if row["field"] == "time_valid_last" and row["measurement"] != "bme680":
                continue
            dimensions = json.loads(row["dimensions"])
            if kind == "host" and dimensions.get("cpu") != "cpu-total":
                continue
            at = row["time"]
            cadence = contract["cadence_minutes"]
            if (
                not row["entity_id"]
                or not start <= at < stop
                or at.second
                or at.microsecond
                or at.minute % cadence
                or row["interval_minutes"] != cadence
                or row["split"] != split_at(at, train_stop, val_stop)
            ):
                raise ValueError("Aggregate cadence, timestamp or split mismatch")
            point = points.setdefault((row["entity_id"], at), {"room": dimensions.get("room", "")})
            room = dimensions.get("room", "")
            if room and point["room"] and room != point["room"]:
                raise ValueError("Conflicting room tags")
            if room:
                point["room"] = room
            name = row["field"]
            value = row["value"] if row["value"] is not None else row["text"]
            if name in point or (
                isinstance(value, float)
                and (not math.isfinite(value) or abs(value) > float(np.finfo(np.float32).max))
            ):
                raise ValueError("Duplicate aggregate or value outside the finite float32 range")
            point[name] = value
    windows: dict[str, list] = {key: [] for key in ("train", "validation", "test")}
    context = contract["context_length"]
    horizon = 1 if one_step else contract["prediction_length"]
    segment: list = []
    previous = None
    for (entity, at), point in sorted(points.items()):
        split = split_at(at, train_stop, val_stop)
        valid = all(isinstance(point.get(c), (float, int)) for c in channels)
        if one_step and kind == "host":
            valid = valid and all(0 <= point[c] <= 100 for c in channels)
        if kind == "device":
            valid = (
                valid
                and point.get("warmed_up_last") == "true"
                and point.get("time_valid_last") != "false"
                and bool(point["room"])
            )
        identity = (entity, point["room"], split)
        reset = (
            (boundaries is not None and (entity, at) in boundaries)
            or point.get("uptime_resets", 0)
            or (
                previous is not None
                and point.get("uptime_last", math.inf) < previous[2].get("uptime_last", -math.inf)
            )
        )
        if (
            not valid
            or previous is None
            or previous[0] != identity
            or (at - previous[1]).total_seconds() != contract["cadence_minutes"] * 60
            or reset
        ):
            segment = []
        previous = identity, at, point
        if not valid:
            continue
        segment.append((at, [point[c] for c in channels]))
        if len(segment) >= context + horizon and (len(segment) - context - horizon) % horizon == 0:
            selected = segment[-context - horizon :]
            times = [t for t, _ in selected[:context]]
            windows[split].append(
                {
                    "entity_id": entity,
                    "room": point["room"],
                    "times": times,
                    "target_times": [t for t, _ in selected[context:]],
                    "past": prepare_input([v for _, v in selected[:context]], times, contract),
                    "future": np.asarray([v for _, v in selected[context:]], dtype=np.float32)[
                        None, ...
                    ],
                }
            )
    if any(file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Advisor source changed while building windows")
    if not one_step and any(not windows[split] for split in windows):
        counts = {split: len(items) for split, items in windows.items()}
        raise ValueError(f"Every split needs a complete forecast window; counts: {counts}")
    return windows, hashes


def checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    return {p.name: file_hash(p) for p in checkpoint.iterdir() if p.is_file()}


def load_candidate(
    input_dir: Path,
    candidate: Path,
    *,
    one_step: bool = False,
    boundaries: set[tuple[str, datetime]] | None = None,
) -> tuple[ForecastModel, dict[str, list], dict[str, str]]:
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
    windows, sources = load_windows(
        input_dir, model.contract, one_step=one_step, boundaries=boundaries
    )
    if sources != model.manifest["dataset"]:
        raise ValueError("Forecast dataset must match the candidate's chronological dataset")
    if any(file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Forecast candidate changed while loading")
    return model, windows, hashes


def evaluate(native: Any, exported: ForecastModel, windows: list) -> dict[str, Any]:
    import torch

    errors, baseline_errors, max_drift = [], [], 0.0
    for window in windows:
        with torch.no_grad():
            expected = native(torch.from_numpy(window["past"])).numpy()
        actual = exported.predict(window["past"][0], window["times"])
        if not np.isfinite(expected).all() or not np.allclose(
            actual, expected, atol=ATOL, rtol=RTOL
        ):
            raise ValueError("Native/ONNX parity failed")
        max_drift = max(max_drift, float(np.max(np.abs(actual - expected))))
        errors.append(actual[0] - window["future"][0])
        baseline_errors.append(window["past"][0, -1] - window["future"][0])
    error, baseline = (
        np.asarray(errors, dtype=np.float64),
        np.asarray(baseline_errors, dtype=np.float64),
    )
    return {
        "windows": len(windows),
        "max_absolute_drift": max_drift,
        "mae_by_horizon_channel": np.abs(error).mean(axis=0).tolist(),
        "rmse_by_horizon_channel": np.sqrt((error**2).mean(axis=0)).tolist(),
        "last_value_mae_by_horizon_channel": np.abs(baseline).mean(axis=0).tolist(),
        "origins": [
            {
                "entity_id": w["entity_id"],
                "room": w["room"],
                "context_start": w["times"][0].isoformat(),
                "last_input": w["times"][-1].isoformat(),
                "first_target": w["target_times"][0].isoformat(),
                "last_target": w["target_times"][-1].isoformat(),
            }
            for w in windows
        ],
    }


def pipeline(
    input_dir: Path,
    checkpoint: Path,
    output: Path,
    *,
    expected_sha256: str,
    profile: str,
    model_version: str,
    epochs: int = 1,
    export_only: bool = False,
) -> dict[str, Any]:
    import copy

    import onnx
    import torch
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    if output.exists():
        raise ValueError("Output already exists; choose a new candidate directory")
    if any(output.resolve().is_relative_to(source.resolve()) for source in (input_dir, checkpoint)):
        raise ValueError("Output must be outside the source dataset and checkpoint")
    if not model_version.strip() or epochs <= 0:
        raise ValueError("A model version and positive epoch count are required")
    before = checkpoint_hashes(checkpoint)
    if before["model.safetensors"] != expected_sha256.lower():
        raise ValueError("Native checkpoint SHA-256 mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = TinyTimeMixerForPrediction.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_safetensors=True,
        num_input_channels=len(PROFILES[profile]["channels"]),
    ).cpu()
    config = model.config
    if (
        config.scaling != "std"
        or config.loss != "mse"
        or config.resolution_prefix_tuning
        or config.exogenous_channel_indices
        or config.prediction_channel_indices
        or config.categorical_vocab_size_list
    ):
        raise ValueError("Unsupported TTM configuration for the Advisor contract")
    contract = forecast_contract(
        profile, config.context_length, config.prediction_filter_length or config.prediction_length
    )
    if (checkpoint / "forecast-contract.json").exists() and read_report(
        checkpoint / "forecast-contract.json"
    ) != contract:
        raise ValueError("Native checkpoint has an incompatible Advisor contract")
    windows, sources = load_windows(input_dir, contract)
    training_start = perf_counter()
    losses = []
    best_loss = math.inf
    if not export_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(epochs):
            model.train()
            for window in windows["train"]:
                optimizer.zero_grad()
                loss = model(
                    past_values=torch.from_numpy(window["past"]),
                    future_values=torch.from_numpy(window["future"]),
                ).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite native training loss")
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    float(
                        model(
                            past_values=torch.from_numpy(w["past"]),
                            future_values=torch.from_numpy(w["future"]),
                        ).loss
                    )
                    for w in windows["validation"]
                ) / len(windows["validation"])
            if not math.isfinite(val_loss):
                raise ValueError("Nonfinite native validation loss")
            losses.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
    model.eval()
    training = {
        "epochs": 0 if export_only else epochs,
        "validation_loss_by_epoch": losses,
        "seconds": perf_counter() - training_start,
        "seed": 7,
        "device": "cpu",
    }

    class NativeForecast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, past_values):
            return self.model(past_values=past_values, return_loss=False).prediction_outputs

    native = NativeForecast().eval()
    manifest = {
        "schema_version": "advisor-onnx-1",
        "model_version": model_version,
        "contract": contract,
        "source_checkpoint": before,
        "dataset": sources,
        "versions": {
            name: version(name) for name in ("torch", "granite-tsfm", "onnx", "onnxruntime")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".advisor-forecast-", dir=output.parent) as temporary:
        staging = Path(temporary) / "candidate"
        staging.mkdir()
        artifact = staging / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                native,
                (torch.from_numpy(windows["validation"][0]["past"]),),
                artifact,
                input_names=["past_values"],
                output_names=["forecast"],
                opset_version=18,
                dynamo=True,
                external_data=False,
                verbose=False,
            )
        graph = onnx.load(artifact)
        onnx.helper.set_model_props(graph, {MANIFEST_KEY: json.dumps(manifest, sort_keys=True)})
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, artifact)
        digest = file_hash(artifact)
        exported = ForecastModel(artifact, digest)
        report = {
            "schema_version": "advisor-validation-1",
            "model_version": model_version,
            "passed": True,
            "deployment_approved": False,
            "deployment_blockers": [
                "Model selection/seasonal benchmarks and measured Pi latency/memory required"
            ],
            "tolerances": {"atol": ATOL, "rtol": RTOL},
            "training": training,
            "train_windows": len(windows["train"]),
            "sources": sources,
            "splits": {
                split: evaluate(native, exported, windows[split])
                for split in ("validation", "test")
            },
        }
        if checkpoint_hashes(checkpoint) != before or any(
            file_hash(input_dir / name) != digest for name, digest in sources.items()
        ):
            raise ValueError("Checkpoint or dataset changed during the pipeline")
        if not export_only:
            model.save_pretrained(staging / "checkpoint")
            write_report(contract, staging / "checkpoint" / "forecast-contract.json")
        write_report({**manifest, "sha256": digest}, staging / "manifest.json")
        write_report(report, staging / "validation-report.json")
        (staging / "model.onnx.sha256").write_text(digest + "\n", encoding="ascii")
        staging.rename(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Trusted local TTM save_pretrained directory"
    )
    parser.add_argument("--sha256", required=True, help="Recorded SHA-256 of model.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = pipeline(
            args.input_dir,
            args.checkpoint,
            args.output_dir,
            expected_sha256=args.sha256,
            profile=args.profile,
            model_version=args.model_version,
            epochs=args.epochs,
            export_only=args.export_only,
        )
        print(
            f"Validated Advisor ONNX candidate: {args.output_dir}; "
            f"deployment approved: {report['deployment_approved']}"
        )
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        pa.ArrowException,
    ) as error:
        print(f"Advisor forecast pipeline failed: {error}")
        return 1
