"""Replay indoor climate advice against an existing Advisor history export."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from edgeops_ai.advisor import MINUTE, Advice
from edgeops_ai.climate import climate_findings
from edgeops_ai.influx_reader import CLIMATE_FIELDS
from edgeops_ai.training.advisor_dataset import timestamp
from edgeops_ai.training.advisor_health import load_devices, split_at
from edgeops_ai.training.advisor_labels import file_hash
from edgeops_ai.training.reports import read_report, write_report
from edgeops_ai.weather import WeatherObservation


def replay(
    input_dir: Path,
    output: Path,
    *,
    humidity_minutes: int = 120,
    forecast_candidate: Path | None = None,
    weather_observations: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Output already exists; choose a new report directory")
    source_paths = {
        "data_card_sha256": input_dir / "data-card.json",
        "device_parquet_sha256": input_dir / "device.parquet",
    }
    if weather_observations is not None:
        source_paths["weather_observations_sha256"] = weather_observations
    hashes = {name: file_hash(path) for name, path in source_paths.items()}
    card = read_report(source_paths["data_card_sha256"])
    if card.get("schema_version") != "advisor-1":
        raise ValueError("Expected an advisor-1 dataset")
    start, stop = timestamp(card["start"]), timestamp(card["stop"])
    train_stop = timestamp(card["split"]["train_stop"])
    validation_stop = timestamp(card["split"]["validation_stop"])
    if not start < train_stop < validation_stop < stop or card["split"].get("shuffle") is not False:
        raise ValueError("Expected chronological, non-shuffled dataset boundaries")
    if hashes["device_parquet_sha256"] != card["tables"]["device"]["sha256"]:
        raise ValueError("Source device Parquet hash mismatch")
    parquet_rows = pq.ParquetFile(source_paths["device_parquet_sha256"]).metadata.num_rows
    if parquet_rows != card["tables"]["device"]["rows"]:
        raise ValueError("Source device Parquet row count mismatch")
    samples = load_devices(input_dir, fields=CLIMATE_FIELDS)
    weather = (
        [
            WeatherObservation.model_validate(item)
            for item in read_report(weather_observations)["observations"]
        ]
        if weather_observations is not None
        else []
    )
    forecasts: list[dict[str, Any]] = []
    forecast_advice: list[Advice] = []
    forecast_metadata: dict[str, Any] | None = None
    if forecast_candidate is not None:
        from edgeops_ai.training.advisor_climate_forecast import forecast_findings

        forecast_advice, forecasts, forecast_metadata = forecast_findings(
            input_dir, forecast_candidate
        )
    findings = climate_findings(
        samples,
        start,
        stop,
        (train_stop, validation_stop),
        humidity_minutes=humidity_minutes,
        forecast_advice=forecast_advice,
        outdoor_weather=weather,
    )
    rows = [
        {
            **asdict(item),
            "start": item.start.isoformat(),
            "stop": item.stop.isoformat(),
            "observed_at": item.observed_at.isoformat(),
            "room": item.entity_id,
            "split": split_at(item.observed_at - MINUTE, train_stop, validation_stop),
        }
        for item in findings
    ]
    report = {
        "schema_version": "advisor-climate-1",
        "start": start.isoformat(),
        "stop": stop.isoformat(),
        "sources": hashes,
        "device_minutes": len(samples),
        "rooms": sorted({sample.room for sample in samples if sample.room}),
        "warmed_up_minutes": sum(sample.values.get("warmed_up") == "true" for sample in samples),
        "thresholds": {
            "humidity_percent": 60,
            "humidity_minutes": humidity_minutes,
            "humidity_recovery_percent": 55,
            "iaq_max_score": 59.8,
            "iaq_minutes": 15,
            "iaq_recovery_score": 69.8,
            "room_cooldown_minutes": 60,
        },
        "advice_counts": dict(Counter(item.code for item in findings)),
        "forecast_available": bool(forecasts),
        "outdoor_weather_available": any(
            "weather_observed_at" in item.evidence for item in findings
        ),
    }
    if forecast_metadata is not None:
        report["forecast"] = forecast_metadata
    if any(file_hash(path) != hashes[name] for name, path in source_paths.items()):
        raise ValueError("Source files changed during replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".advisor-climate-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        write_report(report, staging / "report.json")
        write_report({"findings": rows}, staging / "advice.json")
        if forecast_metadata is not None:
            write_report({"forecasts": forecasts}, staging / "forecasts.json")
        staging.rename(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--humidity-minutes", type=int, default=120)
    parser.add_argument("--forecast-candidate", type=Path)
    parser.add_argument("--weather-observations", type=Path)
    args = parser.parse_args(argv)
    try:
        report = replay(
            args.input_dir,
            args.output_dir,
            humidity_minutes=args.humidity_minutes,
            forecast_candidate=args.forecast_candidate,
            weather_observations=args.weather_observations,
        )
        print(f"Report: {args.output_dir}; advice: {sum(report['advice_counts'].values())}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
        print(f"Advisor climate replay failed: {error}")
        return 1
