"""Read and write JSON evidence reports."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read JSON report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def write_report(report: Mapping[str, Any], output: Path) -> None:
    if output.suffix.lower() != ".json":
        raise ValueError("Report output must use the .json extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
