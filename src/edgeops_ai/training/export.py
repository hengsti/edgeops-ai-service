"""Export an existing trusted scikit-learn artifact to ONNX without retraining."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from edgeops_ai.ml import MODEL_FEATURE_COLUMNS, ModelArtifactError, load_ml_artifact


def export_onnx(source: Path, output: Path, expected_sha256: str) -> str:
    if source.suffix.lower() != ".joblib" or output.suffix.lower() != ".onnx":
        raise ValueError("Export requires a .joblib source and an .onnx output.")
    artifact = load_ml_artifact(source, expected_sha256)
    model = artifact["model"]
    graph = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, len(MODEL_FEATURE_COLUMNS)]))],
        options={"zipmap": False},
    )
    manifest = {key: value for key, value in artifact.items() if key != "model"}
    manifest["labels"] = [str(label) for label in model.classes_]
    onnx.helper.set_model_props(graph, {"edgeops_ai_manifest": json.dumps(manifest)})
    onnx.checker.check_model(graph)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = graph.SerializeToString()
    output.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sha256", required=True, help="Recorded SHA-256 of the trusted input artifact."
    )
    args = parser.parse_args(argv)
    try:
        digest = export_onnx(args.input, args.output, args.sha256)
    except (ModelArtifactError, OSError, ValueError) as error:
        print(f"ONNX export failed: {error}")
        return 1
    print(f"ONNX artifact: {args.output}")
    print(f"MODEL_ARTIFACT_SHA256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
