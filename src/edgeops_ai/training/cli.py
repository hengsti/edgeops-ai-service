"""The edgeops-train command-line entry point."""

import argparse
from collections.abc import Sequence
from importlib import import_module

COMMANDS = {
    "advisor-dataset": "advisor_dataset",
    "advisor-labels": "advisor_labels",
    "advisor-health": "advisor_health",
    "advisor-climate": "advisor_climate",
    "advisor-maintenance": "advisor_maintenance",
    "advisor-forecast": "advisor_forecast",
    "advisor-benchmark": "advisor_benchmark",
    "collect": "dataset",
    "train": "train",
    "evaluate": "evaluate",
    "validate": "validate",
    "pipeline": "pipeline",
    "evaluate-deployed": "deployed",
    "analyze-features": "features",
    "ablate": "ablation",
    "export-onnx": "export",
    "check-distributions": "distributions",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect, train, evaluate and validate EdgeOps models."
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Use <command> --help for options.")
    args = parser.parse_args(argv)
    return import_module(f"edgeops_ai.training.{COMMANDS[args.command]}").main(args.args)
