"""Compatibility wrapper; prefer edgeops-train."""

from edgeops_ai.training.ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
