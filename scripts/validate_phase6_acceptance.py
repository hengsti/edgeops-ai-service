"""Compatibility wrapper; prefer edgeops-train."""

from edgeops_ai.training.validate import acceptance_main

if __name__ == "__main__":
    raise SystemExit(acceptance_main())
