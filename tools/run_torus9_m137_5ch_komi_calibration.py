#!/usr/bin/env python3
"""Retired bespoke M137 5CH komi production launcher.

Historical runs created by older checkouts remain valid and are not modified.
New production calibration/evaluation must be expressed as an Orchestrator V2
run-spec and submitted through ``production_entrypoint run``.
"""
from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "This standalone production launcher is disabled. "
        "Use: python -m gocube_golden.orchestrator_v2.production_entrypoint run <run-spec.json>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
