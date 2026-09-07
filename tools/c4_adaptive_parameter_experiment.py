#!/usr/bin/env python3
"""Backward-compatible filename for the canonical Cube-4 P1..P5 sweep."""

from tools.c4_overnight_experiment import *  # noqa: F401,F403
from tools.c4_overnight_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
