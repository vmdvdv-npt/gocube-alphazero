#!/usr/bin/env python3
"""Deprecated command alias for the canonical Cube-4 overnight sweep.

Keep this filename only so old shell history does not break. All behavior and
state live in ``tools.c4_overnight_experiment``; no second orchestration path is
implemented here.
"""

from tools.c4_overnight_experiment import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
