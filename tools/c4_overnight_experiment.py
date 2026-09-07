#!/usr/bin/env python3
"""Canonical entrypoint for the complete autonomous GoCube overnight sweep."""

from tools.c4_overnight_complete import *  # noqa: F401,F403
from tools.c4_overnight_complete import main


if __name__ == "__main__":
    raise SystemExit(main())
