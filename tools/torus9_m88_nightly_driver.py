#!/usr/bin/env python3
"""Experiment-only Torus9 generation driver for the M88 nightly A/B campaign."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.torus9_m88_nightly_support import install_nightly_profile_validation

install_nightly_profile_validation()

from tools import torus9_staged_sims_driver as _base


def main(argv: Sequence[str] | None = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
