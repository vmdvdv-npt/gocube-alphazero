"""Minimal generic worker for one supervised production generation."""

from __future__ import annotations

import argparse

from .production_generation import run_generation_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    run_generation_worker(args.request, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
