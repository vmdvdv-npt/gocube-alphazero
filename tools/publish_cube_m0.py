#!/usr/bin/env python3
"""Publish a canonical Cube-family M0 after the post-merge preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import torch

from gocube_golden.cube_m0_publisher import publish_cube_m0
from gocube_golden.provenance import capture_code_identity


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _preflight(repo_root: Path, runs_root: Path, lineage_id: str, expected_head: str | None) -> None:
    head = _git(repo_root, "rev-parse", "HEAD")
    origin_main = _git(repo_root, "rev-parse", "origin/main")
    expected = expected_head or origin_main
    if head != expected:
        raise RuntimeError(
            f"production M0 requires clean expected HEAD {expected}; current HEAD is {head}"
        )
    if _git(repo_root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("production M0 requires a clean working tree")
    if not torch.cuda.is_available():
        raise RuntimeError("production M0 preflight requires CUDA")
    usage = shutil.disk_usage(runs_root.parent if runs_root.parent.exists() else runs_root)
    if usage.free < 10 * 1024**3:
        raise RuntimeError(f"insufficient free disk space for Cube4 production: {usage.free} bytes")
    for path in (
        runs_root / "cube4" / "active" / lineage_id,
        runs_root / "cube4" / "archive" / lineage_id,
    ):
        if path.exists():
            raise RuntimeError(f"lineage id is already occupied: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026092301)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-head")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    runs_root = args.runs_root.resolve()
    _preflight(repo_root, runs_root, args.lineage_id, args.expected_head)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    publication = publish_cube_m0(
        size=args.size,
        lineage_id=args.lineage_id,
        effective_config=config,
        seed=args.seed,
        runs_root=runs_root,
        repo_root=repo_root,
        code_identity=capture_code_identity(repo_root),
        require_clean_code=True,
    )
    print(json.dumps(publication.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
