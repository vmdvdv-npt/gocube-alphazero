#!/usr/bin/env python3
"""Create the fresh-history ``new_komi`` Torus9 bootstrap lineage.

This command is intentionally narrow.  It does not select komi and it does not
start self-play/training.  The resulting lineage is blocked until a fresh
5-channel komi calibration selects the real rules komi.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from gocube_golden.torus9_new_komi import create_new_komi_lineage


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help="Canonical M137 .pt path; SHA-256 is pinned by the bootstrap code.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Repository runs root; lineage is created at runs/torus9/active/new_komi.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root, metadata = create_new_komi_lineage(
        source_checkpoint=Path(args.source_checkpoint),
        runs_root=repo_root / args.runs_root,
        converter_git_commit=_git_head(repo_root),
    )
    print(f"new_komi lineage: {root}")
    print(f"bootstrap checkpoint: {root / 'checkpoints' / 'M137-5CH-bootstrap.pt'}")
    print(f"model hash: {metadata['converted_model_hash']}")
    print("training status: BLOCKED_PENDING_KOMI_CALIBRATION")
    print("replay history: fresh-only; no parent replay referenced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
