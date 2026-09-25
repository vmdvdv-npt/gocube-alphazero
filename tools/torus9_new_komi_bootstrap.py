#!/usr/bin/env python3
"""Create the fresh-history ``new_komi`` Torus9 bootstrap lineage.

This command is intentionally narrow. It does not select komi and it does not
start self-play/training. The resulting lineage is blocked until a fresh
5-channel komi calibration selects the real rules komi.

The command also enforces the permanent no-komi neural observation policy:
legacy six-channel models may be read only as conversion sources and cannot be
selected as the trainable ``new_komi`` network.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from gocube_golden.torus9_m137_5ch import Torus9M137FiveChannelGraphNet
from gocube_golden.torus9_new_komi import create_new_komi_lineage
from gocube_golden.torus9_new_komi_guard import (
    assert_new_komi_training_checkpoint_metadata,
    assert_new_komi_training_model,
)


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

    # Fail before creating a lineage if the trainable target ever regresses to
    # the historical 6CH/komi-input representation.
    assert_new_komi_training_model(Torus9M137FiveChannelGraphNet())

    repo_root = Path(__file__).resolve().parents[1]
    root, metadata = create_new_komi_lineage(
        source_checkpoint=Path(args.source_checkpoint),
        runs_root=repo_root / args.runs_root,
        converter_git_commit=_git_head(repo_root),
    )
    # Verify the persisted checkpoint contract too. Any future training binding
    # must apply this same guard before the first optimizer step.
    assert_new_komi_training_checkpoint_metadata(metadata)

    print(f"new_komi lineage: {root}")
    print(f"bootstrap checkpoint: {root / 'checkpoints' / 'M137-5CH-bootstrap.pt'}")
    print(f"model hash: {metadata['converted_model_hash']}")
    print("training status: BLOCKED_PENDING_KOMI_CALIBRATION")
    print("observation policy: 5CH only; komi neural channel forbidden")
    print("replay history: fresh-only; no parent replay referenced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
