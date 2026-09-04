from __future__ import annotations

import argparse
import os

from alphazero.envs.gocube.terminal import CONSERVATIVE_AREA_ADJUDICATOR_V1

from .catalog import CheckpointCatalog
from .manifest import RunManifest, write_run_manifest


def register_run(
    *,
    checkpoint_dir: str,
    run_name: str,
    topology: str,
    size: int,
    rule_set: str,
    komi: float,
    terminal_adjudicator: str = CONSERVATIVE_AREA_ADJUDICATOR_V1,
    force: bool = False,
) -> str:
    if os.path.basename(run_name) != run_name or run_name in {"", ".", ".."}:
        raise ValueError("run-name must be a single directory name")
    run_dir = os.path.join(checkpoint_dir, run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not CheckpointCatalog.checkpoint_files(run_dir):
        raise FileNotFoundError(f"No supported iteration checkpoints found in: {run_dir}")

    manifest = RunManifest.create(
        run_name=run_name,
        topology=topology,
        size=size,
        rule_set=rule_set,
        komi=komi,
        terminal_adjudicator=terminal_adjudicator,
    )
    return write_run_manifest(run_dir, manifest, force=force)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Register a legacy GoCube AlphaZero training run")
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--topology", choices=("torus", "cube"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rule-set", default="chinese")
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    path = register_run(
        checkpoint_dir=cli.checkpoint_dir,
        run_name=cli.run_name,
        topology=cli.topology,
        size=cli.size,
        rule_set=cli.rule_set,
        komi=cli.komi,
        force=cli.force,
    )
    print(path)


if __name__ == "__main__":
    main()
