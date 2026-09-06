import argparse
import os

import pyximport
import torch

pyximport.install()

from alphazero.envs.gocube.evaluation import (
    load_evaluation_checkpoint,
    play_balanced_batched_match,
    prepare_evaluation_args,
)
from alphazero.envs.gocube.game import game_class, legacy_game_class
from alphazero.envs.gocube.integration.manifest import load_run_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate two GoCube AlphaZero checkpoints without training")
    parser.add_argument("--topology", choices=("torus", "cube"), default="cube")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--candidate", type=int, default=5)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--arena-batch-size", type=int, default=16)
    parser.add_argument("--arena-workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    return parser.parse_args(argv)


def validate_cli(cli):
    if cli.candidate < 0 or cli.baseline < 0:
        raise ValueError("checkpoint iterations must be non-negative")
    if cli.games < 2 or cli.games % 2:
        raise ValueError("games must be a positive even number of at least 2")
    if cli.sims < 1:
        raise ValueError("sims must be at least 1")
    if cli.arena_batch_size < 2:
        raise ValueError("arena-batch-size must be at least 2")
    if cli.arena_workers < 1:
        raise ValueError("arena-workers must be at least 1")


def prepare_arena_args(saved_args, game_cls, sims, arena_batch_size=16, arena_workers=1, cuda=None):
    return prepare_evaluation_args(
        saved_args,
        game_cls,
        sims,
        arena_batch_size=arena_batch_size,
        arena_workers=arena_workers,
        cuda=cuda,
    )


def load_checkpoint(game_cls, folder, iteration, *, device=None):
    return load_evaluation_checkpoint(game_cls, folder, iteration, device=device)


def resolve_game_class(checkpoint_dir, run_name, topology, size):
    folder = os.path.join(checkpoint_dir, run_name)
    try:
        manifest = load_run_manifest(folder)
    except Exception:
        return game_class(topology, size)
    if manifest.topology != topology or manifest.size != size:
        raise ValueError("CLI topology/size does not match run manifest")
    return legacy_game_class(manifest.topology, manifest.size, manifest.terminal_adjudicator)


def resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Arena evaluation but torch.cuda.is_available() is false")
    return value


def main():
    cli = parse_args()
    validate_cli(cli)
    device = resolve_device(cli.device)
    game_cls = resolve_game_class(cli.checkpoint_dir, cli.run_name, cli.topology, cli.size)
    folder = os.path.join(cli.checkpoint_dir, cli.run_name)
    candidate = load_checkpoint(game_cls, folder, cli.candidate, device=device)
    baseline = load_checkpoint(game_cls, folder, cli.baseline, device=device)
    args = prepare_arena_args(
        candidate.args,
        game_cls,
        cli.sims,
        arena_batch_size=cli.arena_batch_size,
        arena_workers=cli.arena_workers,
        cuda=device == "cuda",
    )
    result = play_balanced_batched_match(candidate, baseline, game_cls, args, cli.games)
    wins = result["wins"]
    draws = result["draws"]
    effective = sum(wins) + draws
    candidate_score = (wins[0] + 0.5 * draws) / effective if effective else 0.0
    baseline_score = (wins[1] + 0.5 * draws) / effective if effective else 0.0

    print()
    print(f"candidate iteration {cli.candidate}: {wins[0]} wins ({candidate_score:.3f} score rate)")
    print(f"baseline iteration {cli.baseline}: {wins[1]} wins ({baseline_score:.3f} score rate)")
    print(f"draws: {draws}")
    print(f"no-result: {result['no_results']}")
    print(
        f"batched Arena: {result['arena_batch_size']} games in flight per worker, "
        f"{result['arena_workers']} worker(s), exact 50/50 colors"
    )


if __name__ == "__main__":
    main()
