#!/usr/bin/env python3
"""Evaluate two GoCube checkpoints with a fixed, noise-free MCTS Arena."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pyximport
import torch

pyximport.install()

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import game_class


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_interval(score: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Approximate Wilson interval for score rate (draws count as half a point)."""
    if n <= 0:
        return 0.0, 1.0
    denominator = 1.0 + z * z / n
    center = (score + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def load_network(game_cls, checkpoint_path: Path) -> NNetWrapper:
    return NNetWrapper.from_checkpoint(
        game_cls,
        folder=str(checkpoint_path.parent),
        filename=checkpoint_path.name,
        device="cpu",
        load_training_state=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-id", default="candidate")
    parser.add_argument("--reference-id", default="reference")
    parser.add_argument("--topology", default="cube", choices=("cube", "torus"))
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    if args.games < 2 or args.games % 2:
        parser.error("--games must be an even integer >= 2 so colors are balanced")
    if args.sims < 1:
        parser.error("--sims must be >= 1")

    candidate_path = Path(args.candidate).resolve()
    reference_path = Path(args.reference).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    game_cls = game_class(args.topology, args.size, "japanese")
    candidate = load_network(game_cls, candidate_path)
    reference = load_network(game_cls, reference_path)

    eval_args = candidate.args.copy()
    eval_args.cuda = False
    eval_args.numMCTSSims = int(args.sims)
    eval_args.arenaMCTSSims = int(args.sims)
    eval_args.probFastSim = 0.0
    eval_args.add_root_noise = False
    eval_args.add_root_temp = False
    eval_args.startTemp = 0.0
    eval_args.arenaTemp = 0.0
    eval_args.arenaBatched = False
    eval_args.use_draws_for_winrate = True

    players = [
        MCTSPlayer(candidate, game_cls=game_cls, args=eval_args),
        MCTSPlayer(reference, game_cls=game_cls, args=eval_args),
    ]
    arena = Arena(players, game_cls, use_batched_mcts=False, args=eval_args)

    started = time.time()
    wins, draws, _winrates = arena.play_games(args.games, verbose=False, shuffle_players=True)
    elapsed = time.time() - started
    no_results = int(arena.no_results)
    effective_games = int(sum(wins) + draws)
    candidate_points = float(wins[0]) + 0.5 * float(draws)
    reference_points = float(wins[1]) + 0.5 * float(draws)
    candidate_score = candidate_points / effective_games if effective_games else 0.0
    reference_score = reference_points / effective_games if effective_games else 0.0
    ci_low, ci_high = score_interval(candidate_score, effective_games)

    payload = {
        "schema_version": 1,
        "candidate": {
            "id": args.candidate_id,
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "reference": {
            "id": args.reference_id,
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "topology": args.topology,
        "size": args.size,
        "ruleset": "japanese",
        "games_requested": args.games,
        "games_effective": effective_games,
        "candidate_wins": int(wins[0]),
        "reference_wins": int(wins[1]),
        "draws": int(draws),
        "no_results": no_results,
        "candidate_score_rate": candidate_score,
        "reference_score_rate": reference_score,
        "candidate_score_ci95_approx": [ci_low, ci_high],
        "mcts_sims": args.sims,
        "fast_probability": 0.0,
        "root_noise": False,
        "root_temperature": False,
        "action_temperature": 0.0,
        "batched": False,
        "balanced_colors": True,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "seconds_per_game": elapsed / args.games,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)

    print(
        "EVAL RESULT: "
        f"{args.candidate_id} {wins[0]}W / {wins[1]}L / {draws}D / {no_results}NR, "
        f"score={candidate_score:.3f}, ci95~[{ci_low:.3f}, {ci_high:.3f}], "
        f"{elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
