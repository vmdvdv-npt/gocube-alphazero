#!/usr/bin/env python3
"""Compare two GoCube checkpoints with the fixed production Arena contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.diversified_game import diversified_pinned_game_class
from alphazero.envs.gocube.game import game_class
from alphazero.envs.gocube.production_training import summarize_arena_outcomes
from alphazero.utils import const_temp_scaling, get_iter_file


ARENA_SIMS = 50
EXPECTED_KOMI = 0.5
DEFAULT_SEED = 20260906


def _checkpoint_path(run_name: str, iteration: int) -> Path:
    return Path("checkpoint") / run_name / get_iter_file(int(iteration))


def _load_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "args" not in payload or "state_dict" not in payload:
        raise ValueError(f"Checkpoint does not contain saved args/state_dict: {path}")
    return payload


def _require_same_contract(args_a, args_b) -> None:
    for label, args in (("A", args_a), ("B", args_b)):
        komi = float(args.get("gocube_komi", float("nan")))
        if not math.isclose(komi, EXPECTED_KOMI, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Checkpoint {label} violates GoCube komi 0.5 contract: {komi}")

    for key in (
        "gocube_topology",
        "gocube_size",
        "gocube_rule_set",
        "gocube_rules_fingerprint",
        "gocube_observation_schema",
        "gocube_katago_search_contract",
    ):
        if args_a.get(key) != args_b.get(key):
            raise ValueError(
                f"Checkpoint Arena requires matching {key}: "
                f"A={args_a.get(key)!r}, B={args_b.get(key)!r}"
            )


def _load_network(game_cls, path: Path) -> NNetWrapper:
    return NNetWrapper.from_checkpoint(
        game_cls,
        folder=str(path.parent),
        filename=path.name,
        device="cpu",
        load_training_state=False,
    )


def _non_batched_summary(arena, game_cls, games: int, seed: int) -> dict[str, object]:
    outcomes = []
    for game_index in range(games):
        game_seed = int(seed) + game_index
        np.random.seed(game_seed & 0xFFFFFFFF)
        random.seed(game_seed)
        torch.manual_seed(game_seed & 0x7FFFFFFF)
        order = [0, 1] if game_index % 2 == 0 else [1, 0]
        color = "black" if order[0] == 0 else "white"
        final_state, winstate = arena.play_game(False, order)
        has_draw_slot = len(winstate) > game_cls.num_players()
        if has_draw_slot and bool(winstate[-1]):
            result = (
                "no_result"
                if getattr(final_state, "terminal_kind", None) == "no_result"
                else "draw"
            )
        else:
            winner_color = next(
                (idx for idx, won in enumerate(winstate[:game_cls.num_players()]) if bool(won)),
                None,
            )
            if winner_color is None:
                result = "no_result"
            else:
                result = "win" if order[winner_color] == 0 else "loss"
        outcomes.append((color, result))
    return summarize_arena_outcomes(outcomes)


def _batched_summary(arena, games: int, seed: int) -> dict[str, object]:
    np.random.seed(int(seed) & 0xFFFFFFFF)
    random.seed(int(seed))
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    wins, draws, _ = arena.play_games(games, verbose=False, shuffle_players=True)
    no_results = int(arena.no_results)
    scored = int(sum(wins) + draws)
    return {
        "games": int(scored + no_results),
        "scored_games": scored,
        "wins": int(wins[0]),
        "losses": int(wins[1]),
        "draws": int(draws),
        "no_results": no_results,
        "win_rate": (
            (float(wins[0]) + 0.5 * float(draws)) / scored if scored else 0.0
        ),
        "by_color": arena.player_color_results(0),
    }


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return safe or "run"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare arbitrary GoCube run/checkpoint pairs with fixed 50-sim Arena settings"
    )
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--iteration-a", type=int, required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--iteration-b", type=int, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    if args.iteration_a < 0 or args.iteration_b < 0:
        parser.error("checkpoint iterations must be non-negative")
    if args.games < 1:
        parser.error("--games must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    path_a = _checkpoint_path(args.run_a, args.iteration_a)
    path_b = _checkpoint_path(args.run_b, args.iteration_b)
    payload_a = _load_payload(path_a)
    payload_b = _load_payload(path_b)
    saved_a = payload_a["args"]
    saved_b = payload_b["args"]
    _require_same_contract(saved_a, saved_b)

    topology = str(saved_a["gocube_topology"])
    size = int(saved_a["gocube_size"])
    game_cls = diversified_pinned_game_class(game_class(topology, size, "japanese"))
    if game_cls.rules_fingerprint() != saved_a.get("gocube_rules_fingerprint"):
        raise ValueError("Current game implementation does not match checkpoint rules fingerprint")

    network_a = _load_network(game_cls, path_a)
    network_b = _load_network(game_cls, path_b)

    eval_args = saved_a.copy()
    eval_args.cuda = False
    eval_args.workers = int(args.workers)
    eval_args._num_players = game_cls.num_players() + game_cls.has_draw()
    eval_args.numMCTSSims = ARENA_SIMS
    eval_args.arenaMCTSSims = ARENA_SIMS
    eval_args.probFastSim = 0.0
    eval_args.add_root_noise = False
    eval_args.add_root_temp = False
    eval_args.startTemp = 0.0
    eval_args.arenaTemp = 0.0
    eval_args.arenaBatched = bool(args.batched)
    eval_args.arena_batch_size = max(
        1,
        math.ceil(int(args.games) / int(args.workers)),
    )
    eval_args.temp_scaling_fn = const_temp_scaling
    eval_args.use_draws_for_winrate = True

    players = [
        MCTSPlayer(network_a, game_cls=game_cls, args=eval_args),
        MCTSPlayer(network_b, game_cls=game_cls, args=eval_args),
    ]
    arena = Arena(
        players,
        game_cls,
        use_batched_mcts=bool(args.batched),
        args=eval_args,
    )

    started = time.perf_counter()
    if args.batched:
        summary = _batched_summary(arena, int(args.games), int(args.seed))
    else:
        summary = _non_batched_summary(
            arena,
            game_cls,
            int(args.games),
            int(args.seed),
        )
    elapsed = time.perf_counter() - started
    actual_games = int(summary["games"])

    output = {
        "schema_version": 1,
        "run_a": args.run_a,
        "iteration_a": int(args.iteration_a),
        "run_b": args.run_b,
        "iteration_b": int(args.iteration_b),
        "wins": int(summary["wins"]),
        "losses": int(summary["losses"]),
        "draws": int(summary["draws"]),
        "no_results": int(summary["no_results"]),
        "win_rate": float(summary["win_rate"]),
        "by_color": summary["by_color"],
        "number_of_games": actual_games,
        "requested_games": int(args.games),
        "wall_time_seconds": float(elapsed),
        "games_per_second": actual_games / elapsed if elapsed > 0.0 else 0.0,
        "workers": int(args.workers),
        "batched": bool(args.batched),
        "seed": int(args.seed),
        "arena_contract": {
            "search_sims": ARENA_SIMS,
            "fast_search": False,
            "dirichlet_noise": False,
            "root_policy_temperature": False,
            "move_temperature": 0.0,
            "same_search_settings": True,
            "komi": EXPECTED_KOMI,
            "rules_fingerprint": saved_a["gocube_rules_fingerprint"],
        },
    }

    output_dir = Path("arena-results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{_safe_name(args.run_a)}-i{int(args.iteration_a):04d}-vs-"
        f"{_safe_name(args.run_b)}-i{int(args.iteration_b):04d}-seed{int(args.seed)}.json"
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)

    print(
        f"Arena A vs B: {output['wins']}W/{output['losses']}L/"
        f"{output['draws']}D/{output['no_results']}NR, "
        f"winrate={output['win_rate']:.3f}, {output['games_per_second']:.3f} games/s"
    )
    print(f"JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
