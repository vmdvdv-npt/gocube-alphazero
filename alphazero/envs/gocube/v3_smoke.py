from __future__ import annotations

import argparse
import json

import numpy as np

from .game import game_class
from .katago_v3 import NO_RESULT, SCORED


def run_smoke(topology: str, size: int, games: int, seed: int) -> dict[str, object]:
    game_cls = game_class(topology, size, "japanese")
    rng = np.random.default_rng(seed)
    results = []
    for game_index in range(games):
        game = game_cls()
        while not game.win_state().any():
            legal = np.flatnonzero(game.valid_moves())
            if legal.size == 0:
                raise RuntimeError("nonterminal V3 game has zero legal actions")
            pass_action = game.pass_action()
            point_actions = legal[legal != pass_action]
            # Validation policy is intentionally weak/deterministic: it exercises the
            # rule protocol, not model strength. Passes are biased so all three phases
            # receive coverage without a large training run.
            if pass_action in legal and (point_actions.size == 0 or rng.random() < 0.22):
                action = pass_action
            else:
                action = int(rng.choice(point_actions))
            game.play_action(int(action))
        terminal = game.terminal_adjudication
        state = game.semantic_state
        score = None
        if terminal is not None and terminal.score is not None:
            score = {
                "black": terminal.score.black,
                "white": terminal.score.white,
                "margin": terminal.score.margin,
                "winner": terminal.score.winner,
            }
        results.append({
            "game": game_index,
            "terminal_kind": game.terminal_kind,
            "no_result_reason": state.no_result_reason,
            "training_valid": game.has_training_result(),
            "turns": game.turns,
            "score": score,
            "diagnostics": game.diagnostic_counters(),
        })

    def total(key):
        return sum(int(row["diagnostics"][key]) for row in results)

    cleanup1_entered = total("terminal/entered_cleanup1")
    cleanup2_entered = total("terminal/entered_cleanup2")
    cleanup1_moves = total("terminal/cleanup1_moves")
    cleanup2_moves = total("terminal/cleanup2_moves")
    scored_rows = [r for r in results if r["terminal_kind"] == SCORED]
    no_result_rows = [r for r in results if r["terminal_kind"] == NO_RESULT]
    summary = {
        "topology": topology,
        "size": size,
        "games": games,
        "scored": len(scored_rows),
        "no_result": len(no_result_rows),
        "pass_alive_early_end": total("terminal/pass_alive_early_end"),
        "cleanup1_entered": cleanup1_entered,
        "cleanup2_entered": cleanup2_entered,
        "average_cleanup1_moves": cleanup1_moves / cleanup1_entered if cleanup1_entered else 0.0,
        "average_cleanup2_moves": cleanup2_moves / cleanup2_entered if cleanup2_entered else 0.0,
        "cleanup_captures": total("terminal/cleanup_captures"),
        "ko_unblock_actions": total("terminal/ko_unblock_actions"),
        "maximum_game_length": max(r["turns"] for r in results),
        "invalid_training_games": sum(not r["training_valid"] for r in results),
        "exact_score_examples": [r["score"] for r in scored_rows[:5]],
        "no_result_cases": [
            {"game": r["game"], "reason": r["no_result_reason"], "turns": r["turns"]}
            for r in no_result_rows
        ],
        "crashes_exceptions": 0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.games < 32:
        raise ValueError("V3 acceptance smoke requires at least 32 games per topology")
    for topology, size, offset in (("cube", 4, 0), ("torus", 9, 1)):
        print(f"===== {topology.upper()} {size} V3 SMOKE =====")
        print(json.dumps(run_smoke(topology, size, args.games, args.seed + offset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
