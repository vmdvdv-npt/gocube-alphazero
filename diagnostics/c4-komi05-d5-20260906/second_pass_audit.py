#!/usr/bin/env python3
"""Audit the second consecutive PASS in GoCube main phase.

This diagnostic intentionally does not modify training or game rules. For each
recorded game it replays to the position immediately before the second
consecutive PASS in MAIN and asks:

1. Does the network's *score head* prefer a legal non-PASS successor by a
   meaningful margin while preserving at least the same predicted win chance?
2. Does the production MCTS, which consumes only policy/value, still prefer
   PASS at 20 or 50 simulations?

The combination is a direct test for the suspected objective mismatch: score is
learned as an auxiliary target but ignored by MCTS action selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyximport
import torch

pyximport.install(language_level=3)

from alphazero.MCTS import MCTS
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.envs.gocube.katago_v3 import MAIN


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="iteration record directory")
    parser.add_argument("--checkpoint", required=True, help="checkpoint used for these self-play games")
    parser.add_argument("--limit", type=int, default=64, help="maximum double-pass positions to audit")
    parser.add_argument("--sims", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=1.0,
        help="minimum predicted point gain for a non-PASS move to count as score-dominating PASS",
    )
    parser.add_argument(
        "--win-tolerance",
        type=float,
        default=0.005,
        help="allowed predicted win-probability loss when testing score dominance",
    )
    return parser.parse_args()


def _find_second_main_pass(moves: list[dict]) -> int | None:
    for i in range(1, len(moves)):
        prev = moves[i - 1]
        cur = moves[i]
        if (
            prev.get("phase") == MAIN
            and cur.get("phase") == MAIN
            and str(prev.get("move", "")).upper() == "PASS"
            and str(cur.get("move", "")).upper() == "PASS"
        ):
            return i
    return None


def _replay_before(moves: list[dict], stop_index: int) -> Cube4JapaneseGame:
    game = Cube4JapaneseGame()
    for move in moves[:stop_index]:
        game.play_action(int(move["action"]))
    return game


def _full_network(model: NNetWrapper, observations: np.ndarray):
    device = next(model.nnet.parameters()).device
    tensor = torch.from_numpy(observations.astype(np.float32)).to(device)
    model.nnet.eval()
    with torch.no_grad():
        outputs = model.nnet(tensor)
    if len(outputs) < 4:
        raise RuntimeError("Checkpoint has no ownership/score auxiliary heads")
    policy = torch.exp(outputs[0]).cpu().numpy()
    value = torch.exp(outputs[1]).cpu().numpy()
    score = outputs[3].reshape(-1).cpu().numpy()
    return policy, value, score


def _action_name(action: int) -> str:
    if action == Cube4JapaneseGame.pass_action():
        return "PASS"
    return str(Cube4JapaneseGame.point_id_for_action(action))


def _signed_final_score(record: dict) -> float | None:
    score = record.get("final_score")
    if not isinstance(score, dict):
        return None
    try:
        return float(score["black"]) - float(score["white"])
    except (KeyError, TypeError, ValueError):
        return None


def _mcts_snapshot(game: Cube4JapaneseGame, model: NNetWrapper, sims: int) -> dict:
    mcts = MCTS(model.args)
    mcts.search(game, model, int(sims), False, False)
    counts = np.asarray(mcts.counts(game), dtype=np.int64)
    pass_action = game.pass_action()
    best = int(np.argmax(counts))
    total = int(counts.sum())
    return {
        "sims": int(sims),
        "best_action": best,
        "best_move": _action_name(best),
        "pass_visits": int(counts[pass_action]),
        "total_child_visits": total,
        "pass_visit_fraction": float(counts[pass_action] / total) if total else 0.0,
        "pass_is_best": bool(best == pass_action),
    }


def _audit_record(record: dict, model: NNetWrapper, sims_values: list[int], score_threshold: float, win_tolerance: float) -> dict | None:
    moves = record.get("moves") or []
    idx = _find_second_main_pass(moves)
    if idx is None:
        return None

    game = _replay_before(moves, idx)
    state = game.semantic_state
    if state.phase != MAIN or state.consecutive_passes != 1:
        raise RuntimeError(
            f"Replay mismatch for {record.get('game_id')}: phase={state.phase!r}, passes={state.consecutive_passes}"
        )

    recorded_second = moves[idx]
    if int(recorded_second["action"]) != game.pass_action():
        raise RuntimeError(f"Expected recorded second PASS in {record.get('game_id')}")

    player = int(game.player)
    player_name = "black" if player == 0 else "white"
    pass_action = game.pass_action()
    valids = np.asarray(game.valid_moves(), dtype=np.uint8)
    legal_actions = [int(a) for a in np.flatnonzero(valids)]
    nonpass_actions = [a for a in legal_actions if a != pass_action]

    # Raw policy at the exact decision state.
    root_policy, root_value, root_score = _full_network(model, game.observation()[None, ...])
    root_policy = root_policy[0]
    root_value = root_value[0]
    root_score_points = float(root_score[0] * game.logical_topology().point_count)

    successors = []
    observations = []
    for action in legal_actions:
        nxt = game.clone()
        nxt.play_action(action)
        successors.append(nxt)
        observations.append(nxt.observation())
    _, successor_values, successor_scores = _full_network(model, np.stack(observations))

    point_count = game.logical_topology().point_count
    rows = []
    for j, action in enumerate(legal_actions):
        value = successor_values[j]
        win_prob = float(value[player] + 0.5 * value[2])
        signed_score_points = float(successor_scores[j] * point_count)
        score_for_player = signed_score_points if player == 0 else -signed_score_points
        exact_after_action = None
        terminal = successors[j].terminal_adjudication
        if terminal is not None and terminal.score is not None:
            signed_exact = float(terminal.score.black - terminal.score.white)
            exact_after_action = signed_exact if player == 0 else -signed_exact
        rows.append(
            {
                "action": action,
                "move": _action_name(action),
                "root_policy": float(root_policy[action]),
                "win_prob_for_passer": win_prob,
                "pred_score_for_passer": score_for_player,
                "exact_score_for_passer_if_terminal": exact_after_action,
            }
        )

    by_action = {row["action"]: row for row in rows}
    pass_row = by_action[pass_action]
    nonpass_rows = [row for row in rows if row["action"] != pass_action]
    best_score = max(nonpass_rows, key=lambda row: row["pred_score_for_passer"]) if nonpass_rows else None
    best_win = max(nonpass_rows, key=lambda row: row["win_prob_for_passer"]) if nonpass_rows else None

    dominating = []
    for row in nonpass_rows:
        score_gain = row["pred_score_for_passer"] - pass_row["pred_score_for_passer"]
        win_delta = row["win_prob_for_passer"] - pass_row["win_prob_for_passer"]
        if score_gain >= score_threshold and win_delta >= -win_tolerance:
            dominating.append((score_gain, win_delta, row))
    dominating.sort(key=lambda item: (item[0], item[1]), reverse=True)

    final_signed = _signed_final_score(record)
    final_for_passer = None
    if final_signed is not None:
        final_for_passer = final_signed if player == 0 else -final_signed

    return {
        "game_id": record.get("game_id"),
        "iteration": record.get("iteration"),
        "second_pass_move_number": int(recorded_second.get("move_number", idx + 1)),
        "passer": player_name,
        "legal_nonpass_moves": len(nonpass_actions),
        "recorded_final_score_for_passer": final_for_passer,
        "root_pred_score_for_passer": root_score_points if player == 0 else -root_score_points,
        "root_value_win_prob_for_passer": float(root_value[player] + 0.5 * root_value[2]),
        "pass": pass_row,
        "best_nonpass_by_score": best_score,
        "best_nonpass_by_win": best_win,
        "best_pred_score_gain_over_pass": (
            float(best_score["pred_score_for_passer"] - pass_row["pred_score_for_passer"])
            if best_score is not None
            else None
        ),
        "best_pred_win_delta_over_pass": (
            float(best_win["win_prob_for_passer"] - pass_row["win_prob_for_passer"])
            if best_win is not None
            else None
        ),
        "score_dominated_within_win_tolerance": bool(dominating),
        "best_dominating_nonpass": dominating[0][2] if dominating else None,
        "best_dominating_score_gain": float(dominating[0][0]) if dominating else None,
        "best_dominating_win_delta": float(dominating[0][1]) if dominating else None,
        "mcts": [_mcts_snapshot(game, model, sims) for sims in sims_values],
    }


def main() -> int:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    records_dir = Path(args.records)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NNetWrapper.from_checkpoint(
        Cube4JapaneseGame,
        folder=str(checkpoint.parent),
        filename=checkpoint.name,
        device=device,
        load_training_state=False,
    )

    audited = []
    for path in sorted(records_dir.glob("C4-*.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        result = _audit_record(
            record,
            model,
            [int(v) for v in args.sims],
            float(args.score_threshold),
            float(args.win_tolerance),
        )
        if result is None:
            continue
        audited.append(result)
        if len(audited) >= args.limit:
            break

    with output.open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    dominated = [row for row in audited if row["score_dominated_within_win_tolerance"]]
    gains = [row["best_dominating_score_gain"] for row in dominated]
    print(f"device={device}")
    print(f"positions_audited={len(audited)}")
    print(
        "score_dominated_within_win_tolerance="
        f"{len(dominated)}/{len(audited)}"
        if audited
        else "score_dominated_within_win_tolerance=0/0"
    )
    if gains:
        print(f"dominating_score_gain_mean={float(np.mean(gains)):.3f}")
        print(f"dominating_score_gain_max={float(np.max(gains)):.3f}")
    for sims in args.sims:
        rows = [
            next(item for item in row["mcts"] if item["sims"] == int(sims))
            for row in audited
        ]
        pass_best = sum(int(item["pass_is_best"]) for item in rows)
        mean_fraction = float(np.mean([item["pass_visit_fraction"] for item in rows])) if rows else 0.0
        print(f"mcts_{int(sims)}_pass_best={pass_best}/{len(rows)}")
        print(f"mcts_{int(sims)}_mean_pass_visit_fraction={mean_fraction:.4f}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
