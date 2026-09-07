#!/usr/bin/env python3
"""Compare two GoCube checkpoints with the fixed production Arena contract.

Normal batched evaluation uses a coalescing inference server so ready Arena
workers are grouped into larger neural-network calls. Held-out evaluation uses
one frozen suite of self-play prefixes and plays every position twice with
colors swapped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from torch import multiprocessing as mp

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.envs.gocube.diversified_game import diversified_pinned_game_class
from alphazero.envs.gocube.game import game_class
from alphazero.inference_batching import collect_ready_worker_ids
from alphazero.search_contract import SearchOutput
from alphazero.utils import const_temp_scaling, get_iter_file


ARENA_SIMS = 50
EXPECTED_KOMI = 0.5
DEFAULT_SEED = 20260906
HELDOUT_SCHEMA_VERSION = 1


def _checkpoint_path(run_name: str, iteration: int) -> Path:
    return Path("checkpoint") / run_name / get_iter_file(int(iteration))


def _load_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
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


def _resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Arena requested but torch.cuda.is_available() is false")
    return requested


def _load_network(game_cls, path: Path, device: str) -> NNetWrapper:
    return NNetWrapper.from_checkpoint(
        game_cls,
        folder=str(path.parent),
        filename=path.name,
        device=device,
        load_training_state=False,
    )


def _wilson_interval(score: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    score = min(1.0, max(0.0, float(score)))
    denominator = 1.0 + z * z / n
    center = (score + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _summarize_outcomes(outcomes: list[tuple[str, str, float | None]]) -> dict[str, object]:
    wins = sum(result == "win" for _, result, _ in outcomes)
    losses = sum(result == "loss" for _, result, _ in outcomes)
    draws = sum(result == "draw" for _, result, _ in outcomes)
    no_results = sum(result == "no_result" for _, result, _ in outcomes)
    scored = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / scored if scored else 0.0
    ci_low, ci_high = _wilson_interval(win_rate, scored)
    by_color = {}
    for color in ("black", "white"):
        subset = [result for item_color, result, _ in outcomes if item_color == color]
        color_wins = subset.count("win")
        color_losses = subset.count("loss")
        color_draws = subset.count("draw")
        color_no_results = subset.count("no_result")
        by_color[color] = {
            "games": len(subset),
            "wins": color_wins,
            "losses": color_losses,
            "draws": color_draws,
            "no_results": color_no_results,
        }
    margins = [float(margin) for _, _, margin in outcomes if margin is not None and math.isfinite(float(margin))]
    return {
        "games": len(outcomes),
        "scored_games": scored,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "no_results": no_results,
        "win_rate": win_rate,
        "win_rate_ci95": [ci_low, ci_high],
        "by_color": by_color,
        "score_margin_mean_abs": (sum(abs(value) for value in margins) / len(margins)) if margins else None,
    }


def _result_for_model_a(final_state, winstate, player_to_index: list[int]) -> tuple[str, str, float | None]:
    color_for_a = "black" if int(player_to_index[0]) == 0 else "white"
    has_draw_slot = len(winstate) > 2
    if has_draw_slot and bool(winstate[-1]):
        result = "no_result" if getattr(final_state, "terminal_kind", None) == "no_result" else "draw"
    else:
        winner_color = next((idx for idx, won in enumerate(winstate[:2]) if bool(won)), None)
        if winner_color is None:
            result = "no_result"
        else:
            result = "win" if int(player_to_index[winner_color]) == 0 else "loss"
    terminal = getattr(final_state, "terminal_adjudication", None)
    score = getattr(terminal, "score", None) if terminal is not None else None
    margin = getattr(score, "margin", None)
    return color_for_a, result, None if margin is None else float(margin)


def _non_batched_summary(arena, game_cls, games: int, seed: int) -> dict[str, object]:
    outcomes: list[tuple[str, str, float | None]] = []
    lengths: list[int] = []
    for game_index in range(games):
        game_seed = int(seed) + game_index
        np.random.seed(game_seed & 0xFFFFFFFF)
        random.seed(game_seed)
        torch.manual_seed(game_seed & 0x7FFFFFFF)
        order = [0, 1] if game_index % 2 == 0 else [1, 0]
        final_state, winstate = arena.play_game(False, order)
        outcomes.append(_result_for_model_a(final_state, winstate, order))
        lengths.append(int(final_state.turns))
    summary = _summarize_outcomes(outcomes)
    summary["average_game_length"] = sum(lengths) / len(lengths) if lengths else 0.0
    summary["inference_calls"] = None
    summary["mean_inference_batch_rows"] = 1.0
    return summary


def _copy_search_output(
    output: SearchOutput,
    worker_rows: list[tuple[int, int]],
    policy_tensors,
    value_tensors,
    score_tensors,
    ownership_tensors,
) -> None:
    policy = output.policy.detach().cpu()
    value = output.value.detach().cpu()
    score = output.score.detach().cpu() if output.score is not None else None
    ownership = output.ownership.detach().cpu() if output.ownership is not None else None
    expected = sum(rows for _, rows in worker_rows)
    if score is None or ownership is None:
        raise RuntimeError("Checkpoint Arena requires score and ownership search heads")
    if not all(int(tensor.size(0)) == expected for tensor in (policy, value, score, ownership)):
        raise RuntimeError("Checkpoint Arena network returned inconsistent coalesced batch rows")
    offset = 0
    for worker_id, rows in worker_rows:
        end = offset + rows
        policy_tensors[worker_id][:rows].copy_(policy[offset:end])
        value_tensors[worker_id][:rows].copy_(value[offset:end])
        score_tensors[worker_id][:rows].copy_(score[offset:end])
        ownership_tensors[worker_id][:rows].copy_(ownership[offset:end])
        offset = end


def _drain_results(result_queue, agents, outcomes, lengths) -> None:
    while True:
        try:
            final_state, winstate, agent_id = result_queue.get_nowait()
        except Empty:
            return
        mapping = list(agents[int(agent_id)].player_to_index)
        outcomes.append(_result_for_model_a(final_state, winstate, mapping))
        lengths.append(int(final_state.turns))


def _coalesced_batched_summary(
    players,
    game_cls,
    eval_args,
    games: int,
    seed: int,
    wait_ms: float,
) -> dict[str, object]:
    """Run Arena workers with one coalescing inference server per model."""
    np.random.seed(int(seed) & 0xFFFFFFFF)
    random.seed(int(seed))
    torch.manual_seed(int(seed) & 0x7FFFFFFF)

    workers = int(eval_args.workers)
    eval_args.arena_batch_size = 1
    eval_args.gamesPerIteration = int(games)
    point_count = int(game_cls.logical_topology().point_count)

    ready_queue = mp.Queue()
    result_queue = mp.Queue()
    completed = mp.Value("i", 0)
    games_played = mp.Value("i", 0)
    stop_event = mp.Event()
    pause_event = mp.Event()
    batch_ready = [mp.Event() for _ in range(workers)]
    batch_queues = [mp.Queue() for _ in range(workers)]
    policy_tensors = []
    value_tensors = []
    score_tensors = []
    ownership_tensors = []
    agents = []

    for worker_id in range(workers):
        policy = torch.zeros([1, game_cls.action_size()])
        value = torch.zeros([1, game_cls.num_players() + 1])
        score = torch.zeros([1, 1])
        ownership = torch.zeros([1, point_count, 3])
        for tensor in (policy, value, score, ownership):
            tensor.share_memory_()
            if bool(eval_args.cuda):
                tensor.pin_memory()
        policy_tensors.append(policy)
        value_tensors.append(value)
        score_tensors.append(score)
        ownership_tensors.append(ownership)
        input_placeholder = [[] for _ in range(game_cls.num_players())]
        agent = SelfPlayAgent(
            worker_id,
            game_cls,
            ready_queue,
            batch_ready[worker_id],
            input_placeholder,
            policy,
            value,
            batch_queues[worker_id],
            result_queue,
            completed,
            games_played,
            stop_event,
            pause_event,
            eval_args,
            _is_arena=True,
            score_tensor=score,
            ownership_tensor=ownership,
        )
        agent.daemon = True
        agents.append(agent)
        agent.start()

    outcomes: list[tuple[str, str, float | None]] = []
    lengths: list[int] = []
    inference_rows = 0
    inference_calls = 0
    try:
        while completed.value != workers:
            worker_ids = collect_ready_worker_ids(ready_queue, workers, wait_ms)
            if worker_ids:
                data_by_worker = {worker_id: batch_queues[worker_id].get() for worker_id in worker_ids}
                for model_index, player in enumerate(players):
                    chunks = []
                    worker_rows: list[tuple[int, int]] = []
                    for worker_id in worker_ids:
                        batch = data_by_worker[worker_id][model_index]
                        if isinstance(batch, list):
                            continue
                        rows = int(batch.size(0))
                        chunks.append(batch)
                        worker_rows.append((worker_id, rows))
                    if not chunks:
                        continue
                    combined = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
                    output = player.nn.process_for_search(combined)
                    if not isinstance(output, SearchOutput):
                        raise RuntimeError("process_for_search() must return SearchOutput")
                    _copy_search_output(
                        output,
                        worker_rows,
                        policy_tensors,
                        value_tensors,
                        score_tensors,
                        ownership_tensors,
                    )
                    inference_rows += int(combined.size(0))
                    inference_calls += 1
                for worker_id in worker_ids:
                    batch_ready[worker_id].set()
            _drain_results(result_queue, agents, outcomes, lengths)
            dead = sum(not agent.is_alive() for agent in agents)
            if dead > int(completed.value):
                raise RuntimeError("Checkpoint Arena worker exited before reporting completion")
    finally:
        stop_event.set()
        for event in batch_ready:
            event.set()
        for agent in agents:
            agent.join(timeout=10)
            if agent.is_alive():
                agent.terminate()
                agent.join(timeout=2)
        _drain_results(result_queue, agents, outcomes, lengths)
        while len(outcomes) < int(games):
            try:
                final_state, winstate, agent_id = result_queue.get(timeout=0.25)
            except Empty:
                break
            mapping = list(agents[int(agent_id)].player_to_index)
            outcomes.append(_result_for_model_a(final_state, winstate, mapping))
            lengths.append(int(final_state.turns))

    if len(outcomes) != int(games):
        raise RuntimeError(f"coalesced Arena completed {len(outcomes)} games, expected {games}")
    summary = _summarize_outcomes(outcomes)
    summary["average_game_length"] = sum(lengths) / len(lengths) if lengths else 0.0
    summary["inference_calls"] = inference_calls
    summary["mean_inference_batch_rows"] = (
        inference_rows / inference_calls if inference_calls else 0.0
    )
    return summary


def _validate_heldout_suite(payload: dict, saved_args: dict) -> list[dict]:
    if int(payload.get("schema_version", -1)) != HELDOUT_SCHEMA_VERSION:
        raise ValueError("Unsupported held-out Arena suite schema")
    if not math.isclose(float(payload.get("komi", float("nan"))), EXPECTED_KOMI, abs_tol=1e-12):
        raise ValueError("Held-out Arena suite violates komi 0.5 contract")
    if payload.get("rules_fingerprint") != saved_args.get("gocube_rules_fingerprint"):
        raise ValueError("Held-out Arena suite rules fingerprint mismatch")
    positions = payload.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("Held-out Arena suite must contain at least one position")
    normalized = []
    for index, item in enumerate(positions):
        if not isinstance(item, dict):
            raise ValueError(f"Held-out position {index} is not an object")
        actions = item.get("prefix_actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"Held-out position {index} has no prefix_actions")
        normalized.append({**item, "prefix_actions": [int(action) for action in actions]})
    return normalized


def _game_from_prefix(game_cls, actions: list[int]):
    state = game_cls()
    for action in actions:
        if state.win_state().any():
            raise ValueError("Held-out prefix reaches terminal state before its end")
        valid = state.valid_moves()
        action = int(action)
        if action < 0 or action >= len(valid) or not bool(valid[action]):
            raise ValueError(f"Held-out prefix contains illegal action {action}")
        state.play_action(action)
    if state.win_state().any():
        raise ValueError("Held-out prefix itself is terminal")
    return state


def _play_from_position(players, game_cls, actions: list[int], order: list[int]):
    state = _game_from_prefix(game_cls, actions)
    for player in players:
        player.reset()
    start_turns = int(state.turns)
    while not state.win_state().any():
        model_index = int(order[state.player])
        action = players[model_index](state)
        for player in players:
            player.update(state, action)
        state.play_action(int(action))
    return state, state.win_state(), int(state.turns) - start_turns


def _heldout_summary(players, game_cls, positions: list[dict], seed: int) -> dict[str, object]:
    outcomes: list[tuple[str, str, float | None]] = []
    lengths: list[int] = []
    for position_index, position in enumerate(positions):
        actions = position["prefix_actions"]
        for swap in (0, 1):
            game_seed = int(seed) + position_index * 2 + swap
            np.random.seed(game_seed & 0xFFFFFFFF)
            random.seed(game_seed)
            torch.manual_seed(game_seed & 0x7FFFFFFF)
            order = [0, 1] if swap == 0 else [1, 0]
            final_state, winstate, continuation_length = _play_from_position(
                players, game_cls, actions, order
            )
            outcomes.append(_result_for_model_a(final_state, winstate, order))
            lengths.append(continuation_length)
    summary = _summarize_outcomes(outcomes)
    summary["average_game_length"] = sum(lengths) / len(lengths) if lengths else 0.0
    summary["inference_calls"] = None
    summary["mean_inference_batch_rows"] = 1.0
    summary["heldout_positions"] = len(positions)
    summary["paired_color_swaps"] = True
    return summary


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return safe or "run"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare arbitrary GoCube checkpoints with fixed 50-sim Arena settings"
    )
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--iteration-a", type=int, required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--iteration-b", type=int, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--arena-inference-batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--heldout-suite", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.iteration_a < 0 or args.iteration_b < 0:
        parser.error("checkpoint iterations must be non-negative")
    if args.games < 1:
        parser.error("--games must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.arena_inference_batch_wait_ms < 0:
        parser.error("--arena-inference-batch-wait-ms must be non-negative")

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

    device = _resolve_device(args.device)
    network_a = _load_network(game_cls, path_a, device)
    network_b = _load_network(game_cls, path_b, device)

    eval_args = saved_a.copy()
    eval_args.cuda = device == "cuda"
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
    eval_args.arena_batch_size = 1
    eval_args.temp_scaling_fn = const_temp_scaling
    eval_args.use_draws_for_winrate = True
    eval_args.arena_inference_batch_wait_ms = float(args.arena_inference_batch_wait_ms)

    players = [
        MCTSPlayer(network_a, game_cls=game_cls, args=eval_args),
        MCTSPlayer(network_b, game_cls=game_cls, args=eval_args),
    ]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.heldout_suite:
        suite_payload = json.loads(Path(args.heldout_suite).read_text(encoding="utf-8"))
        positions = _validate_heldout_suite(suite_payload, saved_a)
        summary = _heldout_summary(players, game_cls, positions, int(args.seed))
        evaluation_mode = "heldout-paired"
    elif args.batched:
        summary = _coalesced_batched_summary(
            players,
            game_cls,
            eval_args,
            int(args.games),
            int(args.seed),
            float(args.arena_inference_batch_wait_ms),
        )
        evaluation_mode = "batched-coalesced"
    else:
        arena = Arena(players, game_cls, use_batched_mcts=False, args=eval_args)
        summary = _non_batched_summary(arena, game_cls, int(args.games), int(args.seed))
        evaluation_mode = "sequential"
    elapsed = time.perf_counter() - started
    actual_games = int(summary["games"])

    output = {
        "schema_version": 2,
        "run_a": args.run_a,
        "iteration_a": int(args.iteration_a),
        "run_b": args.run_b,
        "iteration_b": int(args.iteration_b),
        "wins": int(summary["wins"]),
        "losses": int(summary["losses"]),
        "draws": int(summary["draws"]),
        "no_results": int(summary["no_results"]),
        "win_rate": float(summary["win_rate"]),
        "win_rate_ci95": summary["win_rate_ci95"],
        "by_color": summary["by_color"],
        "average_game_length": summary.get("average_game_length"),
        "score_margin_mean_abs": summary.get("score_margin_mean_abs"),
        "number_of_games": actual_games,
        "requested_games": (
            2 * int(summary.get("heldout_positions", 0))
            if args.heldout_suite
            else int(args.games)
        ),
        "wall_time_seconds": float(elapsed),
        "games_per_second": actual_games / elapsed if elapsed > 0.0 else 0.0,
        "workers": int(args.workers),
        "batched": evaluation_mode == "batched-coalesced",
        "evaluation_mode": evaluation_mode,
        "device": device,
        "arena_inference_batch_wait_ms": float(args.arena_inference_batch_wait_ms),
        "inference_calls": summary.get("inference_calls"),
        "mean_inference_batch_rows": summary.get("mean_inference_batch_rows"),
        "heldout_suite": str(args.heldout_suite) if args.heldout_suite else None,
        "paired_color_swaps": bool(summary.get("paired_color_swaps", False)),
        "cuda_peak_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if device == "cuda"
            else None
        ),
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
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        suffix = "-heldout" if args.heldout_suite else ""
        output_path = output_dir / (
            f"{_safe_name(args.run_a)}-i{int(args.iteration_a):04d}-vs-"
            f"{_safe_name(args.run_b)}-i{int(args.iteration_b):04d}"
            f"-seed{int(args.seed)}{suffix}.json"
        )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)

    print(
        f"Arena A vs B: {output['wins']}W/{output['losses']}L/"
        f"{output['draws']}D/{output['no_results']}NR, "
        f"winrate={output['win_rate']:.3f}, "
        f"CI95=[{output['win_rate_ci95'][0]:.3f},{output['win_rate_ci95'][1]:.3f}], "
        f"{output['games_per_second']:.3f} games/s, "
        f"mode={evaluation_mode}, device={device}"
    )
    print(f"JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
