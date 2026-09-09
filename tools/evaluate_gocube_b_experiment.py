#!/usr/bin/env python3
"""Evaluate one B0/B1 training seed on the canonical frozen suite.

The command intentionally has no user-settable simulation count.  B4's
production evaluator always uses the registered 50-simulation settings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pyximport

pyximport.install()

if __package__ in (None, ""):
    # Keep the documented ``python tools/<entrypoint>.py`` invocation usable
    # without requiring callers to export PYTHONPATH first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from alphazero.GenericPlayers import MCTSPlayer
from alphazero.envs.gocube.b_evaluation import (
    B_GAMES_PER_POSITION,
    B_HELDOUT_SUITE_ID,
    B_HELDOUT_SUITE_SHA256,
    B_HELDOUT_SUITE_POSITION_COUNT,
    B_STATISTICAL_METHOD_IDENTIFIER,
    authoritative_suite_game_class,
    replay_actions,
    require_registered_b_evaluation_target,
    select_checkpoint_at_or_after,
    semantic_state_fingerprint,
    summarize_game_diagnostics,
    validate_frozen_suite,
    validate_pairing_invariants,
)
from alphazero.envs.gocube.b_experiment_contract import (
    BExperimentContract,
    B_EXTENSION_SEEDS,
    B_EXPERIMENT_CONTRACT_ID,
    B_EXPERIMENT_CONTRACT_VERSION,
    B_SEED_LIST,
    b_evaluation_schedule,
    load_b_experiment_contract,
    resolve_b_experiment_contract,
    validate_extension_seed_decision,
)
from alphazero.envs.gocube.evaluation import prepare_evaluation_args
from alphazero.envs.gocube.katago_v3 import (
    EPISODE_MOVE_LIMIT,
    NO_RESULT,
)
from alphazero.envs.gocube.observation import GoCubeObservationAdapter
from alphazero.envs.gocube.production_contract import GOCUBE_KOMI, require_gocube_komi
from tools.gocube_checkpoint_arena_complete import (
    ARENA_SIMS,
    _authoritative_game_class,
    _load_network,
    _require_compatible_contracts,
    _resolve_checkpoint_contract,
)


def _mapping_value(value: object, key: str, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _require_b_checkpoint(payload: Mapping[str, object], label: str) -> tuple[object, str]:
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise ValueError(f"{label} checkpoint is missing saved args")
    if _mapping_value(args, "gocube_experiment_contract_id") != B_EXPERIMENT_CONTRACT_ID:
        raise ValueError(f"{label} checkpoint is not marked with the B experiment contract")
    if int(_mapping_value(args, "gocube_experiment_contract_version", -1)) != B_EXPERIMENT_CONTRACT_VERSION:
        raise ValueError(f"{label} checkpoint has an unsupported B experiment contract version")
    contract_sha = str(_mapping_value(args, "gocube_experiment_contract_sha256", ""))
    if len(contract_sha) != 64 or any(char not in "0123456789abcdef" for char in contract_sha.lower()):
        raise ValueError(f"{label} checkpoint is missing a valid B experiment contract SHA-256")
    return args, contract_sha.lower()


def _load_checkpoint_payload(path: Path) -> dict[str, object]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "args" not in payload or "state_dict" not in payload:
        raise ValueError(f"Checkpoint does not contain args/state_dict: {path}")
    return payload


def _resolve_checkpoint_source(source: str, milestone: int, clock: str) -> tuple[Path, dict[str, object]]:
    selected = select_checkpoint_at_or_after(source, milestone, clock=clock)
    path = Path(str(selected["path"])).resolve()
    selected["path"] = str(path)
    return path, selected


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)
    random.seed(int(seed))
    torch.manual_seed(int(seed) & 0x7FFFFFFF)


def _raw_outcome(state, winstate, *, order: list[int]) -> tuple[str, str]:
    if len(winstate) > 2 and bool(winstate[2]):
        if getattr(state, "terminal_kind", None) == NO_RESULT:
            return "NO_RESULT", "no_result"
        return "draw", "draw"
    winner_index = next(
        (index for index, won in enumerate(winstate[:2]) if bool(won)), None
    )
    if winner_index is None:
        return "NO_RESULT", "no_result"
    winner_model = int(order[winner_index])
    return ("B1", "win") if winner_model == 1 else ("B0", "loss")


def play_evaluation_game(
    players,
    game_cls,
    position: Mapping[str, object],
    *,
    pair_game_index: int,
    training_seed: int,
    evaluation_seed: int,
    sample_milestone: int = 0,
    move_limit: int | None = None,
) -> dict[str, object]:
    """Play one game from a replayed suite position."""

    if pair_game_index not in (0, 1):
        raise ValueError("pair_game_index must be 0 or 1")
    actions = position.get("actions")
    if not isinstance(actions, list):
        raise ValueError("Evaluation position is missing actions")
    _seed_everything(evaluation_seed)
    state = replay_actions([int(action) for action in actions])
    if state.__class__ is not game_cls:
        raise ValueError("Suite replay did not use the evaluator's authoritative semantic class")
    starting_fingerprint = semantic_state_fingerprint(state)
    expected_fingerprint = position.get(
        "semantic_state_fingerprint", position.get("state_fingerprint")
    )
    if starting_fingerprint != expected_fingerprint:
        raise ValueError(f"Position {position.get('position_id')} starting-state fingerprint mismatch")
    starting_player = "black" if state.player == 0 else "white"
    order = [0, 1] if pair_game_index == 0 else [1, 0]
    b0_color = "black" if order.index(0) == 0 else "white"
    b1_color = "black" if order.index(1) == 0 else "white"
    for player in players:
        player.reset()
    cap = int(move_limit if move_limit is not None else state.episode_move_limit)
    if cap < int(state.episode_move_count):
        raise ValueError("Evaluation move limit is below the frozen starting position depth")
    start_turns = int(state.turns)
    started = time.perf_counter()
    while not state.win_state().any():
        model_index = int(order[state.player])
        action = int(players[model_index](state))
        valid = state.valid_moves()
        if action < 0 or action >= len(valid) or not bool(valid[action]):
            raise RuntimeError(f"B evaluator selected illegal action {action}")
        for player in players:
            player.update(state, action)
        state.play_action(action)
        if not state.win_state().any() and int(state.episode_move_count) >= cap:
            if not state.finalize_episode_due_to_runtime_limit(cap):
                raise RuntimeError("B evaluator reached move limit but rule state did not finalize")
    winner, raw_outcome = _raw_outcome(state, state.win_state(), order=order)
    termination_reason = getattr(state, "termination_reason", None)
    result_provenance = getattr(state, "result_provenance", None)
    move_limit_hit = bool(
        termination_reason == EPISODE_MOVE_LIMIT
        or result_provenance == "runtime"
    )
    record = {
        "training_seed": int(training_seed),
        "sample_milestone": int(sample_milestone),
        "evaluation_seed": int(evaluation_seed),
        "position_id": str(position["position_id"]),
        "pair_game_index": int(pair_game_index),
        "b0_color": b0_color,
        "b1_color": b1_color,
        "starting_player": starting_player,
        "starting_semantic_state_fingerprint": starting_fingerprint,
        "starting_state_fingerprint": starting_fingerprint,
        "winner": winner,
        "raw_outcome": raw_outcome,
        "b1_game_score": {
            "win": 1.0,
            "draw": 0.5,
            "no_result": 0.5,
            "loss": 0.0,
        }[raw_outcome],
        "termination_reason": termination_reason,
        "no_result": raw_outcome == "no_result",
        "draw": raw_outcome == "draw",
        "move_limit": move_limit_hit,
        "moves_played_from_start": int(state.turns) - start_turns,
        "final_total_move_number": int(state.turns),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return record


def evaluate_seed(
    *,
    b0_checkpoint: Path,
    b1_checkpoint: Path,
    suite_path: Path,
    training_seed: int,
    sample_milestone: int,
    sample_clock: str | None = None,
    device: str = "cpu",
    extension_seed_decision: Path | None = None,
    move_limit: int | None = None,
    experiment_contract: BExperimentContract | Mapping[str, object] | None = None,
) -> dict[str, object]:
    if training_seed not in B_SEED_LIST:
        raise ValueError(f"training seed must be one of {B_SEED_LIST}")
    if experiment_contract is None:
        raise ValueError("B evaluator requires the immutable B experiment contract")
    contract, contract_sha256 = resolve_b_experiment_contract(experiment_contract)
    evaluation_schedule = b_evaluation_schedule(contract)
    registered_clock = str(evaluation_schedule["scientific_clock"])
    effective_clock = registered_clock if sample_clock is None else sample_clock
    try:
        registered_milestone = require_registered_b_evaluation_target(
            effective_clock,
            sample_milestone,
            registered_clock=registered_clock,
            registered_milestones=evaluation_schedule["milestone_targets"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("B evaluator received an unregistered scientific target") from exc
    if training_seed in B_EXTENSION_SEEDS and (
        effective_clock != registered_clock
        or registered_milestone != evaluation_schedule["final_milestone"]
    ):
        raise ValueError(
            "Extension seeds can only be evaluated at the final registered B milestone"
        )
    suite_payload, replayed = validate_frozen_suite(suite_path)
    if suite_payload.get("suite_id") != B_HELDOUT_SUITE_ID:
        raise ValueError("B evaluator received a non-canonical suite")
    if sample_milestone < 0:
        raise ValueError("sample milestone must be non-negative")
    b0_selected = select_checkpoint_at_or_after(b0_checkpoint, registered_milestone, clock=effective_clock)
    b1_selected = select_checkpoint_at_or_after(b1_checkpoint, registered_milestone, clock=effective_clock)
    for label, selected in (("B0", b0_selected), ("B1", b1_selected)):
        if selected.get("cumulative_new_samples") is None or selected.get("cumulative_optimizer_examples") is None:
            raise ValueError(
                f"{label} selected checkpoint does not record both cumulative scientific counters"
            )
    b0_checkpoint = Path(str(b0_selected["path"])).resolve()
    b1_checkpoint = Path(str(b1_selected["path"])).resolve()
    payload_a = _load_checkpoint_payload(b0_checkpoint)
    payload_b = _load_checkpoint_payload(b1_checkpoint)
    saved_a, contract_sha_a = _require_b_checkpoint(payload_a, "B0")
    saved_b, contract_sha_b = _require_b_checkpoint(payload_b, "B1")
    if contract_sha_a != contract_sha_b:
        raise ValueError("B0/B1 checkpoint contract SHA mismatch")
    if contract_sha_a != contract_sha256:
        raise ValueError("B checkpoints do not match the immutable B experiment contract")
    if training_seed in B_EXTENSION_SEEDS:
        if extension_seed_decision is None:
            raise ValueError(
                "seeds 3 and 4 require --extension-seed-decision approved by "
                "the pre-registered ambiguity/variance criterion"
            )
        validate_extension_seed_decision(
            extension_seed_decision,
            contract_sha256=contract_sha_a,
            experiment_contract=contract,
        )
    elif extension_seed_decision is not None:
        raise ValueError("--extension-seed-decision is only valid for extension seeds 3 and 4")
    if _mapping_value(saved_a, "gocube_model_profile") != "baseline":
        raise ValueError("--b0-checkpoint does not contain the B0 baseline profile")
    if _mapping_value(saved_b, "gocube_model_profile") != "g1":
        raise ValueError("--b1-checkpoint does not contain the B1 g1 profile")
    contract_a, model_cls_a = _resolve_checkpoint_contract(saved_a, "B0")
    contract_b, model_cls_b = _resolve_checkpoint_contract(saved_b, "B1")
    _require_compatible_contracts(contract_a, contract_b, saved_a, saved_b)
    if contract_a.topology_kind != "cube" or contract_a.topology_size != 4:
        raise ValueError("B evaluator only supports Cube-4 checkpoints")
    require_gocube_komi(contract_a.komi, context="B evaluator checkpoint")
    # The paths may point at a run directory.  Selection is by the scientific
    # cumulative clock, never by matching iteration numbers across profiles.
    if b0_selected["scientific_counter"] < registered_milestone or b1_selected["scientific_counter"] < registered_milestone:
        raise ValueError("Selected checkpoint does not reach the requested scientific milestone")

    semantic_cls = _authoritative_game_class(contract_a)
    if semantic_cls is not authoritative_suite_game_class():
        raise ValueError("B evaluator suite and checkpoint semantic game classes differ")
    network_a = _load_network(model_cls_a, b0_checkpoint, device)
    network_b = _load_network(model_cls_b, b1_checkpoint, device)
    eval_args = prepare_evaluation_args(saved_a, semantic_cls, sims=ARENA_SIMS)
    eval_args.cuda = str(device) == "cuda"
    eval_args.probFastSim = 0.0
    eval_args.add_root_noise = False
    eval_args.add_root_temp = False
    eval_args.startTemp = 0.0
    eval_args.arenaTemp = 0.0
    eval_args.arenaMCTSSims = ARENA_SIMS
    eval_args.numMCTSSims = ARENA_SIMS
    players = [
        MCTSPlayer(
            network_a,
            game_cls=semantic_cls,
            args=eval_args.copy(),
            observation_adapter=GoCubeObservationAdapter(model_cls_a),
        ),
        MCTSPlayer(
            network_b,
            game_cls=semantic_cls,
            args=eval_args.copy(),
            observation_adapter=GoCubeObservationAdapter(model_cls_b),
        ),
    ]
    games: list[dict[str, object]] = []
    for position_index, (position, _state) in enumerate(replayed):
        for pair_game_index in range(B_GAMES_PER_POSITION):
            games.append(
                play_evaluation_game(
                    players,
                    semantic_cls,
                    position,
                    pair_game_index=pair_game_index,
                    training_seed=training_seed,
                    evaluation_seed=(
                        int(training_seed) * 1_000_003
                        + registered_milestone * 1_009
                        + position_index * 2
                        + pair_game_index
                    ),
                    sample_milestone=registered_milestone,
                    move_limit=move_limit,
                )
            )
    grouped = validate_pairing_invariants(
        games,
        expected_position_ids=[position["position_id"] for position, _state in replayed],
    )
    position_results = []
    for position, _state in replayed:
        item = grouped[str(position["position_id"])]
        position_results.append(
            {
                "position_id": str(position["position_id"]),
                "pair_score_b1": float(item["pair_score_b1"]),
                "games": item["games"],
            }
        )
    pair_scores = [float(item["pair_score_b1"]) for item in position_results]
    seed_score = float(np.mean(pair_scores))
    diagnostics = summarize_game_diagnostics(games)
    return {
        "schema_version": 1,
        "evaluation_id": f"gocube-b-evaluation-seed{training_seed}-m{registered_milestone}",
        "experiment_contract_id": B_EXPERIMENT_CONTRACT_ID,
        "experiment_contract_version": B_EXPERIMENT_CONTRACT_VERSION,
        "experiment_contract_sha256": contract_sha_a,
        "heldout_suite_id": B_HELDOUT_SUITE_ID,
        "heldout_suite_sha256": B_HELDOUT_SUITE_SHA256,
        "heldout_suite_path": str(suite_path.resolve()),
        "topology": "cube",
        "size": 4,
        "komi": GOCUBE_KOMI,
        "rules_fingerprint": contract_a.rules_fingerprint,
        "scientific_clock": effective_clock,
        "scientific_milestone": registered_milestone,
        "training_seed": int(training_seed),
        "b0_checkpoint": {
            "path": str(b0_checkpoint.resolve()),
            "iteration": int(b0_selected.get("iteration", -1)),
            "sha256": str(b0_selected.get("checkpoint_sha256", "")),
            "cumulative_new_samples": b0_selected.get("cumulative_new_samples"),
            "cumulative_optimizer_examples": b0_selected.get("cumulative_optimizer_examples"),
            "milestone_target": registered_milestone,
            "overshoot": int(b0_selected["overshoot"]),
            "profile": "baseline",
            "model_contract": contract_a.to_dict(),
        },
        "b1_checkpoint": {
            "path": str(b1_checkpoint.resolve()),
            "iteration": int(b1_selected.get("iteration", -1)),
            "sha256": str(b1_selected.get("checkpoint_sha256", "")),
            "cumulative_new_samples": b1_selected.get("cumulative_new_samples"),
            "cumulative_optimizer_examples": b1_selected.get("cumulative_optimizer_examples"),
            "milestone_target": registered_milestone,
            "overshoot": int(b1_selected["overshoot"]),
            "profile": "g1",
            "model_contract": contract_b.to_dict(),
        },
        "position_count": B_HELDOUT_SUITE_POSITION_COUNT,
        "games_per_position": B_GAMES_PER_POSITION,
        "games": games,
        "position_results": position_results,
        "seed_score_b1": seed_score,
        "seed_delta": seed_score - 0.5,
        "result_semantics": {"win": 1.0, "draw": 0.5, "no_result": 0.5, "loss": 0.0},
        "statistical_method_identifier": B_STATISTICAL_METHOD_IDENTIFIER,
        "search_contract": {
            "mcts_sims": ARENA_SIMS,
            "fast_simulation_probability": 0.0,
            "root_dirichlet_noise": False,
            "root_policy_temperature": False,
            "move_action_temperature": 0.0,
            "arena_temperature": 0.0,
        },
        "diagnostics": diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", required=True)
    parser.add_argument("--b1-checkpoint", required=True)
    parser.add_argument("--suite", "--heldout-suite", dest="suite", required=True)
    parser.add_argument("--experiment-contract", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--sample-milestone", required=True, type=int)
    parser.add_argument("--sample-clock", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--extension-seed-decision", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.training_seed not in B_SEED_LIST:
        parser.error(f"--training-seed must be one of {B_SEED_LIST}")
    if args.sample_milestone < 0:
        parser.error("--sample-milestone must be non-negative")
    try:
        contract, _contract_sha256 = load_b_experiment_contract(args.experiment_contract)
    except Exception as exc:
        parser.error(str(exc))
    payload = evaluate_seed(
        b0_checkpoint=Path(args.b0_checkpoint),
        b1_checkpoint=Path(args.b1_checkpoint),
        suite_path=Path(args.suite),
        training_seed=args.training_seed,
        sample_milestone=args.sample_milestone,
        sample_clock=args.sample_clock,
        device=args.device,
        extension_seed_decision=(Path(args.extension_seed_decision) if args.extension_seed_decision else None),
        experiment_contract=contract,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(
        f"B evaluation seed {args.training_seed}: paired_score={payload['seed_score_b1']:.6f} "
        f"delta={payload['seed_delta']:+.6f}; JSON: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
