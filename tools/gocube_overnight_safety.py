from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from alphazero.GenericPlayers import MCTSPlayer
from alphazero.envs.gocube.observation import GoCubeObservationAdapter
from alphazero.utils import const_temp_scaling
from tools import gocube_checkpoint_arena_complete as arena_impl


HELDOUT_SCHEMA_VERSION = 2
HELDOUT_PROVENANCE = "fresh-evaluation-rollout-v1"
SAMPLE_TIME_RE = re.compile(r"Sample Time:\s*([0-9.]+)s")
INFER_BATCH_RE = re.compile(r"Infer Batch:\s*([0-9.]+)")


def extract_last_progress_metrics(line: str) -> tuple[float | None, float | None]:
    """Return the last progress metrics from a CR-updated console line.

    The progress bar redraws in place using carriage returns, so one newline-
    terminated string can contain many historical `Sample Time` / `Infer Batch`
    values. The benchmark must use the final redraw, not the first match.
    """

    sample_matches = SAMPLE_TIME_RE.findall(str(line))
    batch_matches = INFER_BATCH_RE.findall(str(line))
    sample_time = float(sample_matches[-1]) if sample_matches else None
    infer_batch = float(batch_matches[-1]) if batch_matches else None
    return sample_time, infer_batch


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_path(run_name: str, iteration: int) -> Path:
    return Path("checkpoint") / str(run_name) / f"iteration-{int(iteration):04d}.pkl"


def _rollout_args(saved_args, game_cls, device: str):
    args = saved_args.copy()
    args.cuda = device == "cuda"
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    args.numMCTSSims = arena_impl.ARENA_SIMS
    args.arenaMCTSSims = arena_impl.ARENA_SIMS
    args.probFastSim = 0.0
    args.add_root_noise = False
    args.add_root_temp = False
    # A non-zero fixed action temperature gives multiple representative fresh
    # trajectories while search itself remains deterministic/noise-free.
    args.startTemp = 1.0
    args.arenaTemp = 1.0
    args.temp_scaling_fn = const_temp_scaling
    return args


def _fresh_rollout_actions(player: MCTSPlayer, game_cls, game_seed: int) -> list[int]:
    np.random.seed(int(game_seed) & 0xFFFFFFFF)
    random.seed(int(game_seed))
    torch.manual_seed(int(game_seed) & 0x7FFFFFFF)
    player.reset()
    player.temp = float(player.args.startTemp)
    state = game_cls()
    actions: list[int] = []
    while not state.win_state().any():
        action = int(player(state))
        valid = state.valid_moves()
        if action < 0 or action >= len(valid) or not bool(valid[action]):
            raise RuntimeError(f"Held-out generator selected illegal action {action}")
        player.update(state, action)
        state.play_action(action)
        actions.append(action)
    return actions


def _generate_rollout_positions(*, run_name: str, iteration: int, positions: int,
                                seed: int, device: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoint_path = _checkpoint_path(run_name, iteration)
    payload = arena_impl._load_payload(checkpoint_path)
    saved_args = payload["args"]
    komi = float(saved_args.get("gocube_komi", float("nan")))
    if not math.isclose(komi, arena_impl.EXPECTED_KOMI, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Held-out source checkpoint violates komi 0.5 contract: {komi}")

    contract, model_game_cls = arena_impl._resolve_checkpoint_contract(saved_args, "held-out source")
    game_cls = arena_impl._authoritative_game_class(contract)

    resolved_device = arena_impl._resolve_device(device)
    network = arena_impl._load_network(model_game_cls, checkpoint_path, resolved_device)
    eval_args = _rollout_args(saved_args, game_cls, resolved_device)
    player = MCTSPlayer(
        network,
        game_cls=game_cls,
        args=eval_args,
        observation_adapter=GoCubeObservationAdapter(model_game_cls),
    )

    selected: list[dict[str, object]] = []
    seen_prefixes: set[tuple[int, ...]] = set()
    fractions = (0.25, 0.40, 0.55, 0.70)
    max_attempts = max(64, int(positions) * 16)
    attempt = 0
    while len(selected) < int(positions) and attempt < max_attempts:
        position_index = len(selected)
        rollout_seed = int(seed) + attempt * 1009 + position_index * 65537
        attempt += 1
        actions = _fresh_rollout_actions(player, game_cls, rollout_seed)
        if len(actions) < 8:
            continue
        fraction = fractions[position_index % len(fractions)]
        prefix_len = min(len(actions) - 2, max(2, int(round(len(actions) * fraction))))
        if prefix_len < 2:
            continue
        prefix_actions = tuple(int(action) for action in actions[:prefix_len])
        if prefix_actions in seen_prefixes:
            continue
        seen_prefixes.add(prefix_actions)
        selected.append({
            "position_id": f"H{position_index + 1:03d}",
            "origin": HELDOUT_PROVENANCE,
            "rollout_seed": rollout_seed,
            "rollout_length": len(actions),
            "prefix_length": prefix_len,
            "prefix_actions": list(prefix_actions),
        })

    if len(selected) < int(positions):
        raise RuntimeError(
            f"Could generate only {len(selected)} unique fresh held-out positions, requested {positions}"
        )

    generator = {
        "source_checkpoint": str(checkpoint_path),
        "search_sims": int(arena_impl.ARENA_SIMS),
        "action_temperature": 1.0,
        "root_noise": False,
        "root_policy_temperature": False,
        "device": resolved_device,
    }
    del player
    del network
    if resolved_device == "cuda":
        torch.cuda.empty_cache()
    return selected, generator


def validate_heldout_suite(payload: dict, saved_args: dict) -> list[dict]:
    if int(payload.get("schema_version", -1)) != HELDOUT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported held-out Arena suite schema; expected {HELDOUT_SCHEMA_VERSION} fresh-only schema"
        )
    if payload.get("provenance") != HELDOUT_PROVENANCE:
        raise ValueError("Held-out Arena suite is not a fresh evaluation rollout suite")
    if payload.get("training_data_excluded") is not True:
        raise ValueError("Held-out Arena suite must explicitly exclude all training/replay data")
    if not math.isclose(
        float(payload.get("komi", float("nan"))), arena_impl.EXPECTED_KOMI,
        rel_tol=0.0, abs_tol=1e-12,
    ):
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
        if item.get("origin") != HELDOUT_PROVENANCE:
            raise ValueError(f"Held-out position {index} lacks fresh-rollout provenance")
        actions = item.get("prefix_actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"Held-out position {index} has no prefix_actions")
        normalized.append({**item, "prefix_actions": [int(action) for action in actions]})
    return normalized


def build_fresh_heldout_suite(*, run_name: str, iteration: int, output_path: Path,
                              positions: int, seed: int, device: str = "auto") -> dict[str, object]:
    checkpoint_path = _checkpoint_path(run_name, iteration)
    payload = arena_impl._load_payload(checkpoint_path)
    saved_args = payload["args"]

    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        validate_heldout_suite(existing, saved_args)
        if existing.get("source_run") != run_name or int(existing.get("source_iteration", -1)) != int(iteration):
            raise RuntimeError("Existing held-out suite was built from a different source")
        if len(existing.get("positions", [])) != int(positions):
            raise RuntimeError("Existing held-out suite uses a different position count")
        return existing

    selected, generator = _generate_rollout_positions(
        run_name=run_name,
        iteration=iteration,
        positions=int(positions),
        seed=int(seed),
        device=str(device),
    )
    suite = {
        "schema_version": HELDOUT_SCHEMA_VERSION,
        "provenance": HELDOUT_PROVENANCE,
        "training_data_excluded": True,
        "seed": int(seed),
        "source_run": str(run_name),
        "source_iteration": int(iteration),
        "source_checkpoint": str(checkpoint_path),
        "komi": arena_impl.EXPECTED_KOMI,
        "rules_fingerprint": saved_args["gocube_rules_fingerprint"],
        "generator": generator,
        "positions": selected,
    }
    validate_heldout_suite(suite, saved_args)
    _atomic_json(output_path, suite)
    return suite


def install_fresh_heldout_contract(module):
    """Make checkpoint Arena reject historical training-derived heldout suites."""

    module.HELDOUT_SCHEMA_VERSION = HELDOUT_SCHEMA_VERSION
    module._validate_heldout_suite = validate_heldout_suite
    return module
