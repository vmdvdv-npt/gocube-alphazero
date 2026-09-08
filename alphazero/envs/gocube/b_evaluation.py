"""B4 frozen evaluation and paired statistics.

This module is deliberately B-specific.  The generic Arena remains useful for
diagnostics, but a B experiment has a different statistical unit: one frozen
starting position evaluated twice, with the model colors swapped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .game import Cube4JapaneseGame
from .katago_v3 import (
    KATAGO_REFERENCE_COMMIT,
    KATAGO_RULES_IMPLEMENTATION_VERSION,
    KATAGO_RULES_VERSION,
)
from .diversified_game import diversified_pinned_game_class


B_HELDOUT_SUITE_ID = "gocube-b-heldout-suite-v1"
B_HELDOUT_SUITE_SCHEMA_VERSION = 1
B_HELDOUT_GENERATOR_ID = "gocube-b-heldout-generator-v1"
B_HELDOUT_GENERATOR_MASTER_SEED = 20260908
B_HELDOUT_SUITE_POSITION_COUNT = 16
B_HELDOUT_DEPTH_SCHEDULE = tuple(
    depth for depth in (12, 24, 36, 48) for _ in range(4)
)
B_HELDOUT_POSITION_IDS = tuple(
    f"p{index:02d}-d{depth}"
    for index, depth in enumerate(B_HELDOUT_DEPTH_SCHEDULE)
)
B_STATISTICAL_METHOD_IDENTIFIER = (
    "hierarchical-paired-bootstrap-seeds-to-starting-position-pairs-v1"
)
B_BOOTSTRAP_REPLICATES = 10_000
B_BOOTSTRAP_SEED = 20260908
B_GAMES_PER_POSITION = 2
B_REGISTERED_EVALUATION_CLOCK = "cumulative_new_samples"
B_REGISTERED_EVALUATION_MILESTONES = (
    10_000_000,
    20_000_000,
    30_000_000,
    40_000_000,
)
B_FINAL_EVALUATION_CLOCK = B_REGISTERED_EVALUATION_CLOCK
B_FINAL_EVALUATION_MILESTONE = B_REGISTERED_EVALUATION_MILESTONES[-1]
B_EXTENSION_CRITERION_ID = (
    "extend-to-five-seeds-only-if-mandatory-seed-bootstrap-ambiguity-or-variance-v1"
)

# Updated after the canonical artifact is generated.  Keeping the value in a
# source module makes accidental use of an arbitrary held-out file impossible.
B_HELDOUT_SUITE_SHA256 = "0a0d6b9662532aa1bd6fcfad2662651adfc9fc4f6058082a0ad22f86ab319e9d"
B_HELDOUT_SUITE_HASH = B_HELDOUT_SUITE_SHA256


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_suite_path() -> Path:
    return repository_root() / "evaluation" / "gocube-b-heldout-suite-v1.json"


B_HELDOUT_SUITE_PATH = canonical_suite_path()


def require_registered_b_evaluation_target(
    scientific_clock: str,
    scientific_milestone: object,
    *,
    final_only: bool = False,
) -> int:
    """Reject clocks/targets outside the registered B evaluation schedule."""

    if scientific_clock != B_REGISTERED_EVALUATION_CLOCK:
        raise ValueError(
            "B evaluation requires the registered scientific clock "
            f"{B_REGISTERED_EVALUATION_CLOCK!r}"
        )
    if isinstance(scientific_milestone, bool) or not isinstance(scientific_milestone, int):
        raise ValueError("B evaluation milestone must be an integer")
    milestone = scientific_milestone
    if milestone not in B_REGISTERED_EVALUATION_MILESTONES:
        raise ValueError(
            "B evaluation milestone must be one of "
            + ", ".join(str(value) for value in B_REGISTERED_EVALUATION_MILESTONES)
        )
    if final_only and milestone != B_FINAL_EVALUATION_MILESTONE:
        raise ValueError(
            "B extension approval requires the final registered B milestone "
            f"{B_FINAL_EVALUATION_MILESTONE}"
        )
    return milestone


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.asarray(value).reshape(-1).tolist()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot encode {type(value).__name__} in semantic fingerprint")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _board_hex(board: Any) -> str:
    return bytes(np.asarray(board, dtype=np.uint8).reshape(-1).tolist()).hex()


def semantic_state_payload(state: Any) -> dict[str, object]:
    """Return every rule-relevant field of a V3 state in stable form."""

    state = getattr(state, "semantic_state", state)
    fields = (
        "current_player",
        "turns",
        "consecutive_passes",
        "captures",
        "white_bonus_score",
        "phase",
        "ko_recap_blocked",
        "phase_history",
        "history_since_pass",
        "black_pass_states",
        "white_pass_states",
        "ko_capture_history",
        "second_cleanup_start_colors",
        "cleanup2_moves",
        "main_moves",
        "cleanup1_moves",
        "terminal_kind",
        "no_result_reason",
        "termination_reason",
        "result_provenance",
        "pass_alive_early_end",
        "entered_cleanup1",
        "entered_cleanup2",
        "cleanup_captures",
        "ko_unblock_actions",
    )
    payload: dict[str, object] = {
        "board": _board_hex(state.board),
        # Immediate previous board is part of the exact simple-ko state.  It
        # is kept separately from the longer history fields because the
        # authoritative V3 transition checks it directly.
        "previous_board": (
            None if getattr(state, "previous_board", None) is None
            else _board_hex(state.previous_board)
        ),
    }
    for field in fields:
        payload[field] = getattr(state, field, None)
    return payload


def semantic_state_fingerprint(state: Any) -> str:
    return hashlib.sha256(_canonical_json(semantic_state_payload(state))).hexdigest()


def authoritative_suite_game_class():
    """Resolve the exact profile-neutral semantic class used by B evaluation."""

    return diversified_pinned_game_class(Cube4JapaneseGame)


def authoritative_rules_fingerprint() -> str:
    return str(authoritative_suite_game_class().rules_fingerprint())


def authoritative_rules_implementation() -> str:
    return (
        f"gocube-katago-rules-v{KATAGO_RULES_VERSION}"
        f"-implementation-v{KATAGO_RULES_IMPLEMENTATION_VERSION}"
    )


def _read_bytes(path: str | os.PathLike[str]) -> bytes:
    target = Path(path)
    try:
        content = target.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read frozen heldout suite: {target}") from exc
    if not content:
        raise ValueError("Frozen heldout suite must not be empty")
    return content


def _load_json(path: str | os.PathLike[str]) -> dict[str, object]:
    target = Path(path)
    try:
        payload = json.loads(_read_bytes(target).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Frozen heldout suite is not valid UTF-8 JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Frozen heldout suite must be a JSON object")
    return payload


def _position_actions(position: Mapping[str, object]) -> list[int]:
    actions = position.get("actions")
    if not isinstance(actions, list):
        # ``timeline`` is accepted as a compatibility alias only when it is a
        # plain integer action list.  The committed generator writes actions.
        actions = position.get("timeline")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Heldout position must contain a non-empty actions list")
    normalized: list[int] = []
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("Heldout actions must be integer canonical action IDs")
        normalized.append(int(action))
    return normalized


def replay_actions(actions: Sequence[int]):
    """Replay a suite timeline using the authoritative production game API."""

    game_cls = authoritative_suite_game_class()
    state = game_cls()
    topology = game_cls.logical_topology()
    for ply, action in enumerate(actions, start=1):
        if state.win_state().any():
            raise ValueError(f"Heldout timeline reaches terminal before ply {ply}")
        action = int(action)
        if action == int(topology.pass_action):
            raise ValueError("Frozen suite timelines may contain point moves only, not Pass")
        valid = state.valid_moves()
        if action < 0 or action >= len(valid) or not bool(valid[action]):
            raise ValueError(f"Heldout timeline contains illegal action {action} at ply {ply}")
        state.play_action(action)
    if state.win_state().any():
        raise ValueError("Heldout starting position must be nonterminal")
    return state


def validate_frozen_suite(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = B_HELDOUT_SUITE_SHA256,
) -> tuple[dict[str, object], list[tuple[dict[str, object], object]]]:
    """Validate bytes, schema, identity, and every replayed semantic state."""

    raw = _read_bytes(path)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(
            "Frozen heldout suite SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha}"
        )
    payload = _load_json(path)
    expected_depths = list(B_HELDOUT_DEPTH_SCHEDULE)
    required = (
        "schema_version", "suite_id", "generator_id", "generator_master_seed",
        "topology", "size", "rules_implementation", "rules_fingerprint", "komi",
        "position_count", "depth_schedule", "positions",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("Frozen heldout suite is missing fields: " + ", ".join(missing))
    checks = {
        "schema_version": B_HELDOUT_SUITE_SCHEMA_VERSION,
        "suite_id": B_HELDOUT_SUITE_ID,
        "generator_id": B_HELDOUT_GENERATOR_ID,
        "generator_master_seed": B_HELDOUT_GENERATOR_MASTER_SEED,
        "topology": "cube",
        "size": 4,
        "position_count": B_HELDOUT_SUITE_POSITION_COUNT,
        "depth_schedule": expected_depths,
        "rules_implementation": authoritative_rules_implementation(),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Frozen heldout suite field {key} mismatch: expected {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    try:
        if not math.isclose(float(payload["komi"]), 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Frozen heldout suite komi must be exactly 0.5")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Frozen"):
            raise
        raise ValueError("Frozen heldout suite komi must be numeric 0.5") from exc
    expected_rules = authoritative_rules_fingerprint()
    if payload.get("rules_fingerprint") != expected_rules:
        raise ValueError(
            "Frozen heldout suite rules fingerprint mismatch: "
            f"expected {expected_rules}, got {payload.get('rules_fingerprint')!r}"
        )
    positions = payload.get("positions")
    if not isinstance(positions, list) or len(positions) != B_HELDOUT_SUITE_POSITION_COUNT:
        raise ValueError("Frozen heldout suite must contain exactly 16 positions")

    result: list[tuple[dict[str, object], object]] = []
    seen: set[str] = set()
    for index, raw_position in enumerate(positions):
        if not isinstance(raw_position, dict):
            raise ValueError(f"Heldout position {index} must be an object")
        position = dict(raw_position)
        expected_id = B_HELDOUT_POSITION_IDS[index]
        if position.get("position_id") != expected_id:
            raise ValueError(
                f"Heldout position {index} has wrong position_id: "
                f"expected {expected_id!r}, got {position.get('position_id')!r}"
            )
        if position["position_id"] in seen:
            raise ValueError(f"Duplicate heldout position_id: {position['position_id']}")
        seen.add(str(position["position_id"]))
        actions = _position_actions(position)
        target_ply = int(position.get("target_ply", -1))
        if target_ply != expected_depths[index] or len(actions) != target_ply:
            raise ValueError(f"Heldout position {expected_id} has incorrect target ply")
        state = replay_actions(actions)
        expected_fingerprint = semantic_state_fingerprint(state)
        persisted = position.get("semantic_state_fingerprint", position.get("state_fingerprint"))
        if persisted != expected_fingerprint:
            raise ValueError(
                f"Heldout position {expected_id} semantic-state fingerprint mismatch"
            )
        topology = state.logical_topology()
        derived = {
            "side_to_move": "black" if state.player == 0 else "white",
            "player_to_move": int(state.player),
            "phase": str(state.semantic_state.phase),
            "move_count": int(state.turns),
        }
        for key, expected in derived.items():
            if key in position and position[key] != expected:
                raise ValueError(f"Heldout position {expected_id} {key} does not match replay")
        point_ids = position.get("action_point_ids")
        if point_ids is not None:
            if point_ids != [topology.point_id(action) for action in actions]:
                raise ValueError(f"Heldout position {expected_id} action_point_ids mismatch")
        result.append((position, state))
    return payload, result


def suite_payload_from_positions(positions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Build canonical JSON payload from already generated position records."""

    return {
        "schema_version": B_HELDOUT_SUITE_SCHEMA_VERSION,
        "suite_id": B_HELDOUT_SUITE_ID,
        "generator_id": B_HELDOUT_GENERATOR_ID,
        "generator_master_seed": B_HELDOUT_GENERATOR_MASTER_SEED,
        "topology": "cube",
        "size": 4,
        "rules_implementation": authoritative_rules_implementation(),
        "rules_fingerprint": authoritative_rules_fingerprint(),
        "rules_reference_commit": KATAGO_REFERENCE_COMMIT,
        "komi": 0.5,
        "position_count": B_HELDOUT_SUITE_POSITION_COUNT,
        "depth_schedule": list(B_HELDOUT_DEPTH_SCHEDULE),
        "positions": [dict(position) for position in positions],
    }


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def outcome_score_b1(raw_outcome: object) -> float:
    if isinstance(raw_outcome, (int, float)) and not isinstance(raw_outcome, bool):
        numeric = float(raw_outcome)
        if numeric in (0.0, 0.5, 1.0):
            return numeric
    value = str(raw_outcome).strip().lower().replace("-", "_")
    aliases = {
        "b1_win": "win", "b1win": "win", "win": "win",
        "b1_loss": "loss", "b1loss": "loss", "loss": "loss",
        "draw": "draw", "d": "draw",
        "no_result": "no_result", "noresult": "no_result", "nr": "no_result",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(f"Unknown B1 raw outcome: {raw_outcome!r}")
    return {"win": 1.0, "draw": 0.5, "no_result": 0.5, "loss": 0.0}[normalized]


def pair_score_b1(scores_or_games: Sequence[Any]) -> float:
    """Calculate one paired-position score; exactly two games are required."""

    if len(scores_or_games) != B_GAMES_PER_POSITION:
        raise ValueError("A paired starting position must contain exactly two games")
    values = []
    for item in scores_or_games:
        if isinstance(item, Mapping):
            if "b1_game_score" in item:
                value = float(item["b1_game_score"])
                if value not in (0.0, 0.5, 1.0):
                    raise ValueError("Invalid persisted b1_game_score")
            else:
                value = outcome_score_b1(item.get("raw_outcome"))
        else:
            value = outcome_score_b1(item)
        values.append(value)
    return float(sum(values) / B_GAMES_PER_POSITION)


def validate_pairing_invariants(
    games: Sequence[Mapping[str, object]],
    *,
    expected_position_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate balanced color swaps and return canonical position grouping."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for game in games:
        if not isinstance(game, Mapping):
            raise ValueError("Every B evaluation game must be an object")
        position_id = str(game.get("position_id", ""))
        if not position_id:
            raise ValueError("B evaluation game is missing position_id")
        grouped.setdefault(position_id, []).append(game)
    if expected_position_ids is not None and set(grouped) != set(expected_position_ids):
        raise ValueError("B evaluation positions do not match the frozen suite")
    result: dict[str, dict[str, object]] = {}
    for position_id, records in grouped.items():
        if len(records) != B_GAMES_PER_POSITION:
            raise ValueError(f"Position {position_id} must contain exactly two games")
        indices = [int(record.get("pair_game_index", -1)) for record in records]
        if sorted(indices) != [0, 1]:
            raise ValueError(f"Position {position_id} pair_game_index must be exactly 0 and 1")
        assignments = {
            (str(record.get("b0_color")), str(record.get("b1_color")))
            for record in records
        }
        if assignments != {("black", "white"), ("white", "black")}:
            raise ValueError(f"Position {position_id} must have one balanced color swap")
        fingerprints = {
            record.get("starting_semantic_state_fingerprint", record.get("starting_state_fingerprint"))
            for record in records
        }
        if len(fingerprints) != 1 or None in fingerprints:
            raise ValueError(f"Position {position_id} games do not share one starting state")
        result[position_id] = {
            "position_id": position_id,
            "games": sorted(records, key=lambda record: int(record["pair_game_index"])),
            "pair_score_b1": pair_score_b1(records),
            "starting_semantic_state_fingerprint": next(iter(fingerprints)),
        }
    return result


def hierarchical_paired_bootstrap(
    seed_position_scores: Mapping[int | str, Sequence[float]],
    *,
    replicates: int = B_BOOTSTRAP_REPLICATES,
    seed: int = B_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Resample seeds, then positions within each selected seed."""

    if int(replicates) < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must be between 0 and 1")
    canonical = []
    for key in sorted(seed_position_scores, key=lambda value: int(value)):
        values = np.asarray(list(seed_position_scores[key]), dtype=np.float64)
        if values.ndim != 1 or len(values) < 1:
            raise ValueError("Every seed must contain at least one paired position score")
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 0.25, 0.5, 0.75, 1.0)).all():
            raise ValueError("Paired position scores must be finite quarter-point values")
        canonical.append(values)
    if not canonical:
        raise ValueError("At least one training seed is required for bootstrap")
    position_count = len(canonical[0])
    if any(len(values) != position_count for values in canonical):
        raise ValueError("Every seed must contain the same number of paired positions")
    values = np.stack(canonical, axis=0)
    rng = np.random.default_rng(int(seed))
    seed_indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
    position_indices = rng.integers(
        0,
        position_count,
        size=(int(replicates), len(values), position_count),
    )
    sampled = values[seed_indices[:, :, None], position_indices]
    replicate_scores = sampled.mean(axis=2).mean(axis=1)
    deltas = replicate_scores - 0.5
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.percentile(deltas, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "interval": "percentile",
        "ci95_delta": [float(low), float(high)],
        "ci95_low": float(low),
        "ci95_high": float(high),
        "replicate_delta_min": float(np.min(deltas)),
        "replicate_delta_max": float(np.max(deltas)),
    }


def classify_delta_interval(ci95_delta: Sequence[float]) -> str:
    low, high = float(ci95_delta[0]), float(ci95_delta[1])
    if low > 0.0:
        return "B1_BETTER"
    if high < 0.0:
        return "B0_BETTER"
    return "INCONCLUSIVE"


def extension_seed_decision(
    seed_deltas: Mapping[int | str, float],
    mandatory_bootstrap: Mapping[str, object],
    *,
    experiment_contract_sha256: str,
    scientific_clock: str = B_FINAL_EVALUATION_CLOCK,
    scientific_milestone: int = B_FINAL_EVALUATION_MILESTONE,
) -> dict[str, object]:
    milestone = require_registered_b_evaluation_target(
        scientific_clock,
        scientific_milestone,
        final_only=True,
    )
    mandatory = []
    for index in (0, 1, 2):
        value = seed_deltas.get(index)
        if value is None:
            value = seed_deltas.get(str(index))
        if value is None:
            raise ValueError(f"Missing mandatory seed delta {index}")
        mandatory.append(float(value))
    low, high = (float(value) for value in mandatory_bootstrap["ci95_delta"])
    if not np.isfinite((low, high)).all() or low > high:
        raise ValueError("Mandatory bootstrap interval is invalid")
    ambiguity = low <= 0.0 <= high
    if not np.isfinite(np.asarray(mandatory, dtype=np.float64)).all():
        raise ValueError("Mandatory seed deltas must be finite")
    std = float(np.std(np.asarray(mandatory, dtype=np.float64), ddof=1))
    variance = std >= 0.10
    approved = bool(ambiguity or variance)
    return {
        "schema_version": 1,
        "approved": approved,
        "decision": "extend_to_five" if approved else "stop_at_three",
        "criterion_id": B_EXTENSION_CRITERION_ID,
        "mandatory_seed_count": 3,
        "extension_seed_count": 5,
        "scientific_clock": B_FINAL_EVALUATION_CLOCK,
        "scientific_milestone": milestone,
        "criterion_evidence": {
            "ambiguity_detected": ambiguity,
            "variance_exceeded": variance,
            "mandatory_ci95_low": low,
            "mandatory_ci95_high": high,
            "mandatory_seed_delta_std": std,
        },
        "experiment_contract_sha256": str(experiment_contract_sha256),
    }


def _first_present(mapping: Mapping[str, object], keys: Sequence[str], default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def checkpoint_iteration(path: str | os.PathLike[str]) -> int:
    match = re.search(r"iteration-(\d+)\.pkl$", str(path))
    if not match:
        raise ValueError(f"Checkpoint path does not contain an iteration number: {path}")
    return int(match.group(1))


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter_from_mapping(mapping: Mapping[str, object], clock: str) -> int | None:
    if clock == "cumulative_new_samples":
        keys = (
            "cumulative_new_samples", "cumulative_new_samples_accepted",
            "new_samples_accepted", "cumulative_new_data_samples",
        )
    elif clock == "cumulative_optimizer_examples":
        keys = (
            "cumulative_optimizer_examples", "cumulative_optimizer_examples_seen",
            "optimizer_examples_seen", "total_training_samples", "actual_training_samples",
        )
    else:
        raise ValueError(f"Unsupported scientific clock: {clock}")
    value = _first_present(mapping, keys)
    if value is not None:
        return int(value)
    for nested_key in (
        "training_state", "training", "aggregate_metrics", "cumulative",
        "cumulative_counters", "counters",
    ):
        nested = mapping.get(nested_key)
        if isinstance(nested, Mapping):
            value = _counter_from_mapping(nested, clock)
            if value is not None:
                return value
    return None


def checkpoint_counters(path: str | os.PathLike[str]) -> dict[str, int | None]:
    """Read checkpoint counters without loading network weights."""

    import torch

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be an object: {path}")
    result = {
        "cumulative_new_samples": _counter_from_mapping(payload, "cumulative_new_samples"),
        "cumulative_optimizer_examples": _counter_from_mapping(payload, "cumulative_optimizer_examples"),
    }
    args = payload.get("args")
    if isinstance(args, Mapping):
        for clock in result:
            if result[clock] is None:
                result[clock] = _counter_from_mapping(args, clock)
        # Production checkpoints keep generation counters in the committed
        # iteration manifest next to the data namespace rather than in the
        # network payload.  Read only the manifest for this checkpoint's own
        # iteration; never use a later progress snapshot for an earlier model.
        run_name = _first_present(args, ("run_name", "runName"))
        data_root = _first_present(args, ("data", "data_root"))
        if run_name and data_root:
            iteration = checkpoint_iteration(path)
            roots = [Path(str(data_root))]
            if not roots[0].is_absolute():
                roots.extend((Path.cwd(), repository_root()))
            seen_paths: set[Path] = set()
            for root in roots:
                manifest_path = root / str(run_name) / "records" / f"iteration-{iteration:04d}" / "iteration-manifest.json"
                manifest_path = manifest_path.resolve()
                if manifest_path in seen_paths or not manifest_path.is_file():
                    continue
                seen_paths.add(manifest_path)
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    continue
                if isinstance(manifest, Mapping):
                    for clock in result:
                        if result[clock] is None:
                            result[clock] = _counter_from_mapping(manifest, clock)
                if all(value is not None for value in result.values()):
                    break
    return result


def _checkpoint_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted(path.glob("iteration-*.pkl"), key=checkpoint_iteration)
    if not candidates:
        raise ValueError(f"No iteration checkpoints found under {path}")
    return candidates


def select_checkpoint_at_or_after(
    checkpoint_source: str | os.PathLike[str] | Sequence[Mapping[str, object]],
    milestone: int,
    *,
    clock: str = "cumulative_new_samples",
) -> dict[str, object]:
    """Select the first committed/resumable checkpoint on a cumulative clock.

    The sequence form is useful to orchestration code and tests; path form is
    used by the production evaluator.
    """

    target = int(milestone)
    if target < 0:
        raise ValueError("milestone must be non-negative")
    if isinstance(checkpoint_source, (str, os.PathLike)):
        candidates: list[dict[str, object]] = []
        for path in _checkpoint_candidates(Path(checkpoint_source)):
            counters = checkpoint_counters(path)
            counter = counters.get(clock)
            if counter is None:
                continue
            candidates.append({
                "path": str(path),
                "iteration": checkpoint_iteration(path),
                "checkpoint_sha256": checkpoint_sha256(path),
                **counters,
                "scientific_counter": int(counter),
            })
    else:
        candidates = []
        for item in checkpoint_source:
            if not isinstance(item, Mapping):
                raise ValueError("Checkpoint candidates must be objects")
            counter = _counter_from_mapping(item, clock)
            if counter is None:
                raise ValueError("Checkpoint candidate has no cumulative scientific counter")
            candidates.append({**dict(item), "scientific_counter": int(counter)})
        candidates.sort(key=lambda item: int(item.get("iteration", 0)))
    for selected in sorted(
        candidates,
        key=lambda item: (
            int(item["scientific_counter"]),
            int(item.get("iteration", 0)),
        ),
    ):
        if int(selected["scientific_counter"]) >= target:
            selected["milestone_target"] = target
            selected["overshoot"] = int(selected["scientific_counter"]) - target
            return selected
    raise ValueError(
        f"No committed checkpoint reaches {clock} milestone {target}"
    )


def summarize_game_diagnostics(games: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(games)
    wins_b0 = sum(str(g.get("winner")) == "B0" for g in games)
    wins_b1 = sum(str(g.get("winner")) == "B1" for g in games)
    draws = sum(bool(g.get("draw")) for g in games)
    no_results = sum(bool(g.get("no_result")) for g in games)
    move_limits = sum(bool(g.get("move_limit")) for g in games)
    lengths = [int(g.get("moves_played_from_start", 0)) for g in games]
    black = [g for g in games if str(g.get("b1_color")) == "black"]
    white = [g for g in games if str(g.get("b1_color")) == "white"]

    def color_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "games": len(records),
            "b0_wins": sum(str(g.get("winner")) == "B0" for g in records),
            "b1_wins": sum(str(g.get("winner")) == "B1" for g in records),
            "draws": sum(bool(g.get("draw")) for g in records),
            "no_results": sum(bool(g.get("no_result")) for g in records),
            "b1_score": float(np.mean([float(g["b1_game_score"]) for g in records])) if records else None,
        }
    return {
        "total_games": total,
        "b0_wins": wins_b0,
        "b1_wins": wins_b1,
        "draws": draws,
        "no_results": no_results,
        "no_result_rate": no_results / total if total else 0.0,
        "move_limit_count": move_limits,
        "move_limit_rate": move_limits / total if total else 0.0,
        "average_game_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_game_length": float(np.median(lengths)) if lengths else 0.0,
        "b1_score_when_black": float(np.mean([float(g["b1_game_score"]) for g in black])) if black else None,
        "b1_score_when_white": float(np.mean([float(g["b1_game_score"]) for g in white])) if white else None,
        "raw_color_split": {
            "b1_black": color_summary(black),
            "b1_white": color_summary(white),
        },
    }


# Descriptive aliases keep the statistical contract easy to discover from
# tests and downstream analysis code without introducing a second implementation.
score_b1_outcome = outcome_score_b1
calculate_pair_score = pair_score_b1
hierarchical_paired_bootstrap_v1 = hierarchical_paired_bootstrap
load_frozen_suite = validate_frozen_suite
