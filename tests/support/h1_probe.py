"""Observation-collision probes for H1, independent of any observation schema."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import fields
from typing import Any, Callable, Iterable, Sequence

import numpy as np


def stable_observation_bytes(observation: Any) -> bytes:
    """Encode an observation without Python's process-randomized ``hash``."""

    if hasattr(observation, "tobytes"):
        shape = tuple(int(dimension) for dimension in getattr(observation, "shape", ()))
        dtype = str(getattr(observation, "dtype", "unknown"))
        payload = observation.tobytes(order="C")
        return json.dumps({"shape": shape, "dtype": dtype}, separators=(",", ":")).encode() + b"\0" + payload
    return json.dumps(observation, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def stable_observation_digest(observation: Any) -> str:
    return hashlib.sha256(stable_observation_bytes(observation)).hexdigest()


@dataclass(frozen=True)
class ReachableSample:
    sample_id: str
    state: Any
    actions: tuple[Any, ...] = ()
    category: str = "ordinary"
    seed: int | None = None


@dataclass(frozen=True)
class ObservationCollision:
    observation_digest: str
    sample_ids: tuple[str, ...]
    categories: tuple[str, ...]
    semantic_signatures: tuple[Any, ...]
    semantic_collision: bool


@dataclass(frozen=True)
class ObservationProbeReport:
    samples_examined: int
    observation_groups: int
    collisions: tuple[ObservationCollision, ...]

    @property
    def semantic_collisions(self) -> tuple[ObservationCollision, ...]:
        return tuple(collision for collision in self.collisions if collision.semantic_collision)


def canonical_state_value(value: Any) -> Any:
    """Make state values comparable without Python's process-randomized hash."""

    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), tuple(value.shape), value.tobytes().hex())
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, (tuple, list)):
        return tuple(canonical_state_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), canonical_state_value(item)) for key, item in value.items()))
    return value


def v3_state_key(state: Any) -> tuple[Any, ...]:
    """Return a complete, deterministic identity for a V3 state."""

    return tuple(
        (field.name, canonical_state_value(getattr(state, field.name)))
        for field in fields(state)
    )


def v3_immediate_semantic_signature(state: Any, topology: Any) -> tuple[Any, ...]:
    """Capture the immediate semantics relevant to H1 collision grouping."""

    from alphazero.envs.gocube.katago_v3 import (
        apply_v3_action,
        is_simple_ko_state,
        v3_valid_moves,
    )

    valid = tuple(int(value) for value in v3_valid_moves(state, topology))
    pass_after = None
    if valid[topology.pass_action]:
        after = apply_v3_action(state, topology.pass_action, topology)
        pass_after = (
            after.phase,
            after.current_player,
            after.terminal_kind,
            after.no_result_reason,
        )
    return (
        "valid_moves",
        valid,
        "phase",
        state.phase,
        "side",
        state.current_player,
        "terminal",
        (state.terminal_kind, state.no_result_reason),
        "pass_after",
        pass_after,
        "score_offset",
        (state.white_bonus_score, state.captures, state.second_cleanup_start_colors),
        "simple_ko",
        bool(is_simple_ko_state(state, topology)),
    )


V3_STATE_AUDIT_FIELDS = {
    "board": {"classification": "A"},
    "current_player": {"classification": "A"},
    "turns": {"classification": "D"},
    "consecutive_passes": {"classification": "A"},
    "captures": {"classification": "A"},
    "white_bonus_score": {"classification": "D"},
    "previous_board": {"classification": "A"},
    "phase": {"classification": "A"},
    "ko_recap_blocked": {"classification": "A"},
    "phase_history": {"classification": "D"},
    "history_since_pass": {"classification": "D"},
    "black_pass_states": {"classification": "D"},
    "white_pass_states": {"classification": "D"},
    "ko_capture_history": {"classification": "D"},
    "second_cleanup_start_colors": {"classification": "A"},
    "cleanup2_moves": {"classification": "A"},
    "main_moves": {"classification": "C"},
    "cleanup1_moves": {"classification": "C"},
    "terminal_kind": {"classification": "D"},
    "no_result_reason": {"classification": "D"},
    "termination_reason": {"classification": "D"},
    "result_provenance": {"classification": "D"},
    "pass_alive_early_end": {"classification": "D"},
    "entered_cleanup1": {"classification": "C"},
    "entered_cleanup2": {"classification": "C"},
    "cleanup_captures": {"classification": "C"},
    "ko_unblock_actions": {"classification": "C"},
}


PINNED_WRAPPER_AUDIT_FIELDS = {
    "_pinned_auto_end_pass_alive": "D",
    "_pinned_root_prune_useless_moves": "D",
    "_pinned_selfplay_semantics": "D",
    "_pinned_seki_fork_hack_prob": "D",
    "_pinned_is_search_clone": "D",
    "_pinned_at_search_root": "D",
    "_pinned_started_from_seki_fork": "D",
    "_pinned_start_phase": "D",
    "_pinned_move_history": "D",
    "_pinned_state_history": "D",
    "_pinned_state_history_offset": "D",
    "_pinned_episode_move_count": "D",
    "_pinned_episode_type": "D",
    "_pinned_episode_limit_override": "D",
}


DIVERSIFIED_WRAPPER_AUDIT_FIELDS = {
    "_diverse_early_fork_prob": "D",
    "_diverse_ordinary_fork_prob": "D",
    "_diverse_early_expected_move_prop": "D",
    "_diverse_plain_fork_pool_capacity": "D",
    "_diverse_started_from_plain_fork": "D",
    "_diverse_suppress_plain_fork_generation": "D",
    "_diverse_train_state_history": "D",
    "_diverse_training_history_offset": "D",
}


def find_observation_collisions(
    samples: Iterable[ReachableSample],
    observation_builder: Callable[[Any], Any],
    semantic_signature: Callable[[Any], Any],
) -> ObservationProbeReport:
    """Group reachable samples by canonical observation and compare semantics."""

    buckets: dict[str, list[tuple[ReachableSample, Any]]] = defaultdict(list)
    sample_count = 0
    for sample in samples:
        sample_count += 1
        observation = observation_builder(sample.state)
        buckets[stable_observation_digest(observation)].append((sample, semantic_signature(sample.state)))
    collisions = []
    for digest, bucket in sorted(buckets.items()):
        if len(bucket) < 2:
            continue
        signatures = tuple(item[1] for item in bucket)
        collisions.append(
            ObservationCollision(
                observation_digest=digest,
                sample_ids=tuple(item[0].sample_id for item in bucket),
                categories=tuple(item[0].category for item in bucket),
                semantic_signatures=signatures,
                semantic_collision=len({repr(signature) for signature in signatures}) > 1,
            )
        )
    return ObservationProbeReport(sample_count, len(buckets), tuple(collisions))


def bounded_reachable_states(
    initial_state: Any,
    *,
    legal_actions: Callable[[Any], Iterable[Any]],
    transition: Callable[[Any, Any], Any],
    state_key: Callable[[Any], Any],
    max_depth: int = 6,
    max_states: int = 2_000,
    category: str = "ordinary",
    seed: int | None = None,
) -> tuple[ReachableSample, ...]:
    """Enumerate only callback-approved legal trajectories with fixed limits."""

    if max_depth < 0 or max_states < 1:
        raise ValueError("Reachability limits must be non-negative depth and positive state count")
    queue: list[tuple[Any, tuple[Any, ...], int]] = [(initial_state, (), 0)]
    seen = {repr(state_key(initial_state))}
    samples: list[ReachableSample] = []
    while queue and len(samples) < max_states:
        state, actions, depth = queue.pop(0)
        samples.append(ReachableSample(f"{category}:{len(samples):05d}", state, actions, category, seed))
        if depth >= max_depth:
            continue
        for action in sorted(legal_actions(state), key=repr):
            child = transition(state, action)
            key = repr(state_key(child))
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, actions + (action,), depth + 1))
    return tuple(samples)


def classify_search_result(report: ObservationProbeReport, *, searched_categories: Sequence[str]) -> dict[str, Any]:
    """Produce the bounded-search wording used by the H1 findings report."""

    found_categories = {
        category
        for collision in report.semantic_collisions
        for category in collision.categories
    }
    not_found_categories = []
    for category in searched_categories:
        if category not in found_categories and category not in not_found_categories:
            not_found_categories.append(category)
    return {
        "found": [collision.observation_digest for collision in report.semantic_collisions],
        "found_categories": sorted(found_categories),
        "same_observation_non_ambiguous": [
            collision.observation_digest
            for collision in report.collisions
            if not collision.semantic_collision
        ],
        "not_found_within_search": not_found_categories,
        "bounded": True,
    }
