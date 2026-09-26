"""Deterministic, history-preserving Cube Arena evaluation starts.

The Arena worker receives only an opening action prefix.  Replaying that
prefix through :class:`CubeSearchAdapter` is deliberately the one source of
truth for both the board and the observation/repetition histories.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import random
from typing import Sequence

from .cube_family import cube_family_topology, initial_cube_state
from .cube_observation_v2 import concrete_observation_identity, initial_cube_observation_context
from .cube_search import CubeSearchAdapter, CubeSearchPosition
from .provenance import derive_seed


STARTSET_SCHEMA = "cube-evaluation-startset-v1"
GENERATOR_VERSION = "cube-evaluation-starts-v1"
STRATA_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _state_payload(position: CubeSearchPosition) -> dict[str, object]:
    state = position.game_state
    context = position.observation_context
    return {
        "stones": [int(value) for value in state.stones],
        "side_to_move": int(state.side_to_move),
        "superko_history": [[int(value) for value in board] for board in state.superko_history],
        "consecutive_passes": int(state.consecutive_passes),
        "previous_boards": [list(board) for board in context.previous_boards],
        "previous_action": context.previous_action,
    }


def opening_fingerprint(
    *, size: int, opening_actions: Sequence[int], position: CubeSearchPosition
) -> str:
    """Fingerprint the replayable prefix and all resulting history state."""

    return _fingerprint(
        {
            "schema": STARTSET_SCHEMA,
            "size": int(size),
            "opening_actions": [int(action) for action in opening_actions],
            "state": _state_payload(position),
        }
    )


@dataclass(frozen=True)
class CubeArenaStart:
    start_id: str
    start_kind: str
    opening_actions: tuple[int, ...]
    opening_ply: int
    start_fingerprint: str

    def __post_init__(self) -> None:
        if self.start_kind not in {"empty_control", "diverse"}:
            raise ValueError("Cube Arena start kind is invalid")
        if len(self.opening_actions) != int(self.opening_ply):
            raise ValueError("Cube Arena opening ply does not match opening actions")
        if any(type(action) is not int or action < 0 for action in self.opening_actions):
            raise ValueError("Cube Arena opening actions must be non-negative action indices")
        if not str(self.start_fingerprint).startswith("sha256:"):
            raise ValueError("Cube Arena start fingerprint must be SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "start_id": self.start_id,
            "start_kind": self.start_kind,
            "opening_actions": list(self.opening_actions),
            "opening_ply": int(self.opening_ply),
            "start_fingerprint": self.start_fingerprint,
        }


@dataclass(frozen=True)
class CubeArenaStartset:
    size: int
    master_seed: int
    pairs: int
    game_fingerprint: str
    observation_fingerprint: str
    topology_fingerprint: str
    starts: tuple[CubeArenaStart, ...]
    fingerprint: str

    @property
    def empty_control_pairs(self) -> int:
        return sum(start.start_kind == "empty_control" for start in self.starts)

    @property
    def diverse_pairs(self) -> int:
        return sum(start.start_kind == "diverse" for start in self.starts)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": STARTSET_SCHEMA,
            "generator": GENERATOR_VERSION,
            "generator_version": GENERATOR_VERSION,
            "topology": f"cube{self.size}",
            "size": int(self.size),
            "topology_fingerprint": self.topology_fingerprint,
            "game_fingerprint": self.game_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "master_seed": int(self.master_seed),
            "pairs": int(self.pairs),
            "paired_colors": True,
            "empty_control": True,
            "strata_fractions": list(STRATA_FRACTIONS),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "starts": [start.to_dict() for start in self.starts],
            "startset_fingerprint": self.fingerprint,
        }


def reconstruct_cube_arena_start(
    *, size: int, opening_actions: Sequence[int]
) -> CubeSearchPosition:
    """Rebuild a start using the exact normal Cube search transition path."""

    state = initial_cube_state(size=int(size))
    position = CubeSearchPosition(state, initial_cube_observation_context(state))
    adapter = CubeSearchAdapter()
    topology = position.topology
    for raw_action in opening_actions:
        if type(raw_action) is not int or not 0 <= raw_action < topology.point_count:
            raise ValueError("Cube Arena openings may contain only legal point action indices")
        if raw_action not in adapter.legal_actions(position):
            raise ValueError(f"Cube Arena opening action is illegal: {raw_action}")
        position = adapter.apply_action(position, raw_action)
        if position.is_terminal:
            raise ValueError("Cube Arena opening reaches a terminal position")
    return position


def _candidate_depth(target: int, maximum: int, attempt: int, size: int) -> int:
    # The first bounded block targets the requested stratum.  If a small Cube
    # exhausts the finite number of positions at a shallow depth, later blocks
    # move deterministically to adjacent depths instead of duplicating starts.
    block = attempt // max(32, int(size) * 16)
    offset = block % 5
    return max(1, min(maximum, int(target) + offset))


@lru_cache(maxsize=32)
def build_cube_arena_startset(*, size: int, master_seed: int, pairs: int) -> CubeArenaStartset:
    """Build exactly one control plus ``pairs - 1`` unique diverse starts."""

    if type(size) is not int or not 2 <= size <= 7:
        raise ValueError("Cube Arena startset size must be between 2 and 7")
    if type(master_seed) is not int or master_seed <= 0:
        raise ValueError("Cube Arena startset master_seed must be a positive integer")
    if type(pairs) is not int or pairs <= 0:
        raise ValueError("Cube Arena startset pairs must be positive")

    topology = cube_family_topology(size)
    initial_state = initial_cube_state(size=size)
    initial = CubeSearchPosition(
        initial_state,
        initial_cube_observation_context(initial_state),
    )
    empty_fingerprint = opening_fingerprint(size=size, opening_actions=(), position=initial)
    starts: list[CubeArenaStart] = [
        CubeArenaStart(
            start_id="canonical-empty-board",
            start_kind="empty_control",
            opening_actions=(),
            opening_ply=0,
            start_fingerprint=empty_fingerprint,
        )
    ]
    seen = {empty_fingerprint}
    required = pairs - 1
    maximum_depth = max(8, min(topology.point_count * 2, topology.point_count + 16))
    max_attempts = max(2000, required * 240)
    adapter = CubeSearchAdapter()
    generated = 0
    for attempt in range(max_attempts):
        if generated >= required:
            break
        stratum = generated % len(STRATA_FRACTIONS)
        target = max(1, round(topology.point_count * STRATA_FRACTIONS[stratum]))
        depth = _candidate_depth(target, maximum_depth, attempt, size)
        rng = random.Random(derive_seed(master_seed, GENERATOR_VERSION, size, generated, attempt))
        state = initial_cube_state(size=size)
        position = CubeSearchPosition(state, initial_cube_observation_context(state))
        actions: list[int] = []
        try:
            for _ in range(depth):
                legal_points = tuple(
                    int(action)
                    for action in adapter.legal_actions(position)
                    if type(action) is int and 0 <= action < topology.point_count
                )
                if not legal_points:
                    raise ValueError("opening has no legal non-PASS action")
                action = legal_points[rng.randrange(len(legal_points))]
                actions.append(action)
                position = adapter.apply_action(position, action)
                if position.is_terminal:
                    raise ValueError("opening became terminal")
        except (ValueError, IndexError):
            continue
        fingerprint = opening_fingerprint(size=size, opening_actions=actions, position=position)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        starts.append(
            CubeArenaStart(
                start_id=f"cube{size}-diverse-{generated:04d}",
                start_kind="diverse",
                opening_actions=tuple(actions),
                opening_ply=len(actions),
                start_fingerprint=fingerprint,
            )
        )
        generated += 1
    if generated != required:
        raise ValueError(
            "Cube Arena startset generation failed closed: "
            f"needed {required} unique diverse starts, generated {generated} "
            f"within {max_attempts} bounded attempts"
        )

    descriptor = {
        "schema": STARTSET_SCHEMA,
        "generator": GENERATOR_VERSION,
        "generator_version": GENERATOR_VERSION,
        "topology": f"cube{size}",
        "size": size,
        "topology_fingerprint": topology.fingerprint,
        "game_fingerprint": initial_state.rules_fingerprint,
        "observation_fingerprint": str(
            concrete_observation_identity(topology)["concrete_observation_fingerprint"]
        ),
        "master_seed": master_seed,
        "pairs": pairs,
        "paired_colors": True,
        "empty_control": True,
        "strata_fractions": list(STRATA_FRACTIONS),
    }
    corpus_fingerprint = _fingerprint(
        {"descriptor": descriptor, "starts": [start.to_dict() for start in starts]}
    )
    return CubeArenaStartset(
        size=size,
        master_seed=master_seed,
        pairs=pairs,
        game_fingerprint=str(descriptor["game_fingerprint"]),
        observation_fingerprint=str(descriptor["observation_fingerprint"]),
        topology_fingerprint=topology.fingerprint,
        starts=tuple(starts),
        fingerprint=corpus_fingerprint,
    )


__all__ = [
    "CubeArenaStart",
    "CubeArenaStartset",
    "GENERATOR_VERSION",
    "STARTSET_SCHEMA",
    "STRATA_FRACTIONS",
    "build_cube_arena_startset",
    "opening_fingerprint",
    "reconstruct_cube_arena_start",
]
