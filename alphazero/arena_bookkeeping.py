"""Immutable identity and routing records for batched Arena games.

The Arena worker process is allowed to recycle a slot as soon as a game is
finished.  These records make the identity crossing an IPC boundary explicit;
neither queue order nor the worker's current mutable state is part of the
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, NamedTuple


class ArenaRoutingKey(NamedTuple):
    """Stable key for one search row in one slot generation."""

    worker_id: int
    slot_id: int
    generation: int
    game_id: int


@dataclass(frozen=True)
class ArenaGameIdentity:
    """Immutable identity assigned to one active game slot."""

    game_id: int
    worker_id: int
    slot_id: int
    generation: int
    model_a_color: str
    player_to_index: tuple[int, int]

    @property
    def routing_key(self) -> ArenaRoutingKey:
        return ArenaRoutingKey(
            int(self.worker_id),
            int(self.slot_id),
            int(self.generation),
            int(self.game_id),
        )


@dataclass
class ArenaGameSlot:
    """Runtime snapshot owned by one worker's active slot.

    ``game_state`` and ``mcts_state`` are updated in place by the worker, while
    ``identity`` is replaced only when the slot is recycled for a new game.
    """

    identity: ArenaGameIdentity
    game_state: Any
    mcts_state: Any
    completion_state: str = "active"


@dataclass(frozen=True)
class ArenaResult:
    """Self-contained result snapshot published after a game completes."""

    final_state: Any
    winstate: Any
    game_id: int
    worker_id: int
    slot_id: int
    generation: int
    model_a_color: str
    player_to_index: tuple[int, int]

    @classmethod
    def from_slot(cls, slot: ArenaGameSlot, final_state: Any, winstate: Any) -> "ArenaResult":
        identity = slot.identity
        return cls(
            final_state=final_state,
            winstate=winstate,
            game_id=int(identity.game_id),
            worker_id=int(identity.worker_id),
            slot_id=int(identity.slot_id),
            generation=int(identity.generation),
            model_a_color=str(identity.model_a_color),
            player_to_index=tuple(int(value) for value in identity.player_to_index),
        )


def model_a_color_for_game_id(game_id: int) -> str:
    """Deterministic, global color schedule independent of worker timing."""

    return "black" if int(game_id) % 2 == 0 else "white"


def player_to_index_for_game_id(game_id: int) -> tuple[int, int]:
    """Return semantic-player -> model-index routing for one game."""

    if model_a_color_for_game_id(game_id) == "black":
        return (0, 1)
    return (1, 0)


def arena_game_ids_by_worker(total_games: int, workers: int) -> tuple[tuple[int, ...], ...]:
    """Partition global game IDs into deterministic per-worker quotas.

    The partition is fixed before processes start.  Each ID occurs exactly
    once, and its parity determines color, so worker speed and completion order
    cannot change the aggregate color schedule.
    """

    total_games = int(total_games)
    workers = int(workers)
    if total_games < 0:
        raise ValueError("total_games must be non-negative")
    if workers < 1:
        raise ValueError("workers must be at least one")
    base, remainder = divmod(total_games, workers)
    result: list[tuple[int, ...]] = []
    cursor = 0
    for worker_id in range(workers):
        quota = base + (1 if worker_id < remainder else 0)
        result.append(tuple(range(cursor, cursor + quota)))
        cursor += quota
    if cursor != total_games:
        raise RuntimeError("Arena game ID partition does not cover requested games")
    return tuple(result)


def routing_keys_for_payload(payload: dict, model_index: int) -> tuple[ArenaRoutingKey, ...]:
    """Validate and normalize one model's routing keys from a worker payload."""

    keys_by_model = payload.get("routing_keys")
    if not isinstance(keys_by_model, (list, tuple)):
        raise RuntimeError("Arena inference payload is missing routing_keys")
    if model_index < 0 or model_index >= len(keys_by_model):
        raise RuntimeError("Arena inference payload has an invalid model index")
    keys = tuple(keys_by_model[model_index])
    normalized: list[ArenaRoutingKey] = []
    for key in keys:
        if isinstance(key, ArenaRoutingKey):
            normalized.append(key)
            continue
        if isinstance(key, (list, tuple)) and len(key) == 4:
            normalized.append(ArenaRoutingKey(*(int(value) for value in key)))
            continue
        raise RuntimeError(f"invalid Arena inference routing key: {key!r}")
    return tuple(normalized)


def flatten_routing_keys(keys_by_model: Iterable[Iterable[ArenaRoutingKey]]) -> tuple[ArenaRoutingKey, ...]:
    return tuple(key for model_keys in keys_by_model for key in model_keys)
