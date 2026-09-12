from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol, Sequence

from .arena_contract import DEFAULT_ARENA_CONTRACT
from .provenance import PlayerIdentity
from .search import Evaluator, SearchError, SequentialPUCT
from .state import PASS, GoldenState


@dataclass(frozen=True)
class PlayerContext:
    game_id: str
    pair_id: str
    player_slot: str
    player_id: str
    assigned_color: str
    player_move_index: int
    ply: int
    seed: int
    search_contract_id: str


class Player(Protocol):
    player_id: str
    is_search_player: bool

    def select_action(self, state: GoldenState, context: PlayerContext) -> int | str:
        ...


@dataclass(frozen=True)
class GoodPlayer:
    """Deterministic proof player: play one stone at Point 0, then PASS."""

    player_id: str = "GOOD"
    is_search_player: bool = False

    def select_action(self, state: GoldenState, context: PlayerContext) -> int | str:
        return 0 if context.player_move_index == 0 else PASS


@dataclass(frozen=True)
class BadPlayer:
    """Deterministic proof player: always PASS."""

    player_id: str = "BAD"
    is_search_player: bool = False

    def select_action(self, state: GoldenState, context: PlayerContext) -> int | str:
        return PASS


@dataclass(frozen=True)
class TracePlayer:
    actions: tuple[int | str, ...]
    player_id: str = "TRACE"
    is_search_player: bool = False

    def __init__(self, actions: Sequence[int | str], player_id: str = "TRACE") -> None:
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(self, "player_id", player_id)
        object.__setattr__(self, "is_search_player", False)

    def select_action(self, state: GoldenState, context: PlayerContext) -> int | str:
        if context.player_move_index >= len(self.actions):
            return PASS
        return self.actions[context.player_move_index]


@dataclass(frozen=True)
class SearchPlayer:
    player_id: str
    search: SequentialPUCT
    evaluator: Evaluator
    identity: PlayerIdentity | None = None
    is_search_player: bool = True

    def __post_init__(self) -> None:
        if self.search.settings != DEFAULT_ARENA_CONTRACT.search:
            raise ValueError("SearchPlayer settings must exactly match Golden Arena contract")
        if self.identity is not None:
            self.identity.validate()
            if self.identity.logical_player_id != self.player_id:
                raise ValueError(
                    "SearchPlayer identity logical_player_id must exactly match player_id"
                )
        checkpoint_like = any(
            hasattr(self.evaluator, name)
            for name in ("checkpoint_path", "model_path", "checkpoint_metadata")
        )
        if checkpoint_like and (
            self.identity is None or self.identity.player_kind != "checkpoint"
        ):
            raise ValueError(
                "Checkpoint/model-backed SearchPlayer requires structured checkpoint identity"
            )

    def select_action(self, state: GoldenState, context: PlayerContext) -> int | str:
        # Per-move seed is supplied by Arena and is independent of the opponent's
        # RNG consumption. SequentialPUCT itself is stateless across calls.
        result = self.search.search(state, self.evaluator, seed=context.seed)
        if result.action not in result.legal_actions:
            raise SearchError("Search returned action outside its own legal set")
        return result.action


def derive_child_seed(parent_seed: int, label: str) -> int:
    payload = f"{int(parent_seed)}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
