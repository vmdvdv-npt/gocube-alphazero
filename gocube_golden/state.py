from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
from typing import Iterable, Sequence

from .diagnostics import increment
from .topology import GoldenTopology, TORUS_5X5, TORUS_5X5_TOPOLOGY_FINGERPRINT

RULES_PROFILE_ID = "graph-area-v1"
STAGE0_RULES_FINGERPRINT = "sha256:8eac3337443a70893fa5ad359580f7ba92b18958e06f0d775c29f08791796842"
BASELINE_KOMI = 0.5
LEGACY_FORBIDDEN_KOMI = 7.5
PASS = "PASS"
LIVE_HISTORY = "canonical-live"
SYNTHETIC_HISTORY = "synthetic-test-research"

class LegacyKomiError(ValueError):
    pass

class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

EMPTY, BLACK, WHITE = Stone.EMPTY, Stone.BLACK, Stone.WHITE
BoardKey = tuple[int, ...]

def opponent(color: Stone) -> Stone:
    if color == BLACK:
        return WHITE
    if color == WHITE:
        return BLACK
    raise ValueError("EMPTY has no opponent")

def validate_komi(value: object, *, context: str = "Golden graph-area-v1") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} requires finite numeric komi, got {value!r}")
    try:
        komi = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} requires finite numeric komi, got {value!r}") from exc
    if not math.isfinite(komi):
        raise ValueError(f"{context} requires finite numeric komi, got {value!r}")
    if math.isclose(komi, LEGACY_FORBIDDEN_KOMI, rel_tol=0.0, abs_tol=1e-12):
        raise LegacyKomiError(
            "Golden Stage 1 received forbidden legacy komi 7.5: likely stale-artifact "
            "contamination. Stop and contact the project owner; do not silently coerce to 0.5."
        )
    return komi

def board_key(stones: Sequence[Stone | int]) -> BoardKey:
    return tuple(int(stone) for stone in stones)

def _research_rules_fingerprint(topology: GoldenTopology, komi: float) -> str:
    payload = {
        "rules_profile_id": RULES_PROFILE_ID, "version": 1, "suicide": "forbidden",
        "ko": "positional-superko", "terminal": "two-consecutive-passes", "scoring": "graph-area",
        "topology_fingerprint": topology.fingerprint, "komi": komi, "scope": "golden-stage1-test-research",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()

def rules_fingerprint_for(topology: GoldenTopology, komi: float) -> str:
    komi = validate_komi(komi)
    if topology.fingerprint == TORUS_5X5_TOPOLOGY_FINGERPRINT and math.isclose(
        komi, BASELINE_KOMI, rel_tol=0.0, abs_tol=1e-12
    ):
        return STAGE0_RULES_FINGERPRINT
    return _research_rules_fingerprint(topology, komi)

def _stones(values: Iterable[Stone | int], count: int) -> tuple[Stone, ...]:
    try:
        normalized = tuple(Stone(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid stone value") from exc
    if len(normalized) != count:
        raise ValueError(f"Expected {count} stones, got {len(normalized)}")
    return normalized

def _history(values: Iterable[Sequence[Stone | int]], count: int) -> tuple[BoardKey, ...]:
    history = tuple(board_key(position) for position in values)
    if not history:
        raise ValueError("Superko history must contain the initial board arrangement")
    for position in history:
        if len(position) != count or any(value not in (0, 1, 2) for value in position):
            raise ValueError("Invalid board arrangement in superko history")
    return history

@dataclass(frozen=True)
class GoldenState:
    stones: tuple[Stone, ...]
    side_to_move: Stone
    superko_history: tuple[BoardKey, ...]
    consecutive_passes: int
    topology: GoldenTopology
    rules_id: str
    rules_fingerprint: str
    komi: float
    history_provenance: str = LIVE_HISTORY

    def __post_init__(self) -> None:
        increment("validated_state_constructions")
        increment("full_history_validations")
        object.__setattr__(self, "stones", _stones(self.stones, self.topology.point_count))
        object.__setattr__(self, "_board_key", board_key(self.stones))
        try:
            side = Stone(self.side_to_move)
        except (TypeError, ValueError) as exc:
            raise ValueError("side_to_move must be BLACK or WHITE") from exc
        if side not in (BLACK, WHITE):
            raise ValueError("side_to_move must be BLACK or WHITE")
        object.__setattr__(self, "side_to_move", side)
        history = _history(self.superko_history, self.topology.point_count)
        object.__setattr__(self, "superko_history", history)
        object.__setattr__(self, "_superko_membership", frozenset(history))
        if isinstance(self.consecutive_passes, bool) or not isinstance(self.consecutive_passes, int):
            raise ValueError("consecutive_passes must be an integer")
        if self.consecutive_passes not in (0, 1, 2):
            raise ValueError("consecutive_passes must be 0, 1, or 2")
        if self.rules_id != RULES_PROFILE_ID:
            raise ValueError(f"Golden Stage 1 supports only {RULES_PROFILE_ID}")
        komi = validate_komi(self.komi)
        object.__setattr__(self, "komi", komi)
        expected = rules_fingerprint_for(self.topology, komi)
        if self.rules_fingerprint != expected:
            raise ValueError("Golden rules fingerprint does not match topology/komi identity")
        if self.history_provenance not in (LIVE_HISTORY, SYNTHETIC_HISTORY):
            raise ValueError(f"Unsupported Golden history provenance {self.history_provenance!r}")

        if self.history_provenance == LIVE_HISTORY:
            if history[-1] != self.board_key:
                raise ValueError("Canonical live superko history must end at the current board")
            if len(set(history)) != len(history):
                raise ValueError(
                    "Canonical live superko history cannot contain duplicate stone positions"
                )
        elif self.board_key not in history:
            raise ValueError(
                "Synthetic fixture superko history must still contain the current board arrangement"
            )

    @property
    def board_key(self) -> BoardKey:
        return self._board_key

    @property
    def superko_membership(self) -> frozenset[BoardKey]:
        """Exact immutable membership index for positional-superko lookups."""

        return self._superko_membership

    @classmethod
    def _from_trusted_transition(
        cls,
        parent: "GoldenState",
        *,
        stones: tuple[Stone, ...],
        side_to_move: Stone,
        consecutive_passes: int,
        append_board_key: BoardKey | None,
    ) -> "GoldenState":
        """Construct a child after rules.py has completed local transition checks.

        This intentionally bypasses the external-construction audit.  The parent
        is already validated, and rules.py supplies either the unchanged history
        (PASS) or exactly one non-repeating resulting board key (a point move).
        """

        if append_board_key is None:
            history = parent.superko_history
            membership = parent.superko_membership
        else:
            history = parent.superko_history + (append_board_key,)
            membership = parent.superko_membership | frozenset((append_board_key,))
        child = object.__new__(cls)
        object.__setattr__(child, "stones", stones)
        object.__setattr__(
            child,
            "_board_key",
            append_board_key if append_board_key is not None else parent.board_key,
        )
        object.__setattr__(child, "side_to_move", side_to_move)
        object.__setattr__(child, "superko_history", history)
        object.__setattr__(child, "consecutive_passes", consecutive_passes)
        object.__setattr__(child, "topology", parent.topology)
        object.__setattr__(child, "rules_id", parent.rules_id)
        object.__setattr__(child, "rules_fingerprint", parent.rules_fingerprint)
        object.__setattr__(child, "komi", parent.komi)
        object.__setattr__(child, "history_provenance", parent.history_provenance)
        object.__setattr__(child, "_superko_membership", membership)
        increment("trusted_state_constructions")
        return child

    @property
    def is_terminal(self) -> bool:
        return self.consecutive_passes == 2

    @property
    def is_canonical_live(self) -> bool:
        return self.history_provenance == LIVE_HISTORY

    @property
    def state_key(self) -> tuple[object, ...]:
        return (
            self.board_key, int(self.side_to_move), self.superko_history, self.consecutive_passes,
            self.topology.topology_id, self.topology.fingerprint, self.rules_id,
            self.rules_fingerprint, self.komi, self.history_provenance,
        )

def initial_state(*, topology: GoldenTopology = TORUS_5X5, komi: float = BASELINE_KOMI) -> GoldenState:
    komi = validate_komi(komi)
    stones = tuple(EMPTY for _ in range(topology.point_count))
    return GoldenState(
        stones, BLACK, (board_key(stones),), 0, topology, RULES_PROFILE_ID,
        rules_fingerprint_for(topology, komi), komi, LIVE_HISTORY,
    )

def research_state_from_stones(
    stones: Sequence[Stone | int], *, side_to_move: Stone = BLACK,
    topology: GoldenTopology = TORUS_5X5, komi: float = BASELINE_KOMI,
    superko_history: Iterable[Sequence[Stone | int]] | None = None,
    consecutive_passes: int = 0,
) -> GoldenState:
    """Build an explicitly synthetic test/research position.

    The constructor validates board/history structure but deliberately does not
    claim that the supplied position/history is reachable by legal play.
    Sequential Golden Arena rejects this provenance for real evaluation games.
    """
    komi = validate_komi(komi)
    normalized = _stones(stones, topology.point_count)
    current = board_key(normalized)
    history = (current,) if superko_history is None else _history(superko_history, topology.point_count)
    if current not in history:
        raise ValueError("Provided synthetic superko history must include the current board arrangement")
    return GoldenState(
        normalized, Stone(side_to_move), history, consecutive_passes, topology,
        RULES_PROFILE_ID, rules_fingerprint_for(topology, komi), komi, SYNTHETIC_HISTORY,
    )

def state_from_stones(
    stones: Sequence[Stone | int], *, side_to_move: Stone = BLACK,
    topology: GoldenTopology = TORUS_5X5, komi: float = BASELINE_KOMI,
    superko_history: Iterable[Sequence[Stone | int]] | None = None,
    consecutive_passes: int = 0,
) -> GoldenState:
    """Compatibility alias for Stage-1 synthetic fixtures.

    New code should use ``research_state_from_stones`` so synthetic provenance
    is explicit at the call site.
    """
    return research_state_from_stones(
        stones, side_to_move=side_to_move, topology=topology, komi=komi,
        superko_history=superko_history, consecutive_passes=consecutive_passes,
    )
