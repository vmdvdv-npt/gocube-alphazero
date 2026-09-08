from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from .core import BLACK, EMPTY, WHITE, FinalScore, StoneBreakdown, TerritoryBreakdown, TerritoryPoints, Topology
MAIN = "main"
CLEANUP_1 = "cleanup1"
CLEANUP_2 = "cleanup2"
SCORED = "scored"
NO_RESULT = "no_result"

# The value head has three fixed, player-to-move-relative classes.  These are
# deliberately not called WIN/LOSS/DRAW: a scored draw is represented by a
# mixture of WIN and LOSS, while NO_RESULT is a distinct terminal outcome.
VALUE_WIN = 0
VALUE_LOSS = 1
VALUE_NO_RESULT = 2
VALUE_TARGET_SIZE = 3
VALUE_TARGET_SEMANTICS = "win-loss-noresult-s1-v2"

KATAGO_JAPANESE_ADJUDICATOR_V3 = "gocube-katago-japanese-v3"
OBSERVATION_SCHEMA_V3 = "gocube-observation-v3"
KATAGO_RULES_VERSION = 3
# The scorer/setup contract changed in S1.  Keep this separate from the
# upstream rules number: KataGo still supplies Rules V3, while this is the
# version of our faithful state/adjudicator implementation.
KATAGO_RULES_IMPLEMENTATION_VERSION = 4
KATAGO_REFERENCE_COMMIT = "f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
KATAGO_REFERENCE_VERSION = "1.18.0+ Rules Version 3"

EMERGENCY_MOVE_CAP_FACTOR = 24
EMERGENCY_MOVE_CAP_BASE = 256

# Termination is deliberately represented as two dimensions.  ``terminal_kind``
# describes the value returned by the rule engine, while these values describe
# why that terminal was reached.  Runtime force-scoring is never a formal rule
# result even though it has the same scored terminal kind and numeric labels.
FORMAL_PASS = "formal_pass"
PASS_ALIVE = "pass_alive"
CYCLE = "cycle"
EPISODE_MOVE_LIMIT = "episode_move_limit"
UNKNOWN_LEGACY_TERMINATION = "unknown_legacy_termination"

RESULT_PROVENANCE_FORMAL = "formal"
RESULT_PROVENANCE_RULE_NO_RESULT = "rule_no_result"
RESULT_PROVENANCE_RUNTIME = "runtime"


def episode_move_limit(topology: Topology) -> int:
    """Return the production self-play budget for ``topology``.

    This is a runner budget, not a rule transition.  The legacy constant names
    remain above as import compatibility for diagnostics and old callers, but
    no formal transition consults them.
    """

    point_count = int(topology.point_count)
    if point_count <= 0:
        raise ValueError("topology must contain at least one point")
    return EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * point_count


def rules_fingerprint(topology: Topology, komi: float = 0.5) -> str:
    payload = {
        "adjudicator": KATAGO_JAPANESE_ADJUDICATOR_V3,
        "observationSchema": OBSERVATION_SCHEMA_V3,
        "rulesVersion": KATAGO_RULES_VERSION,
        "rulesImplementationVersion": KATAGO_RULES_IMPLEMENTATION_VERSION,
        "katagoCommit": KATAGO_REFERENCE_COMMIT,
        "koRule": "SIMPLE",
        "scoringRule": "TERRITORY",
        "taxRule": "SEKI",
        "multiStoneSuicide": False,
        "button": False,
        "whiteHandicapBonus": 0,
        "selfPlayOpts": True,
        "topology": topology.kind,
        "size": topology.size,
        "pointCount": topology.point_count,
        "komi": float(komi),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _readonly_board(values: Sequence[int] | np.ndarray, point_count: int) -> np.ndarray:
    board = np.asarray(values, dtype=np.uint8).reshape(-1).copy()
    if board.shape != (point_count,):
        raise ValueError(f"Expected board of length {point_count}, got {board.shape}")
    if not np.isin(board, (EMPTY, BLACK, WHITE)).all():
        raise ValueError("Board contains invalid occupancy values")
    board.flags.writeable = False
    return board


def _board_key(board: np.ndarray) -> bytes:
    return bytes(np.asarray(board, dtype=np.uint8).tolist())


def _state_key(board: np.ndarray, player: int, blocked: Iterable[int]) -> bytes:
    mask = bytearray(len(board))
    for point in blocked:
        mask[point] = 1
    return bytes((player,)) + _board_key(board) + bytes(mask)


@dataclass(frozen=True, eq=False)
class V3State:
    board: np.ndarray
    current_player: int = 0
    turns: int = 0
    consecutive_passes: int = 0
    captures: tuple[int, int] = (0, 0)
    # KataGo's BoardHistory::whiteBonusScore.  It is score-relevant state,
    # not a value that can be inferred from whether MAIN moves happened.
    # GoCube stores captures by capturing player: captures[0] is black's
    # captured-white count and captures[1] is white's captured-black count.
    white_bonus_score: float = 0.0
    previous_board: np.ndarray | None = None
    phase: str = MAIN
    ko_recap_blocked: tuple[int, ...] = ()
    phase_history: tuple[bytes, ...] = ()
    history_since_pass: tuple[bytes, ...] = ()
    black_pass_states: tuple[bytes, ...] = ()
    white_pass_states: tuple[bytes, ...] = ()
    ko_capture_history: tuple[tuple[int, int, bytes], ...] = ()
    second_cleanup_start_colors: bytes | None = None
    cleanup2_moves: tuple[int, int] = (0, 0)
    main_moves: tuple[int, int] = (0, 0)
    cleanup1_moves: tuple[int, int] = (0, 0)
    terminal_kind: str | None = None
    no_result_reason: str | None = None
    termination_reason: str | None = None
    result_provenance: str | None = None
    pass_alive_early_end: bool = False
    entered_cleanup1: bool = False
    entered_cleanup2: bool = False
    cleanup_captures: int = 0
    ko_unblock_actions: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, V3State):
            return NotImplemented
        scalar = (
            self.current_player == other.current_player
            and self.turns == other.turns
            and self.consecutive_passes == other.consecutive_passes
            and self.captures == other.captures
            and self.white_bonus_score == other.white_bonus_score
            and self.phase == other.phase
            and self.ko_recap_blocked == other.ko_recap_blocked
            and self.phase_history == other.phase_history
            and self.history_since_pass == other.history_since_pass
            and self.black_pass_states == other.black_pass_states
            and self.white_pass_states == other.white_pass_states
            and self.ko_capture_history == other.ko_capture_history
            and self.second_cleanup_start_colors == other.second_cleanup_start_colors
            and self.cleanup2_moves == other.cleanup2_moves
            and self.main_moves == other.main_moves
            and self.cleanup1_moves == other.cleanup1_moves
            and self.terminal_kind == other.terminal_kind
            and self.no_result_reason == other.no_result_reason
            and self.termination_reason == other.termination_reason
            and self.result_provenance == other.result_provenance
            and self.pass_alive_early_end == other.pass_alive_early_end
            and self.entered_cleanup1 == other.entered_cleanup1
            and self.entered_cleanup2 == other.entered_cleanup2
            and self.cleanup_captures == other.cleanup_captures
            and self.ko_unblock_actions == other.ko_unblock_actions
        )
        return scalar and np.array_equal(self.board, other.board) and (
            (self.previous_board is None and other.previous_board is None)
            or (
                self.previous_board is not None
                and other.previous_board is not None
                and np.array_equal(self.previous_board, other.previous_board)
            )
        )


@dataclass(frozen=True)
class PassAliveAnalysis:
    pass_alive_black_groups: tuple[tuple[int, ...], ...]
    pass_alive_white_groups: tuple[tuple[int, ...], ...]
    pass_alive_black_territory: tuple[int, ...]
    pass_alive_white_territory: tuple[int, ...]

    @property
    def covered_points(self) -> frozenset[int]:
        points: set[int] = set(self.pass_alive_black_territory)
        points.update(self.pass_alive_white_territory)
        for group in self.pass_alive_black_groups + self.pass_alive_white_groups:
            points.update(group)
        return frozenset(points)


@dataclass(frozen=True)
class IndependentLifeAnalysis:
    black_area: tuple[int, ...]
    white_area: tuple[int, ...]
    black_regions: tuple[tuple[int, ...], ...]
    white_regions: tuple[tuple[int, ...], ...]
    black_territory: tuple[int, ...]
    white_territory: tuple[int, ...]
    dame: tuple[int, ...]
    seki: tuple[int, ...]


@dataclass(frozen=True)
class V3Terminal:
    terminal_kind: str
    score: FinalScore | None
    ownership: np.ndarray | None
    ownership_mask: np.ndarray | None
    reason: str | None = None
    result_provenance: str | None = None

    @property
    def training_valid(self) -> bool:
        return self.terminal_kind == SCORED and self.score is not None

    @property
    def value_target_valid(self) -> bool:
        return self.terminal_kind in (SCORED, NO_RESULT)

    @property
    def score_target_valid(self) -> bool:
        return self.terminal_kind == SCORED and self.score is not None

    @property
    def ownership_target_valid(self) -> bool:
        return (
            self.terminal_kind == SCORED
            and self.ownership is not None
            and self.ownership_mask is not None
        )

    @property
    def winner(self) -> str:
        if self.score is None:
            return "draw"
        return self.score.winner

    @property
    def no_result(self) -> bool:
        return self.terminal_kind == NO_RESULT

    @property
    def termination_reason(self) -> str | None:
        """Named view used by record/target consumers."""

        return self.reason

    @property
    def target_provenance(self) -> str | None:
        return self.result_provenance


@dataclass(frozen=True)
class V3TrainingTargets:
    """All labels derived from one terminal result.

    ``score_target`` intentionally contains NaN for NO_RESULT.  Consumers
    must select active rows before evaluating a score loss; multiplying a NaN
    by a zero mask is not a valid substitute.
    """

    value_target: np.ndarray
    score_target: np.ndarray
    score_mask: np.ndarray
    ownership_target: np.ndarray
    ownership_mask: np.ndarray
    terminal_kind: str
    termination_reason: str | None = None
    result_provenance: str | None = None

    def __post_init__(self):
        if self.value_target.shape != (VALUE_TARGET_SIZE,):
            raise ValueError(f"value_target must have shape ({VALUE_TARGET_SIZE},)")
        if self.score_target.shape != (1,) or self.score_mask.shape != (1,):
            raise ValueError("score targets and mask must have shape (1,)")
        if self.ownership_target.ndim != 2 or self.ownership_target.shape[1] != 3:
            raise ValueError("ownership_target must have shape (point_count, 3)")
        if self.ownership_mask.shape != (self.ownership_target.shape[0],):
            raise ValueError("ownership_mask must have one entry per point")
        arrays = (
            self.value_target,
            self.score_target,
            self.score_mask,
            self.ownership_target,
            self.ownership_mask,
        )
        if any(array.dtype != np.float32 for array in arrays):
            raise ValueError("V3 training targets must use float32 arrays")

    def __iter__(self):
        """Compatibility view for old scored-game callers."""

        yield self.score_target
        yield self.ownership_target
        yield self.ownership_mask

    @property
    def target_provenance(self) -> str | None:
        """Provenance of every target in this bundle."""

        return self.result_provenance


class V3IllegalMove(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def initial_v3_state(topology: Topology) -> V3State:
    board = _readonly_board(np.zeros(topology.point_count, dtype=np.uint8), topology.point_count)
    key = _state_key(board, 0, ())
    return V3State(
        board=board,
        white_bonus_score=0.0,
        phase_history=(key,),
        history_since_pass=(key,),
    )


def _validate_captures(captures: tuple[int, int]) -> tuple[int, int]:
    try:
        normalized = (int(captures[0]), int(captures[1]))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("captures must be a pair of non-negative integers") from exc
    if any(isinstance(value, bool) or value < 0 for value in normalized):
        raise ValueError("captures must be a pair of non-negative integers")
    if tuple(captures) != normalized:
        raise ValueError("captures must contain integer values")
    return normalized


def _validate_color_snapshot(colors: bytes | None, point_count: int) -> bytes | None:
    if colors is None:
        return None
    normalized = bytes(colors)
    if len(normalized) != point_count:
        raise ValueError(
            "second_cleanup_start_colors must contain one occupancy byte per point"
        )
    if any(value not in (EMPTY, BLACK, WHITE) for value in normalized):
        raise ValueError("second_cleanup_start_colors contains invalid occupancy values")
    return normalized


def _boardhistory_clear_white_bonus(board: np.ndarray, captures: tuple[int, int]) -> float:
    """Return the score offset initialized by pinned ``BoardHistory::clear``.

    ``Board::numWhiteCaptures`` counts white stones captured by Black and
    ``Board::numBlackCaptures`` counts black stones captured by White.  The
    public V3 state stores the same information by capturing player, hence the
    subtraction below.  This value is deliberately computed from the explicit
    setup state and capture state, never from move counters.
    """

    black_stones = int(np.count_nonzero(np.asarray(board) == BLACK))
    white_stones = int(np.count_nonzero(np.asarray(board) == WHITE))
    black_captures, white_captures = captures
    return float(black_stones - white_stones - black_captures + white_captures)


def v3_state_from_board(
    topology: Topology,
    *,
    black: Iterable[int] = (),
    white: Iterable[int] = (),
    current_player: int = 0,
    turns: int = 0,
    captures: tuple[int, int] = (0, 0),
    phase: str = MAIN,
    previous_board: np.ndarray | None = None,
    ko_recap_blocked: Iterable[int] = (),
    second_cleanup_start_colors: bytes | None = None,
    cleanup2_moves: tuple[int, int] = (0, 0),
) -> V3State:
    # SCORED/NO_RESULT are retained for explicit test and compatibility
    # fixtures that construct a terminal view directly. New setup/replay
    # inputs should use one of the three encore phases.
    if phase not in (MAIN, CLEANUP_1, CLEANUP_2, SCORED, NO_RESULT):
        raise ValueError(f"unsupported V3 setup phase: {phase!r}")
    if int(current_player) not in (0, 1):
        raise ValueError("current_player must be 0 (black) or 1 (white)")
    current_player = int(current_player)
    captures = _validate_captures(captures)
    board = np.zeros(topology.point_count, dtype=np.uint8)
    for p in black:
        board[p] = BLACK
    for p in white:
        if board[p] != EMPTY:
            raise ValueError("Overlapping fixture stones")
        board[p] = WHITE
    board = _readonly_board(board, topology.point_count)
    blocked = tuple(sorted(set(ko_recap_blocked)))
    key = _state_key(board, current_player, blocked)
    prev = None if previous_board is None else _readonly_board(previous_board, topology.point_count)
    second_start = _validate_color_snapshot(second_cleanup_start_colors, topology.point_count)
    if phase == CLEANUP_2:
        second_start = _board_key(board) if second_start is None else second_start
    elif second_start is not None:
        raise ValueError("second_cleanup_start_colors is only valid in CLEANUP_2")
    return V3State(
        board=board,
        current_player=current_player,
        turns=turns,
        captures=captures,
        white_bonus_score=_boardhistory_clear_white_bonus(board, captures),
        previous_board=prev,
        phase=phase,
        ko_recap_blocked=blocked,
        phase_history=(key,),
        history_since_pass=(key,),
        second_cleanup_start_colors=second_start,
        cleanup2_moves=cleanup2_moves,
        entered_cleanup1=phase in (CLEANUP_1, CLEANUP_2),
        entered_cleanup2=phase == CLEANUP_2,
    )


def _collect_group(board: np.ndarray, start: int, color: int, topology: Topology) -> tuple[set[int], set[int]]:
    group = {start}
    liberties: set[int] = set()
    pending = [start]
    while pending:
        point = pending.pop()
        for neighbor in topology.neighbor_indices(point):
            occupancy = int(board[neighbor])
            if occupancy == EMPTY:
                liberties.add(neighbor)
            elif occupancy == color and neighbor not in group:
                group.add(neighbor)
                pending.append(neighbor)
    return group, liberties


def _all_groups(board: np.ndarray, topology: Topology, color: int | None = None) -> tuple[tuple[int, ...], ...]:
    visited: set[int] = set()
    result: list[tuple[int, ...]] = []
    for point in range(topology.point_count):
        c = int(board[point])
        if c == EMPTY or point in visited or (color is not None and c != color):
            continue
        group, _ = _collect_group(board, point, c, topology)
        visited.update(group)
        result.append(tuple(sorted(group)))
    return tuple(result)


def _components_matching(board: np.ndarray, topology: Topology, allowed) -> tuple[tuple[int, ...], ...]:
    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(topology.point_count):
        if start in visited or not allowed(int(board[start])):
            continue
        visited.add(start)
        pending = [start]
        component: list[int] = []
        while pending:
            point = pending.pop()
            component.append(point)
            for neighbor in topology.neighbor_indices(point):
                if neighbor not in visited and allowed(int(board[neighbor])):
                    visited.add(neighbor)
                    pending.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _empty_regions(board: np.ndarray, topology: Topology) -> tuple[tuple[int, ...], ...]:
    return _components_matching(board, topology, lambda c: c == EMPTY)


def _pseudolegal_candidate(board: np.ndarray, player: int, action: int, topology: Topology) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    if action < 0 or action >= topology.point_count:
        raise V3IllegalMove("invalid-action")
    if int(board[action]) != EMPTY:
        raise V3IllegalMove("occupied")
    stone = BLACK if player == 0 else WHITE
    opponent = WHITE if stone == BLACK else BLACK
    candidate = np.asarray(board).copy()
    candidate[action] = stone
    captured_groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for neighbor in topology.neighbor_indices(action):
        if int(candidate[neighbor]) != opponent or neighbor in seen:
            continue
        group, liberties = _collect_group(candidate, neighbor, opponent, topology)
        seen.update(group)
        if not liberties:
            captured_groups.append(tuple(sorted(group)))
    for group in captured_groups:
        for point in group:
            candidate[point] = EMPTY
    _, own_liberties = _collect_group(candidate, action, stone, topology)
    if not own_liberties:
        raise V3IllegalMove("suicide")
    return candidate, tuple(captured_groups)


def _is_ko_move(old_board: np.ndarray, new_board: np.ndarray, player: int, action: int, captured_groups: tuple[tuple[int, ...], ...], topology: Topology) -> bool:
    captured_points = [p for group in captured_groups for p in group]
    if len(captured_points) != 1:
        return False
    reply = captured_points[0]
    try:
        reply_board, _ = _pseudolegal_candidate(new_board, 1 - player, reply, topology)
    except V3IllegalMove:
        return False
    return np.array_equal(reply_board, old_board)


def is_simple_ko_state(state: V3State, topology: Topology) -> bool:
    """Return whether ``state`` is immediately recapturable simple ko.

    ``previous_board`` is the board from the immediately preceding accepted
    action snapshot.  The test first uses the two-point shape as a cheap
    prefilter, then applies the opponent's actual capture/suicide transition
    at the only possible recapture point and requires exact restoration of
    that snapshot.  Cleanup ko blocks are intentionally ignored here: this is
    a rule fact about the unblocked local recapture, while cleanup policy
    decides whether the point is currently blocked or can be lifted by
    PASS-for-ko.
    """

    if state.phase not in (MAIN, CLEANUP_1, CLEANUP_2) or state.previous_board is None:
        return False

    before = np.asarray(state.previous_board)
    current = np.asarray(state.board)
    if before.shape != current.shape or before.shape != (topology.point_count,):
        return False

    # A simple ko transition has one newly placed opponent stone and one
    # removed stone. This is only a prefilter; it is not the proof.
    changed = np.flatnonzero(before != current)
    if changed.size != 2:
        return False

    opponent_color = WHITE if state.current_player == 0 else BLACK
    placed_points = [
        int(point)
        for point in changed
        if int(before[point]) == EMPTY and int(current[point]) == opponent_color
    ]
    recapture_points = [
        int(point)
        for point in changed
        if int(before[point]) == (BLACK if state.current_player == 0 else WHITE)
        and int(current[point]) == EMPTY
    ]
    if len(placed_points) != 1 or len(recapture_points) != 1:
        return False

    try:
        recaptured, captured_groups = _pseudolegal_candidate(
            current, state.current_player, recapture_points[0], topology
        )
    except V3IllegalMove:
        # In particular, a false-positive single capture often has a
        # suicide-shaped apparent recapture.
        return False
    captured_points = tuple(point for group in captured_groups for point in group)
    return len(captured_points) == 1 and np.array_equal(recaptured, before)


def _pass_for_ko_unblock_target(state: V3State, action: int, topology: Topology) -> int | None:
    """Return the ko-recap block lifted by this KataGo pass-for-ko action."""

    if state.phase not in (CLEANUP_1, CLEANUP_2):
        return None
    opponent = WHITE if state.current_player == 0 else BLACK
    blocked = set(state.ko_recap_blocked)

    if action in blocked and int(state.board[action]) == opponent:
        group, liberties = _collect_group(state.board, action, opponent, topology)
        if len(group) == 1 and len(liberties) == 1:
            return action

    if action < 0 or action >= topology.point_count or int(state.board[action]) != EMPTY:
        return None
    capture_target = None
    for neighbor in topology.neighbor_indices(action):
        neighbor = int(neighbor)
        if int(state.board[neighbor]) != opponent:
            return None
        group, liberties = _collect_group(state.board, neighbor, opponent, topology)
        if len(liberties) == 1 and action in liberties:
            if capture_target is not None:
                return None
            if len(group) != 1:
                return None
            capture_target = neighbor
    if capture_target is None or capture_target not in blocked:
        return None
    return capture_target


def _unblock_action_legal(state: V3State, action: int, topology: Topology) -> bool:
    return _pass_for_ko_unblock_target(state, action, topology) is not None


def _ko_repeat_forbidden(state: V3State, action: int) -> bool:
    coloring = _board_key(state.board)
    return (state.current_player, action, coloring) in state.ko_capture_history


def _legal_placement(state: V3State, action: int, topology: Topology):
    candidate, captured_groups = _pseudolegal_candidate(state.board, state.current_player, action, topology)
    if state.phase == MAIN:
        if state.previous_board is not None and np.array_equal(candidate, state.previous_board):
            raise V3IllegalMove("simple-ko")
        return candidate, captured_groups, False
    if state.phase not in (CLEANUP_1, CLEANUP_2):
        raise V3IllegalMove("not-playing")
    ko_move = _is_ko_move(state.board, candidate, state.current_player, action, captured_groups, topology)
    if ko_move:
        blocked = set(state.ko_recap_blocked)
        if any(any(point in blocked for point in group) for group in captured_groups):
            raise V3IllegalMove("ko-recapture-blocked")
        if _ko_repeat_forbidden(state, action):
            raise V3IllegalMove("ko-repeat-forbidden")
    return candidate, captured_groups, ko_move


def v3_valid_moves(state: V3State, topology: Topology) -> np.ndarray:
    result = np.zeros(topology.action_size, dtype=np.uint8)
    if state.terminal_kind is not None or state.phase not in (MAIN, CLEANUP_1, CLEANUP_2):
        return result
    for action in range(topology.point_count):
        if _unblock_action_legal(state, action, topology):
            result[action] = 1
            continue
        if int(state.board[action]) != EMPTY:
            continue
        try:
            _legal_placement(state, action, topology)
        except V3IllegalMove:
            continue
        result[action] = 1
    result[topology.pass_action] = 1
    return result


def _phase_reset(state: V3State, phase: str, *, second_start: bytes | None = None) -> V3State:
    blocked: tuple[int, ...] = ()
    key = _state_key(state.board, state.current_player, blocked)
    return replace(
        state,
        phase=phase,
        consecutive_passes=0,
        ko_recap_blocked=blocked,
        phase_history=(key,),
        history_since_pass=(key,),
        black_pass_states=(),
        white_pass_states=(),
        ko_capture_history=(),
        second_cleanup_start_colors=second_start if phase == CLEANUP_2 else state.second_cleanup_start_colors,
        termination_reason=None,
        result_provenance=None,
        entered_cleanup1=state.entered_cleanup1 or phase in (CLEANUP_1, CLEANUP_2),
        entered_cleanup2=state.entered_cleanup2 or phase == CLEANUP_2,
    )


def _cycle_check_and_record(state: V3State, *, after_pass: bool) -> V3State:
    key = _state_key(state.board, state.current_player, state.ko_recap_blocked)
    if not after_pass and state.history_since_pass.count(key) >= 2:
        return replace(
            state,
            phase=NO_RESULT,
            terminal_kind=NO_RESULT,
            no_result_reason=CYCLE,
            termination_reason=CYCLE,
            result_provenance=RESULT_PROVENANCE_RULE_NO_RESULT,
        )
    phase_history = state.phase_history + (key,)
    since_pass = (key,) if after_pass else state.history_since_pass + (key,)
    return replace(state, phase_history=phase_history, history_since_pass=since_pass)


def _finish_phase_after_pass(state: V3State) -> V3State:
    if state.phase == MAIN:
        return _phase_reset(state, CLEANUP_1)
    if state.phase == CLEANUP_1:
        return _phase_reset(state, CLEANUP_2, second_start=_board_key(state.board))
    if state.phase == CLEANUP_2:
        return replace(
            state,
            phase=SCORED,
            terminal_kind=SCORED,
            termination_reason=FORMAL_PASS,
            result_provenance=RESULT_PROVENANCE_FORMAL,
        )
    return state


def _pass(state: V3State, topology: Topology) -> V3State:
    pre_key = _state_key(state.board, state.current_player, state.ko_recap_blocked)
    pass_states = state.black_pass_states if state.current_player == 0 else state.white_pass_states
    repeated_pass_state = pre_key in pass_states
    black_pass = state.black_pass_states + (pre_key,) if state.current_player == 0 else state.black_pass_states
    white_pass = state.white_pass_states + (pre_key,) if state.current_player == 1 else state.white_pass_states
    next_state = replace(
        state,
        current_player=1 - state.current_player,
        turns=state.turns + 1,
        consecutive_passes=state.consecutive_passes + 1,
        previous_board=state.board,
        black_pass_states=black_pass,
        white_pass_states=white_pass,
    )
    if next_state.consecutive_passes >= 2 or repeated_pass_state:
        return _finish_phase_after_pass(next_state)
    return _cycle_check_and_record(next_state, after_pass=True)


def _unblock(state: V3State, action: int, topology: Topology) -> V3State:
    target = _pass_for_ko_unblock_target(state, action, topology)
    if target is None:
        raise V3IllegalMove("invalid-pass-for-ko")
    blocked = tuple(p for p in state.ko_recap_blocked if p != target)
    next_state = replace(
        state,
        current_player=1 - state.current_player,
        turns=state.turns + 1,
        consecutive_passes=0,
        previous_board=state.board,
        ko_recap_blocked=blocked,
        ko_unblock_actions=state.ko_unblock_actions + 1,
    )
    return _cycle_check_and_record(next_state, after_pass=False)


def _placement(state: V3State, action: int, topology: Topology) -> V3State:
    candidate, captured_groups, ko_move = _legal_placement(state, action, topology)
    captured = sum(len(group) for group in captured_groups)
    captures = list(state.captures)
    captures[state.current_player] += captured
    blocked = set(state.ko_recap_blocked)
    ko_history = state.ko_capture_history
    if state.phase in (CLEANUP_1, CLEANUP_2):
        if ko_move:
            blocked.add(action)
            ko_history = ko_history + ((state.current_player, action, _board_key(state.board)),)
        blocked = {p for p in blocked if int(candidate[p]) != EMPTY}
    else:
        blocked.clear()
    cleanup2_moves = list(state.cleanup2_moves)
    main_moves = list(state.main_moves)
    cleanup1_moves = list(state.cleanup1_moves)
    white_bonus_score = state.white_bonus_score
    if state.phase == MAIN:
        main_moves[state.current_player] += 1
        white_bonus_score += 1.0 if state.current_player == 0 else -1.0
    elif state.phase == CLEANUP_1:
        cleanup1_moves[state.current_player] += 1
        white_bonus_score += 1.0 if state.current_player == 0 else -1.0
    elif state.phase == CLEANUP_2:
        cleanup2_moves[state.current_player] += 1
    next_state = replace(
        state,
        board=_readonly_board(candidate, topology.point_count),
        current_player=1 - state.current_player,
        turns=state.turns + 1,
        consecutive_passes=0,
        captures=(captures[0], captures[1]),
        white_bonus_score=white_bonus_score,
        previous_board=state.board,
        ko_recap_blocked=tuple(sorted(blocked)),
        ko_capture_history=ko_history,
        cleanup2_moves=(cleanup2_moves[0], cleanup2_moves[1]),
        main_moves=(main_moves[0], main_moves[1]),
        cleanup1_moves=(cleanup1_moves[0], cleanup1_moves[1]),
        cleanup_captures=state.cleanup_captures + (captured if state.phase in (CLEANUP_1, CLEANUP_2) else 0),
    )
    return _cycle_check_and_record(next_state, after_pass=False)


def apply_v3_action(state: V3State, action: int, topology: Topology) -> V3State:
    if state.terminal_kind is not None or state.phase not in (MAIN, CLEANUP_1, CLEANUP_2):
        raise V3IllegalMove("not-playing")
    if action == topology.pass_action:
        next_state = _pass(state, topology)
    elif _unblock_action_legal(state, action, topology):
        next_state = _unblock(state, action, topology)
    else:
        next_state = _placement(state, action, topology)
    return next_state


def _benson_pass_alive_groups(board: np.ndarray, topology: Topology, color: int) -> tuple[tuple[int, ...], ...]:
    groups = _all_groups(board, topology, color)
    if not groups:
        return ()
    owner: dict[int, int] = {}
    for gi, group in enumerate(groups):
        for p in group:
            owner[p] = gi

    regions: list[tuple[tuple[int, ...], set[int], set[int]]] = []
    for region in _components_matching(board, topology, lambda c, own=color: c != own):
        boundary = {
            owner[n]
            for p in region
            for n in topology.neighbor_indices(p)
            if n in owner
        }
        if not boundary:
            continue

        vital = set(boundary)
        for p in region:
            if int(board[p]) != EMPTY:
                continue
            adjacent = {owner[n] for n in topology.neighbor_indices(p) if n in owner}
            vital.intersection_update(adjacent)
        regions.append((region, boundary, vital))

    remaining_groups = set(range(len(groups)))
    remaining_regions = set(range(len(regions)))
    while True:
        remove_groups = {gi for gi in remaining_groups if sum(gi in regions[ri][2] for ri in remaining_regions) < 2}
        remaining_groups -= remove_groups
        remove_regions = {ri for ri in remaining_regions if any(gi not in remaining_groups for gi in regions[ri][1])}
        remaining_regions -= remove_regions
        if not remove_groups and not remove_regions:
            break
    return tuple(groups[gi] for gi in sorted(remaining_groups))


def _pass_alive_territory_for_color(board: np.ndarray, topology: Topology, color: int, pass_alive_groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    alive_points = {p for group in pass_alive_groups for p in group}
    territory: set[int] = set()
    for region in _components_matching(board, topology, lambda c: c != color):
        bordering_own_groups: set[int] = set()
        all_border_alive = True
        for p in region:
            for n in topology.neighbor_indices(p):
                if int(board[n]) == color:
                    bordering_own_groups.add(n)
                    if n not in alive_points:
                        all_border_alive = False
        if not bordering_own_groups or not all_border_alive:
            continue
        adjacent_count = sum(any(n in alive_points for n in topology.neighbor_indices(p)) for p in region)
        if adjacent_count >= len(region) - 1:
            territory.update(region)
    return tuple(sorted(territory))


def pass_alive_analysis(board: np.ndarray, topology: Topology) -> PassAliveAnalysis:
    black_groups = _benson_pass_alive_groups(board, topology, BLACK)
    white_groups = _benson_pass_alive_groups(board, topology, WHITE)
    black_territory = _pass_alive_territory_for_color(board, topology, BLACK, black_groups)
    white_territory = _pass_alive_territory_for_color(board, topology, WHITE, white_groups)
    return PassAliveAnalysis(black_groups, white_groups, black_territory, white_territory)


def all_points_pass_alive(board: np.ndarray, topology: Topology) -> bool:
    return len(pass_alive_analysis(board, topology).covered_points) == topology.point_count


def independent_life_analysis(board: np.ndarray, topology: Topology) -> IndependentLifeAnalysis:
    # KataGo first builds basic area from Benson/pass-alive groups and safe or
    # unsafe territories, then fills any still-unassigned stone with its own
    # color.  This is intentionally separate from the final score pass: a
    # dead opposing stone can be part of a color's basic area without being a
    # scored independent-life point.
    pass_alive = pass_alive_analysis(board, topology)
    basic_area = np.zeros(topology.point_count, dtype=np.uint8)
    for color, groups, territory in (
        (BLACK, pass_alive.pass_alive_black_groups, pass_alive.pass_alive_black_territory),
        (WHITE, pass_alive.pass_alive_white_groups, pass_alive.pass_alive_white_territory),
    ):
        for group in groups:
            basic_area[list(group)] = color
        basic_area[list(territory)] = color
    # Rules V3 also asks KataGo for unsafe large territories. An empty
    # component with no opposing stone and at least one bordering stone is
    # assigned to that color even when the bordering group is not Benson
    # pass-alive; the independent-life pass below decides whether it is seki.
    for color in (BLACK, WHITE):
        opponent = WHITE if color == BLACK else BLACK
        for component in _components_matching(
            board, topology, lambda value, own=color: value != own
        ):
            points = set(component)
            if any(int(board[point]) == opponent for point in points):
                continue
            if not any(
                int(board[neighbor]) == color
                for point in points
                for neighbor in topology.neighbor_indices(point)
            ):
                continue
            for point in points:
                if basic_area[point] == EMPTY:
                    basic_area[point] = color
    for point, value in enumerate(np.asarray(board).reshape(-1)):
        if basic_area[point] == EMPTY and int(value) != EMPTY:
            basic_area[point] = int(value)

    seki_components: set[frozenset[int]] = set()
    for color in (BLACK, WHITE):
        for component in _components_matching(
            basic_area, topology, lambda value, expected=color: value == expected
        ):
            component_points = set(component)
            is_seki = False
            for point in component_points:
                if int(board[point]) == color:
                    _, liberties = _collect_group(board, point, color, topology)
                    if len(liberties) == 1:
                        is_seki = True
                        break
                if any(
                    int(board[neighbor]) == EMPTY and basic_area[neighbor] == EMPTY
                    for neighbor in topology.neighbor_indices(point)
                ):
                    is_seki = True
                    break
            if is_seki:
                seki_components.add(frozenset(component_points))

    black_regions: list[tuple[int, ...]] = []
    white_regions: list[tuple[int, ...]] = []
    black_area: set[int] = set()
    white_area: set[int] = set()
    for color, sink, area in (
        (BLACK, black_regions, black_area),
        (WHITE, white_regions, white_area),
    ):
        for component in _components_matching(
            basic_area, topology, lambda value, expected=color: value == expected
        ):
            if frozenset(component) in seki_components:
                continue
            sink.append(component)
            area.update(component)

    black_territory = {p for p in black_area if int(board[p]) == EMPTY}
    white_territory = {p for p in white_area if int(board[p]) == EMPTY}
    dame = {
        p for p in range(topology.point_count)
        if int(board[p]) == EMPTY and basic_area[p] == EMPTY
    }
    seki = {
        p for component in seki_components for p in component
        if int(board[p]) == EMPTY
    }
    return IndependentLifeAnalysis(
        tuple(sorted(black_area)), tuple(sorted(white_area)), tuple(black_regions), tuple(white_regions),
        tuple(sorted(black_territory)), tuple(sorted(white_territory)), tuple(sorted(dame)), tuple(sorted(seki)),
    )


def _legacy_board_only_life_analysis(board: np.ndarray, topology: Topology) -> IndependentLifeAnalysis:
    """LEGACY/TEST ONLY: reproduce the pre-S1 board-only scoring view.

    Production scoring must call :func:`independent_life_analysis` directly.
    Keeping this helper named and isolated lets historical-audit tooling
    quantify the old result without allowing telemetry to select it.
    """

    empty_regions = _empty_regions(board, topology)
    dame_points: set[int] = set()
    for region in empty_regions:
        colors = {
            int(board[neighbor])
            for point in region
            for neighbor in topology.neighbor_indices(point)
            if int(board[neighbor]) != EMPTY
        }
        if colors == {BLACK, WHITE}:
            dame_points.update(region)
    atari_points: set[int] = set()
    for group in _all_groups(board, topology):
        liberties = {
            neighbor
            for point in group
            for neighbor in topology.neighbor_indices(point)
            if int(board[neighbor]) == EMPTY
        }
        if len(liberties) == 1:
            atari_points.update(group)
    black_regions: list[tuple[int, ...]] = []
    white_regions: list[tuple[int, ...]] = []
    for color, opponent, sink in ((BLACK, WHITE, black_regions), (WHITE, BLACK, white_regions)):
        for component in _components_matching(
            board, topology, lambda value, opponent=opponent: value != opponent
        ):
            points = set(component)
            if points & dame_points or points & atari_points:
                continue
            if not any(int(board[point]) == color for point in component):
                continue
            sink.append(component)
    black_area = {point for region in black_regions for point in region}
    white_area = {point for region in white_regions for point in region}
    black_territory = {point for point in black_area if int(board[point]) == EMPTY}
    white_territory = {point for point in white_area if int(board[point]) == EMPTY}
    assigned_empty = black_territory | white_territory
    remaining_empty = {
        point
        for point in range(topology.point_count)
        if int(board[point]) == EMPTY and point not in assigned_empty
    }
    seki: set[int] = set()
    neutral: set[int] = set()
    for region in empty_regions:
        region_points = set(region)
        if not region_points & remaining_empty:
            continue
        colors = {
            int(board[neighbor])
            for point in region
            for neighbor in topology.neighbor_indices(point)
            if int(board[neighbor]) != EMPTY
        }
        if colors == {BLACK, WHITE}:
            neutral.update(region_points)
        elif colors:
            seki.update(region_points)
        else:
            neutral.update(region_points)
    return IndependentLifeAnalysis(
        tuple(sorted(black_area)), tuple(sorted(white_area)), tuple(black_regions), tuple(white_regions),
        tuple(sorted(black_territory)), tuple(sorted(white_territory)), tuple(sorted(neutral)), tuple(sorted(seki)),
    )


def final_v3_score(state: V3State, topology: Topology, komi: float) -> tuple[FinalScore, np.ndarray, np.ndarray]:
    """Score one formal V3 state using the single production semantics.

    The pinned implementation separates board-area scoring from the
    score-relevant ``whiteBonusScore`` initialized by ``BoardHistory::clear``.
    Setup and replay therefore follow the same path.  In particular,
    ``main_moves`` is retained as telemetry only and must never select a
    scoring algorithm.
    """

    start_colors = (
        _board_key(state.board)
        if state.second_cleanup_start_colors is None
        else state.second_cleanup_start_colors
    )
    board, captures = state.board, state.captures
    life = independent_life_analysis(board, topology)
    black_area = set(life.black_area)
    white_area = set(life.white_area)
    black_score = float(len(black_area))
    white_score = float(len(white_area))
    # ``calculateIndependentLifeArea`` supplies the formal board area.  The
    # remaining stones clause is the same guard used by KataGo's territory
    # scorer: before encore 2 all remaining stones count; in encore 2 only
    # stones that were present at its start count.
    encore2 = state.second_cleanup_start_colors is not None
    for point, value in enumerate(np.asarray(board).reshape(-1)):
        color = int(value)
        if (
            color == BLACK
            and point not in black_area
            and point not in white_area
            and (not encore2 or start_colors[point] == BLACK)
        ):
            black_score += 1.0
        elif (
            color == WHITE
            and point not in white_area
            and point not in black_area
            and (not encore2 or start_colors[point] == WHITE)
        ):
            white_score += 1.0
    formal_black = set(black_area)
    formal_white = set(white_area)
    for point, value in enumerate(np.asarray(board).reshape(-1)):
        color = int(value)
        if color == BLACK and point not in formal_white and (
            point in formal_black or not encore2 or start_colors[point] == BLACK
        ):
            formal_black.add(point)
        elif color == WHITE and point not in formal_black and (
            point in formal_white or not encore2 or start_colors[point] == WHITE
        ):
            formal_white.add(point)
    # Store the white-oriented offset on the state, just as KataGo stores it
    # on BoardHistory.  The public FinalScore keeps the existing breakdown
    # convention: the offset is represented on white so that white-black is
    # authoritative; it is not a territory/prisoner decomposition.
    white_score += float(state.white_bonus_score)
    white_score += float(komi)
    territory = TerritoryBreakdown(black=len(life.black_territory), white=len(life.white_territory), neutral=len(life.dame), seki=len(life.seki))
    territory_points = TerritoryPoints(black=life.black_territory, white=life.white_territory, neutral=life.dame, seki=life.seki)
    winner = "draw" if black_score == white_score else ("black" if black_score > white_score else "white")
    score = FinalScore(
        ruleset="japanese", black=black_score, white=white_score, komi=float(komi), territory=territory,
        territory_points=territory_points,
        stones_on_board=StoneBreakdown(int(np.count_nonzero(board == BLACK)), int(np.count_nonzero(board == WHITE))),
        captures=captures, prisoners=captures, dead_stones=StoneBreakdown(0, 0), winner=winner, margin=abs(black_score - white_score),
    )
    labels = np.full(topology.point_count, 2, dtype=np.int64)
    for p in formal_black:
        labels[p] = 0
    for p in formal_white:
        labels[p] = 1
    ownership = np.zeros((topology.point_count, 3), dtype=np.float32)
    ownership[np.arange(topology.point_count), labels] = 1.0
    ownership_mask = np.ones(topology.point_count, dtype=np.float32)
    return score, ownership, ownership_mask


def terminal_from_state(state: V3State, topology: Topology, komi: float) -> V3Terminal | None:
    if state.terminal_kind == NO_RESULT:
        return V3Terminal(
            NO_RESULT,
            None,
            None,
            None,
            state.termination_reason or state.no_result_reason,
            state.result_provenance or RESULT_PROVENANCE_RULE_NO_RESULT,
        )
    if state.terminal_kind != SCORED:
        return None
    score, ownership, ownership_mask = final_v3_score(state, topology, komi)
    return V3Terminal(
        SCORED,
        score,
        ownership,
        ownership_mask,
        state.termination_reason,
        state.result_provenance or RESULT_PROVENANCE_FORMAL,
    )


def normalized_score_target_v3(terminal: V3Terminal, topology: Topology) -> np.ndarray:
    if not terminal.training_valid or terminal.score is None:
        raise ValueError("Score target requires terminal_kind == SCORED")
    signed = terminal.score.black - terminal.score.white
    return np.asarray([np.clip(signed / topology.point_count, -1.0, 1.0)], dtype=np.float32)


def build_v3_training_targets(
    terminal: V3Terminal,
    side_to_move: int,
    topology: Topology,
) -> V3TrainingTargets:
    """Build every training label from terminal semantics in one place.

    The score remains the canonical black-minus-white normalized value used by
    the existing score head.  Value labels are converted to the player-to-move
    perspective for the individual position being saved.
    """

    if not terminal.value_target_valid:
        raise ValueError(f"Unsupported terminal kind for value training: {terminal.terminal_kind!r}")
    side_to_move = int(side_to_move)
    if side_to_move not in (0, 1):
        raise ValueError(f"side_to_move must be 0 (black) or 1 (white), got {side_to_move}")

    value = np.zeros(VALUE_TARGET_SIZE, dtype=np.float32)
    if terminal.terminal_kind == NO_RESULT:
        value[VALUE_NO_RESULT] = 1.0
        score = np.asarray([np.nan], dtype=np.float32)
        score_mask = np.zeros(1, dtype=np.float32)
        ownership = np.zeros((topology.point_count, 3), dtype=np.float32)
        ownership_mask = np.zeros(topology.point_count, dtype=np.float32)
    else:
        assert terminal.score is not None
        if terminal.score.winner == "draw":
            value[VALUE_WIN] = 0.5
            value[VALUE_LOSS] = 0.5
        else:
            winner = 0 if terminal.score.winner == "black" else 1
            value[VALUE_WIN if winner == side_to_move else VALUE_LOSS] = 1.0
        score = normalized_score_target_v3(terminal, topology)
        score_mask = np.ones(1, dtype=np.float32)
        if terminal.ownership is None or terminal.ownership_mask is None:
            raise ValueError("SCORED terminal is missing ownership targets")
        ownership = np.asarray(terminal.ownership, dtype=np.float32).copy()
        ownership_mask = np.asarray(terminal.ownership_mask, dtype=np.float32).copy()

    result_provenance = terminal.result_provenance
    if result_provenance is None:
        result_provenance = (
            RESULT_PROVENANCE_RULE_NO_RESULT
            if terminal.terminal_kind == NO_RESULT
            else RESULT_PROVENANCE_FORMAL
        )

    return V3TrainingTargets(
        value_target=value,
        score_target=score,
        score_mask=score_mask,
        ownership_target=ownership,
        ownership_mask=ownership_mask,
        terminal_kind=terminal.terminal_kind,
        termination_reason=terminal.reason,
        result_provenance=result_provenance,
    )


def maybe_pass_alive_early_terminal(state: V3State, topology: Topology) -> V3State:
    if state.phase != MAIN or state.terminal_kind is not None:
        return state
    if all_points_pass_alive(state.board, topology):
        return replace(
            state,
            phase=SCORED,
            terminal_kind=SCORED,
            termination_reason=PASS_ALIVE,
            result_provenance=RESULT_PROVENANCE_FORMAL,
            second_cleanup_start_colors=_board_key(state.board),
            pass_alive_early_end=True,
        )
    return state


def ko_repeat_forbidden_mask(state: V3State, topology: Topology) -> np.ndarray:
    mask = np.zeros(topology.point_count, dtype=np.float32)
    if state.phase not in (CLEANUP_1, CLEANUP_2):
        return mask
    coloring = _board_key(state.board)
    for player, action, prior_coloring in state.ko_capture_history:
        if player == state.current_player and prior_coloring == coloring:
            mask[action] = 1.0
    return mask


def repetition_pressure(state: V3State) -> float:
    key = _state_key(state.board, state.current_player, state.ko_recap_blocked)
    return min(1.0, state.history_since_pass.count(key) / 2.0)
