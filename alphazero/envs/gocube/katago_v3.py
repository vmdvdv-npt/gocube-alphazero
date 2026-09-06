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

KATAGO_JAPANESE_ADJUDICATOR_V3 = "gocube-katago-japanese-v3"
OBSERVATION_SCHEMA_V3 = "gocube-observation-v3"
KATAGO_RULES_VERSION = 3
KATAGO_RULES_IMPLEMENTATION_VERSION = 3
KATAGO_REFERENCE_COMMIT = "f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
KATAGO_REFERENCE_VERSION = "1.18.0+ Rules Version 3"

EMERGENCY_MOVE_CAP_FACTOR = 24
EMERGENCY_MOVE_CAP_BASE = 256


def rules_fingerprint(topology: Topology, komi: float = 7.5) -> str:
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

    @property
    def training_valid(self) -> bool:
        return self.terminal_kind == SCORED and self.score is not None

    @property
    def winner(self) -> str:
        if self.score is None:
            return "draw"
        return self.score.winner

    @property
    def no_result(self) -> bool:
        return self.terminal_kind == NO_RESULT


class V3IllegalMove(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def initial_v3_state(topology: Topology) -> V3State:
    board = _readonly_board(np.zeros(topology.point_count, dtype=np.uint8), topology.point_count)
    key = _state_key(board, 0, ())
    return V3State(board=board, phase_history=(key,), history_since_pass=(key,))


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
    return V3State(
        board=board,
        current_player=current_player,
        turns=turns,
        captures=captures,
        previous_board=prev,
        phase=phase,
        ko_recap_blocked=blocked,
        phase_history=(key,),
        history_since_pass=(key,),
        second_cleanup_start_colors=second_cleanup_start_colors,
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


def _pass_for_ko_unblock_target(state: V3State, action: int, topology: Topology) -> int | None:
    """Return the ko-recap block lifted by this KataGo pass-for-ko action."""

    if state.phase not in (CLEANUP_1, CLEANUP_2):
        return None
    opponent = WHITE if state.current_player == 0 else BLACK
    blocked = set(state.ko_recap_blocked)

    # KataGo path 1: play on the blocked opponent stone itself. The move is a
    # pass-for-ko rather than a board placement when that stone is a one-stone
    # chain in atari.
    if action in blocked and int(state.board[action]) == opponent:
        group, liberties = _collect_group(state.board, action, opponent, topology)
        if len(group) == 1 and len(liberties) == 1:
            return action

    # KataGo path 2: play on the empty ko-capture point whose capture target is
    # a blocked opponent stone. Board::getKoCaptureLoc requires every on-board
    # neighbor to be opponent-colored and exactly one capturable one-stone chain.
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
        entered_cleanup1=state.entered_cleanup1 or phase in (CLEANUP_1, CLEANUP_2),
        entered_cleanup2=state.entered_cleanup2 or phase == CLEANUP_2,
    )


def _cycle_check_and_record(state: V3State, *, after_pass: bool) -> V3State:
    key = _state_key(state.board, state.current_player, state.ko_recap_blocked)
    if not after_pass and state.history_since_pass.count(key) >= 2:
        return replace(state, phase=NO_RESULT, terminal_kind=NO_RESULT, no_result_reason="cycle")
    phase_history = state.phase_history + (key,)
    since_pass = (key,) if after_pass else state.history_since_pass + (key,)
    return replace(state, phase_history=phase_history, history_since_pass=since_pass)


def _finish_phase_after_pass(state: V3State) -> V3State:
    if state.phase == MAIN:
        return _phase_reset(state, CLEANUP_1)
    if state.phase == CLEANUP_1:
        return _phase_reset(state, CLEANUP_2, second_start=_board_key(state.board))
    if state.phase == CLEANUP_2:
        return replace(state, phase=SCORED, terminal_kind=SCORED)
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
    if state.phase == MAIN:
        main_moves[state.current_player] += 1
    elif state.phase == CLEANUP_1:
        cleanup1_moves[state.current_player] += 1
    elif state.phase == CLEANUP_2:
        cleanup2_moves[state.current_player] += 1
    next_state = replace(
        state,
        board=_readonly_board(candidate, topology.point_count),
        current_player=1 - state.current_player,
        turns=state.turns + 1,
        consecutive_passes=0,
        captures=(captures[0], captures[1]),
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
    if next_state.terminal_kind is None and next_state.turns >= EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * topology.point_count:
        return replace(next_state, phase=NO_RESULT, terminal_kind=NO_RESULT, no_result_reason="move-cap")
    return next_state


def _benson_pass_alive_groups(board: np.ndarray, topology: Topology, color: int) -> tuple[tuple[int, ...], ...]:
    groups = _all_groups(board, topology, color)
    if not groups:
        return ()
    owner: dict[int, int] = {}
    for gi, group in enumerate(groups):
        for p in group:
            owner[p] = gi

    # Benson regions are maximal connected components that contain no stone of
    # the color being proved alive. They may therefore contain both EMPTY and
    # opponent stones. This is the graph-topology form of the standard Benson
    # construction and deliberately makes no planar edge/corner assumptions.
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

        # A region is vital for a chain iff every EMPTY intersection in that
        # region is a liberty of the chain. Opponent stones are part of the
        # region but are not themselves intersections that must border the
        # chain, and their presence does not invalidate the region.
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
    empty_regions = _empty_regions(board, topology)
    dame_points: set[int] = set()
    for region in empty_regions:
        colors: set[int] = set()
        for p in region:
            for n in topology.neighbor_indices(p):
                c = int(board[n])
                if c != EMPTY:
                    colors.add(c)
        if colors == {BLACK, WHITE}:
            dame_points.update(region)
    atari_points: set[int] = set()
    for group in _all_groups(board, topology):
        liberties: set[int] = set()
        for p in group:
            for n in topology.neighbor_indices(p):
                if int(board[n]) == EMPTY:
                    liberties.add(n)
        if len(liberties) == 1:
            atari_points.update(group)
    black_regions: list[tuple[int, ...]] = []
    white_regions: list[tuple[int, ...]] = []
    for color, opponent, sink in ((BLACK, WHITE, black_regions), (WHITE, BLACK, white_regions)):
        for component in _components_matching(board, topology, lambda c, opp=opponent: c != opp):
            points = set(component)
            if points & dame_points or points & atari_points:
                continue
            if not any(int(board[p]) == color for p in component):
                continue
            sink.append(component)
    black_area = {p for region in black_regions for p in region}
    white_area = {p for region in white_regions for p in region}
    black_territory = {p for p in black_area if int(board[p]) == EMPTY}
    white_territory = {p for p in white_area if int(board[p]) == EMPTY}
    assigned_empty = black_territory | white_territory
    remaining_empty = {p for p in range(topology.point_count) if int(board[p]) == EMPTY and p not in assigned_empty}
    seki: set[int] = set()
    neutral: set[int] = set()
    for region in empty_regions:
        rset = set(region)
        if not (rset & remaining_empty):
            continue
        colors = {int(board[n]) for p in region for n in topology.neighbor_indices(p) if int(board[n]) != EMPTY}
        if colors == {BLACK, WHITE}:
            neutral.update(rset)
        elif colors:
            seki.update(rset)
        else:
            neutral.update(rset)
    return IndependentLifeAnalysis(
        tuple(sorted(black_area)), tuple(sorted(white_area)), tuple(black_regions), tuple(white_regions),
        tuple(sorted(black_territory)), tuple(sorted(white_territory)), tuple(sorted(neutral)), tuple(sorted(seki)),
    )


def _selfplay_remove_stones_in_opponent_pass_alive_territory(board: np.ndarray, captures: tuple[int, int], topology: Topology) -> tuple[np.ndarray, tuple[int, int]]:
    analysis = pass_alive_analysis(board, topology)
    black_pat = set(analysis.pass_alive_black_territory)
    white_pat = set(analysis.pass_alive_white_territory)
    result = np.asarray(board).copy()
    caps = [captures[0], captures[1]]
    remove_black = [p for p in white_pat if int(board[p]) == BLACK]
    remove_white = [p for p in black_pat if int(board[p]) == WHITE]
    for p in remove_black:
        result[p] = EMPTY
        caps[1] += 1
    for p in remove_white:
        result[p] = EMPTY
        caps[0] += 1
    return _readonly_board(result, topology.point_count), (caps[0], caps[1])


def final_v3_score(state: V3State, topology: Topology, komi: float) -> tuple[FinalScore, np.ndarray, np.ndarray]:
    start_colors = _board_key(state.board) if state.second_cleanup_start_colors is None else state.second_cleanup_start_colors
    board, captures = _selfplay_remove_stones_in_opponent_pass_alive_territory(state.board, state.captures, topology)
    life = independent_life_analysis(board, topology)
    black_area = set(life.black_area)
    white_area = set(life.white_area)
    penalties = [0, 0]
    for p in range(topology.point_count):
        c = int(board[p])
        if c == BLACK and p not in black_area and start_colors[p] != BLACK:
            penalties[0] += 1
        elif c == WHITE and p not in white_area and start_colors[p] != WHITE:
            penalties[1] += 1
    territory = TerritoryBreakdown(black=len(life.black_territory), white=len(life.white_territory), neutral=len(life.dame), seki=len(life.seki))
    territory_points = TerritoryPoints(black=life.black_territory, white=life.white_territory, neutral=life.dame, seki=life.seki)
    black_score = float(len(life.black_territory) + captures[0] + state.cleanup2_moves[0] - penalties[0])
    white_score = float(len(life.white_territory) + captures[1] + state.cleanup2_moves[1] - penalties[1] + komi)
    winner = "draw" if black_score == white_score else ("black" if black_score > white_score else "white")
    score = FinalScore(
        ruleset="japanese", black=black_score, white=white_score, komi=float(komi), territory=territory,
        territory_points=territory_points,
        stones_on_board=StoneBreakdown(int(np.count_nonzero(board == BLACK)), int(np.count_nonzero(board == WHITE))),
        captures=captures, prisoners=captures, dead_stones=StoneBreakdown(0, 0), winner=winner, margin=abs(black_score - white_score),
    )
    labels = np.full(topology.point_count, 2, dtype=np.int64)
    for p in life.black_area:
        if int(board[p]) == BLACK or p in life.black_territory:
            labels[p] = 0
    for p in life.white_area:
        if int(board[p]) == WHITE or p in life.white_territory:
            labels[p] = 1
    ownership = np.zeros((topology.point_count, 3), dtype=np.float32)
    ownership[np.arange(topology.point_count), labels] = 1.0
    ownership_mask = np.ones(topology.point_count, dtype=np.float32)
    return score, ownership, ownership_mask


def terminal_from_state(state: V3State, topology: Topology, komi: float) -> V3Terminal | None:
    if state.terminal_kind == NO_RESULT:
        return V3Terminal(NO_RESULT, None, None, None, state.no_result_reason)
    if state.terminal_kind != SCORED:
        return None
    score, ownership, ownership_mask = final_v3_score(state, topology, komi)
    return V3Terminal(SCORED, score, ownership, ownership_mask)


def normalized_score_target_v3(terminal: V3Terminal, topology: Topology) -> np.ndarray:
    if not terminal.training_valid or terminal.score is None:
        raise ValueError("Score target requires terminal_kind == SCORED")
    signed = terminal.score.black - terminal.score.white
    return np.asarray([np.clip(signed / topology.point_count, -1.0, 1.0)], dtype=np.float32)


def maybe_pass_alive_early_terminal(state: V3State, topology: Topology) -> V3State:
    if state.phase != MAIN or state.terminal_kind is not None:
        return state
    if all_points_pass_alive(state.board, topology):
        return replace(
            state,
            phase=SCORED,
            terminal_kind=SCORED,
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
