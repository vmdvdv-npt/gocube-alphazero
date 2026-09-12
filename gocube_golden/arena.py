from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from .arena_contract import (
    ARENA_CONTRACT_ID,
    DEFAULT_ARENA_CONTRACT,
    GOLDEN_MOVE_LIMIT,
    GoldenArenaContract,
)
from .players import Player, PlayerContext, derive_child_seed
from .result import DOUBLE_PASS, Winner, result_from_terminal
from .rules import IllegalMoveError, apply_action
from .search import SearchError
from .state import (
    BLACK,
    WHITE,
    GoldenState,
    initial_state,
)
from .topology import TORUS_5X5, TORUS_5X5_TOPOLOGY_FINGERPRINT

class PlayerSlot(str, Enum):
    A = "A"
    B = "B"

class MappedResult(str, Enum):
    A_WIN = "A_WIN"
    B_WIN = "B_WIN"
    DRAW = "DRAW"

class TerminationReason(str, Enum):
    DOUBLE_PASS = DOUBLE_PASS
    TRUNCATED_MOVE_LIMIT = "TRUNCATED_MOVE_LIMIT"
    ERROR_ILLEGAL_PLAYER_ACTION = "ERROR_ILLEGAL_PLAYER_ACTION"
    ERROR_PLAYER_EXCEPTION = "ERROR_PLAYER_EXCEPTION"
    ERROR_SEARCH = "ERROR_SEARCH"

TECHNICAL_TERMINATIONS = frozenset({
    TerminationReason.TRUNCATED_MOVE_LIMIT,
    TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION,
    TerminationReason.ERROR_PLAYER_EXCEPTION,
    TerminationReason.ERROR_SEARCH,
})

@dataclass(frozen=True)
class ActionEvidence:
    ply: int
    side_to_move: str
    player_slot: str
    player_id: str
    action: int | str
    legal: bool
    error: str | None = None

@dataclass(frozen=True)
class GameRecord:
    game_id: str
    pair_id: str
    rules_fingerprint: str
    topology_fingerprint: str
    komi: float
    start_state_key: tuple[object, ...]
    start_history: tuple[tuple[int, ...], ...]
    start_trace: tuple[int | str, ...]
    player_A_id: str
    player_B_id: str
    black_player: str
    white_player: str
    seed_game: int
    seed_A: int
    seed_B: int
    search_contract_id: str
    search_settings: tuple[tuple[str, object], ...]
    action_trace: tuple[ActionEvidence, ...]
    final_board: tuple[int, ...]
    absolute_rule_result: Winner | None
    mapped_result: MappedResult | None
    black_area: int | None
    white_area: int | None
    margin_black: float | None
    termination_reason: TerminationReason
    error_details: str | None

    @property
    def is_rule_result(self) -> bool:
        return self.termination_reason == TerminationReason.DOUBLE_PASS

    @property
    def is_technical(self) -> bool:
        return self.termination_reason in TECHNICAL_TERMINATIONS

@dataclass(frozen=True)
class PairSummary:
    pair_id: str
    game_ids: tuple[str, ...]
    a_wins: int
    b_wins: int
    draws: int
    technical_failures: int

@dataclass(frozen=True)
class ArenaSummary:
    games: int
    rule_results: int
    a_wins: int
    b_wins: int
    draws: int
    technical_failures: int
    a_as_black: int
    b_as_black: int
    technical_by_reason: tuple[tuple[str, int], ...]
    paired_results: tuple[PairSummary, ...]

def map_absolute_result(winner: Winner, *, black_player: str) -> MappedResult:
    if black_player not in (PlayerSlot.A.value, PlayerSlot.B.value):
        raise ValueError(f"black_player must be A or B, got {black_player!r}")
    if winner == Winner.DRAW:
        return MappedResult.DRAW
    if winner == Winner.BLACK:
        return MappedResult.A_WIN if black_player == PlayerSlot.A.value else MappedResult.B_WIN
    if winner == Winner.WHITE:
        return MappedResult.B_WIN if black_player == PlayerSlot.A.value else MappedResult.A_WIN
    raise ValueError(f"Unknown absolute Golden winner {winner!r}")

def post_action_termination(state: GoldenState, action_count: int) -> TerminationReason | None:
    """Resolve formal rule terminal before the execution watchdog."""
    if state.is_terminal:
        return TerminationReason.DOUBLE_PASS
    if action_count >= GOLDEN_MOVE_LIMIT:
        return TerminationReason.TRUNCATED_MOVE_LIMIT
    return None

def _derive_seed(master_seed: int, *parts: object) -> int:
    text = ":".join([str(int(master_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")

def _slot_for_side(side, black_player: str) -> str:
    if black_player == PlayerSlot.A.value:
        return PlayerSlot.A.value if side == BLACK else PlayerSlot.B.value
    if black_player == PlayerSlot.B.value:
        return PlayerSlot.B.value if side == BLACK else PlayerSlot.A.value
    raise ValueError("black_player must be A or B")

def _player_id_for_slot(record_or_ids, slot: str) -> str:
    if isinstance(record_or_ids, GameRecord):
        return record_or_ids.player_A_id if slot == PlayerSlot.A.value else record_or_ids.player_B_id
    return record_or_ids[0] if slot == PlayerSlot.A.value else record_or_ids[1]

def _replay_start(record: GameRecord) -> GoldenState:
    state = initial_state(topology=TORUS_5X5, komi=record.komi)
    for action in record.start_trace:
        state = apply_action(state, action).after
    return state

def validate_game_record(record: GameRecord) -> None:
    """Fail closed by independently rebuilding evidence from raw traces."""
    if not record.game_id or not record.pair_id:
        raise ValueError("GameRecord requires non-empty game_id and pair_id")
    if record.topology_fingerprint != TORUS_5X5_TOPOLOGY_FINGERPRINT:
        raise ValueError("GameRecord topology fingerprint is not canonical Golden Torus 5x5")
    if record.search_contract_id != ARENA_CONTRACT_ID:
        raise ValueError("GameRecord search contract drift")
    if record.black_player not in ("A", "B") or record.white_player not in ("A", "B"):
        raise ValueError("GameRecord colors must be assigned to A/B slots")
    if record.black_player == record.white_player:
        raise ValueError("Black and White cannot be assigned to the same Arena slot")

    start = _replay_start(record)
    if start.state_key != record.start_state_key:
        raise ValueError("GameRecord start_state_key does not replay exactly")
    if start.superko_history != record.start_history:
        raise ValueError("GameRecord start_history does not replay exactly")
    if start.is_terminal:
        raise ValueError("Arena GameRecord cannot start from terminal state")

    state = start
    applied_count = 0
    illegal_seen = False
    for evidence in record.action_trace:
        expected_slot = _slot_for_side(state.side_to_move, record.black_player)
        expected_id = _player_id_for_slot(record, expected_slot)
        if evidence.ply != applied_count + 1:
            raise ValueError("ActionEvidence ply is not sequential")
        if evidence.side_to_move != state.side_to_move.name:
            raise ValueError("ActionEvidence color does not match Golden state")
        if evidence.player_slot != expected_slot or evidence.player_id != expected_id:
            raise ValueError("ActionEvidence player/color mapping is corrupt")
        try:
            transition = apply_action(state, evidence.action)
        except IllegalMoveError:
            if evidence.legal:
                raise ValueError("ActionEvidence marks an illegal move as legal")
            illegal_seen = True
            break
        if not evidence.legal:
            raise ValueError("ActionEvidence marks a legal move as illegal")
        state = transition.after
        applied_count += 1

    if tuple(int(stone) for stone in state.stones) != record.final_board:
        raise ValueError("GameRecord final_board does not match replayed trace")

    if record.termination_reason == TerminationReason.DOUBLE_PASS:
        if illegal_seen or not state.is_terminal:
            raise ValueError("DOUBLE_PASS record must replay to formal terminal")
        exact = result_from_terminal(state)
        if record.absolute_rule_result != exact.winner:
            raise ValueError("Corrupted absolute winner in GameRecord")
        expected_mapped = map_absolute_result(exact.winner, black_player=record.black_player)
        if record.mapped_result != expected_mapped:
            raise ValueError("Corrupted A/B mapping in GameRecord")
        if record.black_area != exact.black_area or record.white_area != exact.white_area:
            raise ValueError("Corrupted expected area score in GameRecord")
        if record.margin_black != exact.margin_black:
            raise ValueError("Corrupted expected margin in GameRecord")
        if record.error_details is not None:
            raise ValueError("Rule result cannot contain technical error details")
    else:
        if record.absolute_rule_result is not None or record.mapped_result is not None:
            raise ValueError("Technical termination must never contain W/L/D result")
        if record.black_area is not None or record.white_area is not None or record.margin_black is not None:
            raise ValueError("Technical termination must never contain a rule score")
        if record.termination_reason == TerminationReason.TRUNCATED_MOVE_LIMIT:
            if state.is_terminal:
                raise ValueError("Truncation cannot override a formal terminal result")
            if applied_count != GOLDEN_MOVE_LIMIT:
                raise ValueError("Move-limit truncation must occur after exactly 500 applied actions")
        elif record.termination_reason in (
            TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION,
            TerminationReason.ERROR_SEARCH,
        ):
            if not illegal_seen and not record.error_details:
                raise ValueError("Illegal/search error record lacks failure evidence")
        elif record.termination_reason == TerminationReason.ERROR_PLAYER_EXCEPTION:
            if not record.error_details:
                raise ValueError("Player exception record lacks error details")
        else:
            raise ValueError(f"Unsupported technical termination {record.termination_reason!r}")

def recompute_summary(records: Iterable[GameRecord]) -> ArenaSummary:
    raw = tuple(records)
    ids = [record.game_id for record in raw]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game_id in raw Arena records")
    for record in raw:
        validate_game_record(record)

    a_wins = sum(record.mapped_result == MappedResult.A_WIN for record in raw)
    b_wins = sum(record.mapped_result == MappedResult.B_WIN for record in raw)
    draws = sum(record.mapped_result == MappedResult.DRAW for record in raw)
    technical = [record for record in raw if record.is_technical]
    reason_counts: dict[str, int] = {}
    for record in technical:
        reason_counts[record.termination_reason.value] = reason_counts.get(record.termination_reason.value, 0) + 1

    pairs: dict[str, list[GameRecord]] = {}
    for record in raw:
        pairs.setdefault(record.pair_id, []).append(record)
    pair_summaries = tuple(
        PairSummary(
            pair_id=pair_id,
            game_ids=tuple(record.game_id for record in pair_records),
            a_wins=sum(record.mapped_result == MappedResult.A_WIN for record in pair_records),
            b_wins=sum(record.mapped_result == MappedResult.B_WIN for record in pair_records),
            draws=sum(record.mapped_result == MappedResult.DRAW for record in pair_records),
            technical_failures=sum(record.is_technical for record in pair_records),
        )
        for pair_id, pair_records in sorted(pairs.items())
    )
    return ArenaSummary(
        games=len(raw),
        rule_results=sum(record.is_rule_result for record in raw),
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        technical_failures=len(technical),
        a_as_black=sum(record.black_player == "A" for record in raw),
        b_as_black=sum(record.black_player == "B" for record in raw),
        technical_by_reason=tuple(sorted(reason_counts.items())),
        paired_results=pair_summaries,
    )

class SequentialGoldenArena:
    def __init__(
        self,
        *,
        contract: GoldenArenaContract = DEFAULT_ARENA_CONTRACT,
        master_seed: int = 0,
    ) -> None:
        self.contract = contract
        self.master_seed = int(master_seed)
        self._records: list[GameRecord] = []
        self._game_ids: set[str] = set()

    @property
    def records(self) -> tuple[GameRecord, ...]:
        return tuple(self._records)

    def summary(self) -> ArenaSummary:
        return recompute_summary(self.records)

    def _validate_start(
        self,
        start_state: GoldenState,
        start_trace: Sequence[int | str],
        *,
        allow_research_komi: bool,
    ) -> None:
        if not start_state.is_canonical_live:
            raise ValueError("Golden Arena rejects synthetic/unproven start state history")
        if start_state.is_terminal:
            raise ValueError("Golden Arena rejects terminal start state")
        if start_state.topology.fingerprint != self.contract.topology_fingerprint:
            raise ValueError("Golden Arena start topology is not canonical Torus 5x5")
        if allow_research_komi:
            if start_state.komi not in (self.contract.komi, 0.0):
                raise ValueError("Arena research start permits only explicit komi 0.0 or baseline 0.5")
        elif start_state.komi != self.contract.komi:
            raise ValueError("Golden Arena production start komi must match contract 0.5")
        if not allow_research_komi and start_state.rules_fingerprint != self.contract.rules_fingerprint:
            raise ValueError("Golden Arena production rules fingerprint drift")

        replayed = initial_state(topology=start_state.topology, komi=start_state.komi)
        for action in start_trace:
            replayed = apply_action(replayed, action).after
        if replayed.state_key != start_state.state_key:
            raise ValueError("Arena start_state must be exactly reproducible from legal start_trace")

    def play_game(
        self,
        *,
        game_id: str,
        pair_id: str,
        player_A: Player,
        player_B: Player,
        black_player: str,
        start_state: GoldenState | None = None,
        start_trace: Sequence[int | str] = (),
        allow_research_komi: bool = False,
    ) -> GameRecord:
        if game_id in self._game_ids:
            raise ValueError(f"Duplicate game_id rejected: {game_id}")
        if not game_id or not pair_id:
            raise ValueError("game_id and pair_id must be non-empty")
        if black_player not in ("A", "B"):
            raise ValueError("black_player must be A or B")
        white_player = "B" if black_player == "A" else "A"
        start_state = start_state or initial_state()
        self._validate_start(start_state, start_trace, allow_research_komi=allow_research_komi)
        self._game_ids.add(game_id)

        seed_game = _derive_seed(self.master_seed, pair_id, game_id, "game")
        seed_A = _derive_seed(seed_game, "A")
        seed_B = _derive_seed(seed_game, "B")
        players = {"A": player_A, "B": player_B}
        seeds = {"A": seed_A, "B": seed_B}
        move_indices = {"A": 0, "B": 0}
        state = start_state
        action_trace: list[ActionEvidence] = []
        applied_count = 0
        termination: TerminationReason | None = None
        error_details: str | None = None

        while termination is None:
            slot = _slot_for_side(state.side_to_move, black_player)
            player = players[slot]
            player_id = str(player.player_id)
            move_index = move_indices[slot]
            context = PlayerContext(
                game_id=game_id,
                pair_id=pair_id,
                player_slot=slot,
                player_id=player_id,
                assigned_color=state.side_to_move.name,
                player_move_index=move_index,
                ply=applied_count + 1,
                seed=derive_child_seed(seeds[slot], f"move:{move_index}"),
                search_contract_id=self.contract.contract_id,
            )
            try:
                action = player.select_action(state, context)
            except SearchError as exc:
                termination = TerminationReason.ERROR_SEARCH
                error_details = f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                termination = (
                    TerminationReason.ERROR_SEARCH
                    if bool(getattr(player, "is_search_player", False))
                    else TerminationReason.ERROR_PLAYER_EXCEPTION
                )
                error_details = f"{type(exc).__name__}: {exc}"
                break

            try:
                transition = apply_action(state, action)
            except IllegalMoveError as exc:
                reason = (
                    TerminationReason.ERROR_SEARCH
                    if bool(getattr(player, "is_search_player", False))
                    else TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION
                )
                action_trace.append(ActionEvidence(
                    ply=applied_count + 1,
                    side_to_move=state.side_to_move.name,
                    player_slot=slot,
                    player_id=player_id,
                    action=action,
                    legal=False,
                    error=f"{exc.reason.value}: {exc}",
                ))
                termination = reason
                error_details = action_trace[-1].error
                break

            action_trace.append(ActionEvidence(
                ply=applied_count + 1,
                side_to_move=state.side_to_move.name,
                player_slot=slot,
                player_id=player_id,
                action=action,
                legal=True,
            ))
            move_indices[slot] += 1
            applied_count += 1
            state = transition.after
            termination = post_action_termination(state, applied_count)

        absolute = None
        mapped = None
        black_area = None
        white_area = None
        margin = None
        if termination == TerminationReason.DOUBLE_PASS:
            # Absolute Golden result is computed first, with no A/B identity.
            golden_result = result_from_terminal(state)
            absolute = golden_result.winner
            black_area = golden_result.black_area
            white_area = golden_result.white_area
            margin = golden_result.margin_black
            # Mapping happens only after the absolute result exists.
            mapped = map_absolute_result(absolute, black_player=black_player)

        record = GameRecord(
            game_id=game_id,
            pair_id=pair_id,
            rules_fingerprint=state.rules_fingerprint,
            topology_fingerprint=state.topology.fingerprint,
            komi=state.komi,
            start_state_key=start_state.state_key,
            start_history=start_state.superko_history,
            start_trace=tuple(start_trace),
            player_A_id=str(player_A.player_id),
            player_B_id=str(player_B.player_id),
            black_player=black_player,
            white_player=white_player,
            seed_game=seed_game,
            seed_A=seed_A,
            seed_B=seed_B,
            search_contract_id=self.contract.contract_id,
            search_settings=self.contract.search.evidence(),
            action_trace=tuple(action_trace),
            final_board=tuple(int(stone) for stone in state.stones),
            absolute_rule_result=absolute,
            mapped_result=mapped,
            black_area=black_area,
            white_area=white_area,
            margin_black=margin,
            termination_reason=termination,
            error_details=error_details,
        )
        validate_game_record(record)
        self._records.append(record)
        return record

    def play_pair(
        self,
        *,
        pair_id: str,
        player_A: Player,
        player_B: Player,
        start_state: GoldenState | None = None,
        start_trace: Sequence[int | str] = (),
        game_ids: tuple[str, str] | None = None,
        allow_research_komi: bool = False,
    ) -> tuple[GameRecord, GameRecord]:
        start_state = start_state or initial_state()
        first_id, second_id = game_ids or (f"{pair_id}-g1", f"{pair_id}-g2")
        first = self.play_game(
            game_id=first_id, pair_id=pair_id, player_A=player_A, player_B=player_B,
            black_player="A", start_state=start_state, start_trace=start_trace,
            allow_research_komi=allow_research_komi,
        )
        second = self.play_game(
            game_id=second_id, pair_id=pair_id, player_A=player_A, player_B=player_B,
            black_player="B", start_state=start_state, start_trace=start_trace,
            allow_research_komi=allow_research_komi,
        )
        return first, second

def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value

def write_records_jsonl(path: str | Path, records: Iterable[GameRecord]) -> None:
    raw = tuple(records)
    # Validation before persistence prevents a corrupted mutable summary/record
    # from becoming the source of truth.
    for record in raw:
        validate_game_record(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(_jsonable(record), sort_keys=True) + "\n" for record in raw),
        encoding="utf-8",
    )
