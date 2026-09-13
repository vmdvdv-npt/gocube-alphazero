from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Iterable, Sequence
from uuid import uuid4

from .arena_contract import (
    ARENA_CONTRACT_ID,
    DEFAULT_ARENA_CONTRACT,
    GOLDEN_MOVE_LIMIT,
    SEARCH_CONTRACT_FINGERPRINT,
    SEARCH_IMPLEMENTATION_ID,
    GoldenArenaContract,
)
from .experiment_profile import (
    EXPERIMENT_FINGERPRINT,
    PROFILE_ID,
    RULES_PROFILE_ID,
    SEED_DERIVATION_ID,
)
from .players import Player, PlayerContext, derive_child_seed
from .provenance import (
    PROVENANCE_SCHEMA_VERSION,
    CodeIdentity,
    PlayerIdentity,
    RunManifest,
    capture_code_identity,
    derive_game_seeds,
    infer_player_identity,
    run_identity_payload,
    sha256_fingerprint,
)
from .result import DOUBLE_PASS, Winner, result_from_terminal
from .rules import IllegalMoveError, apply_action
from .search import SEARCH_IMPLEMENTATION_FINGERPRINT, SearchError
from .state import BLACK, GoldenState, initial_state
from .topology import TORUS_5X5, TORUS_5X5_TOPOLOGY_FINGERPRINT

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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


TECHNICAL_TERMINATIONS = frozenset(
    {
        TerminationReason.TRUNCATED_MOVE_LIMIT,
        TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION,
        TerminationReason.ERROR_PLAYER_EXCEPTION,
        TerminationReason.ERROR_SEARCH,
    }
)


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
    schema_version: int
    run_id: str
    run_identity_fingerprint: str
    experiment_profile_id: str
    experiment_fingerprint: str
    git_commit_sha: str
    git_tree_sha: str
    git_worktree_clean: bool
    seed_derivation_id: str
    master_seed: int
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
    player_A_identity_fingerprint: str
    player_B_identity_fingerprint: str
    black_player: str
    white_player: str
    seed_game: int
    seed_A: int
    seed_B: int
    search_contract_id: str
    search_contract_fingerprint: str
    search_implementation_id: str
    search_implementation_fingerprint: str
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
    if state.is_terminal:
        return TerminationReason.DOUBLE_PASS
    if action_count >= GOLDEN_MOVE_LIMIT:
        return TerminationReason.TRUNCATED_MOVE_LIMIT
    return None


def _slot_for_side(side, black_player: str) -> str:
    if black_player == PlayerSlot.A.value:
        return PlayerSlot.A.value if side == BLACK else PlayerSlot.B.value
    if black_player == PlayerSlot.B.value:
        return PlayerSlot.B.value if side == BLACK else PlayerSlot.A.value
    raise ValueError("black_player must be A or B")


def _player_id_for_slot(record: GameRecord, slot: str) -> str:
    return record.player_A_id if slot == PlayerSlot.A.value else record.player_B_id


def _replay_start(record: GameRecord) -> GoldenState:
    state = initial_state(topology=TORUS_5X5, komi=record.komi)
    for action in record.start_trace:
        state = apply_action(state, action).after
    return state


def _expected_run_identity(record: GameRecord) -> str:
    return sha256_fingerprint(
        run_identity_payload(
            schema_version=record.schema_version,
            run_id=record.run_id,
            experiment_profile_id=record.experiment_profile_id,
            experiment_fingerprint=record.experiment_fingerprint,
            git_commit_sha=record.git_commit_sha,
            git_tree_sha=record.git_tree_sha,
            git_worktree_clean=record.git_worktree_clean,
            rules_profile_id=RULES_PROFILE_ID,
            rules_fingerprint=DEFAULT_ARENA_CONTRACT.rules_fingerprint,
            topology_fingerprint=DEFAULT_ARENA_CONTRACT.topology_fingerprint,
            komi=DEFAULT_ARENA_CONTRACT.komi,
            arena_contract_id=record.search_contract_id,
            search_implementation_id=record.search_implementation_id,
            search_implementation_fingerprint=record.search_implementation_fingerprint,
            search_contract_fingerprint=record.search_contract_fingerprint,
            search_settings=record.search_settings,
            seed_derivation_id=record.seed_derivation_id,
            master_seed=record.master_seed,
            player_A_identity_fingerprint=record.player_A_identity_fingerprint,
            player_B_identity_fingerprint=record.player_B_identity_fingerprint,
        )
    )


def _validate_record_provenance(record: GameRecord) -> None:
    if record.schema_version != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("GameRecord provenance schema version drift")
    if not record.run_id or not record.game_id or not record.pair_id:
        raise ValueError("GameRecord requires non-empty run_id, game_id and pair_id")
    if not _SHA256_RE.fullmatch(record.run_identity_fingerprint):
        raise ValueError("GameRecord run identity fingerprint is malformed")
    if record.experiment_profile_id != PROFILE_ID or record.experiment_fingerprint != EXPERIMENT_FINGERPRINT:
        raise ValueError("GameRecord experiment identity drift")
    if not _GIT_SHA_RE.fullmatch(record.git_commit_sha) or not _GIT_SHA_RE.fullmatch(record.git_tree_sha):
        raise ValueError("GameRecord git code identity is malformed")
    if not isinstance(record.git_worktree_clean, bool):
        raise ValueError("GameRecord git_worktree_clean must be boolean")
    if record.seed_derivation_id != SEED_DERIVATION_ID:
        raise ValueError("GameRecord seed derivation contract drift")
    if (record.seed_game, record.seed_A, record.seed_B) != derive_game_seeds(record.master_seed, record.pair_id, record.game_id):
        raise ValueError("GameRecord derived seed evidence does not match master seed schedule")
    for fingerprint in (record.player_A_identity_fingerprint, record.player_B_identity_fingerprint):
        if not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError("GameRecord player identity fingerprint is malformed")
    if record.search_contract_id != ARENA_CONTRACT_ID:
        raise ValueError("GameRecord search contract drift")
    if record.search_contract_fingerprint != SEARCH_CONTRACT_FINGERPRINT:
        raise ValueError("GameRecord search contract fingerprint drift")
    if record.search_implementation_id != SEARCH_IMPLEMENTATION_ID:
        raise ValueError("GameRecord search implementation id drift")
    if record.search_implementation_fingerprint != SEARCH_IMPLEMENTATION_FINGERPRINT:
        raise ValueError("GameRecord search implementation fingerprint drift")
    if record.search_settings != DEFAULT_ARENA_CONTRACT.search.evidence():
        raise ValueError("GameRecord search settings drift")
    if record.run_identity_fingerprint != _expected_run_identity(record):
        raise ValueError("GameRecord run identity fingerprint does not recompute")


def validate_game_record(record: GameRecord) -> None:
    _validate_record_provenance(record)
    if record.topology_fingerprint != TORUS_5X5_TOPOLOGY_FINGERPRINT:
        raise ValueError("GameRecord topology fingerprint is not canonical Golden Torus 5x5")
    if record.search_contract_id != ARENA_CONTRACT_ID:
        raise ValueError("GameRecord search contract drift")
    if record.search_implementation_id != SEARCH_IMPLEMENTATION_ID:
        raise ValueError("GameRecord search implementation id drift")
    if record.search_implementation_fingerprint != SEARCH_IMPLEMENTATION_FINGERPRINT:
        raise ValueError("GameRecord search implementation fingerprint drift")
    if record.search_settings != DEFAULT_ARENA_CONTRACT.search.evidence():
        raise ValueError("GameRecord search settings drift")
    if record.black_player not in ("A", "B") or record.white_player not in ("A", "B") or record.black_player == record.white_player:
        raise ValueError("GameRecord colors must be one A/B slot each")

    start = _replay_start(record)
    if record.rules_fingerprint != start.rules_fingerprint:
        raise ValueError("GameRecord rules fingerprint does not match replayed Golden rules")
    if start.state_key != record.start_state_key:
        raise ValueError("GameRecord start_state_key does not replay exactly")
    if start.superko_history != record.start_history:
        raise ValueError("GameRecord start_history does not replay exactly")
    if start.is_terminal:
        raise ValueError("Arena GameRecord cannot start from terminal state")

    state = start
    applied_count = 0
    illegal_seen = False
    for evidence_index, evidence in enumerate(record.action_trace):
        if state.is_terminal:
            raise ValueError("ActionEvidence exists after formal Golden terminal")
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
            if evidence_index != len(record.action_trace) - 1:
                raise ValueError("ActionEvidence trace continues after illegal move")
            break
        if not evidence.legal:
            raise ValueError("ActionEvidence marks a legal move as illegal")
        state = transition.after
        applied_count += 1

    if state.is_terminal and record.termination_reason != TerminationReason.DOUBLE_PASS:
        raise ValueError("Formal Golden terminal requires DOUBLE_PASS termination")
    if tuple(int(stone) for stone in state.stones) != record.final_board:
        raise ValueError("GameRecord final_board does not match replayed trace")

    if record.termination_reason == TerminationReason.DOUBLE_PASS:
        if illegal_seen or not state.is_terminal:
            raise ValueError("DOUBLE_PASS record must replay to formal terminal")
        exact = result_from_terminal(state)
        if record.absolute_rule_result != exact.winner:
            raise ValueError("Corrupted absolute winner in GameRecord")
        if record.mapped_result != map_absolute_result(exact.winner, black_player=record.black_player):
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
                raise ValueError(f"Move-limit truncation must occur after exactly {GOLDEN_MOVE_LIMIT} applied actions")
        elif record.termination_reason in (TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION, TerminationReason.ERROR_SEARCH):
            if not illegal_seen and not record.error_details:
                raise ValueError("Illegal/search error record lacks failure evidence")
        elif record.termination_reason == TerminationReason.ERROR_PLAYER_EXCEPTION:
            if not record.error_details:
                raise ValueError("Player exception record lacks error details")
        else:
            raise ValueError(f"Unsupported technical termination {record.termination_reason!r}")


_PAIR_EQUAL_FIELDS = (
    "run_id",
    "run_identity_fingerprint",
    "experiment_profile_id",
    "experiment_fingerprint",
    "git_commit_sha",
    "git_tree_sha",
    "git_worktree_clean",
    "seed_derivation_id",
    "master_seed",
    "pair_id",
    "rules_fingerprint",
    "topology_fingerprint",
    "komi",
    "start_state_key",
    "start_history",
    "start_trace",
    "player_A_id",
    "player_B_id",
    "player_A_identity_fingerprint",
    "player_B_identity_fingerprint",
    "search_contract_id",
    "search_contract_fingerprint",
    "search_implementation_id",
    "search_implementation_fingerprint",
    "search_settings",
)


def validate_pair_records(records: Iterable[GameRecord]) -> tuple[GameRecord, GameRecord]:
    pair = tuple(records)
    if len(pair) != 2:
        raise ValueError("Golden paired evidence requires exactly two games per pair")
    first, second = pair
    validate_game_record(first)
    validate_game_record(second)
    if first.game_id == second.game_id:
        raise ValueError("Golden pair games must have distinct game_id values")
    for field_name in _PAIR_EQUAL_FIELDS:
        if getattr(first, field_name) != getattr(second, field_name):
            raise ValueError(f"Golden pair provenance/start mismatch: {field_name}")
    if {(first.black_player, first.white_player), (second.black_player, second.white_player)} != {("A", "B"), ("B", "A")}:
        raise ValueError("Golden pair must contain exactly one A-black and one B-black game")
    return first, second


def pair_schedule_from_records(records: Iterable[GameRecord]) -> tuple[tuple[str, str, str], ...]:
    raw = tuple(records)
    grouped: dict[str, list[GameRecord]] = {}
    for record in raw:
        grouped.setdefault(record.pair_id, []).append(record)
    schedule: list[tuple[str, str, str]] = []
    for pair_id, pair_records in sorted(grouped.items()):
        one, two = validate_pair_records(pair_records)
        by_black = {one.black_player: one, two.black_player: two}
        schedule.append((pair_id, by_black["A"].game_id, by_black["B"].game_id))
    return tuple(schedule)


def recompute_summary(records: Iterable[GameRecord]) -> ArenaSummary:
    raw = tuple(records)
    ids = [record.game_id for record in raw]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game_id in raw Arena records")
    schedule = pair_schedule_from_records(raw)
    pairs: dict[str, list[GameRecord]] = {}
    for record in raw:
        pairs.setdefault(record.pair_id, []).append(record)

    a_wins = sum(record.mapped_result == MappedResult.A_WIN for record in raw)
    b_wins = sum(record.mapped_result == MappedResult.B_WIN for record in raw)
    draws = sum(record.mapped_result == MappedResult.DRAW for record in raw)
    technical = [record for record in raw if record.is_technical]
    reason_counts: dict[str, int] = {}
    for record in technical:
        reason_counts[record.termination_reason.value] = reason_counts.get(record.termination_reason.value, 0) + 1

    pair_summaries = tuple(
        PairSummary(
            pair_id=pair_id,
            game_ids=(a_black_game, b_black_game),
            a_wins=sum(record.mapped_result == MappedResult.A_WIN for record in pairs[pair_id]),
            b_wins=sum(record.mapped_result == MappedResult.B_WIN for record in pairs[pair_id]),
            draws=sum(record.mapped_result == MappedResult.DRAW for record in pairs[pair_id]),
            technical_failures=sum(record.is_technical for record in pairs[pair_id]),
        )
        for pair_id, a_black_game, b_black_game in schedule
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
        run_id: str | None = None,
        code_identity: CodeIdentity | None = None,
        require_canonical_code: bool = False,
    ) -> None:
        self.contract = contract
        self.master_seed = int(master_seed)
        self.run_id = run_id or f"golden-{uuid4().hex}"
        self.code_identity = code_identity or capture_code_identity()
        self.code_identity.validate(require_canonical=require_canonical_code)
        self._records: list[GameRecord] = []
        self._game_ids: set[str] = set()
        self._player_A_identity: PlayerIdentity | None = None
        self._player_B_identity: PlayerIdentity | None = None

    @property
    def records(self) -> tuple[GameRecord, ...]:
        return tuple(self._records)

    def summary(self) -> ArenaSummary:
        return recompute_summary(self.records)

    def _bind_player_identities(self, player_A: Player, player_B: Player) -> tuple[PlayerIdentity, PlayerIdentity]:
        identity_A = infer_player_identity(player_A)
        identity_B = infer_player_identity(player_B)
        if self._player_A_identity is None:
            self._player_A_identity = identity_A
            self._player_B_identity = identity_B
        elif self._player_A_identity != identity_A or self._player_B_identity != identity_B:
            raise ValueError("A Golden run cannot change A/B structured player identities between games")
        return identity_A, identity_B

    def _manifest_snapshot(
        self,
        *,
        pair_count: int,
        game_count: int,
        pair_schedule: tuple[tuple[str, str, str], ...],
    ) -> RunManifest:
        if self._player_A_identity is None or self._player_B_identity is None:
            raise ValueError("Cannot build Golden RunManifest before player identities are bound")
        return RunManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            run_id=self.run_id,
            experiment_profile_id=PROFILE_ID,
            experiment_fingerprint=EXPERIMENT_FINGERPRINT,
            git_commit_sha=self.code_identity.git_commit_sha,
            git_tree_sha=self.code_identity.git_tree_sha,
            git_worktree_clean=self.code_identity.working_tree_clean,
            canonical_evidence=self.code_identity.canonical,
            rules_profile_id=RULES_PROFILE_ID,
            rules_fingerprint=self.contract.rules_fingerprint,
            topology_fingerprint=self.contract.topology_fingerprint,
            komi=self.contract.komi,
            arena_contract_id=self.contract.contract_id,
            search_implementation_id=SEARCH_IMPLEMENTATION_ID,
            search_implementation_fingerprint=SEARCH_IMPLEMENTATION_FINGERPRINT,
            search_contract_fingerprint=SEARCH_CONTRACT_FINGERPRINT,
            search_settings=self.contract.search.evidence(),
            seed_derivation_id=SEED_DERIVATION_ID,
            master_seed=self.master_seed,
            pair_count=pair_count,
            game_count=game_count,
            pair_schedule=pair_schedule,
            player_A=self._player_A_identity,
            player_B=self._player_B_identity,
        )

    def manifest(self, *, require_canonical: bool = False) -> RunManifest:
        summary = recompute_summary(self.records)
        schedule = pair_schedule_from_records(self.records)
        manifest = self._manifest_snapshot(pair_count=len(summary.paired_results), game_count=summary.games, pair_schedule=schedule)
        validate_run_evidence(manifest, self.records, require_canonical=require_canonical)
        return manifest

    def _validate_start(self, start_state: GoldenState, start_trace: Sequence[int | str], *, allow_research_komi: bool) -> None:
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
        identity_A, identity_B = self._bind_player_identities(player_A, player_B)
        transient_manifest = self._manifest_snapshot(pair_count=0, game_count=0, pair_schedule=())
        self._game_ids.add(game_id)

        seed_game, seed_A, seed_B = derive_game_seeds(self.master_seed, pair_id, game_id)
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
                termination = TerminationReason.ERROR_SEARCH if bool(getattr(player, "is_search_player", False)) else TerminationReason.ERROR_PLAYER_EXCEPTION
                error_details = f"{type(exc).__name__}: {exc}"
                break

            try:
                transition = apply_action(state, action)
            except IllegalMoveError as exc:
                termination = TerminationReason.ERROR_SEARCH if bool(getattr(player, "is_search_player", False)) else TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION
                action_trace.append(ActionEvidence(
                    ply=applied_count + 1,
                    side_to_move=state.side_to_move.name,
                    player_slot=slot,
                    player_id=player_id,
                    action=action,
                    legal=False,
                    error=f"{exc.reason.value}: {exc}",
                ))
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

        absolute = mapped = None
        black_area = white_area = None
        margin = None
        if termination == TerminationReason.DOUBLE_PASS:
            golden_result = result_from_terminal(state)
            absolute = golden_result.winner
            black_area = golden_result.black_area
            white_area = golden_result.white_area
            margin = golden_result.margin_black
            mapped = map_absolute_result(absolute, black_player=black_player)

        record = GameRecord(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            run_id=self.run_id,
            run_identity_fingerprint=transient_manifest.run_identity_fingerprint,
            experiment_profile_id=PROFILE_ID,
            experiment_fingerprint=EXPERIMENT_FINGERPRINT,
            git_commit_sha=self.code_identity.git_commit_sha,
            git_tree_sha=self.code_identity.git_tree_sha,
            git_worktree_clean=self.code_identity.working_tree_clean,
            seed_derivation_id=SEED_DERIVATION_ID,
            master_seed=self.master_seed,
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
            player_A_identity_fingerprint=identity_A.fingerprint,
            player_B_identity_fingerprint=identity_B.fingerprint,
            black_player=black_player,
            white_player=white_player,
            seed_game=seed_game,
            seed_A=seed_A,
            seed_B=seed_B,
            search_contract_id=self.contract.contract_id,
            search_contract_fingerprint=SEARCH_CONTRACT_FINGERPRINT,
            search_implementation_id=SEARCH_IMPLEMENTATION_ID,
            search_implementation_fingerprint=SEARCH_IMPLEMENTATION_FINGERPRINT,
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
            game_id=first_id,
            pair_id=pair_id,
            player_A=player_A,
            player_B=player_B,
            black_player="A",
            start_state=start_state,
            start_trace=start_trace,
            allow_research_komi=allow_research_komi,
        )
        second = self.play_game(
            game_id=second_id,
            pair_id=pair_id,
            player_A=player_A,
            player_B=player_B,
            black_player="B",
            start_state=start_state,
            start_trace=start_trace,
            allow_research_komi=allow_research_komi,
        )
        validate_pair_records((first, second))
        return first, second


def validate_run_evidence(
    manifest: RunManifest,
    records: Iterable[GameRecord],
    *,
    require_canonical: bool = False,
) -> None:
    raw = tuple(records)
    manifest.validate(require_canonical=require_canonical)
    summary = recompute_summary(raw)
    schedule = pair_schedule_from_records(raw)
    if manifest.game_count != summary.games or manifest.pair_count != len(summary.paired_results):
        raise ValueError("RunManifest counts do not match raw evidence")
    if manifest.pair_schedule != schedule:
        raise ValueError("RunManifest pair_schedule does not match independently reconstructed raw schedule")
    expected_run_identity = manifest.run_identity_fingerprint
    for record in raw:
        if record.run_id != manifest.run_id or record.run_identity_fingerprint != expected_run_identity:
            raise ValueError("GameRecord run identity does not match RunManifest")
        if record.experiment_profile_id != manifest.experiment_profile_id or record.experiment_fingerprint != manifest.experiment_fingerprint:
            raise ValueError("GameRecord experiment identity does not match RunManifest")
        if record.git_commit_sha != manifest.git_commit_sha or record.git_tree_sha != manifest.git_tree_sha or record.git_worktree_clean != manifest.git_worktree_clean:
            raise ValueError("GameRecord git code identity does not match RunManifest")
        if record.master_seed != manifest.master_seed:
            raise ValueError("GameRecord master_seed does not match RunManifest")
        if record.search_contract_id != manifest.arena_contract_id or record.search_contract_fingerprint != manifest.search_contract_fingerprint:
            raise ValueError("GameRecord Arena contract does not match RunManifest")
        if record.search_implementation_id != manifest.search_implementation_id or record.search_implementation_fingerprint != manifest.search_implementation_fingerprint:
            raise ValueError("GameRecord search implementation does not match RunManifest")
        if record.search_settings != manifest.search_settings:
            raise ValueError("GameRecord search settings do not match RunManifest")
        if record.player_A_identity_fingerprint != manifest.player_A.fingerprint or record.player_B_identity_fingerprint != manifest.player_B.fingerprint:
            raise ValueError("GameRecord player identity does not match RunManifest")
        if require_canonical:
            if record.rules_fingerprint != manifest.rules_fingerprint or record.topology_fingerprint != manifest.topology_fingerprint or record.komi != manifest.komi:
                raise ValueError("Canonical GameRecord rules/topology/komi do not match RunManifest")


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
    recompute_summary(raw)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(_jsonable(record), sort_keys=True) + "\n" for record in raw), encoding="utf-8")


def write_run_evidence(
    directory: str | Path,
    manifest: RunManifest,
    records: Iterable[GameRecord],
    *,
    require_canonical: bool = True,
) -> tuple[Path, Path]:
    raw = tuple(records)
    validate_run_evidence(manifest, raw, require_canonical=require_canonical)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    manifest_payload = _jsonable(manifest)
    manifest_payload["run_identity_fingerprint"] = manifest.run_identity_fingerprint
    manifest_payload["manifest_fingerprint"] = manifest.manifest_fingerprint
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_records_jsonl(records_path, raw)
    return manifest_path, records_path
