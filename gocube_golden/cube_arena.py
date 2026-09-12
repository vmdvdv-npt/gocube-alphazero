"""Sequential Golden Arena for the canonical Cube contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .arena_contract import SearchSettings
from .cube_topology import CUBE4_TOPOLOGY
from .cube_training import CUBE_ARENA_CONTRACT_ID, CUBE_WATCHDOG, cube_initial_state
from .players import PlayerContext, derive_child_seed
from .provenance import CodeIdentity, capture_code_identity, derive_game_seeds, sha256_fingerprint
from .result import Winner, result_from_terminal
from .rules import IllegalMoveError, apply_action
from .search import SearchError, SequentialPUCT
from .state import PASS, GoldenState

CUBE_ARENA_SEARCH = SearchSettings(simulations=64, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)
CUBE_ARENA_FINGERPRINT = sha256_fingerprint(
    {
        "contract_id": CUBE_ARENA_CONTRACT_ID,
        "topology_fingerprint": CUBE4_TOPOLOGY.fingerprint,
        "simulations": 64,
        "cpuct": 1.25,
        "fpu": 0.0,
        "root_noise": False,
        "temperature": 0.0,
        "fast_search": False,
        "resign": False,
        "watchdog": CUBE_WATCHDOG,
        "inference_batch_size": 1,
        "inference_coalescing": False,
    }
)


@dataclass(frozen=True)
class CubeSearchPlayer:
    player_id: str
    evaluator: object
    search_settings: SearchSettings = CUBE_ARENA_SEARCH
    is_search_player: bool = True

    def select_action(self, state: GoldenState, context: PlayerContext) -> int | str:
        result = SequentialPUCT(self.search_settings).search(state, self.evaluator, seed=context.seed)
        if result.action not in result.legal_actions:
            raise SearchError("Cube Arena selected an illegal action")
        return result.action


@dataclass(frozen=True)
class CubeArenaRecord:
    run_id: str
    pair_id: str
    game_id: str
    master_seed: int
    game_seed: int
    topology_fingerprint: str
    geometry_fingerprint: str
    rules_fingerprint: str
    komi: float
    player_A_id: str
    player_B_id: str
    black_player: str
    white_player: str
    start_state: dict[str, object]
    start_trace: tuple[int | str, ...]
    action_trace: tuple[dict[str, object], ...]
    final_board: tuple[int, ...]
    formal_result: str | None
    mapped_result: str | None
    black_area: int | None
    white_area: int | None
    margin_black: float | None
    technical_termination: str | None
    error: str | None

    @property
    def is_technical(self) -> bool:
        return self.technical_termination is not None

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _slot_for_side(side: object, black_player: str) -> str:
    side_name = getattr(side, "name", str(side))
    if black_player == "A":
        return "A" if side_name == "BLACK" else "B"
    if black_player == "B":
        return "B" if side_name == "BLACK" else "A"
    raise ValueError("black_player must be A or B")


def _mapped_result(winner: Winner, black_player: str) -> str:
    if winner == Winner.DRAW:
        return "DRAW"
    winner_slot = black_player if winner == Winner.BLACK else ("B" if black_player == "A" else "A")
    return "A_WIN" if winner_slot == "A" else "B_WIN"


def _state_to_dict(state: GoldenState) -> dict[str, object]:
    from .cube_training import cube_state_identity

    return cube_state_identity(state)


class SequentialGoldenCubeArena:
    def __init__(
        self,
        *,
        master_seed: int,
        run_id: str,
        code_identity: CodeIdentity | None = None,
        require_canonical_code: bool = False,
        move_limit: int = CUBE_WATCHDOG,
        search_settings: SearchSettings = CUBE_ARENA_SEARCH,
    ) -> None:
        if move_limit != CUBE_WATCHDOG:
            raise ValueError("Cube Arena watchdog is a hard 20*N = 1920 action invariant")
        if require_canonical_code and search_settings != CUBE_ARENA_SEARCH:
            raise ValueError("Canonical Cube Arena search settings are fixed at 64 simulations")
        self.master_seed = int(master_seed)
        self.run_id = run_id
        self.move_limit = int(move_limit)
        self.search_settings = search_settings
        self.code_identity = code_identity or capture_code_identity()
        self.code_identity.validate(require_canonical=require_canonical_code)
        self.records: list[CubeArenaRecord] = []
        self._game_ids: set[str] = set()

    def _validate_start(self, state: GoldenState, trace: Sequence[int | str]) -> None:
        if state.topology.fingerprint != CUBE4_TOPOLOGY.fingerprint:
            raise ValueError("Cube Arena start topology fingerprint mismatch")
        if state.komi != 0.5:
            raise ValueError("Cube Arena komi must be exactly 0.5")
        if not state.is_canonical_live or state.is_terminal:
            raise ValueError("Cube Arena requires a nonterminal canonical-live start")
        replayed = cube_initial_state()
        for action in trace:
            replayed = apply_action(replayed, action).after
        if replayed.state_key != state.state_key:
            raise ValueError("Cube Arena start is not reproducible from its legal trace")

    def play_game(
        self,
        *,
        game_id: str,
        pair_id: str,
        player_A: CubeSearchPlayer,
        player_B: CubeSearchPlayer,
        black_player: str,
        start_state: GoldenState | None = None,
        start_trace: Sequence[int | str] = (),
    ) -> CubeArenaRecord:
        if game_id in self._game_ids:
            raise ValueError(f"Duplicate Cube Arena game_id: {game_id}")
        state = start_state or cube_initial_state()
        self._validate_start(state, start_trace)
        self._game_ids.add(game_id)
        seed_game, seed_A, seed_B = derive_game_seeds(self.master_seed, pair_id, game_id)
        players = {"A": player_A, "B": player_B}
        seeds = {"A": seed_A, "B": seed_B}
        moves = {"A": 0, "B": 0}
        evidence: list[dict[str, object]] = []
        formal: str | None = None
        mapped: str | None = None
        technical: str | None = None
        error: str | None = None
        applied = 0
        while technical is None and formal is None:
            slot = _slot_for_side(state.side_to_move, black_player)
            player = players[slot]
            move_index = moves[slot]
            context = PlayerContext(
                game_id=game_id,
                pair_id=pair_id,
                player_slot=slot,
                player_id=player.player_id,
                assigned_color=state.side_to_move.name,
                player_move_index=move_index,
                ply=applied + 1,
                seed=derive_child_seed(seeds[slot], f"move:{move_index}"),
                search_contract_id=CUBE_ARENA_CONTRACT_ID,
            )
            try:
                action = player.select_action(state, context)
            except SearchError as exc:
                technical, error = "ERROR_SEARCH", f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                technical, error = "ERROR_PLAYER_EXCEPTION", f"{type(exc).__name__}: {exc}"
                break
            try:
                transition = apply_action(state, action)
            except IllegalMoveError as exc:
                technical, error = "ERROR_ILLEGAL_PLAYER_ACTION", f"{exc.reason.value}: {exc}"
                evidence.append({
                    "ply": applied + 1,
                    "side_to_move": state.side_to_move.name,
                    "player_slot": slot,
                    "player_id": player.player_id,
                    "action": action,
                    "legal": False,
                    "error": error,
                })
                break
            evidence.append({
                "ply": applied + 1,
                "side_to_move": state.side_to_move.name,
                "player_slot": slot,
                "player_id": player.player_id,
                "action": action,
                "legal": True,
            })
            moves[slot] += 1
            applied += 1
            state = transition.after
            if state.is_terminal:
                result = result_from_terminal(state)
                formal = result.winner.value
                mapped = _mapped_result(result.winner, black_player)
            elif applied >= self.move_limit:
                technical, error = "TRUNCATED_MOVE_LIMIT", f"Cube Arena watchdog reached {self.move_limit} actions"
        record = CubeArenaRecord(
            run_id=self.run_id,
            pair_id=pair_id,
            game_id=game_id,
            master_seed=self.master_seed,
            game_seed=seed_game,
            topology_fingerprint=CUBE4_TOPOLOGY.fingerprint,
            geometry_fingerprint=CUBE4_TOPOLOGY.geometry_fingerprint,
            rules_fingerprint=state.rules_fingerprint,
            komi=state.komi,
            player_A_id=player_A.player_id,
            player_B_id=player_B.player_id,
            black_player=black_player,
            white_player="B" if black_player == "A" else "A",
            start_state=_state_to_dict(start_state or cube_initial_state()),
            start_trace=tuple(start_trace),
            action_trace=tuple(evidence),
            final_board=tuple(int(stone) for stone in state.stones),
            formal_result=formal,
            mapped_result=mapped,
            black_area=result.black_area if formal is not None else None,
            white_area=result.white_area if formal is not None else None,
            margin_black=result.margin_black if formal is not None else None,
            technical_termination=technical,
            error=error,
        )
        self.records.append(record)
        return record

    def play_pair(
        self,
        *,
        pair_id: str,
        player_A: CubeSearchPlayer,
        player_B: CubeSearchPlayer,
        start_state: GoldenState | None = None,
        start_trace: Sequence[int | str] = (),
    ) -> tuple[CubeArenaRecord, CubeArenaRecord]:
        first = self.play_game(
            game_id=f"{pair_id}-g1", pair_id=pair_id, player_A=player_A,
            player_B=player_B, black_player="A", start_state=start_state,
            start_trace=start_trace,
        )
        second = self.play_game(
            game_id=f"{pair_id}-g2", pair_id=pair_id, player_A=player_A,
            player_B=player_B, black_player="B", start_state=start_state,
            start_trace=start_trace,
        )
        return first, second


def cube_pair_score(records: Sequence[CubeArenaRecord]) -> float:
    if len(records) != 2 or any(record.is_technical for record in records):
        raise ValueError("Cube pair score requires exactly two nontechnical games")
    scores = {"A_WIN": 1.0, "DRAW": 0.5, "B_WIN": 0.0}
    return sum(scores[str(record.mapped_result)] for record in records) / 2.0


def cube_hoeffding_interval(scores: Sequence[float], *, alpha: float = 0.05) -> tuple[float, float]:
    if not scores or not 0.0 < alpha < 1.0 or any(not 0.0 <= float(score) <= 1.0 for score in scores):
        raise ValueError("Hoeffding input must be nonempty pair scores in [0,1]")
    mean = sum(float(score) for score in scores) / len(scores)
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * len(scores)))
    return max(0.0, mean - radius), min(1.0, mean + radius)


def summarize_cube_arena(records: Iterable[CubeArenaRecord]) -> dict[str, object]:
    raw = tuple(records)

    def summarize_group(group_records: Sequence[CubeArenaRecord]) -> dict[str, object]:
        groups: dict[str, list[CubeArenaRecord]] = {}
        for record in group_records:
            groups.setdefault(record.pair_id, []).append(record)
        group_technical = sum(record.is_technical for record in group_records)
        group_scores = [] if group_technical else [cube_pair_score(group) for _, group in sorted(groups.items())]
        group_wins = {"A_WIN": 0, "B_WIN": 0, "DRAW": 0}
        for record in group_records:
            if record.mapped_result in group_wins:
                group_wins[str(record.mapped_result)] += 1
        group_candidate_scores = {
            "black": [
                1.0 if record.mapped_result == "A_WIN" else 0.5 if record.mapped_result == "DRAW" else 0.0
                for record in group_records if not record.is_technical and record.black_player == "A"
            ],
            "white": [
                1.0 if record.mapped_result == "A_WIN" else 0.5 if record.mapped_result == "DRAW" else 0.0
                for record in group_records if not record.is_technical and record.black_player == "B"
            ],
        }
        black_score = sum(group_candidate_scores["black"]) / len(group_candidate_scores["black"]) if group_candidate_scores["black"] else None
        white_score = sum(group_candidate_scores["white"]) / len(group_candidate_scores["white"]) if group_candidate_scores["white"] else None
        split_fraction = sum(score == 0.5 for score in group_scores) / len(group_scores) if group_scores else None
        return {
            "pairs": len(groups),
            "games": len(group_records),
            "wins_losses_draws": {"A": group_wins["A_WIN"], "B": group_wins["B_WIN"], "DRAW": group_wins["DRAW"]},
            "mean_pair_score": None if group_technical else sum(group_scores) / len(group_scores) if group_scores else None,
            "hoeffding_95": None if group_technical else cube_hoeffding_interval(group_scores) if group_scores else None,
            "candidate_score_as_black": black_score,
            "candidate_score_as_white": white_score,
            "two_zero_pairs": sum(score == 1.0 for score in group_scores),
            "one_one_split_pairs": sum(score == 0.5 for score in group_scores),
            "zero_two_pairs": sum(score == 0.0 for score in group_scores),
            "technical": group_technical,
            "technical_reasons": sorted({str(record.technical_termination) for record in group_records if record.is_technical}),
            "pair_scores": group_scores,
            "color_control": {
                "one_one_split_fraction": split_fraction,
                "black_white_score_gap": abs(black_score - white_score) if black_score is not None and white_score is not None else None,
                "diagnostic": "LOW COLOR-CONTROLLED DISCRIMINATIVE POWER" if split_fraction is not None and split_fraction >= 0.5 else "NOT_FLAGGED",
            },
        }

    result = summarize_group(raw)
    strata: dict[str, tuple[CubeArenaRecord, ...]] = {}
    for record in raw:
        strata.setdefault(str(len(record.start_trace)), tuple())
        strata[str(len(record.start_trace))] = strata[str(len(record.start_trace))] + (record,)
    result["prefix_stratum_breakdown"] = {
        prefix: summarize_group(group) for prefix, group in sorted(strata.items(), key=lambda item: int(item[0]))
    }
    result["arena_fingerprint"] = CUBE_ARENA_FINGERPRINT
    return result


def write_cube_arena_jsonl(path: str | Path, records: Iterable[CubeArenaRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
