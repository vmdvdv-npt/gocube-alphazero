"""Protocol V1 orchestration over the canonical Golden rules and shared move selector."""

from __future__ import annotations

from typing import Mapping, Sequence

from gocube_golden.arena_contract import SEARCH_IMPLEMENTATION_ID
from gocube_golden.result import DOUBLE_PASS, Winner, result_from_terminal
from gocube_golden.rules import apply_action
from gocube_golden.state import BLACK, PASS, GoldenState
from gocube_golden.scoring import score_terminal

from .catalog import CheckpointDescriptor, GOLDEN_TERMINAL_ADJUDICATOR
from .errors import CheckpointIncompatible, GenerationFailed, IntegrationError
from .golden_mapping import GoldenActionMappingError, mapping_for
from .golden_models import GoldenPlayableModel
from .golden_move import (
    GOLDEN_PROTOCOL_RULESET,
    GoldenMoveSelector,
    GoldenPositionContract,
    initial_state_for_position,
    replay_protocol_moves_strict,
    validate_checkpoint_position_compatibility,
    validate_loaded_model,
)


GOLDEN_SEARCH_IMPLEMENTATION_ID = SEARCH_IMPLEMENTATION_ID
TORUS9_INTERACTIVE_MOVE_LIMIT = 500


def _winner_name(winner: Winner) -> str:
    return winner.value.lower()


def _score_payload(score, captures: tuple[int, int]) -> dict[str, object]:
    return {
        "ruleSet": GOLDEN_PROTOCOL_RULESET,
        "black": float(score.black_area),
        "white": float(score.white_area) + float(score.komi),
        "komi": float(score.komi),
        "winner": _winner_name(
            Winner.BLACK if score.margin_black > 0 else Winner.WHITE if score.margin_black < 0 else Winner.DRAW
        ),
        "margin": abs(float(score.margin_black)),
        "captures": list(captures),
        "prisoners": None,
        "territory": {
            "black": score.black_territory,
            "white": score.white_territory,
            "neutral": score.neutral_points,
            "seki": 0,
        },
        "stonesOnBoard": {
            "black": score.black_stones,
            "white": score.white_stones,
        },
        "deadStones": None,
    }


def serialize_golden_result(state: GoldenState, *, captures: tuple[int, int]) -> dict[str, object]:
    if not state.is_terminal:
        raise GenerationFailed("Golden result serialization requires a DOUBLE_PASS terminal state")
    result = result_from_terminal(state)
    score = score_terminal(state)
    return {
        "winner": _winner_name(result.winner),
        "adjudicatorId": GOLDEN_TERMINAL_ADJUDICATOR,
        "fallbackCount": 0,
        "unresolvedCount": 0,
        "cleanupMoveCount": 0,
        "noResult": False,
        "terminationReason": DOUBLE_PASS,
        "resultProvenance": "golden-graph-area-v1",
        "runtimeForced": False,
        "score": _score_payload(score, captures),
    }


def replay_protocol_moves(
    *,
    topology: str,
    size: int,
    moves: Sequence[Mapping[str, object]],
) -> tuple[GoldenState, tuple[int, int]]:
    try:
        return replay_protocol_moves_strict(
            topology=topology,
            size=size,
            rule_set=GOLDEN_PROTOCOL_RULESET,
            komi=0.5,
            moves=moves,
        )
    except IntegrationError as exc:
        raise GenerationFailed(str(exc)) from exc


def replay_protocol_game(game: Mapping[str, object]) -> GoldenState:
    topology = game.get("topology")
    size = game.get("size")
    moves = game.get("moves")
    if (
        not isinstance(topology, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or isinstance(moves, (str, bytes, bytearray))
        or not isinstance(moves, Sequence)
    ):
        raise GenerationFailed("Malformed Protocol V1 Golden game")
    state, captures = replay_protocol_moves(topology=topology, size=size, moves=moves)
    if not state.is_terminal:
        raise GenerationFailed("Protocol Golden replay did not reach DOUBLE_PASS")
    expected_result = serialize_golden_result(state, captures=captures)
    if game.get("result") != expected_result:
        raise GenerationFailed("Protocol Golden result does not match replayed Golden terminal state")
    return state


class GoldenGameGenerator:
    """Generate one deterministic Protocol V1 game from two Golden models."""

    def __init__(self, *, mapping_resolver=mapping_for, move_selector: GoldenMoveSelector | None = None):
        self.mapping_resolver = mapping_resolver
        self.move_selector = move_selector or GoldenMoveSelector(mapping_resolver=mapping_resolver)

    def _compatible(
        self,
        black: CheckpointDescriptor,
        white: CheckpointDescriptor,
    ) -> None:
        fields = (
            "topology",
            "size",
            "rule_set",
            "terminal_adjudicator",
            "profile_id",
            "architecture_id",
            "rules_fingerprint",
            "observation_fingerprint",
            "target_fingerprint",
            "komi",
        )
        differences = [
            field for field in fields
            if getattr(black, field, None) != getattr(white, field, None)
        ]
        if differences:
            field = differences[0]
            raise CheckpointIncompatible(
                f"Golden checkpoints are incompatible for {field}: "
                f"black={getattr(black, field, None)!r}, white={getattr(white, field, None)!r}"
            )
        if black.rule_set != GOLDEN_PROTOCOL_RULESET:
            raise CheckpointIncompatible("Golden graph-area has no valid Protocol V1 ruleSet projection")

    @staticmethod
    def _move_limit(descriptor: CheckpointDescriptor) -> int:
        if descriptor.topology != "torus" or descriptor.size != 9:
            raise CheckpointIncompatible(
                f"No current Golden serving watchdog for {descriptor.topology} size {descriptor.size}"
            )
        return TORUS9_INTERACTIVE_MOVE_LIMIT

    def generate(
        self,
        *,
        black: CheckpointDescriptor,
        white: CheckpointDescriptor,
        black_model: GoldenPlayableModel,
        white_model: GoldenPlayableModel,
        mcts_sims: int,
    ) -> dict[str, object]:
        if isinstance(mcts_sims, bool) or not isinstance(mcts_sims, int) or mcts_sims < 1:
            raise GenerationFailed("mctsSims must be an integer >= 1")
        self._compatible(black, white)
        try:
            mapping = self.mapping_resolver(black.topology, black.size)
        except GoldenActionMappingError as exc:
            raise GenerationFailed(f"Action mapping failure: {exc}") from exc

        position = GoldenPositionContract(
            topology=black.topology,
            size=black.size,
            rule_set=black.rule_set,
            komi=black.komi,
        )
        state = initial_state_for_position(position)

        validate_checkpoint_position_compatibility(position=position, state=state, descriptor=black, mapping=mapping)
        validate_checkpoint_position_compatibility(position=position, state=state, descriptor=white, mapping=mapping)
        validate_loaded_model(black_model, black, mapping)
        validate_loaded_model(white_model, white, mapping)

        models = (black_model, white_model)
        descriptors = (black, white)
        moves: list[dict[str, object]] = []
        captures = [0, 0]
        move_limit = self._move_limit(black)

        for _ in range(move_limit):
            if state.is_terminal:
                break
            current_color = state.side_to_move
            index = 0 if current_color == BLACK else 1
            selection = self.move_selector.select_move(
                state=state,
                descriptor=descriptors[index],
                model=models[index],
                mcts_sims=mcts_sims,
            )
            action = selection.action
            try:
                transition = apply_action(state, action)
            except Exception as exc:
                raise GenerationFailed(f"Golden rules transition failed: {exc}") from exc

            if action != PASS:
                captures[index] += len(transition.captured)
            moves.append(
                {
                    "moveNumber": len(moves) + 1,
                    "color": "black" if current_color == BLACK else "white",
                    "action": mapping.golden_action_to_protocol(action),
                    "captured": mapping.captured_point_ids(transition.captured),
                }
            )
            state = transition.after

        if not state.is_terminal:
            raise GenerationFailed(
                f"Golden game exceeded technical watchdog ({move_limit}) before DOUBLE_PASS"
            )
        return {
            "topology": black.topology,
            "size": black.size,
            "ruleSet": GOLDEN_PROTOCOL_RULESET,
            "komi": 0.5,
            "terminalAdjudicator": GOLDEN_TERMINAL_ADJUDICATOR,
            "mctsSims": mcts_sims,
            "black": {"checkpointId": black.checkpoint_id},
            "white": {"checkpointId": white.checkpoint_id},
            "moves": moves,
            "result": serialize_golden_result(state, captures=(captures[0], captures[1])),
        }


__all__ = [
    "GOLDEN_PROTOCOL_RULESET",
    "GoldenGameGenerator",
    "replay_protocol_game",
    "replay_protocol_moves",
    "serialize_golden_result",
]
