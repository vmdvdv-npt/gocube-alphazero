"""Protocol V1 orchestration over the canonical Golden rules and PUCT."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.cube_training import cube_initial_state
from gocube_golden.result import DOUBLE_PASS, Winner, result_from_terminal
from gocube_golden.rules import apply_action
from gocube_golden.search import SequentialPUCT
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, PASS, GoldenState, initial_state
from gocube_golden.topology import TORUS_9X9
from gocube_golden.scoring import score_terminal

from .catalog import CheckpointDescriptor, GOLDEN_TERMINAL_ADJUDICATOR
from .errors import CheckpointIncompatible, GenerationFailed
from .golden_mapping import GoldenActionMappingError, GoldenProtocolMapping, mapping_for
from .golden_models import GoldenPlayableModel


GOLDEN_PROTOCOL_RULESET = "chinese"
GOLDEN_SEARCH_IMPLEMENTATION_ID = "golden-sequential-puct-v1"
GOLDEN_CPUCT = 1.25
GOLDEN_FPU = 0.0
TORUS9_INTERACTIVE_MOVE_LIMIT = 500
CUBE4_INTERACTIVE_MOVE_LIMIT = 1920


def _winner_name(winner: Winner) -> str:
    return winner.value.lower()


def _score_payload(score, captures: tuple[int, int]) -> dict[str, object]:
    return {
        # Protocol V1's chinese score is area plus komi on White.  Golden's
        # immutable score stores area without komi and carries the signed
        # black margin, so the projection is explicit here.
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
        # Golden graph-area scoring has no cleanup/dead-stone phase.  Null is
        # semantically honest and is accepted by the optional V1 diagnostics.
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
    """Replay Protocol V1 moves through Golden rules for round-trip proof."""

    try:
        mapping = mapping_for(topology, size)
    except GoldenActionMappingError as exc:
        raise GenerationFailed(f"Action mapping failure: {exc}") from exc
    if topology == "cube":
        state = cube_initial_state(komi=0.5)
    elif topology == "torus":
        state = initial_state(topology=TORUS_9X9, komi=0.5)
    else:
        raise GenerationFailed(f"Unsupported Golden replay topology {topology!r}")
    captures = [0, 0]
    for expected_number, move in enumerate(moves, start=1):
        if not isinstance(move, Mapping):
            raise GenerationFailed("Protocol move must be an object")
        if move.get("moveNumber") != expected_number:
            raise GenerationFailed("Protocol moveNumber is not sequential")
        expected_color = "black" if state.side_to_move == BLACK else "white"
        if move.get("color") != expected_color:
            raise GenerationFailed("Protocol move color does not match Golden side_to_move")
        try:
            action = mapping.protocol_action_to_golden(move["action"])
            transition = apply_action(state, action)
            expected_captured = mapping.captured_point_ids(transition.captured)
        except (KeyError, GoldenActionMappingError, ValueError) as exc:
            raise GenerationFailed(f"Protocol move is illegal or unmappable: {exc}") from exc
        if move.get("captured") != expected_captured:
            raise GenerationFailed(
                f"Protocol captures disagree with Golden transition at move {expected_number}"
            )
        if action != PASS:
            captures[0 if state.side_to_move == BLACK else 1] += len(transition.captured)
        state = transition.after
    return state, (captures[0], captures[1])


def replay_protocol_game(game: Mapping[str, object]) -> GoldenState:
    """Replay a generated Protocol game and verify its Golden result."""

    topology = game.get("topology")
    size = game.get("size")
    moves = game.get("moves")
    if not isinstance(topology, str) or isinstance(size, bool) or not isinstance(size, int) or not isinstance(moves, Sequence):
        raise GenerationFailed("Malformed Protocol V1 Golden game")
    state, captures = replay_protocol_moves(topology=topology, size=size, moves=moves)
    if not state.is_terminal:
        raise GenerationFailed("Protocol Golden replay did not reach DOUBLE_PASS")
    expected_result = serialize_golden_result(state, captures=captures)
    if game.get("result") != expected_result:
        raise GenerationFailed("Protocol Golden result does not match replayed Golden terminal state")
    return state


def _validate_model(model: object, descriptor: CheckpointDescriptor, mapping: GoldenProtocolMapping) -> None:
    if not isinstance(model, GoldenPlayableModel):
        raise GenerationFailed(
            f"Golden checkpoint {descriptor.checkpoint_id} did not load as GoldenPlayableModel"
        )
    metadata = model.metadata
    expected = {
        "profile_id": descriptor.profile_id,
        "architecture_id": descriptor.architecture_id,
        "rules_fingerprint": descriptor.rules_fingerprint,
        "observation_fingerprint": descriptor.observation_fingerprint,
        "target_fingerprint": descriptor.target_fingerprint,
        "komi": descriptor.komi,
    }
    if descriptor.profile_id == "gocube-cube4-golden-training-v1":
        expected["profile_id"] = metadata.get("training_profile_id")
        expected["profile_fingerprint"] = metadata.get("training_profile_fingerprint")
    for key, value in expected.items():
        if value is not None and metadata.get(key, metadata.get("training_" + key)) != value:
            raise CheckpointIncompatible(
                f"Golden model {descriptor.checkpoint_id} metadata mismatch for {key}"
            )
    topology_fp = metadata.get("topology_fingerprint")
    if topology_fp != mapping.golden_topology_fingerprint:
        raise CheckpointIncompatible(
            f"Golden model {descriptor.checkpoint_id} topology fingerprint does not match mapping"
        )
    if float(metadata.get("komi", math.nan)) != 0.5:
        raise CheckpointIncompatible("Golden model komi must be exactly 0.5")


class GoldenGameGenerator:
    """Generate one deterministic Protocol V1 game from two Golden models."""

    def __init__(self, *, mapping_resolver=mapping_for):
        self.mapping_resolver = mapping_resolver

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
    def _start_state(descriptor: CheckpointDescriptor) -> GoldenState:
        if descriptor.topology == "cube" and descriptor.size == 4:
            return cube_initial_state(komi=0.5)
        if descriptor.topology == "torus" and descriptor.size == 9:
            return initial_state(topology=TORUS_9X9, komi=0.5)
        raise GenerationFailed(
            f"No Golden initial state is registered for {descriptor.topology} size {descriptor.size}"
        )

    @staticmethod
    def _move_limit(descriptor: CheckpointDescriptor) -> int:
        if descriptor.topology == "cube":
            return CUBE4_INTERACTIVE_MOVE_LIMIT
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
        _validate_model(black_model, black, mapping)
        _validate_model(white_model, white, mapping)

        state = self._start_state(black)
        evaluators = (black_model.evaluator, white_model.evaluator)
        settings = SearchSettings(
            simulations=mcts_sims,
            cpuct=GOLDEN_CPUCT,
            fpu=GOLDEN_FPU,
            root_noise=False,
            fast_search=False,
            resign=False,
            root_policy_temperature=False,
            move_temperature=0.0,
            deterministic_tie_break=True,
        )
        adapter = GoldenSearchAdapter()
        moves: list[dict[str, object]] = []
        captures = [0, 0]
        move_limit = self._move_limit(black)

        for _ in range(move_limit):
            if state.is_terminal:
                break
            current_color = state.side_to_move
            evaluator = evaluators[0 if current_color == BLACK else 1]
            try:
                search_result = SequentialPUCT(settings, adapter=adapter).search(
                    state, evaluator, seed=0
                )
                if search_result.simulations != mcts_sims:
                    raise GenerationFailed(
                        "Golden SequentialPUCT did not execute the requested mctsSims"
                    )
                if search_result.implementation_id != GOLDEN_SEARCH_IMPLEMENTATION_ID:
                    raise GenerationFailed("Golden search implementation provenance drifted")
                action = search_result.action
                if action not in search_result.legal_actions:
                    raise GenerationFailed("Golden search selected an illegal action")
                transition = apply_action(state, action)
            except GenerationFailed:
                raise
            except Exception as exc:
                raise GenerationFailed(f"Golden search/rules transition failed: {exc}") from exc

            if action != PASS:
                captures[0 if current_color == BLACK else 1] += len(transition.captured)
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
