"""Versioned AlphaZero -> GoCube product-boundary export.

The exporter deliberately uses the independent graph checker as its only
source of expected MAIN semantics.  It never asks a GoCube implementation to
produce the expected values that GoCube will later be tested against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .fixtures import VerificationFixture
from .independent_graph import BLACK, WHITE, IndependentIllegalMove, apply_move

PRODUCT_BOUNDARY_SCHEMA = "gocube-product-boundary-v1"
F0_CONTRACT_ID = "gocube-f0-integrated-freeze-v1"
ACTION_CONTRACT = "gocube-action-point-id-pass-v1"
KOMI = 0.5
VERIFIED_V1_SOURCES = frozenset(
    {
        "native_katago",
        "independent_graph",
        "exhaustive_solver",
        "metamorphic",
        "manual_reviewed_graph_proof",
        "hand_proved_invariant",
    }
)


@dataclass(frozen=True)
class ProductBoundaryFixture:
    fixture_id: str
    topology: str
    size: int
    komi: float
    initial_position: Mapping[str, tuple[str, ...]]
    main_action_sequence: tuple[str, ...]
    side_to_move: str
    expected_captures_during_main: tuple[Mapping[str, Any], ...]
    board_after_first_pass: Mapping[str, tuple[str, ...]]
    board_after_second_pass: Mapping[str, tuple[str, ...]]
    training_internal_cleanup: Mapping[str, Any]
    steps: tuple[Mapping[str, Any], ...] = ()
    source_verification_id: str | None = None
    source_verification_status: str = "verified"
    source_repo: str = "vmdvdv-npt/gocube-alphazero"
    f0_contract_id: str = F0_CONTRACT_ID
    action_contract: str = ACTION_CONTRACT
    topology_contract: Mapping[str, Any] = field(default_factory=dict)
    expected_product_boundary: Mapping[str, Any] = field(default_factory=dict)
    expected_endgame_classification: tuple[Mapping[str, Any], ...] = ()
    expected_final_score: Mapping[str, Any] | None = None
    rule_set: str = "japanese"
    schema: str = PRODUCT_BOUNDARY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        actions = [_action_record(action, self.topology_contract) for action in self.main_action_sequence]
        first = {
            "board": _jsonable(self.board_after_first_pass),
            "captures": _captures_at_step(self.steps, self.board_after_first_pass, 1),
            "player_to_move": _player_at_step(self.steps, 1),
            "consecutive_passes": 1,
        }
        second = {
            "board": _jsonable(self.board_after_second_pass),
            "captures": _captures_at_step(self.steps, self.board_after_second_pass, 2),
            "player_to_move": _player_at_step(self.steps, 2),
            "consecutive_passes": 2,
        }
        return {
            "schema": self.schema,
            "fixture_id": self.fixture_id,
            "source_verification_id": self.source_verification_id or self.fixture_id,
            "source_verification_status": self.source_verification_status,
            "provenance": {
                "source_repo": self.source_repo,
                "f0_contract_id": self.f0_contract_id,
                "v1_fixture_id": self.source_verification_id or self.fixture_id,
                "v1_status": self.source_verification_status,
            },
            "topology": self.topology,
            "size": self.size,
            "komi": self.komi,
            "rule_set": self.rule_set,
            "initial_position": _jsonable(self.initial_position),
            "initial_player": self.side_to_move,
            "main_actions": actions,
            "steps": _jsonable(self.steps),
            "after_first_pass": first,
            "after_second_pass": second,
            "expected_product_boundary": _jsonable(
                self.expected_product_boundary
                or {
                    "board": self.board_after_second_pass,
                    "captures": second["captures"],
                    "player_to_move": second["player_to_move"],
                    "consecutive_passes": 2,
                    "phase": "endgame",
                }
            ),
            "expected_endgame_classification": _jsonable(self.expected_endgame_classification),
            "expected_final_score": _jsonable(self.expected_final_score),
            "action_contract": self.action_contract,
            "topology_contract": _jsonable(self.topology_contract),
            # Kept as additive aliases for the pre-V2 groundwork readers.
            "MAIN_action_sequence": list(self.main_action_sequence),
            "side_to_move": self.side_to_move,
            "expected_captures_during_MAIN": _jsonable(self.expected_captures_during_main),
            "board_after_first_PASS": _jsonable(self.board_after_first_pass),
            "board_after_second_PASS": _jsonable(self.board_after_second_pass),
            "training_internal_cleanup": _jsonable(self.training_internal_cleanup),
        }


def export_product_boundary_fixture(
    fixture: VerificationFixture,
    topology: Any,
    *,
    source_verification_id: str | None = None,
    rule_set: str = "japanese",
    expected_endgame_classification: Iterable[Mapping[str, Any]] = (),
    expected_final_score: Mapping[str, Any] | None = None,
    require_verified: bool = False,
) -> ProductBoundaryFixture:
    """Materialize MAIN through the second PASS using the independent checker."""

    if fixture.topology_kind != topology.kind or fixture.size != topology.size:
        raise ValueError("Fixture and topology do not match")
    if require_verified and fixture.oracle not in VERIFIED_V1_SOURCES:
        raise ValueError(
            f"V2 export requires a verified V1 source; {fixture.id} has oracle {fixture.oracle!r}"
        )
    if require_verified and fixture.status != "verified":
        raise ValueError(f"V2 export requires verified V1 fixture: {fixture.id} ({fixture.status})")
    board = fixture.board(topology.index_by_id)
    adjacency = topology.neighbors_by_index
    player = BLACK if fixture.to_move == "black" else WHITE
    consecutive_passes = 0
    first_pass_board: tuple[int, ...] | None = None
    second_pass_board: tuple[int, ...] | None = None
    capture_counts = {"black": 0, "white": 0}
    captures = []
    steps: list[Mapping[str, Any]] = []
    for action_number, action in enumerate(fixture.actions):
        player_before = "black" if player == BLACK else "white"
        if action in ("PASS", "pass"):
            consecutive_passes += 1
            if consecutive_passes == 1:
                first_pass_board = board
            elif consecutive_passes == 2:
                second_pass_board = board
            next_player = WHITE if player == BLACK else BLACK
            steps.append(
                _step(
                    action_number,
                    player_before,
                    _pass_action(),
                    (),
                    board,
                    capture_counts,
                    "white" if next_player == WHITE else "black",
                    consecutive_passes,
                    topology,
                )
            )
            player = WHITE if player == BLACK else BLACK
            continue
        consecutive_passes = 0
        try:
            result = apply_move(board, player, topology.point_index(action), adjacency)
        except (KeyError, IndependentIllegalMove) as error:
            reason = getattr(error, "reason", str(error))
            raise ValueError(f"V2 fixture {fixture.id} has illegal MAIN action {action}: {reason}") from error
        captured_points = tuple(
            topology.point_id(point) for point in sorted(result.captured_points)
        )
        if result.captured_points:
            capture_counts[player_before] += len(result.captured_points)
            captures.append(
                {
                    "action_index": action_number,
                    "action": action,
                    "player": player_before,
                    "points": captured_points,
                }
            )
        board = result.board
        next_player = WHITE if player == BLACK else BLACK
        steps.append(
            _step(
                action_number,
                player_before,
                _place_action(action),
                captured_points,
                board,
                capture_counts,
                "white" if next_player == WHITE else "black",
                consecutive_passes,
                topology,
            )
        )
        player = next_player
    if first_pass_board is None or second_pass_board is None:
        raise ValueError("Product-boundary export requires two MAIN PASS actions")
    first_step = next(step for step in steps if step["consecutive_passes"] == 1)
    second_step = next(step for step in steps if step["consecutive_passes"] == 2)
    topology_contract = _topology_contract(topology)
    return ProductBoundaryFixture(
        fixture_id=fixture.id,
        topology=fixture.topology_kind,
        size=fixture.size,
        komi=KOMI,
        initial_position={"black": fixture.black, "white": fixture.white},
        main_action_sequence=fixture.actions,
        side_to_move=fixture.to_move,
        expected_captures_during_main=tuple(captures),
        board_after_first_pass=_board_colors(first_pass_board, topology),
        board_after_second_pass=_board_colors(second_pass_board, topology),
        training_internal_cleanup=fixture.cleanup_metadata.get("training_internal_cleanup", {}),
        steps=tuple(steps),
        source_verification_id=source_verification_id or fixture.source_id,
        source_verification_status=fixture.status,
        topology_contract=topology_contract,
        expected_product_boundary={
            "board": _board_colors(second_pass_board, topology),
            "captures": dict(capture_counts),
            "player_to_move": "white" if second_step["next_player"] == "white" else "black",
            "consecutive_passes": 2,
            "phase": "endgame",
        },
        expected_endgame_classification=tuple(expected_endgame_classification),
        expected_final_score=expected_final_score,
        rule_set=rule_set,
    )


def export_verified_product_boundary_fixtures(
    fixtures: Iterable[VerificationFixture],
    topology: Any,
    *,
    exclude_families: Iterable[str] = ("ko", "wrap_ko"),
) -> list[ProductBoundaryFixture]:
    """Export deterministic positive-boundary fixtures from verified V1 data.

    Existing V1 tactical fixtures often stop before the product boundary.  A
    pair of explicit MAIN passes is appended for this derived V2 artifact;
    the source verification ID remains the original V1 fixture ID.
    """

    excluded = set(exclude_families)
    exported = []
    for fixture in fixtures:
        if (
            fixture.phase != "main"
            or fixture.family in excluded
            or fixture.oracle not in VERIFIED_V1_SOURCES
            or fixture.status != "verified"
            or fixture.expected.get("legal") is False
        ):
            continue
        actions = tuple(fixture.actions)
        if not any(action in ("PASS", "pass") for action in actions):
            actions += ("PASS", "PASS")
        elif not _has_consecutive_passes(actions):
            actions += ("PASS", "PASS")
        source = fixture
        derived = VerificationFixture(
            **{**source.__dict__, "actions": actions, "source_fixture_id": source.source_id}
        )
        exported.append(
            export_product_boundary_fixture(
                derived,
                topology,
                source_verification_id=source.source_id,
                expected_final_score=source.expected.get("verified_final_score"),
                require_verified=True,
            )
        )
    return exported


def write_product_boundary_json(path: str | Path, fixtures: list[ProductBoundaryFixture]) -> None:
    payload = {"schema": PRODUCT_BOUNDARY_SCHEMA, "fixtures": [fixture.to_dict() for fixture in fixtures]}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _board_colors(board: tuple[int, ...], topology: Any) -> dict[str, tuple[str, ...]]:
    return {
        "black": tuple(topology.point_id(point) for point, value in enumerate(board) if value == BLACK),
        "white": tuple(topology.point_id(point) for point, value in enumerate(board) if value == WHITE),
    }


def _has_consecutive_passes(actions: Iterable[str]) -> bool:
    previous_pass = False
    for action in actions:
        current_pass = action in ("PASS", "pass")
        if current_pass and previous_pass:
            return True
        previous_pass = current_pass
    return False


def _place_action(point_id: str) -> dict[str, Any]:
    return {"type": "place", "point_id": point_id}


def _pass_action() -> dict[str, Any]:
    return {"type": "pass"}


def _action_record(action: str, topology_contract: Mapping[str, Any]) -> dict[str, Any]:
    if action in ("PASS", "pass"):
        return _pass_action()
    mapping = {
        item["alpha_zero_point_id"]: item
        for item in topology_contract.get("point_mapping", ())
    }
    record = mapping.get(action)
    if record is None:
        return _place_action(action)
    return {
        "type": "place",
        "point_id": record["product_point_id"],
        "alpha_zero_point_id": action,
        "alpha_zero_action_index": record["alpha_zero_action_index"],
    }


def _topology_contract(topology: Any) -> dict[str, Any]:
    points = tuple(topology.point_ids)
    return {
        "topology": topology.kind,
        "size": topology.size,
        "point_ids": list(points),
        "point_mapping": [
            {
                "alpha_zero_point_id": point,
                "product_point_id": point,
                "alpha_zero_action_index": index,
            }
            for index, point in enumerate(points)
        ],
        "adjacency": {
            point: [topology.point_id(neighbor) for neighbor in topology.neighbors_by_index[index]]
            for index, point in enumerate(points)
        },
    }


def _step(
    index: int,
    player_before: str,
    action: Mapping[str, Any],
    captured: Iterable[str],
    board: tuple[int, ...],
    captures: Mapping[str, int],
    next_player: str,
    consecutive_passes: int,
    topology: Any,
) -> Mapping[str, Any]:
    return {
        "index": index,
        "player_before": player_before,
        "action": dict(action),
        "legal": True,
        "captured_points": list(captured),
        "board": _board_colors(board, topology),
        "captures": dict(captures),
        "next_player": next_player,
        "consecutive_passes": consecutive_passes,
    }


def _captures_at_step(
    steps: tuple[Mapping[str, Any], ...],
    board: Mapping[str, Any],
    consecutive_passes: int,
) -> Mapping[str, int]:
    for step in steps:
        if step["consecutive_passes"] == consecutive_passes and step["board"] == board:
            return step["captures"]
    return {"black": 0, "white": 0}


def _player_at_step(steps: tuple[Mapping[str, Any], ...], consecutive_passes: int) -> str:
    for step in steps:
        if step["consecutive_passes"] == consecutive_passes:
            return str(step["next_player"])
    raise ValueError(f"Missing pass step {consecutive_passes}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value
