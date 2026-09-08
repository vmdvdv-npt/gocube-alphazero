"""Export format for future GoCube TypeScript MAIN-boundary comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .fixtures import VerificationFixture
from .independent_graph import BLACK, EMPTY, WHITE, IndependentIllegalMove, apply_move

PRODUCT_BOUNDARY_SCHEMA = "gocube-product-boundary-v1"
KOMI = 0.5


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
    schema: str = PRODUCT_BOUNDARY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fixture_id": self.fixture_id,
            "topology": self.topology,
            "size": self.size,
            "komi": self.komi,
            "initial_position": _jsonable(self.initial_position),
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
) -> ProductBoundaryFixture:
    """Materialize MAIN through the second PASS using the independent checker."""

    if fixture.topology_kind != topology.kind or fixture.size != topology.size:
        raise ValueError("Fixture and topology do not match")
    board = fixture.board(topology.index_by_id)
    adjacency = topology.neighbors_by_index
    player = BLACK if fixture.to_move == "black" else WHITE
    pass_count = 0
    first_pass_board: tuple[int, ...] | None = None
    second_pass_board: tuple[int, ...] | None = None
    captures = []
    for action_number, action in enumerate(fixture.actions):
        if action in ("PASS", "pass"):
            pass_count += 1
            if pass_count == 1:
                first_pass_board = board
            elif pass_count == 2:
                second_pass_board = board
            player = WHITE if player == BLACK else BLACK
            continue
        result = apply_move(board, player, topology.point_index(action), adjacency)
        if result.captured_points:
            captures.append(
                {
                    "action_index": action_number,
                    "action": action,
                    "player": "black" if player == BLACK else "white",
                    "points": tuple(topology.point_id(point) for point in sorted(result.captured_points)),
                }
            )
        board = result.board
        player = WHITE if player == BLACK else BLACK
    if first_pass_board is None or second_pass_board is None:
        raise ValueError("Product-boundary export requires two MAIN PASS actions")
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
    )


def write_product_boundary_json(path: str | Path, fixtures: list[ProductBoundaryFixture]) -> None:
    payload = {"schema": PRODUCT_BOUNDARY_SCHEMA, "fixtures": [fixture.to_dict() for fixture in fixtures]}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _board_colors(board: tuple[int, ...], topology: Any) -> dict[str, tuple[str, ...]]:
    return {
        "black": tuple(topology.point_id(point) for point, value in enumerate(board) if value == BLACK),
        "white": tuple(topology.point_id(point) for point, value in enumerate(board) if value == WHITE),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value
