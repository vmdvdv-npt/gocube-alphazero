"""Stable, source-labelled verification fixtures for Cube and product edges."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .independent_graph import board_from_colors

EXPECTED_SOURCES = frozenset(
    {
        "native_katago",
        "independent_graph",
        "exhaustive_solver",
        "metamorphic",
        "product_fixture_pending",
        "hand_proved_invariant",
        "structural-only",
    }
)


@dataclass(frozen=True)
class VerificationFixture:
    id: str
    family: str
    topology_kind: str
    size: int
    black: tuple[str, ...] = ()
    white: tuple[str, ...] = ()
    to_move: str = "black"
    actions: tuple[str, ...] = ()
    expected: Mapping[str, Any] = field(default_factory=dict)
    oracle: str = "independent_graph"
    rotation_policy: str = "all_24"
    rotation_safe: bool = True
    notes: str = ""
    previous_board: tuple[int, ...] | None = None
    phase: str = "main"
    captures: tuple[int, int] = (0, 0)
    ko_state: Mapping[str, Any] = field(default_factory=dict)
    cleanup_metadata: Mapping[str, Any] = field(default_factory=dict)
    external_context: Mapping[str, Any] = field(default_factory=dict)
    second_cleanup_start_colors: tuple[int, ...] | None = None
    seed: int | None = None
    generator_version: str | None = None
    source_fixture_id: str | None = None
    rotation_index: int | None = None

    def __post_init__(self) -> None:
        if self.topology_kind not in ("cube", "torus", "rectangular-test"):
            raise ValueError(f"Unsupported fixture topology: {self.topology_kind}")
        if self.size < 1:
            raise ValueError("Fixture size must be positive")
        if self.to_move not in ("black", "white"):
            raise ValueError("to_move must be 'black' or 'white'")
        if self.oracle not in EXPECTED_SOURCES:
            raise ValueError(f"Unknown fixture provenance: {self.oracle}")
        if set(self.black) & set(self.white):
            raise ValueError(f"Fixture has overlapping stones: {self.id}")
        if any(action not in ("PASS", "pass") and not isinstance(action, str) for action in self.actions):
            raise ValueError("Fixture actions must be canonical point IDs or PASS")

    @property
    def source_id(self) -> str:
        return self.source_fixture_id or self.id

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "family": self.family,
            "topology_kind": self.topology_kind,
            "size": self.size,
            "initial_position": {"black": list(self.black), "white": list(self.white)},
            "black": list(self.black),
            "white": list(self.white),
            "to_move": self.to_move,
            "actions": list(self.actions),
            "expected": _jsonable(self.expected),
            "oracle/source": self.oracle,
            "rotation_policy": self.rotation_policy,
            "rotation_safe": self.rotation_safe,
            "notes": self.notes,
            "previous_board": None if self.previous_board is None else list(self.previous_board),
            "phase": self.phase,
            "captures": list(self.captures),
            "ko_state": _jsonable(self.ko_state),
            "cleanup_metadata": _jsonable(self.cleanup_metadata),
            "external_context": _jsonable(self.external_context),
            "second_cleanup_start_colors": None if self.second_cleanup_start_colors is None else list(self.second_cleanup_start_colors),
            "seed": self.seed,
            "generator_version": self.generator_version,
            "source_fixture_id": self.source_fixture_id,
            "rotation_index": self.rotation_index,
        }
        return data

    def board(self, point_id_to_index: Mapping[str, int]) -> tuple[int, ...]:
        return board_from_colors(
            len(point_id_to_index),
            (point_id_to_index[point] for point in self.black),
            (point_id_to_index[point] for point in self.white),
        )

    def action_indices(self, point_id_to_index: Mapping[str, int], pass_action: int) -> tuple[int, ...]:
        return tuple(
            pass_action if action in ("PASS", "pass") else point_id_to_index[action]
            for action in self.actions
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset, tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _cube4_fixture(
    fixture_id: str,
    family: str,
    *,
    black: Iterable[str] = (),
    white: Iterable[str] = (),
    to_move: str = "black",
    actions: Iterable[str] = (),
    expected: Mapping[str, Any] | None = None,
    oracle: str = "independent_graph",
    notes: str = "",
    rotation_safe: bool = True,
    rotation_policy: str = "all_24",
    **kwargs: Any,
) -> VerificationFixture:
    return VerificationFixture(
        id=fixture_id,
        family=family,
        topology_kind="cube",
        size=4,
        black=tuple(black),
        white=tuple(white),
        to_move=to_move,
        actions=tuple(actions),
        expected=dict(expected or {}),
        oracle=oracle,
        notes=notes,
        rotation_safe=rotation_safe,
        rotation_policy=rotation_policy,
        **kwargs,
    )


def cube_verification_fixtures() -> tuple[VerificationFixture, ...]:
    """Return the small, reviewable Cube corpus used by CI.

    Complex life-and-death fixtures intentionally carry structural-only or
    product-pending provenance.  They are inputs for later proof work, not
    claims that the current scorer is correct.
    """

    q = "front:0:1"
    q_neighbors = ("top:3:1", "front:0:2", "front:1:1", "front:0:0")
    fixtures = (
        _cube4_fixture(
            "cube4_vertex_single_group_001",
            "vertex_groups",
            black=("front:0:0", "left:0:3"),
            expected={"group": ["front:0:0", "left:0:3"], "liberty_count": 5},
            notes="Two canonical points meet across a physical vertex triangle; liberties are unique graph points.",
        ),
        _cube4_fixture(
            "cube4_vertex_three_face_group_001",
            "vertex_groups",
            black=("front:0:0", "left:0:3", "top:3:0"),
            expected={"group_count": 1, "vertex_triangle": ["front:0:0", "left:0:3", "top:3:0"]},
            notes="A group crosses all three faces incident to one vertex.",
        ),
        _cube4_fixture(
            "cube4_vertex_near_miss_001",
            "vertex_groups",
            black=("front:0:1", "left:0:2"),
            expected={"group_count": 2, "not_adjacent": True, "control_points": ["front:0:1", "left:0:2"]},
            notes="Visually close face points are deliberately not connected by the canonical graph.",
        ),
        _cube4_fixture(
            "cube4_inner_shared_liberty_001",
            "vertex_groups",
            black=("front:1:1", "front:1:2"),
            expected={"group": ["front:1:1", "front:1:2"], "liberty_count": 6},
            notes="Connected interior stones share liberties; duplicate incident edges must not inflate the count.",
        ),
        _cube4_fixture(
            "cube4_vertex_capture_001",
            "captures",
            black=("front:0:1", "front:1:0", "left:0:3"),
            white=("front:0:0",),
            actions=("top:3:0",),
            expected={"captured": ["front:0:0"], "capture_count": 1, "legal": True},
            notes="Single-stone capture at a three-face vertex.",
        ),
        _cube4_fixture(
            "cube4_vertex_capture_before_suicide_001",
            "captures",
            black=(
                "front:0:1", "front:1:0", "left:0:2", "left:1:3",
                "top:1:0", "top:2:1", "top:3:2",
            ),
            white=("front:0:0", "left:0:3", "top:2:0", "top:3:1"),
            actions=("top:3:0",),
            expected={"captured_group_count": 3, "capture_count": 4, "legal": True, "own_liberties_after": 4, "captured_anchor": "front:0:0"},
            notes="The played point has no liberty before removal; it is legal only because capture precedes suicide checking.",
        ),
        _cube4_fixture(
            "cube4_vertex_multiple_neighbor_capture_001",
            "captures",
            black=(
                "front:0:3", "front:1:0", "front:1:2", "front:2:1",
                "left:0:3", "top:2:1", "top:3:0", "top:3:2",
            ),
            white=q_neighbors,
            actions=(q,),
            expected={"captured_group_count": 4, "capture_count": 4, "same_group_not_double_counted": True, "legal": True, "captured_neighbors": list(q_neighbors)},
            notes="Four independent opponent groups touch a vertex-adjacent move; each group is removed once.",
        ),
        _cube4_fixture(
            "cube4_vertex_suicide_control_001",
            "captures",
            black=q_neighbors,
            to_move="white",
            actions=(q,),
            expected={"legal": False, "reason": "suicide", "capture_count": 0, "action": q},
            notes="Control for capture-before-suicide: same local shape with no capturable opponent group.",
        ),
        _cube4_fixture(
            "cube4_two_independent_eyes_001",
            "eyes",
            black=("front:1:1", "front:1:3", "front:3:1", "front:3:3"),
            expected={"status": "unknown", "structural": "two_empty_components", "anchor_points": ["front:1:2", "front:2:1"]},
            oracle="structural-only",
            notes="Only empty-region connectivity and borders are asserted; no alive/dead conclusion is made.",
        ),
        _cube4_fixture(
            "cube4_false_eye_control_001",
            "eyes",
            black=("front:0:1", "front:1:0", "front:1:2", "front:2:1"),
            white=("front:1:1",),
            expected={"status": "unknown", "structural": "mixed_border", "empty_anchor": "front:0:0"},
            oracle="structural-only",
            notes="A false-eye-shaped control; this fixture does not claim life-and-death status.",
        ),
        _cube4_fixture(
            "cube4_seam_region_001",
            "eyes",
            black=("front:0:1", "left:0:2"),
            expected={"region_count": "independent_graph", "crosses_seam": True, "anchor_points": ["front:0:1", "left:0:2"]},
            notes="Empty connectivity is measured on the closed surface, not on a flat cube net.",
        ),
        _cube4_fixture(
            "cube4_seki_shared_liberty_001",
            "seki_dame",
            black=("front:1:1", "front:1:2"),
            white=("front:2:1", "front:2:2"),
            expected={"status": "structural-only", "shared_empty_region": True, "score": "unknown", "shared_liberty_points": ["front:0:1"]},
            oracle="structural-only",
            notes="Shared liberties are recorded for later seki proof; no production pass-alive result is used.",
        ),
        _cube4_fixture(
            "cube4_dame_neutral_region_001",
            "seki_dame",
            black=("front:0:0",),
            white=("back:0:0",),
            expected={"status": "structural-only", "neutral_region": True, "score": "unknown", "anchor_points": ["front:1:1", "back:1:1"]},
            oracle="structural-only",
            notes="Control with both colors bordering one connected empty component; neutrality is structural, not a score claim.",
        ),
        _cube4_fixture(
            "cube4_false_simple_ko_001",
            "ko",
            black=("front:0:1", "front:1:0", "front:1:2", "front:2:0"),
            white=("front:1:1",),
            actions=("front:2:1", "front:1:1"),
            expected={"capture_action": "front:2:1", "captured": ["front:1:1"], "recapture_legal": False, "recapture_reason": "suicide", "simple_ko": False},
            oracle="independent_graph",
            notes="Reviewed M1 false positive: two board points change, but the apparent recapture is suicide and does not restore the position.",
            ko_state={"production_heuristic": "diagnostic-only", "known_on_revision": "842de66323e1bd20caaac4cec3df007608190611"},
        ),
        _cube4_fixture(
            "cube4_true_simple_ko_001",
            "ko",
            black=("front:1:0", "left:0:3", "top:3:0"),
            white=("front:0:0", "front:0:2", "front:1:1", "top:3:1"),
            actions=("front:0:1", "front:0:0"),
            expected={"capture_action": "front:0:1", "captured": ["front:0:0"], "recapture_legal_without_ko": True, "restores_initial_board": True, "simple_ko": True},
            notes="The independent capture checker proves positional restoration; the ko prohibition itself remains a separate policy check.",
        ),
        _cube4_fixture(
            "cube4_seam_local_isomorphism_001",
            "seam_tactics",
            black=("front:0:0", "left:0:3"),
            expected={"induced_edges": [["front:0:0", "left:0:3"]], "control": ["front:0:1", "left:0:2"]},
            notes="Positive seam edge and visually-near non-edge are paired in one fixture.",
        ),
        _cube4_fixture(
            "cube4_global_path_three_faces_001",
            "global_connectivity",
            black=("front:0:0", "left:0:3", "top:3:0"),
            expected={"group_count": 1, "path_faces": ["front", "left", "top"], "path_points": ["front:0:0", "left:0:3", "top:3:0"]},
            notes="The group is connected through a three-face vertex triangle.",
        ),
        _cube4_fixture(
            "cube4_global_cut_group_001",
            "global_connectivity",
            black=("front:0:0", "front:0:1", "front:0:2"),
            expected={"before_group_count": 1, "after_removing": {"front:0:1": 2}},
            notes="A graph cut is described structurally; the expected split is checked by independent flood-fill.",
        ),
        _cube4_fixture(
            "cube4_cleanup1_pass_for_ko_001",
            "cleanup",
            black=("front:1:0", "left:0:3", "top:3:0"),
            white=("front:0:0", "front:0:2", "front:1:1", "top:3:1"),
            actions=("front:0:1", "PASS", "PASS"),
            expected={"main_until_first_pass": True, "cleanup_record": "separate", "ko_local_property": "true"},
            phase="cleanup1",
            cleanup_metadata={"training_internal_cleanup": {"actions": [], "phase_transitions": []}},
            rotation_safe=False,
            rotation_policy="history-aware-only",
            notes="Product-boundary groundwork: MAIN and cleanup context are kept separate; full phase proof is future acceptance.",
        ),
        _cube4_fixture(
            "cube4_cleanup2_pass_for_ko_001",
            "cleanup",
            black=("front:1:0", "left:0:3", "top:3:0"),
            white=("front:0:0", "front:0:2", "front:1:1", "top:3:1"),
            actions=("front:0:1", "PASS", "PASS"),
            expected={"main_after_second_pass": True, "cleanup_stage": 2, "cleanup_record": "separate"},
            phase="cleanup2",
            cleanup_metadata={"training_internal_cleanup": {"actions": [], "phase_transitions": []}},
            rotation_safe=False,
            rotation_policy="history-aware-only",
            notes="Second-cleanup placeholder preserving the board/history boundary for the later product comparison.",
        ),
        _cube4_fixture(
            "cube4_pass_alive_intruder_001",
            "s1_intruder_support",
            black=("front:0:0", "front:0:1", "front:1:0", "left:0:3", "top:3:0"),
            white=("front:2:2",),
            expected={"pass_alive_area": "pending_independent_proof", "intruder": {"color": "white", "point": "front:2:2"}, "score": "unknown"},
            oracle="product_fixture_pending",
            notes="S1 support only: preserves an intruder-shaped board and independent graph facts; it is not a scorer oracle.",
        ),
        _cube4_fixture(
            "cube4_early_termination_boundary_001",
            "early_termination",
            actions=("PASS", "PASS"),
            expected={"after_second_pass": "product-boundary", "training_internal_cleanup": "separate"},
            oracle="product_fixture_pending",
            rotation_safe=False,
            rotation_policy="history-aware-only",
            notes="Future GoCube boundary fixture; no TypeScript engine is changed or consulted here.",
            cleanup_metadata={"training_internal_cleanup": {"actions": [], "phase_transitions": [], "final_training_result": None}},
        ),
    )
    return fixtures


def fixture_counts(fixtures: Iterable[VerificationFixture]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fixture in fixtures:
        counts[fixture.family] = counts.get(fixture.family, 0) + 1
    return dict(sorted(counts.items()))


def assert_rotation_split_consistency(
    fixtures: Iterable[VerificationFixture],
    split_by_fixture_id: Mapping[str, str],
) -> None:
    for fixture in fixtures:
        source = fixture.source_id
        if fixture.id not in split_by_fixture_id:
            continue
        if source not in split_by_fixture_id:
            raise AssertionError(f"Source fixture {source} has no corpus split")
        if split_by_fixture_id[source] != split_by_fixture_id[fixture.id]:
            raise AssertionError(f"Rotations of {source} cross corpus splits")


def write_fixture_json(path: str | Path, fixtures: Iterable[VerificationFixture]) -> None:
    payload = [fixture.to_dict() for fixture in fixtures]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
