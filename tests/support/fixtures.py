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
        "manual_reviewed_graph_proof",
        "product_fixture_pending",
        "hand_proved_invariant",
        "structural-only",
    }
)

V1_STATUSES = frozenset(("verified", "explained_difference", "unresolved"))


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
    status: str = "verified"
    evidence: str = ""
    difference_id: str | None = None

    def __post_init__(self) -> None:
        if self.topology_kind not in ("cube", "torus", "rectangular-test"):
            raise ValueError(f"Unsupported fixture topology: {self.topology_kind}")
        if self.size < 1:
            raise ValueError("Fixture size must be positive")
        if self.to_move not in ("black", "white"):
            raise ValueError("to_move must be 'black' or 'white'")
        if self.oracle not in EXPECTED_SOURCES:
            raise ValueError(f"Unknown fixture provenance: {self.oracle}")
        if self.status not in V1_STATUSES:
            raise ValueError(f"Unknown V1 fixture status: {self.status}")
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
            "status": self.status,
            "evidence": self.evidence,
            "difference_id": self.difference_id,
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
    evidence = kwargs.pop("evidence", f"independent graph verification for {family}")
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
        evidence=evidence,
        **kwargs,
    )


def _cube4_black_except(*empty: str, white: Iterable[str] = ()) -> tuple[str, ...]:
    """Build a compact full-board eye fixture without renderer assumptions."""

    excluded = set(empty) | set(white)
    return tuple(
        f"{face}:{row}:{column}"
        for face in ("front", "back", "left", "right", "top", "bottom")
        for row in range(4)
        for column in range(4)
        if f"{face}:{row}:{column}" not in excluded
    )


def cube_verification_fixtures() -> tuple[VerificationFixture, ...]:
    """Return the small, reviewable Cube corpus used by CI.

    Every fixture in this registry is an accepted V1 evidence item.  The
    original groundwork registry was deliberately more conservative; the
    evidence fields below make the V1 upgrade explicit without changing the
    historical groundwork document.
    """

    q = "front:0:1"
    q_neighbors = ("top:3:1", "front:0:2", "front:1:1", "front:0:0")
    two_eye_black = _cube4_black_except("front:0:0", "back:2:2")
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
            expected={"status": "verified", "eye_kind": "closed_region_control", "structural": "one_empty_component", "anchor_points": ["front:1:2", "front:2:1"], "independent_proof": "independent empty-region decomposition"},
            evidence="independent empty-region decomposition",
            notes="Historical groundwork control retained in the V1 corpus; the canonical true-eye fixtures below use closed-board graph positions.",
        ),
        _cube4_fixture(
            "cube4_false_eye_control_001",
            "eyes",
            black=("front:0:1", "front:1:0", "front:1:2", "front:2:1"),
            white=("front:1:1",),
            expected={"status": "verified", "eye_kind": "false_eye", "structural": "mixed_border", "empty_anchor": "front:0:0", "independent_proof": "the empty component has both black and white graph borders"},
            oracle="independent_graph",
            evidence="independent empty-region border-color proof",
            notes="The mixed-border empty component is an independent false-eye control.",
        ),
        _cube4_fixture(
            "cube4_obvious_true_eye_pair_001",
            "eyes",
            black=_cube4_black_except("front:1:1", "back:1:1"),
            expected={"status": "verified", "eye_kind": "obvious_true_eye_pair", "eye_points": ["front:1:1", "back:1:1"], "independent_proof": "two disconnected empty components each bordered only by the same black group"},
            evidence="independent graph empty-region proof; both regions have only black boundary and are non-adjacent",
            notes="Obvious true-eye pair on the closed Cube surface.",
        ),
        _cube4_fixture(
            "cube4_vertex_true_eye_001",
            "eyes",
            black=_cube4_black_except("front:0:0", "front:2:2"),
            expected={"status": "verified", "eye_kind": "vertex_related", "eye_points": ["front:0:0", "front:2:2"], "independent_proof": "vertex empty region is isolated by graph adjacency and has only black boundary"},
            evidence="independent graph proof over the vertex degree-3 neighborhood",
            notes="Vertex-related true-eye pattern; no face/net coordinates are used by the oracle.",
        ),
        _cube4_fixture(
            "cube4_false_eye_graph_001",
            "eyes",
            black=_cube4_black_except("front:0:0", "front:0:1", white=("front:0:1",)),
            white=("front:0:1",),
            expected={"status": "verified", "eye_kind": "false_eye", "eye_point": "front:0:0", "independent_proof": "the empty point's graph region borders both black and white"},
            oracle="independent_graph",
            evidence="independent graph empty-region border-color proof",
            notes="False-eye control with a live white intruder retaining the empty point as its liberty.",
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
            black=_cube4_black_except(
                "front:1:1",
                "front:2:1",
                white=("front:1:2", "front:2:2"),
            ),
            white=("front:1:2", "front:2:2"),
            expected={
                "status": "verified",
                "seki_kind": "settled_mutual_two_liberty",
                "shared_empty_region": True,
                "shared_liberty_points": ["front:1:1", "front:2:1"],
                "independent_proof": "bounded exhaustive continuation proves neither side can force a capture",
                "search_status": "proved_draw",
                "search_depth": 3,
                "explored_nodes": 6,
                "defensive_reply": "the other shared liberty captures the first player's group",
            },
            oracle="exhaustive_solver",
            evidence="independent bounded exhaustive continuation plus explicit legal first-move/defensive-reply table; scoring is checked against pinned rectangular seki-tax",
            external_context={
                "katago_analog": {
                    "fixture_id": "seki-tax",
                    "katago_commit": "f6bc4b19a1686caa2d088b56251e8c11c8be6d51",
                    "assertion": "SCORED terminal with white-minus-black final score 0.5",
                }
            },
            notes="Closed Cube4 settled seki: the only two empty points are shared by one black and one white group; playing either liberty lets the opponent fill the other and capture the mover's group.",
        ),
        _cube4_fixture(
            "cube4_dame_neutral_region_001",
            "seki_dame",
            black=("front:0:0",),
            white=("back:0:0",),
            expected={"status": "verified", "seki_kind": "dame_control", "neutral_region": True, "anchor_points": ["front:1:1", "back:1:1"], "independent_proof": "one empty graph component has both black and white borders"},
            oracle="independent_graph",
            evidence="independent empty-region border-color proof",
            notes="Dame/neutral control; the expected neutral region is derived from graph borders.",
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
            black=_cube4_black_except("front:1:1", "back:1:1", "front:1:0", white=("front:1:0",)),
            white=("front:1:0",),
            expected={"pass_alive_area": "verified_black_area_with_intruder", "intruder": {"color": "white", "point": "front:1:0"}, "independent_proof": "two black-eye regions remain covered by the independently proven black group"},
            oracle="independent_graph",
            evidence="independent graph region/group proof; numeric score is covered by pinned S1 rectangular fixture",
            notes="The intruder position is accepted as a graph/topology fixture; pinned KataGo is not claimed as a Cube scorer oracle.",
        ),
        _cube4_fixture(
            "cube4_early_termination_boundary_001",
            "early_termination",
            actions=("PASS", "PASS"),
            expected={"after_second_pass": "cleanup1-boundary", "training_internal_cleanup": "separate", "independent_proof": "two accepted PASS actions preserve occupancy and consume turns", "verified_final_score": {"rule_set": "japanese", "black": 0, "white": 0.5, "komi": 0.5, "winner": "white", "margin": 0.5}},
            oracle="independent_graph",
            evidence="independent PASS transition proof; GoCube product lifecycle remains V2 scope",
            rotation_safe=False,
            rotation_policy="history-aware-only",
            notes="Future GoCube boundary fixture; no TypeScript engine is changed or consulted here.",
            cleanup_metadata={"training_internal_cleanup": {"actions": [], "phase_transitions": [], "final_training_result": None}},
        ),
        _cube4_fixture(
            "cube4_nonempty_two_eye_score_001",
            "endgame_score",
            black=two_eye_black,
            expected={
                "status": "verified",
                "independent_proof": "one connected black group has two exclusive graph-vital regions",
                "verified_endgame_classification": ({"points": two_eye_black, "status": "alive"},),
                "verified_final_score": {
                    "rule_set": "japanese",
                    "black": 2.0,
                    "white": 0.5,
                    "komi": 0.5,
                    "territory": {"black": 2, "white": 0, "neutral": 0, "seki": 0},
                    "stones_on_board": {"black": 94, "white": 0},
                    "captures": [0, 0],
                    "prisoners": [0, 0],
                    "dead_stones": {"black": 0, "white": 0},
                    "winner": "black",
                    "margin": 1.5,
                },
            },
            oracle="independent_graph",
            evidence="independent graph proof of two exclusive black territory regions plus test-only Japanese territory arithmetic; production score_position is compared against this expected result",
            notes="Non-empty product scoring boundary: the single logical black group is manually marked alive after two MAIN passes; both graph-proven eyes become black territory.",
        ),
    )
    return fixtures


def fixture_counts(fixtures: Iterable[VerificationFixture]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fixture in fixtures:
        counts[fixture.family] = counts.get(fixture.family, 0) + 1
    return dict(sorted(counts.items()))


def torus_verification_fixtures() -> tuple[VerificationFixture, ...]:
    """Return the compact Torus 9 acceptance corpus.

    A rectangular KataGo board cannot serve as the expected result for these
    cases.  Each expected capture/group result is derived from the raw
    wrap-around adjacency by :mod:`independent_graph` and then compared with
    the production V3 transition.
    """

    def torus(
        fixture_id: str,
        family: str,
        *,
        black: Iterable[str] = (),
        white: Iterable[str] = (),
        to_move: str = "black",
        actions: Iterable[str] = (),
        expected: Mapping[str, Any] | None = None,
        evidence: str,
        notes: str,
        rotation_policy: str = "torus_dihedral_smoke",
    ) -> VerificationFixture:
        return VerificationFixture(
            id=fixture_id,
            family=family,
            topology_kind="torus",
            size=9,
            black=tuple(black),
            white=tuple(white),
            to_move=to_move,
            actions=tuple(actions),
            expected=dict(expected or {}),
            oracle="independent_graph",
            rotation_policy=rotation_policy,
            notes=notes,
            evidence=evidence,
        )

    return (
        torus(
            "torus9_wrap_group_001",
            "wrap_group",
            black=("0,4", "8,4"),
            expected={"group": ["0,4", "8,4"], "wrap_axis": "horizontal", "status": "verified"},
            evidence="independent graph connectivity across x=0/x=8 seam",
            notes="The two points are adjacent only through Torus wrap-around.",
        ),
        torus(
            "torus9_wrap_capture_001",
            "wrap_capture",
            black=("8,4", "1,4", "0,3"),
            white=("0,4",),
            actions=("0,5",),
            expected={"captured": ["0,4"], "capture_count": 1, "legal": True, "status": "verified"},
            evidence="independent graph placement/capture over the horizontal wrap edge",
            notes="The final liberty of the white stone is filled from the opposite x seam.",
        ),
        torus(
            "torus9_wrap_ko_001",
            "wrap_ko",
            black=("1,4", "0,3", "0,5"),
            white=("0,4", "7,4", "8,3", "8,5"),
            actions=("8,4", "0,4"),
            expected={"captured": ["0,4"], "recapture_legal_without_ko": True, "restores_initial_board": True, "simple_ko": True, "status": "verified"},
            evidence="independent positional-restoration proof across horizontal Torus wrap",
            notes="The simple-ko proof uses only occupancy and Torus adjacency; production ko policy is checked separately.",
        ),
        torus(
            "torus9_no_cube_triangles_001",
            "topology_invariant",
            expected={"graph_triangles": 0, "degree": 4, "status": "verified"},
            evidence="independent graph clique scan and degree profile",
            notes="Torus has wrap-around edges but no Cube vertex triangles.",
        ),
    )


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
