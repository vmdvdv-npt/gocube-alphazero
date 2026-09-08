"""V1 acceptance tests: independent graph oracle plus pinned KataGo."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from alphazero.envs.gocube.core import (
    BLACK,
    EMPTY,
    WHITE,
    GroupClassification,
    cube_topology,
    initial_state,
    score_position,
    torus_topology,
)
from alphazero.envs.gocube.katago_v3 import (
    CLEANUP_1,
    CLEANUP_2,
    MAIN,
    NO_RESULT,
    SCORED,
    V3IllegalMove,
    _all_groups,
    _collect_group,
    _empty_regions,
    apply_v3_action,
    final_v3_score,
    independent_life_analysis,
    is_simple_ko_state,
    pass_alive_analysis,
    terminal_from_state,
    v3_state_from_board,
    v3_valid_moves,
)
from tests.support.fixtures import (
    cube_verification_fixtures,
    torus_verification_fixtures,
)
from tests.support.independent_graph import (
    IndependentIllegalMove,
    apply_move,
    empty_regions,
    find_group,
    find_groups,
    graph_triangles,
)
from tests.support.independent_endgame import (
    mixed_border_regions,
    prove_opponent_placement_exhaustion,
    prove_settled_seki,
    prove_two_vital_regions,
)
from tests.support.independent_rules import (
    CLEANUP_1 as GRAPH_CLEANUP_1,
    CLEANUP_2 as GRAPH_CLEANUP_2,
    MAIN as GRAPH_MAIN,
    SCORED as GRAPH_SCORED,
    apply_action as graph_apply_action,
    initial_rule_state,
    legal_actions as graph_legal_actions,
)
from tests.support.katago_differential import (
    V1_GENERATOR_VERSION,
    V1_LENGTHS,
    V1_SEEDS,
    run_generated_rectangular_differential,
)
from tests.support.ko import prove_positional_restoration
from tests.support.v1_matrix import REQUIRED_FAMILIES, assert_matrix_accepted, canonical_matrix
from tests.support.rotations import cube_rotations, rotate_fixture
from katago_reference_runner import run_fixture


def _fixture(fixture_id):
    return next(item for item in cube_verification_fixtures() if item.id == fixture_id)


def _state_for_fixture(fixture, topology):
    board = fixture.board(topology.index_by_id)
    return v3_state_from_board(
        topology,
        black=np.flatnonzero(np.asarray(board) == BLACK),
        white=np.flatnonzero(np.asarray(board) == WHITE),
        current_player=0 if fixture.to_move == "black" else 1,
        phase=fixture.phase,
        captures=fixture.captures,
        previous_board=fixture.previous_board,
        ko_recap_blocked=fixture.ko_state.get("blocked_points", ()),
        second_cleanup_start_colors=fixture.second_cleanup_start_colors,
    )


def _action_index(action, topology):
    return topology.pass_action if action in ("PASS", "pass") else topology.point_index(action)


def _assert_graph_and_production_groups(board, topology):
    independent = find_groups(board, topology.neighbors_by_index)
    production = _all_groups(np.asarray(board, dtype=np.uint8), topology)
    assert {group.stones for group in independent} == {frozenset(group) for group in production}
    for group in independent:
        start = min(group.stones)
        _stones, liberties = _collect_group(np.asarray(board, dtype=np.uint8), start, group.color, topology)
        assert liberties == set(group.liberties)


def test_v1_acceptance_registry_has_only_verified_evidence_and_no_production_expected():
    fixtures = cube_verification_fixtures() + torus_verification_fixtures()
    assert fixtures
    assert all(fixture.status in ("verified", "explained_difference") for fixture in fixtures)
    assert all(fixture.oracle != "production_expected" for fixture in fixtures)
    assert all(fixture.evidence for fixture in fixtures if fixture.family not in ("vertex_groups", "captures", "ko", "global_connectivity", "seam_tactics"))
    assert not any("unknown" in repr(fixture.expected).lower() for fixture in fixtures)


def test_independent_rule_oracle_models_pass_history_and_phase_boundaries():
    topology = cube_topology(2)
    state = initial_rule_state((EMPTY,) * topology.point_count)
    for expected_phase, expected_passes in (
        (GRAPH_MAIN, 1),
        (GRAPH_CLEANUP_1, 0),
        (GRAPH_CLEANUP_1, 1),
        (GRAPH_CLEANUP_2, 0),
        (GRAPH_CLEANUP_2, 1),
        (GRAPH_SCORED, 2),
    ):
        assert topology.pass_action in graph_legal_actions(
            state, topology.neighbors_by_index, pass_action=topology.pass_action
        )
        state = graph_apply_action(
            state,
            topology.pass_action,
            topology.neighbors_by_index,
            pass_action=topology.pass_action,
        )
        assert state.phase == expected_phase
        assert state.consecutive_passes == expected_passes
    assert state.terminal_kind == "scored"


def test_independent_support_does_not_import_production_rule_helpers():
    import ast
    from pathlib import Path

    for name in ("independent_graph.py", "independent_rules.py", "independent_endgame.py"):
        tree = ast.parse((Path(__file__).parent / "support" / name).read_text(encoding="utf-8"))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(module.startswith("alphazero.envs.gocube") for module in modules)


@pytest.mark.parametrize("size", (2, 3, 4, 5, 6, 7))
def test_cube_topology_v1_invariants(size):
    topology = cube_topology(size)
    triangles = graph_triangles(topology.neighbors_by_index)
    assert len(triangles) == 8
    assert all(len(topology.neighbor_indices(point)) == 4 for point in range(topology.point_count))
    if size == 4:
        participating = {point for triangle in triangles for point in triangle}
        assert topology.point_count == 96
        assert len(participating) == 24


@pytest.mark.parametrize("size", (9, 13, 19))
def test_torus_topology_v1_has_wrap_degree_four_and_no_cube_triangles(size):
    topology = torus_topology(size)
    assert len(graph_triangles(topology.neighbors_by_index)) == 0
    assert all(len(topology.neighbor_indices(point)) == 4 for point in range(topology.point_count))
    assert topology.point_index(f"0,{size // 2}") in topology.neighbor_indices(topology.point_index(f"{size - 1},{size // 2}"))


@pytest.mark.parametrize("fixture", cube_verification_fixtures(), ids=lambda item: item.id)
def test_cube_graph_groups_regions_and_main_transitions_match_production(fixture):
    topology = cube_topology(fixture.size)
    board = fixture.board(topology.index_by_id)
    _assert_graph_and_production_groups(board, topology)

    independent_regions = empty_regions(board, topology.neighbors_by_index)
    production_regions = _empty_regions(np.asarray(board, dtype=np.uint8), topology)
    assert {region.points for region in independent_regions} == {frozenset(region) for region in production_regions}

    if not fixture.actions:
        return
    independent = initial_rule_state(
        board,
        current_player=BLACK if fixture.to_move == "black" else WHITE,
        phase=GRAPH_MAIN if fixture.phase == MAIN else fixture.phase,
        captures=fixture.captures,
        previous_board=fixture.previous_board,
    )
    production = _state_for_fixture(fixture, topology)
    for action_name in fixture.actions:
        action = _action_index(action_name, topology)
        # The independent graph layer is a full legal-mask oracle only for
        # normal play. Cleanup PASS-for-ko is intentionally KataGo-only.
        if independent.phase == GRAPH_MAIN:
            graph_legal = action in graph_legal_actions(independent, topology.neighbors_by_index, pass_action=topology.pass_action)
            production_legal = bool(v3_valid_moves(production, topology)[action])
            assert production_legal == graph_legal, f"{fixture.id}: action {action_name} legal mismatch"
        try:
            independent_next = graph_apply_action(
                independent,
                action,
                topology.neighbors_by_index,
                pass_action=topology.pass_action,
            )
            independent_error = None
        except IndependentIllegalMove as error:
            independent_next = None
            independent_error = error
        try:
            production_next = apply_v3_action(production, action, topology)
            production_error = None
        except V3IllegalMove as error:
            production_next = None
            production_error = error
        if independent_error is not None:
            assert production_error is not None, f"{fixture.id}: production accepted {action_name!r}"
            independent = independent
            production = production
            continue
        assert production_error is None, f"{fixture.id}: production rejected independent legal action {action_name!r}: {production_error}"
        assert independent_next is not None and production_next is not None
        assert tuple(int(value) for value in production_next.board) == independent_next.board
        assert production_next.current_player == (0 if independent_next.current_player == BLACK else 1)
        assert production_next.captures == independent_next.captures
        assert production_next.phase == {
            GRAPH_MAIN: MAIN,
            GRAPH_CLEANUP_1: CLEANUP_1,
            GRAPH_CLEANUP_2: CLEANUP_2,
            GRAPH_SCORED: SCORED,
        }.get(independent_next.phase, production_next.phase)
        independent, production = independent_next, production_next


@pytest.mark.parametrize(
    "fixture_id, expected_simple_ko",
    (
        ("cube4_true_simple_ko_001", True),
        ("cube4_false_simple_ko_001", False),
    ),
)
def test_cube_ko_is_independently_proved_and_production_matches(fixture_id, expected_simple_ko):
    fixture = _fixture(fixture_id)
    topology = cube_topology(4)
    proof = prove_positional_restoration(
        fixture.board(topology.index_by_id),
        BLACK if fixture.to_move == "black" else WHITE,
        topology.point_index(fixture.actions[0]),
        topology.point_index(fixture.actions[1]),
        topology.neighbors_by_index,
    )
    assert proof.is_simple_ko is expected_simple_ko
    assert proof.recapture_legal_without_ko is fixture.expected.get("recapture_legal_without_ko", False)
    state = _state_for_fixture(fixture, topology)
    captured = apply_v3_action(state, topology.point_index(fixture.actions[0]), topology)
    assert is_simple_ko_state(captured, topology) is expected_simple_ko
    assert v3_valid_moves(captured, topology)[topology.point_index(fixture.actions[1])] == (0 if expected_simple_ko else 0)
    if expected_simple_ko:
        with pytest.raises(V3IllegalMove, match="simple-ko"):
            apply_v3_action(captured, topology.point_index(fixture.actions[1]), topology)
    else:
        with pytest.raises(V3IllegalMove, match="suicide"):
            apply_v3_action(captured, topology.point_index(fixture.actions[1]), topology)


@pytest.mark.parametrize("fixture", torus_verification_fixtures(), ids=lambda item: item.id)
def test_torus_graph_corpus_matches_production(fixture):
    topology = torus_topology(9)
    board = fixture.board(topology.index_by_id)
    _assert_graph_and_production_groups(board, topology)
    if fixture.family == "topology_invariant":
        assert len(graph_triangles(topology.neighbors_by_index)) == fixture.expected["graph_triangles"]
        return
    if not fixture.actions:
        return
    if fixture.family == "wrap_ko":
        proof = prove_positional_restoration(
            board,
            BLACK,
            topology.point_index(fixture.actions[0]),
            topology.point_index(fixture.actions[1]),
            topology.neighbors_by_index,
        )
        assert proof.is_simple_ko
    else:
        result = apply_move(
            board,
            BLACK,
            topology.point_index(fixture.actions[0]),
            topology.neighbors_by_index,
        )
        assert {topology.point_id(point) for point in result.captured_points} == set(fixture.expected.get("captured", ()))
        state = _state_for_fixture(fixture, topology)
        production = apply_v3_action(state, topology.point_index(fixture.actions[0]), topology)
        assert tuple(int(value) for value in production.board) == result.board
        assert production.captures == (result.capture_count, 0)


def test_v1_eye_proofs_are_graph_facts_and_production_follows_them():
    topology = cube_topology(4)
    for fixture_id in ("cube4_obvious_true_eye_pair_001", "cube4_vertex_true_eye_001", "cube4_false_eye_graph_001"):
        fixture = _fixture(fixture_id)
        board = fixture.board(topology.index_by_id)
        regions = empty_regions(board, topology.neighbors_by_index)
        point_ids = {topology.point_id(point) for region in regions for point in region.points}
        expected_points = set(fixture.expected.get("eye_points", ())) or {fixture.expected["eye_point"]}
        assert expected_points.issubset(point_ids)
        state = _state_for_fixture(fixture, topology)
        # Production helpers are the actual side of this comparison; the
        # expected two-eye fact above comes from the independent graph proof.
        analysis = pass_alive_analysis(state.board, topology)
        life = independent_life_analysis(state.board, topology)
        if fixture.expected["eye_kind"] in ("obvious_true_eye_pair", "vertex_related"):
            vital = prove_two_vital_regions(board, BLACK, topology.neighbors_by_index)
            expected_indices = {topology.point_index(point) for point in expected_points}
            assert expected_indices == set().union(*vital.vital_regions)
            assert expected_points.issubset({topology.point_id(point) for point in analysis.pass_alive_black_territory})
            assert expected_points.issubset({topology.point_id(point) for point in life.black_territory})
        else:
            expected_indices = {topology.point_index(point) for point in expected_points}
            independent_dame = set().union(*mixed_border_regions(board, topology.neighbors_by_index))
            assert expected_indices.issubset(independent_dame)
            assert not analysis.pass_alive_black_groups
            assert expected_points.issubset({topology.point_id(point) for point in life.dame})


def test_v1_settled_seki_has_independent_bounded_continuation_proof():
    topology = cube_topology(4)
    seki = _fixture("cube4_seki_shared_liberty_001")
    board = seki.board(topology.index_by_id)
    proof = prove_settled_seki(board, topology.neighbors_by_index)
    shared = {topology.point_index(point) for point in seki.expected["shared_liberty_points"]}
    assert proof.status == "proved_settled_seki"
    assert proof.search_status == "proved_draw"
    assert proof.max_depth == 3
    assert proof.max_depth_reached == 3
    assert proof.nodes == seki.expected["explored_nodes"] == 6
    assert set(proof.shared_liberties) == shared
    assert tuple(action for _player, actions in proof.legal_first_actions for action in actions) == tuple(sorted(shared)) * 2
    assert len(proof.reply_lines) == 4
    for line in proof.reply_lines:
        assert line.defensive_action != line.first_action
        assert line.captured_first_group

    state = _state_for_fixture(seki, topology)
    # These are actual production observations, never the source of the
    # expected seki classification.
    production_pass_alive = pass_alive_analysis(state.board, topology)
    production_life = independent_life_analysis(state.board, topology)
    assert not production_pass_alive.pass_alive_black_groups
    assert not production_pass_alive.pass_alive_white_groups
    assert not shared.intersection(set(production_life.black_territory))
    assert not shared.intersection(set(production_life.white_territory))
    assert shared.issubset(set(production_life.dame))
    scoring_state = replace(
        state,
        phase=CLEANUP_2,
        second_cleanup_start_colors=bytes(board),
    )
    production_score, ownership, ownership_mask = final_v3_score(scoring_state, topology, 0.5)
    assert production_score.territory.neutral == len(shared)
    assert production_score.territory.seki == 0
    for point in shared:
        assert np.array_equal(ownership[point], np.asarray((0.0, 0.0, 1.0)))
        assert ownership_mask[point] == 1.0


def test_v1_dame_is_independently_classified_before_production_comparison():
    topology = cube_topology(4)
    fixture = _fixture("cube4_dame_neutral_region_001")
    board = fixture.board(topology.index_by_id)
    independent_dame = set().union(*mixed_border_regions(board, topology.neighbors_by_index))
    anchors = {topology.point_index(point) for point in fixture.expected["anchor_points"]}
    assert anchors.issubset(independent_dame)

    state = _state_for_fixture(fixture, topology)
    # Production life/scoring is compared to the graph-derived neutral set.
    production_life = independent_life_analysis(state.board, topology)
    assert anchors.issubset(set(production_life.dame))
    assert not anchors.intersection(set(production_life.black_territory))
    assert not anchors.intersection(set(production_life.white_territory))


def test_v1_intruder_has_independent_pass_alive_evidence_and_matching_production_ownership():
    topology = cube_topology(4)
    fixture = _fixture("cube4_pass_alive_intruder_001")
    board = fixture.board(topology.index_by_id)
    exhaustion = prove_opponent_placement_exhaustion(board, BLACK, topology.neighbors_by_index)
    assert exhaustion.proved
    assert exhaustion.opponent_legal_actions == ()
    regions = empty_regions(board, topology.neighbors_by_index)
    assert any(region.bordering_colors == frozenset((BLACK,)) for region in regions)
    intruder = topology.point_index(fixture.expected["intruder"]["point"])
    analysis = pass_alive_analysis(board, topology)
    assert len(analysis.pass_alive_black_groups) == 1
    assert intruder in analysis.pass_alive_black_territory
    assert not analysis.pass_alive_white_groups
    state = replace(_state_for_fixture(fixture, topology), phase=CLEANUP_2, second_cleanup_start_colors=bytes(board))
    score, ownership, ownership_mask = final_v3_score(state, topology, 0.5)
    assert score.territory.black == 2
    assert np.array_equal(ownership[intruder], np.asarray((1.0, 0.0, 0.0)))
    assert ownership_mask[intruder] == 1.0


def test_v1_nonempty_score_fixture_matches_independent_graph_and_japanese_oracles():
    topology = cube_topology(4)
    fixture = _fixture("cube4_nonempty_two_eye_score_001")
    board = fixture.board(topology.index_by_id)
    proof = prove_two_vital_regions(board, BLACK, topology.neighbors_by_index)

    assert len(proof.vital_regions) == 2
    assert {topology.point_id(point) for region in proof.vital_regions for point in region} == {
        "front:0:0",
        "back:2:2",
    }
    assert len(proof.group) == len(fixture.black)

    state = replace(initial_state(topology), board=np.asarray(board, dtype=np.uint8))
    score = score_position(
        state,
        topology,
        [GroupClassification(tuple(sorted(proof.group)), "alive")],
        "japanese",
        0.5,
    )
    actual = {
        "rule_set": score.ruleset,
        "black": score.black,
        "white": score.white,
        "komi": score.komi,
        "territory": {
            "black": score.territory.black,
            "white": score.territory.white,
            "neutral": score.territory.neutral,
            "seki": score.territory.seki,
        },
        "stones_on_board": {
            "black": score.stones_on_board.black,
            "white": score.stones_on_board.white,
        },
        "captures": list(score.captures),
        "prisoners": None if score.prisoners is None else list(score.prisoners),
        "dead_stones": {
            "black": score.dead_stones.black,
            "white": score.dead_stones.white,
        },
        "winner": score.winner,
        "margin": score.margin,
    }
    assert actual == fixture.expected["verified_final_score"]


def test_v1_settled_seki_has_pinned_katago_rectangular_scoring_analog():
    rules_path = Path(__file__).parent / "reference" / "katago" / "rules_fixtures.json"
    analog = next(
        fixture
        for fixture in json.loads(rules_path.read_text(encoding="utf-8"))
        if fixture["id"] == "seki-tax"
    )
    snapshots = run_fixture(analog)
    final = snapshots[-1]
    assert final["phase"] == "SCORED"
    assert final["is_game_finished"] is True
    assert final["is_no_result"] is False
    assert final["final_score"] == 0.5
    assert final["winner"] == "white"


def test_v1_runtime_limit_is_not_a_formal_no_result():
    topology = cube_topology(4)
    from alphazero.envs.gocube.katago_v3 import episode_move_limit

    state = replace(v3_state_from_board(topology), turns=episode_move_limit(topology) - 1)
    after = apply_v3_action(state, topology.pass_action, topology)
    assert after.terminal_kind is None
    assert after.termination_reason is None
    assert after.phase == MAIN


def test_v1_rotations_preserve_independent_outcomes_and_keep_source_provenance():
    topology = cube_topology(4)
    for fixture in cube_verification_fixtures():
        if not fixture.rotation_safe:
            continue
        for rotation in cube_rotations(topology):
            rotated = rotate_fixture(fixture, topology, rotation)
            assert rotated.source_id == fixture.id
            assert rotated.rotation_index == rotation.index
            original_board = fixture.board(topology.index_by_id)
            rotated_board = rotated.board(topology.index_by_id)
            assert sorted(np.flatnonzero(np.asarray(rotated_board) == BLACK)) == sorted(rotation.apply_point(point) for point in np.flatnonzero(np.asarray(original_board) == BLACK))
            if fixture.family == "ko":
                proof = prove_positional_restoration(
                    rotated_board,
                    BLACK,
                    topology.point_index(rotated.actions[0]),
                    topology.point_index(rotated.actions[1]),
                    topology.neighbors_by_index,
                )
                assert proof.is_simple_ko is fixture.expected["simple_ko"]
            elif fixture.actions and fixture.phase == MAIN:
                try:
                    graph = apply_move(
                        rotated_board,
                        BLACK if rotated.to_move == "black" else WHITE,
                        topology.point_index(rotated.actions[0]),
                        topology.neighbors_by_index,
                    )
                except IndependentIllegalMove:
                    assert not v3_valid_moves(_state_for_fixture(rotated, topology), topology)[topology.point_index(rotated.actions[0])]
                else:
                    production = apply_v3_action(_state_for_fixture(rotated, topology), topology.point_index(rotated.actions[0]), topology)
                    assert tuple(int(value) for value in production.board) == graph.board


def test_v1_generated_sequences_use_fixed_seeds_and_compare_every_step():
    assert V1_GENERATOR_VERSION == "gocube-v1-legal-intersection-v1"
    report = run_generated_rectangular_differential()
    assert tuple(item.seed for item in report) == V1_SEEDS
    assert tuple(item.requested_steps for item in report) == V1_LENGTHS
    assert all(item.steps_compared > 0 for item in report)
    assert sum(item.steps_compared for item in report) > len(report)
    assert all(item.stopped in ("bound", "terminal") for item in report)


def test_v1_canonical_matrix_has_all_required_families_and_accepted_statuses():
    rows = canonical_matrix(cube_verification_fixtures(), torus_verification_fixtures())
    assert tuple(row.family for row in rows) == REQUIRED_FAMILIES
    assert_matrix_accepted(rows)
    assert all(row.production_compared for row in rows)


def test_v1_difference_registry_is_versioned_and_contains_no_unresolved_rule_gap():
    path = Path(__file__).parent / "reference" / "gocube_v1_difference_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = {"difference_id", "upstream_behavior", "local_behavior", "reason", "verification", "status"}
    assert registry
    assert all(required <= set(item) for item in registry)
    assert len({item["difference_id"] for item in registry}) == len(registry)
    assert all(item["status"] == "EXPLAINED_DIFFERENCE" for item in registry)
    assert not any(item["difference_id"] in {"board", "legal", "capture", "ko", "score"} for item in registry)
