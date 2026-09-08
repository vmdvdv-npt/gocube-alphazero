from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from alphazero.envs.gocube.core import cube_topology

from tests.support.fixtures import (
    assert_rotation_split_consistency,
    cube_verification_fixtures,
    fixture_counts,
    write_fixture_json,
)
from tests.support.product_boundary import (
    KOMI,
    PRODUCT_BOUNDARY_SCHEMA,
    export_product_boundary_fixture,
    export_verified_product_boundary_fixtures,
    write_product_boundary_json,
)


def test_rotation_safe_corpus_keeps_source_fixture_grouped_in_splits():
    fixtures = cube_verification_fixtures()
    rotated = []
    for fixture in fixtures[:2]:
        rotated.append(fixture.__class__(**{**fixture.__dict__, "id": f"{fixture.id}__r00", "source_fixture_id": fixture.id, "rotation_index": 0}))
    split = {fixture.id: "holdout" for fixture in fixtures[:2]}
    split.update({fixture.id: "holdout" for fixture in rotated})
    assert_rotation_split_consistency(tuple(fixtures[:2]) + tuple(rotated), split)


def test_fixture_json_round_trip_is_machine_readable(tmp_path):
    fixtures = cube_verification_fixtures()
    path = tmp_path / "cube-verification.json"
    write_fixture_json(path, fixtures)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == len(fixtures)
    assert payload[0]["id"] == fixtures[0].id
    assert payload[0]["oracle/source"] == fixtures[0].oracle
    assert all(item["rotation_safe"] is True for item in payload if item["family"] not in ("cleanup", "early_termination"))


def test_product_boundary_export_preserves_main_and_cleanup_sections(tmp_path):
    topology = cube_topology(4)
    fixtures = cube_verification_fixtures()
    cleanup = next(item for item in fixtures if item.id == "cube4_cleanup1_pass_for_ko_001")
    boundary = export_product_boundary_fixture(cleanup, topology)
    assert boundary.schema == PRODUCT_BOUNDARY_SCHEMA
    assert boundary.komi == KOMI == 0.5
    assert boundary.main_action_sequence == cleanup.actions
    assert boundary.expected_captures_during_main[0]["points"] == ("front:0:0",)
    assert boundary.board_after_first_pass == boundary.board_after_second_pass
    assert boundary.training_internal_cleanup == cleanup.cleanup_metadata["training_internal_cleanup"]

    early = next(item for item in fixtures if item.id == "cube4_early_termination_boundary_001")
    early_boundary = export_product_boundary_fixture(early, topology)
    assert early_boundary.board_after_second_pass == {"black": (), "white": ()}
    assert early_boundary.training_internal_cleanup["final_training_result"] is None
    output = tmp_path / "product-boundary.json"
    write_product_boundary_json(output, [boundary, early_boundary])
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == PRODUCT_BOUNDARY_SCHEMA
    assert [item["fixture_id"] for item in document["fixtures"]] == [cleanup.id, early.id]


def test_product_boundary_requires_consecutive_passes_not_any_two_passes():
    topology = cube_topology(4)
    early = next(item for item in cube_verification_fixtures() if item.id == "cube4_early_termination_boundary_001")
    interrupted = replace(
        early,
        id="cube4_pass_move_pass_not_boundary_001",
        actions=("PASS", "front:0:0", "PASS"),
    )
    with pytest.raises(ValueError, match="two MAIN PASS"):
        export_product_boundary_fixture(interrupted, topology)

    resumed = replace(
        interrupted,
        id="cube4_pass_move_pass_pass_boundary_001",
        actions=("PASS", "front:0:0", "PASS", "PASS"),
    )
    boundary = export_product_boundary_fixture(resumed, topology)
    assert "front:0:0" in boundary.board_after_first_pass["white"]


def test_v2_export_is_coupled_to_verified_v1_status():
    topology = cube_topology(4)
    source = next(item for item in cube_verification_fixtures() if item.family == "captures")
    unresolved = replace(source, status="unresolved")
    with pytest.raises(ValueError, match="verified V1"):
        export_product_boundary_fixture(
            unresolved,
            topology,
            require_verified=True,
        )


def test_v2_export_contains_explicit_mapping_steps_and_deterministic_boundary():
    topology = cube_topology(4)
    exported = export_verified_product_boundary_fixtures(cube_verification_fixtures(), topology)
    assert len(exported) >= 10
    for fixture in exported:
        document = fixture.to_dict()
        assert document["source_verification_status"] == "verified"
        assert len(document["topology_contract"]["point_mapping"]) == 96
        assert len(document["topology_contract"]["adjacency"]) == 96
        assert len(document["steps"]) == len(document["main_actions"])
        assert document["after_second_pass"]["consecutive_passes"] == 2
        assert document["expected_product_boundary"]["phase"] == "endgame"

    first = exported[0].to_dict()
    second = exported[0].to_dict()
    assert first == second


def test_corpus_counts_are_reportable_without_a_statistical_rotation_multiplier():
    counts = fixture_counts(cube_verification_fixtures())
    assert sum(counts.values()) == len(cube_verification_fixtures())
    assert counts["ko"] == 2


def test_checked_in_corpus_exports_match_typed_registry():
    root = Path(__file__).parent / "reference"
    corpus = json.loads((root / "cube_verification_fixtures.json").read_text(encoding="utf-8"))
    expected = cube_verification_fixtures()
    assert [item["id"] for item in corpus] == [fixture.id for fixture in expected]
    assert all(item["oracle/source"] for item in corpus)
    boundary = json.loads((root / "gocube_product_boundary_fixtures.json").read_text(encoding="utf-8"))
    assert boundary["schema"] == PRODUCT_BOUNDARY_SCHEMA
    assert len(boundary["fixtures"]) >= 10
    assert all(item["source_verification_status"] == "verified" for item in boundary["fixtures"])
