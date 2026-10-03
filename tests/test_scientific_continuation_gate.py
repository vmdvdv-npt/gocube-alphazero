from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parent / "benchmarks" / "scientific_continuation_gate.py"
SPEC = importlib.util.spec_from_file_location("scientific_continuation_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_canonical_json_hash_ignores_mapping_insertion_order():
    left = {"b": [2, 3], "a": {"z": 1, "x": 4}}
    right = {"a": {"x": 4, "z": 1}, "b": [2, 3]}
    assert gate.sha256_json(left) == gate.sha256_json(right)


def test_output_guard_rejects_repository_paths():
    with pytest.raises(ValueError, match="under /tmp"):
        gate.ensure_tmp_path(Path(__file__).resolve())


def test_first_mapping_difference_reports_first_nested_leaf():
    difference = gate.first_mapping_difference(
        {"ply": 4, "root": {"action": 17, "visits": [2, 1]}},
        {"ply": 4, "root": {"action": 18, "visits": [2, 1]}},
    )
    assert difference == {"path": "root.action", "left": 17, "right": 18}


def test_game_comparison_exposes_first_divergent_root(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    semantic_left = {"game_id": "g0", "positions": [{"ply": 1, "action": 3}]}
    semantic_right = {"game_id": "g0", "positions": [{"ply": 1, "action": 4}]}
    gate.write_jsonl(left, [{"game_id": "g0", "semantic": semantic_left,
                            "semantic_sha256": gate.sha256_json(semantic_left),
                            "raw_record": {"positions": [{"state": {"board": []}}]}}])
    gate.write_jsonl(right, [{"game_id": "g0", "semantic": semantic_right,
                             "semantic_sha256": gate.sha256_json(semantic_right),
                             "raw_record": {"positions": [{"state": {"board": []}}]}}])

    report = gate.compare_games(left, right)

    assert not report["identical"]
    assert report["first_divergence"]["ply_index"] == 1
    assert report["first_divergence"]["game_semantic_diff"] == {
        "path": "positions", "left": [{"ply": 1, "action": 3}],
        "right": [{"ply": 1, "action": 4}],
    }


def test_pcr_worker_order_compares_shared_positions_after_trajectory_split():
    left = {
        "g0": {"semantic": {"game_seed": 41, "positions": [
            {"ply": 1, "search_mode": "cheap", "simulation_cap": 100,
             "training_eligible": False},
            {"ply": 2, "search_mode": "full", "simulation_cap": 500,
             "training_eligible": True},
            {"ply": 3, "search_mode": "cheap", "simulation_cap": 100,
             "training_eligible": False},
        ]}},
    }
    right = {
        "g0": {"semantic": {"game_seed": 41, "positions": [
            {"ply": 1, "search_mode": "cheap", "simulation_cap": 100,
             "training_eligible": False},
            {"ply": 2, "search_mode": "full", "simulation_cap": 500,
             "training_eligible": True},
        ]}},
    }

    report = gate.compare_pcr_decisions_by_position(left, right)

    assert report["identical_on_shared_positions"]
    assert report["shared_positions"] == 2
    assert report["left_only_positions"] == 1
    assert report["right_only_positions"] == 0


def test_pcr_worker_order_reports_common_position_decision_mismatch():
    left = {"g0": {"semantic": {"game_seed": 41, "positions": [
        {"ply": 1, "search_mode": "cheap", "simulation_cap": 100,
         "training_eligible": False}]}}}
    right = {"g0": {"semantic": {"game_seed": 41, "positions": [
        {"ply": 1, "search_mode": "full", "simulation_cap": 500,
         "training_eligible": True}]}}}

    report = gate.compare_pcr_decisions_by_position(left, right)

    assert not report["identical_on_shared_positions"]
    assert report["mismatch_count"] == 1
    assert report["first_mismatch"]["game_id"] == "g0"


def test_repeat_comparison_localizes_root_visit_only_mismatch(tmp_path):
    left_path, right_path = tmp_path / "left.jsonl", tmp_path / "right.jsonl"

    def row(visits):
        semantic = {"game_id": "g0", "positions": [{
            "ply": 1, "root_visits": visits, "training_policy": [0.5, 0.5],
            "selected_action": 0,
            "legal_action_identity": {"actions": [0, 1], "mask_sha256": "mask"},
        }]}
        raw = {"game_seed": 7, "formal_result": "BLACK", "final_action_trace": [0],
               "positions": [{"state": {"stones": []}, "search_seed": 11,
                              "root_visits": visits, "pi": [0.5, 0.5],
                              "selected_action": 0}]}
        return {"game_id": "g0", "semantic": semantic,
                "semantic_sha256": gate.sha256_json(semantic), "raw_record": raw}

    gate.write_jsonl(left_path, [row([1, 1])])
    gate.write_jsonl(right_path, [row([2, 0])])

    report = gate.compare_repeat_root_targets(left_path, right_path)

    assert not report["identical"]
    assert report["identical_games"] == 0
    assert report["root_visit_mismatch_games"] == 1
    assert report["root_visit_mismatch_positions"] == 1
    assert report["action_trace_winner_and_length_equal"]
    assert report["first_divergence"]["changed_action_indices"] == [
        {"action_index": 0, "left": 1, "right": 2},
        {"action_index": 1, "left": 1, "right": 0},
    ]


def test_target_builder_comparison_uses_the_same_scientific_summary(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    summary = {"games": 64, "learner_samples": 6271,
               "field_sha256": {"pi": "abc", "z": "def"},
               "sample_identity_sha256": "identity"}
    gate.write_json(left, {**summary, "source_revision": "A"})
    gate.write_json(right, {**summary, "source_revision": "B"})

    report = gate._compare_target_summaries(left, right, "A vs B")

    assert report["identical"]
    assert report["left_revision"] == "A"
    assert report["right_revision"] == "B"

    gate.write_json(right, {**summary, "field_sha256": {"pi": "different", "z": "def"},
                            "source_revision": "B"})
    report = gate._compare_target_summaries(left, right, "A vs B")

    assert not report["identical"]
    assert report["first_difference"]["path"] == "field_sha256.pi"
