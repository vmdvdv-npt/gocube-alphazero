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


def _trace_fixture(policy_hash="p", root_visits=(1, 0), selected=0):
    request = {
        "session_request_index": 0, "state_sha256": "state",
        "legal_mask_sha256": "legal", "observation_sha256": "observation",
        "policy": [0.75, 0.25], "wdl": [0.2, 0.3, 0.5],
        "policy_sha256": policy_hash, "wdl_sha256": "wdl",
    }
    return {
        "root_identity": {"state_sha256": "root"}, "search_seed": 7,
        "root_priors": [0.75, 0.25], "root_visits": list(root_visits),
        "selected_action": selected, "evaluation_requests": [request],
        "simulations": [{
            "simulation_index": 0, "traversal_path": [{"action": 0}],
            "leaf": {"state_sha256": "leaf"}, "evaluation_request_index": 0,
            "puct_decisions": [{"selected_action": 0, "candidate_scores": [1.0, 0.5]}],
            "root_visits_after": list(root_visits),
        }],
    }


def test_first_divergent_simulation_reports_neural_delta_before_later_puct_choice():
    left = _trace_fixture()
    right = _trace_fixture(policy_hash="p2")

    difference = gate._DIAGNOSTICS.first_divergent_simulation(left, right)

    assert difference["simulation_index"] == 0
    assert difference["category"] == "root_neural_evaluation"
    assert difference["left_policy"] == [0.75, 0.25]
    assert difference["right_policy"] == [0.75, 0.25]


def test_controlled_geometry_causality_requires_fixed_root_one_visit_and_stable_repeats():
    def trace(leaf_policy_hash, leaf_wdl_hash, visits, changed=False):
        root = {
            "session_request_index": 0, "state_sha256": "root-state",
            "legal_mask_sha256": "root-legal", "observation_sha256": "root-observation",
            "policy_sha256": "root-policy", "wdl_sha256": "root-wdl",
            "policy": [0.75, 0.25], "wdl": [0.2, 0.3, 0.5],
        }
        leaf = {
            "session_request_index": 1, "state_sha256": "leaf-state",
            "legal_mask_sha256": "leaf-legal", "observation_sha256": "leaf-observation",
            "policy_sha256": leaf_policy_hash, "wdl_sha256": leaf_wdl_hash,
            "policy": [0.6, 0.4], "wdl": [0.1, 0.2, 0.7],
        }
        return {
            "root_identity": {"state_sha256": "root-state"},
            "search_seed": 12, "simulation_cap": 2,
            "settings": {"cpuct": 1.25, "fpu": 0.0, "deterministic_tie_break": True},
            "noise": {"enabled": True, "seed": 13, "alpha": 0.11, "epsilon": 0.25},
            "root_priors": [0.75, 0.25], "root_visits": list(visits),
            "selected_action": 0,
            "evaluation_requests": [root, leaf],
            "simulations": [
                {"puct_decisions": [{"selected_action": 0}],
                 "traversal_path": [{"action": 0}], "leaf": {"state_sha256": "leaf-state"},
                 "evaluation_request_index": 1, "leaf_utility": 0.2 if changed else 0.1,
                 "backup_utility": 0.2 if changed else 0.1,
                 "root_visits_after": [1, 0]},
                {"puct_decisions": [{"selected_action": 1 if changed else 0}],
                 "traversal_path": [{"action": 1 if changed else 0}],
                 "leaf": {"state_sha256": "next-leaf"},
                 "evaluation_request_index": -1,
                 "root_visits_after": list(visits)},
            ],
        }

    left = trace("leaf-policy-a", "leaf-wdl-a", [1, 1])
    right = trace("leaf-policy-b", "leaf-wdl-b", [2, 0], changed=True)
    report = gate._DIAGNOSTICS.compare_controlled_geometry_traces(
        left, right, left_repeats=[trace("leaf-policy-a", "leaf-wdl-a", [1, 1])],
        right_repeats=[trace("leaf-policy-b", "leaf-wdl-b", [2, 0], changed=True)])

    assert report["status"] == "PROVEN_FOR_THIS_ROOT"
    assert report["first_evaluation_difference"]["request_index"] == 1
    assert report["first_puct_difference"]["simulation_index"] == 1
    assert report["first_selected_path_difference"]["simulation_index"] == 1
    assert report["first_backup_difference"]["simulation_index"] == 0
    assert report["visits_moved_left_to_right"] == 1
    assert report["left_geometry_repeat_stable"]
    assert report["right_geometry_repeat_stable"]


def test_evaluation_stream_serialization_preserves_values_and_request_identity():
    trace = _trace_fixture()
    trace.update({"simulation_cap": 200, "settings": {"cpuct": 1.25, "fpu": 0.0,
                                                       "deterministic_tie_break": True},
                  "noise": {"enabled": True, "alpha": 0.11, "epsilon": 0.25, "seed": 9}})

    stream = gate._DIAGNOSTICS.serialize_evaluation_stream(trace)

    assert stream["schema"] == "torus9-fixed-evaluation-stream-v1"
    assert stream["evaluations"][0]["request"]["observation_sha256"] == "observation"
    assert stream["evaluations"][0]["policy"] == [0.75, 0.25]
    assert stream["evaluations"][0]["wdl"] == [0.2, 0.3, 0.5]


def test_exact_root_trace_comparison_includes_search_result():
    left = _trace_fixture()
    right = _trace_fixture()
    assert gate._DIAGNOSTICS.compare_root_traces(left, right)["identical"]

    right["root_visits"] = [0, 1]
    assert not gate._DIAGNOSTICS.compare_root_traces(left, right)["identical"]


def test_batch_geometry_aggregation_compares_matching_repeat_baseline():
    base = {"scope": "same-process", "repeat": 0, "batch_size": 1,
            "target_row": 0, "policy": [0.5, 0.5], "wdl": [0.2, 0.3, 0.5],
            "policy_logits": [0.0, 0.0], "wdl_logits": [0.0, 0.0, 0.0],
            "policy_sha256": "base", "wdl_sha256": "wdl"}
    batched = {**base, "batch_size": 2, "target_row": 1,
               "policy": [0.5001, 0.4999], "policy_logits": [0.001, -0.001],
               "policy_sha256": "shape", "wdl_sha256": "wdl2"}

    summary = gate._DIAGNOSTICS.aggregate_batch_geometry([base, batched])

    variant = next(item for item in summary["variants"] if item["batch_size"] == 2)
    assert summary["shape_or_row_dependent"]
    assert variant["max_policy_abs_delta_vs_batch1"] == pytest.approx(0.0001)
    assert variant["repeat_deltas"][0]["max_policy_logits_abs_delta"] == pytest.approx(0.001)


def test_same_revision_baseline_and_cross_revision_range_classification():
    def comparison(value):
        return {"comparison": str(value), "root_visit_mismatch_fraction": value / 100,
                "selected_action_mismatch_count": 0, "action_trace_mismatch_games": 0,
                "result_mismatch_games": 0, "max_mismatched_root_l1": value / 100,
                "one_visit_redistribution_count": int(value)}

    baseline = gate._DIAGNOSTICS.aggregate_same_revision_baseline(
        [comparison(1), comparison(3)])
    classification = gate._DIAGNOSTICS.classify_against_baseline(comparison(2), baseline)

    assert baseline["pair_count"] == 2
    assert classification["inside_all_observed_AA_ranges"]
    assert not gate._DIAGNOSTICS.classify_against_baseline(
        comparison(4), baseline)["inside_all_observed_AA_ranges"]


def test_final_verdict_refuses_supported_with_unknown_root_cause_or_mutated_artifacts():
    common = dict(deterministic_core_pass=True, cross_revision_parity_pass=True,
                  production_bitwise_reproducible=False, target_builder_pass=True,
                  optimizer_pass=True, pcr_pass=True, inference_mapping_pass=True,
                  artifact_safety_pass=True, production_baseline_consistent=True)
    unknown = gate._DIAGNOSTICS.final_verdicts(root_cause="UNKNOWN", **common)
    mutated = gate._DIAGNOSTICS.final_verdicts(
        root_cause="PROVEN", **{**common, "artifact_safety_pass": False})

    assert unknown["continuation_safety"] == "NOT SUPPORTED"
    assert unknown["production_concurrent_bitwise_reproducibility"] == "FAIL"
    assert mutated["continuation_safety"] == "NOT SUPPORTED"


def test_audit_paths_remain_under_tmp_and_pure_diagnostics_do_not_write():
    production_path = Path("/home/codex/projects/gocube-alphazero/runs/torus9/active")
    before = tuple(production_path.iterdir())
    with pytest.raises(ValueError, match="under /tmp"):
        gate.ensure_tmp_path(production_path / "diagnostic.json")
    gate._DIAGNOSTICS.compare_root_traces(_trace_fixture(), _trace_fixture())
    assert tuple(production_path.iterdir()) == before
