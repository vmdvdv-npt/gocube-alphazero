from __future__ import annotations

import ast
import copy
import math
from pathlib import Path

import pytest

from alphazero.envs.gocube import torus_golden_contract as contract
from alphazero.envs.gocube.komi_policy import LegacyKomiError, validate_gocube_komi


PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs/gocube/torus_golden_v1.json"
EXPECTED_TOPOLOGY_FP = "sha256:b4097c32d4ab5034b84300fa41f951b353a5fcf0d8226e83922889b5552289ef"
EXPECTED_RULES_FP = "sha256:8eac3337443a70893fa5ad359580f7ba92b18958e06f0d775c29f08791796842"
EXPECTED_OBSERVATION_FP = "sha256:d6e3aecc89f7df84f6e758423da4e3fe9269abeca644070db0be9261b30c6361"
EXPECTED_TARGET_FP = "sha256:6dffad74c4f832741f9cd522aa407f3e88d6f6159ebe56c3dc226a52b4145e41"
EXPECTED_EXPERIMENT_FP = "sha256:1f5a1f821fc2e3dc52c957da9ffe1fc9313fb5b02bd81fd2f452233b57bd576e"


def load() -> dict:
    return contract.load_profile(PROFILE_PATH)


def test_profile_loads_and_fingerprints_verify() -> None:
    profile = load()
    assert profile["profile_id"] == contract.PROFILE_ID
    assert profile["experiment_fingerprint"] == EXPECTED_EXPERIMENT_FP
    assert contract.computed_fingerprints(profile) == {
        "topology": EXPECTED_TOPOLOGY_FP,
        "rules": EXPECTED_RULES_FP,
        "observation": EXPECTED_OBSERVATION_FP,
        "targets": EXPECTED_TARGET_FP,
        "experiment": EXPECTED_EXPERIMENT_FP,
    }


def test_first_proof_pins_komi_point_five() -> None:
    profile = load()
    assert profile["rules"]["komi"] == 0.5
    assert profile["rules"]["komi_policy"]["first_proof_baseline"] == 0.5


def test_general_new_contract_validator_accepts_explicit_finite_alternative_komi() -> None:
    assert validate_gocube_komi(1.5, context="future owner-approved profile") == 1.5
    assert validate_gocube_komi(-2.0, context="future owner-approved profile") == -2.0


def test_legacy_7_5_fails_closed_and_escalates_to_owner() -> None:
    with pytest.raises(LegacyKomiError) as exc_info:
        validate_gocube_komi(7.5, context="torus golden runtime")
    message = str(exc_info.value).lower()
    assert "legacy" in message
    assert "stale" in message
    assert "owner" in message
    assert "do not silently coerce" in message


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "not-a-number", None, True])
def test_invalid_komi_is_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        validate_gocube_komi(value, context="torus golden validation")


def test_torus_5x5_topology_identity_is_stable_and_well_formed() -> None:
    profile = load()
    adjacency = contract.torus_neighbors(5, 5)
    assert len(adjacency) == 25
    for point_id, neighbors in enumerate(adjacency):
        assert len(neighbors) == 4
        assert len(set(neighbors)) == 4
        assert point_id not in neighbors
        assert all(point_id in adjacency[n] for n in neighbors)
    assert adjacency[0] == [20, 1, 5, 4]
    assert contract.compute_topology_fingerprint(profile) == EXPECTED_TOPOLOGY_FP


def test_changing_komi_changes_rules_and_experiment_fingerprints() -> None:
    profile = load()
    changed = copy.deepcopy(profile)
    changed["rules"]["komi"] = 1.5
    assert contract.compute_rules_fingerprint(changed) != contract.compute_rules_fingerprint(profile)
    assert contract.compute_experiment_fingerprint(changed) != contract.compute_experiment_fingerprint(profile)


def test_changing_topology_changes_topology_and_experiment_fingerprints() -> None:
    profile = load()
    changed = copy.deepcopy(profile)
    changed["topology"].update(width=6, number_of_points=30, topology_id="torus-6x5-row-major-test")
    assert contract.compute_topology_fingerprint(changed) != contract.compute_topology_fingerprint(profile)
    assert contract.compute_experiment_fingerprint(changed) != contract.compute_experiment_fingerprint(profile)


def test_changing_target_semantics_changes_target_and_experiment_fingerprints() -> None:
    profile = load()
    changed = copy.deepcopy(profile)
    changed["targets"]["value"]["perspective"] = "absolute-black-white-test"
    assert contract.compute_target_fingerprint(changed) != contract.compute_target_fingerprint(profile)
    assert contract.compute_experiment_fingerprint(changed) != contract.compute_experiment_fingerprint(profile)


def test_legacy_win_loss_no_result_is_not_compatible_with_wdl() -> None:
    profile = load()
    legacy = copy.deepcopy(profile)
    legacy["targets"]["contract_id"] = "legacy-win-loss-no-result-test"
    legacy["targets"]["value"]["vector"] = ["WIN", "LOSS", "NO_RESULT"]
    legacy["targets"]["value"]["legacy_wlnr_compatible"] = True
    assert len(legacy["targets"]["value"]["vector"]) == len(profile["targets"]["value"]["vector"])
    assert not contract.target_contract_compatible(profile, legacy)


def test_technical_termination_cannot_become_training_wdl_target() -> None:
    profile = load()
    assert contract.training_outcome_is_eligible(profile, "BLACK")
    assert contract.training_outcome_is_eligible(profile, "WHITE")
    assert contract.training_outcome_is_eligible(profile, "DRAW")
    assert not contract.training_outcome_is_eligible(profile, "TRUNCATED_MOVE_LIMIT")
    assert not contract.training_outcome_is_eligible(profile, "ERROR")
    assert not contract.training_outcome_is_eligible(profile, "NO_RESULT")


def test_profile_does_not_inherit_japanese_v3_defaults() -> None:
    profile = load()
    assert profile["legacy_inheritance"]["japanese_v3_defaults"] is False
    assert profile["legacy_inheritance"]["implicit_defaults"] is False
    assert profile["legacy_inheritance"]["legacy_torus_factory_9_13_19_size_restriction"] is False

    source = Path(contract.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert all("katago_v3" not in name for name in imports)


def test_new_profile_contains_no_implicit_7_5_runtime_default() -> None:
    profile = load()
    numeric_sentinel_paths: list[tuple[str, ...]] = []

    def walk(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + (str(key),))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(float(value), 7.5):
            numeric_sentinel_paths.append(path)

    walk(profile)
    assert numeric_sentinel_paths == [("rules", "komi_policy", "forbidden_legacy_sentinel")]
    assert profile["rules"]["komi"] == 0.5
    assert "7.5" not in Path(contract.__file__).read_text(encoding="utf-8")


def test_watchdog_is_execution_status_not_rule_result() -> None:
    profile = load()
    watchdog = profile["watchdog"]
    assert watchdog["is_game_rule"] is False
    assert watchdog["action_cap"] == 20 * profile["topology"]["number_of_points"] == 500
    assert watchdog["cap_status"] == "TRUNCATED_MOVE_LIMIT"
    assert watchdog["second_pass_terminal_precedes_truncation"] is True
    assert watchdog["self_play_truncation_warning_fraction"] == 0.05


def test_observation_explicitly_separates_nn_view_from_full_rules_state() -> None:
    observation = load()["observation"]
    assert [c["name"] for c in observation["point_channels"]] == [
        "own_stones",
        "opponent_stones",
        "side_to_move_color",
        "previous_pass",
        "legal_point_mask",
        "komi",
    ]
    assert observation["legal_action_mask"] == {
        "semantics": "full action mask over PointId 0..24 followed by PASS",
        "length": 26,
        "pass_action_index": 25,
    }
    assert observation["nn_observation_contains_full_superko_history"] is False
    assert observation["rules_search_state_contains_full_superko_history"] is True


def test_self_play_and_golden_arena_contracts_are_separate() -> None:
    search = load()["search_contracts"]
    assert search["self_play"]["contract_id"] != search["golden_arena"]["contract_id"]
    arena = search["golden_arena"]
    assert arena["root_noise"] is False
    assert arena["fast_search"] is False
    assert arena["move_temperature"] == 0
    assert arena["root_policy_temperature"] is False
    assert arena["resign"] is False
    assert arena["same_search_budget_for_both_models"] is True
    assert arena["search_settings_source"] == "arena-contract-not-checkpoint"


def test_checkpoint_semantic_identity_is_explicit_and_fail_closed() -> None:
    checkpoint = load()["checkpoint_identity"]
    required = set(checkpoint["required_metadata"])
    assert {
        "rules_profile_id",
        "rules_fingerprint",
        "topology_fingerprint",
        "board_size",
        "point_id_order_identity",
        "komi",
        "observation_schema_id",
        "observation_schema_version",
        "observation_fingerprint",
        "target_contract_id",
        "target_contract_version",
        "target_fingerprint",
        "value_head_semantics",
        "network_heads_and_shapes",
        "parent_or_source_run_identity",
        "model_hash",
    } <= required
    assert checkpoint["mismatch_policy"] == "fail-closed-no-fallback-default-or-coercion"


def test_policy_target_contract_is_plain_root_visit_normalization() -> None:
    policy = load()["targets"]["policy"]
    assert policy["formula"] == "N(a)/sum(N(legal_actions))"
    assert policy["illegal_action_mass"] == 0
    assert policy["legal_action_mass_sum"] == 1
    assert policy["requires_positive_visit_sum"] is True
    assert policy["fast_search"] is False
    assert policy["forced_playout_pruning"] is False
    assert policy["lcb_transform"] is False
