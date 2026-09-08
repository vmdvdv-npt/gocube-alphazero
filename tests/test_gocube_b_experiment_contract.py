from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from alphazero.envs.gocube.b_experiment_contract import (
    ExperimentContractError,
    build_b_experiment_contract,
    diff_effective_configs,
    hash_heldout_suite,
    preflight_b_experiment,
    resolve_b_effective_configs_separate_processes,
    validate_b0_effective_config,
    validate_b0_b1_effective_configs,
    validate_b1_effective_config,
    validate_b_experiment_contract,
    validate_b_experiment_record,
    write_b_experiment_record,
)
from alphazero.envs.gocube.contract_versions import (
    B_EXPERIMENT_CONTRACT_ID,
    B_EXPERIMENT_CONTRACT_VERSION,
)
from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args
from tools import gocube_b_experiment


@pytest.fixture(scope="module")
def canonical():
    contract = build_b_experiment_contract(heldout_suite_hash="a" * 64)
    b0, b1 = resolve_b_effective_configs_separate_processes()
    return contract, b0, b1


def test_canonical_b0_and_b1_configs_validate(canonical):
    contract, b0, b1 = canonical
    validate_b_experiment_contract(contract, b0_effective_config=b0, b1_effective_config=b1)
    assert contract.contract_id == "gocube-b-experiment-contract-v1"
    assert contract.komi == 0.5
    assert contract.training_batch_size == 1024
    assert b0["train_batch_size"] == b1["train_batch_size"] == 1024
    assert contract.result_semantics["win"] == 1.0
    assert contract.result_semantics["draw"] == 0.5
    assert contract.result_semantics["no_result"] == 0.5
    assert contract.result_semantics["loss"] == 0.0
    assert contract.primary_endpoint == "heldout_paired_position_score"
    assert "hierarchical-paired-bootstrap" in contract.statistical_method_identifier
    assert "excluded from the denominator" not in contract.no_result_evaluation_convention
    assert contract.seed_list == (0, 1, 2, 3, 4)
    assert contract.initial_seed_count == 3
    assert contract.extension_seed_count == 5
    assert contract.mandatory_seed_list == (0, 1, 2)
    assert contract.extension_seed_list == (3, 4)
    milestones = contract.evaluation_milestones
    assert milestones["clock"] == "cumulative_new_samples"
    assert milestones["counter"] == "new_samples_accepted"
    assert milestones["target"] == 40_000_000
    assert milestones["milestone_fractions"] == (0.25, 0.5, 0.75, 1.0)
    assert milestones["milestone_targets"] == (10_000_000, 20_000_000, 30_000_000, 40_000_000)
    assert milestones["comparison_rule"] == "paired-at-equal-cumulative-sample-budget-v1"
    assert "bootstrap_iteration" not in milestones
    assert milestones["operational_metadata"]["bootstrap_iteration"] == 7


def test_contract_builder_rejects_missing_heldout_hash():
    with pytest.raises(ExperimentContractError, match="real frozen heldout-suite"):
        build_b_experiment_contract()


def test_heldout_hash_is_computed_from_artifact(tmp_path):
    artifact = tmp_path / "frozen-suite.json"
    artifact.write_bytes(b'{"schema_version":1,"positions":16}\n')
    assert hash_heldout_suite(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_real_preflight_requires_heldout_artifact():
    with pytest.raises(ExperimentContractError, match="--heldout-suite is required"):
        preflight_b_experiment()


@pytest.mark.parametrize(
    "field,value",
    [
        ("komi", 7.5),
        ("training_batch_size", 256),
        ("arena_simulations", 51),
        ("self_play_simulations", 49),
        ("fast_simulations", 19),
        ("train_samples_per_new_sample", 2.0),
        ("rules_id", "different-rules"),
    ],
)
def test_contract_drift_fails_closed(canonical, field, value):
    contract, _b0, _b1 = canonical
    broken = replace(contract, **{field: value})
    with pytest.raises(ExperimentContractError):
        validate_b_experiment_contract(broken)


def test_iteration_based_evaluation_milestones_are_rejected(canonical):
    contract, _b0, _b1 = canonical
    broken = replace(
        contract,
        evaluation_milestones={
            "bootstrap_iteration": 7,
            "health_reference_iteration": 4,
            "arena_anchor_period": 10,
            "heldout_positions": 16,
        },
    )
    with pytest.raises(ExperimentContractError, match="evaluation_milestones"):
        validate_b_experiment_contract(broken)


def test_effective_config_whitelist_allows_only_treatment_structure(canonical):
    contract, b0, b1 = canonical
    allowed = copy.deepcopy(b1)
    allowed["lr"] = float(b0["lr"]) + 0.001
    # The learning-rate assignment is deliberately not a permitted change.
    with pytest.raises(ExperimentContractError, match="lr"):
        validate_b0_b1_effective_configs(b0, allowed)

    structural = copy.deepcopy(b1)
    structural["gocube_network_architecture"] = "test-architecture"
    structural["network_architecture_id"] = "test-architecture"
    assert diff_effective_configs(b0, structural) == []
    with pytest.raises(ExperimentContractError, match="network_architecture"):
        validate_b0_b1_effective_configs(b0, structural, contract=contract)


def test_single_treatment_helpers_validate_canonical_b0_and_b1(canonical):
    contract, b0, b1 = canonical
    validate_b0_effective_config(b0, contract=contract)
    validate_b1_effective_config(b1, contract=contract)


@pytest.mark.parametrize(
    "key,value",
    [("lr", 0.02), ("workers", 8), ("gocube_lr_decay_gamma", 0.2)],
)
def test_same_common_drift_in_b0_and_b1_fails_against_canonical_contract(canonical, key, value):
    contract, b0, b1 = canonical
    broken_b0 = copy.deepcopy(b0)
    broken_b1 = copy.deepcopy(b1)
    broken_b0[key] = value
    broken_b1[key] = value
    assert diff_effective_configs(broken_b0, broken_b1) == []
    with pytest.raises(ExperimentContractError, match=key):
        validate_b0_b1_effective_configs(
            broken_b0,
            broken_b1,
            contract=contract,
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("workers", 8),
        ("gocube_rules_fingerprint", "wrong-rules"),
        ("unexpected_setting", True),
        ("gocube_train_samples_per_new_sample", 2.0),
        ("train_batch_size", 256),
    ],
)
def test_non_whitelisted_effective_drift_fails_closed(canonical, key, value):
    _contract, b0, b1 = canonical
    broken = copy.deepcopy(b1)
    broken[key] = value
    with pytest.raises(ExperimentContractError):
        validate_b0_b1_effective_configs(b0, broken)


def test_separate_process_resolution_is_stable(canonical):
    _contract, b0, b1 = canonical
    next_b0, next_b1 = resolve_b_effective_configs_separate_processes()
    assert next_b0 == b0
    assert next_b1 == b1
    assert diff_effective_configs(next_b0, next_b1) == []


def test_launcher_requires_treatment_and_resolves_profile_itself():
    args = gocube_b_experiment.parse_args(["--treatment", "B1", "--dry-run"])
    command = gocube_b_experiment.training_command(args, python="python")
    assert command[command.index("--model-profile") + 1] == "g1"
    assert command[command.index("--train-batch-size") + 1] == "1024"

    args = gocube_b_experiment.parse_args(["--treatment", "B0", "--dry-run"])
    command = gocube_b_experiment.training_command(args, python="python")
    assert command[command.index("--model-profile") + 1] == "baseline"
    command = gocube_b_experiment.training_command(
        args,
        python="python",
        contract_sha256="a" * 64,
    )
    assert command[command.index("--experiment-contract-sha256") + 1] == "a" * 64
    assert command[command.index("--experiment-contract-id") + 1] == B_EXPERIMENT_CONTRACT_ID


def _canonical_cube4_args(*extra):
    return parse_args(
        [
            "--topology",
            "cube",
            "--size",
            "4",
            "--workers",
            "16",
            "--sims",
            "50",
            "--arena-sims",
            "50",
            "--games-per-iteration",
            "256",
            "--train-batch-size",
            "1024",
            "--fast-game-prob",
            "0.25",
            "--train-samples-per-new-sample",
            "1",
            "--no-arena",
            *extra,
        ]
    )


def test_ordinary_cube4_production_batch_does_not_activate_b_contract():
    _game_cls, args = build_hardened_training_args(_canonical_cube4_args())
    assert args.train_batch_size == 1024
    assert args.gocube_experiment_contract_id is None
    assert args.gocube_experiment_contract_version is None
    assert args.gocube_experiment_contract_sha256 is None


def test_explicit_b_marker_sets_contract_identity_and_version():
    _game_cls, args = build_hardened_training_args(
        _canonical_cube4_args(
            "--experiment-contract-id",
            B_EXPERIMENT_CONTRACT_ID,
            "--experiment-contract-sha256",
            "a" * 64,
        )
    )
    assert args.gocube_experiment_contract_id == B_EXPERIMENT_CONTRACT_ID
    assert args.gocube_experiment_contract_version == B_EXPERIMENT_CONTRACT_VERSION
    assert args.gocube_experiment_contract_sha256 == "a" * 64


def test_b_marker_without_sha_fails_closed():
    with pytest.raises(ValueError, match="both --experiment-contract-id"):
        build_hardened_training_args(
            _canonical_cube4_args("--experiment-contract-id", B_EXPERIMENT_CONTRACT_ID)
        )
    with pytest.raises(ValueError, match="both --experiment-contract-id"):
        build_hardened_training_args(
            _canonical_cube4_args("--experiment-contract-sha256", "a" * 64)
        )


def test_launcher_accepts_mandatory_and_approved_extension_seeds(tmp_path):
    args = gocube_b_experiment.parse_args(["--treatment", "B0", "--seed", "2", "--dry-run"])
    command = gocube_b_experiment.training_command(args, python="python")
    assert command[command.index("--seed") + 1] == "2"

    decision = tmp_path / "extension-decision.json"
    decision.write_text(
        json.dumps(
            {
                "approved": True,
                "decision": "extend_to_five",
                "criterion_id": (
                    "extend-to-five-seeds-only-if-mandatory-seed-bootstrap-ambiguity-or-variance-v1"
                ),
                "mandatory_seed_count": 3,
                "extension_seed_count": 5,
                "scientific_clock": "cumulative_new_samples",
                "scientific_milestone": 40_000_000,
                "criterion_evidence": {
                    "ambiguity_detected": True,
                    "variance_exceeded": False,
                },
            }
        ),
        encoding="utf-8",
    )
    args = gocube_b_experiment.parse_args(
        [
            "--treatment",
            "B1",
            "--seed",
            "3",
            "--extension-seed-decision",
            str(decision),
            "--dry-run",
        ]
    )
    command = gocube_b_experiment.training_command(args, python="python")
    assert command[command.index("--seed") + 1] == "3"
    with pytest.raises(SystemExit):
        gocube_b_experiment.parse_args(["--treatment", "B1", "--seed", "3", "--dry-run"])
    with pytest.raises(SystemExit):
        gocube_b_experiment.parse_args(
            ["--treatment", "B1", "--seed", "0", "--allow-dirty-source", "--dry-run"]
        )


def test_launcher_without_suite_is_dry_run_only(tmp_path):
    contract_path = tmp_path / "must-not-be-written.json"
    assert gocube_b_experiment.main(
        [
            "--treatment",
            "B1",
            "--dry-run",
            "--contract-path",
            str(contract_path),
        ]
    ) == 0
    assert not contract_path.exists()
    with pytest.raises(SystemExit, match="--heldout-suite is required"):
        gocube_b_experiment.main(["--treatment", "B1", "--iterations", "1"])


def test_machine_readable_record_round_trips(canonical, tmp_path):
    contract, b0, b1 = canonical
    path = tmp_path / "gocube-b-experiment-contract.json"
    write_b_experiment_record(path, contract, b0, b1)
    payload = validate_b_experiment_record(path)
    assert payload["experiment_contract"]["training_batch_size"] == 1024
