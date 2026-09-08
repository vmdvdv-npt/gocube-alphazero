from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from alphazero.envs.gocube.b_experiment_contract import (
    ExperimentContractError,
    build_b_experiment_contract,
    diff_effective_configs,
    resolve_b_effective_configs_separate_processes,
    validate_b0_b1_effective_configs,
    validate_b_experiment_contract,
    validate_b_experiment_record,
    write_b_experiment_record,
)
from tools import gocube_b_experiment


@pytest.fixture(scope="module")
def canonical():
    contract = build_b_experiment_contract()
    b0, b1 = resolve_b_effective_configs_separate_processes()
    return contract, b0, b1


def test_canonical_b0_and_b1_configs_validate(canonical):
    contract, b0, b1 = canonical
    validate_b_experiment_contract(contract, b0_effective_config=b0, b1_effective_config=b1)
    assert contract.contract_id == "gocube-b-experiment-contract-v1"
    assert contract.komi == 0.5
    assert contract.training_batch_size == 1024
    assert b0["train_batch_size"] == b1["train_batch_size"] == 1024


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


def test_effective_config_whitelist_allows_only_treatment_structure(canonical):
    _contract, b0, b1 = canonical
    allowed = copy.deepcopy(b1)
    allowed["lr"] = float(b0["lr"]) + 0.001
    # The learning-rate assignment is deliberately not a permitted change.
    with pytest.raises(ExperimentContractError, match="lr"):
        validate_b0_b1_effective_configs(b0, allowed)

    structural = copy.deepcopy(b1)
    structural["gocube_network_architecture"] = "test-architecture"
    structural["network_architecture_id"] = "test-architecture"
    assert diff_effective_configs(b0, structural) == []


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


def test_machine_readable_record_round_trips(canonical, tmp_path):
    contract, b0, b1 = canonical
    path = tmp_path / "gocube-b-experiment-contract.json"
    write_b_experiment_record(path, contract, b0, b1)
    payload = validate_b_experiment_record(path)
    assert payload["experiment_contract"]["training_batch_size"] == 1024
