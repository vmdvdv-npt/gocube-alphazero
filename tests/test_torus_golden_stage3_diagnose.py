from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.torus_golden_stage3_diagnose import (
    DiagnosticError,
    _expected_checkpoint_identities,
    _validate_contracts,
    divergence_metrics,
    masked_nn_prior,
    observation_bytes,
    reachable_collision_fixture,
    target_conflict_for_group,
)
from gocube_golden.neural import ACTION_COUNT, build_observation
from gocube_golden.rules import apply_action
from gocube_golden.state import initial_state
from gocube_golden.stage3_contract import PROFILE_ID, SELFPLAY_CONTRACT_FINGERPRINT, load_profile


def _one_hot(index: int) -> tuple[float, ...]:
    return tuple(1.0 if action == index else 0.0 for action in range(ACTION_COUNT))


def test_divergence_metrics_are_finite_with_zero_probabilities() -> None:
    target = _one_hot(0)
    prior = _one_hot(1)
    metrics = divergence_metrics(target, prior)
    assert all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (float, int)))
    assert metrics["kl_pi_to_p"] > 0.0
    assert metrics["tv_pi_p"] == pytest.approx(1.0)
    assert metrics["l1_pi_p"] == pytest.approx(2.0)
    assert metrics["top1_agreement"] is False


def test_exact_observation_group_key_uses_float32_bytes() -> None:
    first = torch.zeros((6, 25), dtype=torch.float32)
    second = first.clone()
    changed = first.clone()
    changed[0, 0] = 1.0
    assert observation_bytes(first) == observation_bytes(second)
    assert observation_bytes(first) != observation_bytes(changed)
    with pytest.raises(DiagnosticError):
        observation_bytes(torch.zeros((25, 6), dtype=torch.float32))


def test_reachable_transposition_has_same_observation_but_different_identity() -> None:
    fixture = reachable_collision_fixture()
    assert fixture["reachable_from_canonical_empty_state"] is True
    assert fixture["same_observation"] is True
    assert fixture["same_current_legal_mask"] is True
    assert fixture["different_full_state"] is True
    assert fixture["different_superko_history"] is True


def test_target_conflict_floor_is_empirical_mean_cross_entropy() -> None:
    common = {"state_key": "state", "history_key": "history", "z": [1.0, 0.0, 0.0]}
    left = {**common, "pi": list(_one_hot(0))}
    right = {**common, "pi": list(_one_hot(1))}
    conflict = target_conflict_for_group([left, right])
    assert conflict["mean_target"][0:2] == pytest.approx([0.5, 0.5])
    assert conflict["mean_ce_to_mean_target"] == pytest.approx(math.log(2.0))
    assert conflict["mean_pair_tv"] == pytest.approx(1.0)
    assert conflict["state_aliasing"] is False


def test_source_checkpoint_chunk_lineage_is_fail_closed() -> None:
    manifest = {
        "chunks": [
            {"chunk": 1, "source_checkpoint": "M0", "source_model_hash": "m0", "model_hash": "m1", "artifact_sha256": "a1"},
            {"chunk": 2, "source_checkpoint": "M1", "source_model_hash": "m1", "model_hash": "m2", "artifact_sha256": "a2"},
            {"chunk": 3, "source_checkpoint": "M2", "source_model_hash": "m2", "model_hash": "m3", "artifact_sha256": "a3"},
            {"chunk": 4, "source_checkpoint": "M3", "source_model_hash": "m3", "model_hash": "m4", "artifact_sha256": "a4"},
        ],
        "arena": {},
    }
    models, artifacts = _expected_checkpoint_identities(manifest)
    assert models == {"M0": "m0", "M1": "m1", "M2": "m2", "M3": "m3", "M4": "m4"}
    assert artifacts == {"M1": "a1", "M2": "a2", "M3": "a3", "M4": "a4"}
    bad = {**manifest, "chunks": [*manifest["chunks"][:-1], {**manifest["chunks"][-1], "source_model_hash": "wrong"}]}
    with pytest.raises(DiagnosticError, match="conflicting model hash"):
        _expected_checkpoint_identities(bad)


def test_profile_fingerprint_mismatch_fails_closed() -> None:
    profile = load_profile()
    manifest = {
        "canonical": True,
        "run_kind": "canonical",
        "profile_id": PROFILE_ID,
        "profile_fingerprint": "sha256:" + "0" * 64,
        "run_id": "run",
        "self_play_contract": {"fingerprint": SELFPLAY_CONTRACT_FINGERPRINT, "contract_id": "golden-selfplay-search-v1"},
        "arena_contract": {"search_contract_fingerprint": "sha256:" + "0" * 64, "search_implementation_id": "golden-sequential-puct-v1"},
        "preflight": {"observation_fingerprint": profile["observation"]["fingerprint"]},
        "chunks": [{}, {}, {}, {}],
    }
    with pytest.raises(DiagnosticError, match="profile fingerprint"):
        _validate_contracts(__import__("pathlib").Path("run"), manifest, profile)


def test_masked_prior_does_not_assign_probability_to_illegal_action() -> None:
    class FixedModel(torch.nn.Module):
        def forward(self, observation):
            logits = torch.zeros((observation.shape[0], ACTION_COUNT))
            logits[:, 0] = 10.0
            return logits, torch.zeros((observation.shape[0], 3))

    state = initial_state()
    prior = masked_nn_prior(FixedModel(), build_observation(state), (False,) + (True,) * 25)
    assert prior[0] == 0.0
    assert sum(prior) == pytest.approx(1.0)


def test_diagnostic_input_helpers_do_not_mutate_state_or_observation() -> None:
    state = initial_state()
    observation = build_observation(state)
    before_state = state.state_key
    before_observation = observation.clone()
    _ = observation_bytes(observation)
    _ = build_observation(state)
    assert state.state_key == before_state
    assert torch.equal(observation, before_observation)
