from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import gocube_golden.torus9 as public_torus9
from gocube_golden import torus9_monolith
from gocube_golden.selfplay_engine import SelfPlayEngine
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_DIRICHLET_ALPHA,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_KOMI,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from gocube_golden.torus9_selfplay import run_torus9_selfplay_games, torus9_game_seed


def _current_contract():
    return public_torus9.Torus9SelfPlaySearchContract(
        contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        dirichlet_alpha=TORUS9_CURRENT_DIRICHLET_ALPHA,
    )


def test_public_torus9_selfplay_front_door_is_the_shared_engine_adapter():
    assert public_torus9.run_torus9_selfplay_games is run_torus9_selfplay_games
    assert public_torus9.run_torus9_selfplay_games.__module__ == "gocube_golden.torus9_selfplay"
    source = inspect.getsource(run_torus9_selfplay_games)
    assert "SelfPlayEngine" in source
    assert "Torus9SelfPlayRunner" not in source
    assert "ThreadPoolExecutor" not in source
    assert not hasattr(torus9_monolith, "Torus9SelfPlayRunner")
    assert not hasattr(torus9_monolith, "Torus9BatchedPUCT")


def test_generic_engine_has_no_torus_or_legacy_training_dependency():
    source = inspect.getsource(SelfPlayEngine)
    whole_module = inspect.getsource(__import__("gocube_golden.selfplay_engine", fromlist=["*"]))
    assert "Torus9" not in source
    for forbidden in ("alphazero.Coach", "SelfPlayAgent", "NNetWrapper", "GenericPlayers"):
        assert forbidden not in whole_module


def test_current_adapter_preserves_scientific_identity_and_komi():
    assert TORUS9_KOMI == 0.5
    profile = load_torus9_current_profile()
    assert current_torus9_profile_fingerprint(profile) == profile["profile_fingerprint"]
    contract = _current_contract()
    contract.validate()
    assert (contract.simulations, contract.cpuct, contract.fpu) == (64, 1.25, 0.0)
    assert contract.temperature_until_ply == 8
    assert contract.temperature_after == 0.0
    assert contract.dirichlet_epsilon == 0.25
    assert contract.dirichlet_alpha == 0.11


def test_game_seed_is_independent_of_worker_scheduling():
    expected = [torus9_game_seed(202609131002, "run", f"game-{index}") for index in range(8)]
    reordered = {
        game_id: torus9_game_seed(202609131002, "run", game_id)
        for game_id in reversed([f"game-{index}" for index in range(8)])
    }
    assert expected == [reordered[f"game-{index}"] for index in range(8)]


def test_empty_batch_still_crosses_the_selfplay_engine_boundary():
    torch.set_num_threads(1)
    model = public_torus9.Torus9CurrentGraphNet().eval()
    profile_fp = current_torus9_profile_fingerprint(load_torus9_current_profile())
    inference: dict[str, object] = {}
    execution: dict[str, object] = {}
    records = run_torus9_selfplay_games(
        model,
        run_id="boundary-test",
        label="M17",
        artifact="sha256:test",
        master_seed=202609131002,
        profile_fp=profile_fp,
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        game_ids=[],
        workers=16,
        device="cpu",
        contract=_current_contract(),
        coalescing=True,
        inference_batch_cap=16,
        inference_batch_wait_ms=6.0,
        inference_telemetry=inference,
        execution_activity=execution,
    )
    assert records == ()
    assert inference["batch_cap"] == 16
    assert inference["wait_ms"] == 6.0
    assert inference["execution_reference_status"] == "non_recommended"
    assert inference["execution_override_reason"] is None
    assert inference["execution_reference"]["effective_context_ceiling"] == 0
    assert inference["performance_reference"]["status"] == "NOT_COMPARABLE_UNDERFILLED"
    assert execution["configured_workers"] == 16
    assert execution["games_requested"] == 0


def test_current_profile_fingerprint_drift_fails_closed():
    model = public_torus9.Torus9CurrentGraphNet().eval()
    with pytest.raises(ValueError, match="fingerprint drift"):
        run_torus9_selfplay_games(
            model,
            run_id="boundary-test",
            label="M17",
            artifact="sha256:test",
            master_seed=1,
            profile_fp="sha256:wrong",
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            game_ids=[],
            workers=16,
            device="cpu",
            contract=_current_contract(),
            coalescing=True,
            inference_batch_cap=16,
            inference_batch_wait_ms=0.0,
        )
