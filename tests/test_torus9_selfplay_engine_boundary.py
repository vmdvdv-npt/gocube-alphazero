from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import gocube_golden.torus9 as public_torus9
from gocube_golden import torus9_monolith
from gocube_golden.provenance import CodeIdentity
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


REFERENCE_PROFILE_FINGERPRINT = "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775"


def _current_contract():
    return public_torus9.Torus9SelfPlaySearchContract(
        contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        dirichlet_alpha=TORUS9_CURRENT_DIRICHLET_ALPHA,
    )


def test_public_torus9_selfplay_front_door_is_the_new_engine_adapter():
    assert public_torus9.run_torus9_selfplay_games is run_torus9_selfplay_games
    assert public_torus9.run_torus9_selfplay_games is not torus9_monolith.run_torus9_selfplay_games
    assert public_torus9.run_torus9_selfplay_games.__module__ == "gocube_golden.torus9_selfplay"
    assert "ThreadPoolExecutor" not in inspect.getsource(public_torus9.run_torus9_selfplay_games)


def test_generic_engine_has_no_torus_or_legacy_training_dependency():
    source = inspect.getsource(SelfPlayEngine)
    whole_module = inspect.getsource(__import__("gocube_golden.selfplay_engine", fromlist=["*"]))
    assert "Torus9" not in source
    for forbidden in ("alphazero.Coach", "SelfPlayAgent", "NNetWrapper"):
        assert forbidden not in whole_module


def test_current_adapter_preserves_golden_scientific_identity_and_komi():
    assert TORUS9_KOMI == 0.5
    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    assert profile_fp == REFERENCE_PROFILE_FINGERPRINT
    contract = _current_contract()
    contract.validate()
    assert contract.simulations == 64
    assert contract.cpuct == 1.25
    assert contract.fpu == 0.0
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


def test_current_wrapper_routes_even_empty_batch_through_engine_boundary():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260915)
        model = public_torus9.Torus9CurrentGraphNet().eval()
    profile_fp = current_torus9_profile_fingerprint(load_torus9_current_profile())
    inference = {}
    execution = {}
    records = run_torus9_selfplay_games(
        model,
        checkpoint_path=Path("unused-read-only-checkpoint.pt"),
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
    assert execution["configured_workers"] == 16
    assert execution["games_requested"] == 0


def test_current_profile_fingerprint_drift_fails_closed():
    torch.set_num_threads(1)
    model = public_torus9.Torus9CurrentGraphNet().eval()
    with pytest.raises(ValueError, match="fingerprint drift"):
        run_torus9_selfplay_games(
            model,
            checkpoint_path=Path("unused.pt"),
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
            inference_batch_wait_ms=6.0,
        )


def test_cpu_single_lane_exact_game_parity_with_pre_extraction_runner():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026091501)
        model = public_torus9.Torus9CurrentGraphNet().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.point_policy.bias.fill_(-20.0)
            model.pass_policy.bias.fill_(20.0)
    profile_fp = current_torus9_profile_fingerprint(load_torus9_current_profile())
    contract = _current_contract()
    code = CodeIdentity("parity-commit", "parity-tree", True)
    expected = torus9_monolith.Torus9SelfPlayRunner(
        model,
        run_id="parity-run",
        model_checkpoint_label="M17",
        checkpoint_artifact_hash="sha256:parity",
        master_seed=202609131002,
        profile_fp=profile_fp,
        code_identity=code,
        device="cpu",
        contract=contract,
        profile_id=TORUS9_CURRENT_PROFILE_ID,
    ).play_game("game-0000")
    actual = run_torus9_selfplay_games(
        model,
        checkpoint_path=Path("unused-read-only-checkpoint.pt"),
        run_id="parity-run",
        label="M17",
        artifact="sha256:parity",
        master_seed=202609131002,
        profile_fp=profile_fp,
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        game_ids=["game-0000"],
        workers=1,
        code_identity=code,
        device="cpu",
        contract=contract,
        coalescing=True,
        inference_batch_cap=16,
        inference_batch_wait_ms=0.0,
    )
    assert len(actual) == 1
    assert actual[0].to_dict() == expected.to_dict()
