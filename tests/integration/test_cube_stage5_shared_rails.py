from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import torch

from gocube_golden.cube_contract import CUBE_PROFILE_ID, load_profile
from gocube_golden.cube_neural import (
    GoldenCubeGraphNetV1,
    SelfPlayCubeRootNoiseEvaluator,
    apply_cube_root_dirichlet_noise,
    build_cube_action_mask,
    build_cube_observation,
    build_cube_observation_into,
)
from gocube_golden.cube_selfplay import _CubeRootNoiseTransform, run_cube_selfplay_games_shared
from gocube_golden.cube_training import (
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    CubeTrainingSample,
    cube_initial_state,
    cube_state_identity,
    run_cube_selfplay_games,
)
from gocube_golden.cube_training_adapter import CubeTrainingAdapter
from gocube_golden.provenance import capture_code_identity
from gocube_golden.rules import prepare_legal_actions
from gocube_golden.search import Evaluation
from training_engine import TrainingEngine


def _sample(game_id: str) -> dict[str, object]:
    state = cube_initial_state()
    observation = build_cube_observation(state)
    mask = build_cube_action_mask(state)
    visits = [0] * 97
    visits[96] = 64
    pi = [0.0] * 97
    pi[96] = 1.0
    return CubeTrainingSample(
        run_id="stage6-test",
        game_id=game_id,
        ply=1,
        state=cube_state_identity(state),
        side_to_move=state.side_to_move.name,
        observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
        legal_action_mask=mask,
        root_visits=tuple(visits),
        pi=tuple(pi),
        z=(0.0, 1.0, 0.0),
        model_hash="sha256:" + "0" * 64,
        selfplay_contract_fingerprint=DEFAULT_CUBE_SELFPLAY_CONTRACT.fingerprint,
    ).to_dict()


def _selfplay_kwargs(contract):
    profile = load_profile()
    return {
        "run_id": "stage6-test",
        "profile_id": CUBE_PROFILE_ID,
        "profile_fingerprint": profile["profile_fingerprint"],
        "model_checkpoint_label": "M0",
        "checkpoint_artifact_hash": "sha256:" + "0" * 64,
        "master_seed": 2026091402,
        "chunk_id": "test",
        "code_identity": capture_code_identity(),
        "contract": contract,
        "allow_noncanonical_contract": True,
        "device": "cpu",
    }


def test_cube_observation_writer_has_exact_float32_parity():
    state = cube_initial_state()
    expected = build_cube_observation(state)
    destination = torch.empty_like(expected)
    assert build_cube_observation_into(state, destination) is None
    assert torch.equal(expected, destination)


def test_cube_root_noise_uses_one_shared_scientific_primitive():
    state = cube_initial_state()
    legal_context = prepare_legal_actions(state)
    base = Evaluation(policy=tuple(float(index + 1) for index in range(97)), wdl=(0.2, 0.3, 0.5))

    class StubEvaluator:
        def evaluate_prepared(self, _state, _legal_context):
            return base

    serial = SelfPlayCubeRootNoiseEvaluator(StubEvaluator(), state, seed=2026091510)
    shared = _CubeRootNoiseTransform(state, seed=2026091510, epsilon=0.25, alpha=0.30)
    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(2026091510)
    expected = apply_cube_root_dirichlet_noise(
        base.policy,
        legal_context.actions,
        epsilon=0.25,
        alpha=0.30,
        generator=expected_generator,
    )
    assert serial.evaluate_prepared(state, legal_context).policy == shared(base, state, legal_context).policy == expected


def test_cube_public_selfplay_uses_shared_engine_rails():
    model = GoldenCubeGraphNetV1()
    telemetry: dict[str, object] = {}
    contract = replace(DEFAULT_CUBE_SELFPLAY_CONTRACT, simulations=1, watchdog=1)
    records = run_cube_selfplay_games(
        model,
        tuple(f"game-{index}" for index in range(5)),
        workers=2,
        active_games_per_worker=2,
        total_active_contexts=4,
        inference_batch_cap=4,
        inference_batch_wait_ms=1.0,
        inference_telemetry=telemetry,
        **_selfplay_kwargs(contract),
    )
    assert [record.game_id for record in records] == [f"game-{index}" for index in range(5)]
    assert all(record.technical_termination == "TRUNCATED_MOVE_LIMIT" for record in records)
    assert telemetry["shared_memory_transport"] is True
    assert telemetry["global_task_replenishment"] is True
    assert "run_cube_selfplay_games_shared" in inspect.getsource(run_cube_selfplay_games)
    assert "SelfPlayEngine" in inspect.getsource(run_cube_selfplay_games_shared)


def test_cube_training_adapter_uses_training_engine_transaction(tmp_path: Path):
    profile = load_profile()
    adapter = CubeTrainingAdapter(profile=profile)
    state = adapter.create_state(GoldenCubeGraphNetV1(), run_id="stage6-test")
    result = TrainingEngine(adapter).run_iteration(
        state=state,
        generation=1,
        output_dir=tmp_path,
        run_id="stage6-test",
        training_seed=2026091405,
        samples=(_sample("game-0"),),
        device="cpu",
    )
    assert result.checkpoint_metadata["profile_fingerprint"] == profile["profile_fingerprint"]
    assert state.optimizer_updates == 1
    assert state.samples_consumed == 1
    assert len(state.rolling_replay.rows) == 1
    assert state.rolling_replay.total_evictions == 0
    assert (tmp_path / "generation-01.complete.json").is_file()
