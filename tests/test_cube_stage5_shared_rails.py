from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import torch

from gocube_golden.cube_contract import CUBE_PROFILE_ID, load_profile
from gocube_golden.cube_neural import (
    GoldenCubeGraphNetV1,
    build_cube_action_mask,
    build_cube_observation,
    build_cube_observation_into,
)
from gocube_golden.cube_selfplay import CubeSelfPlayAdapter
from gocube_golden.cube_training import (
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    CubeSelfPlayRunner,
    CubeTrainingSample,
    cube_compare_selfplay_evidence,
    cube_initial_state,
    cube_state_identity,
    run_cube_selfplay_games,
)
from gocube_golden.cube_training_adapter import (
    CubeCumulativeReplay,
    CubeTrainingAdapter,
)
from gocube_golden.provenance import capture_code_identity
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
        run_id="stage5-test",
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
        "run_id": "stage5-test",
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
    assert expected.numpy().tobytes() == destination.numpy().tobytes()


def test_cube_public_selfplay_uses_shared_rails_and_replenishes_globally():
    model = GoldenCubeGraphNetV1()
    telemetry: dict[str, object] = {}
    contract = replace(DEFAULT_CUBE_SELFPLAY_CONTRACT, simulations=1, watchdog=1)
    records = run_cube_selfplay_games(
        model,
        tuple(f"game-{index}" for index in range(5)),
        checkpoint_path=None,
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
    assert telemetry["central_inference_owner_pid"] != 0
    assert telemetry["max_inference_batch_rows"] >= 2
    assert telemetry["global_task_replenishment"] is True


def test_cube_selfplay_matches_serial_oracle_on_fixed_cpu_fixture():
    model = GoldenCubeGraphNetV1()
    contract = replace(DEFAULT_CUBE_SELFPLAY_CONTRACT, simulations=1, watchdog=2)
    kwargs = _selfplay_kwargs(contract)
    old = CubeSelfPlayRunner(model, **kwargs).play_game("parity-game")
    new = run_cube_selfplay_games(
        model,
        ("parity-game",),
        checkpoint_path=None,
        workers=1,
        active_games_per_worker=1,
        total_active_contexts=1,
        inference_batch_cap=1,
        inference_batch_wait_ms=0.0,
        **kwargs,
    )
    cube_compare_selfplay_evidence((old,), new)


def test_cube_current_selfplay_path_has_no_process_pool_dependency():
    source = inspect.getsource(run_cube_selfplay_games)
    assert "ProcessPoolExecutor" not in source
    assert "CubeSelfPlayRunner" not in source
    assert "run_cube_selfplay_games_shared" in source


def test_cube_training_adapter_keeps_cumulative_replay_and_transaction(tmp_path: Path):
    profile = load_profile()
    adapter = CubeTrainingAdapter(profile=profile)
    model = GoldenCubeGraphNetV1()
    state = adapter.create_state(model, run_id="stage5-test")
    result = TrainingEngine(adapter).run_iteration(
        state=state,
        generation=1,
        output_dir=tmp_path,
        run_id="stage5-test",
        training_seed=2026091405,
        samples=(_sample("game-0"),),
        device="cpu",
    )
    assert result.checkpoint_metadata["profile_fingerprint"] == profile["profile_fingerprint"]
    assert result.checkpoint_metadata["training_profile_fingerprint"] == profile["profile_fingerprint"]
    assert state.optimizer_updates == 1
    assert state.samples_consumed == 1
    assert len(state.rolling_replay.rows) == 1
    assert state.rolling_replay.total_evictions == 0
    assert (tmp_path / "generation-01.complete.json").is_file()


def test_cube_cumulative_replay_has_no_eviction_cap():
    replay = CubeCumulativeReplay()
    replay.append_generation(1, [{"replay_row_id": "g1", "source_generation": 1}])
    replay.append_generation(2, [{"replay_row_id": "g2", "source_generation": 2}])
    assert [row["replay_row_id"] for row in replay.rows] == ["g1", "g2"]
    assert replay.total_evictions == 0
