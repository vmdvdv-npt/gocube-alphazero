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
from gocube_golden.cube_selfplay import CubeSelfPlayAdapter, _CubeRootNoiseTransform
from gocube_golden.cube_training import (
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    CubeSelfPlayRunner,
    CubeTrainingSample,
    cube_compare_selfplay_evidence,
    cube_initial_state,
    cube_replay_batches,
    cube_state_identity,
    run_cube_selfplay_games,
    train_cube_batch_schedule,
)
from gocube_golden.cube_training_adapter import (
    CubeCumulativeReplay,
    CubeTrainingAdapter,
)
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


def test_cube_root_noise_uses_one_shared_scientific_primitive():
    state = cube_initial_state()
    legal_context = prepare_legal_actions(state)
    base = Evaluation(
        policy=tuple(float(index + 1) for index in range(97)),
        wdl=(0.2, 0.3, 0.5),
    )

    class StubEvaluator:
        def evaluate_prepared(self, _state, _legal_context):
            return base

    serial = SelfPlayCubeRootNoiseEvaluator(StubEvaluator(), state, seed=2026091510)
    shared = _CubeRootNoiseTransform(
        state,
        seed=2026091510,
        epsilon=0.25,
        alpha=0.30,
    )
    serial_output = serial.evaluate_prepared(state, legal_context)
    shared_output = shared(base, state, legal_context)
    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(2026091510)
    expected_policy = apply_cube_root_dirichlet_noise(
        base.policy,
        legal_context.actions,
        epsilon=0.25,
        alpha=0.30,
        generator=expected_generator,
    )
    assert serial_output.policy == shared_output.policy == expected_policy


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


def _assert_nested_exact(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_exact(left_item, right_item)
        return
    assert left == right


def test_cube_old_new_training_is_exact_in_rows_losses_model_adam_and_counters():
    raw_samples = tuple(_sample(f"parity-game-{index:03d}") for index in range(65))
    profile = load_profile()
    adapter = CubeTrainingAdapter(profile=profile)
    new_model = GoldenCubeGraphNetV1()
    old_model = GoldenCubeGraphNetV1()
    old_model.load_state_dict(new_model.state_dict(), strict=True)
    new_replay = CubeCumulativeReplay()
    stamped = adapter.stamp_samples(raw_samples, 1)
    adapter.update_replay(new_replay, 1, stamped)
    new_state = adapter.create_state(new_model, run_id="stage5-parity", replay=new_replay)

    seed = 2026091511
    batches = cube_replay_batches(65, 65, seed=seed, batch_size=64)
    old_samples = tuple(CubeTrainingSample(**sample) for sample in raw_samples)
    old_optimizer, old_metrics = train_cube_batch_schedule(
        old_model,
        old_samples,
        batches,
        learning_rate=0.001,
        weight_decay=0.0,
    )
    new_metrics = adapter.train(new_state, new_replay.rows, seed)

    assert tuple(new_metrics["sampled_batches"]) == batches
    expected_sampled_ids = tuple(
        stamped[index]["replay_row_id"] for batch in batches for index in batch
    )
    assert tuple(new_metrics["sampled_replay_row_ids"]) == expected_sampled_ids
    assert new_metrics["metrics"] == old_metrics["metrics"]
    assert new_metrics["batch_sizes"] == old_metrics["batch_sizes"]
    assert new_metrics["exact_samples_consumed"] == old_metrics["exact_samples_consumed"]
    assert new_metrics["updates"] == old_metrics["updates"]
    for name, parameter in old_model.state_dict().items():
        assert torch.equal(parameter, new_model.state_dict()[name]), name
    from gocube_golden.cube_neural import cube_model_hash

    assert cube_model_hash(old_model) == cube_model_hash(new_model)
    _assert_nested_exact(old_optimizer.state_dict(), new_state.optimizer.state_dict())
    assert new_state.optimizer_updates == old_metrics["updates"] == 2
    assert new_state.samples_consumed == old_metrics["cumulative_samples"] == 65


def test_cube_resume_after_reload_is_exactly_equal_to_continuous_training(tmp_path: Path):
    profile = load_profile()
    code = capture_code_identity()
    generation_a = tuple(_sample(f"resume-a-{index}") for index in range(2))
    generation_b = tuple(_sample(f"resume-b-{index}") for index in range(3))
    initial_model = GoldenCubeGraphNetV1()

    continuous_adapter = CubeTrainingAdapter(profile=profile, code_identity=code)
    continuous_model = GoldenCubeGraphNetV1()
    continuous_model.load_state_dict(initial_model.state_dict(), strict=True)
    continuous_state = continuous_adapter.create_state(
        continuous_model, run_id="stage5-resume", parent_checkpoint_identity=None
    )
    continuous_dir = tmp_path / "continuous"
    continuous_engine = TrainingEngine(continuous_adapter)
    continuous_engine.run_iteration(
        state=continuous_state,
        generation=1,
        output_dir=continuous_dir,
        run_id="stage5-resume",
        training_seed=2026091512,
        samples=generation_a,
        code_identity=code,
        device="cpu",
    )
    continuous_b = continuous_engine.run_iteration(
        state=continuous_state,
        generation=2,
        output_dir=continuous_dir,
        run_id="stage5-resume",
        training_seed=2026091513,
        samples=generation_b,
        code_identity=code,
        device="cpu",
    )

    split_adapter = CubeTrainingAdapter(profile=profile, code_identity=code)
    split_model = GoldenCubeGraphNetV1()
    split_model.load_state_dict(initial_model.state_dict(), strict=True)
    split_state = split_adapter.create_state(
        split_model, run_id="stage5-resume", parent_checkpoint_identity=None
    )
    split_dir = tmp_path / "split"
    split_engine = TrainingEngine(split_adapter)
    split_a = split_engine.run_iteration(
        state=split_state,
        generation=1,
        output_dir=split_dir,
        run_id="stage5-resume",
        training_seed=2026091512,
        samples=generation_a,
        code_identity=code,
        device="cpu",
    )
    reloaded = split_adapter.load_state(
        split_a.artifacts["checkpoint"],
        replay_path=split_a.artifacts["rolling_replay"],
        device="cpu",
    )
    split_b = split_engine.run_iteration(
        state=reloaded,
        generation=2,
        output_dir=split_dir,
        run_id="stage5-resume",
        training_seed=2026091513,
        samples=generation_b,
        code_identity=code,
        device="cpu",
    )

    assert continuous_b.fresh_positions == split_b.fresh_positions == 3

    def stable_training_metrics(metrics):
        return {
            key: value
            for key, value in metrics.items()
            if key != "training_wall_time_sec"
        }

    assert stable_training_metrics(continuous_b.training_metrics) == stable_training_metrics(split_b.training_metrics)
    assert continuous_state.optimizer_updates == reloaded.optimizer_updates == 2
    assert continuous_state.samples_consumed == reloaded.samples_consumed == 5
    assert continuous_state.rolling_replay.rows == reloaded.rolling_replay.rows
    for name, parameter in continuous_state.model.state_dict().items():
        assert torch.equal(parameter, reloaded.model.state_dict()[name]), name
    from gocube_golden.cube_neural import cube_model_hash

    assert cube_model_hash(continuous_state.model) == cube_model_hash(reloaded.model)
    _assert_nested_exact(continuous_state.optimizer.state_dict(), reloaded.optimizer.state_dict())


def test_cube_cumulative_replay_has_no_eviction_cap():
    replay = CubeCumulativeReplay()
    replay.append_generation(1, [{"replay_row_id": "g1", "source_generation": 1}])
    replay.append_generation(2, [{"replay_row_id": "g2", "source_generation": 2}])
    assert [row["replay_row_id"] for row in replay.rows] == ["g1", "g2"]
    assert replay.total_evictions == 0
