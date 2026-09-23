from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch

from gocube_golden.cube_family import cube_family_topology
from gocube_golden.cube_network_v2 import ARCHITECTURE_FINGERPRINT, ARCHITECTURE_ID, CubeGraphNetV2
from gocube_golden.cube_replay_v2 import CubeReplayCodecV2
from gocube_golden.cube_selfplay_contract import CUBE_SELFPLAY_SEMANTICS_FINGERPRINT, CubeSelfPlaySearchContract
from gocube_golden.cube_selfplay_v2 import CubeSelfPlayAdapter, _CubeCooperativeGame
from gocube_golden.cube_training_contract_v2 import CHECKPOINT_SCHEMA, REPLAY_SCHEMA, CubeTrainingConfig, load_cube_training_contract
from gocube_golden.cube_training_targets import CubeTrainingSampleV2, build_cube_training_samples
from gocube_golden.cube_training_v2 import create_cube_m0_state, load_cube_checkpoint, run_cube_training_generation
from gocube_golden.search import Evaluation
from gocube_golden.selfplay_engine import GameFinished, InferenceNeed


def _config(*, replay_generations=2, replay_cap=64, steps=1, batch=1):
    return CubeTrainingConfig(
        learning_rate=0.001,
        batch_size=batch,
        optimizer_steps=steps,
        replay_generations=replay_generations,
        replay_cap=replay_cap,
    )


def _synthetic_sample(adapter, *, game_id: str, ply: int = 1):
    p = adapter.topology.point_count
    a = adapter.topology.action_count
    policy = [0.0] * a
    policy[-1] = 1.0
    return CubeTrainingSampleV2(
        observation=torch.zeros((30, p), dtype=torch.float32),
        policy=tuple(policy),
        wdl=(0, 1, 0),
        ownership=tuple("NEUTRAL" for _ in range(p)),
        score_exact=0.0,
        score_normalized=0.0,
        legal_action_mask=tuple(True for _ in range(a)),
        selected_action=a - 1,
        ply=ply,
        side_to_move="BLACK",
        game_id=game_id,
        size=adapter.size,
        game_identity_fingerprint=adapter.game_fingerprint,
        observation_fingerprint=adapter.observation_fingerprint,
        model_hash="sha256:" + "0" * 64,
        selfplay_semantics_fingerprint=CUBE_SELFPLAY_SEMANTICS_FINGERPRINT,
        search_config_fingerprint="sha256:" + "1" * 64,
    )


def _training_row(adapter, *, generation=1, game_id="synthetic"):
    row = adapter.codec.encode(_synthetic_sample(adapter, game_id=game_id))
    return adapter.stamp_samples((row,), generation)[0]


def _formal_pass_record(size: int):
    topology = cube_family_topology(size)
    model = CubeGraphNetV2(topology=topology).eval()
    adapter = CubeSelfPlayAdapter(
        model,
        size=size,
        contract=CubeSelfPlaySearchContract(
            simulations=1,
            root_noise=False,
            temperature_plies=(1, 1),
            technical_move_limit=3,
        ),
    )
    game = _CubeCooperativeGame(adapter.worker_context, f"cube{size}-formal-pass", None)
    while True:
        step = game.advance()
        if isinstance(step, GameFinished):
            return step.record
        assert isinstance(step, InferenceNeed)
        policy = [0.0] * topology.action_count
        policy[topology.pass_action] = 1.0
        game.resume(Evaluation(policy=tuple(policy), wdl=(0.0, 1.0, 0.0)))


@pytest.mark.parametrize("size", range(2, 8))
def test_cube2_to_cube7_one_step_finite_and_changes_parameters(size):
    adapter, state = create_cube_m0_state(size=size, config=_config(replay_cap=8), seed=1000 + size)
    row = _training_row(adapter, game_id=f"cube{size}")
    before = {name: value.detach().clone() for name, value in state.model.named_parameters()}
    metrics = adapter.train(state, (row,), seed=2000 + size)
    for key in ("policy_loss", "wdl_loss", "ownership_loss", "score_loss", "total_loss"):
        assert math.isfinite(float(metrics[key]))
    assert all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in state.model.parameters())
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in state.model.named_parameters())


def test_stage5_cube4_record_to_common_training_engine_checkpoint(tmp_path: Path):
    record = _formal_pass_record(4)
    samples = build_cube_training_samples(record)
    assert len(samples) == 2
    config = _config(replay_cap=16)
    adapter, state = create_cube_m0_state(size=4, config=config, seed=41)
    result = run_cube_training_generation(
        adapter=adapter,
        state=state,
        generation=1,
        lineage_dir=tmp_path,
        run_id="cube4-stage6",
        training_seed=42,
        records=(record,),
        device="cpu",
    )
    assert result.sample_count_new == 2
    assert result.replay_sample_count == 2
    assert result.optimizer_steps == 1
    assert Path(result.checkpoint_reference["path"]).is_file()
    assert result.engine_result.checkpoint_metadata["checkpoint_schema"] == CHECKPOINT_SCHEMA
    assert result.engine_result.checkpoint_metadata["replay_schema"] == REPLAY_SCHEMA
    assert (tmp_path / "generation-01.complete.json").is_file()


def test_technical_record_produces_zero_training_samples():
    topology = cube_family_topology(2)
    model = CubeGraphNetV2(topology=topology).eval()
    adapter = CubeSelfPlayAdapter(
        model,
        size=2,
        contract=CubeSelfPlaySearchContract(
            simulations=1,
            root_noise=False,
            temperature_plies=(1, 1),
            technical_move_limit=1,
        ),
    )
    game = _CubeCooperativeGame(adapter.worker_context, "technical", None)
    while True:
        step = game.advance()
        if isinstance(step, GameFinished):
            record = step.record
            break
        assert isinstance(step, InferenceNeed)
        policy = [0.0] * topology.action_count
        policy[topology.pass_action] = 1.0
        game.resume(Evaluation(policy=tuple(policy), wdl=(0.0, 1.0, 0.0)))
    assert record.technical_termination == "MOVE_LIMIT"
    assert build_cube_training_samples(record) == ()
    training_adapter, _ = create_cube_m0_state(size=2, config=_config(), seed=7)
    assert training_adapter.build_samples((record,)) == ()


def test_replay_window_cap_provenance_and_deterministic_sampling():
    config = _config(replay_generations=2, replay_cap=3)
    adapter, state = create_cube_m0_state(size=2, config=config, seed=17)
    for generation in (1, 2, 3):
        rows = tuple(
            adapter.codec.encode(_synthetic_sample(adapter, game_id=f"g{generation}-{index}", ply=index + 1))
            for index in range(2)
        )
        stamped = adapter.stamp_samples(rows, generation)
        adapter.update_replay(state.rolling_replay, generation, stamped)
    rows = tuple(adapter.replay_rows(state.rolling_replay))
    assert len(rows) == 3
    assert [row["source_generation"] for row in rows] == [2, 3, 3]
    adapter.validate_replay(rows)

    a1, s1 = create_cube_m0_state(size=2, config=config, seed=99)
    a2, s2 = create_cube_m0_state(size=2, config=config, seed=99)
    m1 = a1.train(s1, (_training_row(a1),), seed=123)
    m2 = a2.train(s2, (_training_row(a2),), seed=123)
    assert m1["sampled_replay_row_ids"] == m2["sampled_replay_row_ids"]
    assert m1["sampling_state"] == m2["sampling_state"]


def _nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _nested_equal(a, b)
    else:
        assert left == right


def test_checkpoint_save_load_restores_model_optimizer_and_sampling(tmp_path: Path):
    record = _formal_pass_record(4)
    config = _config(replay_cap=16)
    adapter, state = create_cube_m0_state(size=4, config=config, seed=51)
    result = run_cube_training_generation(
        adapter=adapter,
        state=state,
        generation=1,
        lineage_dir=tmp_path,
        run_id="save-load",
        training_seed=52,
        records=(record,),
        device="cpu",
    )
    loaded_adapter, loaded_state, metadata = load_cube_checkpoint(
        result.checkpoint_reference["path"],
        config=config,
        replay_path=result.engine_result.artifacts["rolling_replay"],
        expected_size=4,
    )
    assert metadata["generation"] == 1
    assert metadata["concrete_training_config_fingerprint"] == config.fingerprint
    for key, value in state.model.state_dict().items():
        assert torch.equal(value.cpu(), loaded_state.model.state_dict()[key].cpu())
    _nested_equal(state.optimizer.state_dict(), loaded_state.optimizer.state_dict())
    assert state.adapter_state == loaded_state.adapter_state
    loaded_adapter.validate_state(loaded_state)


def test_resume_step2_matches_uninterrupted_cpu_path(tmp_path: Path):
    record = _formal_pass_record(2)
    config = _config(replay_cap=16)

    adapter_a, state_a = create_cube_m0_state(size=2, config=config, seed=61)
    run_cube_training_generation(
        adapter=adapter_a,
        state=state_a,
        generation=1,
        lineage_dir=tmp_path / "a",
        run_id="path-a",
        training_seed=62,
        records=(record,),
        device="cpu",
    )
    run_cube_training_generation(
        adapter=adapter_a,
        state=state_a,
        generation=2,
        lineage_dir=tmp_path / "a",
        run_id="path-a",
        training_seed=63,
        records=(record,),
        device="cpu",
    )

    adapter_b, state_b = create_cube_m0_state(size=2, config=config, seed=61)
    first = run_cube_training_generation(
        adapter=adapter_b,
        state=state_b,
        generation=1,
        lineage_dir=tmp_path / "b",
        run_id="path-b",
        training_seed=62,
        records=(record,),
        device="cpu",
    )
    loaded_adapter, loaded_state, _ = load_cube_checkpoint(
        first.checkpoint_reference["path"],
        config=config,
        replay_path=first.engine_result.artifacts["rolling_replay"],
        expected_size=2,
    )
    run_cube_training_generation(
        adapter=loaded_adapter,
        state=loaded_state,
        generation=2,
        lineage_dir=tmp_path / "b",
        run_id="path-b",
        training_seed=63,
        records=(record,),
        device="cpu",
    )
    for key, value in state_a.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), loaded_state.model.state_dict()[key].cpu(), rtol=0.0, atol=0.0)
    _nested_equal(state_a.optimizer.state_dict(), loaded_state.optimizer.state_dict())
    assert state_a.adapter_state == loaded_state.adapter_state


def test_checkpoint_compatibility_fails_closed(tmp_path: Path):
    record = _formal_pass_record(2)
    config = _config(replay_cap=16)
    adapter, state = create_cube_m0_state(size=2, config=config, seed=71)
    result = run_cube_training_generation(
        adapter=adapter,
        state=state,
        generation=1,
        lineage_dir=tmp_path,
        run_id="compat",
        training_seed=72,
        records=(record,),
    )
    metadata = dict(result.engine_result.checkpoint_metadata)
    kwargs = adapter._metadata_kwargs()
    from gocube_golden.cube_checkpoint_v2 import validate_checkpoint_metadata

    wrong = deepcopy(metadata)
    wrong["observation_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="observation"):
        validate_checkpoint_metadata(wrong, **kwargs)
    wrong = deepcopy(metadata)
    wrong["architecture_id"] = "GoldenCubeGraphNetV1"
    with pytest.raises(ValueError, match="architecture"):
        validate_checkpoint_metadata(wrong, **kwargs)
    wrong = deepcopy(metadata)
    wrong["target_contract_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="target"):
        validate_checkpoint_metadata(wrong, **kwargs)
    wrong = deepcopy(metadata)
    wrong["training_semantics_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="training"):
        validate_checkpoint_metadata(wrong, **kwargs)
    with pytest.raises(ValueError, match="topology size"):
        load_cube_checkpoint(
            result.checkpoint_reference["path"],
            config=config,
            replay_path=result.engine_result.artifacts["rolling_replay"],
            expected_size=3,
        )


def test_static_contract_separates_semantics_from_run_owned_values():
    contract = load_cube_training_contract()
    assert contract["optimizer_family"] == "Adam"
    assert contract["required_architecture"] == {"id": ARCHITECTURE_ID, "fingerprint": ARCHITECTURE_FINGERPRINT}
    for key in ("learning_rate", "batch_size", "optimizer_steps", "replay_generations", "replay_cap"):
        assert key not in {name for name in contract if name != "run_owned_fields"}
        assert key in contract["run_owned_fields"]
