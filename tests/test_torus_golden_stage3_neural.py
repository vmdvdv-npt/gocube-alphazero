from __future__ import annotations

import copy

import pytest
import torch

import gocube_golden as g
from gocube_golden.neural import GoldenNeuralEvaluator
from gocube_golden.stage3_contract import load_profile
from gocube_golden.training import GoldenTrainer, GoldenTrainingSample, load_checkpoint, save_checkpoint


def _state_with_board():
    stones = [g.EMPTY] * 25
    stones[0] = g.BLACK
    stones[1] = g.WHITE
    return g.research_state_from_stones(stones, side_to_move=g.BLACK)


def _sample():
    state = g.initial_state()
    observation = g.build_observation(state)
    visits = tuple([1] * 26)
    pi = tuple(1.0 / 26.0 for _ in range(26))
    return GoldenTrainingSample(
        run_id="test-run",
        game_id="game-00",
        ply=1,
        state={
            "stones": [int(stone) for stone in state.stones],
            "side_to_move": int(state.side_to_move),
            "superko_history": [list(position) for position in state.superko_history],
            "consecutive_passes": state.consecutive_passes,
            "topology_id": state.topology.topology_id,
            "topology_fingerprint": state.topology.fingerprint,
            "rules_id": state.rules_id,
            "rules_fingerprint": state.rules_fingerprint,
            "komi": state.komi,
            "history_provenance": state.history_provenance,
        },
        side_to_move="BLACK",
        observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
        legal_action_mask=tuple([True] * 26),
        root_visits=visits,
        pi=pi,
        z=(1.0, 0.0, 0.0),
        model_hash="sha256:" + "0" * 64,
        selfplay_contract_fingerprint=g.DEFAULT_SELFPLAY_CONTRACT.fingerprint,
    )


def test_observation_channels_mask_history_and_pass_contract():
    state = _state_with_board()
    black = g.build_observation_bundle(state)
    white = g.build_observation_bundle(
        g.research_state_from_stones(state.stones, side_to_move=g.WHITE)
    )
    assert tuple(black.tensor.shape) == (6, 25)
    assert black.tensor.dtype == torch.float32
    assert black.action_mask[25] is True
    assert black.tensor[0, 0] == 1 and black.tensor[1, 1] == 1
    assert white.tensor[0, 1] == 1 and white.tensor[1, 0] == 1
    assert black.tensor[2, 0] == 1 and white.tensor[2, 0] == -1
    assert torch.all(black.tensor[5] == 0.5)

    empty = tuple([g.EMPTY] * 25)
    blocked_board = list(empty)
    blocked_board[0] = g.BLACK
    without_history = g.research_state_from_stones(empty, side_to_move=g.BLACK)
    with_history = g.research_state_from_stones(
        empty, side_to_move=g.BLACK, superko_history=(empty, tuple(blocked_board))
    )
    assert g.build_observation_bundle(without_history).action_mask[0] is True
    assert g.build_observation_bundle(with_history).action_mask[0] is False


def test_terminal_state_is_never_observed_and_network_has_only_two_heads():
    terminal = g.apply_action(g.apply_action(g.initial_state(), g.PASS).after, g.PASS).after
    with pytest.raises(ValueError, match="Terminal"):
        g.build_observation(terminal)
    model = g.GoldenGraphNetV1()
    policy, value = model(g.build_observation(g.initial_state()).unsqueeze(0))
    assert tuple(policy.shape) == (1, 26)
    assert tuple(value.shape) == (1, 3)
    evaluation = GoldenNeuralEvaluator(model).evaluate(g.initial_state())
    assert len(evaluation.policy) == 26
    assert len(evaluation.wdl) == 3
    assert sum(evaluation.wdl) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("winner", "side", "expected"),
    [
        ("BLACK", g.BLACK, (1.0, 0.0, 0.0)),
        ("BLACK", g.WHITE, (0.0, 0.0, 1.0)),
        ("WHITE", g.WHITE, (1.0, 0.0, 0.0)),
        ("WHITE", g.BLACK, (0.0, 0.0, 1.0)),
        ("DRAW", g.BLACK, (0.0, 1.0, 0.0)),
        ("DRAW", g.WHITE, (0.0, 1.0, 0.0)),
    ],
)
def test_z_is_finally_side_to_move_relative(winner, side, expected):
    assert g.z_target(winner, side) == expected


def test_tiny_training_fixture_changes_model_and_checkpoint_round_trips(tmp_path):
    model = g.GoldenGraphNetV1()
    sample = _sample()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.inference_mode():
        initial = model(torch.tensor([sample.observation], dtype=torch.float32))
        initial_loss = float(
            -(torch.tensor([sample.pi]) * torch.log_softmax(initial[0], dim=1)).sum()
            -(torch.tensor([sample.z]) * torch.log_softmax(initial[1], dim=1)).sum()
        )
    trainer = GoldenTrainer(model)
    metrics = trainer.train([sample] * 4, updates=3, batch_size=4, seed=7)
    assert metrics[-1].total_loss < initial_loss
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    checkpoint = tmp_path / "M1.pt"
    profile = load_profile()
    checkpoint_metadata = {
        "parent_or_source_run_identity": "test",
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "rules_profile_id": profile["frozen_identities"]["rules_profile_id"],
        "rules_fingerprint": profile["frozen_identities"]["rules_fingerprint"],
        "topology_fingerprint": profile["frozen_identities"]["topology_fingerprint"],
        "board_size": [5, 5],
        "point_id_order_identity": "row-major-yx:point_id=y*width+x",
        "komi": 0.5,
        "observation_schema_id": profile["observation"]["schema_id"],
        "observation_schema_version": profile["observation"]["schema_version"],
        "observation_fingerprint": profile["observation"]["fingerprint"],
        "target_contract_id": profile["target"]["contract_id"],
        "target_contract_version": profile["target"]["contract_version"],
        "target_fingerprint": profile["target"]["fingerprint"],
        "value_head_semantics": "side-to-move:[WIN,DRAW,LOSS]",
        "network_heads_and_shapes": {"policy": [26], "value": [3]},
        "training_profile_id": profile["profile_id"],
        "training_profile_fingerprint": profile["profile_fingerprint"],
    }
    metadata = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=trainer.optimizer,
        metadata=checkpoint_metadata,
    )
    restored = g.GoldenGraphNetV1()
    restored_trainer = GoldenTrainer(restored)
    loaded = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_trainer.optimizer,
        expected={"model_hash": metadata["model_hash"]},
    )
    assert loaded["model_hash"] == metadata["model_hash"]
    with torch.inference_mode():
        old_logits = model(torch.tensor([sample.observation], dtype=torch.float32))
        new_logits = restored(torch.tensor([sample.observation], dtype=torch.float32))
    assert torch.allclose(old_logits[0], new_logits[0], atol=1e-7, rtol=1e-7)
    assert torch.allclose(old_logits[1], new_logits[1], atol=1e-7, rtol=1e-7)
