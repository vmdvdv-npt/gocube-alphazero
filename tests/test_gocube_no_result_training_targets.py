from types import SimpleNamespace

import numpy as np
import torch

from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube import NO_RESULT, SCORED, Topology, V3Terminal, build_v3_training_targets


def rectangular_topology(width=3, height=3):
    point_ids = tuple(f"{x},{y}" for y in range(height) for x in range(width))
    index = {point_id: point for point, point_id in enumerate(point_ids)}
    neighbors = []
    for y in range(height):
        for x in range(width):
            neighbors.append(tuple(
                index[f"{nx},{ny}"]
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                if 0 <= nx < width and 0 <= ny < height
            ))
    return Topology("rect-test", width, point_ids, tuple(neighbors), index)


def scored_terminal(winner):
    score = SimpleNamespace(black=4.0, white=0.5, winner=winner)
    ownership = np.zeros((9, 3), dtype=np.float32)
    ownership[:, 2] = 1.0
    ownership_mask = np.ones(9, dtype=np.float32)
    return V3Terminal(SCORED, score, ownership, ownership_mask)


def test_no_result_retains_value_class_and_masks_score_and_ownership():
    topology = rectangular_topology()
    targets = build_v3_training_targets(
        V3Terminal(NO_RESULT, None, None, None, "cycle"),
        side_to_move=0,
        topology=topology,
    )

    assert targets.terminal_kind == NO_RESULT
    np.testing.assert_array_equal(targets.value_target, np.array([0, 0, 1], dtype=np.float32))
    assert np.isnan(targets.score_target).all()
    assert targets.score_mask.tolist() == [0.0]
    assert np.array_equal(targets.ownership_target, np.zeros((9, 3), dtype=np.float32))
    assert np.array_equal(targets.ownership_mask, np.zeros(9, dtype=np.float32))


def test_scored_win_loss_and_draw_value_semantics_are_distinct_from_no_result():
    topology = rectangular_topology()
    black = build_v3_training_targets(scored_terminal("black"), 0, topology)
    white = build_v3_training_targets(scored_terminal("black"), 1, topology)
    draw_terminal = V3Terminal(
        SCORED,
        SimpleNamespace(black=2.0, white=2.0, winner="draw"),
        np.zeros((9, 3), dtype=np.float32),
        np.ones(9, dtype=np.float32),
    )
    draw = build_v3_training_targets(draw_terminal, 0, topology)

    np.testing.assert_array_equal(black.value_target, [1, 0, 0])
    np.testing.assert_array_equal(white.value_target, [0, 1, 0])
    np.testing.assert_array_equal(draw.value_target, [0.5, 0.5, 0])
    assert black.score_mask.tolist() == [1.0]
    assert np.isfinite(black.score_target).all()


def _loss_wrapper():
    wrapper = object.__new__(NNetWrapper)
    wrapper.args = SimpleNamespace(
        ownership_loss_weight=0.5,
        score_loss_weight=0.5,
        value_loss_weight=1.0,
    )
    return wrapper


def test_masked_nan_score_rows_do_not_enter_score_loss_or_gradient():
    wrapper = _loss_wrapper()
    outputs = torch.tensor([[0.2], [0.4]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[0.1], [float("nan")]], dtype=torch.float32)
    mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    loss = wrapper.loss_score(targets, outputs, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(outputs.grad).all()
    assert outputs.grad[1].item() == 0.0


def test_all_no_result_auxiliary_rows_produce_differentiable_zero_losses():
    wrapper = _loss_wrapper()
    score_outputs = torch.randn(2, 1, requires_grad=True)
    ownership_outputs = torch.randn(2, 9, 3, requires_grad=True)
    score_targets = torch.full((2, 1), float("nan"))
    score_mask = torch.zeros(2, 1)
    ownership_targets = torch.zeros(2, 9, 3)
    ownership_mask = torch.zeros(2, 9)

    loss = wrapper.loss_score(score_targets, score_outputs, score_mask)
    loss = loss + wrapper.loss_ownership(ownership_targets, ownership_outputs, ownership_mask)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(score_outputs.grad).all()
    assert torch.isfinite(ownership_outputs.grad).all()
    assert torch.count_nonzero(score_outputs.grad) == 0
    assert torch.count_nonzero(ownership_outputs.grad) == 0
