from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import gocube_golden as g
from gocube_golden.neural import AuxiliaryGoldenGraphNet, GoldenGraphNetV1
from gocube_golden.stage4 import (
    first_move_statistics,
    state_from_identity,
    torus_automorphism_permutations,
    _transform_state_identity,
)
from gocube_golden.training import (
    GOLDEN_AUXILIARY_TARGET_SOURCE,
    GoldenTrainingSample,
    ownership_target,
    score_target,
    state_identity,
    z_target,
)


def _terminal(stones, *, side=g.BLACK):
    live = g.research_state_from_stones(stones, side_to_move=side, komi=0.5)
    return g.apply_action(g.apply_action(live, g.PASS).after, g.PASS).after


def _sample():
    live = g.initial_state()
    final = _terminal(live.stones)
    observation = g.build_observation(live)
    return GoldenTrainingSample(
        run_id="aux-test",
        game_id="game-00",
        ply=1,
        state=state_identity(live),
        side_to_move="BLACK",
        observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
        legal_action_mask=tuple([True] * 26),
        root_visits=tuple([1] * 26),
        pi=tuple(1.0 / 26.0 for _ in range(26)),
        z=z_target("WHITE", g.BLACK),
        model_hash="sha256:" + "0" * 64,
        selfplay_contract_fingerprint=g.DEFAULT_SELFPLAY_CONTRACT.fingerprint,
        ownership_target=ownership_target(final, g.BLACK),
        score_target=score_target(final, g.BLACK),
        auxiliary_target_source=GOLDEN_AUXILIARY_TARGET_SOURCE,
    )


def test_ownership_is_exact_referee_output_and_inverts_perspective():
    stones = [g.EMPTY] * 25
    stones[0] = g.BLACK
    stones[1] = g.WHITE
    final = _terminal(stones, side=g.BLACK)
    black = ownership_target(final, g.BLACK)
    white = ownership_target(final, g.WHITE)
    assert black[0] == 0 and black[1] == 1
    assert white[0] == 1 and white[1] == 0
    assert set(black[2:]) == {2}


def test_score_target_sign_color_swap_and_komi_are_pinned():
    final = _terminal([g.EMPTY] * 25, side=g.BLACK)
    assert score_target(final, g.BLACK) == pytest.approx(-0.5)
    assert score_target(final, g.WHITE) == pytest.approx(0.5)
    assert z_target("WHITE", g.BLACK) == (0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="komi"):
        score_target(g.research_state_from_stones([g.EMPTY] * 25, komi=7.5), g.BLACK)


def test_auxiliary_targets_transform_with_full_state_symmetry():
    stones = [g.EMPTY] * 25
    stones[0] = g.BLACK
    stones[6] = g.WHITE
    original = _terminal(stones)
    permutation = torus_automorphism_permutations()[37]
    transformed_identity = _transform_state_identity(state_identity(original), permutation)
    transformed = state_from_identity(transformed_identity)
    source = ownership_target(original, g.BLACK)
    target = ownership_target(transformed, g.BLACK)
    expected = [0] * 25
    for old, new in enumerate(permutation):
        expected[new] = source[old]
    assert target == tuple(expected)
    assert score_target(transformed, g.BLACK) == pytest.approx(score_target(original, g.BLACK))


def test_replay_sample_round_trip_preserves_auxiliary_targets():
    sample = _sample()
    restored = GoldenTrainingSample.from_dict(sample.to_dict())
    restored.validate()
    assert restored.ownership_target == sample.ownership_target
    assert restored.score_target == sample.score_target
    assert restored.to_dict() == sample.to_dict()


def test_shared_m0_policy_wdl_parameters_are_bit_identical_and_baseline_outputs_unchanged():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        control = GoldenGraphNetV1()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        auxiliary = AuxiliaryGoldenGraphNet(variant="wdl+ownership")
    for name, parameter in control.state_dict().items():
        assert torch.equal(parameter, auxiliary.state_dict()[name])
    observation = g.build_observation(g.initial_state()).unsqueeze(0)
    control_outputs = control(observation)
    auxiliary_outputs = auxiliary(observation)
    assert torch.equal(control_outputs[0], auxiliary_outputs[0])
    assert torch.equal(control_outputs[1], auxiliary_outputs[1])


def test_first_move_statistics_excludes_technical_games():
    state = state_identity(g.initial_state())
    valid = SimpleNamespace(
        game_id="valid",
        technical_termination=None,
        start_state=state,
        final_action_trace=(g.PASS, g.PASS),
    )
    technical = SimpleNamespace(game_id="technical", technical_termination="ERROR_SEARCH")
    summary = first_move_statistics((valid, technical), source="test")
    assert summary["games"] == 1
    assert summary["technical_games"] == 1
    assert summary["white_wins"] == 1
    assert summary["technical_excluded_from_wdl_and_statistics"] is True
