import math

import numpy as np

from alphazero.envs.gocube.atomic_io import (
    RECOVERY_CONTRACT,
    atomic_torch_save,
    find_last_valid_contiguous_checkpoint,
    load_replay_marker,
    write_replay_marker,
)
from alphazero.envs.gocube.exploration_contract import (
    KATAGO_PINNED_EXPLORATION_CONTRACT,
    KATAGO_PINNED_EXPLORATION_DEFAULTS,
    apply_lcb_play_selection,
    chosen_move_temperature,
    kata_value_child_weights,
)
from alphazero.envs.gocube.production_hardening import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args


def test_chosen_move_temperature_matches_pinned_interpolate_early():
    defaults = KATAGO_PINNED_EXPLORATION_DEFAULTS
    point_count = 96
    at_start = chosen_move_temperature(
        0,
        point_count,
        early_temperature=defaults["chosen_move_temperature_early"],
        temperature=defaults["chosen_move_temperature"],
        halflife=defaults["chosen_move_temperature_halflife"],
    )
    one_scaled_halflife_turn = math.sqrt(point_count)
    after_one = chosen_move_temperature(
        one_scaled_halflife_turn,
        point_count,
        early_temperature=defaults["chosen_move_temperature_early"],
        temperature=defaults["chosen_move_temperature"],
        halflife=defaults["chosen_move_temperature_halflife"],
    )
    late = chosen_move_temperature(
        500,
        point_count,
        early_temperature=defaults["chosen_move_temperature_early"],
        temperature=defaults["chosen_move_temperature"],
        halflife=defaults["chosen_move_temperature_halflife"],
    )

    assert at_start == 0.75
    assert math.isclose(after_one, 0.45, rel_tol=1e-12)
    assert 0.15 <= late < 0.16


def test_value_weight_exponent_downweights_bad_child_and_preserves_total_weight():
    raw = np.array([20.0, 20.0, 5.0])
    utilities = np.array([0.8, -0.8, 0.1])
    adjusted = kata_value_child_weights(raw, utilities, exponent=0.5)

    assert math.isclose(float(adjusted.sum()), float(raw.sum()), rel_tol=1e-12)
    assert adjusted[0] > raw[0]
    assert adjusted[1] < raw[1]
    assert np.all(adjusted >= 0.0)


def test_lcb_uses_utility_variance_and_can_bonus_best_lcb_move():
    weights = np.array([20.0, 20.0, 12.0])
    visits = np.array([20.0, 20.0, 12.0])
    policy = np.array([0.4, 0.35, 0.25])
    means = np.array([0.60, 0.52, 0.40])
    # Same rough means, deliberately different uncertainty.
    squares = np.array([0.60**2 + 0.002, 0.52**2 + 0.08, 0.40**2 + 0.02])
    weight_sums = np.array([20.0, 20.0, 12.0])
    weight_sq_sums = np.array([20.0, 20.0, 12.0])
    ending = np.zeros(3)

    adjusted, lcbs, radii, best = apply_lcb_play_selection(
        weights,
        visits,
        policy,
        means,
        squares,
        weight_sums,
        weight_sq_sums,
        ending,
        root_player=1,
        utility_range_radius=1.3,
        lcb_stdevs=5.0,
        min_visit_prop_for_lcb=0.15,
    )

    assert best is not None
    assert np.all(np.isfinite(lcbs))
    assert np.all(radii >= 0.0)
    assert adjusted[best] >= weights[best]
    assert radii[0] < radii[1]


def test_hardened_production_args_pin_final_search_and_recovery_contract():
    game_cls, args = build_hardened_training_args(parse_args([]))
    defaults = KATAGO_PINNED_EXPLORATION_DEFAULTS

    assert args.gocube_komi == 0.5
    assert args.gocube_rules_fingerprint == game_cls.rules_fingerprint()
    assert args.gocube_katago_exploration_contract == KATAGO_PINNED_EXPLORATION_CONTRACT
    assert args.gocube_recovery_contract == RECOVERY_CONTRACT
    assert args.gocube_chosen_move_temperature_early == defaults["chosen_move_temperature_early"]
    assert args.gocube_chosen_move_temperature == defaults["chosen_move_temperature"]
    assert args.gocube_use_lcb_for_selection is True
    assert args.gocube_lcb_stdevs == 5.0
    assert args.gocube_min_visit_prop_for_lcb == 0.15
    assert args.gocube_value_weight_exponent == 0.5
    assert game_cls.logical_topology().point_count == 96


def test_contiguous_checkpoint_recovery_ignores_corrupt_and_later_files(tmp_path):
    for iteration in (0, 1):
        atomic_torch_save(
            {"state_dict": {"weight": np.array([iteration])}},
            tmp_path / f"iteration-{iteration:04d}.pkl",
        )
    (tmp_path / "iteration-0002.pkl").write_bytes(b"not-a-checkpoint")
    atomic_torch_save(
        {"state_dict": {"weight": np.array([3])}},
        tmp_path / "iteration-0003.pkl",
    )

    last, ignored = find_last_valid_contiguous_checkpoint(tmp_path)
    assert last == 1
    assert ignored == [2, 3]


def test_replay_completion_marker_is_explicit_and_versioned(tmp_path):
    base = tmp_path / "iteration-0007"
    path = write_replay_marker(str(base), iteration=7, row_count=123)
    marker = load_replay_marker(str(base))

    assert path.endswith("iteration-0007-complete.json")
    assert marker["recovery_contract"] == RECOVERY_CONTRACT
    assert marker["iteration"] == 7
    assert marker["row_count"] == 123
