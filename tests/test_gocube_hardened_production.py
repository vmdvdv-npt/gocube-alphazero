from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import torch

from alphazero.envs.gocube.atomic_io import (
    RECOVERY_CONTRACT,
    REPLAY_TENSOR_SUFFIXES,
    atomic_torch_save,
    find_last_valid_contiguous_checkpoint,
    load_replay_marker,
    replay_marker_path,
    write_replay_marker,
)
from alphazero.envs.gocube.exploration_contract import (
    KATAGO_PINNED_EXPLORATION_CONTRACT,
    KATAGO_PINNED_EXPLORATION_DEFAULTS,
    apply_lcb_play_selection,
    chosen_move_temperature,
    kata_value_child_weights,
    retrospectively_reduce_root_weights,
    student_t3_cdf,
)
from alphazero.envs.gocube.hardened_train import (
    HardenedKataGoSearchCoach,
    build_hardened_training_args,
)
from alphazero.envs.gocube.katago_train import parse_args
from alphazero.utils import const_temp_scaling, get_iter_file


def test_hardened_production_defaults_pin_move_value_lcb_and_komi_contract():
    game_cls, args = build_hardened_training_args(parse_args([]))
    defaults = KATAGO_PINNED_EXPLORATION_DEFAULTS

    assert KATAGO_PINNED_EXPLORATION_CONTRACT == "katago-pinned-exploration-v2"
    assert args.gocube_recovery_contract == RECOVERY_CONTRACT
    assert args.gocube_komi == 0.5
    assert args.numMCTSSims == 50
    assert args.numFastSims == 20
    assert args.gocube_chosen_move_temperature_early == 0.75
    assert args.gocube_chosen_move_temperature == 0.15
    assert args.gocube_chosen_move_temperature_halflife == 19.0
    assert args.gocube_value_weight_exponent == 0.5
    assert args.gocube_use_lcb_for_selection is True
    assert args.gocube_lcb_stdevs == 5.0
    assert args.gocube_min_visit_prop_for_lcb == 0.15
    assert args.gocube_chosen_move_subtract == defaults["chosen_move_subtract"]
    assert args.gocube_chosen_move_prune == defaults["chosen_move_prune"]
    assert args.temp_scaling_fn is const_temp_scaling
    assert game_cls.logical_topology().point_count == 96


def test_chosen_move_temperature_uses_logical_area_scaled_halflife():
    point_count = 96
    early = 0.75
    late = 0.15
    halflife = 19.0

    assert chosen_move_temperature(
        0,
        point_count,
        early_temperature=early,
        temperature=late,
        halflife=halflife,
    ) == early

    one_scaled_halflife_turn = halflife * math.sqrt(point_count) / 19.0
    value = chosen_move_temperature(
        one_scaled_halflife_turn,
        point_count,
        early_temperature=early,
        temperature=late,
        halflife=halflife,
    )
    assert math.isclose(value, late + (early - late) * 0.5, rel_tol=1e-12)

    late_value = chosen_move_temperature(
        100_000,
        point_count,
        early_temperature=early,
        temperature=late,
        halflife=halflife,
    )
    assert math.isclose(late_value, late, rel_tol=0.0, abs_tol=1e-12)


def test_value_weight_distribution_is_pinned_student_t_with_three_degrees_of_freedom():
    assert student_t3_cdf(0.0) == 0.5
    assert math.isclose(student_t3_cdf(2.0) + student_t3_cdf(-2.0), 1.0, rel_tol=1e-12)
    # Heavier tails than a standard Gaussian: Phi(2) is about 0.97725.
    assert student_t3_cdf(2.0) < 0.97


def test_value_weight_exponent_downweights_bad_children_without_changing_total_weight():
    raw = np.array([10.0, 10.0], dtype=np.float64)
    adjusted = kata_value_child_weights(
        raw,
        np.array([0.8, -0.8], dtype=np.float64),
        exponent=0.5,
    )
    assert math.isclose(float(adjusted.sum()), float(raw.sum()), rel_tol=1e-12)
    assert adjusted[0] > raw[0]
    assert adjusted[1] < raw[1]


def test_noisy_root_value_aggregation_applies_pinned_chosen_move_prune_before_normalizing():
    adjusted = kata_value_child_weights(
        np.array([64.0, 0.5], dtype=np.float64),
        np.array([0.2, -0.5], dtype=np.float64),
        exponent=0.5,
        prune=1.0,
    )
    assert adjusted[1] == 0.0
    assert math.isclose(float(adjusted.sum()), 64.5, rel_tol=1e-12)


def test_retrospective_reduction_ceils_non_best_weight_like_pinned_searchresults():
    reduced = retrospectively_reduce_root_weights(
        np.array([10.25, 2.20], dtype=np.float64),
        np.array([0.8, 0.2], dtype=np.float64),
        np.array([0.4, 0.4], dtype=np.float64),
        root_player=1,
        explore_scaling=2.0,
        edge_visits=np.array([10.0, 2.0], dtype=np.float64),
    )
    assert reduced[0] == 10.25
    assert reduced[1] == math.ceil(reduced[1])


def test_lcb_prefers_lower_uncertainty_child_and_increases_its_selection_weight():
    result, lcbs, radii, best_idx = apply_lcb_play_selection(
        np.array([20.0, 20.0], dtype=np.float64),
        np.array([20.0, 20.0], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.10, 0.20], dtype=np.float64),
        np.array([0.0101, 0.29], dtype=np.float64),
        np.array([20.0, 20.0], dtype=np.float64),
        np.array([20.0, 20.0], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        root_player=1,
        utility_range_radius=1.30,
        lcb_stdevs=5.0,
        min_visit_prop_for_lcb=0.15,
    )
    assert best_idx == 0
    assert lcbs[0] > lcbs[1]
    assert radii[0] < radii[1]
    assert result[0] > 20.0
    assert result[1] == 20.0


def _valid_checkpoint(path):
    torch.save({"state_dict": {"weight": torch.tensor([1.0])}}, path)


def test_resume_stops_at_first_missing_or_corrupt_checkpoint_and_ignores_tail(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    _valid_checkpoint(folder / "iteration-0000.pkl")
    _valid_checkpoint(folder / "iteration-0001.pkl")
    (folder / "iteration-0002.pkl").write_bytes(b"truncated")
    _valid_checkpoint(folder / "iteration-0003.pkl")

    last_valid, ignored = find_last_valid_contiguous_checkpoint(folder)
    assert last_valid == 1
    assert ignored == [2, 3]


def test_atomic_torch_save_publishes_a_complete_readable_file(tmp_path):
    target = tmp_path / "checkpoint.pkl"
    atomic_torch_save({"state_dict": {"x": torch.arange(4)}}, target)
    payload = torch.load(target, map_location="cpu")
    assert torch.equal(payload["state_dict"]["x"], torch.arange(4))
    assert not list(tmp_path.glob(".checkpoint.pkl.torch-*"))


class _DummyGame:
    @staticmethod
    def observation_size():
        return (2,)


def _replay_base(tmp_path, run_name, iteration):
    folder = tmp_path / run_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / get_iter_file(iteration).replace(".pkl", "")


def _write_complete_replay(base, rows=3):
    tensors = (
        torch.zeros(rows, 2),
        torch.zeros(rows, 4),
        torch.zeros(rows, 3),
        torch.zeros(rows, 1),
        torch.zeros(rows, 1, 3),
        torch.zeros(rows, 1),
    )
    for suffix, tensor in zip(REPLAY_TENSOR_SUFFIXES, tensors):
        torch.save(tensor, os.fspath(base) + suffix)
    write_replay_marker(os.fspath(base), iteration=1, row_count=rows)


def _replay_loader_coach(tmp_path, run_name):
    coach = object.__new__(HardenedKataGoSearchCoach)
    coach.args = SimpleNamespace(
        data=os.fspath(tmp_path),
        run_name=run_name,
        minTrainHistoryWindow=4,
        trainHistoryIncrementIters=2,
        maxTrainHistoryWindow=20,
    )
    coach.game_cls = _DummyGame
    return coach


def test_replay_loader_requires_completion_marker_and_all_six_tensors(tmp_path):
    run_name = "atomic-replay"
    base = _replay_base(tmp_path, run_name, 1)
    _write_complete_replay(base)
    marker = load_replay_marker(os.fspath(base))
    assert marker["row_count"] == 3

    coach = _replay_loader_coach(tmp_path, run_name)
    datasets, loaded = coach._load_replay_datasets(1)
    assert len(datasets) == 1
    assert loaded == {1: 3}

    os.unlink(replay_marker_path(os.fspath(base)))
    datasets, loaded = coach._load_replay_datasets(1)
    assert datasets == []
    assert loaded == {}

    write_replay_marker(os.fspath(base), iteration=1, row_count=3)
    os.unlink(os.fspath(base) + REPLAY_TENSOR_SUFFIXES[-1])
    datasets, loaded = coach._load_replay_datasets(1)
    assert datasets == []
    assert loaded == {}
