import sys
from argparse import Namespace

import pytest

from alphazero.Coach import Coach
from alphazero.envs.gocube.katago_v3 import KATAGO_REFERENCE_COMMIT, KATAGO_RULES_VERSION
from alphazero.envs.gocube.train import (
    GoCubeCoach,
    build_training_args,
    parse_args,
    print_training_configuration,
)
from alphazero.search_contract import GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE, GOCUBE_SEARCH_CONTRACT


def cli_args(**overrides):
    values = {
        "topology": "torus", "size": 9, "workers": 2, "sims": 20,
        "arena_sims": 100,
        "games_per_iteration": 8, "iterations": 3, "train_batch_size": 1024,
        "train_steps_per_iteration": None,
        "fast_game_prob": 0.75, "endgame_sample_weight": 1,
        "inference_batch_wait_ms": 1.0, "no_arena": False,
        "model_gating": False,
        "smoke": False, "run_name": "gocube-train-test",
    }
    values.update(overrides)
    return Namespace(**values)


def test_v3_contract_and_training_controls_are_forwarded():
    game_cls, args = build_training_args(cli_args())
    assert game_cls.RULESET == "japanese"
    assert game_cls.TERMINAL_ADJUDICATOR_ID == "gocube-katago-japanese-v3"
    assert game_cls.OBSERVATION_SCHEMA == "gocube-observation-v3"
    assert args.numIters == 3
    assert args.gamesPerIteration == 8
    assert args.process_batch_size == 4
    assert args.train_batch_size == 1024
    assert args.numMCTSSims == 20
    assert args.arenaMCTSSims == 100
    assert args.arenaTemp == 0.0
    assert args.arenaBatched is False
    assert args.compareWithBaseline is True
    assert args.compareWithPast is True
    assert args.model_gating is False
    assert args.autoTrainSteps is True
    assert args.train_steps_per_iteration is None
    assert args.probFastSim == 0.75
    assert args.gocube_auxiliary_targets is True
    assert args.gocube_endgame_sample_weight == 1
    assert args.gocube_main_after_pass_weight == 1
    assert args.gocube_cleanup1_weight == 1
    assert args.gocube_cleanup2_weight == 1
    assert args.search_utility_mode == GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE
    assert args.gocube_search_contract == GOCUBE_SEARCH_CONTRACT
    assert args.gocube_win_loss_utility_factor == pytest.approx(1.0)
    assert args.gocube_static_score_utility_factor == pytest.approx(0.0)
    assert args.gocube_dynamic_score_utility_factor == pytest.approx(0.4)
    assert args.gocube_dynamic_score_center_zero_weight == pytest.approx(0.25)
    assert args.gocube_dynamic_score_center_scale == pytest.approx(0.5)
    assert args.gocube_root_ending_bonus_points == pytest.approx(0.5)
    assert args.gocube_score_improvement_threshold_points == pytest.approx(1.0)
    assert args.gocube_win_probability_tolerance == pytest.approx(0.005)
    assert args.gocube_terminal_adjudicator == "gocube-katago-japanese-v3"
    assert args.gocube_observation_schema == "gocube-observation-v3"
    assert args.gocube_rules_fingerprint == game_cls.rules_fingerprint()
    assert args.gocube_katago_rules_version == KATAGO_RULES_VERSION
    assert args.gocube_katago_reference_commit == KATAGO_REFERENCE_COMMIT
    assert args.gocube_game_id_registry == "data/.gocube-game-ids"
    assert args.run_name not in args.gocube_game_id_registry


def test_v3_cli_defaults_are_conservative_pilot_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    cli = parse_args()
    game_cls, args = build_training_args(cli)
    assert game_cls.topology_kind() == "torus"
    assert args.numMCTSSims == 100
    assert args.arenaMCTSSims == 100
    assert args.gamesPerIteration == 256
    assert args.train_batch_size == 256
    assert args.probFastSim == 0.25
    assert args.gocube_endgame_sample_weight == 1
    assert args.gocube_main_after_pass_weight == 1
    assert args.gocube_cleanup1_weight == 1
    assert args.gocube_cleanup2_weight == 1
    assert args.gocube_guard_min_games == 32
    assert args.gocube_early_double_pass_warning_rate == pytest.approx(0.01)
    assert args.gocube_early_double_pass_fatal_rate == pytest.approx(0.05)
    assert args.gocube_cleanup2_warning_fraction == pytest.approx(0.50)
    assert args.gocube_cleanup2_fatal_fraction == pytest.approx(0.70)
    assert args.gocube_score_dominated_pass_fatal_rate == pytest.approx(0.25)
    assert args.gocube_score_audit_min_positions == 16
    assert args.arenaTemp == 0.0
    assert args.compareWithBaseline is True
    assert args.compareWithPast is True
    assert args.model_gating is False
    assert args.autoTrainSteps is True
    assert args.train_steps_per_iteration is None


def test_default_run_name_uses_new_pilot_methodology_namespace():
    _, args = build_training_args(cli_args(run_name=None))
    assert args.run_name == "gocube-torus-9-japanese75-katago-v3-pilot"


def test_smoke_mode_is_one_iteration_without_arena_comparisons():
    _, args = build_training_args(cli_args(iterations=50, smoke=True))
    assert args.numIters == 1
    assert args.compareWithBaseline is False
    assert args.compareWithPast is False
    assert args.model_gating is False
    assert args.autoTrainSteps is False
    assert args.train_steps_per_iteration == 1
    assert args.probFastSim == 0.0


def test_fixed_train_steps_override_disables_auto_mode():
    _, args = build_training_args(cli_args(train_steps_per_iteration=17))
    assert args.autoTrainSteps is False
    assert args.train_steps_per_iteration == 17


def test_arena_on_gating_off_is_default_pilot_configuration():
    _, args = build_training_args(cli_args())
    assert args.compareWithBaseline is True
    assert args.compareWithPast is True
    assert args.model_gating is False


def test_arena_on_gating_on_is_explicit_opt_in():
    _, args = build_training_args(cli_args(model_gating=True))
    assert args.compareWithBaseline is True
    assert args.compareWithPast is True
    assert args.model_gating is True


def test_arena_off_gating_off_is_supported():
    _, args = build_training_args(cli_args(no_arena=True))
    assert args.compareWithBaseline is False
    assert args.compareWithPast is False
    assert args.model_gating is False


def test_arena_off_gating_on_fails_fast():
    with pytest.raises(ValueError, match="model gating requires arena evaluation"):
        build_training_args(cli_args(no_arena=True, model_gating=True))


def test_smoke_with_model_gating_fails_fast():
    with pytest.raises(ValueError, match="model gating requires arena evaluation"):
        build_training_args(cli_args(smoke=True, model_gating=True))


def test_blanket_endgame_oversampling_is_rejected():
    with pytest.raises(ValueError, match="blanket endgame oversampling is disabled"):
        build_training_args(cli_args(endgame_sample_weight=3))


def test_phase_weights_are_independently_configurable():
    _, args = build_training_args(cli_args(
        main_after_pass_weight=2,
        cleanup1_weight=3,
        cleanup2_weight=4,
    ))
    assert args.gocube_main_after_pass_weight == 2
    assert args.gocube_cleanup1_weight == 3
    assert args.gocube_cleanup2_weight == 4


def test_cli_exposes_arena_budget_model_gating_and_fixed_steps(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--arena-sims", "77", "--model-gating", "--train-steps-per-iteration", "17"],
    )
    cli = parse_args()
    assert cli.arena_sims == 77
    assert cli.model_gating is True
    assert cli.no_arena is False
    assert cli.train_steps_per_iteration == 17


def test_training_configuration_reports_training_arena_gating_and_search(capsys):
    _, args = build_training_args(cli_args())
    print_training_configuration(args)
    output = capsys.readouterr().out
    assert "Search:" in output
    assert GOCUBE_SEARCH_CONTRACT in output
    assert "Self-play:" in output
    assert "regular sims = 20" in output
    assert "fast sims = 20" in output
    assert "fast probability = 75%" in output
    assert "Training:" in output
    assert "games/iteration = 8" in output
    assert "batch size = 1024" in output
    assert "step mode = auto" in output
    assert "endgame weight = 1" in output
    assert "phase weights = main-after-pass 1, cleanup1 1, cleanup2 1" in output
    assert "Arena: fixed 100 sims, fast OFF, root noise OFF, root temp OFF, action temp 0" in output
    assert "color-balanced scheduling" in output
    assert "Model gating: OFF" in output


def test_no_arena_advances_self_play_version_after_training_checkpoint(monkeypatch):
    saved = []
    monkeypatch.setattr(Coach, "_save_model", lambda self, model, iteration: saved.append(iteration))
    coach = object.__new__(GoCubeCoach)
    coach.args = Namespace(model_gating=False)
    coach.self_play_iter = 0
    coach._save_model(object(), 1)
    assert saved == [1]
    assert coach.self_play_iter == 1


def test_gating_off_arena_compares_candidate_to_previous_checkpoint(monkeypatch):
    seen_past_iterations = []
    monkeypatch.setattr(Coach, "_save_model", lambda self, model, iteration: None)
    monkeypatch.setattr(
        Coach,
        "compareToPast",
        lambda self, model_iter: seen_past_iterations.append((model_iter, self.self_play_iter)),
    )
    coach = object.__new__(GoCubeCoach)
    coach.args = Namespace(model_gating=False)
    coach.self_play_iter = 4

    coach._save_model(object(), 5)
    assert coach.self_play_iter == 5

    coach.compareToPast(5)

    assert seen_past_iterations == [(5, 4)]
    assert coach.self_play_iter == 5


def test_gating_keeps_self_play_version_owned_by_arena(monkeypatch):
    monkeypatch.setattr(Coach, "_save_model", lambda self, model, iteration: None)
    coach = object.__new__(GoCubeCoach)
    coach.args = Namespace(model_gating=True)
    coach.self_play_iter = 7
    coach._save_model(object(), 8)
    assert coach.self_play_iter == 7


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workers", 0, "workers must be at least 1"),
        ("games_per_iteration", 0, "games-per-iteration must be at least 1"),
        ("iterations", 0, "iterations must be at least 1"),
        ("train_batch_size", 0, "train-batch-size must be at least 1"),
        ("train_steps_per_iteration", 0, "train-steps-per-iteration must be at least 1"),
        ("sims", 0, "sims must be at least 1"),
        ("arena_sims", 0, "arena-sims must be at least 1"),
        ("fast_game_prob", -0.1, "fast-game-prob must be between 0 and 1"),
        ("fast_game_prob", 1.1, "fast-game-prob must be between 0 and 1"),
        ("inference_batch_wait_ms", -0.1, "inference-batch-wait-ms must be non-negative"),
        ("endgame_sample_weight", 0, "endgame-sample-weight must be at least 1"),
        ("main_after_pass_weight", 0, "phase sample weights must be at least 1"),
    ],
)
def test_invalid_training_counts_fail_fast(field, value, message):
    with pytest.raises(ValueError, match=message):
        build_training_args(cli_args(**{field: value}))
