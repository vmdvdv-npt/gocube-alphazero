from argparse import Namespace

import pytest

from alphazero.Coach import Coach
from alphazero.envs.gocube.katago_v3 import KATAGO_REFERENCE_COMMIT, KATAGO_RULES_VERSION
from alphazero.envs.gocube.train import GoCubeCoach, build_training_args


def cli_args(**overrides):
    values = {
        "topology": "torus", "size": 9, "workers": 2, "sims": 20,
        "games_per_iteration": 8, "iterations": 3, "train_batch_size": 1024,
        "fast_game_prob": 0.75, "endgame_sample_weight": 3,
        "inference_batch_wait_ms": 1.0, "no_arena": False,
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
    assert args.compareWithBaseline is True
    assert args.compareWithPast is True
    assert args.model_gating is True
    assert args.autoTrainSteps is True
    assert args.probFastSim == 0.75
    assert args.gocube_auxiliary_targets is True
    assert args.gocube_endgame_sample_weight == 3
    assert args.gocube_terminal_adjudicator == "gocube-katago-japanese-v3"
    assert args.gocube_observation_schema == "gocube-observation-v3"
    assert args.gocube_rules_fingerprint == game_cls.rules_fingerprint()
    assert args.gocube_katago_rules_version == KATAGO_RULES_VERSION
    assert args.gocube_katago_reference_commit == KATAGO_REFERENCE_COMMIT


def test_default_run_name_is_fresh_v3_namespace():
    _, args = build_training_args(cli_args(run_name=None))
    assert args.run_name == "gocube-torus-9-japanese75-katago-v3"


def test_smoke_mode_is_one_iteration_without_arena_comparisons():
    _, args = build_training_args(cli_args(iterations=50, smoke=True))
    assert args.numIters == 1
    assert args.compareWithBaseline is False
    assert args.compareWithPast is False
    assert args.model_gating is False
    assert args.autoTrainSteps is False
    assert args.train_steps_per_iteration == 1
    assert args.probFastSim == 0.0


def test_no_arena_advances_self_play_version_after_training_checkpoint(monkeypatch):
    saved = []
    monkeypatch.setattr(Coach, "_save_model", lambda self, model, iteration: saved.append(iteration))
    coach = object.__new__(GoCubeCoach)
    coach.args = Namespace(model_gating=False)
    coach.self_play_iter = 0
    coach._save_model(object(), 1)
    assert saved == [1]
    assert coach.self_play_iter == 1


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
        ("fast_game_prob", -0.1, "fast-game-prob must be between 0 and 1"),
        ("fast_game_prob", 1.1, "fast-game-prob must be between 0 and 1"),
        ("inference_batch_wait_ms", -0.1, "inference-batch-wait-ms must be non-negative"),
        ("endgame_sample_weight", 0, "endgame-sample-weight must be at least 1"),
    ],
)
def test_invalid_training_counts_fail_fast(field, value, message):
    with pytest.raises(ValueError, match=message):
        build_training_args(cli_args(**{field: value}))
