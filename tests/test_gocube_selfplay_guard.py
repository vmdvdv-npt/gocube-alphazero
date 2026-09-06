from types import SimpleNamespace

from alphazero.envs.gocube.training_guard import evaluate_selfplay_guard


def _args(**overrides):
    values = dict(
        gocube_guard_min_games=32,
        gocube_early_double_pass_warning_rate=0.01,
        gocube_early_double_pass_fatal_rate=0.05,
        gocube_cleanup2_warning_fraction=0.50,
        gocube_cleanup2_fatal_fraction=0.70,
        gocube_score_dominated_pass_fatal_rate=0.25,
        gocube_score_audit_min_positions=16,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _telemetry(**overrides):
    values = dict(
        main_double_pass_within_2=0,
        phase_main_decisions=1000,
        phase_cleanup1_decisions=100,
        phase_cleanup2_decisions=100,
        search_audited_positions=0,
        search_score_dominated_pass=0,
    )
    values.update(overrides)
    return values


def test_normal_iteration_is_valid_and_training_allowed():
    result = evaluate_selfplay_guard(_telemetry(), games=256, args=_args())
    assert result.status == "valid"
    assert result.training_allowed is True
    assert result.fatal_reasons == ()


def test_pathological_early_double_pass_iteration_is_invalid():
    result = evaluate_selfplay_guard(
        _telemetry(main_double_pass_within_2=20),
        games=32,
        args=_args(),
    )
    assert result.status == "invalid_selfplay"
    assert result.training_allowed is False
    assert any("early MAIN double-pass" in reason for reason in result.fatal_reasons)


def test_cleanup2_domination_is_invalid_after_minimum_game_count():
    result = evaluate_selfplay_guard(
        _telemetry(
            phase_main_decisions=100,
            phase_cleanup1_decisions=100,
            phase_cleanup2_decisions=800,
        ),
        games=64,
        args=_args(),
    )
    assert result.status == "invalid_selfplay"
    assert any("CLEANUP_2 decision fraction" in reason for reason in result.fatal_reasons)


def test_score_dominated_pass_guard_requires_minimum_audit_sample():
    below = evaluate_selfplay_guard(
        _telemetry(search_audited_positions=15, search_score_dominated_pass=15),
        games=256,
        args=_args(),
    )
    assert below.status == "valid"

    enough = evaluate_selfplay_guard(
        _telemetry(search_audited_positions=16, search_score_dominated_pass=4),
        games=256,
        args=_args(),
    )
    assert enough.status == "invalid_selfplay"
    assert any("score-dominated second-PASS" in reason for reason in enough.fatal_reasons)


def test_one_or_two_early_games_do_not_trigger_guard_before_minimum_games():
    result = evaluate_selfplay_guard(
        _telemetry(main_double_pass_within_2=2),
        games=2,
        args=_args(),
    )
    assert result.status == "valid"
