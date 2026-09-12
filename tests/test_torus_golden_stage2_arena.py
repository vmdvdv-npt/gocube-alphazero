from __future__ import annotations

from dataclasses import replace

import pytest

import gocube_golden as g


# ---------------------------------------------------------------------------
# Scripted Arena bookkeeping
# ---------------------------------------------------------------------------

def test_good_vs_bad_and_color_swap_map_after_absolute_result():
    arena = g.SequentialGoldenArena(master_seed=17)
    first, second = arena.play_pair(
        pair_id="good-bad", player_A=g.GoodPlayer("GOOD"), player_B=g.BadPlayer("BAD")
    )
    assert first.black_player == "A" and first.white_player == "B"
    assert first.absolute_rule_result == g.Winner.BLACK
    assert first.mapped_result == g.MappedResult.A_WIN
    assert second.black_player == "B" and second.white_player == "A"
    assert second.absolute_rule_result == g.Winner.WHITE
    assert second.mapped_result == g.MappedResult.A_WIN
    assert arena.summary().a_wins == 2


def test_bad_vs_good_identity_swap_maps_correctly():
    arena = g.SequentialGoldenArena(master_seed=17)
    first, second = arena.play_pair(
        pair_id="bad-good", player_A=g.BadPlayer("BAD"), player_B=g.GoodPlayer("GOOD")
    )
    assert first.absolute_rule_result == g.Winner.WHITE
    assert first.mapped_result == g.MappedResult.B_WIN
    assert second.absolute_rule_result == g.Winner.BLACK
    assert second.mapped_result == g.MappedResult.B_WIN


def test_absolute_result_does_not_depend_on_model_name():
    arena1 = g.SequentialGoldenArena()
    r1 = arena1.play_game(
        game_id="names-1", pair_id="p1",
        player_A=g.GoodPlayer("model-z"), player_B=g.BadPlayer("model-a"), black_player="A",
    )
    arena2 = g.SequentialGoldenArena()
    r2 = arena2.play_game(
        game_id="names-2", pair_id="p2",
        player_A=g.GoodPlayer("renamed"), player_B=g.BadPlayer("other"), black_player="A",
    )
    assert r1.absolute_rule_result == r2.absolute_rule_result == g.Winner.BLACK
    assert r1.margin_black == r2.margin_black


def test_corrupted_mapping_is_detected_fail_closed():
    record = g.SequentialGoldenArena().play_game(
        game_id="map", pair_id="p",
        player_A=g.GoodPlayer("A"), player_B=g.BadPlayer("B"), black_player="A",
    )
    corrupted = replace(record, mapped_result=g.MappedResult.B_WIN)
    with pytest.raises(ValueError, match="mapping"):
        g.validate_game_record(corrupted)


def test_corrupted_color_assignment_is_detected_fail_closed():
    record = g.SequentialGoldenArena().play_game(
        game_id="color", pair_id="p",
        player_A=g.GoodPlayer("A"), player_B=g.BadPlayer("B"), black_player="A",
    )
    corrupted = replace(record, black_player="B", white_player="A")
    with pytest.raises(ValueError):
        g.validate_game_record(corrupted)


def test_corrupted_expected_score_is_detected_fail_closed():
    record = g.SequentialGoldenArena().play_game(
        game_id="score", pair_id="p",
        player_A=g.GoodPlayer("A"), player_B=g.BadPlayer("B"), black_player="A",
    )
    corrupted = replace(record, margin_black=record.margin_black + 1.0)
    with pytest.raises(ValueError, match="margin"):
        g.validate_game_record(corrupted)


def test_duplicate_game_id_is_rejected_by_session_and_summary():
    arena = g.SequentialGoldenArena()
    record = arena.play_game(
        game_id="dup", pair_id="p",
        player_A=g.BadPlayer("A"), player_B=g.BadPlayer("B"), black_player="A",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        arena.play_game(
            game_id="dup", pair_id="p2",
            player_A=g.BadPlayer("A"), player_B=g.BadPlayer("B"), black_player="B",
        )
    with pytest.raises(ValueError, match="Duplicate"):
        g.recompute_summary((record, record))


class IllegalPlayer:
    player_id = "ILLEGAL"
    is_search_player = False
    def select_action(self, state, context):
        return 999


class ExplodingPlayer:
    player_id = "BOOM"
    is_search_player = False
    def select_action(self, state, context):
        raise RuntimeError("deliberate player failure")


class IllegalSearchPlayer:
    player_id = "SEARCH-BUG"
    is_search_player = True
    def select_action(self, state, context):
        return 999


def test_illegal_player_action_is_error_not_defeat_or_draw():
    record = g.SequentialGoldenArena().play_game(
        game_id="illegal", pair_id="p",
        player_A=IllegalPlayer(), player_B=g.BadPlayer("B"), black_player="A",
    )
    assert record.termination_reason == g.TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION
    assert record.absolute_rule_result is None
    assert record.mapped_result is None


def test_player_exception_is_typed_error_not_game_result():
    record = g.SequentialGoldenArena().play_game(
        game_id="exception", pair_id="p",
        player_A=ExplodingPlayer(), player_B=g.BadPlayer("B"), black_player="A",
    )
    assert record.termination_reason == g.TerminationReason.ERROR_PLAYER_EXCEPTION
    assert record.absolute_rule_result is None
    assert record.mapped_result is None


def test_search_illegal_action_is_error_search_not_loss():
    record = g.SequentialGoldenArena().play_game(
        game_id="search-illegal", pair_id="p",
        player_A=IllegalSearchPlayer(), player_B=g.BadPlayer("B"), black_player="A",
    )
    assert record.termination_reason == g.TerminationReason.ERROR_SEARCH
    assert record.absolute_rule_result is None
    assert record.mapped_result is None


def test_deterministic_script_repeats_trace_and_seeds():
    kwargs = dict(
        game_id="same", pair_id="pair",
        player_A=g.TracePlayer((0, g.PASS), "A"),
        player_B=g.TracePlayer((g.PASS,), "B"),
        black_player="A",
    )
    r1 = g.SequentialGoldenArena(master_seed=99).play_game(**kwargs)
    r2 = g.SequentialGoldenArena(master_seed=99).play_game(**kwargs)
    assert r1.action_trace == r2.action_trace
    assert (r1.seed_game, r1.seed_A, r1.seed_B) == (r2.seed_game, r2.seed_A, r2.seed_B)


def test_A_equals_A_slot_swap_has_no_artificial_model_advantage():
    same = g.BadPlayer("SAME")
    arena = g.SequentialGoldenArena(master_seed=1)
    first, second = arena.play_pair(pair_id="same-model", player_A=same, player_B=same)
    assert first.mapped_result == g.MappedResult.B_WIN
    assert second.mapped_result == g.MappedResult.A_WIN
    summary = arena.summary()
    assert summary.a_wins == summary.b_wins == 1


def test_draw_mapping_uses_explicit_research_komi_zero_only():
    state = g.initial_state(komi=0.0)
    arena = g.SequentialGoldenArena()
    record = arena.play_game(
        game_id="draw", pair_id="draw",
        player_A=g.BadPlayer("A"), player_B=g.BadPlayer("B"), black_player="A",
        start_state=state, start_trace=(), allow_research_komi=True,
    )
    assert record.absolute_rule_result == g.Winner.DRAW
    assert record.mapped_result == g.MappedResult.DRAW
    assert record.margin_black == 0.0


def test_forbidden_legacy_komi_75_remains_fail_closed():
    with pytest.raises(g.LegacyKomiError):
        g.initial_state(komi=7.5)


def test_watchdog_checks_double_pass_before_move_500_truncation():
    terminal = g.apply_action(g.initial_state(), g.PASS).after
    terminal = g.apply_action(terminal, g.PASS).after
    assert g.post_action_termination(terminal, 500) == g.TerminationReason.DOUBLE_PASS
    assert g.post_action_termination(g.initial_state(), 500) == g.TerminationReason.TRUNCATED_MOVE_LIMIT


def test_technical_termination_cannot_be_corrupted_into_wdl():
    record = g.SequentialGoldenArena().play_game(
        game_id="tech", pair_id="p",
        player_A=IllegalPlayer(), player_B=g.BadPlayer("B"), black_player="A",
    )
    corrupted = replace(
        record,
        absolute_rule_result=g.Winner.WHITE,
        mapped_result=g.MappedResult.B_WIN,
    )
    with pytest.raises(ValueError, match="never contain"):
        g.validate_game_record(corrupted)


def test_summary_is_recomputed_from_raw_records_not_mutable_counters():
    arena = g.SequentialGoldenArena()
    arena.play_pair(pair_id="p", player_A=g.GoodPlayer("A"), player_B=g.BadPlayer("B"))
    s1 = arena.summary()
    s2 = g.recompute_summary(arena.records)
    assert s1 == s2
    assert s2.a_wins == 2 and s2.games == 2 and s2.rule_results == 2


def test_paired_games_share_start_history_but_not_game_id_or_colors():
    # Legal noninitial live start: Black 0, White PASS, then pair begins with Black.
    start = g.initial_state()
    trace = (0, g.PASS)
    for action in trace:
        start = g.apply_action(start, action).after
    arena = g.SequentialGoldenArena()
    one, two = arena.play_pair(
        pair_id="paired-start", player_A=g.BadPlayer("A"), player_B=g.BadPlayer("B"),
        start_state=start, start_trace=trace,
    )
    assert one.game_id != two.game_id
    assert one.start_history == two.start_history == start.superko_history
    assert one.start_state_key == two.start_state_key == start.state_key
    assert one.black_player == "A" and two.black_player == "B"


def test_checkpoint_metadata_cannot_override_arena_settings():
    with pytest.raises(ValueError, match="not allowed"):
        g.reject_checkpoint_arena_overrides({"numMCTSSims": 999})
    with pytest.raises(ValueError, match="not allowed"):
        g.reject_checkpoint_arena_overrides({"arena": {"cpuct": 9.0}})
    g.reject_checkpoint_arena_overrides({"model_hash": "abc", "observation_schema": "future-v1"})
