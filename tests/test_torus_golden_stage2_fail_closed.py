from __future__ import annotations

from dataclasses import replace

import pytest

import gocube_golden as g
import gocube_golden.arena as golden_arena


class IllegalPlayer:
    player_id = "ILLEGAL-EVIDENCE"
    is_search_player = False

    def select_action(self, state, context):
        return 999


class ExplodingPlayer:
    player_id = "EXPLODING-EVIDENCE"
    is_search_player = False

    def select_action(self, state, context):
        raise RuntimeError("deliberate player failure")


class IllegalSearchPlayer:
    player_id = "SEARCH-ILLEGAL-EVIDENCE"
    is_search_player = True

    def select_action(self, state, context):
        return 999


def _rule_record():
    return g.SequentialGoldenArena().play_game(
        game_id="evidence-rule",
        pair_id="evidence-pair",
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )


def _replay_rule_record(record):
    state = g.initial_state(komi=record.komi)
    for action in record.start_trace:
        state = g.apply_action(state, action).after
    for evidence in record.action_trace:
        assert evidence.legal
        state = g.apply_action(state, evidence.action).after
    assert state.is_terminal
    return state


def _post_terminal_corruption(
    record,
    *,
    action,
    legal,
    termination_reason=g.TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION,
):
    state = _replay_rule_record(record)
    slot = record.black_player if state.side_to_move == g.BLACK else record.white_player
    player_id = record.player_A_id if slot == "A" else record.player_B_id
    fake = replace(
        record.action_trace[-1],
        ply=len(record.action_trace) + 1,
        side_to_move=state.side_to_move.name,
        player_slot=slot,
        player_id=player_id,
        action=action,
        legal=legal,
        error=None if legal else "fake post-terminal action",
    )
    return replace(
        record,
        action_trace=record.action_trace + (fake,),
        termination_reason=termination_reason,
        absolute_rule_result=None,
        mapped_result=None,
        black_area=None,
        white_area=None,
        margin_black=None,
        error_details="fake technical termination after formal terminal",
    )


def _technical_override(record, termination_reason):
    return replace(
        record,
        termination_reason=termination_reason,
        absolute_rule_result=None,
        mapped_result=None,
        black_area=None,
        white_area=None,
        margin_black=None,
        error_details="fake technical termination after formal terminal",
    )


def test_game_record_pins_concrete_search_implementation_identity():
    record = _rule_record()
    assert record.search_implementation_id == g.SEARCH_IMPLEMENTATION_ID
    assert record.search_implementation_fingerprint == g.SEARCH_IMPLEMENTATION_FINGERPRINT
    g.validate_game_record(record)


def test_validator_rejects_tampered_search_implementation_id_and_fingerprint():
    record = _rule_record()
    with pytest.raises(ValueError, match="implementation id drift"):
        g.validate_game_record(replace(record, search_implementation_id="other-search"))
    with pytest.raises(ValueError, match="implementation fingerprint drift"):
        g.validate_game_record(
            replace(record, search_implementation_fingerprint="sha256:tampered")
        )


def test_validator_rejects_tampered_rules_fingerprint():
    record = _rule_record()
    with pytest.raises(ValueError, match="rules fingerprint"):
        g.validate_game_record(replace(record, rules_fingerprint="sha256:tampered"))


def test_validator_rejects_tampered_search_settings():
    record = _rule_record()
    tampered = tuple(
        (key, 999 if key == "simulations" else value)
        for key, value in record.search_settings
    )
    with pytest.raises(ValueError, match="search settings drift"):
        g.validate_game_record(replace(record, search_settings=tampered))


def test_validator_rejects_any_trace_tail_after_illegal_action():
    record = g.SequentialGoldenArena().play_game(
        game_id="illegal-tail",
        pair_id="illegal-tail-pair",
        player_A=IllegalPlayer(),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    assert record.termination_reason == g.TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION
    assert len(record.action_trace) == 1
    corrupted = replace(
        record,
        action_trace=record.action_trace + (record.action_trace[-1],),
    )
    with pytest.raises(ValueError, match="continues after illegal move"):
        g.validate_game_record(corrupted)


def test_validator_rejects_illegal_evidence_after_formal_terminal():
    record = _rule_record()
    assert record.termination_reason == g.TerminationReason.DOUBLE_PASS
    corrupted = _post_terminal_corruption(record, action=999, legal=False)

    with pytest.raises(ValueError, match="ActionEvidence exists after formal Golden terminal"):
        g.validate_game_record(corrupted)


def test_validator_rejects_legal_marked_evidence_after_formal_terminal_at_boundary():
    record = _rule_record()
    corrupted = _post_terminal_corruption(record, action=999, legal=True)

    with pytest.raises(ValueError, match="ActionEvidence exists after formal Golden terminal"):
        g.validate_game_record(corrupted)


def test_validator_rejects_third_pass_after_formal_double_pass_terminal():
    record = _rule_record()
    assert record.action_trace[-2].action == g.PASS
    assert record.action_trace[-1].action == g.PASS
    corrupted = _post_terminal_corruption(record, action=g.PASS, legal=False)

    with pytest.raises(ValueError, match="ActionEvidence exists after formal Golden terminal"):
        g.validate_game_record(corrupted)


@pytest.mark.parametrize(
    "termination_reason",
    [
        g.TerminationReason.ERROR_SEARCH,
        g.TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION,
    ],
)
def test_technical_termination_cannot_override_replayed_formal_terminal(
    termination_reason,
):
    record = _rule_record()
    corrupted = _technical_override(record, termination_reason)

    with pytest.raises(ValueError, match="Formal Golden terminal requires DOUBLE_PASS"):
        g.validate_game_record(corrupted)


def test_legitimate_preterminal_technical_failures_remain_valid(monkeypatch):
    illegal = g.SequentialGoldenArena().play_game(
        game_id="preterminal-illegal",
        pair_id="preterminal-illegal-pair",
        player_A=IllegalPlayer(),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    assert illegal.termination_reason == g.TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION
    g.validate_game_record(illegal)

    player_exception = g.SequentialGoldenArena().play_game(
        game_id="preterminal-exception",
        pair_id="preterminal-exception-pair",
        player_A=ExplodingPlayer(),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    assert player_exception.termination_reason == g.TerminationReason.ERROR_PLAYER_EXCEPTION
    g.validate_game_record(player_exception)

    search_error = g.SequentialGoldenArena().play_game(
        game_id="preterminal-search",
        pair_id="preterminal-search-pair",
        player_A=IllegalSearchPlayer(),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    assert search_error.termination_reason == g.TerminationReason.ERROR_SEARCH
    g.validate_game_record(search_error)

    monkeypatch.setattr(golden_arena, "GOLDEN_MOVE_LIMIT", 1)
    truncated = g.SequentialGoldenArena().play_game(
        game_id="preterminal-truncated",
        pair_id="preterminal-truncated-pair",
        player_A=g.TracePlayer((0,), "A"),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    assert truncated.termination_reason == g.TerminationReason.TRUNCATED_MOVE_LIMIT
    g.validate_game_record(truncated)


@pytest.mark.parametrize("simulations", [64.0, 1.5, "64"])
def test_search_settings_simulations_requires_actual_integer(simulations):
    with pytest.raises(ValueError, match="positive integer"):
        g.SearchSettings(simulations=simulations)


def test_search_settings_simulations_accepts_positive_integer():
    assert g.SearchSettings(simulations=64).simulations == 64
