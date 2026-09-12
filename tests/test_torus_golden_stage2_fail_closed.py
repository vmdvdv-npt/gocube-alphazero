from __future__ import annotations

from dataclasses import replace

import pytest

import gocube_golden as g


class IllegalPlayer:
    player_id = "ILLEGAL-EVIDENCE"
    is_search_player = False

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


@pytest.mark.parametrize("simulations", [64.0, 1.5, "64"])
def test_search_settings_simulations_requires_actual_integer(simulations):
    with pytest.raises(ValueError, match="positive integer"):
        g.SearchSettings(simulations=simulations)


def test_search_settings_simulations_accepts_positive_integer():
    assert g.SearchSettings(simulations=64).simulations == 64
