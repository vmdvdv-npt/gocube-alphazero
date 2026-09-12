from __future__ import annotations

import json

import pytest

import gocube_golden as g


CLEAN_CODE = g.CodeIdentity("a" * 40, "b" * 40, True)


class IllegalPlayer:
    player_id = "ILLEGAL-SERIALIZATION"
    is_search_player = False

    def select_action(self, state, context):
        return 999


def arena():
    return g.SequentialGoldenArena(
        master_seed=20260912,
        run_id="serialization-proof",
        code_identity=CLEAN_CODE,
    )


def rule_pair():
    session = arena()
    pair = session.play_pair(
        pair_id="serialization-pair",
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
    )
    return session, pair


def technical_record():
    return arena().play_game(
        game_id="technical-g1",
        pair_id="technical-pair",
        player_A=IllegalPlayer(),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )


def test_double_pass_record_round_trips_to_typed_validated_object():
    _, (record, _) = rule_pair()
    encoded = g.game_record_to_json(record)
    loaded = g.game_record_from_json(encoded)
    assert loaded == record
    assert isinstance(loaded.termination_reason, g.TerminationReason)
    assert isinstance(loaded.absolute_rule_result, g.Winner)
    assert isinstance(loaded.mapped_result, g.MappedResult)
    assert all(isinstance(item, g.ActionEvidence) for item in loaded.action_trace)
    g.validate_game_record(loaded)
    assert g.game_record_to_json(loaded) == encoded


def test_technical_record_round_trips_without_wdl_or_score():
    record = technical_record()
    assert record.termination_reason == g.TerminationReason.ERROR_ILLEGAL_PLAYER_ACTION
    loaded = g.load_and_validate_game_record(g.game_record_to_json(record))
    assert loaded == record
    assert loaded.absolute_rule_result is None
    assert loaded.mapped_result is None
    assert loaded.black_area is None
    assert loaded.white_area is None
    assert loaded.margin_black is None


def test_unknown_termination_enum_is_rejected_during_parse():
    _, (record, _) = rule_pair()
    payload = g.game_record_to_dict(record)
    payload["termination_reason"] = "UNKNOWN_TERMINATION"
    with pytest.raises(g.GoldenSerializationError, match="unknown TerminationReason"):
        g.game_record_from_dict(payload)


def test_missing_required_field_is_rejected():
    _, (record, _) = rule_pair()
    payload = g.game_record_to_dict(record)
    payload.pop("game_id")
    with pytest.raises(g.GoldenSerializationError, match="missing required fields: game_id"):
        g.game_record_from_dict(payload)


def test_malformed_action_evidence_is_rejected():
    _, (record, _) = rule_pair()
    payload = g.game_record_to_dict(record)
    payload["action_trace"][0]["ply"] = "1"
    with pytest.raises(g.GoldenSerializationError, match="action_trace\\[0\\]\\.ply.*integer"):
        g.game_record_from_dict(payload)


def test_wrong_scalar_type_is_not_silently_coerced():
    _, (record, _) = rule_pair()
    payload = g.game_record_to_dict(record)
    payload["seed_game"] = str(record.seed_game)
    with pytest.raises(g.GoldenSerializationError, match="seed_game.*integer"):
        g.game_record_from_dict(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_json_numbers_are_rejected(bad):
    _, (record, _) = rule_pair()
    payload = g.game_record_to_dict(record)
    payload["komi"] = bad
    encoded = json.dumps(payload)
    with pytest.raises(g.GoldenSerializationError, match="non-finite JSON constant"):
        g.game_record_from_json(encoded)


def test_structurally_valid_tamper_parses_but_semantic_validator_rejects():
    _, (record, _) = rule_pair()
    payload = g.game_record_to_dict(record)
    payload["mapped_result"] = g.MappedResult.B_WIN.value
    loaded = g.game_record_from_dict(payload)
    assert loaded.mapped_result == g.MappedResult.B_WIN
    with pytest.raises(ValueError, match="mapping"):
        g.validate_game_record(loaded)


def test_unsupported_schema_version_is_rejected():
    _, (record, _) = rule_pair()
    payload = g.game_record_to_dict(record)
    payload["schema_version"] = record.schema_version + 1
    with pytest.raises(g.GoldenSerializationError, match="unsupported schema version"):
        g.game_record_from_dict(payload)


def test_jsonl_writer_reader_round_trip_preserves_complete_pair(tmp_path):
    session, pair = rule_pair()
    path = tmp_path / "records.jsonl"
    g.write_records_jsonl(path, session.records)
    loaded = g.read_records_jsonl(path)
    assert loaded == pair
    for before, after in zip(pair, loaded, strict=True):
        assert g.game_record_to_json(after) == g.game_record_to_json(before)


def test_jsonl_corruption_is_not_skipped_and_reports_line_number(tmp_path):
    session, _ = rule_pair()
    path = tmp_path / "records.jsonl"
    g.write_records_jsonl(path, session.records)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n{" + "\n", encoding="utf-8")
    with pytest.raises(g.GoldenSerializationError, match="line 2"):
        g.read_records_jsonl(path)


def test_jsonl_blank_line_is_explicitly_rejected(tmp_path):
    session, _ = rule_pair()
    path = tmp_path / "records.jsonl"
    g.write_records_jsonl(path, session.records)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.splitlines()[0] + "\n\n" + text.splitlines()[1] + "\n", encoding="utf-8")
    with pytest.raises(g.GoldenSerializationError, match="line 2: blank lines are forbidden"):
        g.read_records_jsonl(path)
