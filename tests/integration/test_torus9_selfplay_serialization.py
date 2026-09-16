from __future__ import annotations

import json
from pathlib import Path

import gocube_golden as g
from gocube_golden.torus9_serialization import write_torus9_game_records_jsonl


def _record(game_id: str) -> g.Torus9SelfPlayGameRecord:
    position = g.Torus9SelfPlayPosition(
        ply=1,
        state={
            "stones": [0, 1, 0],
            "side_to_move": 1,
            "superko_history": [[0, 1, 0], [1, 1, 0]],
            "nested": {"values": [1, 2, 3]},
        },
        side_to_move="BLACK",
        root_visits=(1, 0, 2),
        pi=(0.25, 0.0, 0.75),
        selected_action=g.PASS,
        search_seed=123,
        model_hash="model-hash",
    )
    return g.Torus9SelfPlayGameRecord(
        run_id="serialization-test",
        game_id=game_id,
        profile_id="profile",
        profile_fingerprint="profile-fingerprint",
        selfplay_contract_id="contract",
        selfplay_contract_fingerprint="contract-fingerprint",
        model_checkpoint_label="M0",
        model_hash="model-hash",
        checkpoint_artifact_hash="artifact-hash",
        git_commit="commit",
        git_tree="tree",
        git_worktree_clean=True,
        master_seed=1,
        game_seed=2,
        start_state={"stones": [0, 0, 0], "superko_history": [[0, 0, 0]]},
        positions=(position,),
        final_action_trace=(g.PASS,),
        formal_result="BLACK",
        technical_termination=None,
        error=None,
        nn_evaluations=7,
    )


def _legacy_bytes(records: tuple[g.Torus9SelfPlayGameRecord, ...]) -> str:
    return "".join(
        json.dumps(record.to_dict(), sort_keys=True) + "\n"
        for record in records
    )


def test_streaming_selfplay_jsonl_is_byte_identical_to_legacy(tmp_path: Path) -> None:
    records = (_record("game-0000"), _record("game-0001"))
    expected = _legacy_bytes(records)
    path = tmp_path / "games.jsonl"

    write_torus9_game_records_jsonl(path, records)

    assert path.read_text(encoding="utf-8") == expected


def test_streaming_writer_bypasses_to_dict_and_accepts_one_pass_iterable(
    tmp_path: Path, monkeypatch
) -> None:
    records = (_record("game-0000"), _record("game-0001"))
    expected = _legacy_bytes(records)

    def fail_to_dict(self):
        raise AssertionError("streaming serializer must not call to_dict/asdict")

    monkeypatch.setattr(g.Torus9SelfPlayGameRecord, "to_dict", fail_to_dict)
    consumed: list[str] = []

    def one_pass():
        for record in records:
            consumed.append(record.game_id)
            yield record

    path = tmp_path / "games.jsonl"
    write_torus9_game_records_jsonl(path, one_pass())

    assert consumed == ["game-0000", "game-0001"]
    assert path.read_text(encoding="utf-8") == expected


def test_nightly_diagnostics_uses_direct_streaming_record_writer() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "torus9_nightly_diagnostics.py"
    ).read_text(encoding="utf-8")
    assert "write_torus9_game_records_jsonl(games_path, records)" in source
    assert "write_jsonl(games_path, [record.to_dict() for record in records])" not in source
