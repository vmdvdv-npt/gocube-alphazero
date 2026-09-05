import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import replace

from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.envs.gocube.records import (
    build_game_record,
    effective_parameter_snapshot,
    game_id_prefix,
    reserve_game_id,
    write_game_record,
    write_iteration_manifest,
)
from alphazero.utils import dotdict


def _reserve_ids(registry, output):
    output.put([reserve_game_id(registry, "C4") for _ in range(8)])


def _reserve_run_ids(registry, run_name, output):
    output.put((run_name, [reserve_game_id(registry, "C4") for _ in range(3)]))


def _example_record(tmp_path, game_id="C4-000001", game_number=1):
    game = Cube4JapaneseGame()
    game.play_action(0)
    game.play_action(game.pass_action())
    final_state = replace(game.semantic_state, phase="scored", terminal_kind="scored")
    game = Cube4JapaneseGame(final_state)
    record_path = os.path.relpath(
        tmp_path / f"{game_id}.json", os.getcwd()
    )
    return build_game_record(
        game=game,
        game_id=game_id,
        run_name="record-test",
        iteration=1,
        game_number=game_number,
        checkpoint={"id": "record-test@0", "path": "checkpoint/iteration-0000.pkl"},
        parameters=effective_parameter_snapshot({"workers": 2, "sims": 1}),
        moves=[
            {"move_number": 1, "player": "black", "phase": "main", "action": 0, "move": "front:0:0"},
            {"move_number": 2, "player": "white", "phase": "main", "action": 96, "move": "PASS"},
        ],
        start_time=1000.0,
        end_time=1001.25,
        winstate=game.win_state(),
        record_path=record_path,
    )


def test_cube_game_id_prefix_and_worker_allocations_are_unique(tmp_path):
    assert game_id_prefix(Cube4JapaneseGame) == "C4"
    context = mp.get_context("fork")
    output = context.Queue()
    workers = [context.Process(target=_reserve_ids, args=(str(tmp_path), output)) for _ in range(3)]
    for worker in workers:
        worker.start()
    batches = [output.get(timeout=10) for _ in workers]
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    ids = [game_id for batch in batches for game_id in batch]
    assert len(ids) == len(set(ids)) == 24
    assert sorted(ids) == [f"C4-{number:06d}" for number in range(1, 25)]


def test_independent_runs_continue_one_global_cube_registry(tmp_path):
    context = mp.get_context("fork")
    output = context.Queue()
    first = context.Process(target=_reserve_run_ids, args=(str(tmp_path), "run-a", output))
    first.start()
    first_run, first_ids = output.get(timeout=10)
    first.join(timeout=10)
    assert first.exitcode == 0

    second = context.Process(target=_reserve_run_ids, args=(str(tmp_path), "run-b", output))
    second.start()
    second_run, second_ids = output.get(timeout=10)
    second.join(timeout=10)
    assert second.exitcode == 0

    assert (first_run, second_run) == ("run-a", "run-b")
    assert first_ids == ["C4-000001", "C4-000002", "C4-000003"]
    assert second_ids == ["C4-000004", "C4-000005", "C4-000006"]
    assert set(first_ids).isdisjoint(second_ids)


def test_record_contains_replay_moves_final_position_and_terminal_metadata(tmp_path):
    record = _example_record(tmp_path)
    required = {
        "schema_version", "game_id", "run_name", "iteration",
        "game_number_inside_iteration", "checkpoint", "start_time", "end_time",
        "duration_seconds", "topology", "size", "rules", "effective_parameters",
        "moves", "number_of_moves", "final_position", "winner", "result",
        "final_score", "final_score_margin", "terminal_kind", "no_result_reason",
        "terminal", "cleanup_endgame_diagnostics", "record_path",
    }
    assert required <= set(record)
    assert record["game_id"] == "C4-000001"
    assert record["moves"][1]["move"] == "PASS"
    assert record["number_of_moves"] == 2
    assert record["final_position"]["board"]
    assert record["terminal_kind"] == "scored"
    assert record["result"] in {"black_win", "white_win", "draw"}
    assert record["final_score"] is not None
    assert record["final_score_margin"] == record["final_score"]["margin"]
    assert record["effective_parameters"] == {"sims": 1, "workers": 2}


def test_effective_parameter_snapshot_includes_dotdict_contents():
    assert effective_parameter_snapshot(dotdict({"workers": 2, "sims": 1})) == {
        "sims": 1,
        "workers": 2,
    }


def test_record_hash_and_iteration_manifest_point_to_existing_records(tmp_path):
    record_dir = tmp_path / "records" / "iteration-0001"
    entries = []
    for number in (1, 2):
        record = _example_record(record_dir, f"C4-{number:06d}", number)
        entry = write_game_record(record_dir, record)
        entry["game_number_inside_iteration"] = number
        entries.append(entry)
        record_bytes = (record_dir / f"C4-{number:06d}.json").read_bytes()
        assert entry["sha256"] == hashlib.sha256(record_bytes).hexdigest()

    manifest_path = write_iteration_manifest(
        record_dir,
        run_name="record-test",
        iteration=1,
        checkpoint={"id": "record-test@0"},
        parameters={"workers": 2},
        records=entries,
        aggregate_metrics={"games": 2, "draws": 0},
    )
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    assert manifest["schema_version"] == 1
    assert [item["game_id"] for item in manifest["records"]] == ["C4-000001", "C4-000002"]
    assert [item["game_number_inside_iteration"] for item in manifest["records"]] == [1, 2]
    for item in manifest["records"]:
        path = os.path.abspath(item["record_path"])
        assert os.path.exists(path)
        assert item["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest()
