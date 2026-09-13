from __future__ import annotations

import time

import pytest

import gocube_golden as g
from gocube_golden.provenance import derive_game_seeds


class SlowGoodPlayer:
    player_id = "GOOD"
    is_search_player = False

    def select_action(self, state, context):
        time.sleep(0.03 if context.game_id.endswith("-g1") else 0.001)
        return 0 if context.player_move_index == 0 else g.PASS


class RaisingFactory:
    def __call__(self):
        raise RuntimeError("initializer failure")


class IllegalPlayer:
    player_id = "ILLEGAL"
    is_search_player = False

    def select_action(self, state, context):
        return 999


def _sequential_records(*, pairs, master_seed, run_id, code_identity):
    arena = g.SequentialGoldenArena(
        master_seed=master_seed,
        run_id=run_id,
        code_identity=code_identity,
    )
    for pair in pairs:
        arena.play_pair(
            pair_id=pair.pair_id,
            player_A=g.GoodPlayer("A"),
            player_B=g.BadPlayer("B"),
            start_state=pair.start_state,
            start_trace=pair.start_trace,
            game_ids=pair.game_ids,
        )
    return arena.records


def test_process_parallel_pair_is_bit_exact_to_sequential_oracle():
    code = g.capture_code_identity()
    pairs = tuple(
        g.PairTask(pair_id=f"pair-{index}", start_trace=())
        for index in range(3)
    )
    parallel = g.ProcessParallelGoldenArena(
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
        workers=2,
        master_seed=91,
        run_id="process-parity-test",
        code_identity=code,
    )
    parallel_records = parallel.play_pairs(pairs)
    sequential_records = _sequential_records(
        pairs=pairs,
        master_seed=91,
        run_id="process-parity-test",
        code_identity=code,
    )
    assert parallel_records == sequential_records
    assert [record.game_id for record in parallel_records] == [
        "pair-0-g1", "pair-0-g2", "pair-1-g1", "pair-1-g2", "pair-2-g1", "pair-2-g2"
    ]
    assert parallel.canonical_schedule == (
        ("pair-0", "pair-0-g1", "pair-0-g2"),
        ("pair-1", "pair-1-g1", "pair-1-g2"),
        ("pair-2", "pair-2-g1", "pair-2-g2"),
    )


@pytest.mark.parametrize("workers", (1, 2, 4))
def test_worker_count_does_not_change_semantic_records(workers):
    code = g.capture_code_identity()
    pairs = tuple(g.PairTask(pair_id=f"stable-{index}") for index in range(2))
    arena = g.ProcessParallelGoldenArena(
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
        workers=workers,
        master_seed=123,
        run_id="worker-count-test",
        code_identity=code,
    )
    assert arena.play_pairs(pairs) == _sequential_records(
        pairs=pairs,
        master_seed=123,
        run_id="worker-count-test",
        code_identity=code,
    )


def test_completion_order_is_not_persisted_order():
    code = g.capture_code_identity()
    pairs = tuple(g.PairTask(pair_id=f"ordered-{index}") for index in range(2))
    arena = g.ProcessParallelGoldenArena(
        player_A=SlowGoodPlayer(),
        player_B=g.BadPlayer("B"),
        workers=2,
        master_seed=7,
        run_id="completion-order-test",
        code_identity=code,
    )
    records = arena.play_pairs(pairs)
    assert tuple(record.game_id for record in records) == (
        "ordered-0-g1", "ordered-0-g2", "ordered-1-g1", "ordered-1-g2"
    )


def test_task_contains_stable_seed_and_checkpoint_identity_evidence():
    code = g.capture_code_identity()
    arena = g.ProcessParallelGoldenArena(
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
        workers=1,
        master_seed=55,
        run_id="task-evidence-test",
        code_identity=code,
    )
    task = arena.build_tasks((g.PairTask(pair_id="evidence"),))[0]
    assert task.checkpoint_A_identity == task.player_A_identity
    assert task.checkpoint_B_identity == task.player_B_identity
    assert (task.master_seed, task.pair_id, task.game_id) == (55, "evidence", "evidence-g1")
    record = arena.play_games((task,))[0]
    assert (record.seed_game, record.seed_A, record.seed_B) == derive_game_seeds(
        55, "evidence", "evidence-g1"
    )


def test_duplicate_game_id_is_rejected_before_pool_start():
    arena = g.ProcessParallelGoldenArena(
        player_A=g.GoodPlayer("A"), player_B=g.BadPlayer("B"), workers=1
    )
    with pytest.raises(ValueError, match="Duplicate game_id"):
        arena.play_pair(pair_id="duplicate", game_ids=("same", "same"))


def test_worker_initializer_failure_is_fail_closed_without_partial_records():
    identity = g.infer_player_identity(g.GoodPlayer("A"))
    arena = g.ProcessParallelGoldenArena(
        player_A=g.PlayerSpec.from_factory(RaisingFactory(), identity=identity),
        player_B=g.BadPlayer("B"),
        workers=2,
        run_id="initializer-failure-test",
    )
    with pytest.raises(g.ArenaWorkerError, match="worker|initializer|failed"):
        arena.play_pairs((g.PairTask(pair_id="will-fail"),))
    assert arena.records == ()
    with pytest.raises(g.ArenaWorkerError, match="failed closed"):
        arena.summary()


def test_technical_game_is_returned_as_technical_and_not_as_draw():
    arena = g.ProcessParallelGoldenArena(
        player_A=IllegalPlayer(), player_B=g.BadPlayer("B"), workers=2
    )
    records = arena.play_pairs((g.PairTask(pair_id="technical"),))
    assert len(records) == 2
    assert all(record.is_technical for record in records)
    assert arena.summary().technical_failures == 2
    assert arena.summary().draws == 0


def test_process_arena_validates_pair_after_all_results_are_collected():
    arena = g.ProcessParallelGoldenArena(
        player_A=g.GoodPlayer("A"), player_B=g.BadPlayer("B"), workers=2
    )
    records = arena.play_pairs((g.PairTask(pair_id="validated"),))
    assert g.validate_pair_records(records)[0].game_id == "validated-g1"
    assert arena.manifest().game_count == 2
