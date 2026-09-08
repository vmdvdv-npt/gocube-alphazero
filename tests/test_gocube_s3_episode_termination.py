from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import alphazero.envs.gocube.katago_v3 as katago_v3
from alphazero.envs.gocube import (
    CLEANUP_1,
    CLEANUP_2,
    CYCLE,
    EPISODE_MOVE_LIMIT,
    FORMAL_PASS,
    PASS_ALIVE,
    RESULT_PROVENANCE_FORMAL,
    RESULT_PROVENANCE_RULE_NO_RESULT,
    RESULT_PROVENANCE_RUNTIME,
    SCORED,
    Cube4JapaneseGame,
    episode_move_limit,
)
from alphazero.envs.gocube.katago_v3 import (
    KATAGO_REFERENCE_COMMIT,
    KATAGO_RULES_VERSION,
    KATAGO_RULES_IMPLEMENTATION_VERSION,
    NO_RESULT,
    _cycle_check_and_record,
    initial_v3_state,
    rules_fingerprint,
    v3_state_from_board,
)
from alphazero.envs.gocube.pinned_game import PinnedCube4JapaneseGame
from alphazero.envs.gocube.pinned_selfplay import PinnedSelfPlayAgent
from alphazero.envs.gocube.records import (
    audit_termination_records,
    build_game_record,
)
from alphazero.envs.gocube.selfplay_semantics import (
    rebase_cleanup_training_state,
    should_stop_episode,
)


def test_s3_rules_identity_bumps_implementation_without_changing_pinned_upstream_rules():
    topology = Cube4JapaneseGame.logical_topology()
    assert KATAGO_RULES_VERSION == 3
    assert KATAGO_RULES_IMPLEMENTATION_VERSION == 5
    assert Cube4JapaneseGame.KATAGO_RULES_IMPLEMENTATION_VERSION == 5
    assert Cube4JapaneseGame.KATAGO_REFERENCE_COMMIT == KATAGO_REFERENCE_COMMIT

    current = rules_fingerprint(topology)
    assert current == Cube4JapaneseGame.rules_fingerprint()
    katago_v3.KATAGO_RULES_IMPLEMENTATION_VERSION = 4
    try:
        legacy_s1 = rules_fingerprint(topology)
    finally:
        katago_v3.KATAGO_RULES_IMPLEMENTATION_VERSION = 5
    assert legacy_s1 != current


def test_cube4_episode_limit_remains_the_production_formula():
    assert episode_move_limit(Cube4JapaneseGame.logical_topology()) == 2560


def test_same_formal_position_has_same_clone_terminal_near_runtime_limit():
    topology = Cube4JapaneseGame.logical_topology()
    limit = episode_move_limit(topology)
    low = PinnedCube4JapaneseGame(v3_state_from_board(topology, turns=0))
    high = PinnedCube4JapaneseGame(v3_state_from_board(topology, turns=limit - 1))
    for game in (low, high):
        game.configure_pinned_selfplay(
            auto_end_pass_alive=False,
            root_prune_useless_moves=False,
            seki_fork_hack_prob=0.0,
        )

    low_clone = low.clone()
    high_clone = high.clone()
    low_clone.play_action(0)
    high_clone.play_action(0)

    assert low_clone.terminal_kind is None
    assert high_clone.terminal_kind is None
    assert low_clone.termination_reason == high_clone.termination_reason is None


def test_runner_hook_force_scores_only_the_real_game():
    game = PinnedCube4JapaneseGame()
    for action in (0, 1, 2):
        game.play_action(action)

    agent = object.__new__(PinnedSelfPlayAgent)
    agent._is_arena = False
    agent._is_warmup = False
    agent._episode_move_limit = 3
    agent.games = [game]
    agent.telemetry = None
    agent._after_game_action(0)

    assert game.terminal_kind == SCORED
    assert game.termination_reason == EPISODE_MOVE_LIMIT
    assert game.result_provenance == RESULT_PROVENANCE_RUNTIME
    targets = game.training_target_bundle()
    assert targets.termination_reason == EPISODE_MOVE_LIMIT
    assert targets.result_provenance == RESULT_PROVENANCE_RUNTIME
    assert targets.score_mask.tolist() == [1.0]
    assert np.all(targets.ownership_mask == 1.0)


def test_formal_pass_and_cycle_keep_distinct_provenance():
    game = PinnedCube4JapaneseGame()
    game.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    for _ in range(6):
        game.play_action(game.pass_action())
    assert game.terminal_kind == SCORED
    assert game.termination_reason == FORMAL_PASS
    assert game.result_provenance == RESULT_PROVENANCE_FORMAL

    topology = Cube4JapaneseGame.logical_topology()
    state = initial_v3_state(topology)
    key = state.history_since_pass[0]
    cycle = _cycle_check_and_record(
        replace(state, history_since_pass=(key, key, key)),
        after_pass=False,
    )
    assert cycle.terminal_kind == NO_RESULT
    assert cycle.termination_reason == CYCLE
    assert cycle.result_provenance == RESULT_PROVENANCE_RULE_NO_RESULT


def test_pass_alive_is_formal_and_never_episode_limit():
    topology = Cube4JapaneseGame.logical_topology()
    empty = (10, 20)
    state = v3_state_from_board(
        topology,
        black=(point for point in range(topology.point_count) if point not in empty),
    )
    game = PinnedCube4JapaneseGame(state)
    game.configure_pinned_selfplay(
        auto_end_pass_alive=True,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    game.play_action(game.pass_action())

    assert game.terminal_kind == SCORED
    assert game.termination_reason == PASS_ALIVE
    assert game.result_provenance == RESULT_PROVENANCE_FORMAL
    assert game.episode_move_count == 1


def test_synthetic_cleanup_and_fork_episodes_have_independent_runtime_counts():
    topology = Cube4JapaneseGame.logical_topology()
    for phase in (CLEANUP_1, CLEANUP_2):
        rebased = rebase_cleanup_training_state(
            initial_v3_state(topology), phase
        )
        game = PinnedCube4JapaneseGame(rebased)
        game.mark_synthetic_cleanup_episode()
        game._pinned_episode_move_count = 2
        assert game.episode_type == "synthetic_cleanup"
        assert game.turns == 0
        assert game.finalize_episode_due_to_runtime_limit(2)
        assert game.termination_reason == EPISODE_MOVE_LIMIT
        assert game.result_provenance == RESULT_PROVENANCE_RUNTIME

    pool = PinnedCube4JapaneseGame._seki_pool()
    pool.clear()
    source = PinnedCube4JapaneseGame()
    source.play_action(0)
    pool.append((source.semantic_state, source._pinned_move_history))
    target = PinnedCube4JapaneseGame()
    assert target.maybe_start_seki_fork(1.0)
    assert target.episode_type == "fork"
    assert target.episode_move_count == 0


def test_runtime_count_is_separate_from_formal_turn_counter():
    game = PinnedCube4JapaneseGame()
    game._state = replace(game.semantic_state, turns=2559)
    game._sync_framework_fields()
    assert game.episode_move_count == 0
    game.play_action(0)
    assert game.turns == 2560
    assert game.episode_move_count == 1
    assert game.terminal_kind is None
    assert not should_stop_episode(game.episode_move_count, game.logical_topology())


def test_record_persists_runtime_termination_and_target_provenance(tmp_path):
    game = PinnedCube4JapaneseGame()
    game._pinned_episode_move_count = 3
    assert game.finalize_episode_due_to_runtime_limit(3)
    record = build_game_record(
        game=game,
        game_id="C4-000001",
        run_name="s3-test",
        iteration=1,
        game_number=1,
        checkpoint={"id": "s3-test@0"},
        parameters={"gocube_episode_move_limit": 3},
        moves=[],
        start_time=1000.0,
        end_time=1001.0,
        winstate=game.win_state(),
        record_path=str(tmp_path / "C4-000001.json"),
    )
    assert record["termination_reason"] == EPISODE_MOVE_LIMIT
    assert record["result_provenance"] == RESULT_PROVENANCE_RUNTIME
    assert record["target_provenance"] == RESULT_PROVENANCE_RUNTIME
    assert record["termination"]["formal_terminal"] is False
    assert record["termination"]["runtime_forced"] is True
    assert record["final_position"]["episode_move_count"] == 3


def test_historical_termination_audit_is_read_only_and_fail_closed(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"terminal_kind": "scored"}), encoding="utf-8")
    report = audit_termination_records([old])
    assert report["counts"] == {"unknown_legacy_termination": 1}
    assert old.read_text(encoding="utf-8") == json.dumps({"terminal_kind": "scored"})


def test_termination_audit_reports_episode_limit_frequency_by_type(tmp_path):
    paths = []
    for index, episode_type in enumerate(("ordinary", "synthetic_cleanup", "fork")):
        path = tmp_path / f"{index}.json"
        path.write_text(
            json.dumps({
                "termination_reason": EPISODE_MOVE_LIMIT,
                "episode_type": episode_type,
            }),
            encoding="utf-8",
        )
        paths.append(path)

    report = audit_termination_records(paths)
    assert report["episode_move_limit_count"] == 3
    assert report["episode_move_limit_fraction"] == 1.0
    assert report["episode_move_limit_by_episode_type"] == {
        "ordinary": 1,
        "synthetic_cleanup": 1,
        "fork": 1,
    }
