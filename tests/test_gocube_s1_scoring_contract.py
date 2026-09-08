from dataclasses import replace

import numpy as np

import alphazero.envs.gocube.katago_v3 as katago_v3
from alphazero.envs.gocube import (
    CLEANUP_1,
    CLEANUP_2,
    MAIN,
    build_v3_training_targets,
    final_v3_score,
    initial_v3_state,
    terminal_from_state,
    v3_state_from_board,
    apply_v3_action,
)
from alphazero.envs.gocube.selfplay_semantics import rebase_cleanup_training_state
from gocube_reference_topology import rectangular_test_topology
from katago_reference_runner import KatagoOracleProcess


def _counterexample_setup(topology):
    black = tuple(p for p in range(topology.point_count) if p not in (6, 7, 18))
    return v3_state_from_board(topology, black=black, white=(7,), phase=MAIN)


def _score_after_passes(state, topology, count):
    for _ in range(count):
        state = apply_v3_action(state, topology.pass_action, topology)
    terminal = terminal_from_state(state, topology, 0.5)
    assert terminal is not None and terminal.score is not None
    return state, terminal


def test_s1_pass_only_counterexample_uses_one_scorer_path(monkeypatch):
    topology = rectangular_test_topology(5, 5)
    state = _counterexample_setup(topology)

    def legacy_must_not_run(*args, **kwargs):
        raise AssertionError("pre-S1 board-only scorer was selected")

    monkeypatch.setattr(katago_v3, "_legacy_board_only_life_analysis", legacy_must_not_run)
    no_telemetry_score = final_v3_score(state, topology, 0.5)[0]
    telemetry_score = final_v3_score(
        replace(state, main_moves=(3, 2)), topology, 0.5
    )[0]

    assert no_telemetry_score.white - no_telemetry_score.black == -3.5
    assert no_telemetry_score.winner == "black"
    assert telemetry_score == no_telemetry_score


def test_boardhistory_clear_offset_preserves_setup_captures_and_rebases():
    topology = rectangular_test_topology(5, 5)
    black = tuple(p for p in range(topology.point_count) if p not in (6, 7, 18))
    state = v3_state_from_board(
        topology,
        black=black,
        white=(7,),
        captures=(2, 1),
        phase=MAIN,
    )
    # 22 black stones - 1 white stone - 2 white captures by Black
    # + 1 black capture by White.
    assert state.white_bonus_score == 20.0
    assert state.captures == (2, 1)

    cleanup1, terminal1 = _score_after_passes(
        rebase_cleanup_training_state(state, CLEANUP_1), topology, 4
    )
    cleanup2, terminal2 = _score_after_passes(
        rebase_cleanup_training_state(state, CLEANUP_2), topology, 2
    )
    assert cleanup1.phase == "scored"
    assert cleanup2.phase == "scored"
    assert terminal1.score == terminal2.score
    assert terminal2.score.white - terminal2.score.black == -4.5
    assert terminal2.score.captures == (2, 1)

    setup = {
        "black": [[p % 5, p // 5] for p in black],
        "white": [[2, 1]],
        # Native KataGo names these counters by captured colour.
        "captures": {"black": 1, "white": 2},
        "encore_phase": 1,
    }
    with KatagoOracleProcess(x_size=5, y_size=5) as oracle:
        reference = oracle.setup(setup)
        for _ in range(4):
            reference = oracle.play("pass")
    assert reference["phase"] == "SCORED"
    assert reference["final_score"] == terminal1.score.white - terminal1.score.black == -4.5


def test_s1_score_ownership_and_targets_come_from_same_terminal_result():
    topology = rectangular_test_topology(5, 5)
    final_state, terminal = _score_after_passes(
        _counterexample_setup(topology), topology, 6
    )
    assert final_state.phase == "scored"
    assert terminal.ownership_target_valid
    assert terminal.ownership_mask is not None
    assert np.all(terminal.ownership_mask == 1.0)
    assert np.all(np.argmax(terminal.ownership, axis=1) == 0)

    targets = build_v3_training_targets(terminal, final_state.current_player, topology)
    assert np.array_equal(targets.value_target, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(targets.score_target, np.asarray([3.5 / 25], dtype=np.float32))
    assert np.all(targets.score_mask == 1.0)
    assert np.array_equal(targets.ownership_target, terminal.ownership)


def test_cleanup_setup_phases_have_explicit_second_start_and_pass_transitions():
    topology = rectangular_test_topology(3, 3)
    board_state = v3_state_from_board(
        topology,
        black=(0, 1, 3),
        white=(4,),
        captures=(1, 2),
        phase=MAIN,
    )
    cleanup1 = rebase_cleanup_training_state(board_state, CLEANUP_1)
    cleanup2 = rebase_cleanup_training_state(board_state, CLEANUP_2)
    assert cleanup1.main_moves == (0, 0)
    assert cleanup1.second_cleanup_start_colors is None
    assert cleanup2.main_moves == (0, 0)
    assert cleanup2.second_cleanup_start_colors == bytes(cleanup2.board.tolist())
    assert cleanup1.white_bonus_score == cleanup2.white_bonus_score == 3.0

    after_c1, _ = _score_after_passes(cleanup1, topology, 4)
    after_c2, _ = _score_after_passes(cleanup2, topology, 2)
    assert after_c1.phase == after_c2.phase == "scored"
