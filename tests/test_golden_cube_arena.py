from __future__ import annotations

from gocube_golden.cube_arena import (
    CUBE_ARENA_FINGERPRINT,
    CubeArenaRecord,
    cube_hoeffding_interval,
    summarize_cube_arena,
)
from gocube_golden.cube_topology import CUBE4_TOPOLOGY
from gocube_golden.cube_training import cube_initial_state, cube_state_identity


def _record(*, game_id: str, pair_id: str, black_player: str, mapped_result: str) -> CubeArenaRecord:
    state = cube_initial_state()
    winner_slot = "A" if mapped_result == "A_WIN" else "B"
    formal_result = "BLACK" if winner_slot == black_player else "WHITE"
    return CubeArenaRecord(
        run_id="arena-test",
        pair_id=pair_id,
        game_id=game_id,
        master_seed=1,
        game_seed=2,
        topology_fingerprint=CUBE4_TOPOLOGY.fingerprint,
        geometry_fingerprint=CUBE4_TOPOLOGY.geometry_fingerprint,
        rules_fingerprint=state.rules_fingerprint,
        komi=0.5,
        player_A_id="candidate",
        player_B_id="reference",
        black_player=black_player,
        white_player="B" if black_player == "A" else "A",
        start_state=cube_state_identity(state),
        start_trace=(0, 1, 2, 3),
        action_trace=(),
        final_board=tuple(0 for _ in range(96)),
        formal_result=formal_result,
        mapped_result=mapped_result,
        black_area=48,
        white_area=48,
        margin_black=0.5,
        technical_termination=None,
        error=None,
    )


def test_cube_arena_summary_is_paired_color_swapped_and_stratified():
    records = (
        _record(game_id="p-g1", pair_id="p", black_player="A", mapped_result="A_WIN"),
        _record(game_id="p-g2", pair_id="p", black_player="B", mapped_result="B_WIN"),
    )
    summary = summarize_cube_arena(records)
    assert summary["arena_fingerprint"] == CUBE_ARENA_FINGERPRINT
    assert summary["pairs"] == 1 and summary["games"] == 2
    assert summary["mean_pair_score"] == 0.5
    assert summary["candidate_score_as_black"] == 1.0
    assert summary["candidate_score_as_white"] == 0.0
    assert summary["prefix_stratum_breakdown"]["4"]["pairs"] == 1
    assert summary["color_control"]["diagnostic"] == "LOW COLOR-CONTROLLED DISCRIMINATIVE POWER"


def test_cube_hoeffding_interval_is_bounded_and_not_normal_approximation():
    lower, upper = cube_hoeffding_interval([1.0] * 64)
    assert lower > 0.5
    assert upper == 1.0
