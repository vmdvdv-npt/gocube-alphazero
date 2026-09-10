import pytest

from alphazero.envs.gocube import Cube2JapaneseGame
from alphazero.envs.gocube.golden_scoring import GoldenOutcome
from tools.gocube_golden_arena import (
    GoldenArenaError,
    GoldenSequentialArena,
    _result_for_model_a,
    golden_game_plan,
)


def test_golden_game_plan_is_paired_and_color_balanced():
    g0 = golden_game_plan(0, 1000)
    g1 = golden_game_plan(1, 1000)
    g2 = golden_game_plan(2, 1000)
    g3 = golden_game_plan(3, 1000)

    assert g0 == {
        "game_id": 0,
        "pair_id": 0,
        "seed": 1000,
        "model_a_color": "black",
    }
    assert g1 == {
        "game_id": 1,
        "pair_id": 0,
        "seed": 1000,
        "model_a_color": "white",
    }
    assert g2["pair_id"] == g3["pair_id"] == 1
    assert g2["seed"] == g3["seed"] == 1001
    assert g2["model_a_color"] == "black"
    assert g3["model_a_color"] == "white"


@pytest.mark.parametrize(
    ("outcome", "a_color", "expected"),
    [
        (GoldenOutcome.BLACK, "black", "win"),
        (GoldenOutcome.BLACK, "white", "loss"),
        (GoldenOutcome.WHITE, "black", "loss"),
        (GoldenOutcome.WHITE, "white", "win"),
        (GoldenOutcome.DRAW, "black", "draw"),
        (GoldenOutcome.DRAW, "white", "draw"),
        (GoldenOutcome.NO_RESULT, "black", "no_result"),
        (GoldenOutcome.NO_RESULT, "white", "no_result"),
    ],
)
def test_model_result_mapping_uses_absolute_color(outcome, a_color, expected):
    assert _result_for_model_a(outcome, a_color) == expected


def test_invalid_model_color_fails_closed():
    with pytest.raises(GoldenArenaError, match="Invalid model A color"):
        _result_for_model_a(GoldenOutcome.BLACK, "player0")


class _PassPlayer:
    def reset(self):
        pass

    def update(self, state, action):
        pass

    def __call__(self, state):
        return int(state.pass_action())


class _OneMoveThenPassPlayer(_PassPlayer):
    def reset(self):
        self.played = False

    def __call__(self, state):
        if not self.played:
            valids = state.valid_moves()
            for action in range(state.logical_topology().point_count):
                if bool(valids[action]):
                    self.played = True
                    return action
        return int(state.pass_action())


class _WinStateForbiddenCube2(Cube2JapaneseGame):
    def win_state(self):
        raise AssertionError("Golden Arena must not call production win_state()")


def test_golden_sequential_arena_never_uses_win_state_for_result_mapping():
    arena = GoldenSequentialArena(
        [_PassPlayer(), _PassPlayer()],
        _WinStateForbiddenCube2,
        base_seed=123,
    )
    summary = arena.play(2)

    # Empty-board formal territory score is 0 for Black and 0.5 for White.
    # The two games swap models, so A loses as Black and wins as White.
    assert summary["black_wins"] == 0
    assert summary["white_wins"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["draws"] == 0
    assert summary["no_results"] == 0
    assert summary["records"][0]["model_a_color"] == "black"
    assert summary["records"][0]["model_a_result"] == "loss"
    assert summary["records"][1]["model_a_color"] == "white"
    assert summary["records"][1]["model_a_result"] == "win"
    assert all(
        record["production_score_agreed"] is True
        for record in summary["records"]
    )


def test_golden_runner_reconstructs_white_bonus_from_moves():
    arena = GoldenSequentialArena(
        [_OneMoveThenPassPlayer(), _PassPlayer()],
        Cube2JapaneseGame,
        base_seed=321,
    )
    record = arena.play_one(0)

    assert record["moves"] >= 7
    assert record["production_score_agreed"] is True
    assert record["score"] is not None
