from types import SimpleNamespace

from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.envs.gocube.train import expected_phase_saved_samples
from alphazero.utils import dotdict


def _game(phase, consecutive_passes=0):
    return SimpleNamespace(
        semantic_state=SimpleNamespace(
            phase=phase,
            consecutive_passes=consecutive_passes,
        )
    )


def test_phase_specific_sampling_weights_are_independent_and_accounted():
    """One synthetic base position in each V3 bucket gets only its own weight."""

    agent = SelfPlayAgent.__new__(SelfPlayAgent)
    agent.args = dotdict({
        "gocube_endgame_sample_weight": 99,  # deprecated blanket knob must be irrelevant
        "gocube_main_after_pass_weight": 2,
        "gocube_cleanup1_weight": 3,
        "gocube_cleanup2_weight": 4,
    })

    cases = (
        (_game("main", 0), "main", 1),
        (_game("main", 1), "main_after_one_pass", 2),
        (_game("cleanup1", 0), "cleanup1", 3),
        (_game("cleanup2", 0), "cleanup2", 4),
    )

    telemetry = {}
    for game, expected_bucket, expected_repeat in cases:
        bucket = agent._phase_bucket(game)
        repeat = agent._phase_weight(bucket)
        assert bucket == expected_bucket
        assert repeat == expected_repeat
        telemetry[f"weighted_samples_{bucket}"] = repeat

    # Four base positions become 1 + 2 + 3 + 4 rows. The old blanket
    # endgame multiplier deliberately has no effect on the result.
    assert expected_phase_saved_samples(telemetry) == 10
