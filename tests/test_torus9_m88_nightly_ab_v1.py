from __future__ import annotations

from tools import torus9_m88_nightly_ab_v1 as nightly
from tools.torus9_m88_nightly_support import install_nightly_profile_validation


def test_campaign_has_expected_six_arms_and_budgets() -> None:
    config = nightly.load_config()
    arms = nightly.arms_from_config(config)

    assert list(arms) == [
        "games-g128",
        "games-g192",
        "games-g384",
        "lr-low",
        "lr-current",
        "lr-high",
    ]
    assert [(arms[key].games, arms[key].iterations, arms[key].steps)
            for key in ("games-g128", "games-g192", "games-g384")] == [
        (128, 6, 160),
        (192, 4, 160),
        (384, 2, 160),
    ]
    assert {arms[key].total_games for key in ("games-g128", "games-g192", "games-g384")} == {768}
    assert [(arms[key].games, arms[key].iterations, arms[key].lr)
            for key in ("lr-low", "lr-current", "lr-high")] == [
        (128, 3, 0.0001),
        (128, 3, 0.0003),
        (128, 3, 0.001),
    ]


def test_nightly_profiles_materialize_on_old_v1_boundary() -> None:
    install_nightly_profile_validation()
    config = nightly.load_config()
    arms = nightly.arms_from_config(config)

    for arm in arms.values():
        spec = nightly.make_run_spec(config, arm)
        generation = spec.payload["generation"]
        driver_config = generation["driver_config"]
        profile = spec.orchestrator_spec.profile_payload

        assert generation["command"][1] == nightly.DRIVER
        assert generation["resume_command"][1] == nightly.DRIVER
        assert driver_config["games"] == arm.games
        assert driver_config["optimizer_steps_per_iteration"] == 160
        assert profile["self_play"]["mcts_simulations"] == 128
        assert profile["training"]["learning_rate"] == arm.lr
        assert profile["replay"]["generations"] == 6
        assert profile["replay"]["cap"] == 40000
        assert spec.payload["arena"]["enabled"] is False
