import pytest

from gocube_golden.torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    M137_FIVE_CHANNEL_CHANNELS,
    Torus9M137FiveChannelGraphNet,
)
from gocube_golden.torus9_monolith import (
    TORUS9_OBSERVATION_CHANNELS,
    Torus9CurrentGraphNet,
)
from gocube_golden.torus9_new_komi_guard import (
    NO_KOMI_OBSERVATION_POLICY_ID,
    assert_new_komi_training_model,
    assert_new_komi_training_observation_contract,
)


def test_new_komi_training_guard_accepts_only_canonical_five_channel_model() -> None:
    evidence = assert_new_komi_training_model(Torus9M137FiveChannelGraphNet())
    assert evidence["policy_id"] == NO_KOMI_OBSERVATION_POLICY_ID
    assert evidence["architecture_id"] == M137_FIVE_CHANNEL_ARCHITECTURE_ID
    assert evidence["input_channels"] == 5
    assert evidence["observation_channels"] == list(M137_FIVE_CHANNEL_CHANNELS)
    assert evidence["komi_channel"] is False


def test_new_komi_training_guard_rejects_legacy_six_channel_model() -> None:
    with pytest.raises(ValueError, match="exactly 5 neural observation channels"):
        assert_new_komi_training_model(Torus9CurrentGraphNet())


def test_new_komi_training_guard_rejects_explicit_komi_channel() -> None:
    channels = (
        "own_stones",
        "opponent_stones",
        "side_to_move_color",
        "previous_pass",
        "komi",
    )
    with pytest.raises(ValueError, match="forbids komi as a neural observation channel"):
        assert_new_komi_training_observation_contract(
            architecture_id=M137_FIVE_CHANNEL_ARCHITECTURE_ID,
            input_channels=5,
            observation_channels=channels,
        )


def test_new_komi_training_guard_rejects_historical_0_5_observation_contract() -> None:
    assert "komi" in TORUS9_OBSERVATION_CHANNELS
    with pytest.raises(ValueError, match="forbids komi as a neural observation channel"):
        assert_new_komi_training_observation_contract(
            architecture_id="GoldenGraphNetV2-Torus9",
            input_channels=6,
            observation_channels=TORUS9_OBSERVATION_CHANNELS,
        )
