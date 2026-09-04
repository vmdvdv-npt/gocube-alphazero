import numpy as np
import pytest

from alphazero.mcts_policy import normalize_masked_policy


def test_normal_policy_is_masked_and_normalized():
    result = normalize_masked_policy(
        np.array([0.2, 0.3, 0.5], dtype=np.float32),
        np.array([1, 0, 1], dtype=np.float32),
    )

    np.testing.assert_allclose(result, [2 / 7, 0, 5 / 7], rtol=1e-6)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-6)


def test_zero_legal_mass_falls_back_to_uniform_legal_policy():
    result = normalize_masked_policy(
        np.array([0.0, 0.7, 0.0, 0.3], dtype=np.float32),
        np.array([1, 0, 1, 0], dtype=np.float32),
    )

    np.testing.assert_array_equal(result, [0.5, 0.0, 0.5, 0.0])


def test_illegal_actions_are_zero_even_with_high_probability():
    result = normalize_masked_policy(
        np.array([0.1, 1000.0, 0.9], dtype=np.float32),
        np.array([1, 0, 1], dtype=np.float32),
    )

    np.testing.assert_allclose(result, [0.1, 0.0, 0.9], rtol=1e-6)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_pathological_legal_values_fall_back_without_nan(bad_value):
    result = normalize_masked_policy(
        np.array([bad_value, 0.9, 0.1], dtype=np.float32),
        np.array([1, 0, 1], dtype=np.float32),
    )

    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, [0.5, 0.0, 0.5])


def test_pathological_illegal_value_stays_zero_without_nan():
    result = normalize_masked_policy(
        np.array([0.25, np.inf, 0.75], dtype=np.float32),
        np.array([1, 0, 1], dtype=np.float32),
    )

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, [0.25, 0.0, 0.75], rtol=1e-6)


def test_single_legal_action_gets_probability_one():
    result = normalize_masked_policy(
        np.array([np.nan, 0.0, np.inf], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
    )

    np.testing.assert_array_equal(result, [0.0, 1.0, 0.0])


def test_no_legal_actions_raises_invariant_error():
    with pytest.raises(ValueError, match='no legal actions'):
        normalize_masked_policy(
            np.array([0.2, 0.8], dtype=np.float32),
            np.array([0, 0], dtype=np.float32),
        )
