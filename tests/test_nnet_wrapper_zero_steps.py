from alphazero.NNetWrapper import NNetWrapper


def test_zero_train_steps_is_safe_noop():
    wrapper = object.__new__(NNetWrapper)
    wrapper.total_steps = 123
    wrapper.current_step = 456
    wrapper.l_pi = 0.25
    wrapper.l_v = 0.5

    result = wrapper.train([], 0)

    assert result == (0.25, 0.5)
    assert wrapper.total_steps == 0
    assert wrapper.current_step == 0
