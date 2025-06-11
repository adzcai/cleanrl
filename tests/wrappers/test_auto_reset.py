import jax.random as jr

from utils.structures import SENTINEL
from wrappers.auto_reset import PrevDone, auto_reset_wrapper


def test_auto_reset_wrapper(dummy_env_params):
    env, params = dummy_env_params
    wrapped = auto_reset_wrapper(env)
    key = jr.key(0)

    # Initial reset
    ts = wrapped.reset(params, key=key)
    assert isinstance(ts.state, PrevDone)
    assert not ts.state.is_last
    assert ts.is_first
    assert ts.discount == SENTINEL
    assert ts.reward == SENTINEL

    # First step (not done)
    for step in range(params.max_horizon - 1):
        ts = wrapped.step(ts.state, None, params, key=key)
        assert not ts.state.is_last
        assert ts.is_mid
        assert ts.discount == 1.0
        assert ts.reward == 1.0

    ts = wrapped.step(ts.state, None, params, key=key)
    assert ts.state.is_last
    assert ts.is_last
    assert ts.discount == 0.0
    assert ts.reward == 1.0

    # After auto-reset, prev_done should be False again
    ts = wrapped.step(ts.state, None, params, key=key)
    assert not ts.state.is_last
    assert ts.is_first
    assert ts.discount == SENTINEL
    assert ts.reward == SENTINEL
