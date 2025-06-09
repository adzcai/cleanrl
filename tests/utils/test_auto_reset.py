import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt

from testing.dummy_env import Params
from utils.auto_reset import PrevDone, auto_reset_wrapper


def test_auto_reset_wrapper(dummy_env):
    params = Params(max_horizon=2)
    wrapped = auto_reset_wrapper(dummy_env)
    key = jr.key(0)

    # Initial reset
    ts = wrapped.reset(params, key=key)
    assert isinstance(ts.state, PrevDone)
    assert ts.state.is_last == False
    assert ts.is_first == True
    assert jnp.isnan(ts.discount)

    # First step (not done)
    ts = wrapped.step(ts.state, None, params, key=key)
    assert ts.state.is_last == False
    assert ts.is_mid == True
    assert ts.discount == 1.0

    # Second step (should trigger auto-reset)
    ts = wrapped.step(ts.state, None, params, key=key)
    assert ts.state.is_last == True
    assert ts.is_last == True
    assert ts.discount == 0.0

    # After auto-reset, prev_done should be False again
    ts = wrapped.step(ts.state, None, params, key=key)
    assert ts.state.is_last == False
    assert ts.is_first == True
    assert jnp.isnan(ts.discount)
