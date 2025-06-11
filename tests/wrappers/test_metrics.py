import jax
import jax.numpy as jnp
import pytest
import jax.random as jr
import numpy.testing as npt

from src.testing.dummy_env import make_dummy_env, simple_rollout
from src.wrappers.metrics import metrics_wrapper
from wrappers.auto_reset import auto_reset_wrapper


def test_metrics_accumulates_and_resets(dummy_env_params):
    # Create dummy env with horizon 3
    env, params = dummy_env_params
    env = auto_reset_wrapper(env)
    wrapped_env = metrics_wrapper(env)

    key1, key2 = jr.split(jr.key(0))

    # Take a single rollout
    ts_seq = simple_rollout(
        wrapped_env, params, action=None, horizon=params.max_horizon, key=key1
    )
    npt.assert_array_equal(
        ts_seq.state.metrics.cum_return, jnp.arange(params.max_horizon + 1)
    )
    npt.assert_array_equal(
        ts_seq.state.metrics.step, jnp.arange(params.max_horizon + 1)
    )

    # Reset and rollout again: metrics should start from zero
    ts_seq2 = simple_rollout(
        wrapped_env, params, action=None, horizon=2 * params.max_horizon + 1, key=key2
    )
    # first trajectory
    npt.assert_array_equal(
        ts_seq2.state.metrics.cum_return[: params.max_horizon + 1],
        jnp.arange(params.max_horizon + 1),
    )
    npt.assert_array_equal(
        ts_seq2.state.metrics.step[: params.max_horizon + 1],
        jnp.arange(params.max_horizon + 1),
    )
    # second trajectory
    npt.assert_array_equal(
        ts_seq2.state.metrics.cum_return[params.max_horizon + 1 :],
        jnp.arange(params.max_horizon + 1),
    )
    npt.assert_array_equal(
        ts_seq2.state.metrics.step[params.max_horizon + 1 :],
        jnp.arange(params.max_horizon + 1),
    )


def test_metrics_increments_each_step():
    env, params = make_dummy_env(max_horizon=5)
    wrapped_env = metrics_wrapper(env)
    key = jax.random.PRNGKey(123)
    ts_seq = simple_rollout(wrapped_env, params, action=0, horizon=5, key=key)
    # Each step should increment by 1
    assert jnp.all(ts_seq.state.metrics.cum_return == jnp.arange(6))
    assert jnp.all(ts_seq.state.metrics.step == jnp.arange(6))


def test_metrics_wrapper_outside_auto_reset_raises(pytree_env):
    """Test that putting metrics_wrapper outside auto_reset_wrapper raises an error or fails."""
    with pytest.raises(ValueError):
        wrapped_env = auto_reset_wrapper(pytree_env)
        wrapped_env = metrics_wrapper(wrapped_env)

    # correct order should not raise an error
    wrapped_env = metrics_wrapper(pytree_env)
    wrapped_env = auto_reset_wrapper(wrapped_env)

