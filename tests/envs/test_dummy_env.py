import jax
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt

from testing.dummy_env import simple_rollout
from utils.structures import SENTINEL, StepType, TimeStep


def test_dummy_env(dummy_env_params):
    env, params = dummy_env_params
    horizon = params.max_horizon
    ts_s = simple_rollout(env, params, None, horizon, key=jr.key(0))
    del env, params

    npt.assert_array_equal(ts_s.reward, jnp.array((SENTINEL,) + (1.0,) * horizon))
    npt.assert_array_equal(ts_s.state, jnp.arange(horizon + 1))
    npt.assert_array_equal(
        ts_s.obs,
        jnp.arange(horizon + 1, dtype=float)[:, jnp.newaxis],
    )
    npt.assert_array_equal(
        ts_s.discount,
        jnp.array((SENTINEL,) + (1.0,) * (horizon - 1) + (0,)),
    )
    npt.assert_array_equal(
        ts_s.step_type,
        jnp.array(
            (StepType.FIRST,) + (StepType.MID,) * (horizon - 1) + (StepType.LAST,)
        ),
    )
