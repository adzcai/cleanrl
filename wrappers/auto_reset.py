import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from wrappers.common import Environment, TAction, TEnvParams, TEnvState, TObs


def auto_reset_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[TObs, TEnvState, TAction, TEnvParams]:
    """Automatically reset the environment after an episode.

    If env.step returns a terminal state (done == True),
    return an initial obs, state from env.reset.
    Note that this means we never observe the actual terminal observation.
    See `StepType`.
    """

    def step(state: TEnvState, action: TAction, params: TEnvParams, *, key: Key[Array, ""]):
        key_reset, key_step = jr.split(key)
        obs_reset, state_reset = env.reset(params, key=key_reset)
        timestep = env.step(state, action, params, key=key_step)
        obs, state = jax.tree.map(
            ft.partial(jnp.where, timestep.done),
            (obs_reset, state_reset),
            (timestep.obs, timestep.state),
        )
        return timestep._replace(obs=obs, state=state)

    return env.wrap(step=step)
