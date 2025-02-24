import dataclasses as dc
import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from wrappers.common import Environment, TAction, TEnvParams, TEnvState, TObs


def auto_reset_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[TObs, TEnvState, TAction, TEnvParams]:
    """Automatically reset the environment once a terminal state is reached.

    Consider an (obs, env_state, reward, done, info).
    Under this wrapper, if done is true,
    obs and env_state correspond to an initial environment.
    We skip the terminal state that was transitioned into.
    """

    def step(key: Key[Array, ""], env_state: TEnvState, action: TAction, params: TEnvParams):
        """Auto-reset.

        If env.step returns a terminal state (done == True),
        return an initial obs, state from env.reset.
        """
        key_reset, key_step = jr.split(key)
        obs_reset, state_reset = env.reset(key_reset, params)
        timestep = env.step(key_step, env_state, action, params)
        obs, env_state = jax.tree.map(
            ft.partial(jnp.where, timestep.done),
            (obs_reset, state_reset),
            (timestep.obs, timestep.env_state),
        )
        return timestep._replace(obs=obs, env_state=env_state)

    return dc.replace(env, step=step)
