import dataclasses as dc
from typing import Annotated

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from wrappers.common import Environment, TAction, TEnvParams, TEnvState, TObs


def _flatten(x: Array):
    """Flatten and preserve scalars."""
    return jax.tree.map(lambda x: x if len(x.shape) == 0 else x.ravel(), x)


def flatten_observation_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[Annotated[TObs, "flat"], TEnvState, TAction, TEnvParams]:
    """Flatten observations."""

    def reset(key: Key[Array, ""], params: TEnvParams):
        obs, state = env.reset(key, params)
        return _flatten(obs), state

    def step(key: Key[Array, ""], env_state: TEnvState, action: TAction, params: TEnvParams):
        timestep = env.step(key, env_state, action, params)
        return timestep._replace(obs=_flatten(timestep.obs))

    return dc.replace(env, reset=reset, step=step)
