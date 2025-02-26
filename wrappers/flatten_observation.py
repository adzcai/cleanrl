from typing import Annotated

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from wrappers.common import Environment, TAction, TEnvParams, TEnvState, TObs


def _flatten(x: Array):
    """Flatten and preserve scalars."""
    return jax.tree.map(lambda x: x if jnp.ndim(x) == 0 else x.ravel(), x)


def flatten_observation_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[Annotated[TObs, "flat"], TEnvState, TAction, TEnvParams]:
    """Flatten observations."""

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        obs, state = env.reset(params, key=key)
        return _flatten(obs), state

    def step(state: TEnvState, action: TAction, params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.step(state, action, params, key=key)
        return timestep._replace(obs=_flatten(timestep.obs))

    return env.wrap(reset=reset, step=step)
