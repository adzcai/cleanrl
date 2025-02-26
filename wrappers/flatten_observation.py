from typing import Annotated

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from wrappers.base import Environment, StepType, TAction, TEnvParams, TEnvState, Timestep, TObs


def _flatten(x: Array):
    """Flatten and preserve scalars."""
    return jax.tree.map(lambda x: x if jnp.ndim(x) == 0 else x.ravel(), x)


def flatten_observation_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[Annotated[TObs, "flat"], TEnvState, TAction, TEnvParams]:
    """Flatten observations."""

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.reset(params, key=key)
        timestep.obs = _flatten(timestep.obs)
        return timestep

    def step(state: TEnvState, action: TAction, params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.step(state, action, params, key=key)
        timestep.obs = _flatten(timestep.obs)
        return timestep

    return env.wrap(name="flatten_observation", reset=reset, step=step)
