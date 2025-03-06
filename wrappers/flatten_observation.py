import math
from typing import Annotated

import dm_env.specs as specs
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from wrappers.base import Environment, TAction, TEnvParams, TEnvState, TObs


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

    def observation_space(params: TEnvParams) -> specs.Array:
        obs_spec = env.observation_space(params)
        assert isinstance(obs_spec, specs.BoundedArray), (
            f"Base environment observation space must be BoundedArray. Got {obs_spec}"
        )
        return specs.BoundedArray(
            shape=(math.prod(obs_spec.shape),),
            dtype=obs_spec.dtype,
            minimum=obs_spec.minimum,
            maximum=obs_spec.maximum,
            name=obs_spec.name,
        )

    return env.wrap(
        name="flatten_observation",
        reset=reset,
        step=step,
        observation_space=observation_space,
    )
