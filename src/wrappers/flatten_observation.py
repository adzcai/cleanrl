import math
from typing import Annotated, TypeVar

import dm_env.specs as specs
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from utils.structures import Environment, TAction, TEnvParams, TEnvState, Wrapper


def _flatten(x: Array):
    """Flatten and preserve scalars."""
    return jax.tree.map(lambda x: x if jnp.ndim(x) == 0 else x.ravel(), x)


TObs = TypeVar("TObs", bound=PyTree[Array])


def flatten_observation_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[Annotated[TObs, "flat"], TEnvState, TAction, TEnvParams]:
    """Flatten observations."""

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.reset(params, key=key)
        timestep.obs = _flatten(timestep.obs)
        return timestep

    def step(
        state: TEnvState, action: TAction, params: TEnvParams, *, key: Key[Array, ""]
    ):
        timestep = env.step(state, action, params, key=key)
        timestep.obs = _flatten(timestep.obs)
        return timestep

    def observation_space(params: TEnvParams) -> specs.Array:
        """Flatten the shape of each array in the observation."""
        obs_spec = env.observation_space(params)
        return jax.tree.map(
            lambda spec: (
                spec.replace(shape=(math.prod(spec.shape),))
                if isinstance(spec, specs.Array) and len(spec.shape) > 0
                else spec
            ),
            obs_spec,
        )

    return Wrapper.overwrite(
        env,
        name="flatten_observation",
        reset=reset,
        step=step,
        observation_space=observation_space,
    )
