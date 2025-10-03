import dataclasses as dc
import math
from typing import Annotated

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from envs.base import Environment
from cleanrl_utils import specs
from cleanrl_utils.envs.utils import TAction, TArrayTree, TEnvParams, TEnvState
from cleanrl_utils.envs.wrap import Wrapper


def _flatten(x: TArrayTree) -> TArrayTree:
    """Flatten and preserve scalars."""
    return jax.tree.map(lambda x: x if jnp.ndim(x) == 0 else x.ravel(), x)


def flatten_observation_wrapper(
    env: Environment[TArrayTree, TEnvState, TAction, TEnvParams],
) -> Environment[Annotated[TArrayTree, "flat"], TEnvState, TAction, TEnvParams]:
    """Flatten observations."""

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.reset(params, key=key)
        timestep.obs = _flatten(timestep.obs)
        return timestep

    def step(state: TEnvState, action: TAction, params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.step(state, action, params, key=key)
        timestep.obs = _flatten(timestep.obs)
        return timestep

    def observation_space(params: TEnvParams) -> specs.SpecTree:
        """Flatten the shape of each array in the observation."""
        obs_spec = env.observation_space(params)
        return jax.tree.map(
            lambda spec: (
                dc.replace(spec, shape=(math.prod(spec.shape),))
                if isinstance(spec, specs.Array) and len(spec.shape) > 0
                else spec
            ),
            obs_spec,
            is_leaf=lambda x: isinstance(x, specs.Array),
        )

    return Wrapper.overwrite(
        env,
        name="flatten_observation",
        reset=reset,
        step=step,
        observation_space=observation_space,
    )
