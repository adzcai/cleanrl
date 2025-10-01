import dataclasses as dc
from typing import Generic

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from envs.base import Environment
from utils.structures import (
    TEnvParams,
    TEnvState,
    TObs,
    dataclass,
)
from wrappers.base import Wrapper


@dataclass
class OAR(Generic[TObs]):
    """Observation that includes the last action and reward."""

    obs: TObs
    action: Integer[Array, ""]
    reward: Float[Array, ""]


def oar_wrapper(
    env: Environment[TObs, TEnvState, Integer[Array, ""], TEnvParams],
) -> Environment[OAR[TObs], TEnvState, Integer[Array, ""], TEnvParams]:
    """Return last action and reward in the observation."""

    def reset_fn(params: TEnvParams, *, key) -> TEnvState:
        ts = env.reset(params, key=key)
        obs = OAR(
            obs=ts.obs,
            action=jnp.array(0),  # No last action on reset
            reward=jnp.array(0.0),  # No reward on reset
        )
        return dc.replace(ts, obs=obs)  # type: ignore

    def step_fn(
        env_state: TEnvState,
        action: Integer[Array, ""],
        params: TEnvParams,
        *,
        key=None,
    ) -> TEnvState:
        ts = env.step(env_state, action, params, key=key)
        # Update the observation to include last action and reward
        obs = OAR(
            obs=ts.obs,
            action=action,
            reward=ts.reward,
        )
        return dc.replace(ts, obs=obs)  # type: ignore

    return Wrapper.overwrite(env, reset=reset_fn, step=step_fn)
