"""Wrapper to automatically reset the environment after an episode."""

import dataclasses as dc
import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from envs.base import Environment
from utils.structures import (
    TAction,
    TDataclass,
    TEnvParams,
    TimeStep,
    TObs,
    dataclass,
)
from wrappers.base import Wrapper


@dataclass
class PrevDone(Wrapper[TDataclass]):
    """We follow the dm_env convention of returning terminal states.

    Note that the TimeStep object has an is_last property
    but not necessarily each environment state.
    """

    is_last: Bool[Array, ""]


def auto_reset_wrapper(
    env: Environment[TObs, TDataclass, TAction, TEnvParams],
) -> Environment[TObs, PrevDone[TDataclass], TAction, TEnvParams]:
    """Automatically reset the environment after an episode.

    The env_state for the wrapped class must be a dataclass.
    """

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.reset(params, key=key)
        return dc.replace(
            timestep,
            state=PrevDone(_inner=timestep.state, is_last=jnp.bool_(False)),
        )

    def step(
        env_state: PrevDone[TDataclass],
        action: TAction,
        params: TEnvParams,
        *,
        key: Key[Array, ""],
    ):
        key_reset, key_step = jr.split(key)
        timestep_reset = env.reset(params, key=key_reset)
        timestep_step = env.step(env_state._inner, action, params, key=key_step)
        timestep: TimeStep[TObs, TDataclass] = jax.tree.map(
            ft.partial(jnp.where, env_state.is_last),
            timestep_reset,
            timestep_step,
        )
        return dc.replace(timestep, state=PrevDone(_inner=timestep.state, is_last=timestep.is_last))

    return Wrapper.overwrite(env, name="auto_reset", reset=reset, step=step)
