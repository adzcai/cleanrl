"""Wrapper to automatically reset the environment after an episode."""

import dataclasses as dc
import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from utils.log_utils import dataclass
from utils.structures import (
    Environment,
    TAction,
    TEnvParams,
    TEnvState,
    TimeStep,
    TObs,
    Wrapper,
)


@dataclass
class PrevDone(Wrapper[TEnvState]):
    """We follow the dm_env convention of returning terminal states."""

    is_last: Bool[Array, ""]


def auto_reset_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[TObs, PrevDone[TEnvState], TAction, TEnvParams]:
    """Automatically reset the environment after an episode."""

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.reset(params, key=key)
        return dc.replace(
            timestep,
            state=PrevDone(_inner=timestep.state, is_last=jnp.bool_(False)),
        )

    def step(
        env_state: PrevDone[TEnvState],
        action: TAction,
        params: TEnvParams,
        *,
        key: Key[Array, ""],
    ):
        key_reset, key_step = jr.split(key)
        timestep_reset = env.reset(params, key=key_reset)
        timestep_step = env.step(env_state._inner, action, params, key=key_step)
        timestep: TimeStep[TObs, TEnvState] = jax.tree.map(
            ft.partial(jnp.where, env_state.is_last),
            timestep_reset,
            timestep_step,
        )
        return dc.replace(timestep, state=PrevDone(_inner=timestep.state, is_last=timestep.is_last))

    return Wrapper.overwrite(env, name="auto_reset", reset=reset, step=step)
