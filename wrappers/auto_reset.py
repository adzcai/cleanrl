import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Key

from log_util import dataclass
from wrappers.base import (
    Environment,
    TAction,
    TEnvParams,
    TEnvState,
    Timestep,
    TObs,
    Wrapper,
)


@dataclass
class PrevDone(Wrapper[TEnvState]):
    """We follow the deepmind convention of returning terminal state"""

    prev_done: Bool[Array, ""]


def auto_reset_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[TObs, PrevDone[TEnvState], TAction, TEnvParams]:
    """Automatically reset the environment after an episode."""

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.reset(params, key=key)
        timestep.state = PrevDone(_inner=timestep.state, prev_done=jnp.asarray(False))
        return timestep

    def step(
        state: PrevDone[TEnvState],
        action: TAction,
        params: TEnvParams,
        *,
        key: Key[Array, ""],
    ):
        key_reset, key_step = jr.split(key)
        timestep_reset = env.reset(params, key=key_reset)
        timestep_step = env.step(state._inner, action, params, key=key_step)
        timestep: Timestep[TObs, TEnvState] = jax.tree.map(
            ft.partial(jnp.where, state.prev_done),
            timestep_reset,
            timestep_step,
        )
        timestep.state = PrevDone(_inner=timestep.state, prev_done=timestep.last)
        return timestep

    return env.wrap(name="auto_reset", reset=reset, step=step)
