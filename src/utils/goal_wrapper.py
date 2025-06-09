"""Wrapper for multitask environments."""

import dataclasses as dc

import jax.numpy as jnp
from dm_env.specs import DiscreteArray as DiscreteArraySpec
from jaxtyping import Array, Integer, Key

from utils.base import (
    Environment,
    GoalObs,
    TAction,
    TEnvParams,
    TEnvState,
    Timestep,
    TObs,
    Wrapper,
)
from utils.log_util import dataclass


@dataclass
class GoalState(Wrapper[TEnvState]):
    """Environment state wrapper that adds an integer goal."""

    goal: Integer[Array, ""]


def goal_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[GoalObs[TObs], GoalState[TEnvState], TAction, TEnvParams]:
    """Turn a non-multi-task environment into a pseudo-multitask environment.

    Sets there to be a singleton goal.
    """

    def reset(
        params: TEnvParams, *, key: Key[Array, ""]
    ) -> Timestep[GoalObs[TObs], GoalState[TEnvState]]:
        timestep = env.reset(params, key=key)
        goal = jnp.zeros((), int)
        return dc.replace(
            timestep,
            obs=GoalObs(obs=timestep.obs, goal=goal),
            state=GoalState(_inner=timestep.state, goal=goal),
        )  # type: ignore

    def step(
        state: GoalState[TEnvState],
        action: TAction,
        params: TEnvParams,
        *,
        key: Key[Array, ""],
    ) -> Timestep[Array, GoalState[TEnvState]]:
        timestep = env.step(state._inner, action, params, key=key)
        return dc.replace(
            timestep,
            obs=GoalObs(obs=timestep.obs, goal=state.goal),
            state=GoalState(_inner=timestep.state, goal=state.goal),
        )  # type: ignore

    return Wrapper.overwrite(
        env,
        name="goal_wrapper",
        step=step,
        reset=reset,
        goal_space=lambda params: DiscreteArraySpec(1, name="goal"),
    )
