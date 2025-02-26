"""Wrapper for multitask environments."""

import jax.numpy as jnp
from dm_env.specs import DiscreteArray as DiscreteArraySpec
from jaxtyping import Array, Integer, Key

from log_util import dataclass
from wrappers.base import (
    Environment,
    GoalObs,
    TAction,
    TEnvParams,
    TEnvState,
    Timestep,
    TObs,
    Wrapper,
)


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

    def reset(params: TEnvParams, *, key: Key[Array, ""]):
        timestep = env.reset(params, key=key)
        goal = jnp.zeros((), int)
        timestep.obs = GoalObs(obs=timestep.obs, goal=goal)
        timestep.state = GoalState(_inner=timestep.state, goal=goal)
        return timestep

    def step(
        state: GoalState[TEnvState], action: TAction, params: TEnvParams, *, key: Key[Array, ""]
    ) -> Timestep[GoalObs[TObs], GoalState[TEnvState]]:
        """Persist the goal throughout a rollout."""
        timestep = env.step(state._inner, action, params, key=key)
        # keep the goal
        timestep.obs = GoalObs(obs=timestep.obs, goal=state.goal)
        timestep.state = GoalState(_inner=timestep.state, goal=state.goal)
        return timestep

    def observation_space(params: TEnvParams):
        space = env.observation_space(params)
        return [space, DiscreteArraySpec(1, name="goal")]

    return env.wrap(name="goal", step=step, reset=reset, observation_space=observation_space)
