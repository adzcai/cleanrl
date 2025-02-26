"""Wrapper for multitask environments."""

import jax.numpy as jnp
from gymnax.environments import spaces
from jaxtyping import Array, Integer, Key

from log_util import dataclass
from wrappers.common import (
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
        obs, state = env.reset(params, key=key)
        goal = jnp.zeros((), int)
        obs = GoalObs(obs=obs, goal=goal)
        state = GoalState(_inner=state, goal=goal)
        return obs, state

    def step(
        state: GoalState[TEnvState], action: TAction, params: TEnvParams, *, key: Key[Array, ""]
    ) -> Timestep[GoalObs[TObs], GoalState[TEnvState]]:
        """Persist the goal throughout a rollout."""
        timestep = env.step(state._inner, action, params, key=key)
        # keep the goal
        obs = GoalObs(obs=timestep.obs, goal=state.goal)
        state = GoalState(_inner=timestep.state, goal=state.goal)
        return timestep._replace(obs=obs, state=state)

    def observation_space(params: TEnvParams):
        space = env.observation_space(params)
        return spaces.Tuple([space, spaces.Discrete(1)])

    return env.wrap(step=step, reset=reset, observation_space=observation_space)
