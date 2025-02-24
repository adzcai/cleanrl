"""Wrapper for multitask environments."""

import dataclasses as dc
from typing import TYPE_CHECKING

import jax.numpy as jnp
from gymnax.environments import spaces
from jaxtyping import Array, Key, UInt

from wrappers.common import (
    Environment,
    GoalObs,
    TAction,
    TEnvParams,
    TEnvState,
    Timestep,
    TObs,
    WrapperState,
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass(frozen=True)
class GoalState(WrapperState[TEnvState]):
    """Environment state wrapper that adds an integer goal."""

    goal: UInt[Array, ""]


def goal_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[GoalObs[TObs], GoalState[TEnvState], TAction, TEnvParams]:
    """Turn a non-multi-task environment into a pseudo-multitask environment.

    Sets there to be a singleton goal.
    """

    def reset(key: Key[Array, ""], params: TEnvParams):
        obs, env_state = env.reset(key, params)
        goal = jnp.zeros((), jnp.uint32)
        obs = GoalObs(obs=obs, goal=goal)
        env_state = GoalState(_env_state=env_state, goal=goal)
        return obs, env_state

    def step(
        key: Key[Array, ""],
        state: GoalState[TEnvState],
        action: TAction,
        params: TEnvParams,
    ) -> Timestep[GoalObs[TObs], GoalState[TEnvState]]:
        """Persist the goal throughout a rollout."""
        timestep = env.step(key, state._env_state, action, params)
        # keep the goal
        obs = GoalObs(obs=timestep.obs, goal=state.goal)
        env_state = GoalState(_env_state=timestep.env_state, goal=state.goal)
        return timestep._replace(obs=obs, env_state=env_state)

    def observation_space(params: TEnvParams):
        space = env.observation_space(params)
        return spaces.Tuple([space, spaces.Discrete(1)])

    return dc.replace(env, step=step, reset=reset, observation_space=observation_space)
