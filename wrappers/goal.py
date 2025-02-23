"""Wrapper for multitask environments."""

import jax.numpy as jnp
from wrappers.common import TObs, TBaseState, Timestep, Wrapper, WrapperState
import gymnax.environments.environment as gymenv
from gymnax.environments import spaces
from typing import TYPE_CHECKING, Generic, NamedTuple
from jaxtyping import Real, UInt, Key, Array

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


class GoalObs(NamedTuple, Generic[TObs]):
    obs: TObs
    goal: UInt[Array, ""]


@dataclass(frozen=True)
class GoalState(WrapperState[TBaseState]):
    goal: UInt[Array, ""]


class GoalWrapper(Wrapper[GoalObs[TObs], TBaseState, GoalState[TBaseState], gymenv.TEnvParams]):
    """Return an observation that contains the current goal.

    The goal is set upon environment reset.
    """

    num_goals: UInt[Array, ""]

    def __init__(self, env: gymenv.Environment[TBaseState, gymenv.TEnvParams], num_goals: int):
        super().__init__(env)
        self.num_goals = jnp.asarray(num_goals, jnp.uint32)

    def step(
        self,
        key: Key[Array, ""],
        state: GoalState[TBaseState],
        action: int | float | Real[Array, ""],
        params: gymenv.TEnvParams,
    ) -> Timestep[GoalObs[TObs], GoalState[TBaseState]]:
        """Persist the goal throughout a rollout."""
        timestep = super().step(key, state, action, params)
        # keep the goal
        obs = GoalObs(obs=timestep.obs, goal=state.goal)
        env_state = GoalState(_env_state=timestep.env_state, goal=state.goal)
        return timestep._replace(obs=obs, env_state=env_state)

    def reset(self, key: Key[Array, ""], params: gymenv.TEnvParams):
        obs, env_state = super().reset(key, params)
        goal = self.get_goal(env_state)
        obs = GoalObs(obs=obs, goal=goal)
        env_state = GoalState(_env_state=env_state, goal=goal)
        return obs, env_state

    def observation_space(self, params: gymenv.TEnvParams):
        space = super().observation_space(params)
        return spaces.Tuple([space, spaces.Discrete(self.num_goals)])

    def get_goal(self, env_state: TBaseState):
        return jnp.zeros((), jnp.uint32)
