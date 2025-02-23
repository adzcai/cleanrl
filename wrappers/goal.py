"""Wrapper for multitask environments."""

import jax.numpy as jnp
from wrappers.common import TObs, TBaseState, Timestep, Wrapper, WrapperState
import gymnax.environments.environment as gymenv
from typing import TYPE_CHECKING
from jaxtyping import Real, UInt, Key, Array

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass(frozen=True)
class GoalState(WrapperState[TBaseState]):
    goal: UInt[Array, ""]


class GoalWrapper(Wrapper[TObs, TBaseState, GoalState[TBaseState], gymenv.TEnvParams]):
    """Return an observation that contains the current goal."""

    def step(
        self,
        key: Key[Array, ""],
        state: GoalState[TBaseState],
        action: int | float | Real[Array, ""],
        params: gymenv.TEnvParams,
    ) -> Timestep[TObs, GoalState[TBaseState]]:
        timestep = super().step(key, state, action, params)
        # keep the goal
        env_state = GoalState(_env_state=timestep.env_state, goal=state.goal)
        return timestep._replace(env_state=env_state, info=timestep.info | {"goal": state.goal})

    def reset(self, key: Key[Array, ""], params: gymenv.TEnvParams):
        obs, env_state = super().reset(key, params)
        env_state = GoalState(_env_state=env_state, goal=self.get_goal(env_state))
        return obs, env_state

    def get_goal(self, env_state: TBaseState):
        return jnp.zeros((), jnp.uint32)
