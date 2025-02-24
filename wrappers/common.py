"""Base classes for gymnax environment wrappers."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

import gymnax.environments.spaces as spaces
from jaxtyping import Array, Bool, Float, Key, UInt
from typeguard import typechecked

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

TEnvState = TypeVar("TEnvState")
TObs = TypeVar("TObs")
TAction = TypeVar("TAction")
TEnvParams = TypeVar("TEnvParams")


class Timestep(NamedTuple, Generic[TObs, TEnvState]):
    """A single interaction timestep.

    Implemented as a NamedTuple to allow for destructuring:

        Timestep(*env.step(key, state, action, params))

    Attributes:
        obs (TObs): The observation of env_state.
        env_state (TEnvState): The updated state of the environment.
        reward (Float[Array, ""]): The reward received after taking an action.
        done (Bool ()): Indicates that the episode has ended.
        info (dict[str, Any]): Additional information from the environment.
    """

    obs: TObs
    env_state: TEnvState
    reward: Float[Array, ""]
    done: Bool[Array, ""]
    info: dict[str, Any]


@typechecked
@dataclass(frozen=True)
class Environment(Generic[TObs, TEnvState, TAction, TEnvParams]):
    """An interface for an interactive environment."""

    reset: Callable[[Key[Array, ""], TEnvParams], tuple[TObs, TEnvState]]
    step: Callable[[Key[Array, ""], TEnvState, TAction, TEnvParams], Timestep[TObs, TEnvState]]
    action_space: Callable[[TEnvParams], spaces.Space]
    observation_space: Callable[[TEnvParams], spaces.Space]
    goal_space: Callable[[TEnvParams], spaces.Space]


class GoalObs(NamedTuple, Generic[TObs]):
    """A tuple of the environment observation and the integer goal."""

    obs: TObs
    goal: UInt[Array, ""]


@dataclass(frozen=True)
class WrapperState(Generic[TEnvState]):
    """Base class for wrapping an existing environment's state.

    Delegates property lookups to the base state."""

    _env_state: TEnvState

    def __getattr__(self, name):
        return getattr(self._env_state, name)
