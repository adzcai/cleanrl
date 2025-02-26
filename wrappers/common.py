"""Base environment API.

See gymnax for converting between different environment APIs:
https://github.com/RobertTLange/gymnax/tree/main/gymnax/wrappers
This API is inspired by navix:
https://github.com/epignatelli/navix/blob/main/navix/environments/environment.py
"""

import dataclasses as dc
from collections.abc import Callable
from enum import IntEnum
from typing import (Any, Generic, NamedTuple, ParamSpec, Protocol, TypeVar,
                    runtime_checkable)

import gymnax.environments.spaces as spaces
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Integer, Key, jaxtyped

from log_util import T, dataclass

P = ParamSpec("P")
TEnvState = TypeVar("TEnvState")
TObs = TypeVar("TObs")
TAction = TypeVar("TAction")
TEnvParams = TypeVar("TEnvParams")


class StepType(IntEnum):
    TRANSITION = 0
    """A standard environment transition."""
    TRUNCATION = 1
    """Time limit reached (this state is not terminal)."""
    TERMINATION = 2
    """This state is terminal."""


class Timestep(NamedTuple, Generic[TObs, TEnvState]):
    obs: TObs
    """The observation of `state`."""
    state: TEnvState
    """The whole state of the environment."""
    reward: Float[Array, ""]
    """The reward emitted upon entering `state` (i.e. from acting in the previous state)."""
    step_type: StepType
    """The `StepType`."""
    info: dict[str, Any]
    """Additional information."""

    @property
    def terminated(self):
        return self.step_type == StepType.TERMINATION

    @property
    def truncated(self):
        return self.step_type == StepType.TRUNCATION

    @property
    def done(self):
        return jnp.logical_or(self.terminated, self.truncated)


@dataclass
class Wrapper(Generic[T]):
    """Base dataclass for assigning additional properties to an object.

    Delegates property lookups to the inner object."""

    _inner: T | None

    @property
    def inner(self):
        """Return the wrapped object or self."""
        return self._inner if self._inner is not None else self

    def wrap(self, **kwargs):
        return dc.replace(self, **kwargs, _inner=self)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@runtime_checkable
class ResetFn(Protocol[TObs, TEnvState, TEnvParams]):
    def __call__(self, params: TEnvParams, *, key: Key[Array, ""]) -> tuple[TObs, TEnvState]:
        """Sample a new state."""


@runtime_checkable
class StepFn(Protocol[TObs, TEnvState, TAction, TEnvParams]):
    def __call__(
        self, env_state: TEnvState, action: TAction, params: TEnvParams, *, key: Key[Array, ""]
    ) -> Timestep[TObs, TEnvState]:
        """Step the environment. See `Timestep`."""


@jaxtyped(typechecker=typechecker)
@dataclass
class Environment(Wrapper["Environment"], Generic[TObs, TEnvState, TAction, TEnvParams]):
    """An interface for an interactive environment.

    We allow the action, observation, and goal spaces to vary.
    """

    reset: ResetFn[TObs, TEnvState, TEnvParams]
    step: StepFn[TObs, TEnvState, TAction, TEnvParams]
    action_space: Callable[[TEnvParams], spaces.Space]
    observation_space: Callable[[TEnvParams], spaces.Space]
    goal_space: Callable[[TEnvParams], spaces.Space]


class GoalObs(NamedTuple, Generic[TObs]):
    obs: TObs
    """The original observation."""
    goal: Integer[Array, ""]
    """The goal (for multitask environments)."""
