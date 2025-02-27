"""Base environment API.

See gymnax for converting between different environment APIs:
https://github.com/RobertTLange/gymnax/tree/main/gymnax/wrappers
This API is inspired by dm_env:
https://github.com/google-deepmind/dm_env/blob/master/docs/index.md
"""

import dataclasses as dc
from collections.abc import Callable
from enum import IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import jax.numpy as jnp
from dm_env.specs import Array as ArraySpec
from jaxtyping import Array, Float, Integer, Key, PyTree

from log_util import T, dataclass

P = ParamSpec("P")
TEnvState = TypeVar("TEnvState")
TObs = TypeVar("TObs")
TAction = TypeVar("TAction")
TEnvParams = TypeVar("TEnvParams")


class StepType(IntEnum):
    FIRST = auto()
    """This state was returned by `reset`."""
    MID = auto()
    """A standard environment transition."""
    LAST = auto()
    """Final state of an episode (not necessarily terminal)."""


@dataclass
class Timestep(Generic[TObs, TEnvState]):
    state: TEnvState
    """The whole state of the environment."""
    obs: TObs
    """The observation of `state`."""
    reward: Float[Array, ""]
    """The reward emitted upon entering `state` (i.e. from acting in the previous state). `nan` for initial states."""
    discount: Float[Array, ""]
    """The discount between `reward` and the value of `state`. `nan` for initial states."""
    step_type: Annotated[Integer[Array, ""], StepType]
    """The `StepType` of `state`."""
    info: dict[str, Any]
    """Additional information."""

    @classmethod
    def initial(cls, obs: TObs, state: TEnvState, info: dict[str, Any]):
        """Construct an initial timestep."""
        return cls(
            obs=obs,
            state=state,
            reward=jnp.asarray(0.0),  # TODO should be None
            discount=jnp.asarray(0.0),  # same
            step_type=jnp.asarray(StepType.FIRST),
            info=info,
        )

    @property
    def first(self):
        return self.step_type == StepType.FIRST

    @property
    def mid(self):
        return self.step_type == StepType.MID

    @property
    def last(self):
        return self.step_type == StepType.LAST


@dataclass
class Wrapper(Generic[T]):
    """Base dataclass for assigning additional properties to an object.

    Delegates property lookups to the inner object."""

    _inner: T | None

    def wrap(self, **kwargs):
        return dc.replace(self, **kwargs, _inner=self)

    if not TYPE_CHECKING:  # better static type hints

        def __getattr__(self, name):
            return getattr(self._inner, name)


@runtime_checkable
class ResetFn(Protocol[TObs, TEnvState, TEnvParams]):
    def __call__(self, params: TEnvParams, *, key: Key[Array, ""]) -> Timestep[TObs, TEnvState]:
        """Sample a new state."""


@runtime_checkable
class StepFn(Protocol[TObs, TEnvState, TAction, TEnvParams]):
    def __call__(
        self, env_state: TEnvState, action: TAction, params: TEnvParams, *, key: Key[Array, ""]
    ) -> Timestep[TObs, TEnvState]:
        """Step the environment. See `Timestep`."""


@dataclass
class Environment(Wrapper["Environment"], Generic[TObs, TEnvState, TAction, TEnvParams]):
    """An interface for an interactive environment.

    We allow the action, observation, and goal spaces to vary.
    """

    name: str
    reset: ResetFn[TObs, TEnvState, TEnvParams]
    step: StepFn[TObs, TEnvState, TAction, TEnvParams]
    action_space: Callable[[TEnvParams], PyTree[ArraySpec]]
    observation_space: Callable[[TEnvParams], PyTree[ArraySpec]]
    goal_space: Callable[[TEnvParams], PyTree[ArraySpec]]

    @property
    def fullname(self):
        """Add all the wrapper names."""
        env = self
        name = self.name
        while getattr(env, "_inner", None) is not None:
            env = env._inner
            name = env.name + " > " + name
        return name


class GoalObs(NamedTuple, Generic[TObs]):
    obs: TObs
    """The original observation."""
    goal: Integer[Array, ""]
    """The goal (for multitask environments)."""
