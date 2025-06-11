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

from utils.log_utils import dataclass

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    TDataclass = TypeVar("TDataclass", bound="DataclassInstance")
else:
    TDataclass = TypeVar("TDataclass")


try:
    import gymnax

    GYMNAX_INSTALLED = True
except ImportError:
    GYMNAX_INSTALLED = False

try:
    import navix

    NAVIX_INSTALLED = True
except ImportError:
    NAVIX_INSTALLED = False

P = ParamSpec("P")
TEnvState = TypeVar("TEnvState")
TObs = TypeVar("TObs")
TAction = TypeVar("TAction", contravariant=True)
TEnvParams = TypeVar("TEnvParams", contravariant=True)


class StepType(IntEnum):
    FIRST = auto()
    """This state was returned by `reset`."""
    MID = auto()
    """A standard environment transition."""
    LAST = auto()
    """Final state of an episode (not necessarily terminal)."""


class Prediction(NamedTuple):
    policy_logits: Float[Array, " num_actions"]
    value_logits: Float[Array, " num_value_bins"]


SENTINEL = -1 << 30  # sentinel for initial reward and discount


@dataclass
class TimeStep(Generic[TObs, TEnvState]):
    reward: Float[Array, ""]
    """The reward emitted upon entering `state` (i.e. from acting in the previous state). `nan` for initial states."""
    state: TEnvState
    """The whole state of the environment."""
    obs: TObs
    """The observation of `state`."""
    discount: Float[Array, ""]
    """The discount between `reward` and the value of `state`. `nan` for initial states."""
    step_type: Annotated[Integer[Array, ""], StepType]
    """The `StepType` of `state`."""
    info: dict[str, Any]
    """Additional information."""

    @classmethod
    def initial(cls, obs: TObs, state: TEnvState, info: dict[str, Any]):
        """Construct an initial timestep."""
        # insert a sentinel for reward and discount
        return cls(
            reward=jnp.array(SENTINEL, dtype=float),
            state=state,
            obs=obs,
            discount=jnp.array(SENTINEL, dtype=float),
            step_type=jnp.array(StepType.FIRST, dtype=int),
            info=info,
        )

    @property
    def is_first(self):
        return self.step_type == StepType.FIRST

    @property
    def is_mid(self):
        return self.step_type == StepType.MID

    @property
    def is_last(self):
        return self.step_type == StepType.LAST


@dataclass
class Wrapper(Generic[TDataclass]):
    """Base dataclass for assigning additional properties to an object.

    Delegates property lookups to the inner object."""

    _inner: TDataclass

    @classmethod
    def overwrite(cls, obj, **kwargs):
        """Replace the properties of the inner object"""
        return dc.replace(obj, **kwargs, _inner=obj)

    if not TYPE_CHECKING:  # better static type hints

        def __getattr__(self, name):
            return getattr(self._inner, name)


@runtime_checkable
class ResetFn(Protocol[TObs, TEnvState, TEnvParams]):
    def __call__(
        self, params: TEnvParams, *, key: Key[Array, ""]
    ) -> TimeStep[TObs, TEnvState]:
        """Sample a new state."""
        raise NotImplementedError(
            "ResetFn must be implemented by the environment. "
            "It should return an initial Timestep with the state, observation, and info."
        )


@runtime_checkable
class StepFn(Protocol[TObs, TEnvState, TAction, TEnvParams]):
    def __call__(
        self,
        env_state: TEnvState,
        action: TAction,
        params: TEnvParams,
        *,
        key: Key[Array, ""] | None = None,
    ) -> TimeStep[TObs, TEnvState]:
        """Step the environment. See `Timestep`."""
        raise NotImplementedError(
            "StepFn must be implemented by the environment. "
            "It should return a Timestep with the new state, observation, and info."
        )


@dataclass
class Environment(
    Wrapper["Environment"], Generic[TObs, TEnvState, TAction, TEnvParams]
):
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


class Transition(NamedTuple, Generic[TObs, TEnvState]):
    """A single transition. May be batched into a trajectory."""

    time_step: TimeStep[TObs, TEnvState]
    """The timestep that was acted in."""
    action: Integer[Array, ""]
    """The action taken from `timestep`."""
    pred: Prediction
    """The prediction of the value of `timestep`."""
    mcts_probs: Float[Array, " num_actions"]
    """The MCTS action probability distribution from acting in timestep."""
