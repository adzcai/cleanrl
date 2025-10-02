"""Base environment API.

See gymnax for converting between different environment APIs:
https://github.com/RobertTLange/gymnax/tree/main/gymnax/wrappers
This API is inspired by dm_env:
https://github.com/google-deepmind/dm_env/blob/master/docs/index.md
"""

import functools as ft
from collections.abc import Collection
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
from jaxtyping import Array, Float, Integer, Key

from cleanrl_utils.log_utils import typecheck

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from chex import dataclass

else:
    from typing import Any as DataclassInstance

    if True:
        from chex import dataclass as dataclass
    else:
        from chex import dataclass as _dataclass

        def dataclass(cls=None, /, **kwargs):
            """Typecheck all dataclass fields."""
            if cls is None:
                return ft.partial(dataclass, **kwargs)
            return typecheck(_dataclass(cls, **kwargs))


P = ParamSpec("P")
ArrayTree = Float[Array, " ..."] | Collection["ArrayTree"] | DataclassInstance
TDataclass = TypeVar("TDataclass", bound=DataclassInstance)
TArrayTree = TypeVar("TArrayTree", bound=ArrayTree)
TEnvState = TypeVar("TEnvState", bound=ArrayTree)
TObs = TypeVar("TObs")
TAction = TypeVar("TAction", contravariant=True)
TEnvParams = TypeVar("TEnvParams", bound=ArrayTree | None, contravariant=True)


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


@runtime_checkable
class ResetFn(Protocol[TObs, TEnvState, TEnvParams]):
    def __call__(self, params: TEnvParams, *, key: Key[Array, ""]) -> TimeStep[TObs, TEnvState]:
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
class GoalObs(Generic[TObs]):
    obs: TObs
    """The original observation."""
    goal: Integer[Array, ""]
    """The goal (for multitask environments)."""


@dataclass
class Transition(Generic[TObs, TEnvState]):
    """A single transition. May be batched into a trajectory."""

    time_step: TimeStep[TObs, TEnvState]
    """The timestep that was acted in."""
    action: Integer[Array, ""]
    """The action taken from `timestep`."""
    pred: Prediction
    """The prediction of the value of `timestep`."""
    mcts_probs: Float[Array, " num_actions"]
    """The MCTS action probability distribution from acting in timestep."""
