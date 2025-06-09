import dataclasses as dc
import functools as ft
from typing import TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Key

from utils.base import (
    Environment,
    TAction,
    TDataclass,
    Timestep,
    TObs,
    Wrapper,
)
from utils.log_util import dataclass

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    # redefine TEnvState and TEnvParams to be dataclasses
    TEnvState = TypeVar("TEnvState", bound="DataclassInstance")
    TEnvParams = TypeVar("TEnvParams", bound="DataclassInstance")
else:
    TEnvState = TypeVar("TDataclass")
    TEnvParams = TypeVar("TDataclass")


@dataclass
class Metrics:
    """Stores episode length and return.

    Properties:
        current_return (Float ()): Cumulative return of the current episode (undiscounted).
        current_length (Integer ()): Number of steps taken in current episode.
        episode_return (Float ()): Returned episode return on termination.
        episode_length (Integer ()): Returned episode length on termination.
    """

    current_return: Float[Array, ""]
    current_length: Integer[Array, ""]
    episode_return: Float[Array, ""]
    episode_length: Integer[Array, ""]


@dataclass
class LogState(Wrapper[TDataclass]):
    metrics: Metrics


def log_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[TObs, LogState[TEnvState], TAction, TEnvParams]:
    """Log interactions and episode rewards.

    env, env_params = log_wrapper(env, env_params)
    """

    init_metrics = Metrics(
        current_return=jnp.zeros((), float),
        current_length=jnp.zeros((), int),
        episode_return=jnp.zeros((), float),
        episode_length=jnp.zeros((), int),
    )

    def reset(
        params: TEnvParams, *, key: Key[Array, ""]
    ) -> Timestep[TObs, LogState[TEnvState]]:
        timestep = env.reset(params, key=key)
        return dc.replace(
            timestep,
            state=LogState(_inner=timestep.state, metrics=init_metrics),
            info=timestep.info | {"metrics": init_metrics},
        )  # type: ignore

    def step(
        state: LogState[TEnvState],
        action: TAction,
        params: TEnvParams,
        *,
        key: Key[Array, ""],
    ) -> Timestep[TObs, LogState[TEnvState]]:
        timestep = env.step(state._inner, action, params, key=key)
        updated_return = state.metrics.current_return + timestep.reward
        updated_length = state.metrics.current_length + 1
        done_metrics = Metrics(
            current_return=init_metrics.current_return,
            current_length=init_metrics.current_length,
            episode_return=updated_return,
            episode_length=updated_length,
        )
        continue_metrics = Metrics(
            current_return=updated_return,
            current_length=updated_length,
            episode_return=state.metrics.current_return,
            episode_length=state.metrics.episode_length,
        )
        metrics = jax.tree.map(
            ft.partial(jnp.where, timestep.is_last), done_metrics, continue_metrics
        )
        return dc.replace(
            timestep,
            state=LogState(_inner=timestep.state, metrics=metrics),
            info=timestep.info | {"metrics": metrics},
        )  # type: ignore

    return Wrapper.overwrite(env, name="log", reset=reset, step=step)
