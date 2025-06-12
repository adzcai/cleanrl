import dataclasses as dc

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Key

from envs.base import Environment
from utils.structures import (
    TAction,
    TEnvParams,
    TEnvState,
    TimeStep,
    TObs,
    dataclass,
)
from wrappers.base import Wrapper


@dataclass
class Metrics:
    """Stores episode length and return.

    Properties:
        cum_return (Float ()): Cumulative undiscounted return
            including the reward obtained upon entering this timestep.
        step (Integer ()): Number of steps taken before this timestep in the current episode.
    """

    cum_return: Float[Array, ""]
    step: Integer[Array, ""]


@dataclass
class LogState(Wrapper[TEnvState]):
    metrics: Metrics


def metrics_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
) -> Environment[TObs, LogState[TEnvState], TAction, TEnvParams]:
    """Log interactions and episode rewards.

    Ensure this goes inside any auto reset wrappers
    """

    env_inner = env
    while getattr(env_inner, "_inner", None) is not None:
        if env_inner.name == "auto_reset":
            raise ValueError(
                "Metrics wrapper should be applied before auto_reset wrapper."
            )
        env_inner = env_inner._inner  # type: ignore

    init_metrics = Metrics(
        cum_return=jnp.zeros((), float),
        step=jnp.zeros((), int),
    )

    def reset(
        params: TEnvParams, *, key: Key[Array, ""]
    ) -> TimeStep[TObs, LogState[TEnvState]]:
        time_step = env.reset(params, key=key)
        return dc.replace(
            time_step,
            state=LogState(_inner=time_step.state, metrics=init_metrics),
        )  # type: ignore

    def step(
        state: LogState[TEnvState],
        action: TAction,
        params: TEnvParams,
        *,
        key: Key[Array, ""],
    ) -> TimeStep[TObs, LogState[TEnvState]]:
        time_step = env.step(state._inner, action, params, key=key)
        metrics = jax.tree.map(
            lambda init, updated: jnp.where(time_step.is_first, init, updated),
            init_metrics,
            Metrics(
                cum_return=state.metrics.cum_return + time_step.reward,
                step=state.metrics.step + 1,
            ),
        )
        return dc.replace(
            time_step,
            state=LogState(_inner=time_step.state, metrics=metrics),
        )  # type: ignore

    return Wrapper.overwrite(env, name="log", reset=reset, step=step)
