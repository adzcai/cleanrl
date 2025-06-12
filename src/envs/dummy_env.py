"""A minimal dummy environment for testing wrappers and environment interfaces."""

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer, Key

from envs.base import Environment
from utils import specs
from utils.structures import (
    StepType,
    TAction,
    TEnvParams,
    TEnvState,
    TimeStep,
    TObs,
    dataclass,
)


@dataclass
class Params:
    max_horizon: int
    """Number of steps before reaching a terminal state."""


def dummy_reset(params: Params, *, key):
    return TimeStep.initial(
        obs=jnp.zeros((1,)),
        state=jnp.int_(0),
        info={},
    )


def dummy_step(env_state, action, params: Params, *, key=None):
    env_state = env_state + 1
    terminal = env_state >= params.max_horizon
    return TimeStep(
        obs=jnp.full((1,), env_state, dtype=float),
        state=env_state,
        reward=jnp.float_(1.0),
        discount=1.0 - terminal,
        step_type=jnp.where(terminal, StepType.LAST, StepType.MID),
        info={},
    )


DummyEnv = Environment[Float[Array, " 1"], Integer[Array, ""], None, Params]


def make_dummy_env(max_horizon: int) -> tuple[DummyEnv, Params]:
    return Environment(
        _inner=None,  # type: ignore
        name="dummy",
        reset=dummy_reset,
        step=dummy_step,
        action_space=lambda params: specs.BoundedArray.discrete(1, name="action"),
        observation_space=lambda params: specs.BoundedArray(
            shape=(1,),
            dtype=jnp.float_,
            minimum=0,
            maximum=params.max_horizon,
            name="observation",
        ),
        goal_space=lambda params: specs.BoundedArray.discrete(1, name="goal"),
    ), Params(max_horizon=max_horizon)


def simple_rollout(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
    params: TEnvParams,
    action: TAction,
    horizon: int,
    *,
    key: Key[Array, ""],
) -> TimeStep[TObs, TEnvState]:
    """For testing. Run a rollout that takes a constant action."""
    init_ts = env.reset(params, key=key)

    def step_fn(ts: TimeStep, key: Key[Array, ""]):
        next_ts = env.step(ts.state, action, params, key=key)
        return next_ts, next_ts

    _, ts_s = jax.lax.scan(step_fn, init_ts, jr.split(key, horizon))

    # append init_ts to the sequence
    ts_s = jax.tree.map(
        lambda init, xs: jnp.concatenate([init[jnp.newaxis, ...], xs], axis=0),
        init_ts,
        ts_s,
    )

    return ts_s
