"""A minimal dummy environment for testing wrappers and environment interfaces."""

import dm_env.specs as specs
import jax.numpy as jnp
from jaxtyping import Array, Integer

from utils.structures import Environment, StepType, Timestep
from utils.log_utils import dataclass


@dataclass
class Params:
    max_horizon: int
    """Number of steps before reaching a terminal state."""


def dummy_reset(params: Params, *, key):
    return Timestep.initial(
        obs=jnp.zeros((1,)),
        state=jnp.int_(0),
        info={},
    )


def dummy_step(env_state, action, params: Params, *, key):
    env_state = env_state + 1
    terminal = env_state >= params.max_horizon
    return Timestep(
        obs=jnp.zeros((1,)),
        state=env_state,
        reward=jnp.float_(1.0),
        discount=1.0 - terminal,
        step_type=jnp.where(terminal, StepType.LAST, StepType.MID),
        info={},
    )


def make_dummy_env(
    max_horizon: int,
) -> Environment[Integer[Array, ""], Integer[Array, ""], None, Params]:
    return Environment(
        _inner=None,  # type: ignore
        name="dummy",
        reset=dummy_reset,
        step=dummy_step,
        action_space=lambda params: specs.DiscreteArray(num_values=1, name="action"),
        observation_space=lambda params: specs.BoundedArray(
            (1,),
            dtype=jnp.float_,
            minimum=0,
            maximum=params.max_horizon,
            name="observation",
        ),
        goal_space=lambda params: None,
    ), Params(max_horizon=max_horizon)
