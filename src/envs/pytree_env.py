import dm_env.specs as specs
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from utils.log_utils import dataclass
from utils.structures import Environment, StepType, TimeStep


@dataclass
class Observation:
    a: Integer[Array, "3"]
    b: Float[Array, "2 2"]
    scalar: Float[Array, ""]


def reset(params, *, key):
    obs = Observation(
        a=jnp.arange(3),
        b=jnp.zeros((2, 2)),
        scalar=jnp.float_(0.0),
    )
    return TimeStep.initial(obs=obs, state=0, info={})


def step(env_state: int, action, params, *, key):
    state = env_state + 1
    obs = Observation(
        a=jnp.arange(3) + state,
        b=jnp.full((2, 2), state, dtype=jnp.float32),
        scalar=jnp.float_(jnp.pi * state),
    )
    return TimeStep(
        obs=obs,
        state=state,
        reward=jnp.float_(1.0),
        discount=jnp.float_(1.0),
        step_type=jnp.int_(StepType.MID),
        info={},
    )


def observation_space(params):
    return Observation(
        a=specs.BoundedArray(
            shape=(3,), dtype=jnp.float32, minimum=0.0, maximum=100.0, name="a"
        ),  # type: ignore
        b=specs.BoundedArray(
            shape=(2, 2), dtype=jnp.float32, minimum=0.0, maximum=100.0, name="b"
        ),  # type: ignore
        scalar=specs.BoundedArray(
            shape=(),
            dtype=jnp.float32,
            minimum=-100.0,
            maximum=100.0,
            name="scalar",
        ),  # type: ignore
    )


def make_pytree_env() -> Environment[Observation, int, None, None]:
    """Create a dummy environment with a pytree observation."""
    return Environment(
        _inner=None,  # type: ignore
        name="pytree_env",
        reset=reset,
        step=step,
        action_space=lambda params: None,
        observation_space=observation_space,
        goal_space=lambda params: None,
    )
