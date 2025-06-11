import dm_env.specs as specs
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from testing.pytree_env import Observation
from wrappers.flatten_observation import flatten_observation_wrapper


@pytest.fixture
def wrapped(pytree_env):
    return flatten_observation_wrapper(pytree_env)


def test_flatten_observation_pytree_step(pytree_env, wrapped):
    assert wrapped._inner is pytree_env
    assert wrapped.name == "flatten_observation"
    assert wrapped.fullname == "pytree_env > flatten_observation"

    # test initial reset
    ts = wrapped.reset(None, key=jr.key(0))
    assert np.array_equal(ts.obs.a, np.arange(3))
    assert np.array_equal(ts.obs.b, np.zeros(4))
    assert ts.obs.scalar == 0.0

    # test steps
    ts = wrapped.step(ts.state, None, None, key=jr.key(0))
    assert np.array_equal(ts.obs.a, np.arange(3) + 1)
    assert np.array_equal(ts.obs.b, np.ones(4))
    assert ts.obs.scalar == np.pi


def test_flatten_observation_space(wrapped):
    obs_space = wrapped.observation_space(None)
    assert isinstance(obs_space, Observation)
    assert obs_space.a.shape == (3,)
    assert obs_space.b.shape == (4,)
    assert obs_space.scalar.shape == ()
    assert obs_space.scalar.minimum == -100.0  # type: ignore
    assert obs_space.scalar.maximum == 100.0  # type: ignore
