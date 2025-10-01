import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.testing as npt
import pytest

from envs.dummy_env import make_dummy_env, simple_rollout
from envs.pytree_env import Observation, make_pytree_env
from utils.structures import SENTINEL, GoalObs, StepType
from wrappers.auto_reset import PrevDone, auto_reset_wrapper
from wrappers.flatten_observation import flatten_observation_wrapper
from wrappers.goal_wrapper import GoalState, goal_wrapper
from wrappers.metrics import metrics_wrapper


@pytest.fixture(params=(1, 2, 5))
def dummy_env_params(request):
    return make_dummy_env(max_horizon=request.param)


@pytest.fixture
def pytree_env():
    return make_pytree_env()


def test_dummy_env(dummy_env_params):
    env, params = dummy_env_params
    horizon = params.max_horizon
    ts_s = simple_rollout(env, params, None, horizon, key=jr.key(0))
    del env, params

    npt.assert_array_equal(ts_s.reward, jnp.array((SENTINEL,) + (1.0,) * horizon))
    npt.assert_array_equal(ts_s.state, jnp.arange(horizon + 1))
    npt.assert_array_equal(
        ts_s.obs,
        jnp.arange(horizon + 1, dtype=float)[:, jnp.newaxis],
    )
    npt.assert_array_equal(
        ts_s.discount,
        jnp.array((SENTINEL,) + (1.0,) * (horizon - 1) + (0,)),
    )
    npt.assert_array_equal(
        ts_s.step_type,
        jnp.array((StepType.FIRST,) + (StepType.MID,) * (horizon - 1) + (StepType.LAST,)),
    )


def test_auto_reset_wrapper(dummy_env_params):
    env, params = dummy_env_params
    wrapped = auto_reset_wrapper(env)
    key = jr.key(0)

    # Initial reset
    ts = wrapped.reset(params, key=key)
    assert isinstance(ts.state, PrevDone)
    assert not ts.state.is_last
    assert ts.is_first
    assert ts.discount == SENTINEL
    assert ts.reward == SENTINEL

    # First step (not done)
    for step in range(params.max_horizon - 1):
        ts = wrapped.step(ts.state, None, params, key=key)
        assert not ts.state.is_last
        assert ts.is_mid
        assert ts.discount == 1.0
        assert ts.reward == 1.0

    ts = wrapped.step(ts.state, None, params, key=key)
    assert ts.state.is_last
    assert ts.is_last
    assert ts.discount == 0.0
    assert ts.reward == 1.0

    # After auto-reset, prev_done should be False again
    ts = wrapped.step(ts.state, None, params, key=key)
    assert not ts.state.is_last
    assert ts.is_first
    assert ts.discount == SENTINEL
    assert ts.reward == SENTINEL


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


@pytest.fixture
def goal_wrapped(dummy_env_params):
    return goal_wrapper(dummy_env_params[0])


def test_goal_wrapper_reset_and_step(dummy_env_params, goal_wrapped):
    assert goal_wrapped._inner is dummy_env_params[0]
    assert goal_wrapped.name == "goal_wrapper"
    assert goal_wrapped.fullname == "dummy > goal_wrapper"

    # Reset
    ts = goal_wrapped.reset(dummy_env_params[1], key=jr.key(0))
    assert isinstance(ts.state, GoalState)
    assert hasattr(ts.state, "goal")
    assert ts.state.goal == 0
    assert isinstance(ts.obs, GoalObs)
    assert ts.obs.goal == 0

    # Step
    ts2 = goal_wrapped.step(ts.state, None, dummy_env_params[1], key=jr.key(0))
    assert isinstance(ts2.state, GoalState)
    assert ts2.state.goal == 0
    assert ts2.obs.goal == 0


def test_goal_wrapper_observation_space(dummy_env_params, goal_wrapped):
    _, params = dummy_env_params
    # Should be a list: [original_space, DiscreteArraySpec]
    obs_space = goal_wrapped.observation_space(params)
    assert isinstance(obs_space, list)
    assert obs_space[1].num_values == 1
    assert getattr(obs_space[1], "name", None) == "goal"


def test_metrics_accumulates_and_resets(dummy_env_params):
    # Create dummy env with horizon 3
    env, params = dummy_env_params
    env = metrics_wrapper(env)
    env = auto_reset_wrapper(env)

    key1, key2 = jr.split(jr.key(0))

    # Take a single rollout
    ts_seq = simple_rollout(env, params, action=None, horizon=params.max_horizon, key=key1)
    npt.assert_array_equal(ts_seq.state.metrics.cum_return, jnp.arange(params.max_horizon + 1))
    npt.assert_array_equal(ts_seq.state.metrics.step, jnp.arange(params.max_horizon + 1))

    # Reset and rollout again: metrics should start from zero
    ts_seq2 = simple_rollout(env, params, action=None, horizon=2 * params.max_horizon + 1, key=key2)
    # first trajectory
    npt.assert_array_equal(
        ts_seq2.state.metrics.cum_return[: params.max_horizon + 1],
        jnp.arange(params.max_horizon + 1),
    )
    npt.assert_array_equal(
        ts_seq2.state.metrics.step[: params.max_horizon + 1],
        jnp.arange(params.max_horizon + 1),
    )
    # second trajectory
    npt.assert_array_equal(
        ts_seq2.state.metrics.cum_return[params.max_horizon + 1 :],
        jnp.arange(params.max_horizon + 1),
    )
    npt.assert_array_equal(
        ts_seq2.state.metrics.step[params.max_horizon + 1 :],
        jnp.arange(params.max_horizon + 1),
    )


def test_metrics_increments_each_step():
    env, params = make_dummy_env(max_horizon=5)
    wrapped_env = metrics_wrapper(env)
    key = jax.random.PRNGKey(123)
    ts_seq = simple_rollout(wrapped_env, params, action=0, horizon=5, key=key)
    # Each step should increment by 1
    assert jnp.all(ts_seq.state.metrics.cum_return == jnp.arange(6))
    assert jnp.all(ts_seq.state.metrics.step == jnp.arange(6))


def test_metrics_wrapper_outside_auto_reset_raises(pytree_env):
    with pytest.raises(ValueError):
        wrapped_env = auto_reset_wrapper(pytree_env)
        wrapped_env = metrics_wrapper(wrapped_env)

    # correct order should not raise an error
    wrapped_env = metrics_wrapper(pytree_env)
    wrapped_env = auto_reset_wrapper(wrapped_env)
