import jax.random as jr
import pytest

from testing.dummy_env import Params
from utils.base import GoalObs
from utils.goal_wrapper import GoalState, goal_wrapper


@pytest.fixture
def wrapped(dummy_env):
    return goal_wrapper(dummy_env)


@pytest.fixture
def params():
    return Params(max_horizon=10)


def test_goal_wrapper_reset_and_step(dummy_env, params, wrapped):
    assert wrapped._inner is dummy_env
    assert wrapped.name == "goal_wrapper"
    assert wrapped.fullname == "dummy > goal_wrapper"

    # Reset
    ts = wrapped.reset(params, key=jr.key(0))
    assert isinstance(ts.state, GoalState)
    assert hasattr(ts.state, "goal")
    assert ts.state.goal == 0
    assert isinstance(ts.obs, GoalObs)
    assert ts.obs.goal == 0

    # Step
    ts2 = wrapped.step(ts.state, None, params, key=jr.key(0))
    assert isinstance(ts2.state, GoalState)
    assert ts2.state.goal == 0
    assert ts2.obs.goal == 0


def test_goal_wrapper_observation_space(wrapped, params):
    # Should be a list: [original_space, DiscreteArraySpec]
    obs_space = wrapped.observation_space(params)
    assert isinstance(obs_space, list)
    assert obs_space[1].num_values == 1
    assert getattr(obs_space[1], "name", None) == "goal"
