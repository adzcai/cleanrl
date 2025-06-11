import jax.random as jr
import pytest

from testing.dummy_env import Params
from utils.structures import GoalObs
from wrappers.goal_wrapper import GoalState, goal_wrapper


@pytest.fixture
def wrapped(dummy_env_params):
    return goal_wrapper(dummy_env_params[0])


def test_goal_wrapper_reset_and_step(dummy_env_params, wrapped):
    assert wrapped._inner is dummy_env_params[0]
    assert wrapped.name == "goal_wrapper"
    assert wrapped.fullname == "dummy > goal_wrapper"

    # Reset
    ts = wrapped.reset(dummy_env_params[1], key=jr.key(0))
    assert isinstance(ts.state, GoalState)
    assert hasattr(ts.state, "goal")
    assert ts.state.goal == 0
    assert isinstance(ts.obs, GoalObs)
    assert ts.obs.goal == 0

    # Step
    ts2 = wrapped.step(ts.state, None, dummy_env_params[1], key=jr.key(0))
    assert isinstance(ts2.state, GoalState)
    assert ts2.state.goal == 0
    assert ts2.obs.goal == 0


def test_goal_wrapper_observation_space(dummy_env_params, wrapped):
    # Should be a list: [original_space, DiscreteArraySpec]
    obs_space = wrapped.observation_space(dummy_env_params[1])
    assert isinstance(obs_space, list)
    assert obs_space[1].num_values == 1
    assert getattr(obs_space[1], "name", None) == "goal"
