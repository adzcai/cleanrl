import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from experiments import muzero
from experiments.config import (
    TrainConfig,
    dict_to_dataclass,
    get_args,
)
from utils.structures import GoalObs


@pytest.mark.parametrize("max_horizon", [3, 5])
def test_muzero_learns_value_on_dummy_env(max_horizon):
    # Setup
    config = get_args(["src/configs/dummy.yaml"])
    config = dict_to_dataclass(TrainConfig, config)
    train_fn = muzero.make_train(config)
    key = jr.key(config.seed)

    final_iter_state, _ = train_fn(key)

    # Get trained params and net
    params = final_iter_state.param_state.params
    net_static = params  # eqx.partition not needed for dummy
    # Evaluate value prediction at initial state

    dummy_obs = jnp.int_(0)
    dummy_goal = jnp.int_(0)
    goal_obs = GoalObs(obs=dummy_obs, goal=dummy_goal)
    # The value function should predict close to max_horizon (all rewards are 1)
    pred = params.actor_critic(
        params.world_model.init_hidden_dyn(), params.embed_goal(dummy_goal)
    )
    # Convert logits to value
    value = jax.nn.softmax(pred.value_logits)
    # Weighted sum over bins
    bin_vals = jnp.linspace(
        config.value.min_value, config.value.max_value, config.value.num_value_bins
    )
    pred_value = (value * bin_vals).sum()
    assert pred_value > max_horizon - 1, (
        f"Predicted value {pred_value} too low for horizon {max_horizon}"
    )


@pytest.mark.skip(reason="Disabled for now")
def test_muzero_catch():
    config = get_args(["src/configs/catch.yaml"])
    config = dict_to_dataclass(TrainConfig, config)
    train_fn = muzero.make_train(config)
    key = jr.key(config.seed)

    final_iter_state, _ = train_fn(key)
