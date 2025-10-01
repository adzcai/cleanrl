import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt
import pytest

from cleanrl import muzero
from cleanrl_utils.config import (
    TrainConfig,
    dict_to_dataclass,
    get_args,
)


def test_muzero_learns_value_on_dummy_env():
    # Setup
    config = get_args(["src/configs/dummy.yaml"])
    config = dict_to_dataclass(TrainConfig, config)
    train_fn = muzero.make_train(config)
    env, env_params, num_actions, net_static = muzero.make_env_and_network(config)
    key_train, key_reset = jr.split(jr.key(config.seed), 2)

    final_iter_state, _ = train_fn(key_train)

    # Get trained params and net
    params = final_iter_state.param_state.params
    net = eqx.combine(params, net_static)
    del params, net_static

    ts = env.reset(env_params, key=key_reset)
    _, (reward_s, pred_s) = net.world_model_rollout(ts.obs, jnp.zeros((1,), dtype=int))
    npt.assert_allclose(config.value.logits_to_value(pred_s.value_logits), 1.0)

    ts = env.step(ts.state, jnp.int_(0), env_params)
    assert ts.is_last
    _, (reward_s, pred_s) = net.world_model_rollout(ts.obs, jnp.zeros((1,), dtype=int))
    npt.assert_allclose(config.value.logits_to_value(pred_s.value_logits), 0.0)


@pytest.mark.skip(reason="Disabled for now")
def test_muzero_catch():
    config = get_args(["src/configs/catch.yaml"])
    config = dict_to_dataclass(TrainConfig, config)
    train_fn = muzero.make_train(config)
    key = jr.key(config.seed)

    final_iter_state, _ = train_fn(key)
