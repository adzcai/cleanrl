import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.testing as npt
from jaxtyping import Array, Integer

from envs.dummy_env import simple_rollout
from envs.multi_catch import BaseObs, make_multi_catch
from utils.jax_utils import BootstrapConfig, bootstrap, roll_into_matrix, tree_slice
from utils.log_utils import exec_loop
from utils.prioritized_buffer import PrioritizedBuffer
from utils.structures import GoalObs, Prediction, Transition
from wrappers.auto_reset import auto_reset_wrapper


def test_roll_into_matrix():
    x = jnp.arange(4)
    npt.assert_array_equal(
        roll_into_matrix(x),
        jnp.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 1, 2],
            ]
        ),
    )


def test_roll_into_matrix_2d():
    x = jnp.arange(6).reshape((2, 3))
    npt.assert_array_equal(
        roll_into_matrix(x),
        jnp.array(
            [
                [
                    [0, 1, 2],
                    [3, 4, 5],
                ],
                [
                    [3, 4, 5],
                    [0, 1, 2],
                ],
            ]
        ),
    )


def test_bootstrap_dummy(dummy_env_params):
    env, params = dummy_env_params
    ts_s = simple_rollout(env, params, None, params.max_horizon, key=jr.key(0))
    zero_s = jnp.zeros((params.max_horizon + 1, 1))
    txn_s = Transition(
        time_step=ts_s,
        action=jnp.zeros((params.max_horizon + 1,), dtype=jnp.int_),
        pred=Prediction(policy_logits=zero_s, value_logits=zero_s),
        mcts_probs=zero_s,
    )

    gamma = 0.9

    boot_value_sh, _ = bootstrap(
        predict_s=lambda obs, action_s: (obs * action_s, None),
        txn_s=txn_s,
        config=BootstrapConfig(
            discount=gamma,
            lambda_gae=1.0,
        ),
    )

    assert boot_value_sh.shape == (params.max_horizon, params.max_horizon)
    assert boot_value_sh[0, 0] == (1 - gamma**params.max_horizon) / (1 - gamma)
    # assert boot_value_sh.item() == 1.0


def test_bootstrap_catch():
    env, params = make_multi_catch()
    env = auto_reset_wrapper(env)
    num_timesteps = 3 * params.rows
    ts_s = simple_rollout(env, params, jnp.int_(1), num_timesteps - 1, key=jr.key(0))
    zero_s = jnp.zeros((num_timesteps, 1))
    txn_s = Transition(
        time_step=ts_s,
        action=jnp.zeros((num_timesteps,), dtype=jnp.int_),
        pred=Prediction(policy_logits=zero_s, value_logits=zero_s),
        mcts_probs=zero_s,
    )

    gamma = 0.9  # debugging

    def predict_s(obs: GoalObs[BaseObs], action_s: Integer[Array, " horizon"]):
        # ball_row = jnp.argmax(obs.obs[:-1, :], axis=0)
        return jnp.arange(action_s.size, dtype=float), None

    boot_value_sh, _ = bootstrap(
        predict_s=predict_s,
        txn_s=txn_s,
        config=BootstrapConfig(
            discount=gamma,
            lambda_gae=1.0,
        ),
    )
    assert boot_value_sh.shape == (num_timesteps - 1, num_timesteps - 1)
    last_sh = roll_into_matrix(ts_s.is_last)
    npt.assert_equal(np.asarray(boot_value_sh[last_sh[:-1, :-1]]), 0.0)
    # assert jnp.all(boot_value_sh[0, :] == (1 - gamma**n) / (1 - gamma))


def test_prioritized_buffer_sentinel_td_error():
    """Prioritized buffer shouldn't produce large negative TD errors due to sentinel values."""
    # Setup environment and buffer
    buffer = PrioritizedBuffer.new(
        batch_size=32,
        max_length=32,
        sample_length=10,
    )

    env, params = make_multi_catch()
    env = auto_reset_wrapper(env)

    key_reset, key_rollout = jr.split(jr.key(0), 2)
    init_ts = jax.vmap(ft.partial(env.reset, params))(key=key_reset)
    x = tree_slice(init_ts, 0)
    buf_state = buffer.init(x)

    def step_fn(carry=(init_ts, buf_state), key=key_rollout):
        ts, buf_state = carry
        actions = jr.randint(key, (buffer.batch_size,), 0, 3)
        next_ts = jax.vmap(
            lambda key, action: env.step(ts.state, action, params, key=key)
        )(jr.split(key, buffer.batch_size), actions)
        buf_state = buf_state.add(ts, next_ts)
        return (next_ts, buf_state), None

    (final_ts, buf_state), _ = exec_loop(buffer.max_length)(step_fn)
