"""Implementation of the MuZero algorithm.

1. Model definitions
2. Data structures for loop carries
3. Loss function
"""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr

import mctx
import equinox as eqx
import optax
import flashbax as fbx
import rlax

import gymnax
from gymnax.environments.environment import Environment, EnvState
from gymnax.wrappers import FlattenObservationWrapper
from gymnax.visualize import Visualizer

import tempfile
import wandb
import matplotlib
import matplotlib.pyplot as plt

from functools import partial
from typing import NamedTuple, Annotated as Batched
from jaxtyping import Bool, Integer, Key, Float, Array, PyTree

from log_util import get_norm_data, log_values, tree_slice, visualize_catch


matplotlib.use("agg")


class WorldModelRNN(eqx.Module):
    """Parameterizes the recurrent world model network.

    Takes `(hidden, action) -> (hidden, reward)`.
    """

    cell: eqx.nn.GRUCell
    reward_head: eqx.nn.MLP

    def __init__(
        self,
        num_actions: int,
        rnn_size: int,
        mlp_size: int,
        mlp_depth: int,
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_reward = jr.split(key)
        self.cell = eqx.nn.GRUCell(num_actions, rnn_size, key=key_cell)
        self.reward_head = eqx.nn.MLP(
            rnn_size, "scalar", mlp_size, mlp_depth, key=key_reward
        )

    def __call__(
        self, hidden: Float[Array, "rnn_size"], actions: Integer[Array, "horizon"]
    ):
        return jax.lax.scan(
            lambda carry, x: self.step(carry, x),
            hidden,
            actions,
        )

    def step(
        self, hidden: Float[Array, "rnn_size"], action: Integer[Array, ""]
    ) -> tuple[Float[Array, "rnn_size"], Float[Array, ""]]:
        """Reset to initial hidden state if done."""
        # TODO handle terminal states
        action = jax.nn.one_hot(action, self.cell.input_size)
        hidden = self.cell(action, hidden)
        return hidden, self.reward_head(hidden)

    def init_hidden(self) -> Float[Array, "rnn_size"]:
        return jnp.zeros(self.cell.hidden_size)


class ActorCritic(eqx.Module):
    """Parameterizes the actor-critic network."""

    policy_head: eqx.nn.MLP
    value_head: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        num_actions: int,
        mlp_size: int,
        mlp_depth: int,
        *,
        key: Key[Array, ""],
    ):
        key_policy, key_value = jr.split(key)
        self.policy_head = eqx.nn.MLP(
            in_size, num_actions, mlp_size, mlp_depth, key=key_policy
        )
        self.value_head = eqx.nn.MLP(
            in_size, "scalar", mlp_size, mlp_depth, key=key_value
        )

    def __call__(self, hidden: Float[Array, "rnn_size"]):
        return self.policy_head(hidden), self.value_head(hidden)


class Network(eqx.Module):
    """The entire MuZero network parameters."""

    projection: eqx.nn.MLP
    world_model: WorldModelRNN
    actor_critic: ActorCritic

    def __init__(self, obs_size: int, num_actions: int, *, key: Key[Array, ""]):
        key_projection, key_world_model, key_actor_critic = jr.split(key, 3)

        self.projection = eqx.nn.MLP(
            in_size=obs_size,
            out_size=config.rnn_size,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            key=key_projection,
        )
        self.world_model = WorldModelRNN(
            rnn_size=config.rnn_size,
            num_actions=num_actions,
            mlp_size=config.mlp_size,
            mlp_depth=config.mlp_depth,
            key=key_world_model,
        )
        self.actor_critic = ActorCritic(
            in_size=config.rnn_size,
            num_actions=num_actions,
            mlp_size=config.mlp_size,
            mlp_depth=config.mlp_depth,
            key=key_actor_critic,
        )

    def __call__(
        self, obs: Float[Array, "obs_size"], actions: Integer[Array, "horizon"]
    ):
        @partial(
            jax.lax.scan,
            init=self.projection(obs),
            xs=actions,
        )
        def rollout(hidden: Float[Array, "rnn_size"], action: Integer[Array, ""]):
            logits, value = self.actor_critic(hidden)
            hidden, reward = self.world_model.step(hidden, action)
            return hidden, (reward, logits, value)

        _, outputs = rollout
        return outputs


class RolloutState(NamedTuple):
    """The observation for acting and the true environment state for rollouts."""

    obs: Float[Array, "obs_size"]
    env_state: EnvState
    is_initial: Bool[Array, ""]


class Transition(NamedTuple):
    """A single transition. The quantities are computed from the given observation state."""

    rollout_state: RolloutState
    action: Integer[Array, ""]
    reward: Float[Array, ""]
    policy_logits: Float[Array, "num_actions"]


class ParamState(NamedTuple):
    """Each outer update iteration."""

    params: Network
    opt_state: optax.OptState


class IterationState(NamedTuple):
    """Carry for iterations.

    Each iteration consists of a rollout phase followed by an optimization phase.
    """

    tally: int
    param_state: ParamState
    target_params: Network
    rollout_state: RolloutState
    buffer_state: fbx.trajectory_buffer.TrajectoryBufferState[Transition]


class Config(NamedTuple):
    env: str = "Catch-bsuite"
    num_evals: int = 10

    # environment collecting
    total_transitions: int = 100_000
    horizon: int = 60
    num_envs: int = 8
    num_mcts_simulations: int = 64

    target_update_freq: int = 4
    target_params_step: float = 1.0

    # network architecture
    rnn_size: int = 128
    mlp_size: int = 64
    mlp_depth: int = 2

    # optimization
    num_gradients: int = 4
    minibatch_size: int = 4
    num_epochs: int = 8
    max_gradient_norm: float = 0.5
    learning_rate: float = 5e-3

    # loss function
    discount: float = 0.99
    value_coefficient: float = 0.5
    reward_coefficient: float = 0.8
    num_steps_bootstrap: int = 100


def muzero_loss(
    net: Network,
    target_net: Network,
    trajectory: Batched[Transition, "horizon"],
):
    """MuZero loss.

    Here we align the targets at each timestep to maximize signal.
    """
    horizon_axis = jnp.arange(trajectory.action.size)

    def roll(ary):
        return jax.vmap(partial(jnp.roll, ary, axis=0))(-horizon_axis)

    policy_targets: Float[Array, "horizon num_actions"] = jax.nn.softmax(
        trajectory.policy_logits, axis=-1
    )
    action_rolled, reward_rolled, is_initial_rolled, policy_targets_rolled = (
        roll(trajectory.action),
        roll(trajectory.reward),
        roll(trajectory.rollout_state.is_initial),
        roll(policy_targets),
    )

    # e.g. values[i, j] is the predicted value at time i+j,
    # based on the observation at time i
    pred_rewards, pred_policy_logits, pred_values = jax.vmap(net)(
        trajectory.rollout_state.obs, action_rolled
    )

    # bootstrap target values
    # discard final transition
    # (too much hassle to pass around final transition)
    _, _, target_pred_values = jax.vmap(target_net)(
        trajectory.rollout_state.obs, action_rolled
    )
    bootstrapped_values: Float[Array, "length length-1"] = jax.vmap(
        rlax.n_step_bootstrapped_returns, (0, 0, 0, None)
    )(
        reward_rolled[:, :-1],
        jnp.where(is_initial_rolled[:, 1:], 0.0, config.discount),
        target_pred_values[:, 1:],
        config.num_steps_bootstrap,
    )

    # top left triangle of matrix
    mask = (
        horizon_axis[:, jnp.newaxis] + horizon_axis[jnp.newaxis, :]
    ) < horizon_axis.size

    # value loss
    value_losses = optax.l2_loss(pred_values[:-1, :-1], bootstrapped_values[:-1, :])
    value_loss = jnp.mean(value_losses, where=mask[:-1, :-1])

    # policy loss
    policy_losses = optax.softmax_cross_entropy(
        pred_policy_logits, policy_targets_rolled
    )
    chex.assert_equal_shape([policy_losses, mask])
    policy_loss = jnp.mean(policy_losses, where=mask)

    # reward model loss
    reward_losses = optax.l2_loss(pred_rewards, reward_rolled)
    reward_loss = jnp.mean(reward_losses, where=mask)

    # total
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        + config.reward_coefficient * reward_loss
    )

    # logging
    td_error = jnp.mean(
        bootstrapped_values - pred_values[:-1, :-1], where=mask[:-1, :-1]
    )
    return loss, {
        "train/loss": loss,
        "train/td_error": td_error,
        "train/policy_loss": policy_loss,
        "train/value_loss": value_loss,
        "train/reward_loss": reward_loss,
    }


def make_train(config: Config):
    num_iterations = config.total_transitions // (config.horizon * config.num_envs)
    total_gradients = num_iterations * config.num_gradients
    # eval_freq = num_iterations // config.num_evals

    env, env_params = gymnax.make(config.env)
    obs_shape = env.observation_space(env_params).shape
    num_actions = env.action_space(env_params).n
    env: Environment = FlattenObservationWrapper(env)
    # env: Environment = LogWrapper(env) # doesn't respect the envstate

    # decay learning rate over three orders of magnitude
    lr = optax.cosine_decay_schedule(
        config.learning_rate,
        total_gradients,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm), optax.adamw(lr, eps=1e-5)
    )

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=config.num_envs,
        sample_batch_size=config.minibatch_size,
        sample_sequence_length=config.horizon,
        period=1,
        min_length_time_axis=config.num_envs * config.horizon,
        max_length_time_axis=config.total_transitions // 2,
    )

    def train(key: Key[Array, ""]):
        key_reset, key_networks, key_iter = jr.split(key, 3)

        init_obs, init_env_state = jax.vmap(env.reset, (0, None))(
            jr.split(key_reset, config.num_envs), env_params
        )
        init_rollout_state = RolloutState(
            init_obs,
            init_env_state,
            jnp.ones(config.num_envs, dtype=bool),
        )
        init_net = Network(
            obs_size=init_obs.shape[-1], num_actions=num_actions, key=key_networks
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)
        init_buffer_state = buffer.init(
            Transition(
                rollout_state=tree_slice(init_rollout_state, 0),
                action=jnp.zeros((), int),
                reward=jnp.zeros(()),
                policy_logits=jnp.empty(num_actions),
            )
        )

        @partial(
            jax.lax.scan,
            init=IterationState(
                tally=jnp.zeros((), int),
                param_state=ParamState(init_params, optim.init(init_params)),
                target_params=init_params,
                rollout_state=init_rollout_state,
                buffer_state=init_buffer_state,
            ),
            xs=jr.split(key_iter, num_iterations),
        )
        def iteration_step(iter_state: IterationState, key: Key[Array, ""]):
            """A single iteration of optimization.

            1. Collect a batch of rollouts in parallel.
            2. Run a few epochs of SGD (or some optimization algorithm) on the batch.
            """
            key_rollout, key_optim = jr.split(key)

            # leading dimension num_parallel_envs
            rollout_state, trajectories = rollout(
                eqx.combine(iter_state.param_state.params, net_static),
                iter_state.rollout_state,
                key_rollout,
            )
            buffer_state = buffer.add(iter_state.buffer_state, trajectories)

            @partial(
                jax.lax.scan,
                init=iter_state.param_state,
                xs=jr.split(key_optim, config.num_epochs),
            )
            def optimize_step(param_state: ParamState, key: Key[Array, ""]):
                trajectories = buffer.sample(buffer_state, key).experience

                @partial(jax.value_and_grad, has_aux=True)
                def loss_fn(params: Network):
                    """Average the loss across a batch of trjaectories."""
                    loss, aux = jax.vmap(muzero_loss, (None, None, 0))(
                        eqx.combine(params, net_static),
                        eqx.combine(iter_state.target_params, net_static),
                        trajectories,
                    )
                    chex.assert_shape(loss, (config.minibatch_size,))
                    return jnp.mean(loss), aux

                (loss, aux), grads = loss_fn(param_state.params)
                updates, opt_state = optim.update(
                    grads, param_state.opt_state, param_state.params
                )
                params = optax.apply_updates(param_state.params, updates)

                log_values(
                    jax.tree.map(jnp.mean, aux)
                    | get_norm_data(updates, "train/params/gradient")
                    | get_norm_data(params, "train/params/norm"),
                )

                return ParamState(params, opt_state), loss

            param_state, losses = optimize_step

            log_values(
                {
                    "train/average_total_reward": jnp.sum(trajectories.reward)
                    / jnp.sum(trajectories.rollout_state.is_initial)
                },
            )

            target_params = jax.tree.map(
                lambda new, old: jnp.where(
                    iter_state.tally % config.target_update_freq == 0,
                    optax.incremental_update(new, old, config.target_params_step),
                    old,
                ),
                param_state.params,
                iter_state.target_params,
            )

            return IterationState(
                tally=iter_state.tally + 1,
                param_state=param_state,
                target_params=target_params,
                rollout_state=rollout_state,
                buffer_state=buffer_state,
            ), losses

        return iteration_step

    def rollout(
        net: Network, obs_state: Batched[RolloutState, "num_envs"], key: Key[Array, ""]
    ) -> tuple[
        Batched[RolloutState, "num_envs"], Batched[Transition, "num_envs horizon"]
    ]:
        """Collect a batch of rollouts from the environment."""

        @partial(
            jax.lax.scan,
            init=obs_state,
            xs=jr.split(key, config.horizon),
        )
        def rollout_step(
            obs_state: Batched[RolloutState, "num_envs"], key: Key[Array, ""]
        ):
            """A single environment interaction."""
            key_action, key_step = jr.split(key)

            # choose action
            hiddens = jax.vmap(net.projection)(obs_state.obs)
            logits, values = jax.vmap(net.actor_critic)(hiddens)
            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=values,
                embedding=hiddens,
            )

            def mcts_recurrent_fn(
                params: None,  # params constant during rollout
                rng: Float[Key, ""],
                actions: Integer[Array, "num_envs"],
                hiddens: Float[Array, "num_envs rnn_size"],
            ):
                hiddens, reward = jax.vmap(net.world_model.step)(hiddens, actions)
                logits, values = jax.vmap(net.actor_critic)(hiddens)
                return mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=jnp.full_like(actions, config.discount),
                    prior_logits=logits,
                    value=values,
                ), hiddens

            output = mctx.muzero_policy(
                params=None,
                rng_key=key_action,
                root=root,
                recurrent_fn=mcts_recurrent_fn,
                num_simulations=config.num_mcts_simulations,
                max_depth=config.horizon,
            )

            # step environment
            obs, env_states, rewards, is_initials, infos = jax.vmap(
                env.step, (0, 0, 0, None)
            )(
                jr.split(key_step, config.num_envs),
                obs_state.env_state,
                output.action,
                env_params,
            )

            return RolloutState(obs, env_states, is_initials), Transition(
                rollout_state=obs_state,
                action=output.action,
                reward=rewards,
                policy_logits=output.action_weights,
            )

        final_rollout_state, trajectories = rollout_step
        # put `num_envs` axis before `horizon` axis
        return final_rollout_state, jax.tree.map(
            lambda x: x.swapaxes(0, 1), trajectories
        )

    return train


if __name__ == "__main__":
    config = Config()
    train = jax.jit(make_train(config))
    with wandb.init(
        project="jax-rl",
        config=config._asdict(),
    ) as run:
        out = jax.block_until_ready(train(jr.key(0)))
    print("Done training.")
