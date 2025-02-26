"""Implementation of AlphaZero with a recurrent actor-critic network."""

import dataclasses as dc
import functools as ft
import sys
from pathlib import Path
from typing import Annotated as Batched
from typing import Generic, Literal, NamedTuple

import chex
import distrax
import dm_env.specs as specs
import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import matplotlib.pyplot as plt
import mctx
import optax
import rlax
import yaml
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Integer, Key, UInt, jaxtyped
from matplotlib.axes import Axes

import wandb
from config import ArchConfig, TrainConfig, main
from log_util import exec_loop, get_norm_data, log_values, roll_into_matrix, tree_slice
from prioritized_buffer import BufferState, PrioritizedBuffer
from wrappers.base import StepType, TEnvState, Timestep, TObs
from wrappers.goal_wrapper import GoalObs
from wrappers.log import Metrics
from wrappers.multi_catch import get_action_name, visualize_catch
from wrappers.translate import make_env

matplotlib.use("agg")  # enable plotting inside jax callback

Obs = GoalObs[Float[Array, " *obs_size"]]


class WorldModelRNN(eqx.Module):
    """Parameterizes the recurrent world model.

    Takes (hidden, action, goal_embedding) -> (hidden, reward).
    """

    cell: eqx.nn.GRUCell
    """The recurrent cell for the world model"""
    reward_head: eqx.nn.MLP
    """The reward model. Takes in hidden state and goal embedding."""

    def __init__(
        self,
        config: ArchConfig,
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_reward = jr.split(key)
        self.cell = eqx.nn.GRUCell(config.num_actions, config.rnn_size, key=key_cell)
        self.reward_head = eqx.nn.MLP(
            config.rnn_size + config.num_goals,
            config.num_value_bins,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_reward,
        )

    def step(
        self,
        hidden: Float[Array, " rnn_size"],
        action: Integer[Array, ""],
        goal_embedding: Float[Array, " mlp_size"],
    ) -> tuple[Float[Array, " rnn_size"], Float[Array, " num_value_bins"]]:
        """Transitions to the next embedding and emits predicted reward."""
        # TODO handle continue predictor
        action = jax.nn.one_hot(action, self.cell.input_size)
        hidden = self.cell(action, hidden)
        return hidden, self.reward_head(jnp.concat([hidden, goal_embedding]))

    def init_hidden(self):
        return jnp.zeros(self.cell.hidden_size)


class Prediction(NamedTuple):
    policy_logits: Float[Array, " num_actions"]
    value_logits: Float[Array, " num_value_bins"]


class ActorCritic(eqx.Module):
    """Parameterizes the actor-critic network."""

    policy_head: eqx.nn.MLP
    """Policy network. Takes world model hidden state and goal embedding."""
    value_head: eqx.nn.MLP
    """Value network. Takes world model hidden state and goal embedding."""

    def __init__(
        self,
        config: ArchConfig,
        *,
        key: Key[Array, ""],
    ):
        key_policy, key_value = jr.split(key)
        self.policy_head = eqx.nn.MLP(
            config.rnn_size + config.num_goals,
            config.num_actions,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_policy,
        )
        self.value_head = eqx.nn.MLP(
            config.rnn_size + config.num_goals,
            config.num_value_bins,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_value,
        )

    def __call__(
        self, hidden: Float[Array, " rnn_size"], goal_embedding: Float[Array, " mlp_size"]
    ):
        """Predict action and value logits from the hidden state."""
        input = jnp.concat([hidden, goal_embedding])
        return Prediction(
            policy_logits=self.policy_head(input),
            value_logits=self.value_head(input),
        )


class MuZeroNetwork(eqx.Module):
    """The entire MuZero network parameters."""

    projection: eqx.nn.MLP
    """Embed the observation into world model recurrent state."""
    world_model: WorldModelRNN
    """The recurrent world model."""
    actor_critic: ActorCritic
    """The actor and critic networks."""
    num_goals: int

    def __init__(
        self,
        config: ArchConfig,
        obs_size: tuple[int, ...],
        *,
        key: Key[Array, ""],
    ):
        key_projection, key_world_model, key_actor_critic = jr.split(key, 3)
        self.num_goals = config.num_goals

        if config.kind == "mlp":
            self.projection = eqx.nn.MLP(
                in_size=obs_size[0],
                out_size=config.rnn_size,
                width_size=config.mlp_size,
                depth=config.mlp_depth,
                key=key_projection,
            )
        elif config.kind == "cnn":
            key_cnn, key_linear = jr.split(key)
            # (C, H, W)
            cnn = eqx.nn.Conv2d(
                in_channels=obs_size,
                out_channels=config.mlp_size,
                kernel_size=3,
                key=key_cnn,
            )
            flat_size = cnn(jnp.empty(obs_size)).size
            self.projection = eqx.nn.Sequential(
                [
                    cnn,
                    eqx.nn.Lambda(jnp.ravel),
                    eqx.nn.Linear(
                        in_features=flat_size, out_features=config.rnn_size, key=key_linear
                    ),
                ]
            )

        self.world_model = WorldModelRNN(config=config, key=key_world_model)
        self.actor_critic = ActorCritic(config=config, key=key_actor_critic)

    def __call__(
        self,
        goal_obs: Obs,
        action: Integer[Array, " horizon"],
    ):
        return jax.lax.scan(
            lambda hidden, action: self.step(hidden, action, goal_obs.goal),
            self.projection(goal_obs.obs),
            action,
        )

    @jaxtyped(typechecker=typechecker)
    def step(
        self,
        hidden: Float[Array, " rnn_size"],
        action: Integer[Array, ""],
        goal: Integer[Array, ""],
    ):
        goal_embedding = self.embed_goal(goal)
        pred = self.actor_critic(hidden, goal_embedding)
        hidden, reward = self.world_model.step(hidden, action, goal_embedding)
        return hidden, (reward, pred)

    def embed_goal(self, goal: Integer[Array, ""]):
        return jax.nn.one_hot(goal, self.num_goals)


class Transition(NamedTuple, Generic[TObs, TEnvState]):
    """A single transition. May be batched into a trajectory."""

    timestep: Timestep[TObs, TEnvState]
    """The timestep that was acted in."""
    action: Integer[Array, ""]
    """The action taken from `timestep`."""
    pred: Prediction
    """The prediction of the value of `timestep`."""
    mcts_probs: Float[Array, " num_actions"]
    """The MCTS action probability distribution from acting in timestep."""


class ParamState(NamedTuple):
    """Carried during optimization."""

    params: MuZeroNetwork
    opt_state: optax.OptState
    buffer_state: BufferState


class IterState(NamedTuple):
    """Carried across algorithm iterations."""

    step: Integer[Array, ""]
    timesteps: Batched[Timestep, " num_envs"]
    param_state: ParamState
    target_params: MuZeroNetwork


class LossStatistics(NamedTuple):
    """Quantities computed for the loss."""

    loss: Float[Array, ""]
    # policy
    mcts_entropy: Float[Array, " horizon"]
    policy_entropy: Float[Array, " horizon"]
    policy_logits: Float[Array, " horizon num_actions"]
    policy_loss: Float[Array, " horizon"]
    # value
    value_logits: Float[Array, " horizon num_value_bins"]
    target_value_logits: Float[Array, " horizon num_value_bins"]
    bootstrapped_return: Float[Array, " horizon-1"]
    value_loss: Float[Array, " horizon"]
    td_error: Float[Array, " horizon"]


def make_train(config: TrainConfig):
    num_iters = config.collection.total_transitions // (
        config.collection.num_envs * config.env.horizon
    )
    max_horizon = num_iters * config.env.horizon
    num_grad_updates = num_iters * config.optim.num_minibatches
    eval_freq = num_iters // config.eval.num_evals

    # construct environment
    env, env_params = make_env(config.env)
    action_space: specs.DiscreteArray = env.action_space(env_params)
    num_actions = action_space.num_values
    action_dtype = action_space.dtype
    print(env.fullname)

    env_reset_batch = jax.vmap(env.reset, in_axes=(None,))
    env_step_batch = jax.vmap(env.step, in_axes=(0, 0, None))  # map over state and action

    lr = optax.cosine_decay_schedule(config.optim.lr_init, num_grad_updates)
    optim = optax.chain(
        optax.clip_by_global_norm(config.optim.max_grad_norm),
        optax.adamw(lr),
    )

    buffer = PrioritizedBuffer.new(
        batch_size=config.collection.num_envs,
        max_time=max_horizon,
        horizon=config.env.horizon,
    )

    def logits_to_value(
        value_logits: Float[Array, " horizon num_value_bins"],
    ) -> Float[Array, " horizon"]:
        """Convert from logits for two-hot encoding to scalar values."""
        if config.value.num_value_bins == "scalar":
            return value_logits
        else:
            return rlax.transform_from_2hot(
                jax.nn.softmax(value_logits, axis=-1),
                config.value.min_value,
                config.value.max_value,
                config.value.num_value_bins,
            )

    def value_to_probs(value: Float[Array, " horizon"]) -> Float[Array, " horizon num_value_bins"]:
        """Convert from scalar values to probabilities for two-hot encoding."""
        if config.value.num_value_bins == "scalar":
            return value
        else:
            return rlax.transform_to_2hot(
                value,
                config.value.min_value,
                config.value.max_value,
                config.value.num_value_bins,
            )

    def get_invalid_actions(env_state: gymnax.EnvState) -> Bool[Array, " num_actions"]:
        if env.name == "Catch-bsuite":
            catch_state: gymnax.environments.bsuite.catch.EnvState = env_state
            catch_env: gymnax.environments.bsuite.catch.Catch = env
            return jnp.select(
                [
                    catch_state.paddle_x == 0,
                    catch_state.paddle_x == catch_env.columns - 1,
                ],
                [
                    jnp.array([True, False, False]),
                    jnp.array([False, False, True]),
                ],
                default=jnp.zeros(3, jnp.bool),
            )

        return None

    def train(key: Key[Array, ""]):
        key_reset, key_net, key_iter = jr.split(key, 3)

        print("Config\n" + "=" * 25)
        yaml.safe_dump(dc.asdict(config), sys.stdout)
        print("=" * 25)

        init_timesteps = env_reset_batch(
            env_params, key=jr.split(key_reset, config.collection.num_envs)
        )
        # initialize all state
        init_net = MuZeroNetwork(
            config=dc.replace(
                config.arch,
                num_actions=num_actions,
                num_value_bins=config.value.num_value_bins,
                num_goals=env.goal_space(env_params).num_values,
            ),
            obs_size=init_timesteps.obs.obs.shape[1:],
            key=key_net,
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)
        init_hiddens = jnp.broadcast_to(
            init_net.world_model.init_hidden(), (config.collection.num_envs, config.arch.rnn_size)
        )
        init_buffer_state = buffer.init(
            Transition(
                timestep=tree_slice(init_timesteps, 0),
                action=jnp.empty((), action_dtype),
                pred=init_net.actor_critic(
                    init_hiddens[0], init_net.embed_goal(init_timesteps.obs.goal[0])
                ),
                mcts_probs=jnp.empty(num_actions, init_hiddens.dtype),
            )
        )

        @exec_loop(
            IterState(
                step=0,
                timesteps=init_timesteps,
                param_state=ParamState(
                    params=init_params,
                    opt_state=optim.init(init_params),
                    buffer_state=init_buffer_state,
                ),
                target_params=init_params,
            ),
            num_iters,
            key_iter,
        )
        def iterate(iter_state: IterState, key: Key[Array, ""]):
            """A single iteration of optimization.

            1. Collect a batch of rollouts in parallel.
            2. Run a few epochs of SGD (or some optimization algorithm) on the batch.
            """
            key_rollout, key_optim, key_evaluate = jr.split(key, 3)

            # collect data
            rollout_states, trajectories = rollout(
                eqx.combine(iter_state.param_state.params, net_static),
                iter_state.timesteps,
                key_rollout,
            )
            buffer_state = buffer.add(iter_state.param_state.buffer_state, trajectories)

            metrics: Metrics = trajectories.timestep.info["metrics"]
            mean_return = jnp.mean(
                metrics.episode_return,
                where=trajectories.timestep.step_type == StepType.LAST,
            )
            log_values({"iter/step": iter_state.step, "iter/mean_return": mean_return})

            buffer_available = buffer.num_available(buffer_state) >= config.optim.batch_size

            @exec_loop(
                iter_state.param_state._replace(buffer_state=buffer_state),
                config.optim.num_minibatches,
                key_optim,
                cond=buffer_available,
            )
            def optimize_step(param_state: ParamState, key: Key[Array, ""]):
                batch = jax.vmap(buffer.sample, in_axes=(None, None))(
                    param_state.buffer_state,
                    config.eval.warnings,
                    key=jr.split(key, config.optim.batch_size),
                )
                trajectories: Batched[Transition, " batch_size horizon"] = batch.experience

                if False:

                    def debug(batch, free):
                        """Flashbax returns zero when JAX_ENABLE_X64 is not set."""
                        trajectories = batch.experience
                        for i in range(config.optim.batch_size):
                            weights = trajectories.mcts_probs[i]
                            if not jnp.allclose(jnp.sum(weights, axis=-1), 1) and not jnp.all(
                                jnp.isnan(weights)
                            ):
                                print("INVALID OUTPUT DISTRIBUTION BROKEN")
                                print(weights)
                                print(free)
                                print(batch.indices)
                                print(batch.priorities)
                                print("Trajectory with broken weights:")
                                print(tree_slice(trajectories, i))

                    jax.debug.callback(debug, batch, buffer_available)

                @ft.partial(jax.value_and_grad, has_aux=True)
                def loss_grad(params: MuZeroNetwork):
                    """Average the loss across a batch of trjaectories."""
                    losses, aux = loss_trajectory(
                        params, iter_state.target_params, trajectories, net_static
                    )
                    chex.assert_shape(losses, (config.optim.batch_size,))
                    # TODO add importance reweighting
                    return jnp.mean(losses), aux

                (_, aux), grads = loss_grad(param_state.params)
                # value_and_grad drops types
                aux: LossStatistics
                grads: MuZeroNetwork
                updates, opt_state = optim.update(grads, param_state.opt_state, param_state.params)
                params = optax.apply_updates(param_state.params, updates)

                # prioritize trajectories with high TD error
                priorities = jnp.mean(jnp.abs(aux.td_error), axis=-1)
                buffer_state = buffer.set_priorities(
                    param_state.buffer_state,
                    batch.idx,
                    priorities**config.optim.priority_exponent,
                )

                log_values(
                    {f"train/mean_{key}": jnp.mean(value) for key, value in aux._asdict().items()}
                    | get_norm_data(updates, "updates/norm")
                    | get_norm_data(params, "params/norm")
                )

                return ParamState(
                    params=params,
                    opt_state=opt_state,
                    buffer_state=buffer_state,
                ), None

            del buffer_state  # replaced by param_state.buffer_state
            param_state, _ = optimize_step

            target_params = jax.tree.map(
                lambda online, target: jnp.where(
                    iter_state.step % config.bootstrap.target_update_freq == 0,
                    optax.incremental_update(online, target, config.bootstrap.target_update_size),
                    target,
                ),
                param_state.params,
                iter_state.target_params,
            )

            # evaluate every eval_freq steps
            eval_return = jax.lax.cond(
                (iter_state.step % eval_freq == 0) & buffer_available,
                ft.partial(evaluate, num_envs=config.eval.num_eval_envs, net_static=net_static),
                lambda *_: -jnp.inf,  # avoid nan to support debugging
                # cond only accepts positional arguments
                param_state.params,
                target_params,
                param_state.buffer_state,
                key_evaluate,
            )

            jax.debug.print(
                "step {step}/{num_iters}. "
                "seen {transitions}/{total_transitions} transitions. "
                "buffer full {full}. "
                "mean return {mean_return}",
                step=iter_state.step,
                num_iters=num_iters,
                transitions=param_state.buffer_state.pos * config.collection.num_envs,
                total_transitions=config.collection.total_transitions,
                full=param_state.buffer_state.pos >= buffer.max_time,
                mean_return=mean_return,
            )

            if config.eval.warnings:
                valid = jax.tree.map(lambda x: jnp.any(jnp.isnan(x)), param_state.params)

                @ft.partial(jax.debug.callback, valid=valid)
                def debug(valid):
                    if not valid:
                        raise ValueError("NaN parameters.")

            return IterState(
                step=iter_state.step + 1,
                timesteps=rollout_states,
                param_state=param_state,
                target_params=target_params,
            ), eval_return

        final_iter_state, eval_returns = iterate
        return final_iter_state, eval_returns

    def rollout(
        net: MuZeroNetwork,
        init_timesteps: Batched[Timestep, " num_envs"],
        key: Key[Array, ""],
    ) -> tuple[Batched[Timestep, " num_envs"], Batched[Transition, " num_envs horizon"]]:
        @exec_loop(init_timesteps, config.env.horizon, key)
        def rollout_step(timesteps: Batched[Timestep, " num_envs"], key: Key[Array, ""]):
            key_action, key_step = jr.split(key)

            preds, outs = act_mcts(net, timesteps, key_action)
            actions = outs.action.astype(action_dtype)
            next_timesteps = env_step_batch(
                timesteps.state,
                actions,
                env_params,
                key=jr.split(key_step, outs.action.shape[0]),
            )
            return next_timesteps, Transition(
                # contains the reward and network predictions computed from the rollout state
                timestep=timesteps,
                action=actions,
                pred=preds,
                mcts_probs=outs.action_weights,
            )

        final_rollout_state, trajectories = rollout_step
        trajectories = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), trajectories
        )  # swap horizon and num_envs axes

        return final_rollout_state, trajectories

    # @jaxtyped(typechecker=typechecker)
    def act_mcts(
        net: MuZeroNetwork,
        timesteps: Batched[Timestep, " num_envs"],
        key: Key[Array, ""],
    ) -> tuple[Prediction, mctx.PolicyOutput[mctx.GumbelMuZeroExtraData]]:
        """Take a single action from the timestep using mcts.

        Returns:
            tuple[Prediction, mctx.PolicyOutput[mctx.GumbelMuZeroExtraData]]: The value and mcts output from the timestep.
        """
        init_hiddens = jax.vmap(net.projection)(timesteps.obs.obs)
        goal_embeddings = jax.vmap(net.embed_goal)(timesteps.obs.goal)
        preds = jax.vmap(net.actor_critic)(init_hiddens, goal_embeddings)

        root = mctx.RootFnOutput(
            prior_logits=preds.policy_logits,
            value=logits_to_value(preds.value_logits),
            # embedding contains current env_state and the hidden to pass to next net call
            embedding=init_hiddens,
        )

        invalid_actions = None  # jax.vmap(get_invalid_actions)(rollout_states.env_state)

        def mcts_recurrent_fn(
            _: None,  # closure net
            key: Key[Array, ""],
            actions: Integer[Array, " num_envs"],
            hiddens: Float[Array, " num_envs rnn_size"],
        ):
            """Returns the logits and value for the newly created node."""
            hiddens, rewards = jax.vmap(net.world_model.step)(hiddens, actions, goal_embeddings)
            preds = jax.vmap(net.actor_critic)(hiddens, goal_embeddings)
            return mctx.RecurrentFnOutput(
                reward=logits_to_value(rewards),
                # TODO termination
                discount=jnp.full(actions.shape[0], config.bootstrap.discount),
                prior_logits=preds.policy_logits,
                value=logits_to_value(preds.value_logits),
            ), hiddens

        out = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=mcts_recurrent_fn,
            num_simulations=config.collection.num_mcts_simulations,
            invalid_actions=invalid_actions,
            max_depth=config.collection.mcts_depth,
        )

        return preds, out

    def bootstrap(
        net: MuZeroNetwork,
        trajectory: Batched[Transition, " horizon"],
    ) -> tuple[
        Batched[Prediction, " horizon horizon"],
        Float[Array, " horizon horizon-1"],
    ]:
        """Abstracted out for plotting.

        Note that we use `Timestep.discount` rather than `Timestep.step_type`.
        """
        action_rolled, reward_rolled, discount_rolled = map(
            roll_into_matrix,
            (trajectory.action, trajectory.timestep.reward, trajectory.timestep.discount),
        )

        _, (_, pred) = jax.vmap(net)(trajectory.timestep.obs, action_rolled)
        value = logits_to_value(pred.value_logits)
        bootstrapped_return: Float[Array, " horizon horizon-1"] = jax.vmap(
            rlax.lambda_returns, (0, 0, 0, None)
        )(
            jnp.nan_to_num(reward_rolled)[:, 1:],
            jnp.nan_to_num(discount_rolled)[:, 1:] * config.bootstrap.discount,
            value[:, 1:],
            config.bootstrap.lambda_gae,
        )

        return pred, bootstrapped_return

    # jit to ease debugging
    @ft.partial(jax.jit, static_argnames=("net_static",))
    @ft.partial(jax.vmap, in_axes=(None, None, 0, None))
    def loss_trajectory(
        params: MuZeroNetwork,
        target_params: MuZeroNetwork,
        trajectory: Batched[Transition, " horizon"],
        net_static: MuZeroNetwork,
    ):
        """MuZero loss.

        Here we align the targets at each timestep to maximize signal.
        """
        net = eqx.combine(params, net_static)
        target_net = eqx.combine(target_params, net_static)

        # e.g. online_value_logits[i, j] is the array of predicted value logits at time i+j,
        # based on the observation at time i
        # o0 | a0 a1 ...
        # o1 | a1 a2 ...
        action_rolled = roll_into_matrix(trajectory.action)
        _, (online_reward, online_pred) = jax.vmap(net)(trajectory.timestep.obs, action_rolled)

        # bootstrap target values
        # discard final transition
        # (too much hassle to pass around final transition)
        target_pred, bootstrapped_return = bootstrap(target_net, trajectory)

        # value loss
        online_value = logits_to_value(online_pred.value_logits)
        if config.value.num_value_bins == "scalar":
            value_losses = rlax.l2_loss(
                predictions=online_value[:-1, :-1],
                targets=bootstrapped_return[:-1, :],
            )
        else:
            bootstrapped_value_probs = value_to_probs(bootstrapped_return)
            value_losses = distrax.Categorical(
                probs=bootstrapped_value_probs[:-1, :],
            ).cross_entropy(distrax.Categorical(logits=online_pred.value_logits[:-1, :-1]))

        # policy loss
        mcts_probs_rolled = roll_into_matrix(trajectory.mcts_probs)
        policy_losses = distrax.Categorical(
            probs=mcts_probs_rolled,
        ).cross_entropy(distrax.Categorical(logits=online_pred.policy_logits))

        # reward model loss
        reward_rolled = roll_into_matrix(trajectory.timestep.reward)
        if config.value.num_value_bins == "scalar":
            reward_losses = rlax.l2_loss(
                predictions=online_reward[:-1, :-1], targets=reward_rolled[1:, 1:]
            )
        else:
            reward_losses = distrax.Categorical(
                probs=value_to_probs(reward_rolled[1:, 1:]),
            ).cross_entropy(distrax.Categorical(logits=online_reward[:-1, :-1]))

        # top left triangle of matrix
        horizon_axis = jnp.arange(trajectory.action.size)
        mask = horizon_axis[:, jnp.newaxis] + horizon_axis[jnp.newaxis, :]
        mask = horizon_axis.size - mask
        mask = mask / (mask.sum(where=mask > 0))

        # multiplying by mask accounts for multiple-counting timesteps
        # and taking the mean averages across the horizon
        policy_loss = jnp.mean(policy_losses * mask, where=mask > 0)
        value_loss = jnp.mean(value_losses * mask[:-1, :-1], where=mask[:-1, :-1] > 0)
        reward_loss = jnp.mean(reward_losses * mask[:-1, :-1], where=mask[:-1, :-1] > 0)

        loss = (
            policy_loss
            + config.optim.value_coef * value_loss
            + config.optim.reward_coef * reward_loss
        )

        # logging
        mcts_dist = distrax.Categorical(probs=mcts_probs_rolled)
        online_policy_dist = distrax.Categorical(logits=online_pred.policy_logits)
        return loss, tree_slice(
            LossStatistics(
                loss=loss[jnp.newaxis],
                # policy
                mcts_entropy=mcts_dist.entropy(),
                policy_entropy=online_policy_dist.entropy(),
                policy_logits=online_pred.policy_logits,
                policy_loss=policy_losses,
                # value
                value_logits=online_pred.value_logits,
                target_value_logits=target_pred.value_logits,
                bootstrapped_return=bootstrapped_return,
                value_loss=value_losses,
                td_error=bootstrapped_return[:-1, :] - online_value[:-1, :-1],
            ),
            0,
        )

    def evaluate(
        params: MuZeroNetwork,
        target_params: MuZeroNetwork,
        buffer_state: BufferState,
        key: Key[Array, ""],
        *,
        num_envs: int,
        net_static: MuZeroNetwork,
    ) -> Float[Array, ""]:
        """Evaluate a batch of rollouts using the raw policy.

        Returns mean return across the batch.
        """
        key_reset, key_rollout, key_sample = jr.split(key, 3)

        net = eqx.combine(params, net_static)

        key_reset = jr.split(key_reset, num_envs)
        init_timesteps = env_reset_batch(env_params, key=key_reset)

        @exec_loop(init_timesteps, config.eval.eval_horizon, key_rollout)
        def rollout_step(timesteps: Batched[Timestep, " num_envs"], key: Key[Array, ""]):
            key_action, key_step, key_mcts = jr.split(key, 3)

            @ft.partial(jax.value_and_grad, has_aux=True)
            def predict(obs: Float[Array, " *obs_size"], goal: UInt[Array, ""]):
                """Differentiate with respect to value to obtain saliency map."""
                hidden = net.projection(obs)
                goal_embedding = net.embed_goal(goal)
                pred = net.actor_critic(hidden, goal_embedding)
                value = logits_to_value(pred.value_logits[jnp.newaxis])[0]
                return value, pred.policy_logits

            (_, policy_logits), obs_grads = jax.vmap(predict)(timesteps.obs.obs, timesteps.obs.goal)
            actions = jr.categorical(key_action, logits=policy_logits, axis=-1)
            next_timesteps = env_step_batch(
                timesteps.state, actions, env_params, key=jr.split(key_step, num_envs)
            )

            preds, outs = act_mcts(net, timesteps, key_mcts)

            return next_timesteps, (
                Transition(
                    timestep=timesteps,
                    action=actions,
                    pred=preds,
                    mcts_probs=outs.action_weights,
                ),
                obs_grads,
            )

        rollout_states, (trajectories, obs_grads) = rollout_step
        trajectories, obs_grads = jax.tree.map(
            lambda x: x.swapaxes(0, 1), (trajectories, obs_grads)
        )
        trajectories: Batched[Transition, " num_envs horizon"]

        target_net = eqx.combine(target_params, net_static)
        _, bootstrapped_returns = jax.vmap(bootstrap, in_axes=(None, 0))(target_net, trajectories)

        jax.debug.callback(
            visualize_callback,
            trajectories,
            bootstrapped_returns[:, 0],
            jax.vmap(visualize_catch)(trajectories.timestep.state, obs_grads)
            if config.env.env_name in ["Catch-bsuite", "MultiCatch"]
            else None,
            prefix="eval",
        )

        metrics: Batched[Metrics, " num_envs"] = trajectories.timestep.info["metrics"]
        eval_return = jnp.mean(
            metrics.episode_return,
            where=trajectories.timestep.step_type == StepType.LAST,
        )
        log_values({"eval/mean_return": eval_return})
        jax.debug.print("eval mean return {}", eval_return)

        # debug training procedure
        batch = jax.vmap(buffer.sample, in_axes=(None,))(
            buffer_state, key=jr.split(key_sample, num_envs)
        )
        sampled_trajectories = batch.experience
        _, aux = loss_trajectory(params, target_params, sampled_trajectories, net_static)

        jax.debug.callback(
            visualize_callback,
            trajectories=sampled_trajectories._replace(
                pred=Prediction(value_logits=aux.value_logits, policy_logits=aux.policy_logits)
            ),
            bootstrapped_returns=aux.bootstrapped_return,
            video=jax.vmap(visualize_catch)(sampled_trajectories.timestep.state)
            if config.env.env_name in ["Catch-bsuite", "MultiCatch"]
            else None,
            prefix="visualize",
            priorities=batch.priority,
        )

        return eval_return

    def visualize_callback(
        trajectories: Batched[Transition, " num_envs horizon"],
        bootstrapped_returns: Float[Array, " num_envs horizon-1"],
        video: Float[Array, " num_envs horizon 3 height width"] | None,
        prefix: str,
        priorities: Float[Array, " num_envs"] | None = None,
    ) -> None:
        """Visualize a batch of trajectories.

        Args:
            trajectories (Transition (num_envs, horizon)): The batch of trajectories.
            bootstrapped_returns (Float (num_envs, horizon)): The bootstrapped target returns.
            video (Float (num_envs, horizon, 3, height, width)): A batch of sequences of images to imshow.
            prefix (str): The wandb prefix to log under.
            priorities (Float (num_envs), optional): The priorities of the trajectories. Defaults to None.
        """
        if not wandb.run or wandb.run.disabled:
            return

        fig_width = 3
        rows_per_env = 2
        if config.value.num_value_bins != "scalar":
            rows_per_env += 1
        num_envs, horizon = trajectories.reward.shape

        fig_stats, ax = plt.subplots(
            nrows=num_envs * rows_per_env,
            figsize=(2 * fig_width, 4 * num_envs * rows_per_env),
        )
        for i in range(num_envs):
            j = 0

            ax_stats: Axes = ax[rows_per_env * i + j]
            j += 1
            ax_stats.set_title(
                f"T{i}" + (f" priority {priorities[i]:.2f}" if priorities is not None else "")
            )
            plot_statistics(ax_stats, tree_slice(trajectories, i), bootstrapped_returns[i])

            ax_policy: Axes = ax[rows_per_env * i + j]
            j += 1
            ax_policy.set_title(f"Trajectory {i} Policy and MCTS")
            plot_compare_dists(
                ax_policy,
                jax.nn.softmax(trajectories.pred.policy_logits[i], axis=-1),
                trajectories.mcts_probs[i],
            )

            if config.value.num_value_bins != "scalar":
                ax_value: Axes = ax[rows_per_env * i + j]
                j += 1
                ax_value.set_title(f"Trajectory {i} Value and Bootstrapped")
                plot_compare_dists(
                    ax_value,
                    jax.nn.softmax(trajectories.pred.value_logits[i, :-1], axis=-1),
                    value_to_probs(bootstrapped_returns[i]),
                )

            # legend above first axes
            if i == 0:
                # move legend to right of plot
                box = ax_stats.get_position()
                ax_stats.set_position(
                    [
                        box.x0,
                        box.y0,
                        box.width * 0.8,
                        box.height,
                    ]
                )
                ax_stats.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                ax_policy.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3)
                if config.value.num_value_bins != "scalar":
                    ax_value.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3)

        # use wandb.Image since otherwise wandb uses plotly,
        # which breaks the legends
        obj = {f"{prefix}/statistics": wandb.Image(fig_stats)}

        # separate figure for plotting video
        if video is not None:
            fig_video, ax_video = plt.subplots(
                nrows=num_envs,
                ncols=horizon,
                squeeze=False,
                figsize=(horizon * 2, num_envs * 3),
            )

            # wandb.Video(np.asarray(video), fps=10),

            for i in range(num_envs):
                for h in range(horizon):
                    ax_stats: Axes = ax_video[i, h]
                    step_type = trajectories.timestep.step_type[i, h].item()

                    if step_type == StepType.FIRST:
                        step_type = "F"
                    elif step_type == StepType.MID:
                        step_type = "M"
                    elif step_type == StepType.LAST:
                        step_type = "L"
                    a = get_action_name(trajectories.action[i, h].item())
                    # c h w -> h w c
                    ax_stats.imshow(jnp.permute_dims(video[i, h], (1, 2, 0)))
                    ax_stats.grid(True)
                    ax_stats.xaxis.set_major_locator(plt.MultipleLocator(1))
                    ax_stats.yaxis.set_major_locator(plt.MultipleLocator(1))
                    ax_stats.set_title(f"{h=} {step_type} {a=}")
                    ax_stats.axis("off")
            obj[f"{prefix}/trajectories"] = wandb.Image(fig_video)

        wandb.log(obj)
        plt.close(fig_stats)
        if video is not None:
            plt.close(fig_video)

    def plot_statistics(
        ax: Axes,
        trajectory: Batched[Transition, " horizon"],
        bootstrapped_return: Float[Array, " horizon"],
    ):
        horizon = jnp.arange(trajectory.action.size)

        # misc
        initial_idx = horizon[trajectory.timestep.step_type]
        ax.plot(initial_idx, jnp.zeros_like(initial_idx), "k>", label="initial", alpha=0.5)

        # value
        online_value = logits_to_value(trajectory.pred.value_logits)
        ax.plot(horizon, trajectory.reward, "r+", label="reward")
        ax.plot(horizon, online_value, "bo", label="online value")
        ax.plot(horizon[:-1], bootstrapped_return, "mo", label="bootstrapped returns", alpha=0.5)
        ax.fill_between(
            horizon[:-1], bootstrapped_return, online_value[:-1], alpha=0.3, label="TD error"
        )

        # entropy
        policy_dist = distrax.Categorical(logits=trajectory.pred.policy_logits)
        mcts_dist = distrax.Categorical(probs=trajectory.mcts_probs)
        ax.plot(horizon, policy_dist.entropy(), "b:", label="policy entropy")
        ax.plot(horizon, mcts_dist.entropy(), "g:", label="MCTS entropy")
        if config.value.num_value_bins != "scalar":
            value_dist = distrax.Categorical(logits=trajectory.pred.value_logits)
            ax.plot(horizon, value_dist.entropy(), "y:", label="value entropy")

        # loss
        ax.plot(horizon, mcts_dist.kl_divergence(policy_dist), "c--", label="policy / mcts kl")
        if config.value.num_value_bins != "scalar":
            bootstrapped_dist = distrax.Categorical(probs=value_to_probs(bootstrapped_return))
            value_loss = bootstrapped_dist.kl_divergence(value_dist[:-1])
            ax.plot(horizon[:-1], value_loss, "r--", label="value / bootstrap kl")
        else:
            value_loss = rlax.l2_loss(online_value[:-1], bootstrapped_return)
            ax.plot(horizon[:-1], value_loss, "r--", label="value / bootstrap l2")

        ax.set_xticks(horizon, horizon)
        ax.set_ylim(config.value.min_value, config.value.max_value)

    def plot_compare_dists(
        ax: Axes, p0: Float[Array, " horizon n"], p1: Float[Array, " horizon n"]
    ):
        chex.assert_equal_shape([p0, p1])
        horizon, num_actions = p0.shape
        x = jnp.arange(horizon)

        ax.stackplot(
            x,
            jnp.concat([p0, p1], axis=1).T,
            labels=[f"Policy {a}" for a in range(num_actions)]
            + [f"MCTS {a}" for a in range(num_actions)],
        )

        ax.set_xticks(x, x)
        ax.set_ylim(0, 2)

    return train


if __name__ == "__main__":
    main(TrainConfig, make_train, Path(__file__).name)
