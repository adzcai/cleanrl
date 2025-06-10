"""Implementation of MuZero with a recurrent actor-critic network."""

import dataclasses as dc
import functools as ft
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated as Batched
from typing import NamedTuple

import chex
import distrax
import dm_env.specs as specs
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mctx
import optax
import rlax
import yaml
from jaxtyping import Array, Bool, Float, Integer, Key, PyTree, UInt

from experiments.config import ArchConfig, TrainConfig, main
from utils.rl_utils import bootstrap, roll_into_matrix
from utils.structures import Prediction, StepType, TEnvState, Timestep, TObs, Transition
from utils.goal_wrapper import GoalObs
from utils.housemaze import HouseMazeObs
from utils.log import Metrics
from utils.log_utils import (
    exec_loop,
    get_norm_data,
    log_values,
    print_bytes,
    scale_gradient,
    tree_slice,
    typecheck,
)
from utils.prioritized_buffer import BufferState, PrioritizedBuffer
from utils.translate import (
    SUPPORTED_VIDEO_ENVS,
    make_env,
    visualize_env_state_frame,
)
from utils.visualize import visualize_callback


class MLPConcatArgs(eqx.nn.MLP):
    """An MLP that concatenates its arguments.

    Useful so that all embeddings can be passed as projection(goal, obs).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: callable,
        *,
        key: Key[Array, ""],
    ):
        """Initialize the MLP with concatenated inputs.

        Args:
            in_size: Size of each input after concatenation
            out_size: Size of the output
            width_size: Width of the hidden layers
            depth: Number of hidden layers
            activation: Activation function to use
            key: Random key for initialization
        """
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )

    def __call__(self, *x, key=None):
        return super().__call__(jnp.concat(x), key=key)


class WorldModelRNN(eqx.Module):
    """Parameterizes the recurrent dynamics model and reward function.

    We use `hidden_dyn` to mean the dynamics model recurrent state
    and `hidden_obs` for the recurrent policy's state (see `ActorCritic`).
    """

    linear: eqx.nn.Linear
    """The recurrent cell for the dynamics model."""
    reward_head: eqx.nn.Linear
    """The reward model."""
    norm1: eqx.nn.LayerNorm
    """Layer normalization for the state-action representation."""
    norm2: eqx.nn.LayerNorm
    """Layer normalization for the MLP."""
    embed_action: eqx.nn.Embedding
    """Embed actions to `rnn_size`."""
    mlp: eqx.nn.MLP
    """MLP embedding."""
    gradient_scale: float
    """Scale the gradients at each step."""

    def __init__(
        self,
        config: ArchConfig,
        num_actions: int,
        num_value_bins: int,
        gradient_scale: float,
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_reward, key_embed_action, key_mlp = jr.split(key, 4)
        self.embed_action = eqx.nn.Embedding(
            num_actions, config.rnn_size, key=key_embed_action
        )
        self.norm1 = eqx.nn.LayerNorm(config.rnn_size)
        self.norm2 = eqx.nn.LayerNorm(config.rnn_size)
        self.linear = eqx.nn.Linear(
            config.rnn_size, config.rnn_size, use_bias=False, key=key_cell
        )
        self.reward_head = eqx.nn.Linear(
            config.rnn_size + config.goal_dim, num_value_bins, key=key_reward
        )
        # TODO use haiku initialization
        self.mlp = eqx.nn.MLP(
            in_size=config.rnn_size,
            out_size=config.rnn_size,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            activation=getattr(jax.nn, config.activation),
            key=key_mlp,
        )
        self.gradient_scale = gradient_scale

    @typecheck
    def step(
        self,
        hidden_dyn: Float[Array, " rnn_size"],
        action: Integer[Array, ""],
        goal_embedding: Float[Array, " goal_dim"],
    ) -> tuple[Float[Array, " rnn_size"], Float[Array, " num_value_bins"]]:
        """An imagined state transition (`env.step`).

        Transitions to the next hidden state and emits predicted reward from the transition.
        Refer to the `Transition` module in `neurorl/library/muzero_mlps.py`.
        """
        # TODO handle continue predictor
        out = jax.nn.relu(hidden_dyn) + self.embed_action(action)
        out = hidden_dyn + self.linear(self.norm1(out))  # "fake env.step"
        out = out + self.mlp(self.norm2(out))
        # TODO also use resnet-like structure for policy
        out = scale_gradient(out, self.gradient_scale)
        reward = self.reward_head(jnp.concat([out, goal_embedding]))
        return out, reward

    def init_hidden(self):
        return jnp.zeros(self.linear.in_features)


class ActorCritic(eqx.Module):
    """Parameterizes the actor-critic networks."""

    policy_head: MLPConcatArgs
    """Policy network. `(hidden_dyn, goal) -> action distribution`"""
    value_head: MLPConcatArgs
    """Value network. `(hidden_dyn, goal) -> value distribution`"""

    def __init__(
        self,
        config: ArchConfig,
        num_actions: int,
        num_value_bins: int,
        *,
        key: Key[Array, ""],
    ):
        key_policy, key_value = jr.split(key)
        self.policy_head = MLPConcatArgs(
            config.rnn_size + config.goal_dim,
            num_actions,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_policy,
        )
        self.value_head = MLPConcatArgs(
            config.rnn_size + config.goal_dim,
            num_value_bins,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_value,
        )

    def __call__(
        self,
        hidden: Float[Array, " rnn_size"],
        goal_embedding: Float[Array, " goal_dim"],
    ):
        """Predict action and value logits from the hidden state."""
        return Prediction(
            policy_logits=self.policy_head(hidden, goal_embedding),
            value_logits=self.value_head(hidden, goal_embedding),
        )


class HouseMazeEmbedding(eqx.Module):
    """Embed a HouseMaze observation.

    Based on `preplay-AI/networks/CategoricalHouzemazeObsEncoder`.
    """

    dim_offset: Integer[Array, " num_leaves"] = eqx.field(static=True)
    """Offsets of the observation components in the concatenated encoding."""
    embed_combined: eqx.nn.Embedding
    """Embedding matrix for the concatenated encoding. (total_dims,) -> (obs_dim,)"""
    norm: eqx.nn.LayerNorm
    """Normalization of embedding."""
    mlp: MLPConcatArgs
    """Combine the embeddings into a single hidden state via projection."""

    def __init__(
        self,
        config: ArchConfig,
        obs_spec: PyTree[specs.BoundedArray],
        *,
        key: Key[Array, ""],
    ):
        key_embed, key_mlp = jr.split(key)

        leaf_specs: list[specs.BoundedArray] = jax.tree.leaves(obs_spec)
        num_values = [spec.maximum - spec.minimum + 1 for spec in leaf_specs]
        total_dims = int(sum(num_values))  # number of distinct categories
        self.dim_offset = jnp.cumsum(jnp.asarray([0] + num_values[:-1]))
        self.embed_combined = eqx.nn.Embedding(
            total_dims, config.obs_dim, key=key_embed
        )

        obs_flattened_size = sum(math.prod(spec.shape) for spec in leaf_specs)
        assert config.obs_dim is not None, "obs_dim must be set in the config"
        self.norm = eqx.nn.LayerNorm(config.obs_dim)  # we map over the components
        self.mlp = MLPConcatArgs(
            in_size=obs_flattened_size * config.obs_dim + config.goal_dim,
            out_size=config.rnn_size,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            activation=getattr(jax.nn, config.activation),
            key=key_mlp,
        )

    def __call__(
        self,
        obs: HouseMazeObs,
        goal_embedding: Float[Array, " goal_dim"],
        *,
        key: Key[Array, ""] | None = None,
    ):
        leaves = jax.tree.leaves(obs)
        obs_flat = jnp.concat(
            [jnp.ravel(x + offset) for x, offset in zip(leaves, self.dim_offset)]
        )
        obs_embedding: Float[Array, " obs_flattened_size obs_dim"] = jax.vmap(
            self.embed_combined
        )(obs_flat)
        obs_embedding = jax.vmap(self.norm)(obs_embedding)
        obs_embedding = self.mlp(jnp.ravel(obs_embedding), goal_embedding)
        return obs_embedding


class CNNEmbedding(eqx.Module):
    cnn: eqx.nn.Conv2d
    """Embed an observation using a CNN projection."""
    projection: eqx.nn.Linear

    def __init__(self, obs_size, mlp_size, rnn_size, *, key: Key[Array, ""]):
        key_cnn, key_linear = jr.split(key)
        # TODO implement cnn projection
        # (C, H, W)
        self.cnn = eqx.nn.Conv2d(
            in_channels=obs_size,
            out_channels=mlp_size,
            kernel_size=3,
            key=key_cnn,
        )
        flat_size = self.cnn(jnp.empty(obs_size)).size
        self.projection = eqx.nn.Linear(
            in_features=flat_size, out_features=rnn_size, key=key_linear
        )

    def __call__(self, obs, goal_embedding: Float[Array, " goal_dim"]):
        """Embed the observation and goal into a recurrent state."""
        obs_embedding = jnp.ravel(self.cnn(obs))
        obs_embedding = self.projection(obs_embedding)
        return jnp.concat([obs_embedding, goal_embedding])


class MuZeroNetwork(eqx.Module):
    """The entire MuZero network parameters."""

    projection: Callable[[PyTree[Array], PyTree[Array]], Array]
    """Embed the observation into world model recurrent state."""
    world_model: WorldModelRNN
    """The recurrent world model."""
    actor_critic: ActorCritic
    """The actor and critic networks."""
    embed_goal: eqx.nn.Embedding

    def __init__(
        self,
        config: ArchConfig,
        obs_spec: PyTree[specs.Array],
        num_actions: int,
        num_value_bins: int,
        num_goals: int,
        world_model_gradient_scale: float,
        *,
        key: Key[Array, ""],
    ):
        key_projection, key_world_model, key_actor_critic, key_embed_goal = jr.split(
            key, 4
        )

        if config.projection_kind == "mlp":
            # ensure flattened input
            obs_size = obs_spec.shape
            assert len(obs_size) == 1, f"MLP expects flattened input. Got {obs_size}"
            self.projection = MLPConcatArgs(
                in_size=obs_size[0] + config.goal_dim,
                out_size=config.rnn_size,
                width_size=config.mlp_size,
                depth=config.mlp_depth,
                activation=getattr(jax.nn, config.activation),
                key=key_projection,
            )
        elif config.projection_kind == "cnn":
            self.projection = CNNEmbedding(
                obs_spec.shape, config.mlp_size, config.rnn_size, key=key_projection
            )
        elif config.projection_kind == "housemaze":
            self.projection = HouseMazeEmbedding(
                config=config, obs_spec=obs_spec, key=key_projection
            )

        self.world_model = WorldModelRNN(
            config=config,
            num_actions=num_actions,
            num_value_bins=num_value_bins,
            gradient_scale=world_model_gradient_scale,
            key=key_world_model,
        )
        self.actor_critic = ActorCritic(
            config=config,
            num_actions=num_actions,
            num_value_bins=num_value_bins,
            key=key_actor_critic,
        )
        if False:
            # initialize the goal embedder like a word embedding matrix
            # standard normal truncated
            linear = eqx.nn.Embedding(num_goals, config.goal_dim, key=jr.key(0))
            weights = jr.truncated_normal(
                key_embed_goal, -1.96, 1.96, linear.weight.shape
            )
            self.embed_goal = eqx.tree_at(lambda x: x.weight, linear, weights)
        else:
            # default initialization
            self.embed_goal = eqx.nn.Embedding(
                num_goals, config.goal_dim, key=key_embed_goal
            )

    # @typecheck
    def __call__(
        self,
        goal_obs: GoalObs[Float[Array, " *obs_size"]],
        action: Integer[Array, " horizon"],
    ) -> tuple[
        Float[Array, " rnn_size"],
        tuple[Float[Array, " horizon"], Batched[Prediction, " horizon"]],
    ]:
        """Roll out an imagined trajectory by taking the actions from goal_obs.

        Args:
            goal_obs (GoalObs[Float (*obs_size,)]): The observation to roll out from. The goal is preserved throughout.
            action (Integer (horizon,)): The series of actions to take.

        Returns:
            tuple[Float (rnn_size,), tuple[Float (horizon,), Prediction (horizon,)]]: The resulting hidden world state;
                The predicted rewards for taking the given actions (including from `goal_obs`);
                The predicted policy logits and values at the imagined states (including `goal_obs`).
        """
        goal_embedding = self.embed_goal(goal_obs.goal)
        if isinstance(self.projection, HouseMazeEmbedding):
            init_hidden_dyn = self.projection(
                goal_obs.obs,  # type: ignore
                goal_embedding,
            )
        else:
            init_hidden_dyn = self.projection(goal_obs.obs, goal_embedding)
        return jax.lax.scan(
            lambda hidden_dyn, action: self.step(hidden_dyn, action, goal_embedding),
            init_hidden_dyn,
            action,
        )

    @typecheck
    def step(
        self,
        hidden_dyn: Float[Array, " rnn_size"],
        action: Integer[Array, ""],
        goal_embedding: Float[Array, " goal_dim"],
    ) -> tuple[
        Float[Array, " rnn_size"], tuple[Float[Array, " num_value_bins"], Prediction]
    ]:
        """Evaluate the actor-critic on `hidden` and take one imagined step using the world model.

        Args:
            hidden (Float (rnn_size,)): The hidden world state.
            action (int): The action taken from the represented state.
            goal (int): The current goal.

        Returns:
            tuple[Float[Array, " rnn_size"], tuple[Float[Array, ""], Prediction]]: The resulting hidden world state;
                The predicted reward for taking the given action;
                The predicted policy logits and value of `hidden`.
        """
        # First update the hidden state through the world model
        hidden_dyn, reward = self.world_model.step(hidden_dyn, action, goal_embedding)

        # Then make predictions based on the updated hidden state
        pred = self.actor_critic(hidden_dyn, goal_embedding)

        # Scale gradients to prevent vanishing/exploding
        hidden_dyn = scale_gradient(hidden_dyn, self.world_model.gradient_scale)

        return hidden_dyn, (reward, pred)


class ParamState(NamedTuple):
    """Carried during optimization."""

    params: MuZeroNetwork
    opt_state: optax.OptState
    buffer_state: BufferState
    """Gets updated within SGD due to priority sampling"""


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
    mcts_entropy: Float[Array | chex.Array, " horizon"]
    policy_entropy: Float[Array | chex.Array, " horizon"]
    policy_logits: Float[Array, " horizon num_actions"]
    policy_loss: Float[Array | chex.Array, " horizon"]
    # value
    value_logits: Float[Array, " horizon num_value_bins"]
    target_value_logits: Float[Array, " horizon num_value_bins"]
    bootstrapped_return: Float[Array, " horizon-1"]
    value_loss: Float[Array | chex.Array, " horizon"]
    td_error: Float[Array, " horizon"]


def make_train(config: TrainConfig):
    num_iters = config.collection.total_transitions // (
        config.collection.num_envs * config.collection.num_timesteps
    )
    num_grad_updates = num_iters * config.optim.num_minibatches
    eval_freq = num_iters // config.eval.num_evals

    # construct environment
    env, env_params = make_env(config.env)
    action_space: specs.DiscreteArray = env.action_space(env_params)
    num_actions = action_space.num_values
    num_goals = env.goal_space(env_params).num_values
    action_dtype = action_space.dtype

    env_reset_batch = jax.vmap(env.reset, in_axes=(None,))
    env_step_batch = jax.vmap(
        env.step, in_axes=(0, 0, None)
    )  # map over state and action

    lr = optax.warmup_exponential_decay_schedule(
        init_value=1e-5,
        peak_value=config.optim.lr_init,
        warmup_steps=int(config.optim.warmup_frac * num_grad_updates),
        transition_steps=(
            ((1 - config.optim.warmup_frac) * num_grad_updates)
            // config.optim.num_stairs
        ),
        decay_rate=config.optim.decay_rate,
        staircase=True,
    )
    optim = optax.chain(
        optax.contrib.ademamix(lr, weight_decay=1e-4),
        optax.clip_by_global_norm(config.optim.max_grad_norm),
    )

    buffer_time_axis = int(
        config.collection.total_transitions
        / (config.collection.num_envs * config.collection.buffer_size_denominator)
    )
    buffer_args = dict(
        batch_size=config.collection.num_envs,
        max_time=buffer_time_axis,
        horizon=config.optim.num_timesteps,
    )
    buffer = PrioritizedBuffer.new(**buffer_args)

    assert (
        num_value_bins := config.value.num_value_bins
    ) != "scalar", "num_value_bins must be a dict with value parameters"

    if False:

        def get_invalid_actions(
            env_state: gymnax.EnvState,
        ) -> Bool[Array, " num_actions"]:
            """TODO currently not used"""
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

        print(env.fullname)
        print("=" * 25)
        yaml.safe_dump(dc.asdict(config), sys.stdout)
        print("=" * 25)
        yaml.safe_dump(buffer_args, sys.stdout)
        print("=" * 25)

        init_timesteps = env_reset_batch(
            env_params, key=jr.split(key_reset, config.collection.num_envs)
        )
        # initialize all state
        init_net = MuZeroNetwork(
            config=config.arch,
            obs_spec=env.observation_space(env_params),
            num_actions=num_actions,
            num_value_bins=num_value_bins,
            num_goals=num_goals,
            world_model_gradient_scale=config.optim.world_model_gradient_scale,
            key=key_net,
        )

        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)
        init_hiddens = jnp.broadcast_to(
            init_net.world_model.init_hidden(),
            (config.collection.num_envs, config.arch.rnn_size),
        )

        init_transition = Transition(
            timestep=tree_slice(init_timesteps, 0),
            action=jnp.empty((), action_dtype),
            pred=init_net.actor_critic(
                init_hiddens[0], init_net.embed_goal(init_timesteps.obs.goal[0])
            ),
            mcts_probs=jnp.empty(num_actions, init_hiddens.dtype),
        )
        init_buffer_state = buffer.init(init_transition)
        print("buffer size (bytes)")
        print_bytes(init_buffer_state)

        @exec_loop(num_iters)
        def iterate(
            iter_state=IterState(
                step=jnp.int_(0),
                timesteps=init_timesteps,
                param_state=ParamState(
                    params=init_params,
                    opt_state=optim.init(init_params),
                    buffer_state=init_buffer_state,
                ),
                target_params=init_params,
            ),
            key=key_iter,
        ):
            """A single iteration of optimization.

            1. Collect a batch of rollouts in parallel and add it to the buffer.
            2. Run a few epochs of local optimization using sampled data from the buffer.
            """
            key_rollout, key_optim, key_evaluate = jr.split(key, 3)

            # collect data
            next_timesteps, trajectories = rollout(
                eqx.combine(iter_state.param_state.params, net_static),
                iter_state.timesteps,
                key=key_rollout,
            )
            buffer_state = buffer.add(iter_state.param_state.buffer_state, trajectories)

            metrics: Metrics = trajectories.timestep.info["metrics"]
            mean_return = jnp.mean(
                metrics.cum_return, where=trajectories.timestep.is_last
            )
            mean_length = jnp.mean(metrics.step, where=trajectories.timestep.is_last)
            log_values(
                {
                    "iter/step": iter_state.step,
                    "iter/mean_return": mean_return,
                    "iter/mean_length": mean_length,
                }
            )

            buffer_available = jnp.bool_(
                buffer.num_available(buffer_state) >= config.optim.batch_size
            )

            @exec_loop(config.optim.num_minibatches, cond=buffer_available)
            def optimize_step(
                param_state=iter_state.param_state._replace(buffer_state=buffer_state),
                key=key_optim,
            ):
                """Sample batch of trajectories and do a single 'SGD step'"""
                key_sample, key_reanalyze = jr.split(key, 2)
                batch = jax.vmap(buffer.sample, in_axes=(None, None))(
                    param_state.buffer_state,
                    config.eval.warnings,
                    key=jr.split(key_sample, config.optim.batch_size),
                )
                trajectories: Batched[Transition, " batch_size horizon"] = (
                    batch.experience
                )

                # if False:
                # "reanalyze" the policy targets via mcts with target parameters
                horizon = trajectories.action.shape[1]
                net = eqx.combine(iter_state.target_params, net_static)
                _, mcts_out = jax.vmap(act_mcts, in_axes=(None, 1), out_axes=1)(
                    net, trajectories.timestep, key=jr.split(key_reanalyze, horizon)
                )
                trajectories = trajectories._replace(mcts_probs=mcts_out.action_weights)

                if False:

                    def debug(batch, free):
                        """Flashbax returns zero when JAX_ENABLE_X64 is not set."""
                        trajectories = batch.experience
                        for i in range(config.optim.batch_size):
                            weights = trajectories.mcts_probs[i]
                            if not jnp.allclose(
                                jnp.sum(weights, axis=-1), 1
                            ) and not jnp.all(jnp.isnan(weights)):
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
                    return jnp.mean(losses), aux

                    # TODO add importance reweighting
                    importance_weights = (
                        jnp.reciprocal(batch.priority)
                        ** config.optim.importance_exponent
                    )
                    importance_weights /= jnp.max(importance_weights, keepdims=True)
                    return jnp.mean(importance_weights * losses), aux

                (_, aux), grads = loss_grad(param_state.params)
                # value_and_grad drops types
                aux: Batched[LossStatistics, " batch_size"]
                grads: MuZeroNetwork
                updates, opt_state = optim.update(
                    grads, param_state.opt_state, param_state.params  # type: ignore
                )
                params: MuZeroNetwork = optax.apply_updates(param_state.params, updates)  # type: ignore

                # prioritize trajectories with high TD error
                # priorities = jnp.mean(jnp.abs(aux.td_error), axis=-1)
                # buffer_state = buffer.set_priorities(
                #     param_state.buffer_state,
                #     batch.idx,
                #     priorities**config.optim.priority_exponent,
                # )

                # Get the current learning rate from the optimizer
                log_values(
                    {
                        f"train/mean_{key}": jnp.mean(value)
                        for key, value in aux._asdict().items()
                    }
                    | get_norm_data(updates, "updates/norm")
                    | get_norm_data(params, "params/norm")
                )

                # carry updated ParamState (scanned loop)
                return (
                    ParamState(
                        params=params,
                        opt_state=opt_state,
                        buffer_state=param_state.buffer_state,
                    ),
                    None,
                )

            del buffer_state  # replaced by param_state.buffer_state
            param_state, _ = optimize_step

            # occasionally update target
            target_params = jax.tree.map(
                lambda online, target: jnp.where(
                    iter_state.step % config.bootstrap.target_update_freq == 0,
                    optax.incremental_update(
                        online, target, config.bootstrap.target_update_size
                    ),  # type: ignore
                    target,
                ),
                param_state.params,
                iter_state.target_params,
            )

            # evaluate every eval_freq steps
            eval_return = jax.lax.cond(
                (iter_state.step % eval_freq == 0) & buffer_available,
                ft.partial(
                    evaluate, num_envs=config.eval.num_eval_envs, net_static=net_static
                ),
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
                "available {available}. "
                "mean return {mean_return:.02f}. "
                "mean length {mean_length:.02f}. "
                "{num_episodes} episodes collected.",
                step=iter_state.step,
                num_iters=num_iters,
                transitions=param_state.buffer_state.pos * config.collection.num_envs,
                total_transitions=config.collection.total_transitions,
                available=buffer_available,
                mean_return=mean_return,
                mean_length=mean_length,
                num_episodes=jnp.sum(trajectories.timestep.is_last),
            )

            if config.eval.warnings:
                valid = jax.tree.map(
                    lambda x: jnp.any(jnp.isnan(x)), param_state.params
                )

                @ft.partial(jax.debug.callback, valid=valid)
                def debug(valid):
                    if not valid:
                        raise ValueError("NaN parameters.")

            return (
                IterState(
                    step=iter_state.step + 1,
                    timesteps=next_timesteps,
                    param_state=param_state,
                    target_params=target_params,
                ),
                eval_return,
            )

        final_iter_state, eval_returns = iterate
        return final_iter_state, eval_returns

    def rollout(
        net: MuZeroNetwork,
        init_timesteps: Batched[Timestep, " num_envs"],
        *,
        key: Key[Array, ""],
    ) -> tuple[
        Batched[Timestep, " num_envs"], Batched[Transition, " num_envs horizon"]
    ]:
        """Collect a batch of trajectories."""

        @exec_loop(config.collection.num_timesteps)
        def rollout_step(timesteps=init_timesteps, key=key):
            """Take a single step in the true environment using the MCTS policy."""
            key_action, key_step = jr.split(key)

            preds, outs = act_mcts(net, timesteps, key=key_action)
            actions = jnp.asarray(outs.action, dtype=action_dtype)
            next_timesteps = env_step_batch(
                timesteps.state,
                actions,
                env_params,
                key=jr.split(key_step, config.collection.num_envs),
            )
            return next_timesteps, Transition(
                # contains the reward and network predictions computed from the rollout state
                timestep=timesteps,
                action=actions,
                pred=preds,
                mcts_probs=jnp.asarray(outs.action_weights, dtype=action_dtype),
            )

        final_timestep, trajectories = rollout_step
        trajectories = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), trajectories
        )  # swap horizon and num_envs axes

        return final_timestep, trajectories

    def act_mcts(
        net: MuZeroNetwork,
        timesteps: Batched[Timestep[GoalObs[TObs], TEnvState], " num_envs"],
        *,
        key: Key[Array, ""],
    ) -> tuple[Prediction, mctx.PolicyOutput]:
        """Take a single batch of actions from the batch of timesteps using MCTS.

        Returns:
            tuple[Prediction, mctx.PolicyOutput]: The value of `timestep`
                predicted by the critic network
                and the MCTS output of planning from the timestep.
        """
        goal_embeddings = jax.vmap(net.embed_goal)(timesteps.obs.goal)
        init_hiddens = jax.vmap(net.projection)(timesteps.obs.obs, goal_embeddings)  # type: ignore
        preds = jax.vmap(net.actor_critic)(init_hiddens, goal_embeddings)

        root = mctx.RootFnOutput(
            prior_logits=preds.policy_logits,  # type: ignore
            value=config.value.logits_to_value(preds.value_logits),  # type: ignore
            # embedding contains current env_state and the hidden to pass to next net call
            embedding=init_hiddens,  # type: ignore
        )

        invalid_actions = (
            None  # jax.vmap(get_invalid_actions)(rollout_states.env_state)
        )

        @typecheck
        def mcts_recurrent_fn(
            _: None,  # closure net
            key: Key[Array, ""],
            actions: Integer[Array, " num_envs"],
            hidden_dyns: Float[Array, " num_envs rnn_size"],
        ) -> tuple[mctx.RecurrentFnOutput, Float[Array, " num_envs rnn_size"]]:
            """World model transition function.

            Returns:
                tuple[mctx.RecurrentFnOutput, Float (num_envs, rnn_size)]: The logits and value
                    computed by the actor-critic network for the newly created node;
                    The hidden state representing the state following `hiddens`.
            """
            hidden_dyns, rewards = jax.vmap(net.world_model.step)(
                hidden_dyns, actions, goal_embeddings
            )
            preds = jax.vmap(net.actor_critic)(hidden_dyns, goal_embeddings)
            return (
                mctx.RecurrentFnOutput(
                    reward=config.value.logits_to_value(rewards),  # type: ignore
                    # TODO termination
                    discount=jnp.full(actions.shape[0], config.bootstrap.discount),  # type: ignore
                    prior_logits=preds.policy_logits,  # type: ignore
                    value=config.value.logits_to_value(preds.value_logits),  # type: ignore
                ),
                hidden_dyns,
            )

        out = mctx.gumbel_muzero_policy(
            params=None,  # type: ignore
            rng_key=key,
            root=root,
            recurrent_fn=mcts_recurrent_fn,  # type: ignore
            num_simulations=config.collection.num_mcts_simulations,
            invalid_actions=invalid_actions,
            max_depth=config.collection.mcts_depth,
        )

        return preds, out

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
        _, (online_reward, online_pred) = jax.vmap(net)(
            trajectory.timestep.obs, action_rolled
        )

        # bootstrap target values
        def predict(
            obs_seq: Batched[GoalObs, " horizon"],
            action_seq: Integer[Array, " horizon"],
        ):
            """Predict the value sequence from the target network."""
            _, (_, target_pred) = target_net(obs_seq, action_seq)
            value_seq = config.value.logits_to_value(target_pred.value_logits)
            return value_seq, target_pred

        bootstrapped_return, target_pred = bootstrap(predict, trajectory, config)

        # value loss
        # cross-entropy of value predictions on n-step bootstrapped returns of target network
        online_value = config.value.logits_to_value(online_pred.value_logits)
        bootstrapped_value_probs = config.value.value_to_probs(bootstrapped_return)
        bootstrapped_value_dist = distrax.Categorical(
            probs=bootstrapped_value_probs[:-1, :]
        )
        online_value_dist = distrax.Categorical(
            logits=online_pred.value_logits[:-1, :-1]
        )
        value_losses = bootstrapped_value_dist.cross_entropy(online_value_dist)

        # policy loss
        # cross-entropy of policy predictions on MCTS action visit proportions
        # replace terminal timesteps with uniform
        mcts_probs: Float[Array, " horizon num_actions"] = jnp.where(
            trajectory.timestep.is_last[:, jnp.newaxis],
            jnp.ones_like(trajectory.mcts_probs) / num_actions,
            trajectory.mcts_probs,
        )
        mcts_dist = distrax.Categorical(probs=mcts_probs)
        online_policy_dist = distrax.Categorical(logits=online_pred.policy_logits)
        policy_losses = mcts_dist.cross_entropy(online_policy_dist)

        # reward model loss
        # online_reward[i, j] is the reward obtained from acting in state i+j
        # while reward_rolled[i, j] is the reward obtained upon entering state i+j
        reward_rolled = roll_into_matrix(trajectory.timestep.reward[1:])
        reward_dist = distrax.Categorical(
            probs=config.value.value_to_probs(reward_rolled)
        )
        online_reward_dist = distrax.Categorical(logits=online_reward[:-1, :-1])
        reward_losses = reward_dist.cross_entropy(online_reward_dist)

        # top left triangle of matrix
        horizon = trajectory.action.size
        horizon_axis = jnp.arange(horizon)
        mask = horizon_axis[:, jnp.newaxis] + horizon_axis[jnp.newaxis, :]
        mask = horizon - mask
        mask = mask / (mask.sum(where=mask > 0, keepdims=True))

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
        init_timesteps: Batched[Timestep, " num_envs"] = env_reset_batch(
            env_params, key=key_reset
        )

        @exec_loop(config.eval.num_timesteps)
        def rollout_step(timesteps=init_timesteps, key=key_rollout):
            key_action, key_step, key_mcts = jr.split(key, 3)

            @ft.partial(jax.value_and_grad, has_aux=True, allow_int=True)
            def predict(
                obs: Float[Array, " *obs_size"],
                goal: UInt[Array, ""],
                *,
                key: Key[Array, ""],
            ):
                """Differentiate with respect to value to obtain saliency map."""
                goal_embedding = net.embed_goal(goal)
                hidden = net.projection(obs, goal_embedding)
                pred = net.actor_critic(hidden, goal_embedding)
                action = jr.categorical(key, pred.policy_logits)
                # track the predicted reward for plotting
                _, reward = net.world_model.step(hidden, action, goal_embedding)
                value = config.value.logits_to_value(pred.value_logits[jnp.newaxis])[0]
                return value, (reward, action)

            (_, (reward_preds, actions)), obs_grads = jax.vmap(predict)(
                timesteps.obs.obs,
                timesteps.obs.goal,
                key=jr.split(key_action, num_envs),
            )

            next_timesteps = env_step_batch(
                timesteps.state, actions, env_params, key=jr.split(key_step, num_envs)
            )

            preds, outs = act_mcts(net, timesteps, key=key_mcts)

            return next_timesteps, (
                Transition(
                    timestep=timesteps,
                    action=actions,
                    pred=preds,
                    mcts_probs=jnp.asarray(outs.action_weights),
                ),
                obs_grads,
                reward_preds,
            )

        (trajectories, obs_grads, reward_preds) = jax.tree.map(
            lambda x: x.swapaxes(0, 1), rollout_step[1]
        )
        trajectories: Batched[Transition, " num_envs horizon"]

        target_net = eqx.combine(target_params, net_static)

        # bootstrap target values
        def predict(
            obs_seq: Batched[GoalObs, " horizon"],
            action_seq: Integer[Array, " horizon"],
        ):
            """Predict the value sequence from the target network."""
            _, (_, target_pred) = target_net(obs_seq, action_seq)
            value_seq = config.value.logits_to_value(target_pred.value_logits)
            return value_seq, None

        bootstrapped_returns, _ = jax.vmap(bootstrap, in_axes=(None, 0, None))(
            predict, trajectories, config
        )

        jax.debug.callback(
            visualize_callback,
            trajectories=trajectories,
            bootstrapped_returns=bootstrapped_returns[:, 0],
            value_cfg=config.value,
            env_name=config.env.name,
            video=(
                jax.vmap(ft.partial(visualize_env_state_frame, config.env.name))(
                    trajectories.timestep.state, maps=obs_grads
                )
                if config.env.name in SUPPORTED_VIDEO_ENVS
                else None
            ),
            prefix="eval",
            predicted_rewards=reward_preds,
        )

        metrics: Batched[Metrics, " num_envs"] = trajectories.timestep.info["metrics"]
        final_step_mask = trajectories.timestep.is_last

        def compute_goal_return(goal):
            goal_mask = final_step_mask & (trajectories.timestep.obs.goal == goal)
            return jnp.mean(metrics.cum_return, where=goal_mask)

        eval_return = jnp.mean(metrics.cum_return, where=final_step_mask)
        goal_returns = jax.vmap(compute_goal_return)(jnp.arange(num_goals))
        goal_returns_dict = {
            f"eval/mean_return/goal_{goal}": return_val
            for goal, return_val in enumerate(goal_returns)
        }
        log_values({"eval/mean_return/overall": eval_return} | goal_returns_dict)
        jax.debug.print("eval mean return {}", eval_return)

        # debug training procedure
        batch = jax.vmap(buffer.sample, in_axes=(None,))(
            buffer_state, key=jr.split(key_sample, num_envs)
        )
        sampled_trajectories = batch.experience
        _, aux = loss_trajectory(
            params, target_params, sampled_trajectories, net_static
        )
        aux: LossStatistics

        jax.debug.callback(
            visualize_callback,
            # plot the predictions
            value_cfg=config.value,
            env_name=config.env.name,
            trajectories=sampled_trajectories._replace(
                pred=Prediction(
                    value_logits=aux.value_logits,
                    policy_logits=aux.policy_logits,
                )
            ),
            bootstrapped_returns=aux.bootstrapped_return,
            video=(
                jax.vmap(ft.partial(visualize_env_state_frame, config.env.name))(
                    sampled_trajectories.timestep.state
                )
                if config.env.name in SUPPORTED_VIDEO_ENVS
                else None
            ),
            prefix="visualize",
            priorities=batch.priority,
        )

        return eval_return

    return train


if __name__ == "__main__":
    main(TrainConfig, make_train, Path(__file__).name)
