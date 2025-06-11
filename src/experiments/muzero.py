"""Implementation of MuZero with a recurrent actor-critic network."""

import dataclasses as dc
import functools as ft
import importlib.util
import math
import sys
from collections.abc import Callable
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
import yaml
from jaxtyping import Array, Bool, Float, Integer, Key, PyTree

from envs.housemaze import HouseMazeObs
from envs.translate import make_env
from experiments.config import (
    ArchConfig,
    BootstrapConfig,
    LossConfig,
    TrainConfig,
    ValueConfig,
    main,
)
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
from utils.rl_utils import bootstrap, roll_into_matrix
from utils.structures import (
    Prediction,
    TimeStep,
    TObs,
    Transition,
)
from utils.visualize import visualize
from wrappers.goal_wrapper import GoalObs


class MLPConcatArgs(eqx.nn.MLP):
    """An MLP that concatenates its arguments.

    Useful so that all embeddings can be passed as projection(goal, obs).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        # TODO maybe relu(LayerNorm(hidden_dyn)) instead
        out = jax.nn.relu(hidden_dyn) + self.embed_action(action)
        out = hidden_dyn + self.linear(self.norm1(out))  # "fake env.step"
        out = out + self.mlp(self.norm2(out))
        # TODO also use resnet-like structure for policy
        out = scale_gradient(out, self.gradient_scale)
        reward = self.reward_head(jnp.concat([out, goal_embedding]))
        return out, reward


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
    def world_model_rollout(
        self,
        goal_obs: GoalObs[Float[Array, " *obs_size"]],
        action_s: Integer[Array, " horizon"],
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
        init_hidden_dyn = self.projection(goal_obs.obs, goal_embedding)
        return jax.lax.scan(
            lambda hidden_dyn, action: self.step(hidden_dyn, action, goal_embedding),
            init_hidden_dyn,
            action_s,
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
            hidden_dyn (Float (rnn_size,)): The hidden world state.
            action (int): The action taken from the represented state.
            goal (int): The current goal.

        Returns:
            tuple[Float[Array, " rnn_size"], tuple[Float[Array, ""], Prediction]]: The resulting hidden world state;
                The predicted reward for taking the given action;
                The predicted policy logits and value of the original `hidden_dyn` state.
        """
        # Predict value and policy based on the current hidden state
        pred = self.actor_critic(hidden_dyn, goal_embedding)
        # Then update the hidden state through the world model
        next_hidden_dyn, reward = self.world_model.step(
            hidden_dyn, action, goal_embedding
        )
        # Scale gradients to prevent vanishing/exploding
        next_hidden_dyn = scale_gradient(
            next_hidden_dyn, self.world_model.gradient_scale
        )

        return next_hidden_dyn, (reward, pred)


class ParamState(NamedTuple):
    """Carried during optimization."""

    params: MuZeroNetwork
    opt_state: optax.OptState
    buffer_state: BufferState
    """Gets updated within SGD due to priority sampling"""


class IterState(NamedTuple):
    """Carried across algorithm iterations."""

    step: Integer[Array, ""]
    timestep_b: Batched[TimeStep, " num_envs"]
    param_state: ParamState
    target_params: MuZeroNetwork


# jit to ease debugging
# @ft.partial(jax.jit, static_argnames=("net_static",))
@ft.partial(jax.vmap, in_axes=(None, None, 0, None, None, None, None))
def loss_trajectory(
    params: MuZeroNetwork,
    target_params: MuZeroNetwork,
    txn_s: Batched[Transition, " horizon"],
    net_static: MuZeroNetwork,
    value_cfg: ValueConfig,
    bootstrap_cfg: BootstrapConfig,
    loss_cfg: LossConfig,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """MuZero loss.

    Here we align the targets at each timestep to maximize signal.
    The return values get logged to wandb.
    """
    # e.g. online_value_logits[i, j] is the array of predicted value logits at time i+j,
    # based on the observation at time i
    # o0 | a0 a1 ...
    # o1 | a1 a2 ...
    action_sh = roll_into_matrix(txn_s.action)
    net = eqx.combine(params, net_static)
    _, (online_reward_sh, online_pred_sh) = jax.vmap(net.world_model_rollout)(
        txn_s.time_step.obs, action_sh
    )
    del net

    # top left triangle of matrix
    # multiplying by mask accounts for multiple-counting timesteps
    # and taking the mean averages across the horizon
    horizon = txn_s.action.size
    axis_s = jnp.arange(horizon)
    mask_sh = axis_s[:, jnp.newaxis] + axis_s[jnp.newaxis, :]
    mask_sh = horizon - mask_sh
    mask_sh = mask_sh / (mask_sh.sum(where=mask_sh > 0, keepdims=True))

    def weighted_mean(
        x: Float[chex.Array, " horizon horizon"],
        mask: Float[Array, " horizon horizon"],
    ) -> Float[Array, ""]:
        """Compute the mean of the upper triangle of a matrix."""
        return jnp.mean(x * mask, where=mask > 0)

    policy_loss, policy_stats = compute_policy_loss(
        txn_s.mcts_probs,
        online_pred_sh.policy_logits,
        weighted_mean=ft.partial(weighted_mean, mask=mask_sh),
    )
    target_net = eqx.combine(target_params, net_static)
    value_loss, value_stats = compute_value_loss(
        target_net,
        txn_s,
        value_logits_sh=online_pred_sh.value_logits,
        weighted_mean=ft.partial(weighted_mean, mask=mask_sh[:-1, :-1]),
        value_cfg=value_cfg,
        bootstrap_cfg=bootstrap_cfg,
    )
    del target_net
    reward_loss, reward_stats = compute_reward_loss(
        txn_s.time_step.reward,
        online_reward_sh,
        value_cfg=value_cfg,
        weighted_mean=ft.partial(weighted_mean, mask=mask_sh[:-1, :-1]),
    )

    loss = (
        policy_loss
        + loss_cfg.value_coef * value_loss
        + loss_cfg.reward_coef * reward_loss
    )

    return (
        loss,
        dict(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            reward_loss=reward_loss,
        )
        | policy_stats
        | value_stats
        | reward_stats,
    )


def compute_value_loss(
    target_net: MuZeroNetwork,
    txn_s: Batched[Transition, " horizon"],
    value_logits_sh: Float[Array, " horizon horizon num_value_bins"],
    weighted_mean: Callable[[Float[chex.Array, " horizon horizon"]], Float[Array, ""]],
    value_cfg: ValueConfig,
    bootstrap_cfg: BootstrapConfig,
):
    """Compute the value loss.

    The cross-entropy of the value predictions
    on n-step bootstrapped returns of the target network.
    """

    def predict(
        obs: GoalObs,
        action_s: Integer[Array, " horizon"],
    ):
        """Predict the value sequence from the target network."""
        _, (_, target_pred_s) = target_net.world_model_rollout(obs, action_s)
        value_s = value_cfg.logits_to_value(target_pred_s.value_logits)
        return value_s

    boot_value_sh = bootstrap(predict, txn_s, bootstrap_cfg)
    boot_value_dist_sh = distrax.Categorical(
        probs=value_cfg.value_to_probs(boot_value_sh)
    )
    online_value_dist_sh = distrax.Categorical(logits=value_logits_sh[:-1, :-1, :])
    value_loss = weighted_mean(boot_value_dist_sh.cross_entropy(online_value_dist_sh))  # type: ignore

    # statistics
    online_value_sh = value_cfg.logits_to_value(value_logits_sh)[:-1, :-1]
    value_stats = dict(
        online_value_entropy=online_value_dist_sh.entropy(),
        online_value=online_value_sh,
        bootstrapped_value_entropy=boot_value_dist_sh.entropy(),
        bootstrapped_value=boot_value_sh,
        td_error=boot_value_sh - online_value_sh,
    )

    return value_loss, jax.tree.map(weighted_mean, value_stats)


def compute_policy_loss(
    mcts_probs_s: Float[Array, " horizon num_actions"],
    policy_logits_sh: Float[Array, " horizon horizon num_actions"],
    weighted_mean: Callable[[Float[chex.Array, " horizon horizon"]], Float[Array, ""]],
):
    """Compute the policy loss.

    The cross-entropy of the policy predictions on MCTS action visit proportions.
    Expects `mcts_probs_s` to be the MCTS distribution from the target network
    or the uniform distribution for the last step.
    """

    mcts_dist_sh = distrax.Categorical(probs=roll_into_matrix(mcts_probs_s))
    online_policy_dist_sh = distrax.Categorical(logits=policy_logits_sh)
    policy_loss = weighted_mean(mcts_dist_sh.cross_entropy(online_policy_dist_sh))

    policy_stats = dict(
        mcts_entropy=mcts_dist_sh.entropy(),
        policy_entropy=online_policy_dist_sh.entropy(),
    )

    return policy_loss, jax.tree.map(weighted_mean, policy_stats)


def compute_reward_loss(
    reward_s: Float[Array, " horizon"],
    online_reward_sh: Float[Array, " horizon horizon num_value_bins"],
    value_cfg: ValueConfig,
    weighted_mean: Callable[[Float[chex.Array, " horizon horizon"]], Float[Array, ""]],
):
    """Compute the reward loss.

    The cross-entropy of the reward predictions on the rewards obtained from experience.
    """
    # careful for off-by-one:
    # online_reward[i, j] is the reward obtained from *acting in* state i+j
    # while reward_rolled[i, j] is the reward obtained upon *entering* state i+j
    reward_sh = roll_into_matrix(reward_s[1:])
    reward_dist_sh = distrax.Categorical(probs=value_cfg.value_to_probs(reward_sh))
    online_reward_dist_sh = distrax.Categorical(logits=online_reward_sh[:-1, :-1])
    reward_loss_sh = reward_dist_sh.cross_entropy(online_reward_dist_sh)

    reward_stats = dict(reward_entropy=online_reward_dist_sh.entropy())

    return weighted_mean(reward_loss_sh), jax.tree.map(weighted_mean, reward_stats)


def get_static(config: TrainConfig):
    # construct environment
    env, env_params = make_env(config.env)
    action_space: specs.DiscreteArray = env.action_space(env_params)
    num_actions = action_space.num_values
    num_goals = env.goal_space(env_params).num_values

    net = MuZeroNetwork(
        config=config.arch,
        obs_spec=env.observation_space(env_params),
        num_actions=num_actions,
        num_value_bins=config.value.num_value_bins,
        num_goals=num_goals,
        world_model_gradient_scale=config.optim.world_model_gradient_scale,
        key=jr.key(0),
    )
    _, net_static = eqx.partition(net, eqx.is_inexact_array)

    return env, env_params, net_static


def make_train(config: TrainConfig):
    num_grad_updates = config.collection.num_iters * config.optim.num_minibatches
    eval_freq = config.collection.num_iters // config.eval.num_evals

    # construct environment
    env, env_params = make_env(config.env)
    action_space: specs.DiscreteArray = env.action_space(env_params)
    num_actions = action_space.num_values
    num_goals = env.goal_space(env_params).num_values
    action_dtype = action_space.dtype

    env_reset_b = jax.vmap(env.reset, in_axes=(None,))
    env_step_b = jax.vmap(env.step, in_axes=(0, 0, None))  # map over state and action

    lr_schedule = optax.warmup_exponential_decay_schedule(
        init_value=1e-5,
        peak_value=config.lr.lr_init,
        warmup_steps=int(config.lr.warmup_frac * num_grad_updates),
        transition_steps=int(
            ((1 - config.lr.warmup_frac) * num_grad_updates) // config.lr.num_stairs
        ),
        decay_rate=config.lr.decay_rate,
        staircase=True,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(config.optim.max_grad_norm),
        # https://optax.readthedocs.io/en/latest/getting_started.html#accessing-learning-rate
        optax.inject_hyperparams(optax.contrib.ademamix)(
            learning_rate=lr_schedule, weight_decay=1e-4
        ),
    )

    buffer_args = dict(
        batch_size=config.collection.num_envs,
        max_time=config.collection.buffer_time_axis,
        horizon=config.optim.num_time_steps,
    )
    buffer = PrioritizedBuffer.new(**buffer_args)

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

        # initialize all state
        init_net = MuZeroNetwork(
            config=config.arch,
            obs_spec=env.observation_space(env_params),
            num_actions=num_actions,
            num_value_bins=config.value.num_value_bins,
            num_goals=num_goals,
            world_model_gradient_scale=config.optim.world_model_gradient_scale,
            key=key_net,
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)

        init_timestep_b = env_reset_b(
            env_params, key=jr.split(key_reset, config.collection.num_envs)
        )
        init_transition = Transition(
            time_step=tree_slice(init_timestep_b, 0),
            action=jnp.empty((), action_dtype),
            pred=init_net.actor_critic(
                jnp.empty(config.arch.rnn_size, float),
                init_net.embed_goal(init_timestep_b.obs.goal[0]),
            ),
            mcts_probs=jnp.empty(num_actions, float),
        )
        init_buffer_state = buffer.init(init_transition)
        buffer_state_bytes = sum(
            x.nbytes for x in jax.tree.leaves(init_buffer_state) if eqx.is_array(x)
        )
        print(f"buffer size (bytes): {buffer_state_bytes}")
        print_bytes(init_buffer_state)

        @exec_loop(config.collection.num_iters)
        def iterate(
            iter_state=IterState(
                step=jnp.int_(0),
                timestep_b=init_timestep_b,
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
            next_timestep_b, txn_bs = rollout(
                eqx.combine(iter_state.param_state.params, net_static),
                iter_state.timestep_b,
                key=key_rollout,
            )
            buffer_state = buffer.add(iter_state.param_state.buffer_state, txn_bs)

            metrics = txn_bs.time_step.state.metrics
            mean_return = jnp.mean(
                metrics.cum_return,
                where=txn_bs.time_step.is_last,
            )
            mean_length = jnp.mean(metrics.step, where=txn_bs.time_step.is_last)
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
                txn_bs: Batched[Transition, " batch_size horizon"] = batch.experience

                # "reanalyze" the policy targets via mcts with target parameters
                # predict uniform distribution for the last step
                mcts_probs_bs = jax.vmap(act_mcts, in_axes=(None, 1), out_axes=1)(
                    eqx.combine(iter_state.target_params, net_static),
                    txn_bs.time_step.obs,
                    key=jr.split(key_reanalyze, config.optim.num_time_steps),
                )[1].action_weights
                txn_bs = txn_bs._replace(
                    mcts_probs=jnp.where(
                        # TODO this should be for terminal states not truncated
                        # jnp.isclose(txn_bs.time_step.discount, 0.0),
                        txn_bs.time_step.is_last[..., jnp.newaxis],
                        jnp.ones(num_actions) / num_actions,
                        mcts_probs_bs,
                    )
                )

                if False:

                    def debug(batch, free):
                        """Flashbax returns zero when JAX_ENABLE_X64 is not set."""
                        txn_bs = batch.experience
                        for i in range(config.optim.batch_size):
                            weights = txn_bs.mcts_probs[i]
                            if not jnp.allclose(
                                jnp.sum(weights, axis=-1), 1
                            ) and not jnp.all(jnp.isnan(weights)):
                                print("INVALID OUTPUT DISTRIBUTION BROKEN")
                                print(weights)
                                print(free)
                                print(batch.indices)
                                print(batch.priorities)
                                print("Trajectory with broken weights:")
                                print(tree_slice(txn_bs, i))

                    jax.debug.callback(debug, batch, buffer_available)

                @ft.partial(jax.value_and_grad, has_aux=True)
                def loss_grad(params: MuZeroNetwork):
                    """Average the loss across a batch of trjaectories."""
                    losses, aux = loss_trajectory(
                        params,
                        iter_state.target_params,
                        txn_bs,
                        net_static,
                        config.value,
                        config.bootstrap,
                        config.loss,
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
                aux: dict[str, Float[Array, " batch_size"]]
                grads: MuZeroNetwork
                updates, opt_state = optim.update(
                    grads,
                    param_state.opt_state,
                    param_state.params,  # type: ignore
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
                    {f"train/{key}": jnp.mean(value) for key, value in aux.items()}
                    | get_norm_data(updates, "updates/norm")
                    | get_norm_data(params, "params/norm")
                    | {"train/lr": opt_state[1].hyperparams["learning_rate"]}  # type: ignore
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
                    iter_state.step % config.optim.target_update_freq == 0,
                    optax.incremental_update(
                        online, target, config.optim.target_update_size
                    ),  # type: ignore
                    target,
                ),
                param_state.params,
                iter_state.target_params,
            )

            jax.debug.print(
                "step {step}/{num_iters}. "
                "seen {transitions}/{total_transitions} transitions. "
                "available {available}. "
                "mean return {mean_return:.02f}. "
                "mean length {mean_length:.02f}. "
                "{num_episodes} episodes collected.",
                step=iter_state.step,
                num_iters=config.collection.num_iters,
                transitions=param_state.buffer_state.pos * config.collection.num_envs,
                total_transitions=config.collection.total_transitions,
                available=buffer_available,
                mean_return=mean_return,
                mean_length=mean_length,
                num_episodes=jnp.sum(txn_bs.time_step.is_last),
            )

            if config.eval.warnings:
                valid = jax.tree.map(
                    lambda x: jnp.any(jnp.isnan(x)), param_state.params
                )

                @ft.partial(jax.debug.callback, valid=valid)
                def debug(valid):
                    if not valid:
                        raise ValueError("NaN parameters.")

            # evaluate every eval_freq steps
            jax.lax.cond(
                (iter_state.step % eval_freq == 0) & buffer_available,
                ft.partial(
                    visualize, num_envs=config.eval.num_eval_envs, net_static=net_static
                ),
                lambda *_: -jnp.inf,  # avoid nan to support debugging
                # cond only accepts positional arguments
                param_state.params,
                target_params,
                param_state.buffer_state,
                key_evaluate,
            )

            return (
                IterState(
                    step=iter_state.step + 1,
                    timestep_b=next_timestep_b,
                    param_state=param_state,
                    target_params=target_params,
                ),
                None,
            )

        final_iter_state, _ = iterate
        return final_iter_state, None

    def rollout(
        net: MuZeroNetwork,
        init_time_step_b: Batched[TimeStep, " num_envs"],
        *,
        key: Key[Array, ""],
    ) -> tuple[
        Batched[TimeStep, " num_envs"], Batched[Transition, " num_envs horizon"]
    ]:
        """Collect a batch of trajectories."""

        @exec_loop(config.collection.num_time_steps)
        def rollout_step(time_step_b=init_time_step_b, key=key):
            """Take a single step in the true environment using the MCTS policy."""
            key_action, key_step = jr.split(key)

            pred_b, mcts_out_b = act_mcts(net, time_step_b.obs, key=key_action)
            action_b = jnp.asarray(mcts_out_b.action, dtype=action_dtype)
            next_time_step_b = env_step_b(
                time_step_b.state,
                action_b,
                env_params,
                key=jr.split(key_step, config.collection.num_envs),
            )
            return next_time_step_b, Transition(
                # contains the reward and network predictions computed from the rollout state
                time_step=time_step_b,
                action=action_b,
                pred=pred_b,
                mcts_probs=jnp.asarray(mcts_out_b.action_weights, dtype=action_dtype),
            )

        final_timestep_b, txn_sb = rollout_step
        txn_bs = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), txn_sb
        )  # swap horizon and num_envs axes

        return final_timestep_b, txn_bs

    def act_mcts(
        net: MuZeroNetwork,
        obs_b: Batched[GoalObs[TObs], " num_envs"],
        *,
        key: Key[Array, ""],
    ) -> tuple[Prediction, mctx.PolicyOutput]:
        """Take a single batch of actions from the batch of timesteps using MCTS.

        Returns:
            tuple[Prediction, mctx.PolicyOutput]: The value of `timestep`
                predicted by the critic network
                and the MCTS output of planning from the timestep.
        """
        goal_embedding_b = jax.vmap(net.embed_goal)(obs_b.goal)
        init_hidden_dyn_b = jax.vmap(net.projection)(obs_b.obs, goal_embedding_b)
        del obs_b  # not needed anymore
        pred_b = jax.vmap(net.actor_critic)(init_hidden_dyn_b, goal_embedding_b)

        root = mctx.RootFnOutput(
            prior_logits=pred_b.policy_logits,  # type: ignore
            value=config.value.logits_to_value(pred_b.value_logits),  # type: ignore
            # embedding contains current env_state and the hidden to pass to next net call
            embedding=init_hidden_dyn_b,  # type: ignore
        )

        invalid_actions = (
            None  # jax.vmap(get_invalid_actions)(rollout_states.env_state)
        )

        @typecheck
        def mcts_recurrent_fn(
            _: None,  # closure net
            key: Key[Array, ""],
            action_b: Integer[Array, " num_envs"],
            hidden_dyn_b: Float[Array, " num_envs rnn_size"],
        ) -> tuple[mctx.RecurrentFnOutput, Float[Array, " num_envs rnn_size"]]:
            """World model transition function.

            Returns:
                tuple[mctx.RecurrentFnOutput, Float (num_envs, rnn_size)]: The logits and value
                    computed by the actor-critic network for the newly created node;
                    The hidden state representing the state following `hiddens`.
            """
            hidden_dyn_b, reward_b = jax.vmap(net.world_model.step)(
                hidden_dyn_b, action_b, goal_embedding_b
            )
            pred_b = jax.vmap(net.actor_critic)(hidden_dyn_b, goal_embedding_b)
            return (
                mctx.RecurrentFnOutput(
                    reward=config.value.logits_to_value(reward_b),  # type: ignore
                    # TODO termination
                    discount=jnp.full(action_b.shape[0], config.bootstrap.discount),  # type: ignore
                    prior_logits=pred_b.policy_logits,  # type: ignore
                    value=config.value.logits_to_value(pred_b.value_logits),  # type: ignore
                ),
                hidden_dyn_b,
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

        return pred_b, out

    return train


if __name__ == "__main__":
    spec = importlib.util.find_spec("experiments.muzero")
    if spec and spec.origin:
        main(TrainConfig, make_train, spec.origin)
    else:
        print("Could not find the path to the module.")
