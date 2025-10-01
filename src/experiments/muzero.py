"""Implementation of MuZero with a recurrent actor-critic network."""

import dataclasses as dc
import functools as ft
import importlib.util
import math
import sys
from collections.abc import Callable
from typing import Annotated as Batched
from typing import Generic, NamedTuple

import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mctx
import optax
import yaml
from jaxtyping import Array, Bool, Float, Integer, Key, PyTree

from envs.housemaze_env import HouseMazeObs
from envs.translate import make_env
from experiments.config import (
    ArchConfig,
    LossConfig,
    TrainConfig,
    ValueConfig,
    main,
)
from utils import specs
from utils.jax_utils import (
    BootstrapConfig,
    bootstrap,
    get_network_size,
    get_weight_mask,
    roll_into_matrix,
    scale_gradient,
    tree_slice,
)
from utils.log_utils import (
    exec_loop,
    get_norm_data,
    log_values,
    print_bytes,
    typecheck,
)
from utils.prioritized_buffer import BufferState, PrioritizedBuffer
from utils.structures import (
    Prediction,
    TEnvState,
    TimeStep,
    TObs,
    Transition,
)
from utils.visualize import (
    SUPPORTED_VIDEO_ENVS,
    visualize_env_state_frame,
    visualize_trajectory,
)
from wrappers.goal_wrapper import GoalObs
from wrappers.oar_wrapper import OAR


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

    We use `dyn` to mean the dynamics model recurrent state
    and `hidden_obs` for the recurrent policy's state (see `ActorCritic`).
    """

    reward_head: MLPConcatArgs
    """The reward model."""
    embed_action: eqx.nn.Embedding
    """Embed actions to `dyn_size`."""
    mlp: eqx.nn.MLP
    """MLP embedding."""

    def __init__(
        self,
        config: ArchConfig,
        num_actions: int,
        num_value_bins: int,
        *,
        key: Key[Array, ""],
    ):
        key_reward, key_embed_action, key_mlp = jr.split(key, 3)
        self.embed_action = eqx.nn.Embedding(num_actions, config.dyn_size, key=key_embed_action)
        self.reward_head = MLPConcatArgs(
            in_size=config.dyn_size + config.goal_dim,
            out_size=num_value_bins,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            activation=getattr(jax.nn, config.activation),
            key=key_reward,
        )
        # TODO use haiku initialization
        self.mlp = eqx.nn.MLP(
            in_size=config.dyn_size,
            out_size=config.dyn_size,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            activation=getattr(jax.nn, config.activation),
            use_bias=False,
            use_final_bias=False,
            key=key_mlp,
        )

    @typecheck
    def step(
        self,
        dyn: Float[Array, " dyn_size"],
        action: Integer[Array, ""],
        goal_embedding: Float[Array, " goal_dim"],
    ) -> tuple[Float[Array, " dyn_size"], Float[Array, " num_value_bins"]]:
        """An imagined state transition (`env.step`).

        Transitions to the next hidden state and emits predicted reward from the transition.
        Refer to `SimpleTransition` in `neurorl/library/muzero_mlps.py`
        and the `transition_fn` in `neurorl/configs/catch_trainer.py`
        (inside `make_muzero_networks/make_core_module`).
        """
        # TODO handle continue predictor
        out = jax.nn.relu(dyn) + self.embed_action(action)
        out = self.mlp(out)
        # TODO also use resnet-like structure for policy
        reward_logits = self.reward_head(out, goal_embedding)
        return out, reward_logits


class WorldModelGRU(eqx.Module):
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
        num_actions: int,
        num_value_bins: int,
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_reward = jr.split(key)
        self.cell = eqx.nn.GRUCell(num_actions, config.dyn_size, key=key_cell)
        self.reward_head = eqx.nn.MLP(
            config.dyn_size + config.goal_dim,
            num_value_bins,
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
        action = jax.nn.one_hot(action, self.cell.input_size)
        hidden = self.cell(action, hidden)
        return hidden, self.reward_head(jnp.concat([hidden, goal_embedding]))


class WorldModelResNet(eqx.Module):
    """Parameterizes the recurrent dynamics model and reward function.

    We use `dyn` to mean the dynamics model recurrent state
    and `hidden_obs` for the recurrent policy's state (see `ActorCritic`).
    """

    linear: eqx.nn.Linear
    """The recurrent cell for the dynamics model."""
    reward_head: MLPConcatArgs
    """The reward model."""
    norm1: eqx.nn.LayerNorm
    """Layer normalization for the state-action representation."""
    norm2: eqx.nn.LayerNorm
    """Layer normalization for the MLP."""
    embed_action: eqx.nn.Embedding
    """Embed actions to `dyn_size`."""
    mlp: eqx.nn.MLP
    """MLP embedding."""

    def __init__(
        self,
        config: ArchConfig,
        num_actions: int,
        num_value_bins: int,
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_reward, key_embed_action, key_mlp = jr.split(key, 4)
        self.embed_action = eqx.nn.Embedding(num_actions, config.dyn_size, key=key_embed_action)
        self.norm1 = eqx.nn.LayerNorm(config.dyn_size)
        self.norm2 = eqx.nn.LayerNorm(config.dyn_size)
        self.linear = eqx.nn.Linear(config.dyn_size, config.dyn_size, use_bias=False, key=key_cell)
        self.reward_head = MLPConcatArgs(
            in_size=config.dyn_size + config.goal_dim,
            out_size=num_value_bins,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            activation=getattr(jax.nn, config.activation),
            key=key_reward,
        )
        # TODO use haiku initialization
        self.mlp = eqx.nn.MLP(
            in_size=config.dyn_size,
            out_size=config.dyn_size,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            activation=getattr(jax.nn, config.activation),
            key=key_mlp,
        )

    @typecheck
    def step(
        self,
        dyn: Float[Array, " dyn_size"],
        action: Integer[Array, ""],
        goal_embedding: Float[Array, " goal_dim"],
    ) -> tuple[Float[Array, " dyn_size"], Float[Array, " num_value_bins"]]:
        """An imagined state transition (`env.step`).

        Transitions to the next hidden state and emits predicted reward from the transition.
        Refer to `Transition` in `neurorl/library/muzero_mlps.py`
        and the `transition_fn` in `neurorl/configs/catch_trainer.py`
        (inside `make_muzero_networks/make_core_module`).
        """
        # TODO handle continue predictor
        out = jax.nn.relu(dyn) + self.embed_action(action)
        out = dyn + self.linear(self.norm1(out))  # "fake env.step"
        out = out + self.mlp(self.norm2(out))
        # TODO also use resnet-like structure for policy
        reward_logits = self.reward_head(out, goal_embedding)
        return out, reward_logits


class ActorCritic(eqx.Module):
    """Parameterizes the actor-critic networks."""

    policy_head: MLPConcatArgs
    """Policy network. `(dyn, goal) -> action distribution`"""
    value_head: MLPConcatArgs
    """Value network. `(dyn, goal) -> value distribution`"""

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
            config.dyn_size + config.goal_dim,
            num_actions,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_policy,
        )
        self.value_head = MLPConcatArgs(
            config.dyn_size + config.goal_dim,
            num_value_bins,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_value,
        )

    def __call__(
        self,
        hidden: Float[Array, " dyn_size"],
        goal_embedding: Float[Array, " goal_dim"],
    ) -> Prediction:
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
        self.embed_combined = eqx.nn.Embedding(total_dims, config.obs_dim, key=key_embed)

        obs_flattened_size = sum(math.prod(spec.shape) for spec in leaf_specs)
        assert config.obs_dim is not None, "obs_dim must be set in the config"
        self.norm = eqx.nn.LayerNorm(config.obs_dim)  # we map over the components
        self.mlp = MLPConcatArgs(
            in_size=obs_flattened_size * config.obs_dim + config.goal_dim,
            out_size=config.dyn_size,
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
        obs_flat = jnp.concat([jnp.ravel(x + offset) for x, offset in zip(leaves, self.dim_offset)])
        obs_embedding: Float[Array, " obs_flattened_size obs_dim"] = jax.vmap(self.embed_combined)(obs_flat)
        obs_embedding = jax.vmap(self.norm)(obs_embedding)
        obs_embedding = self.mlp(jnp.ravel(obs_embedding), goal_embedding)
        return obs_embedding


class CNNEmbedding(eqx.Module):
    cnn: eqx.nn.Conv2d
    """Embed an observation using a CNN projection."""
    projection: eqx.nn.Linear

    def __init__(self, obs_size, mlp_size, dyn_size, *, key: Key[Array, ""]):
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
        self.projection = eqx.nn.Linear(in_features=flat_size, out_features=dyn_size, key=key_linear)

    def __call__(self, obs, goal_embedding: Float[Array, " goal_dim"]):
        """Embed the observation and goal into a recurrent state."""
        obs_embedding = jnp.ravel(self.cnn(obs))
        obs_embedding = self.projection(obs_embedding)
        return jnp.concat([obs_embedding, goal_embedding])


class OAREmbedding(eqx.Module):
    """Embed an OAR observation into a recurrent state."""

    mlp: MLPConcatArgs
    num_actions: int = eqx.field(static=True)

    def __init__(self, *args, num_actions: int, **kwargs):
        self.mlp = MLPConcatArgs(*args, **kwargs)
        self.num_actions = num_actions

    def __call__(
        self,
        x: OAR[Float[Array, " *obs_size"]],
        goal_embedding: Float[Array, " goal_dim"],
        *,
        key=None,
    ) -> Float[Array, " dyn_size"]:
        """Embed the OAR observation and goal into a recurrent state."""
        # OAR is a tuple of (obs, action, reward)
        obs = jnp.ravel(x.obs)
        reward = jnp.tanh(x.reward)
        action = jax.nn.one_hot(x.action, self.num_actions)
        obs_embedding = self.mlp(obs, reward, action, goal_embedding)
        return obs_embedding


class MuZeroNetwork(eqx.Module, Generic[TObs]):
    """The entire MuZero network parameters."""

    embed_observation: Callable[[TObs, Float[Array, " goal_dim"]], Float[Array, " dyn_size"]]
    """Embed the observation into world model recurrent state."""
    embed_goal: eqx.nn.Embedding
    """Embed the goal into a goal embedding (goal_dim)."""
    actor_critic: ActorCritic
    """The actor and critic networks."""
    world_model: WorldModelGRU
    """The recurrent world model."""
    gradient_scale: float
    """Scale the gradients at each step of the world model."""

    def __init__(
        self,
        config: ArchConfig,
        obs_spec: PyTree[specs.Array],
        num_actions: int,
        num_value_bins: int,
        num_goals: int,
        *,
        key: Key[Array, ""],
    ):
        key_projection, key_world_model, key_actor_critic, key_embed_goal = jr.split(key, 4)

        if isinstance(obs_spec, OAR):
            self.embed_observation = OAREmbedding(
                in_size=obs_spec.obs.size + 1 + num_actions + config.goal_dim,
                out_size=config.dyn_size,
                width_size=config.mlp_size,
                depth=config.mlp_depth,
                activation=getattr(jax.nn, config.activation),
                num_actions=num_actions,
                key=key_projection,
            )  # type: ignore

        elif config.projection_kind == "mlp":
            # ensure flattened input
            obs_size = obs_spec[0].shape if isinstance(obs_spec, list) else obs_spec.shape
            assert len(obs_size) == 1, f"MLP expects flattened input. Got {obs_size}"
            self.embed_observation = MLPConcatArgs(
                in_size=obs_size[0] + config.goal_dim,
                out_size=config.dyn_size,
                width_size=config.mlp_size,
                depth=config.mlp_depth,
                activation=getattr(jax.nn, config.activation),
                key=key_projection,
            )
        elif config.projection_kind == "cnn":
            self.embed_observation = CNNEmbedding(obs_spec.shape, config.mlp_size, config.dyn_size, key=key_projection)
        elif config.projection_kind == "housemaze":
            self.embed_observation = HouseMazeEmbedding(config=config, obs_spec=obs_spec, key=key_projection)  # type: ignore

        self.embed_goal = eqx.nn.Embedding(num_goals, config.goal_dim, key=key_embed_goal)
        self.actor_critic = ActorCritic(
            config=config,
            num_actions=num_actions,
            num_value_bins=num_value_bins,
            key=key_actor_critic,
        )
        self.world_model = WorldModelGRU(
            config=config,
            num_actions=num_actions,
            num_value_bins=num_value_bins,
            key=key_world_model,
        )
        self.gradient_scale = config.world_model_gradient_scale

    # @typecheck
    def world_model_rollout(
        self,
        goal_obs: GoalObs[TObs],
        action_s: Integer[Array, " horizon"],
    ) -> tuple[
        Float[Array, " dyn_size"],
        tuple[Float[Array, " horizon"], Batched[Prediction, " horizon"]],
    ]:
        """Roll out an imagined trajectory by taking the actions from goal_obs.

        Args:
            goal_obs (GoalObs[Float (*obs_size,)]): The observation to roll out from. The goal is preserved throughout.
            action (Integer (horizon,)): The series of actions to take.

        Returns:
            tuple[Float (dyn_size,), tuple[Float (horizon,), Prediction (horizon,)]]: The resulting hidden world state;
                The predicted rewards for taking the given actions (including from `goal_obs`);
                The predicted policy logits and values at the imagined states (including `goal_obs`).
        """
        goal_embedding = self.embed_goal(goal_obs.goal)
        init_dyn = self.embed_observation(goal_obs.obs, goal_embedding)

        def step(dyn, action):
            pred = self.actor_critic(dyn, goal_embedding)
            next_dyn, reward = self.world_model.step(dyn, action, goal_embedding)
            # Scale gradients to prevent vanishing/exploding
            next_dyn = scale_gradient(next_dyn, self.gradient_scale)
            return next_dyn, (reward, pred)

        return jax.lax.scan(step, init_dyn, action_s)

    def predict_value_s(
        self,
        value_cfg: ValueConfig,
        obs: GoalObs,
        action_s: Integer[Array, " horizon"],
    ):
        """Predict the value sequence from the target network."""
        _, (reward_s, target_pred_s) = self.world_model_rollout(obs, action_s)
        value_s = value_cfg.logits_to_value(target_pred_s.value_logits)
        return value_s, reward_s


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
    _, (online_reward_sh, online_pred_sh) = jax.vmap(net.world_model_rollout)(txn_s.time_step.obs, action_sh)
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
        return jnp.sum(x * mask, where=mask > 0)

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

    loss = loss_cfg.policy_coef * policy_loss + loss_cfg.value_coef * value_loss + loss_cfg.reward_coef * reward_loss

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

    boot_value_sh, _ = bootstrap(ft.partial(target_net.predict_value_s, value_cfg), txn_s, bootstrap_cfg)
    boot_value_dist_sh = distrax.Categorical(probs=value_cfg.value_to_probs(boot_value_sh))
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

    return value_loss, jax.tree.map(weighted_mean, value_stats) | {
        "max_abs_td_error": jnp.max(jnp.abs(value_stats["td_error"][:, 0]))  # type: ignore
    }


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


def make_env_and_network(config: TrainConfig):
    """Construct the environment and network from the configuration."""
    env, env_params = make_env(config.env)
    action_space = env.action_space(env_params)
    assert isinstance(action_space, specs.BoundedArray), f"Expected BoundedArray action space, got {type(action_space)}"
    num_actions = action_space.num_values
    num_goals = env.goal_space(env_params).num_values

    net = MuZeroNetwork(
        config=config.arch,
        obs_spec=env.observation_space(env_params),
        num_actions=num_actions,
        num_value_bins=config.value.num_value_bins,
        num_goals=num_goals,
        key=jr.key(0),
    )
    params, net_static = eqx.partition(net, eqx.is_inexact_array)
    nparams, nbytes = get_network_size(params)
    print(f"{get_network_size(params)=}")
    print(f"{get_network_size(params.embed_observation)=}")
    print(f"{get_network_size(params.embed_goal)=}")
    print(f"{get_network_size(params.world_model)=}")
    print(f"{get_network_size(params.actor_critic.policy_head)=}")
    print(f"{get_network_size(params.actor_critic.value_head)=}")

    return env, env_params, num_actions, net_static


def make_train(config: TrainConfig):
    """Return a jittable training function for MuZero."""
    # construct environment
    # use the initial parameters to construct optimizer weight decay mask
    env, env_params, num_actions, net_static = make_env_and_network(config)
    num_goals = env.goal_space(env_params).num_values

    env_reset_b = jax.vmap(env.reset, in_axes=(None,))
    env_step_b = jax.vmap(env.step, in_axes=(0, 0, None))  # map over state and action

    # https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.exponential_decay
    # decay by a factor of `decay_rate` every `transition_steps`

    # only apply weight decay to weights
    lr_config = dict(**config.lr)
    lr_name = lr_config.pop("name")
    lr_schedule = getattr(optax, lr_name)(**lr_config)
    del lr_name, lr_config

    optim = ft.partial(optax.adamw, eps=1e-3, mask=get_weight_mask(net_static))
    optim = optax.chain(
        optax.clip_by_global_norm(jnp.pi),
        # https://optax.readthedocs.io/en/latest/getting_started.html#accessing-learning-rate
        optax.inject_hyperparams(optim)(learning_rate=lr_schedule),
    )
    del net_static

    buffer = PrioritizedBuffer.new(**dc.asdict(config.buffer))

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
        del key

        print(env.fullname)
        print("=" * 25)
        yaml.safe_dump(dc.asdict(config), sys.stdout)
        print("=" * 25)

        # ========== Initialize all state ==========

        init_net = MuZeroNetwork(
            config=config.arch,
            obs_spec=env.observation_space(env_params),
            num_actions=num_actions,
            num_value_bins=config.value.num_value_bins,
            num_goals=num_goals,
            key=key_net,
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)

        init_timestep_b = env_reset_b(env_params, key=jr.split(key_reset, config.buffer.batch_size))
        init_transition = Transition(
            time_step=tree_slice(init_timestep_b, 0),
            action=jnp.empty((), int),
            pred=init_net.actor_critic(
                jnp.empty(config.arch.dyn_size, float),
                init_net.embed_goal(init_timestep_b.obs.goal[0]),
            ),
            mcts_probs=jnp.empty(num_actions, float),
        )
        init_buffer_state = buffer.init(init_transition)
        del init_transition
        buffer_state_bytes = sum(x.nbytes for x in jax.tree.leaves(init_buffer_state) if eqx.is_array(x))
        print(f"buffer size (bytes): {buffer_state_bytes}")
        print_bytes(init_buffer_state)

        # ========== Training loop ==========

        @exec_loop(config.optim.num_iters)
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
            key_rollout, key_optim = jr.split(key)
            del key

            # collect data
            next_timestep_b, txn_bs = rollout(
                eqx.combine(iter_state.param_state.params, net_static),
                iter_state.timestep_b,
                key=key_rollout,
            )
            del key_rollout
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

            buffer_available = jnp.bool_(buffer.num_available(buffer_state) >= config.optim.batch_size)

            # ========== OPTIMIZATION STEP ==========

            target_net = eqx.combine(iter_state.target_params, net_static)
            act_mcts_bs = jax.vmap(act_mcts_b, in_axes=(None, 1), out_axes=1)

            @exec_loop(config.optim.num_updates_per_iter, cond=buffer_available)
            def optimize_step(
                param_state=iter_state.param_state._replace(buffer_state=buffer_state),
                key=key_optim,
            ):
                """Sample batch of trajectories and do a single 'SGD step'"""
                key_sample, key_reanalyze = jr.split(key, 2)
                del key
                batch = jax.vmap(buffer.sample, in_axes=(None, None))(
                    param_state.buffer_state,
                    config.eval.warnings,
                    key=jr.split(key_sample, config.optim.batch_size),
                )
                del key_sample
                txn_bs: Batched[Transition, " batch_size horizon"] = batch.experience

                # "reanalyze" the policy targets via mcts with target parameters
                _, mcts_out_bs = act_mcts_bs(
                    target_net,
                    txn_bs.time_step.obs,
                    key=jr.split(key_reanalyze, config.buffer.sample_length),
                )
                del key_reanalyze
                # predict uniform distribution for the last step
                mcts_probs_bs = jnp.where(
                    # TODO this should be for terminal states not truncated
                    # jnp.isclose(txn_bs.time_step.discount, 0.0),
                    txn_bs.time_step.is_last[..., jnp.newaxis],
                    jnp.ones((num_actions,)) / num_actions,
                    mcts_out_bs.action_weights,
                )
                txn_bs = txn_bs._replace(mcts_probs=mcts_probs_bs)

                if False:

                    def debug(batch, free):
                        """Flashbax returns zero when JAX_ENABLE_X64 is not set."""
                        txn_bs = batch.experience
                        for i in range(config.optim.batch_size):
                            weights = txn_bs.mcts_probs[i]
                            if not jnp.allclose(jnp.sum(weights, axis=-1), 1) and not jnp.all(jnp.isnan(weights)):
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
                    importance_weights = jnp.reciprocal(batch.priority) ** config.optim.importance_exponent
                    importance_weights /= jnp.max(importance_weights, keepdims=True)
                    return jnp.mean(importance_weights * losses), aux

                (_, aux), grads = loss_grad(param_state.params)
                # value_and_grad drops types
                aux: dict[str, Float[Array, " batch_size"]]
                grads: MuZeroNetwork
                updates, opt_state = optim.update(
                    grads,  # type: ignore
                    param_state.opt_state,
                    param_state.params,  # type: ignore
                )
                params: MuZeroNetwork = optax.apply_updates(param_state.params, updates)  # type: ignore

                # prioritize trajectories with high TD error
                if False:
                    priorities = jnp.mean(jnp.abs(aux["td_error"]), axis=-1)
                    buffer_state = buffer.set_priorities(
                        param_state.buffer_state,
                        batch.idx,
                        priorities**config.optim.priority_exponent,
                    )

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

            del buffer_state, key_optim  # replaced by param_state.buffer_state
            param_state, _ = optimize_step

            # occasionally update target
            target_params: MuZeroNetwork = jax.tree.map(
                ft.partial(jnp.where, iter_state.step % config.optim.target_update_freq == 0),
                param_state.params,
                iter_state.target_params,
            )

            jax.debug.print(
                "iter {step}/{num_iters}. "
                "trained {trained}. "
                "seen {transitions}/{total_transitions} txns. "
                "update {num_updates}/{total_updates}. "
                "{num_episodes} episodes: "
                "return {mean_return:.02f}. "
                "length {mean_length:.02f}. ",
                step=iter_state.step,
                trained=buffer_available,
                num_iters=config.optim.num_iters,
                transitions=param_state.buffer_state.pos * config.buffer.batch_size,
                total_transitions=config.optim.num_iters * config.buffer.sample_length * config.buffer.batch_size,
                num_updates=config.optim.num_updates_per_iter * iter_state.step,
                total_updates=config.optim.total_updates,
                num_episodes=jnp.sum(txn_bs.time_step.is_last),
                mean_return=mean_return,
                mean_length=mean_length,
            )

            if config.eval.warnings:
                valid = jax.tree.map(lambda x: jnp.any(jnp.isnan(x)), param_state.params)

                @ft.partial(jax.debug.callback, valid=valid)
                def debug(valid):
                    if not valid:
                        raise ValueError("NaN parameters.")

            # visualization
            target_net = eqx.combine(target_params, net_static)
            txn_s = tree_slice(txn_bs, 0)
            viz_boot_value_s, viz_reward_logits_s = bootstrap(
                ft.partial(target_net.predict_value_s, config.value),
                txn_s,
                config.bootstrap,
            )
            video = (
                jax.vmap(ft.partial(visualize_env_state_frame, config.env.name))(txn_s.time_step.state)
                if config.env.name in SUPPORTED_VIDEO_ENVS
                else None
            )
            jax.lax.cond(
                (iter_state.step % config.eval.eval_freq == 0) & buffer_available,
                # env name is invalid jax type (str) so pass statically
                ft.partial(jax.debug.callback, visualize_trajectory, config.env.name),
                lambda *args: None,
                config.value,
                txn_s,
                viz_boot_value_s[:, 0],  # first prediction (no model rollout)
                viz_reward_logits_s[:, 0],  # first prediction (no model rollout)
                video,
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
        init_time_step_b: Batched[TimeStep[GoalObs[TObs], TEnvState], " num_envs"],
        *,
        key: Key[Array, ""],
    ) -> tuple[
        Batched[TimeStep[GoalObs[TObs], TEnvState], " num_envs"],
        Batched[Transition[GoalObs[TObs], TEnvState], " num_envs horizon"],
    ]:
        """Collect a batch of trajectories."""

        @exec_loop(config.buffer.sample_length)
        def rollout_step(time_step_b=init_time_step_b, key=key):
            """Take a single step in the true environment using the MCTS policy."""
            key_action, key_step = jr.split(key)

            pred_b, mcts_out_b = act_mcts_b(net, time_step_b.obs, key=key_action)
            action_b = jnp.asarray(mcts_out_b.action, dtype=int)
            next_time_step_b = env_step_b(
                time_step_b.state,
                action_b,
                env_params,
                key=jr.split(key_step, config.buffer.batch_size),
            )
            return next_time_step_b, Transition(
                # contains the reward and network predictions computed from the rollout state
                time_step=time_step_b,
                action=action_b,
                pred=pred_b,
                mcts_probs=jnp.asarray(mcts_out_b.action_weights, dtype=float),
            )

        final_timestep_b, txn_sb = rollout_step
        txn_bs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), txn_sb)  # swap horizon and num_envs axes

        return final_timestep_b, txn_bs

    def act_mcts_b(
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
        init_dyn_b = jax.vmap(net.embed_observation)(obs_b.obs, goal_embedding_b)
        del obs_b  # not needed anymore
        pred_b = jax.vmap(net.actor_critic)(init_dyn_b, goal_embedding_b)

        root = mctx.RootFnOutput(
            prior_logits=pred_b.policy_logits,  # type: ignore
            value=config.value.logits_to_value(pred_b.value_logits),  # type: ignore
            # embedding contains current env_state and the hidden to pass to next net call
            embedding=init_dyn_b,  # type: ignore
        )

        invalid_actions = None  # jax.vmap(get_invalid_actions)(rollout_states.env_state)

        @typecheck
        def mcts_recurrent_fn(
            _: None,  # closure net
            key: Key[Array, ""],
            action_b: Integer[Array, " num_envs"],
            dyn_b: Float[Array, " num_envs dyn_size"],
        ) -> tuple[mctx.RecurrentFnOutput, Float[Array, " num_envs dyn_size"]]:
            """World model transition function.

            Returns:
                tuple[mctx.RecurrentFnOutput, Float (num_envs, dyn_size)]: The logits and value
                    computed by the actor-critic network for the newly created node;
                    The hidden state representing the state following `hiddens`.
            """
            next_dyn_b, reward_logits_b = jax.vmap(net.world_model.step)(dyn_b, action_b, goal_embedding_b)
            pred_b = jax.vmap(net.actor_critic)(next_dyn_b, goal_embedding_b)
            return (
                mctx.RecurrentFnOutput(
                    reward=config.value.logits_to_value(reward_logits_b),  # type: ignore
                    # TODO termination
                    discount=jnp.full(action_b.shape[0], config.bootstrap.discount),  # type: ignore
                    prior_logits=pred_b.policy_logits,  # type: ignore
                    value=config.value.logits_to_value(pred_b.value_logits),  # type: ignore
                ),
                next_dyn_b,
            )

        out = mctx.gumbel_muzero_policy(
            params=None,  # type: ignore
            rng_key=key,
            root=root,
            recurrent_fn=mcts_recurrent_fn,  # type: ignore
            invalid_actions=invalid_actions,
            **config.mcts,
        )

        return pred_b, out

    return train


if __name__ == "__main__":
    spec = importlib.util.find_spec("experiments.muzero")
    if spec and spec.origin:
        main(TrainConfig, make_train, spec.origin)
    else:
        print("Could not find the path to the module.")
