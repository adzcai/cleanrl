"""Implementation of AlphaZero with a recurrent actor-critic network.

1. Config
2. Data structures for loop carries
3. Model definitions
4. Training loop
5. Rollout function
6. Evaluation function
"""

# jax
from dataclasses import asdict, field
import jax
import jax.numpy as jnp
import jax.random as jr

# jax ecosystem
import chex
import distrax
import equinox as eqx
import optax

# rl
import rlax
import mctx
import flashbax as fbx
import gymnax
from gymnax.wrappers import FlattenObservationWrapper

# logging
import wandb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import yaml

# typing
from typeguard import typechecked as typechecker
from jaxtyping import Bool, Integer, Float, Key, Array, jaxtyped
from typing import Literal, NamedTuple, Annotated as Batched

# util
import functools as ft
from log_util import (
    exec_loop,
    get_norm_data,
    log_values,
    tree_slice,
    roll_into_matrix,
    visualize_catch,
)

matplotlib.use("agg")  # enable plotting inside jax callback


@chex.dataclass(frozen=True)
class ArchConfig:
    """Network architecture"""

    rnn_size: int = 128
    mlp_size: int = 32
    mlp_depth: int = 2
    activation: Literal["relu", "tanh"] = "relu"


@chex.dataclass(frozen=True)
class Config:
    env_name: str = "Catch-bsuite"
    arch: ArchConfig = field(default_factory=ArchConfig)
    # collection
    total_transitions: int = 500_000
    num_envs: int = 64  # more parallel data collection
    horizon: int = 9  # also mcts max depth
    # two-hot value
    num_value_bins: int | Literal["scalar"] = 5
    min_value: float = -1.0
    max_value: float = 1.0
    # search
    num_mcts_simulations: int = 4  # stronger policy improvement
    # bootstrapping
    discount: float = 0.997  # R2D2 standard
    lambda_gae: float = 0.95
    # optimization
    num_minibatches: int = 6  # number of gradient descent steps per iteration
    batch_size: int = 16  # reduce gradient variance
    lr_init: float = 1e-2
    max_grad_norm: float = 2 * jnp.pi
    # prioritized replay
    priority_exponent: float = 0.6
    priority_td_norm_ord: float = 1.0  # norm of TD error (1 for abs, inf for max)
    # target
    target_update_freq: int = 4
    target_update_size: float = 3 / 4
    # loss
    value_coef: float = 0.4  # scale the value loss
    reward_coef: float = 0.6  # scale the reward loss
    # evaluation
    num_evals: int = 6
    num_eval_envs: int = 2


debug_config = Config(
    arch=ArchConfig(
        rnn_size=1,
        mlp_size=1,
        mlp_depth=0,
    ),
    total_transitions=100,
    num_envs=1,
    horizon=9,
    num_value_bins="scalar",
    num_mcts_simulations=1,
    num_minibatches=1,
    batch_size=1,
    num_evals=1,
    num_eval_envs=1,
)


class WorldModelRNN(eqx.Module):
    """Parameterizes the recurrent world model.

    Takes (hidden, action) -> (hidden, reward).
    """

    cell: eqx.nn.GRUCell
    reward_head: eqx.nn.MLP

    def __init__(
        self,
        config: ArchConfig,
        num_actions: int,
        num_value_bins: int | Literal["scalar"],
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_reward = jr.split(key)
        self.cell = eqx.nn.GRUCell(num_actions, config.rnn_size, key=key_cell)
        self.reward_head = eqx.nn.MLP(
            config.rnn_size,
            num_value_bins,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_reward,
        )

    def step(
        self, hidden: Float[Array, " rnn_size"], action: Integer[Array, ""]
    ) -> tuple[Float[Array, " rnn_size"], Float[Array, " num_value_bins"]]:
        """Transitions to the next embedding and emits predicted reward."""
        # TODO handle continue predictor
        action = jax.nn.one_hot(action, self.cell.input_size)
        hidden = self.cell(action, hidden)
        return hidden, self.reward_head(hidden)

    def init_hidden(self):
        return jnp.zeros(self.cell.hidden_size)


class Prediction(NamedTuple):
    policy_logits: Float[Array, " num_actions"]
    value_logits: Float[Array, " num_value_bins"]


class ActorCritic(eqx.Module):
    """Parameterizes the actor-critic network."""

    policy_head: eqx.nn.MLP
    value_head: eqx.nn.MLP

    def __init__(
        self,
        config: ArchConfig,
        num_actions: int,
        num_value_bins: int | Literal["scalar"],
        *,
        key: Key[Array, ""],
    ):
        key_policy, key_value = jr.split(key)
        self.policy_head = eqx.nn.MLP(
            config.rnn_size,
            num_actions,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_policy,
        )
        self.value_head = eqx.nn.MLP(
            config.rnn_size,
            num_value_bins,
            config.mlp_size,
            config.mlp_depth,
            getattr(jax.nn, config.activation),
            key=key_value,
        )

    def __call__(self, hidden: Float[Array, " rnn_size"]):
        """Predict action and value logits from the hidden state."""
        return Prediction(self.policy_head(hidden), self.value_head(hidden))


class MuZeroNetwork(eqx.Module):
    """The entire MuZero network parameters."""

    projection: eqx.nn.MLP
    world_model: WorldModelRNN
    actor_critic: ActorCritic

    def __init__(
        self,
        config: ArchConfig,
        obs_size: int,
        num_actions: int,
        num_value_bins: int,
        *,
        key: Key[Array, ""],
    ):
        key_projection, key_world_model, key_actor_critic = jr.split(key, 3)

        self.projection = eqx.nn.MLP(
            in_size=obs_size,
            out_size=config.rnn_size,
            width_size=config.mlp_size,
            depth=config.mlp_depth,
            key=key_projection,
        )
        self.world_model = WorldModelRNN(
            config=config,
            num_actions=num_actions,
            num_value_bins=num_value_bins,
            key=key_world_model,
        )
        self.actor_critic = ActorCritic(
            config=config,
            num_actions=num_actions,
            num_value_bins=num_value_bins,
            key=key_actor_critic,
        )

    def __call__(self, obs: Float[Array, " obs_size"], action: Integer[Array, " horizon"]):
        return jax.lax.scan(
            lambda hidden, action: self.step(hidden, action),
            self.projection(obs),
            action,
        )

    def step(self, hidden: Float[Array, " rnn_size"], action: Integer[Array, ""]):
        pred = self.actor_critic(hidden)
        hidden, reward = self.world_model.step(hidden, action)
        return hidden, (reward, pred)


class RolloutState(NamedTuple):
    """Carried when rolling out the environment."""

    obs: Float[Array, " obs_size"]
    env_state: gymnax.EnvState
    initial: Bool[Array, ""]


class Transition(NamedTuple):
    """A single transition. May be batched into a trajectory."""

    rollout_state: RolloutState
    action: Integer[Array, ""]
    reward: Float[Array, " "]
    pred: Prediction
    mcts_probs: Float[Array, " num_actions"]


BufferState = fbx.prioritised_trajectory_buffer.PrioritisedTrajectoryBufferState[Transition]
TrajectorySample = fbx.prioritised_trajectory_buffer.PrioritisedTrajectoryBufferSample[Transition]
TrajectoryBuffer = fbx.prioritised_trajectory_buffer.PrioritisedTrajectoryBuffer[
    Transition,
    BufferState,
    TrajectorySample,
]


class ParamState(NamedTuple):
    """Carried during optimization."""

    params: MuZeroNetwork
    opt_state: optax.OptState
    buffer_state: BufferState


class IterState(NamedTuple):
    """Carried across algorithm iterations."""

    step: Integer[Array, ""]
    rollout_states: Batched[RolloutState, "num_envs"]
    param_state: ParamState
    target_params: MuZeroNetwork


class LossStatistics(NamedTuple):
    """Quantities computed for the loss."""

    loss: Float[Array, " "]
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


@jaxtyped(typechecker=typechecker)
def make_train(config: Config):
    num_iters = config.total_transitions // (config.num_envs * config.horizon)
    max_horizon = num_iters * config.horizon
    num_grad_updates = num_iters * config.num_minibatches
    eval_freq = num_iters // config.num_evals

    env, env_params = gymnax.make(config.env_name)
    env_params = env_params.replace(max_steps_in_episode=max_horizon)  # don't truncate
    obs_shape = env.observation_space(env_params).shape  # for visualization
    num_actions = env.action_space(env_params).n
    env: gymnax.environments.environment.Environment = FlattenObservationWrapper(env)

    env_reset_batch = jax.vmap(env.reset, in_axes=(0, None))  # map over key
    env_step_batch = jax.vmap(env.step, in_axes=(0, 0, 0, None))  # map over key, state, action

    lr = optax.cosine_decay_schedule(config.lr_init, num_grad_updates)
    optim = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(lr),
    )

    # ceil(batch_size / num_envs)
    trajectories_per_batch = (config.batch_size - 1) // config.num_envs + 1
    # save to dict for logging
    buffer_args = dict(
        add_batch_size=config.num_envs,
        sample_batch_size=config.batch_size,
        sample_sequence_length=config.horizon,
        period=config.horizon // 2,  # avoid some correlations
        min_length_time_axis=trajectories_per_batch * config.horizon,
        max_length_time_axis=max_horizon,  # keep all transitions
        priority_exponent=config.priority_exponent,
    )
    buffer: TrajectoryBuffer = fbx.make_prioritised_trajectory_buffer(**buffer_args)

    def logits_to_value(
        value_logits: Float[Array, " horizon num_value_bins"],
    ) -> Float[Array, " horizon"]:
        """Convert from logits for two-hot encoding to scalar values."""
        if config.num_value_bins == "scalar":
            return value_logits
        else:
            return rlax.transform_from_2hot(
                jax.nn.softmax(value_logits, axis=-1),
                config.min_value,
                config.max_value,
                config.num_value_bins,
            )

    def value_to_probs(
        value: Float[Array, " horizon"],
    ) -> Float[Array, " horizon num_value_bins"]:
        """Convert from scalar values to probabilities for two-hot encoding."""
        if config.num_value_bins == "scalar":
            return value
        else:
            return rlax.transform_to_2hot(
                value,
                config.min_value,
                config.max_value,
                config.num_value_bins,
            )

    def get_invalid_actions(
        env_state: gymnax.EnvState,
    ) -> Bool[Array, "num_actions"]:
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

        sys.stdout.write("Config\n" + "=" * 20 + "\n")
        yaml.safe_dump(asdict(config), sys.stdout)
        sys.stdout.write("=" * 20 + "\n")
        sys.stdout.write("Buffer arguments\n" + "=" * 20 + "\n")
        yaml.safe_dump(buffer_args, sys.stdout)
        sys.stdout.write("=" * 20 + "\n")

        init_obs, init_env_states = env_reset_batch(
            jr.split(key_reset, config.num_envs), env_params
        )

        # initialize all state
        init_net = MuZeroNetwork(
            config=config.arch,
            obs_size=init_obs.shape[-1],
            num_actions=num_actions,
            num_value_bins=config.num_value_bins,
            key=key_net,
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)
        init_hiddens = jnp.broadcast_to(
            init_net.world_model.init_hidden(), (config.num_envs, config.arch.rnn_size)
        )
        init_rollout_states = RolloutState(
            obs=init_obs,
            env_state=init_env_states,
            initial=jnp.ones(config.num_envs, jnp.bool),
        )
        init_buffer_state = buffer.init(
            Transition(
                rollout_state=tree_slice(init_rollout_states, 0),
                action=env.action_space(env_params).sample(jr.key(0)),
                reward=jnp.empty((), init_hiddens.dtype),
                pred=init_net.actor_critic(init_hiddens[0]),
                mcts_probs=jnp.empty(num_actions, init_hiddens.dtype),
            )
        )

        @exec_loop(
            IterState(
                step=0,
                rollout_states=init_rollout_states,
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
                iter_state.rollout_states,
                key_rollout,
            )
            buffer_state = buffer.add(iter_state.param_state.buffer_state, trajectories)

            mean_return = jnp.sum(trajectories.reward) / jnp.sum(trajectories.rollout_state.initial)
            log_values({"iter/step": iter_state.step, "iter/mean_return": mean_return})

            buffer_available = buffer.can_sample(buffer_state)

            @exec_loop(
                iter_state.param_state._replace(buffer_state=buffer_state),
                config.num_minibatches,
                key_optim,
                cond=buffer_available,
            )
            def optimize_step(param_state: ParamState, key: Key[Array, ""]):
                batch = buffer.sample(param_state.buffer_state, key)
                trajectories: Batched[Transition, "batch_size horizon"] = batch.experience

                @ft.partial(jax.value_and_grad, has_aux=True)
                def loss_grad(params: MuZeroNetwork):
                    """Average the loss across a batch of trjaectories."""
                    losses, aux = loss_trajectory(
                        params, iter_state.target_params, trajectories, net_static
                    )
                    chex.assert_shape(losses, (config.batch_size,))
                    # TODO add importance reweighting
                    return jnp.mean(losses), aux

                (_, aux), grads = loss_grad(param_state.params)
                # value_and_grad drops types
                aux: LossStatistics
                grads: MuZeroNetwork
                updates, opt_state = optim.update(grads, param_state.opt_state, param_state.params)
                params = optax.apply_updates(param_state.params, updates)

                # prioritize trajectories with high TD error
                # priorities = jnp.linalg.vector_norm(
                #     aux.td_error, axis=-1, ord=config.priority_td_norm_ord
                # )
                priorities = jnp.mean(jnp.abs(aux.td_error), axis=-1)
                buffer_state = buffer.set_priorities(
                    param_state.buffer_state,
                    batch.indices,
                    priorities,
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

            param_state, _ = optimize_step

            target_params = jax.tree.map(
                lambda online, target: jnp.where(
                    iter_state.step % config.target_update_freq == 0,
                    optax.incremental_update(online, target, config.target_update_size),
                    target,
                ),
                param_state.params,
                iter_state.target_params,
            )

            # evaluate every eval_freq steps
            eval_return = jax.lax.cond(
                (iter_state.step % eval_freq == 0) & buffer_available,
                ft.partial(
                    evaluate,
                    num_envs=config.num_eval_envs,
                    net_static=net_static,
                ),
                lambda *_: -2 * jnp.pi,
                # cond only accepts positional arguments
                param_state.params,
                target_params,
                buffer_state,
                key_evaluate,
            )

            jax.debug.print(
                "step {step}/{num_iters}. "
                "seen {transitions}/{total_transitions} transitions. "
                "mean return {mean_return}",
                step=iter_state.step,
                num_iters=num_iters,
                transitions=param_state.buffer_state.current_index * config.num_envs,
                total_transitions=config.total_transitions,
                mean_return=mean_return,
            )

            return IterState(
                step=iter_state.step + 1,
                rollout_states=rollout_states,
                param_state=param_state,
                target_params=target_params,
            ), eval_return

        final_iter_state, eval_returns = iterate
        return final_iter_state, eval_returns

    def rollout(
        net: MuZeroNetwork,
        init_rollout_state: Batched[RolloutState, "num_envs"],
        key: Key[Array, ""],
    ) -> tuple[Batched[RolloutState, "num_envs"], Batched[Transition, "num_envs horizon"]]:
        @exec_loop(init_rollout_state, config.horizon, key)
        def rollout_step(rollout_states: Batched[RolloutState, "num_envs"], key: Key[Array, ""]):
            key_action, key_step = jr.split(key)

            preds, outs = act_mcts(net, rollout_states, key_action)
            obs, env_states, rewards, initials, infos = env_step_batch(
                jr.split(key_step, outs.action.shape[0]),
                rollout_states.env_state,
                outs.action,
                env_params,
            )
            return RolloutState(
                obs=obs,
                env_state=env_states,
                initial=initials,
            ), Transition(
                # contains the reward and network predictions computed from the rollout state
                rollout_state=rollout_states,
                action=outs.action,
                reward=rewards,
                pred=preds,
                mcts_probs=outs.action_weights,
            )

        final_rollout_state, transitions = rollout_step
        transitions = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )  # swap horizon and num_envs axes

        return final_rollout_state, transitions

    def act_mcts(
        net: MuZeroNetwork,
        rollout_states: Batched[RolloutState, "num_envs"],
        key: Key[Array, ""],
    ):
        """Take a single action via MCTS with the world model."""
        init_hiddens = jax.vmap(net.projection)(rollout_states.obs)
        preds = jax.vmap(net.actor_critic)(init_hiddens)

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
            hiddens, rewards = jax.vmap(net.world_model.step)(hiddens, actions)
            preds = jax.vmap(net.actor_critic)(hiddens)
            return mctx.RecurrentFnOutput(
                reward=logits_to_value(rewards),
                # TODO termination
                discount=jnp.full(actions.shape[0], config.discount),
                prior_logits=preds.policy_logits,
                value=logits_to_value(preds.value_logits),
            ), hiddens

        out = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=mcts_recurrent_fn,
            num_simulations=config.num_mcts_simulations,
            invalid_actions=invalid_actions,
            max_depth=config.horizon,
        )

        return preds, out

    def bootstrap(
        net: MuZeroNetwork,
        trajectory: Batched[Transition, "horizon"],
    ) -> tuple[
        Batched[Prediction, "horizon horizon"],
        Float[Array, "horizon horizon-1"],
    ]:
        """Abstracted out for plotting."""
        action_rolled, reward_rolled, initial_rolled = map(
            roll_into_matrix,
            (trajectory.action, trajectory.reward, trajectory.rollout_state.initial),
        )

        _, (_, pred) = jax.vmap(net)(trajectory.rollout_state.obs, action_rolled)
        value = logits_to_value(pred.value_logits)
        bootstrapped_return: Float[Array, "horizon horizon-1"] = jax.vmap(
            rlax.lambda_returns, (0, 0, 0, None)
        )(
            reward_rolled[:, :-1],
            jnp.where(initial_rolled[:, 1:], 0.0, config.discount),
            value[:, 1:],
            config.lambda_gae,
        )

        return pred, bootstrapped_return

    # jit to ease debugging
    @ft.partial(jax.vmap, in_axes=(None, None, 0, None))
    @ft.partial(jax.jit, static_argnames=("net_static",))
    def loss_trajectory(
        params: MuZeroNetwork,
        target_params: MuZeroNetwork,
        trajectory: Batched[Transition, "horizon"],
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
        _, (online_reward, online_pred) = jax.vmap(net)(trajectory.rollout_state.obs, action_rolled)

        # bootstrap target values
        # discard final transition
        # (too much hassle to pass around final transition)
        target_pred, bootstrapped_return = bootstrap(target_net, trajectory)

        # value loss
        online_value = logits_to_value(online_pred.value_logits)
        if config.num_value_bins == "scalar":
            value_losses = rlax.l2_loss(
                predictions=online_value[:-1, :-1],
                targets=bootstrapped_return[:-1, :],
            )
        else:
            bootstrapped_value_probs = value_to_probs(bootstrapped_return)
            value_losses = rlax.categorical_cross_entropy(
                labels=bootstrapped_value_probs[:-1, :],
                logits=online_pred.value_logits[:-1, :-1],
            )

        # policy loss
        mcts_probs_rolled = roll_into_matrix(trajectory.mcts_probs)
        policy_losses = rlax.categorical_cross_entropy(
            labels=mcts_probs_rolled,
            logits=online_pred.policy_logits,
        )

        # reward model loss
        reward_rolled = roll_into_matrix(trajectory.reward)
        if config.num_value_bins == "scalar":
            reward_losses = rlax.l2_loss(predictions=online_reward, targets=reward_rolled)
        else:
            reward_losses = rlax.categorical_cross_entropy(
                labels=value_to_probs(reward_rolled),
                logits=online_reward,
            )

        # top left triangle of matrix
        horizon_axis = jnp.arange(trajectory.action.size)
        mask = horizon_axis[:, jnp.newaxis] + horizon_axis[jnp.newaxis, :]
        mask = horizon_axis.size - mask
        mask = mask / (mask.sum(where=mask > 0))

        policy_loss = jnp.mean(policy_losses * mask, where=mask > 0)
        value_loss = jnp.mean(value_losses * mask[:-1, :-1], where=mask[:-1, :-1] > 0)
        reward_loss = jnp.mean(reward_losses * mask, where=mask > 0)

        # def debug(n, broken, mcts, p):
        #     if n and not broken:
        #         print("BROKEN")
        #         print(mcts)
        #         print(p)

        # all_broken = jnp.isnan(policy_loss) & jnp.isnan(value_loss) & jnp.isnan(reward_loss)
        # jax.debug.callback(
        #     debug,
        #     jnp.isnan(policy_loss),
        #     all_broken,
        #     mcts_probs_rolled[0],
        #     online_pred.policy_logits,
        # )
        # jax.debug.print("{}", mcts_probs_rolled[0])

        loss = policy_loss + config.value_coef * value_loss + config.reward_coef * reward_loss

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
    ) -> Float[Array, " "]:
        """Evaluate a batch of rollouts using the raw policy.

        Returns mean return across the batch.
        """
        key_reset, key_rollout, key_sample = jr.split(key, 3)

        net = eqx.combine(params, net_static)

        key_reset = jr.split(key_reset, num_envs)
        obs, env_state = env_reset_batch(key_reset, env_params)

        init_rollout_states = RolloutState(
            obs=obs,
            env_state=env_state,
            initial=jnp.zeros(num_envs, dtype=jnp.bool),
        )

        @exec_loop(
            init_rollout_states,
            config.horizon,
            key_rollout,
        )
        def rollout_step(rollout_states: Batched[RolloutState, "num_envs"], key: Key[Array, ""]):
            key_action, key_step, key_mcts = jr.split(key, 3)

            @ft.partial(jax.value_and_grad, has_aux=True)
            def predict(obs: Float[Array, " obs_size"]):
                """Differentiate with respect to value to obtain saliency map."""
                hidden = net.projection(obs)
                pred = net.actor_critic(hidden)
                value = logits_to_value(pred.value_logits[jnp.newaxis])[0]
                return value, pred.policy_logits

            (_, policy_logits), obs_grads = jax.vmap(predict)(rollout_states.obs)
            actions = jr.categorical(key_action, logits=policy_logits, axis=-1)
            obs, env_states, rewards, initials, infos = env_step_batch(
                jr.split(key_step, num_envs),
                rollout_states.env_state,
                actions,
                env_params,
            )

            preds, outs = act_mcts(net, rollout_states, key_mcts)

            return RolloutState(obs=obs, env_state=env_states, initial=initials), (
                Transition(
                    rollout_state=rollout_states,
                    action=actions,
                    reward=rewards,
                    pred=preds,
                    mcts_probs=outs.action_weights,
                ),
                obs_grads,
            )

        _, (trajectories, obs_grads) = rollout_step
        trajectories, obs_grads = jax.tree.map(
            lambda x: x.swapaxes(0, 1), (trajectories, obs_grads)
        )
        trajectories: Batched[Transition, "num_envs horizon"]

        target_net = eqx.combine(target_params, net_static)
        _, bootstrapped_returns = jax.vmap(bootstrap, in_axes=(None, 0))(target_net, trajectories)

        jax.debug.callback(
            visualize_callback,
            trajectories,
            bootstrapped_returns[:, 0],
            jax.vmap(visualize_catch, in_axes=(None, 0, 0))(
                obs_shape,
                trajectories.rollout_state.env_state,
                obs_grads,
            ),
            prefix="eval",
        )

        eval_return = jnp.sum(trajectories.reward) / jnp.sum(trajectories.rollout_state.initial)
        log_values({"eval/mean_return": eval_return})

        # debug training procedure

        batch = buffer.sample(buffer_state, key_sample)
        sampled_trajectories: Batched[Transition, "num_envs horizon"] = tree_slice(
            batch.experience, slice(0, num_envs)
        )
        _, aux = loss_trajectory(params, target_params, sampled_trajectories, net_static)

        jax.debug.callback(
            visualize_callback,
            trajectories=sampled_trajectories._replace(
                pred=Prediction(
                    value_logits=aux.value_logits,
                    policy_logits=aux.policy_logits,
                )
            ),
            bootstrapped_returns=aux.bootstrapped_return,
            video=jax.vmap(visualize_catch, (None, 0))(
                obs_shape,
                sampled_trajectories.rollout_state.env_state,
            ),
            prefix="visualize",
            priorities=batch.priorities,
            # target_value_logits=aux["train/target_value_logits"],
        )

        return eval_return

    def visualize_callback(
        trajectories: Batched[Transition, "num_envs horizon"],
        bootstrapped_returns: Float[Array, " num_envs horizon-1"],
        video: Float[Array, " num_envs horizon 3 height width"],
        prefix: str,
        priorities: Float[Array, " num_envs"] | None = None,
    ) -> None:
        if not wandb.run:
            return

        fig_width = 3
        rows_per_env = 2
        if config.num_value_bins != "scalar":
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

            if config.num_value_bins != "scalar":
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
                if config.num_value_bins != "scalar":
                    ax_value.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3)

        # separate figure for plotting video
        fig_video, ax_video = plt.subplots(
            nrows=num_envs,
            ncols=horizon,
            squeeze=False,
            figsize=(horizon * 2, num_envs * 3),
        )

        for i in range(num_envs):
            for h in range(horizon):
                ax_stats: Axes = ax_video[i, h]
                initial = trajectories.rollout_state.initial[i, h].item()
                a = trajectories.action[i, h].item()
                # c h w -> h w c
                ax_stats.imshow(jnp.permute_dims(video[i, h], (1, 2, 0)))
                ax_stats.set_title(f"{h=} {initial=} {a=}")
                ax_stats.axis("off")

        wandb.log(
            {
                # "eval/video": wandb.Video(np.asarray(video), fps=10),
                # use wandb.Image since otherwise wandb uses plotly,
                # which breaks the legends
                f"{prefix}/statistics": wandb.Image(fig_stats),
                f"{prefix}/trajectories": wandb.Image(fig_video),
            }
        )

        plt.close(fig_stats)
        plt.close(fig_video)

    def plot_statistics(
        ax: Axes,
        trajectory: Batched[Transition, "horizon"],
        bootstrapped_return: Float[Array, " horizon"],
    ):
        horizon = jnp.arange(trajectory.action.size)

        # misc
        initial_idx = horizon[trajectory.rollout_state.initial]
        ax.plot(
            initial_idx,
            jnp.zeros_like(initial_idx),
            "k>",
            label="initial",
            alpha=0.5,
        )

        # value
        online_value = logits_to_value(trajectory.pred.value_logits)
        ax.plot(horizon, trajectory.reward, "r+", label="reward")
        ax.plot(horizon, online_value, "bo", label="online value")
        ax.plot(
            horizon[:-1],
            bootstrapped_return,
            "mo",
            label="bootstrapped returns",
            alpha=0.5,
        )
        ax.fill_between(
            horizon[:-1],
            bootstrapped_return,
            online_value[:-1],
            alpha=0.3,
            label="TD error",
        )

        # entropy
        policy_dist = distrax.Categorical(logits=trajectory.pred.policy_logits)
        mcts_dist = distrax.Categorical(probs=trajectory.mcts_probs)
        ax.plot(horizon, policy_dist.entropy(), "b:", label="policy entropy")
        ax.plot(horizon, mcts_dist.entropy(), "g:", label="MCTS entropy")
        if config.num_value_bins != "scalar":
            value_dist = distrax.Categorical(logits=trajectory.pred.value_logits)
            ax.plot(horizon, value_dist.entropy(), "y:", label="value entropy")

        # loss
        ax.plot(horizon, mcts_dist.kl_divergence(policy_dist), "c--", label="policy / mcts kl")
        if config.num_value_bins != "scalar":
            bootstrapped_dist = distrax.Categorical(probs=value_to_probs(bootstrapped_return))
            value_loss = bootstrapped_dist.kl_divergence(value_dist[:-1])
            ax.plot(horizon[:-1], value_loss, "r--", label="value / bootstrap kl")
        else:
            value_loss = rlax.l2_loss(online_value[:-1], bootstrapped_return)
            ax.plot(horizon[:-1], value_loss, "r--", label="value / bootstrap l2")

        ax.set_xticks(horizon, horizon)
        ax.set_ylim(config.min_value, config.max_value)

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
    default_config = Config()
    seed = 0

    WANDB_PROJECT = "alphazero"

    if len(sys.argv) >= 2:
        if sys.argv[1] == "sweep":
            num_runs = 6
            config_values = jax.tree.map(lambda x: {"value": x}, default_config._asdict()) | {
                "lr_init": {"min": 1e-5, "max": 1e-3},
                "num_minibatches": {"values": [64, 128]},
            }
            sweep_config = {
                "method": "random",
                "metric": {"goal": "maximize", "name": "sweep/mean_reward"},
                "parameters": config_values,
            }

            def run():
                with wandb.init(project=WANDB_PROJECT):
                    train = make_train(wandb.config)
                    _, final_reward = jax.jit(train)(jr.key(seed))
                    wandb.log({"sweep/mean_reward": final_reward})

            sweep_id = (
                sys.argv[2]
                if len(sys.argv) >= 3
                else wandb.sweep(sweep_config, project=WANDB_PROJECT)
            )

            wandb.agent(
                sweep_id,
                function=run,
                project=WANDB_PROJECT,
                count=num_runs,
            )

        elif sys.argv[1] == "debug":
            train = make_train(debug_config)
            with jax.disable_jit():
                output = train(jr.key(seed))
            jax.block_until_ready(output)
    else:
        train = make_train(default_config)
        with wandb.init(
            project=WANDB_PROJECT,
            config=asdict(default_config),
            mode="disabled",
        ) as run:
            output = jax.jit(train)(jr.key(seed))
            # with jax.profiler.trace(f"/tmp/{WANDB_PROJECT}-trace", create_perfetto_link=True):
            _, mean_eval_reward = jax.block_until_ready(output)
            sys.stdout.write(f"{mean_eval_reward=}\n")

    sys.stdout.write("Done training\n")
