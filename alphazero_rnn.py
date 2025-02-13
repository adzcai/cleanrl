"""Implementation of AlphaZero with a recurrent actor-critic network.

1. Config
2. Data structures for loop carries
3. Model definitions
4. Training loop
5. Rollout function
6. Evaluation function
"""

# jax
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
from jaxtyping import Bool, Integer, Float, Key, Array
from typing import Literal, NamedTuple, Annotated as Batched

# util
import functools as ft
from log_util import (
    exec_loop,
    get_norm_data,
    log_values,
    tree_slice,
    visualize_catch,
)

matplotlib.use("agg")  # enable plotting inside jax callback


class Config(NamedTuple):
    env_name: str = "Catch-bsuite"
    # network architecture
    rnn_size: int = 32
    mlp_size: int = 32
    mlp_depth: int = 2
    activation: Literal["relu", "tanh"] = "tanh"
    # collection
    total_transitions: int = 10_000
    num_envs: int = 16  # more parallel data collection. too large crashes fbx
    horizon: int = 15  # also mcts max depth
    # two-hot value
    num_value_bins: int | Literal["scalar"] = 4
    min_value: float = -2.0
    max_value: float = 2.0
    # search
    num_mcts_simulations: int = 64  # stronger policy improvement
    # bootstrapping
    discount: float = 0.997  # R2D2 standard
    bootstrap_n: int = 5
    lambda_gae: float = 0.95
    # optimization
    num_minibatches: int = 1  # number of gradient descent steps per iteration
    batch_size: int = 1  # reduce gradient variance
    lr_init: float = 1e-2
    max_grad_norm: float = 0.5
    # target
    target_update_freq: int = 8
    target_update_size: float = 0.6
    # loss
    value_coef: float = 1.0
    # evaluation
    num_evals: int = 4
    num_eval_envs: int = 4


class Prediction(NamedTuple):
    policy_logits: Float[Array, "num_actions"]
    value_logits: Float[Array, "num_value_bins"]


class ActorCriticRNN(eqx.Module):
    cell: eqx.nn.GRUCell
    policy_head: eqx.nn.MLP
    value_head: eqx.nn.MLP

    def __init__(
        self,
        obs_size: int,
        rnn_size: int,
        mlp_size: int,
        mlp_depth: int,
        num_actions: int,
        num_value_bins: int | Literal["scalar"],
        activation: Literal["relu", "tanh"],
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_policy, key_value = jr.split(key, 3)

        activation = jax.nn.relu if activation == "relu" else jax.nn.tanh

        self.cell = (
            eqx.nn.GRUCell(obs_size, rnn_size, key=key_cell) if rnn_size > 0 else None
        )

        in_size = rnn_size if self.is_rnn else obs_size

        self.policy_head = eqx.nn.MLP(
            in_size, num_actions, mlp_size, mlp_depth, activation, key=key_policy
        )
        self.value_head = eqx.nn.MLP(
            in_size, num_value_bins, mlp_size, mlp_depth, activation, key=key_value
        )

    def __call__(
        self,
        hidden: Float[Array, "rnn_size"],
        obs: Float[Array, "horizon obs_size"],
        is_initial: Bool[Array, "horizon"],
    ):
        return jax.lax.scan(
            lambda hidden, x: self.step(hidden, *x),
            hidden,
            (obs, is_initial),
        )

    def step(
        self,
        hidden: Float[Array, "rnn_size"],
        obs: Float[Array, "obs_size"],
        is_initial: Bool[Array, ""],
    ):
        """Returns the predicted policy and value for this observation."""
        hidden = jnp.where(is_initial, self.init_hidden(), hidden)
        hidden = self.cell(obs, hidden) if self.is_rnn else obs
        pred = Prediction(
            policy_logits=self.policy_head(hidden),
            value_logits=self.value_head(hidden),
        )
        return hidden, pred

    def init_hidden(self):
        return jnp.zeros(self.cell.hidden_size if self.is_rnn else 0)

    @property
    def is_rnn(self):
        return self.cell is not None


class Unobs(NamedTuple):
    """`hidden` is the representation of `env_state`"""

    env_state: gymnax.EnvState
    hidden: Float[Array, "rnn_size"]
    is_initial: Bool[Array, ""]


class RolloutState(NamedTuple):
    """Carried when rolling out the environment."""

    obs: Float[Array, "obs_size"]
    unobs: Unobs


class Transition(NamedTuple):
    """A single transition. May be batched into a trajectory."""

    rollout_state: RolloutState
    reward: Float[Array, ""]
    pred: Prediction
    mcts_logits: Float[Array, "num_actions"]


class ParamState(NamedTuple):
    """Carried during optimization."""

    params: ActorCriticRNN
    opt_state: optax.OptState


BufferState = fbx.trajectory_buffer.TrajectoryBufferState[Transition]


class IterState(NamedTuple):
    """Carried across algorithm iterations."""

    step: Integer[Array, ""]
    rollout_states: Batched[RolloutState, "num_envs"]
    param_state: ParamState
    target_params: ActorCriticRNN
    buffer_state: BufferState


def make_train(config: Config):
    num_iters = config.total_transitions // (config.num_envs * config.horizon)
    num_updates = num_iters * config.num_minibatches
    eval_freq = num_iters // config.num_evals

    env, env_params = gymnax.make(config.env_name)
    obs_shape = env.observation_space(env_params).shape  # for visualization
    num_actions = env.action_space(env_params).n
    env = FlattenObservationWrapper(env)

    env_reset_batch = jax.vmap(env.reset, (0, None))  # map over key
    env_step_batch = jax.vmap(env.step, (0, 0, 0, None))  # map over key, state, action

    lr = optax.cosine_decay_schedule(config.lr_init, num_updates)
    optim = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(lr),
    )

    # sample config.batch_size trajectories of config.horizon transitions

    buffer_args = dict(
        add_batch_size=config.num_envs,
        sample_batch_size=config.batch_size,
        sample_sequence_length=config.horizon,
        period=config.horizon // 4,  # avoid some correlations
        min_length_time_axis=(
            # ceil(batch_size / num_envs)
            config.batch_size + (config.num_envs - 1) // config.num_envs
        )
        * config.horizon,
        max_length_time_axis=num_iters * config.horizon,  # keep all transitions
    )
    buffer = fbx.make_trajectory_buffer(**buffer_args)

    def logits_to_values(
        value_logits: Float[Array, "horizon num_value_bins"],
    ) -> Float[Array, "horizon"]:
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

    def train(key: Key[Array, ""]):
        key_reset, key_net, key_iter, key_evaluate = jr.split(key, 4)

        sys.stdout.write("Config\n" + "=" * 20 + "\n")
        yaml.safe_dump(config._asdict(), sys.stdout)
        sys.stdout.write("=" * 20 + "\n")
        sys.stdout.write("Buffer arguments\n" + "=" * 20 + "\n")
        yaml.safe_dump(buffer_args, sys.stdout)
        sys.stdout.write("=" * 20 + "\n")

        obs, env_state = env_reset_batch(
            jr.split(key_reset, config.num_envs), env_params
        )

        # initialize all state
        init_net = ActorCriticRNN(
            obs_size=obs.shape[-1],
            rnn_size=config.rnn_size,
            mlp_size=config.mlp_size,
            mlp_depth=config.mlp_depth,
            num_actions=num_actions,
            num_value_bins=config.num_value_bins,
            activation=config.activation,
            key=key_net,
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)
        init_rollout_state = RolloutState(
            obs=obs,
            unobs=Unobs(
                env_state=env_state,
                hidden=jnp.broadcast_to(
                    init_net.init_hidden(),
                    (config.num_envs, config.rnn_size),
                ),
                is_initial=jnp.ones(config.num_envs, bool),
            ),
        )
        init_buffer_state = buffer.init(
            Transition(
                rollout_state=tree_slice(init_rollout_state, 0),
                reward=jnp.empty((), float),
                pred=init_net.step(init_net.init_hidden(), obs[0], True)[1],
                mcts_logits=jnp.empty(num_actions, float),
            )
        )

        @exec_loop(
            IterState(
                step=0,
                rollout_states=init_rollout_state,
                param_state=ParamState(
                    params=init_params, opt_state=optim.init(init_params)
                ),
                target_params=init_params,
                buffer_state=init_buffer_state,
            ),
            num_iters,
            key_iter,
        )
        def iterate(iter_state: IterState, key: Key[Array, ""]):
            key_rollout, key_optim = jr.split(key)

            # collect data
            rollout_states, trajectories = rollout(
                eqx.combine(iter_state.param_state.params, net_static),
                iter_state.rollout_states,
                key_rollout,
            )
            buffer_state: BufferState = buffer.add(
                iter_state.buffer_state, trajectories
            )

            log_values(
                {
                    "train/mean_return": jnp.sum(trajectories.reward)
                    / jnp.sum(trajectories.rollout_state.unobs.is_initial),
                }
            )

            def optimize_step(param_state: ParamState, key: Key[Array, ""]):
                trajectories: Batched[Transition, "batch_size horizon"] = buffer.sample(
                    buffer_state, key
                ).experience

                @ft.partial(jax.value_and_grad, has_aux=True)
                def loss_grad(params: ActorCriticRNN):
                    """Average the loss across a batch of trjaectories."""
                    losses, aux = jax.vmap(loss_trajectory, (None, None, 0))(
                        eqx.combine(params, net_static),
                        eqx.combine(iter_state.target_params, net_static),
                        trajectories,
                    )
                    chex.assert_shape(losses, (config.batch_size,))
                    return jnp.mean(losses), aux

                (_, aux), grads = loss_grad(param_state.params)
                updates, opt_state = optim.update(
                    grads, param_state.opt_state, param_state.params
                )
                params = optax.apply_updates(param_state.params, updates)

                log_values(
                    jax.tree.map(jnp.mean, aux)
                    | get_norm_data(updates, "updates/norm")
                    | get_norm_data(params, "params/norm")
                )

                return ParamState(params, opt_state), None

            param_state, _ = jax.lax.cond(
                buffer.can_sample(buffer_state),
                lambda param_state, key: exec_loop(
                    param_state,
                    config.num_minibatches,
                    key,
                )(optimize_step),
                lambda param_state, _: (param_state, None),
                iter_state.param_state,
                key_optim,
            )
            param_state: ParamState

            target_params = jax.tree.map(
                lambda new, old: jnp.where(
                    iter_state.step % config.target_update_freq == 0,
                    optax.incremental_update(new, old, config.target_update_size),
                    old,
                ),
                param_state.params,
                iter_state.target_params,
            )

            # evaluate every eval_freq steps
            eval_return = jax.lax.cond(
                iter_state.step % eval_freq == 0 & buffer.can_sample(buffer_state),
                ft.partial(
                    evaluate,
                    num_envs=config.num_eval_envs,
                    net_static=net_static,
                ),
                lambda *_: jnp.nan,
                # cond only accepts positional arguments
                param_state.params,
                target_params,
                buffer_state,
                key_evaluate,
            )

            return IterState(
                step=iter_state.step + 1,
                rollout_states=rollout_states,
                param_state=param_state,
                target_params=target_params,
                buffer_state=buffer_state,
            ), eval_return

        final_iter_state, eval_returns = iterate
        return final_iter_state, eval_returns

    def rollout(
        net: ActorCriticRNN,
        init_rollout_state: Batched[RolloutState, "num_envs"],
        key: Key[Array, ""],
    ) -> tuple[
        Batched[RolloutState, "num_envs"], Batched[Transition, "num_envs horizon"]
    ]:
        @exec_loop(init_rollout_state, config.horizon, key)
        def rollout_step(
            rollout_states: Batched[RolloutState, "num_envs"], key: Key[Array, ""]
        ):
            key_action, key_step = jr.split(key)

            hiddens, preds, out = act_mcts(net, rollout_states, key_action)
            obs, env_states, rewards, is_initials, infos = env_step_batch(
                jr.split(key_step, out.action.size),
                rollout_states.unobs.env_state,
                out.action,
                env_params,
            )
            return RolloutState(
                obs=obs,
                unobs=Unobs(
                    env_state=env_states,
                    hidden=hiddens,
                    is_initial=is_initials,
                ),
            ), Transition(
                # contains the reward and policy logits computed from the rollout state
                rollout_state=rollout_states,
                reward=rewards,
                pred=preds,
                mcts_logits=out.action_weights,
            )

        final_rollout_state, transitions = rollout_step
        return final_rollout_state, jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )

    def act_mcts(
        net: ActorCriticRNN,
        rollout_state: Batched[RolloutState, "num_envs"],
        key: Key[Array, ""],
    ):
        hiddens, preds = jax.vmap(net.step)(
            rollout_state.unobs.hidden,
            rollout_state.obs,
            rollout_state.unobs.is_initial,
        )

        root = mctx.RootFnOutput(
            prior_logits=preds.policy_logits,
            value=logits_to_values(preds.value_logits),
            # embedding contains current env_state and the hidden to pass to next net call
            embedding=rollout_state.unobs._replace(hidden=hiddens),
        )

        def mcts_recurrent_fn(
            _: None,  # closure net
            key: Key[Array, ""],
            action: Integer[Array, "num_envs"],
            unobs: Batched[Unobs, "num_envs"],
        ):
            """Returns the logits and value for the newly created node."""
            obs, env_states, rewards, is_initials, infos = env_step_batch(
                jr.split(key, action.size), unobs.env_state, action, env_params
            )
            hiddens, preds = jax.vmap(net.step)(unobs.hidden, obs, is_initials)
            return mctx.RecurrentFnOutput(
                reward=rewards,
                # careful for off-by-one
                # if new state is initial,
                # preds.value_logits is for a new trajectory
                discount=jnp.where(is_initials, 0.0, config.discount),
                prior_logits=preds.policy_logits,
                value=logits_to_values(preds.value_logits),
            ), Unobs(env_state=env_states, hidden=hiddens, is_initial=is_initials)

        out = mctx.muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=mcts_recurrent_fn,
            num_simulations=config.num_mcts_simulations,
            max_depth=config.horizon,
        )

        return hiddens, preds, out

    def loss_trajectory(
        net: ActorCriticRNN,
        target_net: ActorCriticRNN,
        trajectory: Batched[Transition, "horizon"],
    ):
        """Compute behavior cloning loss for the `trajectory`. Bootstrap values from `target_net`."""
        init_hidden = trajectory.rollout_state.unobs.hidden[0, :]
        obs = trajectory.rollout_state.obs
        is_initials = trajectory.rollout_state.unobs.is_initial

        _, net_pred = net(init_hidden, obs, is_initials)
        chex.assert_shape(
            [trajectory.mcts_logits, net_pred.policy_logits],
            (config.horizon, num_actions),
        )

        # policy loss
        mcts_dist = distrax.Categorical(logits=trajectory.mcts_logits)
        pred_policy_dist = distrax.Categorical(logits=net_pred.policy_logits)
        policy_losses = mcts_dist.cross_entropy(pred_policy_dist)
        policy_loss = jnp.mean(policy_losses)

        # value loss
        # bootstrap from target network
        _, target_pred = target_net(init_hidden, obs, is_initials)
        target_values = logits_to_values(target_pred.value_logits)
        bootstrapped_returns = rlax.n_step_bootstrapped_returns(
            # careful for off-by-one
            # reward[0] is the reward obtained from rollout_state[0]
            # is_initial[1] iff rollout_state[1] is part of new trajectory
            trajectory.reward[:-1],
            jnp.where(is_initials[1:], 0.0, config.discount),
            target_values[1:],
            config.bootstrap_n,
            config.lambda_gae,
        )
        value_losses = (
            rlax.l2_loss(
                predictions=logits_to_values(net_pred.value_logits)[:-1],
                targets=bootstrapped_returns,
            )
            if config.num_value_bins == "scalar"
            else distrax.Categorical(
                probs=rlax.transform_to_2hot(
                    bootstrapped_returns[:, jnp.newaxis],
                    config.min_value,
                    config.max_value,
                    config.num_value_bins,
                )
            ).kl_divergence(
                distrax.Categorical(logits=net_pred.value_logits[:-1, :]),
            )
        )
        value_loss = jnp.mean(value_losses)

        jax.debug.callback(
            visualize_cb,
            trajectories=jax.tree.map(
                lambda x: x[jnp.newaxis], trajectory._replace(pred=net_pred)
            ),
            bootstrapped_returns=bootstrapped_returns[jnp.newaxis],
            video=visualize_catch(
                obs_shape,
                jax.tree.map(
                    lambda x: x[jnp.newaxis], trajectory.rollout_state.unobs.env_state
                ),
            ),
            prefix="voir",
            target_value_logits=target_pred.value_logits[jnp.newaxis],
        )

        # logging
        pred_value = logits_to_values(net_pred.value_logits)
        td_error = bootstrapped_returns - pred_value[:-1]

        loss = policy_loss + config.value_coef * value_loss
        return loss, {
            "train/loss": loss,
            "train/mcts_entropy": mcts_dist.entropy(),
            "train/policy_entropy": pred_policy_dist.entropy(),
            "train/value_logits": net_pred.value_logits,
            "train/target_value_logits": target_pred.value_logits,
            "train/policy_logits": net_pred.policy_logits,
            "train/bootstrapped_returns": bootstrapped_returns,
            "train/policy_loss": policy_losses,
            "train/value_loss": value_losses,
            "train/td_error": td_error,
        }

    def evaluate(
        params: ActorCriticRNN,
        target_params: ActorCriticRNN,
        buffer_state: BufferState,
        key: Key[Array, ""],
        *,
        num_envs: int,
        net_static: ActorCriticRNN,
    ) -> Float[Array, ""]:
        """Evaluate a batch of rollouts using the raw policy.

        Returns mean return across the batch.
        """
        key_reset, key_rollout, key_sample = jr.split(key, 3)

        net = eqx.combine(params, net_static)

        obs, env_state = env_reset_batch(jr.split(key_reset, num_envs), env_params)

        init_rollout_state = RolloutState(
            obs=obs,
            unobs=Unobs(
                env_state=env_state,
                hidden=jnp.broadcast_to(net.init_hidden(), (num_envs, config.rnn_size)),
                is_initial=jnp.zeros(num_envs, dtype=bool),
            ),
        )

        @exec_loop(
            init_rollout_state,
            config.horizon,
            key_rollout,
        )
        def rollout_step(
            rollout_states: Batched[RolloutState, "num_envs"], key: Key[Array, ""]
        ):
            key_action, key_step, key_mcts = jr.split(key, 3)

            @ft.partial(jax.value_and_grad, has_aux=True)
            def predict(obs: Float[Array, "obs_size"], unobs: Unobs):
                """Differentiate with respect to value to obtain saliency map."""
                hidden, pred = net.step(
                    unobs.hidden,
                    obs,
                    unobs.is_initial,
                )

                value = logits_to_values(pred.value_logits[jnp.newaxis])[0]
                return value, (hidden, pred.policy_logits)

            (values, (hiddens, policy_logits)), obs_grads = jax.vmap(predict)(
                rollout_states.obs, rollout_states.unobs
            )
            action = jr.categorical(key_action, logits=policy_logits, axis=-1)
            obs, env_states, rewards, is_initials, info = env_step_batch(
                jr.split(key_step, num_envs),
                rollout_states.unobs.env_state,
                action,
                env_params,
            )

            _, pred, mcts_out = act_mcts(net, rollout_states, key_mcts)

            return RolloutState(
                obs=obs,
                unobs=Unobs(
                    env_state=env_states,
                    hidden=hiddens,
                    is_initial=is_initials,
                ),
            ), (
                Transition(
                    rollout_state=rollout_states,
                    reward=rewards,
                    pred=pred,
                    mcts_logits=mcts_out.action_weights,
                ),
                obs_grads,
            )

        _, (trajectories, obs_grads) = rollout_step
        trajectories: Batched[Transition, "num_envs horizon"] = jax.tree.map(
            lambda x: x.swapaxes(0, 1), trajectories
        )

        is_initial = trajectories.rollout_state.unobs.is_initial
        bootstrapped_returns = jax.vmap(
            rlax.n_step_bootstrapped_returns,
            (0, 0, 0, None, None),
        )(
            trajectories.reward[:, :-1],
            jnp.where(is_initial[:, 1:], 0.0, config.discount),
            logits_to_values(trajectories.pred.value_logits)[:, 1:],
            config.bootstrap_n,
            config.lambda_gae,
        )

        jax.debug.callback(
            visualize_cb,
            trajectories,
            bootstrapped_returns,
            visualize_catch(
                obs_shape,
                trajectories.rollout_state.unobs.env_state,
                obs_grads,
            ),
            prefix="eval",
        )

        eval_return = jnp.sum(trajectories.reward) / jnp.sum(is_initial)

        log_values({"eval/mean_return": eval_return})

        # # debug training procedure
        # target_net = eqx.combine(target_params, net_static)
        # jax.debug.print("can sample? {}", buffer.can_sample(buffer_state))
        # sampled_trajectories: Batched[Transition, "num_envs horizon"] = buffer.sample(
        #     buffer_state, key_sample
        # ).experience
        # sampled_trajectories = tree_slice(sampled_trajectories, slice(0, num_envs))
        # _, aux = jax.vmap(loss_trajectory, (None, None, 0))(
        #     net, target_net, sampled_trajectories
        # )

        # jax.debug.callback(
        #     visualize_cb,
        #     trajectories=sampled_trajectories._replace(
        #         pred=Prediction(
        #             value_logits=aux["train/value_logits"],
        #             policy_logits=aux["train/policy_logits"],
        #         )
        #     ),
        #     bootstrapped_returns=aux["train/bootstrapped_returns"],
        #     video=visualize_catch(
        #         obs_shape,
        #         sampled_trajectories.rollout_state.unobs.env_state,
        #     ),
        #     prefix="visualize",
        #     # target_value_logits=aux["train/target_value_logits"],
        # )

        jax.debug.print(
            "seen {} transitions. mean eval return {}",
            buffer_state.current_index * config.num_envs,
            eval_return,
        )

        return eval_return

    def visualize_cb(
        trajectories: Batched[Transition, "num_envs horizon"],
        bootstrapped_returns: Float[Array, "num_envs horizon-1"],
        video: Float[Array, "num_envs horizon 3 height width"],
        prefix: str,
        target_value_logits: Float[Array, "num_envs horizon num_value_bins"]
        | None = None,
    ) -> None:
        if not wandb.run:
            return

        fig_width = 3
        rows_per_env = 5 if target_value_logits is not None else 4
        num_envs, horizon = trajectories.reward.shape
        horizon = jnp.arange(horizon)

        fig, ax = plt.subplots(
            nrows=num_envs * rows_per_env,
            figsize=(2 * fig_width, 4 * num_envs * rows_per_env),
        )
        for i in range(num_envs):
            policy_dist = distrax.Categorical(logits=trajectories.pred.policy_logits[i])
            mcts_dist = distrax.Categorical(logits=trajectories.mcts_logits[i])
            value_dist = distrax.Categorical(logits=trajectories.pred.value_logits[i])
            values = logits_to_values(value_dist.logits)
            bins = jnp.linspace(
                config.min_value, config.max_value, config.num_value_bins
            )
            # compute variance of value distribution
            value_var = jnp.sum(
                jnp.square(bins[jnp.newaxis, :] - values[:, jnp.newaxis])
                * value_dist.probs,
                axis=-1,
            )

            axes: Axes = ax[rows_per_env * i]
            axes.set_title(f"Trajectory {i}")
            axes.plot(horizon, trajectories.reward[i], "r+", label="reward")
            axes.errorbar(
                horizon + 1 / 3,
                values,
                yerr=jnp.sqrt(value_var),
                fmt="bo",
                label="value",
                alpha=0.5,
            )
            if target_value_logits is not None:
                target_values = logits_to_values(target_value_logits[i])
                target_value_var = jnp.sum(
                    jnp.square(bins[jnp.newaxis, :] - target_values[:, jnp.newaxis])
                    * jax.nn.softmax(target_value_logits[i], axis=-1),
                    axis=-1,
                )
                axes.errorbar(
                    horizon + 2 / 3,
                    target_values,
                    yerr=jnp.sqrt(target_value_var),
                    fmt="go",
                    label="target value",
                    alpha=0.5,
                )
            axes.plot(
                horizon,
                jnp.linalg.norm(trajectories.rollout_state.unobs.hidden[i], axis=-1),
                "cx",
                label="hidden norm",
                alpha=0.5,
            )
            initial_indices = horizon[trajectories.rollout_state.unobs.is_initial[i]]
            axes.plot(
                initial_indices,
                -jnp.ones_like(initial_indices),
                "k|",
                label="initial",
                alpha=0.5,
            )
            axes.plot(
                horizon[:-1],
                bootstrapped_returns[i],
                "mo",
                label="bootstrapped returns",
                alpha=0.5,
            )
            axes.fill_between(
                horizon[:-1],
                bootstrapped_returns[i],
                values[:-1],
                alpha=0.3,
                label="TD error",
            )
            axes.plot(
                horizon,
                policy_dist.entropy(),
                "b",
                label="policy entropy",
            )
            axes.plot(
                horizon,
                mcts_dist.entropy(),
                "g",
                label="MCTS entropy",
            )
            axes.plot(
                horizon,
                policy_dist.kl_divergence(mcts_dist),
                "c--",
                label="policy / mcts kl",
            )
            axes.plot(
                horizon[:-1],
                distrax.Categorical(
                    probs=rlax.transform_to_2hot(
                        bootstrapped_returns[i],
                        config.min_value,
                        config.max_value,
                        config.num_value_bins,
                    )
                ).kl_divergence(value_dist[:-1]),
                "r--",
                label="value / bootstrap kl",
            )
            axes.set_ylim(config.min_value, config.max_value)

            # policy distribution
            policy_axes: Axes = ax[rows_per_env * i + 1]
            policy_axes.set_title(f"Trajectory {i} policy")
            policy_axes.stackplot(
                horizon,
                policy_dist.probs.swapaxes(0, 1),
                labels=range(num_actions),
            )
            policy_axes.set_ylim(0, 1)

            # mcts policy distribution
            mcts_axes: Axes = ax[rows_per_env * i + 2]
            mcts_axes.set_title(f"Trajectory {i} MCTS")
            mcts_axes.stackplot(
                horizon,
                mcts_dist.probs.swapaxes(0, 1),
                labels=range(num_actions),
            )
            mcts_axes.set_ylim(0, 1)

            value_axes: Axes = ax[rows_per_env * i + 3]
            value_axes.set_title(f"Trajectory {i} value")
            value_axes.stackplot(
                horizon,
                value_dist.probs.swapaxes(0, 1),
                labels=bins,
            )
            value_axes.set_ylim(0, 1)

            if target_value_logits is not None:
                target_value_axes: Axes = ax[rows_per_env * i + 4]
                target_value_axes.set_title(f"Trajectory {i} target value")
                target_value_axes.stackplot(
                    horizon,
                    jax.nn.softmax(target_value_logits[i], axis=-1).swapaxes(0, 1),
                    labels=bins,
                )
                target_value_axes.set_ylim(0, 1)

            if i == 0:
                axes.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.20),
                    ncol=3,
                )
                policy_axes.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.20),
                    ncol=3,
                )

        fig_trajectory, ax_trajectory = plt.subplots(
            nrows=num_envs,
            ncols=config.horizon,
            squeeze=False,
            figsize=(config.horizon * 2, num_envs * 3),
        )

        for i in range(num_envs):
            for j in horizon:
                axes: Axes = ax_trajectory[i, j]
                # c h w -> h w c
                axes.imshow(jnp.permute_dims(video[i, j], (1, 2, 0)))
                axes.set_title(f"Trajectory {i} step {j}")
                axes.axis("off")

        wandb.log(
            {
                # "eval/video": wandb.Video(np.asarray(video), fps=10),
                # use wandb.Image since otherwise wandb uses plotly,
                # which breaks the legends
                f"{prefix}/statistics": wandb.Image(fig),
                f"{prefix}/trajectories": wandb.Image(fig_trajectory),
            }
        )

        plt.close(fig)
        plt.close(fig_trajectory)

    return train


if __name__ == "__main__":
    default_config = Config()
    seed = 0

    WANDB_PROJECT = "alphazero"

    if len(sys.argv) >= 2:
        if sys.argv[1] == "sweep":
            num_runs = 6
            config_values = jax.tree.map(
                lambda x: {"value": x}, default_config._asdict()
            ) | {
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
            train = make_train(default_config)
            with jax.disable_jit():
                output = train(jr.key(seed))
            jax.block_until_ready(output)
    else:
        train = make_train(default_config)
        with wandb.init(project=WANDB_PROJECT, config=default_config._asdict()):
            output = jax.jit(train)(jr.key(seed))
            # with jax.profiler.trace(f"/tmp/{WANDB_PROJECT}-trace", create_perfetto_link=True):
            _, mean_eval_reward = jax.block_until_ready(output)
            sys.stdout.write(f"{mean_eval_reward=}\n")

    sys.stdout.write("Done training\n")
