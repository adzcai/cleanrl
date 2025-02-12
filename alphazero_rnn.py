"""Implementation of AlphaZero with a recurrent actor-critic network.

1. Config
2. Data structures for loop carries
3. Model definitions
4. Training loop
5. Rollout function
6. Evaluation function
"""

import distrax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import equinox as eqx
import flashbax as fbx
import rlax
import mctx
import optax
import gymnax
from gymnax.wrappers import FlattenObservationWrapper

import wandb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import yaml
import functools as ft

from jaxtyping import Bool, Integer, Float, Key, Array
from typing import Callable, Literal, NamedTuple, TypeVar, Annotated as Batched

from log_util import get_norm_data, log_values, tree_slice, visualize_catch

matplotlib.use("agg")  # enable plotting inside jax callback


class Config(NamedTuple):
    env_name: str = "Catch-bsuite"
    # network architecture
    rnn_size: int = 64
    mlp_size: int = 64
    mlp_depth: int = 2
    activation: Literal["relu", "tanh"] = "tanh"
    # collection
    total_transitions: int = 500_000
    num_envs: int = 8
    horizon: int = 10
    # two-hot value
    num_value_bins: int | Literal["scalar"] = "scalar"
    min_value: float = -2.0
    max_value: float = 2.0
    # search
    num_mcts_simulations: int = 20
    # bootstrapping
    discount: float = 0.99
    bootstrap_n: int = 10
    lambda_gae: float = 0.95
    # optimization
    num_minibatches: int = 4
    batch_size: int = 12
    lr_init: float = 1e-3
    max_grad_norm: float = 1.0
    # target
    target_update_freq: int = 4
    target_update_size: float = 1.0
    # loss
    value_coef: float = 1.0
    # evaluation
    num_evals: int = 4
    num_eval_envs: int = 4


class ObsState(NamedTuple):
    obs: Float[Array, "obs_size"]
    is_initial: Bool[Array, ""]


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

        self.cell = eqx.nn.GRUCell(obs_size, rnn_size, key=key_cell)
        self.policy_head = eqx.nn.MLP(
            rnn_size, num_actions, mlp_size, mlp_depth, activation, key=key_policy
        )
        self.value_head = eqx.nn.MLP(
            rnn_size, num_value_bins, mlp_size, mlp_depth, activation, key=key_value
        )

    def __call__(
        self, hidden: Float[Array, "rnn_size"], obs_state: Batched[ObsState, "horizon"]
    ):
        return jax.lax.scan(
            lambda hidden, obs: self.step(hidden, obs), hidden, obs_state
        )

    def step(self, hidden: Float[Array, "rnn_size"], obs_state: ObsState):
        """Returns the predicted policy and value for this observation."""
        hidden = jnp.where(obs_state.is_initial, self.init_hidden(), hidden)
        hidden = self.cell(obs_state.obs, hidden)
        return hidden, Prediction(self.policy_head(hidden), self.value_head(hidden))

    def init_hidden(self):
        return jnp.zeros(self.cell.hidden_size)


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def exec_loop(init: Carry, length: int, key: Key[Array, ""]):
    """Scan the decorated function for `length` steps.

    The motivation is that loops are easier to read
    when the target and iter are in front.
    """

    def decorator(
        f: Callable[[Carry, X], tuple[Carry, Y]],
    ) -> tuple[Carry, Batched[Y, "length"]]:
        return jax.lax.scan(f, init, jr.split(key, length))

    return decorator


class Unobs(NamedTuple):
    """`hidden` is the representation of `env_state`"""

    env_state: gymnax.EnvState
    hidden: Float[Array, "rnn_size"]


class RolloutState(NamedTuple):
    """Carried when rolling out the environment."""

    obs_state: ObsState
    unobs: Unobs


class Transition(NamedTuple):
    """A single transition. May be batched into a trajectory."""

    rollout_state: RolloutState
    reward: Float[Array, ""]
    policy_logits: Float[Array, "num_actions"]


class ParamState(NamedTuple):
    """Carried during optimization."""

    params: ActorCriticRNN
    opt_state: optax.OptState


class IterState(NamedTuple):
    """Carried across algorithm iterations."""

    step: Integer[Array, ""]
    rollout_states: Batched[RolloutState, "num_envs"]
    param_state: ParamState
    target_params: ActorCriticRNN
    buffer_state: fbx.trajectory_buffer.TrajectoryBufferState[Transition]


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
        optax.clip_by_global_norm(config.max_grad_norm), optax.adamw(lr, eps=1e-5)
    )

    # sample config.batch_size trajectories of config.horizon transitions
    buffer_args = dict(
        add_batch_size=config.num_envs,
        sample_batch_size=config.batch_size,
        sample_sequence_length=config.horizon,
        period=config.horizon // 2,  # avoid some correlation
        min_length_time_axis=config.horizon * config.batch_size,
        max_length_time_axis=config.total_transitions // 2,  # keep recent transitions
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
        key_net, key_iter, key_evaluate = jr.split(key, 3)

        sys.stdout.write("Config\n" + "=" * 20 + "\n")
        yaml.safe_dump(config._asdict(), sys.stdout)
        sys.stdout.write("=" * 20 + "\n")
        sys.stdout.write("Buffer arguments\n" + "=" * 20 + "\n")
        yaml.safe_dump(buffer_args, sys.stdout)
        sys.stdout.write("=" * 20 + "\n")

        obs, env_state = env_reset_batch(jr.split(key, config.num_envs), env_params)

        # initialize all state
        init_net = ActorCriticRNN(
            obs.shape[-1],
            config.rnn_size,
            config.mlp_size,
            config.mlp_depth,
            num_actions,
            config.num_value_bins,
            config.activation,
            key=key_net,
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)
        init_rollout_state = RolloutState(
            obs_state=ObsState(obs=obs, is_initial=jnp.ones(config.num_envs, bool)),
            unobs=Unobs(
                env_state=env_state,
                # hidden reinitialized for initial states
                hidden=jnp.empty((config.num_envs, config.rnn_size)),
            ),
        )
        init_buffer_state = buffer.init(
            Transition(
                rollout_state=tree_slice(init_rollout_state, 0),
                reward=jnp.empty((), float),
                policy_logits=jnp.empty(num_actions),
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
            buffer_state = buffer.add(iter_state.buffer_state, trajectories)

            log_values(
                {
                    "train/reward_per_trajectory": jnp.sum(trajectories.reward)
                    / jnp.sum(trajectories.rollout_state.obs_state.is_initial),
                }
            )

            # gradient updates
            @exec_loop(
                iter_state.param_state,
                config.num_minibatches,
                key_optim,
            )
            def optimize_step(param_state: ParamState, key: Key[Array, ""]):
                trajectories = buffer.sample(buffer_state, key).experience

                @ft.partial(jax.value_and_grad, has_aux=True)
                def loss_grad(params: ActorCriticRNN):
                    """Average the loss across a batch of trjaectories."""
                    losses, aux = jax.vmap(loss_trajectory, (None, None, 0))(
                        eqx.combine(params, net_static),
                        eqx.combine(iter_state.target_params, net_static),
                        trajectories,
                    )

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

            param_state, _ = optimize_step

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
            jax.lax.cond(
                iter_state.step % eval_freq == 0,
                ft.partial(
                    evaluate, num_envs=config.num_eval_envs, net_static=net_static
                ),
                lambda params, key: jnp.nan,
                param_state.params,
                key,
            )

            return IterState(
                step=iter_state.step + 1,
                rollout_states=rollout_states,
                param_state=param_state,
                target_params=target_params,
                buffer_state=buffer_state,
            ), None

        final_iter_state, _ = iterate

        final_mean_eval_reward = evaluate(
            final_iter_state.param_state.params,
            key_evaluate,
            num_envs=config.num_eval_envs,
            net_static=net_static,
        )
        return final_iter_state, final_mean_eval_reward

    def rollout(
        net: ActorCriticRNN,
        init_rollout_state: Batched[RolloutState, "num_envs"],
        key: Key[Array, ""],
    ) -> tuple[
        Batched[RolloutState, "num_envs"], Batched[Transition, "num_envs horizon"]
    ]:
        @exec_loop(init_rollout_state, config.horizon, key)
        def rollout_step(
            rollout_state: Batched[RolloutState, "num_envs"], key: Key[Array, ""]
        ):
            key_action, key_step = jr.split(key)

            hiddens, preds = jax.vmap(net.step)(
                rollout_state.unobs.hidden, rollout_state.obs_state
            )

            root = mctx.RootFnOutput(
                prior_logits=preds.policy_logits,
                value=logits_to_values(preds.value_logits),
                # embedding contains current env_state and the hidden to pass to next net call
                embedding=rollout_state.unobs._replace(hidden=hiddens),
            )

            def mcts_recurrent_fn(
                _: None,
                key: Key[Array, ""],
                action: Integer[Array, "num_envs"],
                unobs: Batched[Unobs, "num_envs"],
            ):
                """Returns the logits and value for the newly created node."""
                obs, env_states, rewards, is_initials, infos = env_step_batch(
                    jr.split(key, config.num_envs), unobs.env_state, action, env_params
                )
                hiddens, preds = jax.vmap(net.step)(
                    unobs.hidden, ObsState(obs, is_initials)
                )
                return mctx.RecurrentFnOutput(
                    reward=rewards,
                    discount=jnp.where(is_initials, 0.0, config.discount),
                    prior_logits=preds.policy_logits,
                    value=logits_to_values(preds.value_logits),
                ), Unobs(env_states, hiddens)

            out = mctx.muzero_policy(
                params=None,
                rng_key=key_action,
                root=root,
                recurrent_fn=mcts_recurrent_fn,
                num_simulations=config.num_mcts_simulations,
                max_depth=config.horizon,
            )

            obs, env_state, reward, is_initial, info = env_step_batch(
                jr.split(key_step, config.num_envs),
                rollout_state.unobs.env_state,
                out.action,
                env_params,
            )

            return RolloutState(
                obs_state=ObsState(obs=obs, is_initial=is_initial),
                unobs=Unobs(env_state=env_state, hidden=hiddens),
            ), Transition(
                # contains the reward and policy logits computed from the rollout state
                rollout_state=rollout_state,
                reward=reward,
                policy_logits=out.action_weights,
            )

        final_rollout_state, transitions = rollout_step
        return final_rollout_state, jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )

    def loss_trajectory(
        net: ActorCriticRNN,
        target_net: ActorCriticRNN,
        trajectory: Batched[Transition, "horizon"],
    ):
        """Compute behavior cloning loss for the `trajectory`. Bootstrap values from `target_net`."""
        init_hidden = trajectory.rollout_state.unobs.hidden[0, :]
        obs_seq = trajectory.rollout_state.obs_state

        _, net_pred = net(init_hidden, obs_seq)

        # policy loss
        target_policy_probs = jax.nn.softmax(trajectory.policy_logits, axis=-1)
        policy_losses = rlax.categorical_cross_entropy(
            target_policy_probs, net_pred.policy_logits
        )
        policy_loss = jnp.mean(policy_losses)

        # value loss
        # bootstrap from target network
        _, target_pred = target_net(init_hidden, obs_seq)
        target_values = logits_to_values(target_pred.value_logits)
        bootstrapped_values = rlax.n_step_bootstrapped_returns(
            trajectory.reward[:-1],
            jnp.where(obs_seq.is_initial[1:], 0.0, config.discount),
            target_values[1:],
            config.bootstrap_n,
            config.lambda_gae,
        )
        value_losses = (
            rlax.l2_loss(net_pred.value_logits[:-1], bootstrapped_values)
            if config.num_value_bins == "scalar"
            else rlax.categorical_cross_entropy(
                rlax.transform_to_2hot(
                    bootstrapped_values[:, jnp.newaxis],
                    config.min_value,
                    config.max_value,
                    config.num_value_bins,
                ),
                net_pred.value_logits[:-1, :],
            )
        )
        value_loss = jnp.mean(value_losses)

        # logging
        td_error = bootstrapped_values - logits_to_values(net_pred.value_logits)[:-1]
        policy_entropy = distrax.Categorical(logits=net_pred.policy_logits).entropy()

        loss = policy_loss + config.value_coef * value_loss
        return loss, {
            "train/loss": loss,
            "train/policy_entropy": policy_entropy,
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
            "train/td_error": td_error,
        }

    def evaluate(
        params: ActorCriticRNN,
        key: Key[Array, ""],
        *,
        num_envs: int,
        net_static: ActorCriticRNN,
    ) -> Float[Array, ""]:
        """Evaluate a batch of rollouts using the raw policy.

        Returns mean return across the batch.
        """
        key_reset, key_rollout = jr.split(key)

        net = eqx.combine(params, net_static)

        obs, env_state = env_reset_batch(jr.split(key_reset, num_envs), env_params)

        init_rollout_state = RolloutState(
            ObsState(obs, jnp.zeros(num_envs, dtype=bool)),
            Unobs(
                env_state,
                jnp.broadcast_to(net.init_hidden(), (num_envs, config.rnn_size)),
            ),
        )

        @exec_loop(
            init_rollout_state,
            config.horizon,
            key_rollout,
        )
        def rollout_step(
            rollout_state: Batched[RolloutState, "num_envs"], key: Key[Array, ""]
        ):
            key_action, key_step = jr.split(key)

            @ft.partial(jax.value_and_grad, has_aux=True)
            def predict(obs: Float[Array, "obs_size"], rollout_state: RolloutState):
                """Differentiate with respect to value to obtain saliency map."""
                hidden, pred = net.step(
                    rollout_state.unobs.hidden,
                    rollout_state.obs_state._replace(obs=obs),
                )
                value = logits_to_values(pred.value_logits[jnp.newaxis])[0]
                return value, (hidden, pred.policy_logits)

            (values, (hidden, policy_logits)), obs_grads = jax.vmap(predict)(
                rollout_state.obs_state.obs, rollout_state
            )
            action = jr.categorical(key_action, policy_logits, axis=-1)
            obs, env_states, rewards, is_initials, info = env_step_batch(
                jr.split(key_step, num_envs),
                rollout_state.unobs.env_state,
                action,
                env_params,
            )

            return RolloutState(
                ObsState(obs, is_initials), Unobs(env_states, hidden)
            ), (
                rollout_state,
                rewards,
                policy_logits,
                values,
                obs_grads,
            )

        _, (rollout_states, rewards, policy_logits, values, grads) = rollout_step

        bootstrapped_returns = jax.vmap(rlax.lambda_returns, (1, 1, 1, None), 1)(
            rewards[:-1, :],
            jnp.where(rollout_states.obs_state.is_initial[1:, :], 0.0, config.discount),
            values[1:, :],
            config.lambda_gae,
        )
        td_errors = bootstrapped_returns - values[:-1, :]

        @ft.partial(
            jax.debug.callback,
            video=visualize_catch(
                obs_shape=obs_shape,
                env_states=rollout_states.unobs.env_state,
                maps=grads,
            ),
            rewards=rewards,
            policy_logits=policy_logits,
            values=values,
            td_errors=td_errors,
        )
        def visualize(
            video: Float[Array, "num_envs horizon channel height width"],
            rewards: Float[Array, "horizon num_envs"],
            policy_logits: Float[Array, "horizon num_envs num_actions"],
            values: Float[Array, "horizon num_envs"],
            td_errors: Float[Array, "horizon-1 num_envs"],
        ):
            if not wandb.run:
                return

            ncols = 4
            nrows = num_envs // ncols
            horizon = list(range(config.horizon))

            fig, ax = plt.subplots(nrows=nrows * 2, ncols=ncols)
            for i in range(num_envs):
                r, c = divmod(i, ncols)
                axes: Axes = ax[2 * r, c]
                axes.set_title(f"Trajectory {i}")
                axes.plot(horizon, rewards[:, i], "ro", label="reward")
                axes.plot(horizon, values[:, i], "bo", label="value")
                axes.plot(horizon[:-1], td_errors[:, i], "go", label="TD error")
                axes.plot(
                    horizon,
                    distrax.Categorical(logits=policy_logits[:, i]).entropy(),
                )
                axes.legend()

                policy_axes: Axes = ax[2 * r + 1, c]
                for action in range(num_actions):
                    policy_axes.plot(
                        horizon, policy_logits[:, i, action], label=rf"$\pi({action})$"
                    )
                policy_axes.legend()

            wandb.log(
                {
                    "eval/video": wandb.Video(np.asarray(video), fps=10),
                    "eval/statistics": fig,
                }
            )

            plt.close(fig)

        num_trajectories = jnp.sum(rollout_states.obs_state.is_initial)
        mean_return = jnp.sum(rewards) / num_trajectories
        return mean_return

    return train


if __name__ == "__main__":
    default_config = Config()

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
                "metric": {"goal": "maximize", "name": "eval/final/mean_reward"},
                "parameters": config_values,
            }

            def run():
                with wandb.init(project=WANDB_PROJECT):
                    train = make_train(wandb.config)
                    _, final_reward = jax.jit(train)(jr.key(42))
                    wandb.log({"eval/final/mean_reward": final_reward})

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
                output = train(jr.key(42))
            jax.block_until_ready(output)
    else:
        train = make_train(default_config)
        with wandb.init(project=WANDB_PROJECT, config=default_config._asdict()):
            output = jax.jit(train)(jr.key(42))
            # with jax.profiler.trace(f"/tmp/{WANDB_PROJECT}-trace", create_perfetto_link=True):
            _, mean_eval_reward = jax.block_until_ready(output)
            sys.stdout.write(f"{mean_eval_reward=}\n")

    sys.stdout.write("Done training\n")
