"""Implementation of AlphaZero with a recurrent actor-critic network.

1. Config
2. Data structures for loop carries
3. Model definitions
4. Training loop
5. Rollout function
6. Evaluation function
"""

import dataclasses as dc
import functools as ft
import sys
from pathlib import Path
from typing import Annotated as Batched
from typing import Literal, NamedTuple

import chex
import distrax
import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import mctx
import optax
import rlax
import yaml
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Integer, Key, jaxtyped

from envs.multi_catch import visualize_catch
from envs.translate import make_env
from cleanrl_utils.config import ArchConfig, TrainConfig, main
from cleanrl_utils.jax_utils import tree_slice
from cleanrl_utils.log_utils import exec_loop, get_norm_data, log_values
from utils.prioritized_buffer import BufferState, PrioritizedBuffer

matplotlib.use("agg")  # enable plotting inside jax callback


class Prediction(NamedTuple):
    policy_logits: Float[Array, " num_actions"]
    value_logits: Float[Array, " num_value_bins"]


class ActorCriticRNN(eqx.Module):
    cell: eqx.nn.GRUCell
    policy_head: eqx.nn.MLP
    value_head: eqx.nn.MLP

    def __init__(
        self,
        arch: ArchConfig,
        obs_size: int,
        num_actions: int,
        num_value_bins: int | Literal["scalar"],
        *,
        key: Key[Array, ""],
    ):
        key_cell, key_policy, key_value = jr.split(key, 3)

        activation = jax.nn.relu if arch.activation == "relu" else jax.nn.tanh

        self.cell = eqx.nn.GRUCell(obs_size, arch.dyn_size, key=key_cell) if arch.dyn_size > 0 else None

        in_size = arch.dyn_size if self.is_rnn else obs_size

        self.policy_head = eqx.nn.MLP(
            in_size,
            num_actions,
            arch.mlp_size,
            arch.mlp_depth,
            activation,
            key=key_policy,
        )
        self.value_head = eqx.nn.MLP(
            in_size,
            num_value_bins,
            arch.mlp_size,
            arch.mlp_depth,
            activation,
            key=key_value,
        )

    def __call__(
        self,
        hidden: Float[Array, " rnn_size"],
        obs: Float[Array, " horizon obs_size"],
        is_initial: Bool[Array, " horizon"],
    ):
        """Predict across a sequence of observations."""
        return jax.lax.scan(
            lambda hidden, x: self.step(hidden, *x),
            hidden,
            (obs, is_initial),
        )

    def step(
        self,
        hidden: Float[Array, " rnn_size"],
        obs: Float[Array, " obs_size"],
        is_initial: Bool[Array, ""],
    ):
        """Returns the predicted policy and value for this observation."""
        if self.is_rnn:
            hidden = jnp.where(is_initial, self.init_hidden(), hidden)
            hidden = input = self.cell(obs, hidden)
        else:
            hidden, input = jnp.empty(0), obs
        pred = Prediction(
            policy_logits=self.policy_head(input),
            value_logits=self.value_head(input),
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
    hidden: Float[Array, " rnn_size"]
    initial: Bool[Array, ""]


class RolloutState(NamedTuple):
    """Carried when rolling out the environment."""

    obs: Float[Array, " obs_size"]
    unobs: Unobs

    def as_inputs(self):
        """Convenience for passing to network step."""
        return self.unobs.hidden, self.obs, self.unobs.initial


class Transition(NamedTuple):
    """A single transition. May be batched into a trajectory."""

    rollout_state: RolloutState
    action: Integer[Array, ""]
    reward: Float[Array, ""]
    pred: Prediction
    mcts_probs: Float[Array, " num_actions"]


class ParamState(NamedTuple):
    """Carried during optimization."""

    params: ActorCriticRNN
    buffer_state: BufferState[Transition]
    opt_state: optax.OptState


class IterState(NamedTuple):
    """Carried across algorithm iterations."""

    step: Integer[Array, ""]
    rollout_states: Batched[RolloutState, " num_envs"]
    param_state: ParamState
    target_params: ActorCriticRNN


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


@jaxtyped(typechecker=typechecker)
def make_train(config: TrainConfig):
    num_iters = config.mcts.total_transitions // (config.mcts.num_envs * config.env.horizon)
    num_updates = num_iters * config.optim.num_updates_per_iter
    eval_freq = num_iters // config.eval.eval_freq

    env, env_params = make_env(config.env, goal=False)
    env_params = env_params.replace(max_steps_in_episode=config.mcts.total_transitions)  # don't truncate
    num_actions = env.action_space(env_params).n
    # env: gymnax.environments.environment.Environment = FlattenObservationWrapper(env)

    env_reset_batch = jax.vmap(env.reset, in_axes=(None))  # map over key
    env_step_batch = jax.vmap(env.step, in_axes=(0, 0, None))  # map over key, state, action

    lr = optax.cosine_decay_schedule(config.optim.lr_init, num_updates)
    optim = optax.chain(
        optax.clip_by_global_norm(config.optim.max_grad_norm),
        optax.adamw(lr),
    )

    buffer = PrioritizedBuffer.new(
        batch_size=config.mcts.num_envs,
        max_length=num_iters * config.env.horizon,
        sample_length=config.env.horizon,
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

    def value_to_probs(
        value: Float[Array, " horizon"],
    ) -> Float[Array, " horizon num_value_bins"]:
        """Convert from scalar values to logits for two-hot encoding."""
        if config.value.num_value_bins == "scalar":
            return value
        else:
            return rlax.transform_to_2hot(
                value,
                config.value.min_value,
                config.value.max_value,
                config.value.num_value_bins,
            )

    def get_invalid_actions(
        env_state: gymnax.EnvState,
    ) -> Bool[Array, " num_actions"]:
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
        key_reset, key_net, key_iter, key_evaluate = jr.split(key, 4)

        sys.stdout.write("Config\n" + "=" * 20 + "\n")
        yaml.safe_dump(dc.asdict(config), sys.stdout)
        sys.stdout.write("=" * 20 + "\n")

        init_obs, init_env_states = env_reset_batch(
            env_params,
            key=jr.split(key_reset, config.mcts.num_envs),
        )

        # initialize all state
        init_net = ActorCriticRNN(
            arch=config.arch,
            obs_size=init_obs.shape[-1],
            num_actions=num_actions,
            num_value_bins=config.value.num_value_bins,
            key=key_net,
        )
        init_params, net_static = eqx.partition(init_net, eqx.is_inexact_array)
        init_hiddens = jnp.broadcast_to(init_net.init_hidden(), (config.mcts.num_envs, config.arch.dyn_size))
        init_rollout_states = RolloutState(
            obs=init_obs,
            unobs=Unobs(
                env_state=init_env_states,
                hidden=init_hiddens,
                initial=jnp.ones(config.mcts.num_envs, jnp.bool),
            ),
        )
        init_buffer_state = buffer.init(
            Transition(
                rollout_state=tree_slice(init_rollout_states, 0),
                reward=jnp.empty((), init_hiddens.dtype),
                action=env.action_space(env_params).sample(jr.key(0)),
                pred=init_net.step(*tree_slice(init_rollout_states.as_inputs(), 0))[1],
                mcts_probs=jnp.empty(num_actions, init_hiddens.dtype),
            )
        )

        @exec_loop(num_iters)
        def iterate(
            iter_state=IterState(
                step=jnp.int_(0),
                rollout_states=init_rollout_states,
                param_state=ParamState(
                    params=init_params,
                    opt_state=optim.init(init_params),
                    buffer_state=init_buffer_state,
                ),
                target_params=init_params,
            ),
            key=key_iter,
        ):
            key_rollout, key_optim = jr.split(key)

            # collect data
            rollout_states, trajectories = rollout(
                eqx.combine(iter_state.param_state.params, net_static),
                iter_state.rollout_states,
                key_rollout,
            )
            buffer_state = buffer.add(iter_state.param_state.buffer_state, trajectories)

            mean_return = jnp.sum(trajectories.reward) / jnp.sum(trajectories.rollout_state.unobs.initial)
            log_values({"iter/step": iter_state.step, "iter/mean_return": mean_return})

            buffer_available = buffer.num_available(buffer_state) >= config.optim.batch_size

            @exec_loop(
                config.optim.num_updates_per_iter,
                cond=buffer_available,
            )
            def optimize_step(
                param_state=iter_state.param_state._replace(buffer_state=buffer_state),
                key=key_optim,
            ):
                batch = jax.vmap(buffer.sample, (None, None))(
                    param_state.buffer_state,
                    config.eval.warnings,
                    key=jr.split(key, config.optim.batch_size),
                )
                trajectories: Batched[Transition, " batch_size horizon"] = batch.experience

                @ft.partial(jax.value_and_grad, has_aux=True)
                def loss_grad(params: ActorCriticRNN):
                    """Average the loss across a batch of trjaectories."""
                    losses, aux = loss_trajectory(
                        eqx.combine(params, net_static),
                        eqx.combine(iter_state.target_params, net_static),
                        trajectories,
                    )
                    chex.assert_shape(losses, (config.optim.batch_size,))
                    # TODO add importance reweighting
                    return jnp.mean(losses), aux

                (_, aux), grads = loss_grad(param_state.params)
                # value_and_grad drops types
                aux: LossStatistics
                grads: ActorCriticRNN
                updates, opt_state = optim.update(grads, param_state.opt_state, param_state.params)
                params = optax.apply_updates(param_state.params, updates)

                # prioritize trajectories with high TD error
                buffer_state = buffer.set_priorities(
                    param_state.buffer_state,
                    batch.idx,
                    jnp.mean(jnp.abs(aux.td_error), axis=-1),
                )

                log_values(
                    {f"train/mean_{key}": jnp.mean(value) for key, value in aux._asdict().items()}
                    | get_norm_data(updates, "updates/norm")
                    | get_norm_data(params, "params/norm")
                )

                return (
                    ParamState(
                        params=params,
                        opt_state=opt_state,
                        buffer_state=buffer_state,
                    ),
                    None,
                )

            param_state, _ = optimize_step

            target_params = jax.tree.map(
                lambda new, old: jnp.where(
                    iter_state.step % config.bootstrap.target_update_freq == 0,
                    optax.incremental_update(new, old, config.bootstrap.target_update_size),
                    old,
                ),
                param_state.params,
                iter_state.target_params,
            )

            # evaluate every eval_freq steps
            eval_return = jax.lax.cond(
                (iter_state.step % eval_freq == 0) & buffer_available,
                ft.partial(
                    evaluate,
                    num_envs=config.eval.num_eval_envs,
                    net_static=net_static,
                ),
                lambda *_: jnp.nan,
                # cond only accepts positional arguments
                param_state.params,
                target_params,
                buffer_state,
                key_evaluate,
            )

            jax.debug.print(
                "step {}/{}. seen {}/{} transitions. mean return {}",
                iter_state.step,
                num_iters,
                iter_state.step * config.mcts.num_envs * config.env.horizon,
                config.mcts.total_transitions,
                mean_return,
            )

            return (
                IterState(
                    step=iter_state.step + 1,
                    rollout_states=rollout_states,
                    param_state=param_state,
                    target_params=target_params,
                ),
                eval_return,
            )

        final_iter_state, eval_returns = iterate
        return final_iter_state, eval_returns

    def rollout(
        net: ActorCriticRNN,
        init_rollout_state: Batched[RolloutState, " num_envs"],
        key: Key[Array, ""],
    ) -> tuple[Batched[RolloutState, " num_envs"], Batched[Transition, " num_envs horizon"]]:
        @exec_loop(config.env.horizon)
        def rollout_step(rollout_states=init_rollout_state, key=key):
            key_action, key_step = jr.split(key)

            hiddens, preds, out = act_mcts(net, rollout_states, key_action)
            timesteps = env_step_batch(
                rollout_states.unobs.env_state,
                out.action,
                env_params,
                key=jr.split(key_step, out.action.size),
            )
            return RolloutState(
                obs=timesteps.obs,
                unobs=Unobs(
                    env_state=timesteps.state,
                    hidden=hiddens,
                    initial=timesteps.is_last,
                ),
            ), Transition(
                # contains the reward and network predictions computed from the rollout state
                rollout_state=rollout_states,
                action=out.action,
                reward=timesteps.reward,
                pred=preds,
                mcts_probs=out.action_weights,
            )

        final_rollout_state, transitions = rollout_step
        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)  # swap horizon and batch axes
        return final_rollout_state, transitions

    def act_mcts(
        net: ActorCriticRNN,
        rollout_state: Batched[RolloutState, " num_envs"],
        key: Key[Array, ""],
    ):
        hiddens, preds = jax.vmap(net.step)(*rollout_state.as_inputs())

        root = mctx.RootFnOutput(
            prior_logits=preds.policy_logits,
            value=logits_to_value(preds.value_logits),
            # embedding contains current env_state and the hidden to pass to next net call
            embedding=rollout_state.unobs._replace(hidden=hiddens),
        )

        invalid_actions = jax.vmap(get_invalid_actions)(rollout_state.unobs.env_state)

        def mcts_recurrent_fn(
            _: None,  # closure net
            key: Key[Array, ""],
            action: Integer[Array, " num_envs"],
            unobs: Batched[Unobs, " num_envs"],
        ):
            """Returns the logits and value for the newly created node."""
            obs, env_states, rewards, is_initials, infos = env_step_batch(
                unobs.env_state,
                action,
                env_params,
                key=jr.split(key, action.size),
            )
            hiddens, preds = jax.vmap(net.step)(unobs.hidden, obs, is_initials)
            return mctx.RecurrentFnOutput(
                reward=rewards,
                # careful for off-by-one
                # if new state is initial,
                # preds.value_logits is for a new trajectory
                discount=jnp.where(is_initials, 0.0, config.bootstrap.discount),
                prior_logits=preds.policy_logits,
                value=logits_to_value(preds.value_logits),
            ), Unobs(env_state=env_states, hidden=hiddens, initial=is_initials)

        out = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=mcts_recurrent_fn,
            num_simulations=config.mcts.num_mcts_simulations,
            invalid_actions=invalid_actions,
            max_depth=config.env.horizon,
        )

        return hiddens, preds, out

    def bootstrap(net: ActorCriticRNN, trajectory: Batched[Transition, " horizon"]):
        """Bootstrap values from the target network."""
        unobs = trajectory.rollout_state.unobs
        _, pred = net(unobs.hidden[0, :], trajectory.rollout_state.obs, unobs.initial)
        value = logits_to_value(pred.value_logits)
        # bootstrapped_returns = rlax.n_step_bootstrapped_returns(
        return_ = rlax.lambda_returns(
            # careful for off-by-one
            # reward[0] is the reward obtained from acting in rollout_state[0]
            # is_initials[1] iff rollout_state[1] is part of new trajectory
            trajectory.reward[:-1],
            jnp.where(unobs.initial[1:], 0.0, config.bootstrap.discount),
            value[1:],
            # config.bootstrap_n,
            config.bootstrap.lambda_gae,
        )
        return pred, return_

    @ft.partial(jax.vmap, in_axes=(None, None, 0))
    def loss_trajectory(
        net: ActorCriticRNN,
        target_net: ActorCriticRNN,
        trajectory: Batched[Transition, " horizon"],
    ):
        """Compute behavior cloning loss for the `trajectory`. Bootstrap values from `target_net`."""
        init_hidden = trajectory.rollout_state.unobs.hidden[0, :]
        obs = trajectory.rollout_state.obs
        initial = trajectory.rollout_state.unobs.initial

        _, online_pred = net(init_hidden, obs, initial)
        chex.assert_shape(
            [trajectory.mcts_probs, online_pred.policy_logits],
            (config.env.horizon, num_actions),
        )

        # policy loss
        mcts_dist = distrax.Categorical(probs=trajectory.mcts_probs)
        online_policy_dist = distrax.Categorical(logits=online_pred.policy_logits)
        policy_loss = mcts_dist.kl_divergence(online_policy_dist)

        # value loss
        # bootstrap from target network
        target_pred, bootstrapped_return = bootstrap(target_net, trajectory)
        online_values = logits_to_value(online_pred.value_logits)
        if config.value.num_value_bins == "scalar":
            value_loss = rlax.l2_loss(online_values[:-1], bootstrapped_return)
        else:
            online_value_dist = distrax.Categorical(logits=online_pred.value_logits)
            bootstrapped_probs = value_to_probs(bootstrapped_return)
            bootstrapped_dist = distrax.Categorical(probs=bootstrapped_probs)
            value_loss = bootstrapped_dist.kl_divergence(online_value_dist[:-1])

        if False:  # debugging
            jax.debug.callback(
                visualize_callback,
                trajectories=jax.tree.map(lambda x: x[jnp.newaxis], trajectory._replace(pred=online_pred)),
                bootstrapped_returns=bootstrapped_return[jnp.newaxis],
                video=visualize_catch(
                    trajectory.rollout_state.unobs.env_state,
                ),
                prefix="voir",
                target_value_logits=target_pred.value_logits[jnp.newaxis],
            )

        loss = jnp.mean(policy_loss) + config.optim.value_coef * jnp.mean(value_loss)
        return loss, LossStatistics(
            loss=loss,
            # policy
            mcts_entropy=mcts_dist.entropy(),
            policy_entropy=online_policy_dist.entropy(),
            policy_logits=online_pred.policy_logits,
            policy_loss=policy_loss,
            # value
            value_logits=online_pred.value_logits,
            target_value_logits=target_pred.value_logits,
            bootstrapped_return=bootstrapped_return,
            value_loss=value_loss,
            td_error=bootstrapped_return - online_values[:-1],
        )

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

        obs, env_state = env_reset_batch(env_params, key=jr.split(key_reset, num_envs))

        init_rollout_states = RolloutState(
            obs=obs,
            unobs=Unobs(
                env_state=env_state,
                hidden=jnp.broadcast_to(net.init_hidden(), (num_envs, config.arch.dyn_size)),
                initial=jnp.zeros(num_envs, dtype=bool),
            ),
        )

        @exec_loop(
            init_rollout_states,
            config.env.horizon,
            key_rollout,
        )
        def rollout_step(rollout_states: Batched[RolloutState, " num_envs"], key: Key[Array, ""]):
            key_action, key_step, key_mcts = jr.split(key, 3)

            @ft.partial(jax.value_and_grad, has_aux=True)
            def predict(obs: Float[Array, " obs_size"], unobs: Unobs):
                """Differentiate with respect to value to obtain saliency map."""
                hidden, pred = net.step(
                    unobs.hidden,
                    obs,
                    unobs.initial,
                )

                value = logits_to_value(pred.value_logits[jnp.newaxis])[0]
                return value, (hidden, pred.policy_logits)

            (values, (hiddens, policy_logits)), obs_grads = jax.vmap(predict)(rollout_states.obs, rollout_states.unobs)
            actions = jr.categorical(key_action, logits=policy_logits, axis=-1)
            timesteps = env_step_batch(
                rollout_states.unobs.env_state,
                actions,
                env_params,
                key=jr.split(key_step, num_envs),
            )

            _, preds, mcts_outs = act_mcts(net, rollout_states, key_mcts)

            return RolloutState(
                obs=timesteps.obs,
                unobs=Unobs(
                    env_state=timesteps.state,
                    hidden=hiddens,
                    initial=timesteps.is_last,
                ),
            ), (
                Transition(
                    rollout_state=rollout_states,
                    action=actions,
                    reward=timesteps.reward,
                    pred=preds,
                    mcts_probs=mcts_outs.action_weights,
                ),
                obs_grads,
            )

        _, (trajectories, obs_grads) = rollout_step
        trajectories, obs_grads = jax.tree.map(lambda x: x.swapaxes(0, 1), (trajectories, obs_grads))
        trajectories: Batched[Transition, " num_envs horizon"]

        target_net = eqx.combine(target_params, net_static)
        _, bootstrapped_returns = jax.vmap(bootstrap, in_axes=(None, 0))(target_net, trajectories)

        jax.debug.callback(
            visualize_callback,
            trajectories,
            bootstrapped_returns,
            jax.vmap(visualize_catch, in_axes=(0, 0))(
                trajectories.rollout_state.unobs.env_state,
                obs_grads,
            ),
            prefix="eval",
        )

        eval_return = jnp.sum(trajectories.reward) / jnp.sum(trajectories.rollout_state.unobs.initial)
        log_values({"eval/mean_return": eval_return})

        # debug training procedure
        if False:
            batch = buffer.sample(buffer_state, key_sample)
            sampled_trajectories: Batched[Transition, " num_envs horizon"] = tree_slice(batch.experience, slice(0, num_envs))
            _, aux = loss_trajectory(net, target_net, sampled_trajectories)

            jax.debug.callback(
                visualize_callback,
                trajectories=sampled_trajectories._replace(
                    pred=Prediction(
                        value_logits=aux.value_logits,
                        policy_logits=aux.policy_logits,
                    )
                ),
                bootstrapped_returns=aux.bootstrapped_return,
                video=jax.vmap(visualize_catch, in_axes=(0,))(
                    sampled_trajectories.rollout_state.unobs.env_state,
                ),
                prefix="visualize",
                priorities=batch.priorities,
                # target_value_logits=aux["train/target_value_logits"],
            )

        return eval_return

    return train


if __name__ == "__main__":
    main(TrainConfig, make_train, Path(__file__).name)
