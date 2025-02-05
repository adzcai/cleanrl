from functools import partial
from typing import NamedTuple
from jaxtyping import Bool, Integer, Float, Array


import jax
import jax.numpy as jnp
import jax.random as rand


import tempfile
import wandb
import matplotlib
import matplotlib.pyplot as plt


import equinox as eqx
import optax
import rlax
import gymnax
from gymnax.environments.environment import Environment, EnvState
from gymnax.wrappers import FlattenObservationWrapper
from gymnax.visualize import Visualizer

matplotlib.use("agg")


class ObsWithDone(NamedTuple):
    obs: Float[Array, "*batch obs_size"]
    env_state: EnvState
    done: Bool[Array, "*batch"]


class ActorCriticRNN(eqx.Module):
    """Parameterizes the actor-critic network."""

    cell: eqx.nn.GRUCell
    policy_head: eqx.nn.MLP
    value_head: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        rnn_size: int,
        num_actions: int,
        mlp_size: int,
        mlp_depth: int,
        *,
        key: rand.PRNGKey,
    ):
        key_policy, key_value, key_cell = rand.split(key, 3)
        self.cell = eqx.nn.GRUCell(in_size, rnn_size, key=key_cell)
        self.policy_head = eqx.nn.MLP(
            rnn_size, num_actions, mlp_size, mlp_depth, key=key_policy
        )
        self.value_head = eqx.nn.MLP(
            rnn_size, "scalar", mlp_size, mlp_depth, key=key_value
        )

    def __call__(self, hidden: Float[Array, "rnn_size"], inputs: ObsWithDone):
        hidden, outputs = jax.lax.scan(
            lambda carry, x: self.step(carry, x),
            hidden,
            inputs,
        )
        return hidden, outputs

    def step(self, hidden: Float[Array, "rnn_size"], input: ObsWithDone):
        """Reset to initial hidden state if done."""
        hidden = jnp.where(input.done, self.init_hidden(), hidden)
        hidden = self.cell(input.obs, hidden)
        return hidden, (
            self.policy_head(hidden),
            self.value_head(hidden),
        )

    def init_hidden(self):
        return jnp.zeros(self.cell.hidden_size)


class Transition(NamedTuple):
    """A single transition. May be batched into a trajectory."""

    obs: Float[Array, "*batch obs_size"]
    env_state: EnvState
    action: Integer[Array, "*batch"]
    reward: Float[Array, "*batch"]
    logits: Float[Array, "*batch num_actions"]
    value: Float[Array, "*batch"]
    done: Bool[Array, "*batch"]


class RolloutState(NamedTuple):
    """Carried when rolling out the environment."""

    obs_with_done: ObsWithDone
    hidden: Float[Array, "rnn_size"]


class ParamState(NamedTuple):
    """Each outer update iteration."""

    params: ActorCriticRNN
    opt_state: optax.OptState


class UpdateState(NamedTuple):
    param_state: ParamState
    rollout_state: RolloutState


class Minibatch(NamedTuple):
    hidden: Float[Array, "batch rnn_size"]
    trajectories: Transition
    advantages: Float[Array, "batch horizon"]


class Config(NamedTuple):
    env: str = "Catch-bsuite"

    # environment collecting
    num_total_transitions: int = 50_000
    max_horizon: int = 120
    num_parallel_envs: int = 16

    # network architecture
    rnn_size: int = 128
    mlp_size: int = 64
    mlp_depth: int = 2

    # optimization
    num_minibatches: int = 4
    num_epochs: int = 2
    max_gradient_norm: float = 0.5
    learning_rate: float = 1e-4

    # loss function
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001


def make_train(config: Config):
    num_transitions_per_iteration = config.max_horizon * config.num_parallel_envs
    num_iterations = config.num_total_transitions // num_transitions_per_iteration
    num_gradient_steps = num_iterations * config.num_epochs * config.num_minibatches

    lr = optax.linear_schedule(
        config.learning_rate,
        0.0,
        num_gradient_steps,
    )

    env, env_params = gymnax.make(config.env)
    env: Environment = FlattenObservationWrapper(env)
    # env: Environment = LogWrapper(env) # doesn't respect the envstate

    num_actions = env.action_space(env_params).n

    optim = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm), optax.adamw(lr)
    )

    def train(key):
        key_reset, key_network = rand.split(key)

        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            rand.split(key_reset, config.num_parallel_envs), env_params
        )

        network = ActorCriticRNN(
            in_size=obs.shape[-1],
            rnn_size=config.rnn_size,
            num_actions=num_actions,
            mlp_size=config.mlp_size,
            mlp_depth=config.mlp_depth,
            key=key_network,
        )

        params, network_static = eqx.partition(network, eqx.is_inexact_array)

        hidden = network.init_hidden()
        rollout_state = RolloutState(
            ObsWithDone(
                obs,
                env_state,
                jnp.zeros(config.num_parallel_envs, dtype=bool),
            ),
            jnp.broadcast_to(hidden, (config.num_parallel_envs, *hidden.shape)),
        )

        # collect rollouts
        def rollout(
            params: ActorCriticRNN, init_rollout_state: RolloutState, key: rand.PRNGKey
        ):
            model = eqx.combine(params, network_static)

            @partial(
                jax.lax.scan,
                init=init_rollout_state,
                xs=rand.split(key, config.max_horizon),
            )
            def rollout_step(rollout_state: RolloutState, key: rand.PRNGKey):
                """Rollout in a single environment."""
                key_action, key_step = rand.split(key)

                # choose action
                hidden, (logits, value) = model.step(
                    rollout_state.hidden, rollout_state.obs_with_done
                )
                action = rand.categorical(key_action, logits)

                # step environment
                obs, env_state, reward, done, info = env.step(
                    key_step, rollout_state.obs_with_done.env_state, action, env_params
                )

                rollout_state = RolloutState(
                    ObsWithDone(obs, env_state, done),
                    hidden,
                )
                transition = Transition(
                    obs=rollout_state.obs_with_done.obs,
                    env_state=rollout_state.obs_with_done.env_state,
                    action=action,
                    reward=reward,
                    logits=logits,
                    value=value,
                    done=rollout_state.obs_with_done.done,
                )
                return rollout_state, transition

            rollout_state, trajectory = rollout_step

            # rlax convention
            done_offset = jnp.append(
                trajectory.done[1:], rollout_state.obs_with_done.done
            )
            _, (_, value_offset) = model.step(
                rollout_state.hidden, rollout_state.obs_with_done
            )
            value_offset = jnp.append(
                trajectory.value,
                value_offset,
            )
            advantages = rlax.truncated_generalized_advantage_estimation(
                trajectory.reward,
                jnp.where(done_offset, 0.0, config.discount),
                config.gae_lambda,
                value_offset,
            )

            return rollout_state, trajectory, advantages

        @partial(
            jax.lax.scan,
            init=UpdateState(
                ParamState(params, optim.init(params)),
                rollout_state=rollout_state,
            ),
            xs=rand.split(key, num_iterations),
        )
        def iteration_step(update_state: UpdateState, key: rand.PRNGKey):
            key_rollout, key_shuffle, key_eval, key_visualize = rand.split(key, 4)

            rollout_state, trajectories, advantages = jax.vmap(rollout, (None, 0, 0))(
                update_state.param_state.params,
                update_state.rollout_state,
                rand.split(key_rollout, config.num_parallel_envs),
            )

            @partial(
                jax.lax.scan,
                init=update_state.param_state,
                xs=rand.split(key_shuffle, config.num_epochs),
            )
            def epoch_step(param_state: ParamState, key: rand.PRNGKey):
                """For each epoch, reorder the minibatches"""
                permutation = rand.permutation(key, config.num_parallel_envs)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x[permutation], (config.num_minibatches, -1, *x.shape[1:])
                    ),
                    Minibatch(rollout_state.hidden, trajectories, advantages),
                )

                @partial(
                    jax.lax.scan,
                    init=param_state,
                    xs=minibatches,
                )
                def gradient_step(
                    update_state: ParamState,
                    minibatch: Minibatch,
                ) -> tuple[ParamState, Float[Array, ""]]:
                    def trajectory_loss(params: ActorCriticRNN, minibatch: Minibatch):
                        model = eqx.combine(params, network_static)
                        hidden, trajectory, advantages = minibatch

                        _, (logits, value) = model(
                            hidden,
                            ObsWithDone(
                                trajectory.obs, trajectory.env_state, trajectory.done
                            ),
                        )

                        # value loss
                        clipped_values = trajectory.value + jnp.clip(
                            value - trajectory.value,
                            -config.clip_epsilon,
                            config.clip_epsilon,
                        )
                        target_values = trajectory.value + advantages
                        value_loss = jnp.mean(
                            jnp.maximum(
                                rlax.l2_loss(value, target_values),
                                rlax.l2_loss(clipped_values, target_values),
                            )
                        )

                        # policy loss
                        action_idx = (jnp.arange(advantages.size), trajectory.action)
                        ratios = jnp.exp(
                            logits[action_idx] - trajectory.logits[action_idx]
                        )
                        normalized_advantages = (advantages - jnp.mean(advantages)) / (
                            jnp.std(advantages) + 1e-8
                        )
                        pg_loss = rlax.clipped_surrogate_pg_loss(
                            ratios, normalized_advantages, config.clip_epsilon
                        )

                        # entropy loss
                        entropy_loss = rlax.entropy_loss(
                            logits, jnp.ones_like(advantages)
                        )

                        # total
                        return (
                            pg_loss
                            + config.value_coefficient * value_loss
                            + config.entropy_coefficient * entropy_loss
                        ), (pg_loss, value_loss, entropy_loss)

                    def loss_fn(params: ActorCriticRNN):
                        loss, aux = jax.vmap(trajectory_loss, in_axes=(None, 0))(
                            params, minibatch
                        )
                        return jnp.mean(loss), aux

                    (loss, (pg_loss, value_loss, entropy_loss)), grad = (
                        jax.value_and_grad(loss_fn, has_aux=True)(update_state.params)
                    )
                    updates, opt_state = optim.update(
                        grad, update_state.opt_state, update_state.params
                    )
                    params = optax.apply_updates(update_state.params, updates)

                    jax.debug.callback(
                        wandb.log,
                        {
                            "train/total_reward": jnp.sum(
                                minibatch.trajectories.reward
                            ),
                            "train/loss": loss,
                            "train/pg_loss": jnp.mean(pg_loss),
                            "train/value_loss": jnp.mean(value_loss),
                            "train/entropy_loss": jnp.mean(entropy_loss),
                        }
                        | {
                            f"train/gradient{jax.tree_util.keystr(keys)}": jnp.linalg.norm(
                                update
                            )
                            for keys, update in jax.tree.leaves_with_path(updates)
                            if update is not None
                        },
                    )

                    return ParamState(params, opt_state), loss

                return gradient_step

            param_state, losses = epoch_step

            def eval_model(
                params: ActorCriticRNN, rollout_state: RolloutState, key: rand.PRNGKey
            ):
                with tempfile.NamedTemporaryFile(
                    suffix=".gif", delete=False
                ) as f_model:
                    _, trajectory, _ = rollout(params, rollout_state, key)
                    vis = Visualizer(
                        env,
                        env_params,
                        [
                            jax.tree.map(lambda x: x[i], trajectory.env_state)
                            for i in range(trajectory.obs.shape[0])
                        ],
                        jnp.cumsum(trajectory.reward),
                    )
                    vis.animate(f_model.name)

                    wandb.log(
                        {
                            "eval/model/rollout": wandb.Image(f_model.name),
                            "eval/model/rewards": jnp.sum(trajectory.reward),
                        }
                    )

                    plt.close(vis.fig)

            jax.lax.cond(
                rand.bernoulli(key_eval, 0.2),
                partial(jax.debug.callback, eval_model),
                lambda *args: None,
                param_state.params,
                jax.tree.map(lambda x: x[0], update_state.rollout_state),
                key_visualize,
            )

            return UpdateState(param_state, rollout_state), losses

        return iteration_step

    return train


if __name__ == "__main__":
    config = Config()
    key = rand.PRNGKey(184)
    train = jax.jit(make_train(config))

    with wandb.init(
        project="jax-rl",
        config=config._asdict(),
    ) as run:
        # with jax.disable_jit():
        out = jax.block_until_ready(train(key))
    print("Done training.")
