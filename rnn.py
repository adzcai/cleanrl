from functools import partial
from typing import NamedTuple
from jaxtyping import Bool, Integer, Key, Float, Array, PyTree


import jax
import jax.numpy as jnp
import jax.random as rand


import tempfile
import mctx
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


# jax.config.update("jax_enable_x64", True)
matplotlib.use("agg")


class WorldModelRNN(eqx.Module):
    """Parameterizes the actor-critic network."""

    cell: eqx.nn.GRUCell
    reward_head: eqx.nn.MLP

    def __init__(
        self,
        num_actions: int,
        rnn_size: int,
        mlp_size: int,
        mlp_depth: int,
        *,
        key: rand.PRNGKey,
    ):
        key_cell, key_reward = rand.split(key)
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

    def step(self, hidden: Float[Array, "rnn_size"], action: Integer[Array, ""]):
        """Reset to initial hidden state if done."""
        # TODO handle terminal states
        action = jax.nn.one_hot(action, self.cell.input_size)
        hidden = self.cell(action, hidden)
        return hidden, self.reward_head(hidden)

    def init_hidden(self):
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
        key: rand.PRNGKey,
    ):
        key_policy, key_value = rand.split(key)
        self.policy_head = eqx.nn.MLP(
            in_size, num_actions, mlp_size, mlp_depth, key=key_policy
        )
        self.value_head = eqx.nn.MLP(
            in_size, "scalar", mlp_size, mlp_depth, key=key_value
        )

    def __call__(self, hidden: Float[Array, "rnn_size"]):
        return self.policy_head(hidden), self.value_head(hidden)


class Networks(eqx.Module):
    """Network parameters."""

    projection: eqx.nn.Linear
    world_model: WorldModelRNN
    actor_critic: ActorCritic

    def __init__(self, obs_size: int, num_actions: int, *, key: rand.PRNGKey):
        key_projector, key_world_model, key_actor_critic = rand.split(key, 3)

        self.projection = eqx.nn.Linear(
            in_features=obs_size,
            out_features=config.rnn_size,
            key=key_projector,
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


class ObsWithDone(NamedTuple):
    obs: Float[Array, "*batch obs_size"]
    env_state: EnvState
    done: Bool[Array, "*batch"]


class Transition(NamedTuple):
    """A single transition. May be batched into a trajectory."""

    obs_with_done: ObsWithDone
    action: Integer[Array, "*batch"]
    reward: Float[Array, "*batch"]
    logits: Float[Array, "*batch num_actions"]


class ParamState(NamedTuple):
    """Each outer update iteration."""

    params: Networks
    opt_state: optax.OptState


class UpdateState(NamedTuple):
    param_state: ParamState
    rollout_state: ObsWithDone


class Config(NamedTuple):
    env: str = "Catch-bsuite"
    visualizations_dir: str = tempfile.gettempdir()
    seed: int = 184

    # environment collecting
    num_total_transitions: int = 40_000
    max_horizon: int = 40
    num_parallel_envs: int = 8

    # network architecture
    rnn_size: int = 32
    mlp_size: int = 16
    mlp_depth: int = 2

    # mcts
    num_mcts_simulations: int = 36

    # optimization
    num_minibatches: int = 4
    num_epochs: int = 4
    max_gradient_norm: float = 0.5
    learning_rate: float = 5e-3
    end_learning_rate: float = 1e-5

    # loss function
    discount: float = 0.99
    value_coefficient: float = 0.5
    reward_coefficient: float = 0.5
    num_steps: int = 100


def muzero_loss(
    params: Networks,
    trajectory: Transition,
    static: Networks,
):
    """MuZero loss."""
    networks = eqx.combine(params, static)

    rewards, logits, values = networks(
        trajectory.obs_with_done.obs[0], trajectory.action
    )
    # discard final transition
    target_values = rlax.n_step_bootstrapped_returns(
        trajectory.reward[:-1],
        jnp.where(trajectory.obs_with_done.done[1:], 0.0, config.discount),
        values[1:],
        config.num_steps,
    )

    # value loss
    value_losses = optax.l2_loss(values[:-1], target_values)
    value_loss = jnp.mean(value_losses)

    # policy loss
    policy_targets = jax.nn.softmax(trajectory.logits, axis=-1)
    policy_losses = optax.softmax_cross_entropy(logits, policy_targets)
    policy_loss = jnp.mean(policy_losses)

    # reward model loss
    reward_losses = optax.l2_loss(rewards, trajectory.reward)
    reward_loss = jnp.mean(reward_losses)

    # total
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        + config.reward_coefficient * reward_loss
    )
    return loss, {
        "train/loss": loss,
        "train/policy_loss": policy_loss,
        "train/value_loss": value_loss,
        "train/reward_loss": reward_loss,
    }


def get_norm_data(tree: PyTree, prefix: str):
    return {
        f"{prefix}{jax.tree_util.keystr(keys)}": jnp.linalg.norm(ary)
        for keys, ary in jax.tree.leaves_with_path(tree)
        if ary is not None
    }


def make_train(config: Config):
    num_transitions_per_iteration = config.max_horizon * config.num_parallel_envs
    num_iterations = config.num_total_transitions // num_transitions_per_iteration
    num_gradient_steps = num_iterations * config.num_epochs * config.num_minibatches

    lr = optax.linear_schedule(
        config.learning_rate,
        config.end_learning_rate,
        num_gradient_steps,
    )

    env, env_params = gymnax.make(config.env)
    env: Environment = FlattenObservationWrapper(env)
    # env: Environment = LogWrapper(env) # doesn't respect the envstate

    num_actions = env.action_space(env_params).n

    optim = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm), optax.adam(lr, eps=1e-5)
    )

    def train(key):
        key_reset, key_networks = rand.split(key)

        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            rand.split(key_reset, config.num_parallel_envs), env_params
        )
        rollout_state = ObsWithDone(
            obs,
            env_state,
            jnp.zeros(config.num_parallel_envs, dtype=bool),
        )

        networks = Networks(
            obs_size=obs.shape[-1], num_actions=num_actions, key=key_networks
        )
        params, network_static = eqx.partition(networks, eqx.is_inexact_array)

        def mcts_recurrent_fn(
            networks: Networks,
            rng: Float[Key, "batch"],
            action: Integer[Array, "batch"],
            world_state: Float[Array, "batch rnn_size"],
        ):
            hidden, reward = jax.vmap(networks.world_model.step)(world_state, action)
            logits, value = jax.vmap(networks.actor_critic)(hidden)
            return mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(value, config.discount),
                prior_logits=logits,
                value=value,
            ), hidden

        # collect rollouts
        def rollout(params: Networks, obs_state: ObsWithDone, key: rand.PRNGKey):
            """Collect a rollout from the environment and estimate the policy's advantage at each transition."""
            networks = eqx.combine(params, network_static)

            @partial(
                jax.lax.scan,
                init=obs_state,
                xs=rand.split(key, config.max_horizon),
            )
            def rollout_step(obs_state: ObsWithDone, key: rand.PRNGKey):
                """A single environment interaction."""
                key_action, key_step = rand.split(key)

                # choose action
                hidden = networks.projection(obs_state.obs)
                logits, value = networks.actor_critic(hidden)
                root = mctx.RootFnOutput(
                    prior_logits=logits,
                    value=value,
                    embedding=hidden,
                )
                output = mctx.muzero_policy(
                    networks,
                    key_action,
                    jax.tree.map(lambda x: x[jnp.newaxis, ...], root),
                    mcts_recurrent_fn,
                    config.num_mcts_simulations,
                    max_depth=config.max_horizon,
                )

                logits = output.action_weights[0]
                action = output.action[0]

                # step environment
                obs, env_state, reward, done, info = env.step(
                    key_step, obs_state.env_state, action, env_params
                )

                return ObsWithDone(obs, env_state, done), Transition(
                    obs_with_done=obs_state,
                    action=action,
                    reward=reward,
                    logits=logits,
                )

            return rollout_step

        @partial(
            jax.lax.scan,
            init=UpdateState(
                ParamState(params, optim.init(params)),
                rollout_state,
            ),
            xs=rand.split(key, num_iterations),
        )
        def iteration_step(update_state: UpdateState, key: rand.PRNGKey):
            """A single iteration of optimization.

            1. Collect a batch of rollouts in parallel.
            2. Run a few epochs of SGD (or some optimization algorithm) on the batch.
            """
            key_rollout, key_shuffle, key_eval, key_visualize, key_select_trajectory = (
                rand.split(key, 5)
            )

            # leading dimension num_parallel_envs
            rollout_state, trajectories = jax.vmap(rollout, (None, 0, 0))(
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
                    trajectories,
                )

                @partial(
                    jax.lax.scan,
                    init=param_state,
                    xs=minibatches,
                )
                def gradient_step(
                    update_state: ParamState,
                    trajectories: Transition,
                ) -> tuple[ParamState, Float[Array, ""]]:
                    @partial(jax.value_and_grad, has_aux=True)
                    def loss_fn(params: Networks):
                        """Average the loss across a batch of trjaectories."""
                        loss, aux = jax.vmap(muzero_loss, in_axes=(None, 0, None))(
                            params, trajectories, network_static
                        )
                        aux = jax.tree.map(partial(jnp.mean, axis=0), aux)
                        return jnp.mean(loss), aux

                    (loss, aux), grad = loss_fn(update_state.params)
                    updates, opt_state = optim.update(
                        grad, update_state.opt_state, update_state.params
                    )
                    params = optax.apply_updates(update_state.params, updates)

                    jax.debug.callback(
                        wandb.log,
                        aux
                        | get_norm_data(updates, "train/params/gradient")
                        | get_norm_data(params, "train/params/norm"),
                    )

                    return ParamState(params, opt_state), loss

                return gradient_step

            param_state, losses = epoch_step

            def eval_model(
                params: Networks,
                rollout_state: ObsWithDone,
                train_trajectory: Transition,
                key: rand.PRNGKey,
            ):
                def visualize_trajectory(
                    trajectory: Transition,
                    key: rand.PRNGKey,
                ):
                    name = rand.randint(key, (), 0, 1 << 28)
                    path = f"{config.visualizations_dir}/{name:07X}.gif"

                    vis = Visualizer(
                        env,
                        env_params,
                        [
                            jax.tree.map(
                                lambda x: x[i], trajectory.obs_with_done.env_state
                            )
                            for i in range(trajectory.obs_with_done.obs.shape[0])
                        ],
                        jnp.cumsum(trajectory.reward),
                    )
                    vis.animate(path)
                    plt.close(vis.fig)

                    return path

                key_eval, key_train, key_rollout = rand.split(key, 3)
                _obs_state, eval_trajectory = rollout(
                    params, rollout_state, key_rollout
                )
                eval_path = visualize_trajectory(eval_trajectory, key_eval)
                train_path = visualize_trajectory(train_trajectory, key_train)

                wandb.log(
                    {
                        "eval/rollout": wandb.Image(eval_path),
                        "eval/rewards": jnp.sum(eval_trajectory.reward),
                        "train/rollout": wandb.Image(train_path),
                    }
                )

            idx = rand.randint(key_select_trajectory, (), 0, config.num_parallel_envs)

            # sometimes plot model performance
            jax.lax.cond(
                rand.bernoulli(key_eval, 0.2),
                partial(jax.debug.callback, eval_model),
                lambda *args: None,
                param_state.params,
                jax.tree.map(lambda x: x[0], update_state.rollout_state),
                jax.tree.map(lambda x: x[idx], trajectories),
                key_visualize,
            )

            jax.debug.callback(
                wandb.log,
                {
                    "train/average_total_reward": jnp.mean(
                        jnp.sum(trajectories.reward, axis=-1)
                    ),
                },
            )

            return UpdateState(param_state, rollout_state), losses

        return iteration_step

    return train


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        config = Config(visualizations_dir=tempdir)
        key = rand.PRNGKey(config.seed)
        train = jax.jit(make_train(config))

        with wandb.init(
            project="jax-rl",
            config=config._asdict(),
        ) as run:
            # with jax.disable_jit():
            out = jax.block_until_ready(train(key))
        print("Done training.")
