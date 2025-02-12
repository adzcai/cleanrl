import jax
import jax.numpy as jnp
import jax.random as rand

import equinox as eqx
import optax
import rlax
import mctx
import flashbax as fbx

import gymnax
from gymnax.environments.environment import Environment
from gymnax.wrappers import FlattenObservationWrapper
from gymnax.visualize import Visualizer

import tempfile
import wandb
import matplotlib
import matplotlib.pyplot as plt

from functools import partial
from typing import Literal, NamedTuple
from jaxtyping import Bool, Integer, Key, Float, Array, PyTree, jaxtyped
from typeguard import typechecked
from batched import Batched


matplotlib.use("agg")


class ObsState(NamedTuple):
    """An observation and whether the episode terminated."""

    obs: Float[Array, "obs_size"]
    is_initial: Bool[Array, ""]


class ActorCriticOutput(NamedTuple):
    policy_logits: Float[Array, "num_actions"]
    value_logits: Float[Array, "num_value_bins"]


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
        num_value_bins: int | Literal["scalar"],
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
            rnn_size, num_value_bins, mlp_size, mlp_depth, key=key_value
        )

    def __call__(self, hidden: Float[Array, "rnn_size"], inputs: Batched[ObsState]):
        return jax.lax.scan(
            jax.tree_util.Partial(self.step),
            hidden,
            inputs,
        )

    def step(self, hidden: Float[Array, "rnn_size"], input: ObsState):
        """Reset to initial hidden state if done."""
        hidden = jnp.where(input.is_initial, self.init_hidden(), hidden)
        hidden = self.cell(input.obs, hidden)
        return hidden, ActorCriticOutput(
            self.policy_head(hidden),
            self.value_head(hidden),
        )

    def init_hidden(self):
        return jnp.zeros(self.cell.hidden_size)


class UnobsState(NamedTuple):
    """Used in MCTS. The unobserved state of the environment."""

    env_state: gymnax.EnvState
    hidden: Float[Array, "rnn_size"]


class RolloutState(NamedTuple):
    """Carried when rolling out the environment."""

    obs_state: ObsState
    unobs_state: UnobsState


class Transition(NamedTuple):
    """A single transition. May be batched into a trajectory."""

    rollout_state: RolloutState
    action: Integer[Array, ""]
    reward: Float[Array, ""]
    logits: Float[Array, "num_actions"]


class ParamState(NamedTuple):
    """Each outer update iteration."""

    params: ActorCriticRNN
    opt_state: optax.OptState


class IterState(NamedTuple):
    num_iterations: Integer[Array, ""]
    param_state: ParamState
    target_params: ActorCriticRNN
    buffer_state: fbx.trajectory_buffer.TrajectoryBufferState[Transition]
    rollout_state: Batched[RolloutState]


class Config(NamedTuple):
    env: str = "Catch-bsuite"
    visualizations_dir: str = tempfile.gettempdir()
    seed: int = 184
    num_evals: int = 4

    # environment collecting
    num_total_transitions: int = 50_000
    max_horizon: int = 100
    num_parallel_envs: int = 8
    num_mcts_simulations: int = 48

    # network architecture
    rnn_size: int = 64
    mlp_size: int = 32
    mlp_depth: int = 3

    # optimization
    num_gradient_steps_per_iteration: int = 20
    max_gradient_norm: float = 0.5
    learning_rate: float = 1e-3
    end_learning_rate: float = 1e-6
    target_update_frequency: int = 4
    target_step_size: float = 1.0

    # two-hot encoding
    min_value: float = -12.0
    max_value: float = 12.0
    num_value_bins: int = 10

    # loss function
    discount: float = 0.99
    value_coefficient: float = 0.5
    lambda_gae: float = 0.95


def get_norm_data(tree: PyTree[Array], prefix: str):
    """Utility function for logging norms of pytree leaves."""
    return {
        f"{prefix}{jax.tree_util.keystr(keys)}": jnp.linalg.norm(ary)
        for keys, ary in jax.tree.leaves_with_path(tree)
        if ary is not None
    }


def tree_slice(tree: PyTree, at: int):
    """Utility function for slicing pytree leaves."""
    return jax.tree.map(lambda x: x[at], tree)


def make_train(config: Config):
    num_transitions_per_iteration = config.max_horizon * config.num_parallel_envs
    num_iterations = config.num_total_transitions // num_transitions_per_iteration
    num_gradient_steps = num_iterations * config.num_gradient_steps_per_iteration
    eval_frequency = num_iterations // config.num_evals

    env, env_params = gymnax.make(config.env)
    env: Environment = FlattenObservationWrapper(env)
    # env: Environment = LogWrapper(env)  # doesn't respect the envstate

    num_actions = env.action_space(env_params).n

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=config.num_parallel_envs,
        sample_batch_size=config.num_parallel_envs,
        sample_sequence_length=config.max_horizon,
        period=1,
        min_length_time_axis=num_transitions_per_iteration,
        max_size=int(1e6),
    )

    lr = optax.linear_schedule(
        config.learning_rate,
        config.end_learning_rate,
        num_gradient_steps,
    )

    optim = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm), optax.adamw(lr, eps=1e-5)
    )

    tx = rlax.twohot_pair(config.min_value, config.max_value, config.num_value_bins)

    @jaxtyped(typechecker=typechecked)
    def train(key):
        """Train the agent.

        All randomness goes into this function."""
        key_reset, key_network = rand.split(key)

        # initial states
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            rand.split(key_reset, config.num_parallel_envs), env_params
        )

        # initialize network
        network = ActorCriticRNN(
            in_size=obs.shape[-1],
            rnn_size=config.rnn_size,
            num_actions=num_actions,
            num_value_bins=config.num_value_bins,
            mlp_size=config.mlp_size,
            mlp_depth=config.mlp_depth,
            key=key_network,
        )
        params, network_static = eqx.partition(network, eqx.is_inexact_array)

        init_hidden = network.init_hidden()
        init_rollout_state = RolloutState(
            ObsState(
                obs,
                jnp.zeros(config.num_parallel_envs, dtype=bool),
            ),
            UnobsState(
                env_state,
                jnp.broadcast_to(
                    init_hidden, (config.num_parallel_envs, *init_hidden.shape)
                ),
            ),
        )

        # init buffer
        init_buffer_state = buffer.init(
            Transition(
                rollout_state=tree_slice(init_rollout_state, 0),
                action=0,
                reward=0.0,
                logits=jnp.zeros(num_actions),
            )
        )
        init_iter_state = IterState(
            0,
            ParamState(params, optim.init(params)),
            params,
            init_buffer_state,
            init_rollout_state,
        )

        # main loop
        @partial(
            jax.lax.scan,
            init=init_iter_state,
            xs=rand.split(key, num_iterations),
        )
        def iteration_step(iter_state: IterState, key: rand.PRNGKey):
            """A single iteration of optimization.

            1. Collect a batch of rollouts in parallel.
            2. Run a few epochs of SGD (or some optimization algorithm) on the batch.
            """
            key_rollout, key_shuffle, key_visualize = rand.split(key, 3)

            # collect rollouts and update buffer
            # leading dimension num_parallel_envs
            rollout_state, trajectories = jax.vmap(rollout, (None, 0, 0))(
                eqx.combine(iter_state.param_state.params, network_static),
                iter_state.rollout_state,
                rand.split(key_rollout, config.num_parallel_envs),
            )
            buffer_state = buffer.add(iter_state.buffer_state, trajectories)

            # optimize
            param_state, losses = jax.lax.scan(
                partial(
                    optimize_step,
                    buffer_state,
                    iter_state.target_params,
                    network_static=network_static,
                ),
                iter_state.param_state,
                rand.split(key_shuffle, config.num_gradient_steps_per_iteration),
            )

            jax.debug.callback(
                wandb.log,
                {
                    "train/metrics/mean_mcts_reward": jnp.sum(trajectories.reward)
                    / config.num_parallel_envs,
                },
            )

            # evaluate
            jax.lax.cond(
                iter_state.num_iterations % eval_frequency == 0,
                partial(eval_model, network_static=network_static),
                lambda *args: None,
                param_state.params,
                tree_slice(iter_state.rollout_state, 0),
                key_visualize,
            )

            target_params = jax.lax.cond(
                iter_state.num_iterations % config.target_update_frequency == 0,
                lambda new_params, old_params: optax.incremental_update(
                    new_params,
                    old_params,
                    config.target_step_size,
                ),
                lambda _, old_params: old_params,
                param_state.params,
                iter_state.target_params,
            )

            return IterState(
                iter_state.num_iterations + 1,
                param_state,
                target_params,
                buffer_state,
                rollout_state,
            ), losses

        return iteration_step

    def mcts_recurrent_fn(
        model: ActorCriticRNN,
        rng: Float[Key, "batch"],
        action: Integer[Array, "batch"],
        world_state: UnobsState,
    ):
        obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rand.split(rng, action.size), world_state.env_state, action, env_params)
        hidden, (logits, value) = jax.vmap(model.step)(
            world_state.hidden, ObsState(obs, done)
        )
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(done, 0.0, config.discount),
            prior_logits=logits,
            value=tx.apply_inv(value),
        ), UnobsState(env_state, hidden)

    # collect rollouts
    def rollout(
        model: ActorCriticRNN,
        init_rollout_state: RolloutState,
        key: rand.PRNGKey,
    ):
        """Collect a rollout from the environment and estimate the policy's advantage at each transition."""

        @partial(
            jax.lax.scan,
            init=init_rollout_state,
            xs=rand.split(key, config.max_horizon),
        )
        def rollout_step(rollout_state: RolloutState, key: rand.PRNGKey):
            """A single environment interaction."""
            key_action, key_step = rand.split(key)

            # choose action
            hidden, (logits, value) = model.step(
                rollout_state.unobs_state.hidden, rollout_state.obs_state
            )
            root = jax.tree.map(
                lambda x: x[jnp.newaxis, ...],
                mctx.RootFnOutput(
                    prior_logits=logits,
                    value=tx.apply_inv(value).reshape(()),
                    embedding=UnobsState(rollout_state.unobs_state.env_state, hidden),
                ),
            )
            output = mctx.muzero_policy(
                model,
                key_action,
                root,
                mcts_recurrent_fn,
                config.num_mcts_simulations,
                max_depth=config.max_horizon,
            )

            logits = output.action_weights[0]
            action = output.action[0]

            # step environment
            obs, env_state, reward, done, info = env.step(
                key_step, rollout_state.unobs_state.env_state, action, env_params
            )

            return RolloutState(
                ObsState(obs, done),
                UnobsState(env_state, hidden),
            ), Transition(
                rollout_state,
                action,
                reward,
                logits,
            )

        return rollout_step

    def optimize_step(
        buffer_state: fbx.trajectory_buffer.TrajectoryBufferState[Transition],
        target_params: ActorCriticRNN,
        param_state: ParamState,
        key: rand.PRNGKey,
        *,
        network_static: ActorCriticRNN,
    ):
        """For each epoch, reorder the minibatches"""
        minibatch = buffer.sample(buffer_state, key).experience

        @partial(jax.value_and_grad, has_aux=True)
        def loss_fn(params: ActorCriticRNN):
            """Average the loss across a batch of trjaectories."""
            loss, aux = jax.vmap(
                partial(bc_loss, network_static=network_static), (None, None, 0)
            )(
                params,
                target_params,
                minibatch,
            )
            return jnp.mean(loss), aux

        (loss, aux), grad = loss_fn(param_state.params)
        updates, opt_state = optim.update(
            grad,
            param_state.opt_state,
            param_state.params,
        )
        params = optax.apply_updates(param_state.params, updates)

        jax.debug.callback(
            wandb.log,
            jax.tree.map(jnp.mean, aux)
            | get_norm_data(updates, "train/params/gradient")
            | get_norm_data(params, "train/params/norm"),
        )

        return ParamState(params, opt_state), loss

    def bc_loss(
        params: ActorCriticRNN,
        target_params: ActorCriticRNN,
        trajectory: Transition,
        *,
        network_static: ActorCriticRNN,
    ):
        """Behavior cloning loss for the policy."""
        model = eqx.combine(params, network_static)
        init_hidden = trajectory.rollout_state.unobs_state.hidden[0]
        obs = trajectory.rollout_state.obs_state
        _, preds = model(init_hidden, obs)

        # value loss
        target_model = eqx.combine(target_params, network_static)
        _, (_, target_value_logits) = target_model(init_hidden, obs)
        bootstrapped_probs = rlax.transformed_lambda_returns(
            tx,
            trajectory.reward[:-1],
            jnp.where(
                trajectory.rollout_state.obs_state.is_initial[1:], 0.0, config.discount
            ),
            target_value_logits[1:],
            config.lambda_gae,
        )

        value_losses = rlax.categorical_cross_entropy(
            bootstrapped_probs,
            preds.value_logits[:-1],
        )
        target_values = tx.apply_inv(jax.nn.softmax(target_value_logits, axis=-1))
        value_probs = jax.nn.softmax(preds.value_logits, axis=-1)
        predicted_values = tx.apply_inv(value_probs)

        value_entropy = jax.vmap(rlax.softmax().entropy)(preds.value_logits)
        value_loss = jnp.mean(value_losses)
        td_errors = target_values - predicted_values

        # policy loss
        policy_targets = jax.nn.softmax(trajectory.logits, axis=-1)
        policy_losses = optax.softmax_cross_entropy(preds.policy_logits, policy_targets)
        policy_loss = jnp.mean(policy_losses)

        # total
        loss = policy_loss + config.value_coefficient * value_loss

        return loss, {
            "train/metrics/mean_bootstrapped_value": jnp.mean(
                tx.apply_inv(bootstrapped_probs)
            ),
            "train/metrics/mean_target_value": jnp.mean(target_values),
            "train/metrics/mean_predicted_value": jnp.mean(predicted_values),
            "train/metrics/mean_td_error": jnp.mean(td_errors),
            "train/metrics/mean_value_entropy": jnp.mean(value_entropy),
            "train/metrics/loss": loss,
            "train/metrics/policy_loss": policy_loss,
            "train/metrics/value_loss": value_loss,
        }

    def eval_model(
        params: ActorCriticRNN,
        rollout_state: gymnax.EnvState,
        key: rand.PRNGKey,
        *,
        network_static: ActorCriticRNN,
    ):
        key_rollout, key_id = rand.split(key)
        _, eval_trajectory = rollout(
            params, rollout_state, key_rollout, network_static=network_static
        )
        id = rand.randint(key_id, (), 0, 1 << 28)

        def visualize_trajectory(
            env_states: EnvState,
            rewards: Float[Array, "horizon"],
            id: Integer[Array, ""],
        ):
            path = f"{config.visualizations_dir}/{id.item():07X}.gif"

            # visualizer expects a list of env_states
            vis = Visualizer(
                env,
                env_params,
                [tree_slice(env_states, i) for i in range(rewards.size)],
                jnp.cumsum(rewards),
            )
            vis.animate(path)

            wandb.log(
                {
                    "eval/rollout": wandb.Image(path),
                    "eval/rewards": jnp.sum(rewards),
                }
            )

            plt.close(vis.fig)

        jax.debug.callback(
            visualize_trajectory,
            eval_trajectory.rollout_state.unobs_state.env_state,
            eval_trajectory.reward,
            id,
        )

    return train, (mcts_recurrent_fn, rollout, optimize_step, bc_loss, eval_model)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        config = Config(visualizations_dir=tempdir)
        key = rand.PRNGKey(config.seed)
        train = jax.jit(make_train(config)[0])

        with wandb.init(
            project="jax-rl",
            config=config._asdict(),
        ) as run:
            # with jax.disable_jit():
            out = jax.block_until_ready(train(key))
        print("Done training.")
