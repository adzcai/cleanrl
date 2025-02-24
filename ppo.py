import functools as ft
from typing import Literal, NamedTuple

import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import rlax
from gymnax.wrappers import FlattenObservationWrapper
from jaxtyping import Array, Float

import wandb


class NetworkOutput(NamedTuple):
    logits: Float[Array, "num_actions"]
    value_logits: float


class Network(eqx.Module):
    value_net: eqx.nn.MLP
    policy_net: eqx.nn.MLP

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        num_value_bins: int | Literal["scalar"],
        mlp_size: int,
        mlp_depth: int,
        *,
        key: jr.PRNGKey,
    ):
        value_key, policy_key = jr.split(key)
        self.value_net = eqx.nn.MLP(
            obs_size, num_value_bins, mlp_size, mlp_depth, jax.nn.tanh, key=value_key
        )
        self.policy_net = eqx.nn.MLP(
            obs_size, num_actions, mlp_size, mlp_depth, jax.nn.tanh, key=policy_key
        )

    def __call__(self, obs):
        return NetworkOutput(self.policy_net(obs), self.value_net(obs))


class ParamsState(NamedTuple):
    params: Network
    opt_state: optax.OptState


class RolloutState(NamedTuple):
    obs: Float[Array, "obs_size"]
    env_state: gymnax.EnvState


class Transition(NamedTuple):
    rollout_state: RolloutState
    action: int
    reward: float
    done: bool
    out: NetworkOutput


class IterState(NamedTuple):
    rollout_states: RolloutState
    params_state: ParamsState


class Config(NamedTuple):
    env_name: str = "Catch-bsuite"

    # data collection
    num_transitions: int = 500_000
    num_envs: int = 8
    horizon: int = 100

    # network architecture
    mlp_size: int = 64
    mlp_depth: int = 2

    # optimization
    lr_init: float = 2e-2
    lr_end: float = 0.0
    num_updates: int = 4
    batch_size: int = 4

    # loss function and bootstrapping
    discount: float = 0.99
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lambda_gae: float = 0.95

    # two-hot
    num_value_bins: int | Literal["scalar"] = 10
    min_value: float = -10
    max_value: float = 10


def make_train(config: Config):
    num_iters = config.num_transitions // (config.num_envs * config.horizon)
    num_opt_steps = num_iters * config.num_updates
    env, env_params = gymnax.make(config.env_name)
    env = FlattenObservationWrapper(env)
    lr = optax.linear_schedule(config.lr_init, config.lr_end, num_opt_steps)

    optim = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(lr, eps=1e-5),
    )

    tx_pair = rlax.twohot_pair(config.min_value, config.max_value, config.num_value_bins)

    def train(key: jr.PRNGKey):
        key_net, key_reset, key_iterate = jr.split(key, 3)

        # initial states
        obs, env_state = jax.vmap(env.reset, (0, None))(
            jr.split(key_reset, config.num_envs), env_params
        )

        # initialize network
        net = Network(
            obs_size=obs.shape[-1],
            num_actions=env.action_space(env_params).n,
            num_value_bins=config.num_value_bins,
            mlp_size=config.mlp_size,
            mlp_depth=config.mlp_depth,
            key=key_net,
        )
        params, net_static = eqx.partition(net, eqx.is_inexact_array)

        @ft.partial(
            jax.lax.scan,
            init=IterState(
                RolloutState(obs, env_state),
                ParamsState(params, optim.init(params)),
            ),
            xs=jr.split(key_iterate, num_iters),
        )
        def iterate(iter_state: IterState, key: jr.PRNGKey):
            """One iteration of rollouts and optimization."""
            rollout_states, (trajectories, advantages, value_targets) = jax.vmap(
                rollout, (None, 0, 0)
            )(
                eqx.combine(iter_state.params_state.params, net_static),
                iter_state.rollout_states,
                jr.split(key, config.num_envs),
            )

            @ft.partial(
                jax.lax.scan,
                init=iter_state.params_state,
                xs=jr.split(key, config.num_updates),
            )
            def optimize_step(
                params_state: ParamsState,
                key: jr.PRNGKey,
            ):
                idx = jr.randint(key, config.batch_size, 0, config.num_envs)
                batch = jax.tree.map(lambda x: x[idx], (trajectories, advantages, value_targets))

                @ft.partial(jax.value_and_grad, has_aux=True)
                def loss_fn(params: Network):
                    net = eqx.combine(params, net_static)
                    losses, aux = jax.vmap(loss_trajectory, (None, 0, 0, 0))(net, *batch)
                    return jnp.mean(losses), aux

                loss_aux, grads = loss_fn(params_state.params)
                updates, opt_state = optim.update(
                    grads, params_state.opt_state, params_state.params
                )
                params = optax.apply_updates(params_state.params, updates)
                return ParamsState(params, opt_state), loss_aux

            params_state, (losses, auxs) = optimize_step

            jax.debug.callback(
                wandb.log,
                {
                    "train/loss": jnp.mean(losses),
                    "train/reward": jnp.sum(trajectories.reward) / config.num_envs,
                }
                | jax.tree.map(jnp.mean, auxs),
            )

            return IterState(rollout_states, params_state), None

        final_iter_state, _ = iterate
        return final_iter_state

    def rollout(net: Network, init_rollout_state: RolloutState, key: jr.PRNGKey):
        """Collect a single trajectory."""

        @ft.partial(jax.lax.scan, init=init_rollout_state, xs=jr.split(key, config.horizon))
        def rollout_step(rollout_state: RolloutState, key: jr.PRNGKey):
            key_action, key_step = jr.split(key)
            preds = net(rollout_state.obs)
            action = jr.categorical(key_action, preds.logits)
            obs, env_state, reward, done, info = env.step(
                key_step, rollout_state.env_state, action, env_params
            )

            return RolloutState(obs, env_state), Transition(
                rollout_state, action, reward, done, preds
            )

        final_rollout_state, trajectory = rollout_step

        final_value_logits = net(final_rollout_state.obs).value_logits
        value_logits = jnp.vstack([trajectory.out.value_logits, final_value_logits])
        values = tx_pair.apply_inv(jax.nn.softmax(value_logits))
        advantages = rlax.truncated_generalized_advantage_estimation(
            trajectory.reward,
            jnp.where(trajectory.done, 0.0, config.discount),
            config.lambda_gae,
            values,
        )
        value_target_probs = tx_pair.apply(advantages + values[:-1])

        return final_rollout_state, (trajectory, advantages, value_target_probs)

    def loss_trajectory(
        net: Network,
        trajectory: Transition,
        advantages: Float[Array, "horizon"],
        value_target_probs: Float[Array, "horizon num_value_bins"],
    ):
        preds = jax.vmap(net)(trajectory.rollout_state.obs)

        # policy
        idx = (jnp.arange(config.horizon), trajectory.action)
        ratios = jnp.exp(preds.logits[idx] - trajectory.out.logits[idx])
        policy_loss = rlax.clipped_surrogate_pg_loss(ratios, advantages, config.clip_eps)

        # value
        value_losses = rlax.categorical_cross_entropy(
            value_target_probs,
            preds.value_logits,
        )
        value_loss = jnp.mean(value_losses)
        td_error = tx_pair.apply_inv(value_target_probs) - tx_pair.apply_inv(
            jax.nn.softmax(preds.value_logits)
        )

        # entropy
        entropy_loss = rlax.entropy_loss(preds.logits, jnp.ones(config.horizon))

        return (
            policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss
        ), {
            "train/td_error": td_error,
            "train/value_loss": value_loss,
            "train/policy_loss": policy_loss,
        }

    return train, (rollout, loss_trajectory)


if __name__ == "__main__":
    config = Config()
    train, _ = make_train(config)
    train_jit = jax.jit(train)

    DEBUG = False

    if DEBUG:
        with jax.disable_jit():
            out = train_jit(jr.PRNGKey(0))
            jax.block_until_ready(out)
    else:
        with wandb.init(project="jax-rl", config=config._asdict(), tags=["ppo", "categorical"]):
            out = train_jit(jr.PRNGKey(0))
            jax.block_until_ready(out)
