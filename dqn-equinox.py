# %%
import tempfile
import jax
import jax.numpy as jnp
import jax.random as rand

import gymnax
from gymnax.environments.environment import Environment
from gymnax.visualize import Visualizer
import wandb

import matplotlib
import matplotlib.pyplot as plt

import equinox as eqx
import optax
import distrax

from typing import NamedTuple

# from collections.abc import Callable
from jaxtyping import Bool, Integer, Float, Array
from functools import partial
import math


matplotlib.use("agg")


# %%
class Config(NamedTuple):
    env_name: str
    eps_init: float
    eps_end: float
    eps_steps: int
    eps_begin: int

    lr_init: float
    lr_end: float

    buffer_size: int
    batch_size: int

    steps_per_update: int
    collect_workers: int
    update_steps: int
    updates_per_target: int
    target_update_size: float

    updates_per_eval: int
    eval_steps: int

    discount: float


class Transition(NamedTuple):
    obs: Float[Array, "..."]
    env_state: gymnax.EnvState
    action: Integer[Array, ""]
    reward: Float[Array, ""]
    done: Bool[Array, ""]

    @property
    def size(self):
        return self.action.size


# %%
class _ObsAndGymnaxState(NamedTuple):
    obs: Float[Array, "..."]
    env_state: gymnax.EnvState


def collect_transitions(
    rng: rand.PRNGKey,
    params: eqx.Module,
    init_step: int,
    n: int,
    # static
    model_static: eqx.Module,
    env: gymnax.environments.environment.Environment,
    env_params: gymnax.EnvParams,
    epsilon_schedule: optax.Schedule,
):
    model = eqx.combine(params, model_static)

    def env_step(carry: _ObsAndGymnaxState, x: tuple[Integer[Array, ""], rand.PRNGKey]):
        step, rng = x
        rng_act, rng_step, rng_reset = rand.split(rng, 3)

        # choose action
        logits = model(carry.obs.ravel())
        action = distrax.EpsilonGreedy(
            logits, epsilon_schedule(init_step + step)
        ).sample(seed=rng_act)

        # step environment
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, carry.env_state, action, env_params
        )

        next_obs, next_env_state = jax.lax.cond(
            done,
            lambda: env.reset(rng_reset, env_params),
            lambda: (next_obs, next_env_state),
        )

        return _ObsAndGymnaxState(
            obs=next_obs,
            env_state=next_env_state,
        ), Transition(
            obs=carry.obs,
            env_state=carry.env_state,
            action=action,
            reward=reward,
            done=done,
        )

    rng_reset, rng_step = rand.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    _, transitions = jax.lax.scan(
        env_step,
        _ObsAndGymnaxState(obs, env_state),
        (
            jnp.arange(n),
            rand.split(rng_step, n),
        ),
    )
    return transitions


def collect_transitions_distributed(
    f, rng: rand.PRNGKey, params: eqx.Module, init_step: int
):
    transitions_batch = jax.vmap(f, in_axes=(0, None, None, None))(
        rand.split(rng, config.collect_workers),
        params,
        init_step,
        config.steps_per_update // config.collect_workers,
    )
    return jax.tree.map(
        lambda leaf: jnp.reshape(leaf, (-1, *leaf.shape[2:])), transitions_batch
    )


# %%
@partial(jax.vmap, in_axes=(None, None, 0, 0, None))
def loss_batch(
    params: eqx.Module,
    target_params: eqx.Module,
    txn: Transition,
    next_txn: Transition,
    model_static: eqx.Module,
):
    model = eqx.combine(params, model_static)
    target_model = eqx.combine(target_params, model_static)

    target = txn.reward + jax.lax.cond(
        txn.done,
        lambda: 0.0,
        lambda: config.discount * jnp.max(target_model(next_txn.obs.ravel())),
    )

    return (target - model(txn.obs.ravel())[txn.action]) ** 2


def loss_fn(
    params: eqx.Module,
    target_params: eqx.Module,
    txn: Transition,
    next_txn: Transition,
    model_static: eqx.Module,
):
    losses = loss_batch(params, target_params, txn, next_txn, model_static)
    return jnp.mean(losses)


class Buffer(NamedTuple):
    data: Transition
    count: Integer[Array, ""]

    @classmethod
    def init(cls, env: Environment, env_params: gymnax.EnvParams, size: int):
        obs, env_state = env.reset(rand.PRNGKey(0), env_params)
        txn = Transition(
            obs=jnp.zeros_like(obs),
            env_state=env_state,
            action=env.action_space(env_params).sample(rand.PRNGKey(0)),
            reward=jnp.zeros(()),
            done=jnp.array(False),
        )
        return Buffer(
            data=jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (size, *leaf.shape)), txn
            ),
            count=0,
        )

    def add_batch(self, txns: Transition):
        idx = self.count + jnp.arange(txns.size) % self.data.size
        data = jax.tree.map(
            lambda batch, txns: batch.at[idx].set(txns),
            self.data,
            txns,
        )
        return Buffer(data, self.count + txns.size)

    def sample(self, rng, n) -> Transition:
        idxs = rand.randint(
            rng,
            (n,),
            1,
            jnp.minimum(self.count, self.data.size),
        )
        return jax.tree_map(
            lambda ary: ary[idxs - 1, ...],
            self.data,
        ), jax.tree_map(
            lambda ary: ary[idxs, ...],
            self.data,
        )


class _TrainCarry(NamedTuple):
    params: eqx.Module
    target_params: eqx.Module
    opt_state: optax.OptState
    buffer: Buffer


def train_step(
    carry: _TrainCarry,
    x: tuple[Integer[Array, ""], rand.PRNGKey],
    collect_transitions,
    optim: optax.GradientTransformation,
    model_static: eqx.Module,
):
    step, rng = x
    rng_collect, rng_buffer, rng_eval = rand.split(rng, 3)

    # collect transitions
    init_step = step * config.steps_per_update
    transitions = collect_transitions_distributed(
        collect_transitions, rng_collect, carry.params, init_step
    )
    buffer = carry.buffer.add_batch(transitions)

    loss, grads = jax.value_and_grad(loss_fn)(
        carry.params,
        carry.target_params,
        *buffer.sample(rng_buffer, config.batch_size),
        model_static,
    )
    updates, opt_state = optim.update(grads, carry.opt_state, carry.params)
    params = eqx.apply_updates(carry.params, updates)

    target_params = jax.lax.cond(
        step % config.updates_per_target == 0,
        lambda: optax.incremental_update(
            carry.params,
            carry.target_params,
            config.target_update_size,
        ),
        lambda: carry.target_params,
    )

    jax.debug.callback(
        wandb.log,
        {
            "train/total_reward": jnp.sum(transitions.reward),
            "train/loss": loss,
        }
        | {
            f"train/gradient{jax.tree_util.keystr(keys)}": jnp.linalg.norm(update)
            for keys, update in jax.tree.leaves_with_path(updates)
            if update is not None
        },
    )

    def eval_model(rng, params):
        with tempfile.NamedTemporaryFile(suffix=".gif") as f:
            txns = visualize(rng, params, f.name, config.eval_steps)
            wandb.log(
                {
                    "eval/rollout": wandb.Image(f.name),
                    "eval/rewards": jnp.sum(txns.reward),
                }
            )

    jax.lax.cond(
        step % config.updates_per_eval == 0,
        partial(jax.debug.callback, eval_model),
        lambda *args: None,
        rng_eval,
        params,
    )

    return _TrainCarry(params, target_params, opt_state, buffer), loss


@jax.jit
def train(rng):
    rng_model, rng_steps = rand.split(rng)

    env, env_params = gymnax.make(config.env_name)

    params, model_static = get_model(rng_model, env, env_params)

    # optimization
    optim = optax.adamw(
        optax.linear_schedule(config.lr_init, config.lr_end, config.update_steps)
    )
    opt_state = optim.init(params)

    # transition collection
    epsilon_schedule = optax.linear_schedule(
        config.eps_init, config.eps_end, config.eps_steps, config.eps_begin
    )

    collect_transitions_ = partial(
        collect_transitions,
        model_static=model_static,
        env=env,
        env_params=env_params,
        epsilon_schedule=epsilon_schedule,
    )

    # training loop
    train_step_ = partial(
        train_step,
        collect_transitions=collect_transitions_,
        optim=optim,
        model_static=model_static,
    )
    init_carry = _TrainCarry(
        params, params, opt_state, Buffer.init(env, env_params, config.buffer_size)
    )
    carry, losses = jax.lax.scan(
        train_step_,
        init_carry,
        (jnp.arange(config.update_steps), rand.split(rng_steps, config.update_steps)),
    )
    return carry, losses


def get_model(rng: rand.PRNGKey, env: Environment, env_params: gymnax.EnvParams):
    """Initialize model"""
    # model initialization
    model = eqx.nn.MLP(
        in_size=math.prod(env.observation_space(env_params).shape),
        out_size=env.action_space(env_params).n,
        width_size=128,
        depth=2,
        key=rng,
    )
    params, model_static = eqx.partition(model, eqx.is_inexact_array)
    return params, model_static


# %%
@partial(jax.jit, static_argnames=("env", "env_params", "n"))
def visualize_(
    rng: rand.PRNGKey,
    env: Environment,
    env_params: gymnax.EnvParams,
    params: eqx.Module,
    n: int,
):
    rng_model, rng_reset, rng_collect, rng_fname = rand.split(rng, 4)

    _, model_static = get_model(rng_model, env, env_params)

    _, env_state = env.reset(rng_reset, env_params)
    transitions = collect_transitions(
        rng_collect,
        params,
        0,
        n,
        env=env,
        env_params=env_params,
        model_static=model_static,
        epsilon_schedule=optax.constant_schedule(0.0),
    )
    env_states = jax.tree.transpose(
        jax.tree.structure(env_state),
        None,
        jax.tree.map(lambda leaf: list(leaf), transitions.env_state),
    )
    return transitions, env_states


def visualize(rng, params: eqx.Module, fname: str, n: int) -> Transition:
    env, env_params = gymnax.make(config.env_name)
    transitions, env_states = visualize_(rng, env, env_params, params, n)
    vis = Visualizer(env, env_params, env_states, jnp.cumsum(transitions.reward))
    vis.animate(fname)
    plt.close(vis.fig)
    return transitions


# %%
if __name__ == "__main__":
    # %%
    rng = rand.PRNGKey(0)

    # %%
    rng, rng_train, rng_model = rand.split(rng, 3)

    config = Config(
        env_name="Catch-bsuite",
        eps_init=1.0,
        eps_end=0.01,
        eps_steps=20_000,
        eps_begin=500,
        lr_init=0.01,
        lr_end=0.0001,
        buffer_size=1000,
        batch_size=400,
        steps_per_update=240,
        collect_workers=12,
        update_steps=5_000,
        updates_per_target=20,
        target_update_size=0.9,
        updates_per_eval=1000,
        eval_steps=200,
        discount=0.99,
    )

    wandb.init(
        project="jax-rl",
        config=config._asdict(),
    )

    carry, losses = jax.block_until_ready(train(rng_train))
