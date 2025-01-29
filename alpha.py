# %%
import tempfile
import jax
import jax.numpy as jnp
import jax.random as rand

import gymnax
from gymnax.environments.environment import Environment
from gymnax.wrappers import FlattenObservationWrapper
from gymnax.visualize import Visualizer
import wandb

import matplotlib
import matplotlib.pyplot as plt

import equinox as eqx
import optax
import mctx

from typing import NamedTuple, Optional
from collections.abc import Callable
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
    update_batch_size: int
    bootstrap_after: int

    steps_per_update: int
    collect_workers: int
    update_steps: int
    updates_per_target: int
    target_update_size: float

    updates_per_eval: int
    eval_steps: int

    discount: float

    mcts_batch_dim: int
    num_mcts_simulations: int


class Transition(NamedTuple):
    obs: Float[Array, "..."]
    env_state: gymnax.EnvState
    logits: Integer[Array, " a"]
    reward: Float[Array, ""]
    done: Bool[Array, ""]

    @property
    def size(self):
        return self.done.size


# %%
class ObsWithGymnaxState(NamedTuple):
    obs: Float[Array, "..."]
    env_state: gymnax.EnvState


class EnvStateWithTerminal(NamedTuple):
    env_state: gymnax.EnvState
    terminal: bool


def mcts_recurrent_fn(
    params_pv: eqx.Module,
    rng: rand.PRNGKey,
    action_batch: Integer[Array, " b"],
    rec_batch: EnvStateWithTerminal,
    model_pv_static: eqx.Module,
    env: Environment,
    env_params: gymnax.EnvParams,
) -> tuple[mctx.RecurrentFnOutput, EnvStateWithTerminal]:
    """Recurrent function for MCTS accounting for terminal states."""

    def step(rng: rand.PRNGKey, action: Integer[Array, ""], env_state: gymnax.EnvState):
        next_obs, next_env_state, reward, done, info = env.step(
            rng, env_state, action, env_params
        )
        model_pv = eqx.combine(params_pv, model_pv_static)
        logits, value = model_pv(next_obs)
        return mctx.RecurrentFnOutput(
            reward=reward, discount=config.discount, prior_logits=logits, value=value
        ), EnvStateWithTerminal(next_env_state, done)

    def step_with_terminal(
        rng: rand.PRNGKey, action: Integer[Array, ""], rec: EnvStateWithTerminal
    ):
        return jax.lax.cond(
            rec.terminal,
            lambda *_args: (
                mctx.RecurrentFnOutput(
                    reward=0.0,
                    discount=0.0,
                    prior_logits=jnp.zeros(env.num_actions),
                    value=0.0,
                ),
                rec,
            ),
            step,
            rng,
            action,
            rec.env_state,
        )

    return jax.vmap(step_with_terminal)(
        rand.split(rng, action_batch.size), action_batch, rec_batch
    )


def choose_action_mcts(
    rng: rand.PRNGKey,
    params_pv: eqx.Module,
    obs: Float[Array, "..."],
    env_state: gymnax.EnvState,
    model_pv_static: eqx.Module,
    env: Environment,
    env_params: gymnax.EnvParams,
    batch_dim: int,
    num_simulations: int,
) -> mctx.PolicyOutput[None]:
    model_pv = eqx.combine(params_pv, model_pv_static)
    logits, value = model_pv(obs)

    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=EnvStateWithTerminal(env_state, jnp.array(False)),
    )
    root = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_dim, *leaf.shape)), root
    )

    output = mctx.muzero_policy(
        params=params_pv,
        rng_key=rng,
        root=root,
        recurrent_fn=partial(
            mcts_recurrent_fn,
            model_pv_static=model_pv_static,
            env=env,
            env_params=env_params,
        ),
        num_simulations=num_simulations,
    )

    return output


def collect_transitions_mcts(
    rng: rand.PRNGKey,
    params_pv: eqx.Module,
    num_transitions: int,
    model_pv_static: eqx.Module,
    env: Environment,
    env_params: gymnax.EnvParams,
    batch_dim: int,
    num_mcts_simulations: int,
):
    def step(carry: ObsWithGymnaxState, rng: rand.PRNGKey):
        rng_policy, rng_step = rand.split(rng)

        output = choose_action_mcts(
            rng=rng_policy,
            params_pv=params_pv,
            obs=carry.obs,
            env_state=carry.env_state,
            model_pv_static=model_pv_static,
            env=env,
            env_params=env_params,
            batch_dim=batch_dim,
            num_simulations=num_mcts_simulations,
        )

        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, carry.env_state, output.action[0], env_params
        )

        return ObsWithGymnaxState(next_obs, next_env_state), Transition(
            obs=carry.obs,
            env_state=carry.env_state,
            logits=output.action_weights[0],
            reward=reward,
            done=done,
        )

    rng_reset, rng_steps = rand.split(rng)
    _, transitions = jax.lax.scan(
        step,
        ObsWithGymnaxState(*env.reset(rng_reset, env_params)),
        rand.split(rng_steps, num_transitions),
    )

    return transitions


class ComputeValueCarry(NamedTuple):
    steps_since_terminal: int


class ComputeValueInput(NamedTuple):
    txn: Transition
    bootstrap_value: float


def compute_returns(txns: Transition):
    def step(return_to_go: float, txn: Transition):
        return_to_go = txn.reward + jax.lax.select(
            txn.done,
            0.0,
            config.discount * return_to_go,
        )
        return return_to_go, return_to_go

    _, returns = jax.lax.scan(
        step,
        0.0,
        txns,
        reverse=True,
    )
    return returns


def compute_n_step_returns(
    txns: Transition, values: Float[Array, " b"], bootstrap_after: int
):
    returns = compute_returns(txns)

    def loop_n_step_returns(
        steps_to_terminal: int, x: tuple[bool, float, float, float]
    ):
        done, return_to_go, future_return, future_value = x
        steps_to_terminal = jnp.where(done, 0, steps_to_terminal + 1)
        n_step_return = jnp.where(
            steps_to_terminal < bootstrap_after,
            return_to_go,
            return_to_go
            + config.discount**bootstrap_after * (future_value - future_return),
        )
        return steps_to_terminal, n_step_return

    _, n_step_returns = jax.lax.scan(
        loop_n_step_returns,
        0,
        (
            txns.done,
            returns,
            jnp.pad(returns[bootstrap_after:], (0, bootstrap_after)),
            jnp.pad(values[bootstrap_after:], (0, bootstrap_after)),
        ),
        reverse=True,
    )

    return n_step_returns


def loss_fn(
    params: eqx.Module,
    txns: Transition,
    returns: Float[Array, " b"],
    model_static: eqx.Module,
):
    model = eqx.combine(params, model_static)
    logits, values = jax.vmap(model)(txns.obs)

    value_loss = optax.l2_loss(values, returns)
    policy_loss = optax.softmax_cross_entropy(logits, txns.logits)
    regularizer = optax.global_norm(params)
    return jnp.mean(value_loss) + jnp.mean(policy_loss) + jnp.mean(regularizer)


class Buffer(NamedTuple):
    data: Transition
    returns: Float[Array, " b"]
    count: Integer[Array, ""]

    @classmethod
    def init(cls, env: Environment, env_params: gymnax.EnvParams, size: int):
        obs, env_state = env.reset(rand.PRNGKey(0), env_params)
        txn = Transition(
            obs=jnp.zeros_like(obs),
            env_state=env_state,
            logits=jnp.zeros(env.num_actions),
            reward=jnp.zeros(()),
            done=jnp.array(False),
        )
        return Buffer(
            data=jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (size, *leaf.shape)), txn
            ),
            returns=jnp.zeros(size),
            count=0,
        )

    def add_batch(self, txns: Transition, returns: Float[Array, " b"]):
        idx = self.count + jnp.arange(txns.size) % self.data.size
        data = jax.tree.map(
            lambda batch, txns: batch.at[idx].set(txns),
            self.data,
            txns,
        )
        return Buffer(data, self.returns.at[idx].set(returns), self.count + txns.size)

    def sample(self, rng, n) -> Transition:
        idxs = rand.randint(
            rng,
            (n,),
            1,
            jnp.minimum(self.count, self.data.size),
        )
        return jax.tree_map(
            lambda ary: ary[idxs, ...],
            self.data,
        ), self.returns[idxs]


class TrainState(NamedTuple):
    params: eqx.Module
    target_params: eqx.Module
    opt_state: optax.OptState
    buffer: Buffer


def train_step(
    carry: TrainState,
    x: tuple[Integer[Array, ""], rand.PRNGKey],
    collect_transitions: Callable[[rand.PRNGKey, eqx.Module, int], Transition],
    optim: optax.GradientTransformation,
    model_static: eqx.Module,
) -> TrainState:
    step, rng = x
    rng_collect, rng_buffer, rng_eval = rand.split(rng, 3)

    # collect transitions
    transitions = collect_transitions(
        rng_collect, carry.params, config.steps_per_update
    )

    target_model = eqx.combine(carry.target_params, model_static)
    _, target_values = jax.vmap(target_model)(transitions.obs)
    n_step_returns = compute_n_step_returns(
        transitions, target_values, config.bootstrap_after
    )
    buffer = carry.buffer.add_batch(transitions, n_step_returns)

    loss, grads = jax.value_and_grad(loss_fn)(
        carry.params,
        *buffer.sample(rng_buffer, config.update_batch_size),
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

    return TrainState(params, target_params, opt_state, buffer), loss


def make_env(env_name: str):
    env, env_params = gymnax.make(env_name)
    if env_name in ["Catch-bsuite"]:
        env = FlattenObservationWrapper(env)
    return env, env_params


@jax.jit
def train(rng):
    rng_model, rng_steps = rand.split(rng)

    env, env_params = make_env(config.env_name)

    params, model_static = get_model_pv(rng_model, env, env_params)

    # optimization
    optim = optax.adamw(
        optax.linear_schedule(config.lr_init, config.lr_end, config.update_steps)
    )
    opt_state = optim.init(params)

    # training loop
    train_step_ = partial(
        train_step,
        collect_transitions=partial(
            collect_transitions_mcts,
            model_pv_static=model_static,
            env=env,
            env_params=env_params,
            batch_dim=config.mcts_batch_dim,
            num_mcts_simulations=config.num_mcts_simulations,
        ),
        optim=optim,
        model_static=model_static,
    )
    init_carry = TrainState(
        params, params, opt_state, Buffer.init(env, env_params, config.buffer_size)
    )
    carry, losses = jax.lax.scan(
        train_step_,
        init_carry,
        (jnp.arange(config.update_steps), rand.split(rng_steps, config.update_steps)),
    )
    return carry, losses


class ModelPV(eqx.Module):
    base: eqx.nn.MLP
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear

    def __init__(
        self, in_size: int, n_actions: int, width_size: int = 128, *, key: rand.PRNGKey
    ):
        rng_base, rng_policy, rng_value = rand.split(key, 3)
        self.base = eqx.nn.MLP(
            in_size=in_size,
            out_size=width_size,
            width_size=width_size,
            depth=2,
            key=rng_base,
        )
        self.policy_head = eqx.nn.Linear(width_size, n_actions, key=rng_policy)
        self.value_head = eqx.nn.Linear(width_size, "scalar", key=rng_value)

    def __call__(self, obs: Float[Array, "..."], *, key: Optional[rand.PRNGKey] = None):
        base = self.base(obs)
        return self.policy_head(base), self.value_head(base)


def get_model_pv(rng: rand.PRNGKey, env: Environment, env_params: gymnax.EnvParams):
    in_size = math.prod(env.observation_space(env_params).shape)
    n_actions = env.action_space(env_params).n
    model = ModelPV(in_size, n_actions, key=rng)
    params, model_static = eqx.partition(model, eqx.is_inexact_array)
    return params, model_static


# %%
@partial(jax.jit, static_argnames=("env", "env_params", "num_transitions"))
def visualize_(
    rng: rand.PRNGKey,
    env: Environment,
    env_params: gymnax.EnvParams,
    params: eqx.Module,
    num_transitions: int,
):
    rng_model, rng_reset, rng_collect = rand.split(rng, 3)

    _, model_static = get_model_pv(rng_model, env, env_params)

    _, env_state = env.reset(rng_reset, env_params)
    transitions = collect_transitions_mcts(
        rng_collect,
        params,
        num_transitions,
        model_pv_static=model_static,
        env=env,
        env_params=env_params,
        batch_dim=config.mcts_batch_dim,
        num_mcts_simulations=config.num_mcts_simulations,
    )
    env_states = jax.tree.transpose(
        jax.tree.structure(env_state),
        None,
        jax.tree.map(lambda leaf: list(leaf), transitions.env_state),
    )
    return transitions, env_states


def visualize(rng, params: eqx.Module, fname: str, n: int) -> Transition:
    env, env_params = make_env(config.env_name)
    transitions, env_states = visualize_(rng, env, env_params, params, n)
    vis = Visualizer(env, env_params, env_states, jnp.cumsum(transitions.reward))
    vis.animate(fname)
    plt.close(vis.fig)
    return transitions


# %%
if __name__ == "__main__":
    rng = rand.PRNGKey(0)

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
        update_batch_size=400,
        bootstrap_after=1,
        steps_per_update=240,
        collect_workers=12,
        update_steps=5_000,
        updates_per_target=20,
        target_update_size=0.9,
        updates_per_eval=1000,
        eval_steps=200,
        discount=0.99,
        mcts_batch_dim=8,
        num_mcts_simulations=48,
    )

    wandb.init(
        project="jax-rl",
        config=config._asdict(),
    )

    carry, losses = jax.block_until_ready(train(rng_train))

# %%
