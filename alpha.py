# %%
import tempfile
import jax
import jax.numpy as jnp
import jax.random as rand
import chex

import gymnax
from gymnax.environments.environment import Environment
from gymnax.wrappers import FlattenObservationWrapper
from gymnax.visualize import Visualizer
import pygraphviz
import wandb

import matplotlib
import matplotlib.pyplot as plt

import equinox as eqx
import optax
import mctx

from typing import NamedTuple, Optional
from collections.abc import Callable, Sequence
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

    network_width_size: int

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
    mcts_max_depth: int


# %%
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
    """For the recurrent function"""

    obs: Float[Array, "..."]
    env_state: gymnax.EnvState


# %%
def mcts_recurrent_fn(
    params_pv: eqx.Module,
    rng: rand.PRNGKey,
    action_batch: Integer[Array, " b"],
    rec_batch: gymnax.EnvState,
    model_pv_static: eqx.Module,
    env: Environment,
    env_params: gymnax.EnvParams,
) -> tuple[mctx.RecurrentFnOutput, gymnax.EnvState]:
    """Recurrent function (world model) for mcts."""
    model_pv = eqx.combine(params_pv, model_pv_static)

    def step(rng: rand.PRNGKey, action: Integer[Array, ""], env_state: gymnax.EnvState):
        next_obs, next_env_state, reward, done, info = env.step(
            rng, env_state, action, env_params
        )
        logits, value = model_pv(next_obs)
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(done, 0.0, config.discount),
            prior_logits=logits,
            value=value,
        ), next_env_state

    return jax.vmap(step)(rand.split(rng, action_batch.size), action_batch, rec_batch)


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
        embedding=env_state,
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
        max_depth=config.mcts_max_depth,
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


def collect_transitions_model(
    rng: rand.PRNGKey,
    params_pv: eqx.Module,
    num_transitions: int,
    model_pv_static: eqx.Module,
    env: Environment,
    env_params: gymnax.EnvParams,
):
    model = eqx.combine(params_pv, model_pv_static)

    def step(carry: ObsWithGymnaxState, rng: rand.PRNGKey):
        logits, _ = model(carry.obs)
        action = jnp.argmax(logits)

        next_obs, next_env_state, reward, done, info = env.step(
            rng, carry.env_state, action, env_params
        )

        return ObsWithGymnaxState(next_obs, next_env_state), Transition(
            obs=carry.obs,
            env_state=carry.env_state,
            logits=logits,
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


def compute_returns(txns: Transition, discount: float):
    def step(return_to_go: float, txn: Transition):
        return_to_go = txn.reward + jax.lax.select(
            txn.done,
            0.0,
            discount * return_to_go,
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
    txns: Transition,
    values: Float[Array, " b"],
    bootstrap_after: int,
    discount: float,
):
    returns = compute_returns(txns, discount)

    def loop_n_step_returns(
        steps_to_terminal: int, x: tuple[bool, float, float, float]
    ):
        done, return_to_go, future_return, future_value = x
        steps_to_terminal = jnp.where(done, 0, steps_to_terminal + 1)
        n_step_return = jnp.where(
            steps_to_terminal < bootstrap_after,
            return_to_go,
            return_to_go + discount**bootstrap_after * (future_value - future_return),
        )
        return steps_to_terminal, n_step_return

    bootstrap_after = jnp.minimum(bootstrap_after, txns.size)

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
    labels = jax.nn.softmax(txns.logits, axis=-1)
    policy_loss = optax.softmax_cross_entropy(logits, labels)
    return jnp.mean(value_loss) + jnp.mean(policy_loss)


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
        transitions,
        target_values,
        config.bootstrap_after,
        config.discount,
    )
    buffer = carry.buffer.add_batch(transitions, n_step_returns)

    loss, grads = jax.value_and_grad(loss_fn)(
        carry.params,
        *buffer.sample(rng_buffer, config.update_batch_size),
        model_static,
    )
    updates, opt_state = optim.update(grads, carry.opt_state, carry.params)
    params = optax.apply_updates(carry.params, updates)

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

    def eval_model(rng: rand.PRNGKey, params: eqx.Module):
        with tempfile.NamedTemporaryFile(suffix=".gif") as f_model:
            with tempfile.NamedTemporaryFile(suffix=".gif") as f_mcts:
                with tempfile.NamedTemporaryFile(suffix=".png") as f_search:
                    txns_model, txns_mcts, tree = visualize(
                        rng,
                        params,
                        model_static,
                        f_model.name,
                        f_mcts.name,
                        f_search.name,
                        config.eval_steps,
                    )
                    wandb.log(
                        {
                            "eval/model/rollout": wandb.Image(f_model.name),
                            "eval/model/rewards": jnp.sum(txns_model.reward),
                            "eval/mcts/rollout": wandb.Image(f_model.name),
                            "eval/mcts/rewards": jnp.sum(txns_mcts.reward),
                            "eval/mcts/search_tree": wandb.Image(f_search.name),
                        }
                    )

    jax.lax.cond(
        step % config.updates_per_eval == 0,
        partial(jax.debug.callback, eval_model),
        lambda *_: None,
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

    params, model_static = get_model_pv(rng_model, config.network_width_size, env, env_params)

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
        self, in_size: int, n_actions: int, width_size: int, *, key: rand.PRNGKey
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


def get_model_pv(rng: rand.PRNGKey, width_size: int, env: Environment, env_params: gymnax.EnvParams):
    in_size = math.prod(env.observation_space(env_params).shape)
    n_actions = env.action_space(env_params).n
    model = ModelPV(in_size, n_actions, width_size, key=rng)
    params, model_static = eqx.partition(model, eqx.is_inexact_array)
    return params, model_static


# %%
@partial(
    jax.jit,
    static_argnames=("env", "env_params", "model_static", "num_transitions", "search"),
)
def visualize_model_behavior(
    rng: rand.PRNGKey,
    env: Environment,
    env_params: gymnax.EnvParams,
    params: eqx.Module,
    model_static: eqx.Module,
    num_transitions: int,
    search: bool,
):
    rng_reset, rng_collect = rand.split(rng)

    if search:
        transitions = collect_transitions_mcts(
            rng_collect,
            params,
            num_transitions,
            model_static,
            env,
            env_params,
            batch_dim=config.mcts_batch_dim,
            num_mcts_simulations=config.num_mcts_simulations,
        )
    else:
        transitions = collect_transitions_model(
            rng_collect,
            params,
            num_transitions,
            model_pv_static=model_static,
            env=env,
            env_params=env_params,
        )

    obs, env_state = env.reset(rng_reset, env_params)
    env_states = jax.tree.transpose(
        jax.tree.structure(env_state),
        None,
        jax.tree.map(lambda leaf: list(leaf), transitions.env_state),
    )
    return transitions, env_states


def visualize(
    rng: rand.PRNGKey,
    params: eqx.Module,
    model_static: eqx.Module,
    fname_model: str,
    fname_mcts: str,
    fname_tree: str,
    n: int,
) -> Transition:
    rng_model, rng_mcts, rng_reset, rng_tree = rand.split(rng, 4)

    env, env_params = make_env(config.env_name)
    transitions_model, env_states_model = visualize_model_behavior(
        rng_model, env, env_params, params, model_static, n, search=False
    )
    vis = Visualizer(
        env, env_params, env_states_model, jnp.cumsum(transitions_model.reward)
    )
    vis.animate(fname_model)
    plt.close(vis.fig)

    transitions_mcts, env_states_mcts = visualize_model_behavior(
        rng_mcts, env, env_params, params, model_static, n, search=True
    )
    vis = Visualizer(
        env, env_params, env_states_mcts, jnp.cumsum(transitions_mcts.reward)
    )
    vis.animate(fname_mcts)
    plt.close(vis.fig)

    obs, env_state = env.reset(rng_reset, env_params)
    tree = choose_action_mcts(
        rng_tree,
        params,
        obs,
        env_state,
        model_static,
        env,
        env_params,
        batch_dim=config.mcts_batch_dim,
        num_simulations=config.num_mcts_simulations,
    ).search_tree
    graph = convert_tree_to_graph(tree)
    graph.draw(fname_tree, prog="dot")

    return (
        transitions_model,
        transitions_mcts,
        tree,
    )


# %%
def convert_tree_to_graph(
    tree: mctx.Tree, action_labels: Optional[Sequence[str]] = None, batch_index: int = 0
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.

    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.

    Returns:
      A Graphviz graph representation of `tree`.
    """
    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = range(tree.num_actions)
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}."
        )

    def node_to_str(node_i, reward=0, discount=1):
        ball_x = tree.embeddings.ball_x[batch_index, node_i].item()
        ball_y = tree.embeddings.ball_y[batch_index, node_i].item()
        paddle_y = tree.embeddings.paddle_y[batch_index, node_i].item()
        paddle_x = tree.embeddings.paddle_x[batch_index, node_i].item()
        return (
            f"{node_i}\n"
            f"Ball: {(ball_y, ball_x)}\n"
            f"Paddle: {(paddle_y, paddle_x)}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"{action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
            f"p: {probs[a_i]:.2f}\n"
        )

    graph = pygraphviz.AGraph(directed=True)

    # Add root
    graph.add_node(0, label=node_to_str(node_i=0), color="green")
    # Add all other nodes and connect them up.
    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            children_i = tree.children_index[batch_index, node_i, a_i]
            if children_i >= 0:
                graph.add_node(
                    children_i,
                    label=node_to_str(
                        node_i=children_i,
                        reward=tree.children_rewards[batch_index, node_i, a_i],
                        discount=tree.children_discounts[batch_index, node_i, a_i],
                    ),
                    color="red",
                )
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    return graph


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
        network_width_size=32,
        buffer_size=1000,
        update_batch_size=400,
        bootstrap_after=10,
        steps_per_update=240,
        collect_workers=12,
        update_steps=5_000,
        updates_per_target=20,
        target_update_size=0.9,
        updates_per_eval=1000,
        eval_steps=200,
        discount=0.99,
        mcts_batch_dim=1,
        num_mcts_simulations=48,
        mcts_max_depth=64,
    )

    with wandb.init(
        project="jax-rl",
        config=config._asdict(),
    ) as run:
        carry, losses = jax.block_until_ready(train(rng_train))

# %%
