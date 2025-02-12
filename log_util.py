import inspect
import sys
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
import mctx
import pygraphviz
import wandb
import yaml

from typing import Annotated as Batched, Callable
from gymnax.environments.bsuite.catch import EnvState as CatchEnvState


def get_norm_data(tree: PyTree[Float[Array, "..."]], prefix: str):
    """For logging norms of pytree leaves."""
    return {
        f"{prefix}{jax.tree_util.keystr(keys)}": jnp.linalg.norm(ary)
        for keys, ary in jax.tree.leaves_with_path(tree)
        if ary is not None
    }


def log_values(data: dict[str, Float[Array, ""]]):
    """Log a dict of values to wandb or terminal."""

    def log(data: dict[str, Float[Array, ""]]):
        data = jax.tree.map(lambda x: x.item(), data)
        if wandb.run is not None:
            wandb.log(data)
        else:
            yaml.safe_dump(data, sys.stdout)

    jax.debug.callback(log, data)


def tree_slice(tree: PyTree[Array], at: int):
    return jax.tree.map(lambda x: x[at], tree)


def exec_callback(f: Callable[[...], ...]):
    """A decorator for executing callbacks."""
    args, kwargs = [], {}
    for param in inspect.signature(f).parameters.values():
        if param.default == inspect.Parameter.empty:
            raise ValueError(f"All parameters of {f} must have default values")
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(param.default)
        else:
            kwargs[param.name] = param.default

    jax.debug.callback(
        f,
        *args,
        **kwargs,
    )

    return f


def visualize_catch(
    obs_shape: tuple[int, int],
    env_states: Batched[CatchEnvState, "num_envs horizon"],
    maps: Float[Array, "num_envs horizon obs_size"],
) -> Float[Array, "num_envs horizon channel height width"]:
    """Turn a sequence of Catch environment states into a wandb.Video matrix."""

    horizon, num_envs = env_states.time.shape
    horizon_grid, batch_grid = jnp.mgrid[:horizon, :num_envs]

    maps = jnp.reshape(
        jnp.astype(maps * 255.0, jnp.uint8),
        (horizon, num_envs, *obs_shape),
    )

    video = (
        jnp.empty((horizon, num_envs, 3, *obs_shape), dtype=jnp.uint8)
        .at[horizon_grid, batch_grid, 0, env_states.ball_y, env_states.ball_x]
        .set(255)
        .at[horizon_grid, batch_grid, 1, env_states.paddle_y, env_states.paddle_x]
        .set(255)
        .at[:, :, 2, :, :]
        .set(maps)
    )

    return jnp.swapaxes(video, 0, 1)


def convert_tree_to_graph(
    tree: mctx.Tree, action_labels: list[str] | None = None, batch_index: int = 0
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.

    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.

    Returns:
      A Graphviz graph representation of `tree`.
    """
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
        # ball_x = tree.embeddings.ball_x[batch_index, node_i].item()
        # ball_y = tree.embeddings.ball_y[batch_index, node_i].item()
        # paddle_y = tree.embeddings.paddle_y[batch_index, node_i].item()
        # paddle_x = tree.embeddings.paddle_x[batch_index, node_i].item()
        return (
            f"{node_i}\n"
            # f"Ball: {(ball_y, ball_x)}\n"
            # f"Paddle: {(paddle_y, paddle_x)}\n"
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
