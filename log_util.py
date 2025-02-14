# jax
import jax
import jax.numpy as jnp
import jax.random as jr
import mctx

# visualization
import pygraphviz
import wandb

# util
import inspect
import sys
import yaml

# typing
from jaxtyping import Bool, Float, Array, Key, PyTree
from typing import Annotated as Batched, Any, Callable, NamedTuple, TypeVar
from gymnax.environments.bsuite.catch import EnvState as CatchEnvState


def get_norm_data(tree: PyTree[Float[Array, "..."]], prefix: str):
    """For logging norms of pytree leaves."""
    return {
        f"{prefix}{jax.tree_util.keystr(keys)}": jnp.sqrt(jnp.mean(jnp.square(ary)))
        for keys, ary in jax.tree.leaves_with_path(tree)
        if ary is not None
    }


def replace(f: NamedTuple, key: str, value: Any):
    """Replace nested namedtuple fields."""
    idx = key.find(".")
    if idx == -1:
        return f._replace(**{key: value})

    key, rest = key[:idx], key[idx + 1 :]
    prop = getattr(f, key)
    prop = replace(prop, rest, value)
    return f._replace(**{key: prop})


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


def exec_callback(f: Callable | None, *, cond=True):
    """A decorator for executing callbacks that applies the default arguments."""
    signature = inspect.signature(f)
    signature.apply_defaults()
    jax.debug.callback(
        f,
        *signature.args,
        **signature.kwargs,
    )
    return f


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def exec_loop(init: Carry, length: int, key: Key[Array, ""], cond: Bool[Array, ""] | None = None):
    """Scan the decorated function for `length` steps.

    The motivation is that loops are easier to read
    when the target and iter are in front.
    """

    def decorator(
        f: Callable[[Carry, X], tuple[Carry, Y]],
    ) -> tuple[Carry, Batched[Y, "length"]]:
        if cond is None:
            return jax.lax.scan(f, init, jr.split(key, length))
        else:
            return jax.lax.cond(
                cond,
                lambda init, key: jax.lax.scan(f, init, jr.split(key, length)),
                lambda init, key: (init, None),
                init,
                key,
            )

    return decorator


def visualize_catch(
    obs_shape: tuple[int, int],
    env_states: Batched[CatchEnvState, "horizon"],
    maps: Float[Array, "horizon obs_size"] | None = None,
) -> Float[Array, "horizon channel height width"]:
    """Turn a sequence of Catch environment states into a wandb.Video matrix."""

    horizon = env_states.time.size
    horizon_grid = jnp.arange(horizon)
    video_shape = (horizon, 3, *obs_shape)

    if maps is not None:
        # rescale to [0, 255]
        maps_min = jnp.min(maps, keepdims=True)
        maps = 255 * (maps - maps_min) / (jnp.max(maps, keepdims=True) - maps_min)
        maps = maps.astype(jnp.uint8)
        maps = jnp.reshape(maps, (horizon, 1, *obs_shape))
        maps = jnp.broadcast_to(maps, video_shape)
    else:
        maps = jnp.full(video_shape, 255, dtype=jnp.uint8)

    video = (
        maps.at[horizon_grid, :, env_states.ball_y, env_states.ball_x]
        .set(jnp.array([255, 0, 0], dtype=jnp.uint8))
        .at[horizon_grid, :, env_states.paddle_y, env_states.paddle_x]
        .set(jnp.array([0, 255, 0], dtype=jnp.uint8))
    )

    return video


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
