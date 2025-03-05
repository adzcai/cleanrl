import dataclasses as dc
import functools as ft
import inspect
import sys
from typing import TYPE_CHECKING
from typing import Annotated as Batched
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
import mctx
from omegaconf import DictConfig, ListConfig, OmegaConf
import yaml
from beartype import beartype as typechecker
from chex import dataclass
from jaxtyping import Array, Bool, Float, Key, PyTree, jaxtyped

import wandb

T = TypeVar("T")

try:
    import pygraphviz
except ImportError:
    pass


if not TYPE_CHECKING:  # runtime check dataclasses
    _dataclass = dataclass

    def dataclass(cls=None, /, **kwargs):
        """Typecheck all dataclass fields."""
        if cls is None:
            return ft.partial(dataclass, **kwargs)
        return jaxtyped(typechecker=typechecker)(_dataclass(cls, **kwargs))


def typecheck(f):
    return jaxtyped(f, typechecker=typechecker)


def get_norm_data(tree: PyTree[Float[Array, " ..."]], prefix: str):
    """For logging root-mean-squares of pytree leaves."""
    return {
        f"{prefix}{jax.tree_util.keystr(keys)}": jnp.sqrt(jnp.mean(jnp.square(ary)))
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


def exec_callback(f: Callable):
    """A decorator for executing callbacks that applies the default arguments."""
    bound = inspect.signature(f).bind()
    bound.apply_defaults()
    jax.debug.callback(f, *bound.args, **bound.kwargs)
    return f


def scale_gradient(x: Float[Array, "n"], factor: float):
    return x * factor + jax.lax.stop_gradient((1 - factor) * x)


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def roll_into_matrix(ary: Float[Array, " n *size"]) -> Float[Array, " n n *size"]:
    return jax.vmap(jnp.roll, in_axes=(None, 0, None))(ary, -jnp.arange(ary.shape[0]), 0)


def exec_loop(init: Carry, length: int, key: Key[Array, ""], cond: Bool[Array, ""] | None = None):
    """Scan the decorated function for `length` steps.

    The motivation is that loops are easier to read
    when the target and iter are in front.
    """

    def decorator(
        f: Callable[[Carry, X], tuple[Carry, Y]],
    ) -> tuple[Carry, Batched[Y, " length"]]:
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


def convert_tree_to_graph(
    tree: mctx.Tree, action_labels: list[str] | None = None, batch_index: int = 0
) -> "pygraphviz.AGraph":
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


def dict_to_dataclass(cls: type[T], obj: dict) -> T:
    """Cast a dictionary to a dataclass instance.

    Args:
        cls (type[T]): The dataclass to cast to.
        obj (dict): The dictionary matching the dataclass fields.

    Raises:
        ValueError: If any required arguments are missing.

    Returns:
        T: The dataclass instance.
    """
    out = {}
    for field in dc.fields(cls):
        if field.name in obj:
            value = obj[field.name]
        elif field.default is not dc.MISSING:
            value = field.default
        elif field.default_factory is not dc.MISSING:
            value = field.default_factory()
        else:
            raise ValueError(f"Field {field.name} missing when constructing {cls}")
        if dc.is_dataclass(field.type):
            value = dict_to_dataclass(field.type, value)
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_object(value)
        out[field.name] = value
    return cls(**out)
