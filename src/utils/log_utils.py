import functools as ft
import inspect
import sys
from typing import TYPE_CHECKING, Callable, TypeVar
from typing import Annotated as Batched

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import yaml
from beartype import beartype as typechecker
from chex import dataclass
from jaxtyping import Array, Bool, Float, Integer, Key, PyTree, jaxtyped

import wandb

if False and not TYPE_CHECKING:  # runtime check dataclasses
    _dataclass = dataclass

    def dataclass(cls=None, /, **kwargs):
        """Typecheck all dataclass fields."""
        if cls is None:
            return ft.partial(dataclass, **kwargs)
        return typecheck(_dataclass(cls, **kwargs))


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
    """Log a dict of values to wandb (or terminal if wandb is disabled)."""

    def log(data: dict[str, Float[Array, ""]]):
        data = jax.tree.map(lambda x: x.item(), data)
        if wandb.run is not None:
            wandb.log(data)
        else:
            yaml.safe_dump(data, sys.stdout)

    jax.debug.callback(log, data)


def tree_slice(
    tree: PyTree[Array], at: int | tuple[int | Integer[Array, ""], ...] | slice
) -> PyTree[Array]:
    """Slice each leaf of a pytree at the given index or slice."""
    return jax.tree.map(lambda x: x[at], tree)


def exec_callback(f: Callable):
    """A decorator for executing callbacks that applies the default arguments."""
    bound = inspect.signature(f).bind()
    bound.apply_defaults()
    jax.debug.callback(f, *bound.args, **bound.kwargs)
    return f


def scale_gradient(x: Float[Array, " n"], factor: float):
    return x * factor + jax.lax.stop_gradient((1 - factor) * x)


Carry = TypeVar("Carry")
Y = TypeVar("Y")


def exec_loop(length: int, *, cond: Bool[Array, ""] | None = None):
    """Scan the decorated function for `length` steps.

    The motivation is that loops are easier to read
    when the target and iter are in front.
    """

    def decorator(
        f: Callable[[Carry, Key[Array, ""]], tuple[Carry, Y]],
    ) -> tuple[Carry, Batched[Y, " length"]]:
        # read init and rng key from default arguments
        signature = inspect.signature(f).parameters
        init, key = map(lambda x: x.default, signature.values())

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


def print_bytes(x) -> None:
    eqx.tree_pprint(jax.tree.map(lambda x: x.nbytes if eqx.is_array(x) else None, x))
