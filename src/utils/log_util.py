import dataclasses as dc
import functools as ft
import inspect
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from typing import Annotated as Batched

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import yaml
from beartype import beartype as typechecker
from chex import dataclass
from jaxtyping import Array, Bool, Float, Key, PyTree, jaxtyped
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    TDataclass = TypeVar("TDataclass", bound="DataclassInstance")
else:
    TDataclass = TypeVar("TDataclass")

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


def tree_slice(tree: PyTree[Array], at: int | tuple[int, ...] | slice) -> PyTree[Array]:
    """Slice each leaf of a pytree at the given index or slice."""
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
Y = TypeVar("Y")


def roll_into_matrix(ary: Float[Array, " n *size"]) -> Float[Array, " n n *size"]:
    return jax.vmap(jnp.roll, in_axes=(None, 0, None))(
        ary, -jnp.arange(ary.shape[0]), 0
    )


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


def dict_to_dataclass(cls: type[TDataclass], obj: Mapping[str, Any]) -> TDataclass:
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
        if dc.is_dataclass(tp := field.type):
            value = dict_to_dataclass(tp, value)  # type: ignore
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_object(value)
        out[field.name] = value
    return cls(**out)


def print_bytes(x) -> None:
    eqx.tree_pprint(jax.tree.map(lambda x: x.nbytes if eqx.is_array(x) else None, x))
