import inspect
import sys
from typing import Annotated as Batched
from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb
import yaml
from beartype import beartype
from jaxtyping import Array, Bool, Float, Key, PyTree, jaxtyped


def get_norm_data(tree: PyTree[Float[Array, " ..."]], prefix: str):
    """For logging root-mean-squares of pytree leaves."""
    return {
        f"{prefix}{jax.tree_util.keystr(keys)}": jnp.sqrt(jnp.mean(jnp.square(ary)))
        for keys, ary in jax.tree.leaves_with_path(tree)
        if ary is not None
    }


def log_values(data: dict[str, Float[Array, ""]]):
    """Log a dict of values to wandb (or terminal if wandb is disabled)."""

    @exec_callback
    def log(data=data):
        data = jax.tree.map(lambda x: x.item(), data)
        if wandb.run is None or wandb.run.disabled:
            yaml.safe_dump(data, sys.stdout)
        else:
            wandb.log(data)


def exec_callback(f: Callable):
    """A decorator for executing callbacks that applies the default arguments."""
    bound = inspect.signature(f).bind()
    bound.apply_defaults()
    jax.debug.callback(f, *bound.args, **bound.kwargs)
    return f


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


def typecheck(f):
    return jaxtyped(f, typechecker=beartype)
