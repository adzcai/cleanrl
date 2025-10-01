import math
from collections.abc import Collection
from typing import Union

import jax
import jax.numpy as jnp

from utils.structures import DataclassInstance, dataclass

SpecTree = Union["Array", Collection["SpecTree"], "DataclassInstance"]


@dataclass
class Array:
    """A specification for an array."""

    shape: tuple[int, ...]
    """Shape of the array."""
    dtype: jax.typing.DTypeLike
    """Data type of the array."""
    name: str = ""
    """Name of the array specification."""

    def asarray(self) -> jnp.ndarray:
        """Cast this to an array for type checking"""
        return self  # type: ignore


@dataclass
class BoundedArray(Array):
    minimum: int | float = -jnp.inf
    """Minimum value of the array."""
    maximum: int | float = jnp.inf
    """Maximum value of the array."""

    @classmethod
    def discrete(cls, num_values: int, shape=(), **kwargs) -> "BoundedArray":
        """Create a bounded array for discrete values."""
        return cls(
            shape=shape,
            dtype=jnp.int32,
            minimum=0,
            maximum=num_values - 1,
            **kwargs,
        )

    @property
    def num_values(self) -> int:
        """Number of integer values in the array."""
        return math.floor(self.maximum) - math.ceil(self.minimum) + 1
