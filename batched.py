import functools as ft
import typing

import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

_T = typing.TypeVar("_T")


class _FakeBatched(typing.Generic[_T]):
    pass


_FakeBatched.__name__ = "Batched"
_FakeBatched.__qualname__ = "Batched"
_FakeBatched.__module__ = "builtins"


class _MetaBatched(type):
    def __instancecheck__(cls, obj) -> bool:
        @jaxtyped(typechecker=typechecker)
        def accepts_basetype(x: cls.basetype): ...

        def is_basetype(x):
            try:
                accepts_basetype(x)
            except TypeError:
                return False
            else:
                return True

        return jnp.all(jax.vmap(is_basetype)(obj))

    @ft.lru_cache(maxsize=None)
    def __getitem__(cls, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "Type annotations must now include both an "
                "array type and a shape. For example `Batched[Float[Array, 'foo'], 'bar']`."
            )

        basetype_, structure_ = item

        if not isinstance(structure_, str):
            raise ValueError(
                "The structure annotation `struct` in "
                "`Batched[leaftype, struct]` must be be a string, "
                f"e.g. `Batched[leaftype, 'T']`. Got '{structure_}'."
            )

        structure_parts = structure_.strip().split()

        if not all(s.isidentifier() for s in structure_parts):
            raise ValueError(
                f"The structure annotation `struct` in `Batched[leaftype, struct]` must be a valid Python identifier. "
                f"Got '{structure_}', which is not a valid Python identifier."
            )

        name = str(_FakeBatched[basetype_])

        class X(Batched):
            basetype = basetype_
            structure = structure_

        X.__name__ = name
        X.__qualname__ = name
        if getattr(typing, "GENERATING_DOCUMENTATION", False):
            X.__module__ = "builtins"
        else:
            X.__module__ = "jaxtyping"
        return X


if typing.TYPE_CHECKING:
    from typing import Annotated as Batched
else:
    # Can't do `class Batched(Generic[_T]): ...` because we need to override the
    # instancecheck for Batched[foo], but subclassing
    # `type(Generic[int])`, i.e. `typing._GenericAlias` is disallowed.
    Batched = _MetaBatched("Batched", (), {})
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        Batched.__module__ = "builtins"
    else:
        Batched.__module__ = "jaxtyping"
    Batched.__doc__ = """Represents a batch of some other PyTree."""
