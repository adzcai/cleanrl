import dataclasses as dc
from typing import TYPE_CHECKING, Generic

from utils.structures import TDataclass, dataclass


@dataclass
class Wrapper(Generic[TDataclass]):
    """Base dataclass for assigning additional properties to an object.

    Delegates property lookups to the inner object.

    The wrapped object must be a TDataclass to enable field replacement.
    """

    _inner: TDataclass

    @classmethod
    def overwrite(cls, obj, **kwargs):
        """Replace the properties of the inner object"""
        return dc.replace(obj, **kwargs, _inner=obj)

    if not TYPE_CHECKING:  # better static type hints

        def __getattr__(self, name):
            return getattr(self._inner, name)
