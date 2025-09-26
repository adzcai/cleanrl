from jaxtyping import Float, Array
import jax.numpy as jnp
from enum import Enum
from typing import ClassVar


class CellType(Enum):
    START = "S", 0.0
    WALL = "W", 0.0
    GOAL = "G", 2.0
    PIT = "P", -1.0
    EMPTY = "0", 0.0

    @property
    def char(self):
        return self.value[0]

    @property
    def reward(self):
        return self.value[1]

    @property
    def index(self):
        return self.index_of[self.char]

    get_reward: ClassVar[Float[Array, " N"]]
    index_of: ClassVar[dict[str, int]]


CellType.get_reward = jnp.asarray([value.reward for value in CellType.__members__.values()])
CellType.index_of = {char.value[0]: i for i, char in enumerate(CellType.__members__.values())}

SIMPLE_MAP = """
00G00
0PWW0
S0000
"""
