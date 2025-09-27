from enum import Enum, auto

import jax.numpy as jnp


class CellType(Enum):
    EMPTY = auto(), ".", 0.0
    START = auto(), "S", 0.0
    GOAL = auto(), "G", 2.0
    WALL = auto(), "W", 0.0
    PIT = auto(), "P", -1.0

    def __init__(self, i, letter, reward) -> None:
        self.index, self.letter, self.reward = i-1, letter, reward

cell_to_reward = jnp.asarray([cell.reward for cell in CellType])
letter_to_cell = {cell.letter: i for i, cell in enumerate(CellType)}

SIMPLE_MAP = """
..G..
.PWW.
S....
"""

LARGER_MAP = """
...G.
..PW.
..PW.
S....
"""
