from functools import cached_property
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.table import Table
from matplotlib.patches import Circle
from typing import ClassVar
import distrax
import jax.numpy as jnp
from jaxtyping import Shaped, Integer, UInt8, Float, Array, Key
import equinox as eqx
from jax import vmap
from jax.lax import scan

from ilx.maps import CellType


class MDP(eqx.Module):
    d0: Shaped[distrax.Categorical, " S"]
    P: Shaped[distrax.Categorical, "S A S"]
    R: Float[Array, "S A S"]
    γ: float

    @cached_property
    def R_(self):
        return jnp.einsum("sap,sap->sa", self.P.probs, self.R)


class GridEnv(MDP):
    action_map: ClassVar[Integer[Array, "A 2"]] = jnp.asarray(
        [(-1, 0), (0, 1), (1, 0), (0, -1)]
    )
    char_grid: UInt8[Array, "H W"]
    goal_cell: UInt8[Array, " 2"]

    def __init__(self, map_text: str, γ: float) -> None:
        self.char_grid = jnp.asarray(
            [
                [CellType.index_of[char] for char in line]
                for line in map_text.strip().splitlines()
            ]
        )
        init_state = jnp.argwhere(
            self.char_grid.ravel() == CellType.START.index, size=1
        )
        (self.goal_cell,) = jnp.argwhere(self.char_grid == CellType.GOAL.index, size=1)
        s, a = jnp.mgrid[:self.S, :self.A]
        s_, r_ = vmap(vmap(self._step))(s, a)
        P = jnp.zeros((self.S, self.A, self.S)).at[s, a, s_].set(1)  # deterministic
        R = jnp.zeros((self.S, self.A, self.S)).at[s, a, s_].set(r_)

        super().__init__(
            d0=distrax.Categorical(probs=jnp.zeros(self.S).at[init_state].set(1)),
            P=distrax.Categorical(probs=P),
            R=R,
            γ=γ,
        )

    @property
    def S(self) -> int:
        return self.char_grid.size

    @property
    def A(self) -> int:
        return 4

    @property
    def height(self) -> int:
        return self.char_grid.shape[0]

    @property
    def width(self) -> int:
        return self.char_grid.shape[1]

    def _step(self, s: int, a: int):
        r, c = self.get_cell(s)
        dr, dc = self.action_map[a]
        r_ = jnp.clip(r + dr, 0, self.height - 1)
        c_ = jnp.clip(c + dc, 0, self.width - 1)
        index_, obstacle_ = self.is_obstacle(r_, c_)
        s_ = jnp.where(obstacle_, s, self.get_state(r_, c_))
        reward_ = jnp.where(obstacle_, 0, CellType.get_reward[index_])
        return s_, reward_
    
    def step(self, s: int, a: int, *, key: Key[Array, ""]):
        s_ = self.P[s, a].sample(seed=key)
        return s_, self.R[s, a, s_]

    def is_obstacle(self, r: UInt8[Array, ""] | int, c: UInt8[Array, ""] | int):
        char = self.char_grid[r, c]
        return char, char == CellType.WALL.index

    def get_cell(self, s: int):
        return divmod(s, self.width)

    def get_state(self, row: UInt8[Array, ""] | int, col: UInt8[Array, ""] | int):
        return row * self.width + col

    def get_features(self, s: int, a: int):
        row, col = self.get_cell(s)
        dr, dc = self.action_map[a]
        goal_r, goal_c = self.goal_cell

        _, is_obstacle = self.is_obstacle(row + dr, col + dc)

        row /= self.height - 1
        col /= self.width - 1
        goal_r /= self.height - 1
        goal_c /= self.width - 1

        return row, col, dr, dc, row - goal_r, col - goal_c, is_obstacle
    
    def solve_value_iteration(self):
        def step(Q: Float[Array, "S A"]):
            V = jnp.max(Q, axis=1)
            return self.R_ + self.γ * jnp.einsum("sap,p->sa", self.P.probs, V)
        Q, _ = scan(lambda Q, _: (step(Q), Q), jnp.zeros((self.S, self.A)), length=20)
        return Q
    
    def stationary(self, π: Shaped[distrax.Categorical, "S A"]):
        # P(S' | S) = Σ P(S' | S, A) π(A | S)
        P = jnp.einsum("sap,sa->sp", self.P.probs, π.probs)
        d = jnp.linalg.solve(jnp.eye(self.S) - self.γ * P.T, (1-self.γ) * self.d0.probs)
        return distrax.Categorical(probs=d)

    def draw(self, V: Float[Array, " S"], π: distrax.Categorical, filename: str, title: str, d: distrax.Categorical):
        "Render environment"
        
        # Calculate figure size to maintain aspect ratio
        if self.width >= self.height:
            figsize = (8, 8 * self.height / self.width)
        else:
            figsize = (8 * self.width / self.height, 8)

        fig = plt.figure(frameon=False, figsize=figsize)
        ax = fig.add_subplot(111, aspect="equal")
        ax.set_axis_off()

        if title:
            ax.set_title(title)

        # Create table that fills the entire axes
        tb = Table(ax, loc='center')
        ax.add_table(tb)

        # get colors
        dots: list[tuple[int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                index, is_obstacle = self.is_obstacle(r, c)
                reward = CellType.get_reward[index]
                
                if is_obstacle:
                    color = "black"
                elif reward == 0:
                    color = "white"
                elif reward > 0:
                    color = "green"
                else:
                    color = "red"
                if not is_obstacle and (r, c) != tuple(self.goal_cell):
                    dots.append((r, c))
                # Each cell takes up 1/width by 1/height of the table
                cell_width = 1.0 / self.width
                cell_height = 1.0 / self.height
                tb.add_cell(r, c, cell_width, cell_height, facecolor=color)

        ax.figure.canvas.draw()  # need to run draw to define cell bboxes below.

        cells = tb.get_celld()
        viridis = cm.get_cmap("viridis")
        for r, c in dots:
            box = cells[r, c].properties()["bbox"]
            s = self.get_state(r, c)
            radius = 0.2 * jnp.sqrt(d.prob(s) * self.S).item() / max(self.width, self.height)
            # Fix circle center calculation - bbox has x0, y0, x1, y1 attributes
            center_x = (box.x0 + box.x1) / 2
            center_y = (box.y0 + box.y1) / 2
            circle = Circle((center_x, center_y), fc=viridis(V[s].item()), radius=radius, linewidth=0)
            ax.add_patch(circle)

            for a, (dr, dc) in enumerate(self.action_map):
                if (p := π[s].prob(a)) > 0:
                    ax.arrow(
                        center_x, center_y, dc / 40 * p, -dr / 40 * p, color="k", width=0.005 * p
                    )

        fig.tight_layout()
        fig.savefig(filename)
