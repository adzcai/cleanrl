from typing import ClassVar

import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import dataclass
from distrax import Categorical
from jax import lax, vmap
from jaxtyping import Array, Float, Integer, Key, Shaped, UInt8
from matplotlib.patches import Circle
from matplotlib.table import Table

from ilx.core.maps import CellType, cell_to_reward, letter_to_cell

Policy = Shaped[Categorical, "S A"]


@dataclass
class TabularMDP:
    d0: Shaped[Categorical, " S"]
    P: Shaped[Categorical, "S A S"]
    R: Float[Array, "S A S"]
    γ: float
    features: Float[Array, "S A D"]

    @property
    def S(self):
        return self.R.shape[0]

    @property
    def A(self):
        return self.R.shape[1]

    @property
    def D(self):
        return self.features.shape[2]

    def step(self, s: int, a: int, *, key: Key[Array, ""]):
        s_ = self.P[s, a].sample(seed=key)
        r_ = self.R[s, a, s_]
        return s_, r_

    def π_to_P(self, π: Policy):
        P = jnp.einsum("sap, sa -> sp", self.P.probs, π.probs)
        return Categorical(probs=P)

    def π_to_V(self, π: Policy):
        P = self.π_to_P(π)
        R = jnp.einsum("sap, sa, sap -> s", self.P.probs, π.probs, self.R)
        return jnp.linalg.solve(jnp.eye(self.S) - self.γ * P.probs, R)

    def V_to_Q(self, V: Float[Array, " S"]):
        return jnp.einsum(
            "sap, sap -> sa",
            self.P.probs,
            self.R + self.γ * V[jnp.newaxis, jnp.newaxis, :],
        )

    def value_iteration(self):
        def step(Q, _):
            V = jnp.max(Q, axis=-1)
            return self.V_to_Q(V), None

        return lax.scan(step, jnp.zeros((self.S, self.A)), length=10, unroll=True)[0]

    def π_to_stationary(self, π: Policy):
        d = jnp.linalg.solve(
            jnp.eye(self.S) - self.γ * self.π_to_P(π).probs.T,
            (1 - self.γ) * self.d0.probs,
        )
        return Categorical(probs=d)
    
    def π_to_μ(self, π: Policy):
        d = self.π_to_stationary(π)
        return Categorical(probs=jnp.ravel(d.probs[:, jnp.newaxis] * π.probs))

    def π_to_return(self, π: Policy):
        d = self.π_to_stationary(π)
        return jnp.einsum(
            "s, sap, sap, sa ->", d.probs, self.P.probs, self.R, π.probs
        ) / (1 - self.γ)

    def softmax_π(self, w: Float[Array, " D"]):
        return Categorical(logits=self.features @ w)


def Q_to_greedy(Q: Float[Array, "S A"]):
    index = jnp.arange(Q.shape[0]), Q.argmax(axis=1)
    return Categorical(probs=jnp.zeros_like(Q).at[index].set(1))


class GridEnv(TabularMDP):
    action_map: ClassVar[Integer[Array, "A 2"]] = jnp.asarray(
        [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]
    )
    grid: UInt8[Array, "rows cols"]
    state_to_pos: UInt8[Array, "S 2"]
    pos_to_state: UInt8[Array, "rows cols"]
    goal_pos: UInt8[Array, " 2"]
    features: Float[Array, "S A D"]

    def __init__(self, map_text: str, γ: float) -> None:
        self.grid = jnp.asarray(
            [
                [letter_to_cell[char] for char in line]
                for line in map_text.strip().splitlines()
            ]
        )
        wall_mask = self.grid != CellType.WALL.index
        self.state_to_pos = jnp.argwhere(wall_mask)
        self.pos_to_state = jnp.where(
            wall_mask.ravel(), jnp.cumsum(wall_mask) - 1, -1
        ).reshape(*self.bounds)

        S, A = len(self.state_to_pos), len(self.action_map)
        init_pos = jnp.argwhere(self.grid == CellType.START.index, size=1)[0]
        d0 = jnp.zeros(S).at[self.pos_to_state[*init_pos]].set(1)
        self.d0 = Categorical(probs=d0)
        self.goal_pos = jnp.argwhere(self.grid == CellType.GOAL.index, size=1)[0]
        self.γ = γ

        def step(s: int, a: int):
            pos_ = self.state_to_pos[s] + self.action_map[a]
            blocked_ = self.blocked(s, a)
            s_ = jnp.where(blocked_, s, self.pos_to_state[*pos_])
            r_ = jnp.where(blocked_, 0, cell_to_reward[self.grid[*pos_]])
            return s_, r_

        s, a = jnp.mgrid[:S, :A]
        s_, r_ = vmap(vmap(step))(s, a)
        self.P = Categorical(probs=jnp.zeros((S, A, S)).at[s, a, s_].set(1))
        self.R = jnp.zeros((S, A, S)).at[s, a, s_].set(r_)
        self.features = vmap(vmap(self.get_features))(s, a)


    @property
    def bounds(self):
        return jnp.asarray(self.grid.shape)

    def blocked(self, s: int, a: int):
        pos_ = self.state_to_pos[s] + self.action_map[a]
        return (
            (self.grid[*self.state_to_pos[s]] == CellType.GOAL.index)
            | (self.pos_to_state[*pos_] == -1)
            | jnp.any((pos_ < 0) | (pos_ >= self.bounds))
        )

    def gridify(self, state_map: Float[Array, " S"]):
        def extract(row: int, col: int):
            s = self.pos_to_state[row, col]
            return jnp.where(s == -1, jnp.nan, state_map[s])

        return vmap(vmap(extract))(*jnp.mgrid[: self.bounds[0], : self.bounds[1]])

    def get_features(self, s: int, a: int):
        pos, d = self.state_to_pos[s], self.action_map[a]
        blocked_ = self.blocked(s, a)
        diff = pos - self.goal_pos
        pos /= self.bounds - 1
        diff /= self.bounds - 1
        return jnp.asarray([1, *pos, *d, *diff, blocked_])

    def draw(self, π: Policy, filename: str, title: str):
        scale = max(*self.bounds)
        figsize = 8 * self.bounds[::-1] / scale
        fig = plt.figure(frameon=False, figsize=figsize)
        ax = fig.add_subplot(111, aspect="equal")
        ax.set_axis_off()
        ax.set_title(title)

        tb = Table(ax, loc="center")
        ax.add_table(tb)

        for row in range(self.bounds[0]):
            for col in range(self.bounds[1]):
                reward = cell_to_reward[self.grid[row, col]]

                if self.grid[row, col] == CellType.WALL.index:
                    color = "black"
                elif reward == 0:
                    color = "white"
                elif reward > 0:
                    color = "green"
                else:
                    color = "red"

                tb.add_cell(
                    row,
                    col,
                    *1 / self.bounds[::-1],
                    text=self.pos_to_state[row, col],
                    facecolor=color,
                )

        ax.figure.canvas.draw()  # need to run draw to define cell bboxes below.

        cells = tb.get_celld()
        cmap = plt.get_cmap("RdBu")
        V = self.π_to_V(π)
        d = self.π_to_stationary(π)
        for row in range(self.bounds[0]):
            for col in range(self.bounds[1]):
                box = cells[row, col].properties()["bbox"]
                s = self.pos_to_state[row, col]
                if s == -1:
                    continue
                radius = 0.2 * jnp.sqrt(d.prob(s) * self.S).item() / scale
                center: tuple[float, float] = (box.max + box.min) / 2
                circle = Circle(center, fc=cmap(V[s]), radius=radius, linewidth=0)
                ax.add_patch(circle)

                for a, dir in enumerate(self.action_map):
                    if (p := π[s].prob(a)) > 0:
                        dr, dc = dir * p / 40
                        ax.arrow(*center, dc, -dr, color="k", width=0.005 * p)

        fig.tight_layout()
        fig.savefig(filename)
