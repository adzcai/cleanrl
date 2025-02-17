"""A prioritized buffer for replay."""

from dataclasses import replace
from typing import TYPE_CHECKING, Generic, TypeVar, Annotated as Batched
import warnings

from log_util import tree_slice

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr

from jaxtyping import UInt, Key, Float, Array, PyTree


Experience = TypeVar("Experience", bound=PyTree[Array])


@dataclass(frozen=True)
class SumTreeState:
    nodes: Float[Array, " max_size"]
    max_priority: Float[Array, ""]  # new trajectories are assigned max priority


@dataclass(frozen=True)
class BufferState(Generic[Experience]):
    data: Batched[Experience, " batch_size max_time"]
    pos: UInt[Array, ""]
    priority_state: SumTreeState


@dataclass(frozen=True)
class Sample(Generic[Experience]):
    experience: Batched[Experience, " n horizon"]
    idx: UInt[Array, " n"]
    priority: Float[Array, " n"]


@dataclass(frozen=True)
class PrioritizedBuffer(Generic[Experience]):
    """A buffer that contains experiences and priority weights.

    We use "idx" to refer to the flattened index (across batches)
    and "pos" to refer to the time index.
    i.e. idx = batch * max_time + pos.
    """

    batch_size: int
    max_time: int
    horizon: int
    priority_tree: "SumTree"

    @classmethod
    def new(cls, batch_size: int, max_time: int, horizon: int):
        return cls(
            batch_size=batch_size,
            max_time=max_time,
            horizon=horizon,
            priority_tree=SumTree.new(batch_size * max_time),
        )

    def init(self, experience: Experience) -> BufferState[Experience]:
        """Initialize buffer from a dummy experience."""
        data = jax.tree.map(
            lambda x: jnp.empty_like(x, shape=(self.batch_size, self.max_time, *x.shape)),
            experience,
        )

        return BufferState(
            data=data,
            pos=jnp.uint(0),
            priority_state=self.priority_tree.init(),
        )

    def add(
        self,
        state: BufferState[Experience],
        experience: Batched[Experience, "batch_size horizon"],
    ) -> BufferState[Experience]:
        """Add a batch of trajectories.

        Args:
            state (BufferState[Experience]): The current buffer state.
            trajectories (Experience (batch_size, horizon)): The batch of trajectories to add.

        Returns:
            BufferState[Experience]: The updated buffer.
        """
        horizon = jax.tree.leaves(experience)[0].shape[1]
        pos_ary = (state.pos + jnp.arange(horizon)) % self.max_time
        data = jax.tree.map(
            lambda whole, new: whole.at[:, pos_ary].set(new), state.data, experience
        )
        state = replace(state, data=data, pos=state.pos + horizon)
        # enable the trajectories starting up to this point
        pos_enabled = jnp.maximum(0, state.pos - self.horizon - jnp.arange(horizon)) % self.max_time
        pos_enabled = jnp.broadcast_to(pos_enabled, (self.batch_size, horizon))
        idx_enabled = jax.vmap(self.pos_to_flat, in_axes=(1,))(pos_enabled)
        state = self.set_priorities(
            state,
            idx_enabled.ravel(),
            state.priority_state.max_priority[jnp.newaxis],
        )
        return state

    def pos_to_flat(self, pos: UInt[Array, " batch_size"]) -> UInt[Array, " batch_size"]:
        # e.g. suppose max_time is 4 and batch_size is 3
        # then pos_to_flat([2 1 3]) = [2 1 3] + [0 4 8] = [2 5 11]
        return pos + self.max_time * jnp.arange(self.batch_size)

    def set_priorities(
        self,
        state: BufferState[Experience],
        idx: UInt[Array, " batch_size"],
        priorities: Float[Array, " batch_size"],
    ) -> BufferState[Experience]:
        """Set priorities for a batch of trajectories."""
        priority_state = self.priority_tree.update(state.priority_state, idx, priorities)
        return replace(state, priority_state=priority_state)

    def sample(
        self, state: BufferState[Experience], key: Key[Array, ""]
    ) -> Batched[Sample[Experience], "horizon"]:
        """Sample a trajectory from the buffer."""
        idx, priority = self.priority_tree.sample(state.priority_state, key)
        batch, pos = divmod(idx, self.max_time)
        pos_ary = (pos + jnp.arange(self.horizon)) % self.max_time
        return Sample(
            experience=tree_slice(state.data, (batch, pos_ary)),
            idx=idx,
            priority=priority,
        )

    def calibrate(self, state: BufferState[Experience]):
        return replace(state, priority_state=self.priority_tree.recompute(state.priority_state))

    def num_available(self, state: BufferState[Experience]) -> int:
        """The number of possible trajectories (overlapping and across batch)."""
        size = self.batch_size * (jnp.minimum(state.pos, self.max_time) - self.horizon + 1)
        # could also count nonzero in self.priority_tree.nodes[(1 << self.priority_tree.depth) - 1 :]
        return size


@dataclass(frozen=True)
class SumTree:
    """Contains the priorities in a sum tree.

    Throughout the implementation,
    `index` indicates an index into the leaves and
    `idx` indicates an index into the flattened tree structure
    (offset by `self.leaf_idx`).
    """

    size: int
    depth: int

    @classmethod
    def new(cls, size: int):
        depth = int(jnp.ceil(jnp.log2(size)))  # root at zero
        return cls(
            size=size,
            depth=depth,
        )

    def init(self):
        nodes = jnp.zeros((1 << (self.depth + 1)) - 1)
        return SumTreeState(
            nodes=nodes,
            max_priority=jnp.float_(1.0),
        )

    def update(
        self, state: SumTreeState, indices: UInt[Array, " n"], priority: Float[Array, " n"]
    ) -> SumTreeState:
        """Set priorities at the given indices.

        Args:
            indices (UInt (n,)): The indices to update.
            priority (Float (n,)): The new values.

        Returns:
            SumTree: The updated tree.
        """
        # we only update using the first priority for each index
        # turn others into no-ops (add zero)
        unique, first = jnp.unique(
            indices, return_index=True, size=indices.size, fill_value=self.size
        )
        mask = unique < self.size
        idx = jnp.where(mask, self.leaf_idx + unique, 0)
        nodes = state.nodes
        delta = jnp.where(mask, priority[first] - nodes[idx], 0.0)

        for _ in range(self.depth + 1):  # include root
            # accumulate deltas
            nodes = nodes.at[idx].add(delta)
            idx = (idx - 1) >> 1

        return replace(
            state,
            nodes=nodes,
            max_priority=jnp.maximum(jnp.max(priority), state.max_priority),
        )

    def sample(self, state: SumTreeState, key: Key[Array, ""]) -> UInt[Array, " n"]:
        idx = jnp.uint(0)
        value = jr.uniform(key, maxval=state.nodes[idx])
        for _ in range(self.depth):
            left_idx = (idx << 1) + 1
            left_sum = state.nodes[left_idx]
            idx = jnp.where(value < left_sum, left_idx, left_idx + 1)
            value = jnp.where(value < left_sum, value, value - left_sum)

        def debug(error):
            if error:
                raise ValueError("Inconsistency in the priority tree! Invalid index encountered")

        jax.debug.callback(debug, jnp.any(idx - self.leaf_idx >= self.size))

        return idx - self.leaf_idx, state.nodes[idx]

    def recompute(self, state: SumTreeState):
        """Recomputes the entire tree."""
        nodes = state.nodes
        start, end = self.leaf_idx, nodes.size
        for _ in range(self.depth):
            start, end = start >> 1, start
            idx = jnp.arange(start, end)
            left_idx = (idx << 1) + 1
            right_idx = (idx << 1) + 2
            nodes = nodes.at[idx].set(nodes[left_idx] + nodes[right_idx])
        return replace(state, nodes=nodes)

    @property
    def leaf_idx(self):
        return (1 << self.depth) - 1
