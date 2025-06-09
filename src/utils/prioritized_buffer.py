"""A prioritized buffer for replay."""

import dataclasses as dc
from typing import Annotated as Batched
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Key, PyTree, UInt

from utils.log_util import dataclass, exec_callback, tree_slice

TExperience = TypeVar("TExperience", bound=PyTree[Array])


@dataclass
class SumTreeState:
    nodes: Float[Array, " max_size"]
    max_priority: Float[Array, ""]  # new trajectories are assigned max priority
    step: UInt[Array, ""]  # calibrate every some updates


@dataclass
class BufferState(Generic[TExperience]):
    data: Batched[TExperience, " batch_size max_time"]
    pos: UInt[Array, ""]
    priority_state: SumTreeState


@dataclass
class Sample(Generic[TExperience]):
    experience: Batched[TExperience, " horizon"]
    idx: UInt[Array, ""]
    priority: Float[Array, ""]


@dataclass
class PrioritizedBuffer(Generic[TExperience]):
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

    def init(self, experience: TExperience) -> BufferState[TExperience]:
        """Initialize buffer from a dummy experience."""
        data = jax.tree.map(
            lambda x: jnp.empty_like(x, shape=(self.batch_size, self.max_time, *x.shape)),
            experience,
        )

        return BufferState(
            data=data,
            pos=jnp.zeros((), jnp.uint32),
            priority_state=self.priority_tree.init(),
        )

    def add(
        self,
        state: BufferState[TExperience],
        experience: Batched[TExperience, " batch_size horizon"],
    ) -> BufferState[TExperience]:
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
        state = dc.replace(state, data=data, pos=state.pos + horizon)
        # enable the trajectories starting up to this point
        # TODO currently incorrect for initial trajectory if too short
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
        state: BufferState[TExperience],
        idx: UInt[Array, " batch_size"],
        priorities: Float[Array, " batch_size"],
    ) -> BufferState[TExperience]:
        """Set priorities for a batch of trajectories."""
        priority_state = self.priority_tree.update(state.priority_state, idx, priorities)
        return dc.replace(state, priority_state=priority_state)

    def sample(
        self, state: BufferState[TExperience], debug=False, *, key: Key[Array, ""]
    ) -> Batched[Sample[TExperience], " horizon"]:
        """Sample a trajectory from the buffer."""
        key_sample, key_fallback = jr.split(key)
        idx, priority = self.priority_tree.sample(state.priority_state, key_sample, debug=debug)
        batch, pos = divmod(idx, self.max_time)

        # ensure valid indices
        # sample from uniform if invalid
        pos_avail = self.available_pos(state)
        fallback_pos = jr.randint(key_fallback, pos.shape, 0, pos_avail)
        pos = jnp.where(pos < pos_avail, pos, fallback_pos)

        # take trajectories
        pos_ary = (pos + jnp.arange(self.horizon)) % self.max_time
        return Sample(
            experience=tree_slice(state.data, (batch, pos_ary)),
            idx=idx,
            priority=priority,
        )

    def available_pos(self, state: BufferState[TExperience]) -> int:
        """The number of available initial positions (along the horizon axis).

        i.e. (randint(0, available_pos) + range(horizon)) % max_time
        is entirely populated.
        """
        return jnp.where(
            state.pos < self.horizon,
            0,
            jnp.minimum(state.pos, self.max_time) - self.horizon + 1,
        )

    def num_available(self, state: BufferState[TExperience]) -> int:
        """The number of possible trajectories (overlapping and across batch)."""
        size = self.batch_size * self.available_pos(state)
        # could also count nonzero in self.priority_tree.nodes[(1 << self.priority_tree.depth) - 1 :]
        return size


@dataclass
class SumTree:
    """Contains the priorities in a sum tree.

    Throughout the implementation,
    `index` indicates an index into the leaves and
    `idx` indicates an index into the flattened tree structure
    (offset by `self.leaf_idx`).
    """

    size: int
    depth: int
    calibrate_freq: int

    @classmethod
    def new(cls, size: int, calibrate_freq: int = 64):
        depth = int(jnp.ceil(jnp.log2(size)))  # root at zero
        return cls(
            size=size,
            depth=depth,
            calibrate_freq=calibrate_freq,
        )

    def init(self):
        nodes = jnp.zeros((1 << (self.depth + 1)) - 1)
        return SumTreeState(
            nodes=nodes,
            max_priority=jnp.ones((), float),
            step=jnp.zeros((), jnp.uint32),
        )

    def update(
        self,
        state: SumTreeState,
        indices: UInt[Array, " n"],
        priority: Float[Array, " n"],
    ) -> SumTreeState:
        """Set priorities at the given indices.

        To mitigate floating point errors,
        every calibrate_freq updates,
        recompute the tree.

        Args:
            indices (UInt (n,)): The indices to update.
            priority (Float (n,)): The new values.

        Returns:
            SumTree: The updated tree.
        """
        return jax.lax.cond(
            state.step % self.calibrate_freq == 0,
            self._recompute,
            self._update,
            state,
            indices,
            priority,
        )

    def _update(
        self,
        state: SumTreeState,
        indices: UInt[Array, " n"],
        priority: Float[Array, " n"],
    ) -> SumTreeState:
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

        return dc.replace(
            state,
            nodes=nodes,
            max_priority=jnp.maximum(jnp.max(priority), state.max_priority),
            step=state.step + 1,
        )

    def _recompute(
        self,
        state: SumTreeState,
        indices: UInt[Array, " n"],
        priority: Float[Array, " n"],
    ):
        """Sets the priorities and recomputes the entire tree."""
        nodes = state.nodes.at[indices + self.leaf_idx].set(priority)
        idx = jnp.arange(self.leaf_idx, nodes.size)
        for _ in range(self.depth):
            idx = (idx - 1) >> 1
            left_idx = (idx << 1) + 1
            right_idx = (idx << 1) + 2
            nodes = nodes.at[idx].set(nodes[left_idx] + nodes[right_idx])
        return dc.replace(
            state,
            nodes=nodes,
            max_priority=jnp.max(nodes[self.leaf_idx :]),
            step=state.step + 1,
        )

    def sample(self, state: SumTreeState, key: Key[Array, ""], debug=False) -> UInt[Array, ""]:
        """Sample a single index from the priorities.

        Raises:
            ValueError: If debug is enabled and an out-of-bounds index is returned.

        Returns:
            tuple[UInt (), Float ()]: The sampled index; The priority of the sampled item.
        """
        idx = jnp.zeros((), jnp.uint32)
        value = jr.uniform(key, maxval=state.nodes[idx])
        for _ in range(self.depth):
            left_idx = (idx << 1) + 1
            left_sum = state.nodes[left_idx]
            idx = jnp.where(value < left_sum, left_idx, left_idx + 1)
            value = jnp.where(value < left_sum, value, value - left_sum)

        if debug:

            @exec_callback
            def debug(
                error=jnp.any(idx - self.leaf_idx >= self.size),
                top=state.nodes[0],
                total=state.nodes[self.leaf_idx :].sum(),
            ):
                if error:
                    raise IndexError(
                        "Invalid index encountered when sampling from the priority tree. "
                        "This is most likely due to accumulated floating-point errors. "
                        f"Difference was {top - total:.3e} "
                        f"(relative {(top - total) / top:.3e}). "
                        "Pass a smaller calibrate_freq to mitigate this."
                    )

        return idx - self.leaf_idx, state.nodes[idx]

    @property
    def leaf_idx(self):
        return (1 << self.depth) - 1

    def pprint(self, state: SumTreeState):
        """Pretty print the sum tree structure for debugging.

        Args:
            state (SumTreeState): The current state of the sum tree.
        """
        nodes = state.nodes
        depth = self.depth
        leaf_idx = self.leaf_idx

        # Width settings
        node_width = 7  # Width for each number
        total_cols = 2**depth

        for level in range(depth + 1):
            # Calculate indices for this level
            start_idx = (1 << level) - 1
            end_idx = (1 << (level + 1)) - 1
            level_nodes = nodes[start_idx:end_idx]

            # Calculate positions for this level
            positions = []
            segment_width = total_cols // (1 << level)
            for i in range(1 << level):
                pos = (segment_width // 2) + i * segment_width
                positions.append(pos)

            # Print level number and nodes
            line = [" " * node_width] * total_cols
            for pos, val in zip(positions, level_nodes):
                line[pos] = f"{val:7.2f}"

            print(f"Level {level}:", "".join(line))

        # Print separator and leaf values
        print("-" * (total_cols * node_width))
        print("Leaf values:", nodes[leaf_idx : leaf_idx + self.size])
