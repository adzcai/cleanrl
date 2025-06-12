import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jaxtyping import Array, Float

from utils.prioritized_buffer import PrioritizedBuffer


@pytest.fixture
def dummy_experience() -> dict[str, Float[Array, ""]]:
    # Simple dummy experience: a dict with a single array
    return {"obs": jnp.zeros((4,))}


@pytest.fixture
def buffer(dummy_experience):
    buf = PrioritizedBuffer.new(
        batch_size=2,
        max_time=5,
        sample_len=3,
    )
    state = buf.init(dummy_experience)
    return buf, state


def test_init_shape(buffer, dummy_experience):
    buf, state = buffer
    # Check data shape
    assert state.data["obs"].shape == (buf.batch_size, buf.max_time, 4)
    # Check pos and priority_state
    assert state.pos == 0
    assert (
        state.priority_state.nodes.shape[0] == (1 << (buf.priority_tree.depth + 1)) - 1
    )


def test_add_and_sample(buffer, dummy_experience):
    buf, state = buffer
    # Add a batch of experiences
    batch = {
        "obs": jnp.arange(buf.batch_size * buf.sample_len * 4).reshape(
            buf.batch_size, buf.sample_len, 4
        )
    }
    state = buf.add(state, batch)
    # After adding, pos should be advanced
    assert state.pos == buf.sample_len
    # Sample from buffer
    key = jax.random.PRNGKey(0)
    sample = buf.sample(state, key=key)
    # Check sample shape
    assert sample.experience["obs"].shape == (buf.sample_len, 4)
    assert sample.idx.shape == ()
    assert sample.priority.shape == ()


def test_set_priorities(buffer, dummy_experience):
    buf, state = buffer
    idx = jnp.array([0, 1], dtype=jnp.uint32)
    priorities = jnp.array([0.5, 2.0])
    state = buf.set_priorities(state, idx, priorities)
    # Check that max_priority is updated
    assert jnp.isclose(state.priority_state.max_priority, 2.0)


def test_priority_sum_matches_root(buffer, dummy_experience):
    buf, state = buffer
    # Add a batch of experiences
    batch = {
        "obs": jnp.arange(buf.batch_size * buf.sample_len * 4).reshape(
            buf.batch_size, buf.sample_len, 4
        )
    }
    state = buf.add(state, batch)
    # Set known priorities for all possible indices
    num_leaves = buf.priority_tree.size
    idx = jnp.arange(num_leaves, dtype=jnp.uint32)
    priorities = jnp.arange(1, num_leaves + 1)
    state = buf.set_priorities(state, idx, priorities)
    # The root node should equal the sum of all priorities
    root_val = state.priority_state.nodes[0]
    expected_sum = priorities.sum()
    assert jnp.isclose(root_val, expected_sum), f"Root {root_val}, sum {expected_sum}"


def test_multiple_priority_updates(buffer, dummy_experience):
    buf, state = buffer
    # Add a batch of experiences
    batch = {
        "obs": jnp.arange(buf.batch_size * buf.sample_len * 4).reshape(
            buf.batch_size, buf.sample_len, 4
        )
    }
    state = buf.add(state, batch)
    num_leaves = buf.priority_tree.size
    idx = jnp.arange(num_leaves, dtype=jnp.uint32)
    # Repeatedly update priorities in a cycle, increasing each time
    for i in range(1, 11):
        priorities = jnp.full((num_leaves,), i)
        state = buf.set_priorities(state, idx, priorities)
        # After each update, root should be sum of all priorities
        root_val = state.priority_state.nodes[0]
        expected_sum = priorities.sum()
        assert jnp.isclose(root_val, expected_sum), (
            f"Cycle {i}: Root {root_val}, sum {expected_sum}"
        )
        # Max priority should be updated
        assert jnp.isclose(state.priority_state.max_priority, float(i))


def test_num_available(buffer, dummy_experience):
    buf, state = buffer
    # Initially, not enough data for a trajectory
    assert buf.num_available(state) == 0
    # Add enough data
    batch = {"obs": jnp.ones((buf.batch_size, buf.sample_len, 4))}
    state = buf.add(state, batch)
    # Now should have available trajectories
    assert buf.num_available(state) > 0


def test_new_experience_priority_is_max(buffer, dummy_experience):
    buf, state = buffer
    max_priority = state.priority_state.max_priority
    # Add a new experience
    batch = {"obs": jnp.ones((buf.batch_size, buf.sample_len, 4))}

    state = buf.add(state, batch)
    assert state.pos == buf.sample_len
    idx_pos_zero = buf.pos_to_flat(jnp.zeros((buf.batch_size,), dtype=int))
    priorities = state.priority_state.nodes[buf.priority_tree.leaf_idx :]
    desired = jnp.zeros_like(priorities).at[idx_pos_zero].set(max_priority)
    npt.assert_array_equal(priorities, desired)

    new_priorities = max_priority + jnp.arange(buf.batch_size, dtype=float)
    state = buf.set_priorities(state, idx_pos_zero, new_priorities)

    batch = {"obs": jnp.ones((buf.batch_size, 2, 4))}
    state = buf.add(state, batch)
    next_idx = jax.vmap(buf.pos_to_flat)(
        jnp.ones((buf.batch_size,), dtype=int)[jnp.newaxis, :]
        + jnp.arange(2)[:, jnp.newaxis]
    ).ravel()
    priorities = state.priority_state.nodes[buf.priority_tree.leaf_idx :]
    desired = (
        jnp.zeros_like(priorities)
        .at[idx_pos_zero]
        .set(new_priorities)
        .at[next_idx]
        .set(jnp.max(new_priorities))
    )
    npt.assert_array_equal(priorities, desired)


def test_buffer_overfill(buffer, dummy_experience):
    buf, state = buffer  # (2, 5)
    # Fill the buffer to its max capacity
    batch = {"obs": jnp.ones((buf.batch_size, buf.sample_len, 4))}
    state = buf.add(state, batch)
    new_batch = {"obs": -jnp.ones((buf.batch_size, buf.sample_len, 4))}
    state = buf.add(state, new_batch)

    # Now the buffer should be full
    assert state.pos > buf.max_time
    assert state.pos == buf.sample_len * 2

    desired = jnp.broadcast_to(
        jnp.array([-1.0, 1.0, 1.0, -1.0, -1.0])[jnp.newaxis, :, jnp.newaxis],
        (buf.batch_size, 5, 4),
    )

    # Check that the first entries are now zeros
    npt.assert_array_equal(state.data["obs"], desired)


def test_buffer_overfill_priorities(buffer, dummy_experience):
    buf, state = buffer  # (2, 5)
    # Fill the buffer to its max capacity
    batch = {"obs": jnp.ones((buf.batch_size, buf.sample_len, 4))}
    state = buf.add(state, batch)

    # Check that the priorities are set correctly
    desired_priorities = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    priorities = state.priority_state.nodes[
        buf.priority_tree.leaf_idx : buf.priority_tree.leaf_idx + buf.max_time
    ]
    npt.assert_array_equal(priorities, desired_priorities)

    new_batch = {"obs": -jnp.ones((buf.batch_size, buf.sample_len, 4))}
    state = buf.add(state, new_batch)

    # Now the buffer should be full
    priorities = state.priority_state.nodes[
        buf.priority_tree.leaf_idx : buf.priority_tree.leaf_idx + buf.max_time
    ]
    desired_priorities = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])

    # Check that the first entries are now zeros
    npt.assert_array_equal(priorities, desired_priorities)
