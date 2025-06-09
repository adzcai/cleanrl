import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree

from utils.prioritized_buffer import PrioritizedBuffer


@pytest.fixture
def dummy_experience():
    # Simple dummy experience: a dict with a single array
    return {"obs": jnp.zeros((4,))}


@pytest.fixture
def buffer(dummy_experience):
    batch_size = 2
    max_time = 5
    horizon = 3
    buf = PrioritizedBuffer.new(batch_size, max_time, horizon)
    state = buf.init(dummy_experience)
    return buf, state


def test_init_shape(buffer, dummy_experience):
    buf, state = buffer
    # Check data shape
    assert state.data["obs"].shape == (buf.batch_size, buf.max_time, 4)
    # Check pos and priority_state
    assert state.pos == 0
    assert state.priority_state.nodes.shape[0] == (1 << (buf.priority_tree.depth + 1)) - 1


def test_add_and_sample(buffer, dummy_experience):
    buf, state = buffer
    # Add a batch of experiences
    batch = {
        "obs": jnp.arange(buf.batch_size * buf.horizon * 4).reshape(buf.batch_size, buf.horizon, 4)
    }
    state = buf.add(state, batch)
    # After adding, pos should be advanced
    assert state.pos == buf.horizon
    # Sample from buffer
    key = jax.random.PRNGKey(0)
    sample = buf.sample(state, key=key)
    # Check sample shape
    assert sample.experience["obs"].shape == (buf.horizon, 4)
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
        "obs": jnp.arange(buf.batch_size * buf.horizon * 4).reshape(buf.batch_size, buf.horizon, 4)
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
        "obs": jnp.arange(buf.batch_size * buf.horizon * 4).reshape(buf.batch_size, buf.horizon, 4)
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
    batch = {"obs": jnp.ones((buf.batch_size, buf.horizon, 4))}
    state = buf.add(state, batch)
    # Now should have available trajectories
    assert buf.num_available(state) > 0
