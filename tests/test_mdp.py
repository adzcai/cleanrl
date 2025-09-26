import distrax
import jax
import jax.numpy as jnp
import pytest

from ilx.maps import SIMPLE_MAP, CellType
from ilx.mdp import GridEnv

jax.config.update("jax_disable_jit", True)

@pytest.fixture
def env():
    return GridEnv(SIMPLE_MAP, 0.9)

def test_reward(env):
    s = env.get_state(0, 1)
    a = 1
    s_, r_ = env._step(s, a)
    assert s_ == env.get_state(0, 2), f"{s_} is not goal state"
    assert r_ == CellType.GOAL.reward, f"{r_} does not match goal reward"
    assert env.R[s, a, s_] == r_, "reward tensor does not match"
    
    s_, r_ = env._step(env.get_state(2, 1), 0)
    assert s_ == env.get_state(1, 1), f"{s_} is not pit state"
    assert r_ == CellType.PIT.reward, f"{r_} does not match pit reward"

def test_wall(env):
    s = env.get_state(2, 2)
    s_, r_ = env._step(s, 0)
    assert r_ == 0, f"{r_} should be zero for no-op"
    assert s_ == s, f"{s_} should not change because of wall"

def test_draw(env):
    π = distrax.Categorical(logits=jnp.zeros((env.S, env.A)))
    d = distrax.Categorical(logits=jnp.zeros(env.S))
    env.draw(jnp.zeros(env.S), π, "artifacts/test_draw.png", "simple map", d)

def test_value_iteration(env):
    Q = env.solve_value_iteration()
    π = Q.argmax(axis=1)
    index = jnp.arange(env.S), π
    V = Q[index]
    π = jnp.zeros((env.S, env.A)).at[index].set(1)
    π = distrax.Categorical(probs=π)
    d = env.stationary(π)
    env.draw(V, π, "artifacts/test_value_iteration.png", "simple map", d)