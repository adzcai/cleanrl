import distrax
import jax.numpy as jnp
import jax.random as jr

from ilx.core.maps import CellType


def test_env(simple_env):
    assert simple_env.S == 3 * 5 - 2
    assert simple_env.pos_to_state[2, 4] == simple_env.S - 1

    assert jnp.array_equal(
        simple_env.P.probs.argmax(axis=-1),
        jnp.asarray(
            [
                [0, 5, 0, 1],
                [1, 6, 0, 2],
                [2, 2, 2, 2],
                [3, 3, 2, 4],
                [4, 7, 3, 4],
                [0, 8, 5, 6],
                [1, 9, 5, 6],
                [4, 12, 7, 7],
                [5, 8, 8, 9],
                [6, 9, 8, 10],
                [10, 10, 9, 11],
                [11, 11, 10, 12],
                [7, 12, 11, 12],
            ]
        ),
    )


def test_reward(simple_env):
    s = simple_env.pos_to_state[0, 1]
    a = 3
    s_, r_ = simple_env.step(s, a, key=jr.key(0))
    assert s_ == simple_env.pos_to_state[0, 2], f"{s_} is not goal state"
    assert r_ == CellType.GOAL.reward, f"{r_} does not match goal reward"
    assert simple_env.R[s, a, s_] == r_, "reward tensor does not match"

    s = simple_env.pos_to_state[2, 1]
    s_, r_ = simple_env.step(s, 0, key=jr.key(0))
    assert s_ == simple_env.pos_to_state[1, 1], f"{s_} is not pit state"
    assert r_ == CellType.PIT.reward, f"{r_} does not match pit reward"


def test_larger(larger_env):
    print(jnp.argwhere(larger_env.R))
    print(larger_env.R[jnp.nonzero(larger_env.R)])


def test_wall(simple_env):
    s = simple_env.pos_to_state[2, 2]
    s_, r_ = simple_env.step(s, 0, key=jr.key(0))
    assert r_ == 0, f"{r_} should be zero for no-op"
    assert s_ == s, f"{s_} should not change because of wall"


def test_draw(simple_env):
    π = distrax.Categorical(logits=jnp.zeros((simple_env.S, simple_env.A)))
    simple_env.draw(π, "artifacts/test_draw.png", "simple map")


def test_value_iteration(env):
    Q = env.value_iteration()
    π = Q.argmax(axis=1)
    index = jnp.arange(env.S), π
    π = jnp.zeros((env.S, env.A)).at[index].set(1)
    π = distrax.Categorical(probs=π)
    env.draw(π, f"artifacts/test_value_iteration_{env.S}.png", "simple map")


def test_return(env):
    π = env.softmax_π(jnp.ones(env.D))

    a = env.π_to_return(π)

    d = env.π_to_stationary(π)
    V = env.π_to_V(π)
    V_ = jnp.einsum("sap, sa, p -> s", env.P.probs, π.probs, V)
    b = d.probs @ (V - env.γ * V_) / (1 - env.γ)

    assert jnp.allclose(a, b), f"{a} != {b}"
