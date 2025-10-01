import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax, vmap
from jax.scipy.optimize import minimize
from jaxtyping import Array, Float, UInt

from cleanrl_utils.envs.grid_env import LARGER_MAP, GridEnv, Q_to_greedy


def main(env: GridEnv, n_iters=4):
    π_expert = Q_to_greedy(env.value_iteration())

    def iterate(_: tuple[Float[Array, " D"], Float[Array, " S"]], count: UInt[Array, ""]):
        w, d = _
        d += env.π_to_stationary(env.softmax_π(w)).probs

        def loss(w: Float[Array, " D"]):
            return (d / count) @ π_expert.cross_entropy(env.softmax_π(w))

        results = minimize(loss, w, method="BFGS")
        return (results.x, d), (w, results.fun)

    (w_fit, _), (ws, losses) = lax.scan(iterate, (jnp.zeros(env.D), jnp.zeros(env.S)), 1 + jnp.arange(n_iters))

    regrets = env.π_to_return(π_expert) - vmap(env.π_to_return)(vmap(env.softmax_π)(ws))
    plt.title("dagger stats")
    plt.plot(losses, label="loss")
    plt.plot(regrets, label="regret")
    plt.legend()
    plt.savefig("artifacts/dagger-losses.png")

    env.draw(env.softmax_π(w_fit), "artifacts/dagger-learner.png", "dataset aggregation")


if __name__ == "__main__":
    main(GridEnv(LARGER_MAP, 0.99))
