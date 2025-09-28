import jax.numpy as jnp
import optax
from jax import hessian, lax, value_and_grad
from jaxtyping import Array, Float
from matplotlib import pyplot as plt

from ilx.core.maps import SIMPLE_MAP
from ilx.core.mdp import GridEnv, Q_to_greedy


def main(env: GridEnv, lr=0.5, n_iters=50):
    optim = optax.adamw(optax.exponential_decay(lr, 100, 0.001))

    def step(carry: tuple[Float[Array, " D"], optax.OptState], _):
        w, opt_state = carry
        π = env.softmax_π(w)
        μ = env.π_to_μ(π)
        π_hessian = -hessian(lambda w: env.softmax_π(w).logits.ravel())(w)
        fisher = jnp.einsum("m, mcd -> cd", μ.probs, π_hessian) / (1 - env.γ)
        r, grads = value_and_grad(lambda w: -env.π_to_return(env.softmax_π(w)))(w)
        grads = jnp.linalg.solve(fisher + 1e-4 * jnp.eye(env.D), grads)
        updates, opt_state = optim.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return (w, opt_state), -r

    w = jnp.zeros(env.D)
    opt_state = optim.init(w)
    (w_fit, _), returns = lax.scan(step, (w, opt_state), length=n_iters)

    regret = env.π_to_return(Q_to_greedy(env.value_iteration())) - returns
    plt.plot(regret, label="regret")
    plt.savefig("artifacts/npg-losses.png")

    env.draw(env.softmax_π(w_fit), "artifacts/npg-learner.png", "learner")


if __name__ == "__main__":
    main(GridEnv(SIMPLE_MAP, 0.99))
