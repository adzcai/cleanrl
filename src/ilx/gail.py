from distrax import Bernoulli
from jax.nn import sigmoid
import jax.numpy as jnp
import optax
from jax import hessian, lax, value_and_grad
from jaxtyping import Array, Float
from matplotlib import pyplot as plt

from ilx.core.maps import SIMPLE_MAP
from ilx.core.mdp import GridEnv, Q_to_greedy


def main(env: GridEnv, lr=0.5, n_iters=50):
    π_expert = Q_to_greedy(env.value_iteration())
    μ_expert = env.π_to_μ(π_expert)
    optim_d = optax.adamw(optax.exponential_decay(lr, 100, 0.001))
    optim_π = optax.adamw(optax.exponential_decay(lr, 100, 0.001))

    def step(carry: tuple[Float[Array, " D"], Float[Array, " D"], optax.OptState, optax.OptState], _):
        w_d, w_π, opt_state_d, opt_state_π = carry

        π = env.softmax_π(w_π)
        μ = env.π_to_μ(π)

        @value_and_grad
        def loss_d(w_d):
            pred = sigmoid(env.features @ w_d).ravel()
            return -jnp.sum(μ.probs * jnp.log(pred) + μ_expert.probs * jnp.log(1 - pred))

        l_d, grads_d = loss_d(w_d)
        updates_d, opt_state_d = optim_d.update(grads_d, opt_state_d, w_d)
        w_d = optax.apply_updates(w_d, updates_d)

        @value_and_grad
        def loss_π(w_π):
            π = env.softmax_π(w_π)
            return -env.π_to_return(π) - env.π_to_stationary(π).probs @ π.entropy()

        π_hessian = -hessian(lambda w: env.softmax_π(w).logits.ravel())(w_π)
        fisher = jnp.einsum("m, mcd -> cd", μ.probs, π_hessian) / (1 - env.γ)
        r, grads = loss_π(w_π)
        grads = jnp.linalg.solve(fisher + 1e-4 * jnp.eye(env.D), grads)
        updates_π, opt_state_π = optim_π.update(grads, opt_state_π, w_π)
        w_π = optax.apply_updates(w_π, updates_π)

        return (w_d, w_π, opt_state_d, opt_state_π), (-l_d, -r)

    w_d, w_π = jnp.zeros((2, env.D))
    opt_state_d, opt_state_π = optim_d.init(w_d), optim_π.init(w_π)
    (w_d_fit, w_π_fit, _, _), (losses_d, returns) = lax.scan(step, (w_d, w_π, opt_state_d, opt_state_π), length=n_iters)

    regret = env.π_to_return(Q_to_greedy(env.value_iteration())) - returns
    plt.plot(regret, label="regret")
    plt.plot(losses_d, label="discriminator losses")
    plt.legend()
    plt.savefig("artifacts/gail-losses.png")

    env.draw(env.softmax_π(w_π_fit), "artifacts/gail-learner.png", "learner")


if __name__ == "__main__":
    main(GridEnv(SIMPLE_MAP, 0.99))
