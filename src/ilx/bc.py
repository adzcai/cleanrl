import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import lax, value_and_grad, vmap
from jaxtyping import Array, Float

from ilx.core.maps import LARGER_MAP
from ilx.core.mdp import GridEnv, Q_to_greedy


def main(env: GridEnv, lr=0.5, n_iters=20):
    π_expert = Q_to_greedy(env.value_iteration())
    d_expert = env.π_to_stationary(π_expert)

    @value_and_grad
    def loss(w: Float[Array, " D"]):
        return d_expert.probs @ π_expert.kl_divergence(env.softmax_π(w))

    optim = optax.adamw(optax.exponential_decay(lr, 100, 0.001))
    w = jnp.zeros(env.D)
    opt_state = optim.init(w)

    def step(carry: tuple[optax.Params, optax.OptState], _):
        w, opt_state = carry
        l, grads = loss(w)
        updates, opt_state = optim.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return (w, opt_state), (w, l)

    (w_fit, _), (ws, losses) = lax.scan(step, (w, opt_state), length=n_iters)

    regrets = env.π_to_return(π_expert) - vmap(env.π_to_return)(vmap(env.softmax_π)(ws))
    plt.title("behavior cloning stats")
    plt.plot(losses, label="loss")
    plt.plot(regrets, label="regret")
    plt.legend()
    plt.savefig("artifacts/bc-losses.png")

    env.draw(env.softmax_π(w_fit), "artifacts/bc-learner.png", "check behavior cloning")


if __name__ == "__main__":
    main(GridEnv(LARGER_MAP, 0.99))
