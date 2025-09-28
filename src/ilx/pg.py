import jax.numpy as jnp
import optax
from jax import lax, value_and_grad
from matplotlib import pyplot as plt

from ilx.core.maps import SIMPLE_MAP
from ilx.core.mdp import GridEnv, Q_to_greedy


def main(env: GridEnv, lr = 0.5, n_iters=50):
    optim = optax.adamw(optax.exponential_decay(lr, 100, 0.001))

    def step(carry: tuple[optax.Params, optax.OptState], _):
        w, opt_state = carry
        r, grads = value_and_grad(lambda w: -env.π_to_return(env.softmax_π(w)))(w)
        updates, opt_state = optim.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return (w, opt_state), -r
    
    w = jnp.zeros(env.D)
    opt_state = optim.init(w)
    (w_fit, _), returns = lax.scan(step, (w, opt_state), length=n_iters)

    regret = env.π_to_return(Q_to_greedy(env.value_iteration())) - returns
    plt.plot(regret, label="regret")
    plt.savefig("artifacts/pg-losses.png")

    env.draw(env.softmax_π(w_fit), "artifacts/pg-learner.png", "learner")

if __name__ == "__main__":
    main(GridEnv(SIMPLE_MAP, 0.99))