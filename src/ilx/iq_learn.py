import jax.numpy as jnp
import optax
from distrax import Categorical
from jax import lax, value_and_grad, vmap
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from matplotlib import pyplot as plt

from ilx.core.maps import SIMPLE_MAP
from ilx.core.mdp import GridEnv, Q_to_greedy


def main(env: GridEnv, lr=0.5, β=0.01, n_iters=50, f="chisq"):
    π_expert = Q_to_greedy(env.value_iteration())
    μ_expert = env.π_to_μ(π_expert)
    optim = optax.adamw(optax.exponential_decay(lr, 50, 0.01))

    def f_dual(c):
        if f == "chisq":
            return c * c / 4 + c
        elif f == "kl_rev":
            return jnp.exp(c-1)
        else:
            raise ValueError(f"f {f} not recognized")

    @value_and_grad
    def loss(w):
        Q = env.features @ w
        V = logsumexp(Q, axis=1)
        return (1 - env.γ) * env.d0.probs @ V + μ_expert.probs @ f_dual(
            env.γ * env.P.probs @ V - Q
        ).ravel()

    def step(carry: tuple[Float[Array, " D"], optax.OptState], _):
        w, opt_state = carry
        r, grads = loss(w)
        updates, opt_state = optim.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return (w, opt_state), (w, r)

    w = jnp.zeros(env.D)
    opt_state = optim.init(w)
    (w_fit, _), (ws, losses) = lax.scan(step, (w, opt_state), length=n_iters)

    returns = vmap(lambda w: env.π_to_return(Categorical(logits=env.features @ w)))(ws)
    regret = env.π_to_return(π_expert) - returns
    plt.plot(regret, label="regret")
    plt.plot(losses, label="IQ-Learn loss")
    plt.legend()
    plt.savefig("artifacts/iq-learn-losses.png")

    env.draw(env.softmax_π(w_fit), "artifacts/iq-learn-learner.png", "learner")


if __name__ == "__main__":
    main(GridEnv(SIMPLE_MAP, 0.99))
