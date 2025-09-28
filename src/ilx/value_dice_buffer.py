import jax.numpy as jnp
import optax
from jax import lax, value_and_grad, vmap
from jaxtyping import Array, Float
from matplotlib import pyplot as plt

from ilx.core.maps import SIMPLE_MAP
from ilx.core.mdp import GridEnv, Q_to_greedy


def main(env: GridEnv, lr_Q=0.5, lr_π=0.1, n_iters=200, α=0.1):
    π_expert = Q_to_greedy(env.value_iteration())
    μ_expert = env.π_to_μ(π_expert)
    optim_Q = optax.adamw(optax.exponential_decay(lr_Q, 100, 0.1))
    optim_π = optax.adamw(optax.exponential_decay(lr_π, 100, 0.1))

    def step(
        carry: tuple[
            Float[Array, " D"],
            Float[Array, " D"],
            optax.OptState,
            optax.OptState,
            Float[Array, " SA"],
        ],
        count: int,
    ):
        w_Q, w_π, opt_state_Q, opt_state_π, buf = carry

        buf += env.π_to_μ(env.softmax_π(w_π)).probs
        μ_mix = (1 - α) * μ_expert.probs + α * buf / count

        def loss(w_Q, w_π):
            Q = env.features @ w_Q
            π = env.softmax_π(w_π)
            R = Q - env.γ * jnp.einsum("sap, pb, pb -> sa", env.P.probs, π.probs, Q)
            value_π = (1 - env.γ) * jnp.einsum("s, sa, sa ->", env.d0.probs, π.probs, Q)
            value_buf = (buf / count) @ R.ravel()
            return jnp.log(μ_mix @ jnp.exp(R).ravel()) - (
                (1 - α) * value_π + α * value_buf
            )

        l_Q, grads_Q = value_and_grad(loss, 0)(w_Q, w_π)
        updates_Q, opt_state_Q = optim_Q.update(grads_Q, opt_state_Q, w_Q)
        w_Q = optax.apply_updates(w_Q, updates_Q)

        _, grads_π = value_and_grad(lambda w_Q, w_π: -loss(w_Q, w_π), 1)(w_Q, w_π)
        updates_π, opt_state_π = optim_π.update(grads_π, opt_state_π, w_π)
        w_π = optax.apply_updates(w_π, updates_π)

        return (w_Q, w_π, opt_state_Q, opt_state_π, buf), (l_Q, w_π)

    w_Q, w_π = jnp.zeros((2, env.D))
    opt_state_Q, opt_state_π = optim_Q.init(w_Q), optim_π.init(w_π)
    (_, w_π_fit, _, _, _), (losses, w_πs) = lax.scan(
        step,
        (w_Q, w_π, opt_state_Q, opt_state_π, jnp.zeros(env.S * env.A)),
        jnp.arange(n_iters) + 1,
    )

    regrets = env.π_to_return(π_expert) - vmap(env.π_to_return)(
        vmap(env.softmax_π)(w_πs)
    )
    plt.plot(regrets, label="regret")
    plt.plot(losses, label="loss")
    plt.legend()
    plt.savefig("artifacts/value-dice-buffer-losses.png")

    idx = jnp.argmin(regrets)

    env.draw(env.softmax_π(w_π_fit), "artifacts/value-dice-buffer-learner.png", "learner")
    env.draw(env.softmax_π(w_πs[idx]), "artifacts/value-dice-buffer-best.png", "learner")


if __name__ == "__main__":
    main(GridEnv(SIMPLE_MAP, 0.99))
