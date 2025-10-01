import jax.numpy as jnp
import optax
from distrax import Categorical
from jax import lax, value_and_grad, vmap
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from matplotlib import pyplot as plt

from cleanrl_utils.envs.grid_env import SIMPLE_MAP, GridEnv, Q_to_greedy


def main(env: GridEnv, lr_Q=0.5, lr_x=1, n_iters=50, f_name="chisq"):
    π_expert = Q_to_greedy(env.value_iteration())
    μ_expert = env.π_to_μ(π_expert)

    optim_Q = optax.adamw(optax.exponential_decay(lr_Q, 50, 0.01))
    optim_x = optax.adamw(optax.exponential_decay(lr_x, 50, 0.001))

    def f(x):
        if f_name == "chisq":
            return (x - 1) ** 2
        else:
            raise NotImplementedError(f"f {f_name} not recognized")

    def loss(w_Q: Float[Array, " D"], w_x: Float[Array, " D"]):
        Q = env.features @ w_Q
        V = logsumexp(Q, axis=1)
        x = env.features @ w_x
        loss_expert = (env.γ * env.P.probs @ V - Q) * x - f(x)
        return (1 - env.γ) * env.d0.probs @ V + μ_expert.probs @ loss_expert.ravel()

    def step(
        carry: tuple[Float[Array, " D"], Float[Array, " D"], optax.OptState, optax.OptState],
        _,
    ):
        w_Q, w_x, opt_state_Q, opt_state_x = carry

        l, grads_Q = value_and_grad(loss, 0)(w_Q, w_x)
        updates_Q, opt_state_Q = optim_Q.update(grads_Q, opt_state_Q, w_Q)
        w_Q = optax.apply_updates(w_Q, updates_Q)

        _, grads_x = value_and_grad(lambda w_Q, w_x: -loss(w_Q, w_x), 1)(w_Q, w_x)
        updates_x, opt_state_x = optim_x.update(grads_x, opt_state_x, w_x)
        w_x = optax.apply_updates(w_x, updates_x)

        return (w_Q, w_x, opt_state_Q, opt_state_x), (l, w_Q)

    w_Q, w_x = jnp.zeros((2, env.D))
    opt_state_Q, opt_state_x = optim_Q.init(w_Q), optim_x.init(w_x)
    (w_Q_fit, _, _, _), (losses, w_Qs) = lax.scan(step, (w_Q, w_x, opt_state_Q, opt_state_x), length=n_iters)

    returns = vmap(lambda w: env.π_to_return(Categorical(logits=env.features @ w)))(w_Qs)
    regrets = env.π_to_return(π_expert) - returns
    plt.plot(regrets, label="regret")
    plt.plot(losses, label="IQ-Learn loss")
    plt.legend()
    plt.savefig("artifacts/iq-learn-dual-losses.png")

    env.draw(env.softmax_π(w_Q_fit), "artifacts/iq-learn-dual-learner.png", "learner")


if __name__ == "__main__":
    main(GridEnv(SIMPLE_MAP, 0.99))
