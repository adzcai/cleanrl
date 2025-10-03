import os
import random
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from jax import lax, value_and_grad
from jax.scipy.special import logsumexp
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.envs.env_tabular import GridEnv
from cleanrl_utils.jax_utils import f_divergence


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "simple"
    """the id of the environment"""
    total_timesteps: int = 100
    """total timesteps of the experiments"""
    learning_rate: float = 0.5
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""
    f_divergence: str = "chisq"
    """the f divergence to use"""
    proximal: bool = False
    """whether to minimize KL divergence to current policy"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    env = GridEnv(args.env_id, args.gamma)

    # expert data
    π_expert = env.Q_to_greedy(env.value_iteration())
    μ_expert = env.π_to_μ(π_expert)
    optim = optax.adamw(optax.exponential_decay(args.learning_rate, 50, 0.1))

    # networks
    w = jnp.zeros(env.D)
    opt_state = optim.init(w)

    def update(w: optax.Params, opt_state: optax.OptState):
        π = env.softmax_π(w)
        value = env.π_to_return(π)

        def loss(w):
            Q = env.features @ w
            V = jnp.log(jnp.sum(jnp.exp(Q) * (π.probs if args.proximal else 1), axis=1))
            loss_expert = f_divergence(args.f_divergence, env.γ * env.P.probs @ V - Q, dual=True)
            return (1 - env.γ) * env.d0.probs @ V + μ_expert.probs @ loss_expert.ravel()

        loss_val, grads = value_and_grad(loss)(w)
        updates, opt_state = optim.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return (w, opt_state), (value, loss_val)

    (w_fit, _), (values, losses) = lax.scan(lambda carry, _: update(*carry), (w, opt_state), length=args.total_timesteps)

    regrets = env.π_to_return(π_expert) - values
    for i, (regret, loss_val) in enumerate(zip(regrets.tolist(), losses.tolist())):
        writer.add_scalar("losses/irl_loss", loss_val, i)
        writer.add_scalar("charts/episodic_regret", regret, i)

    fig = env.draw(env.softmax_π(w_fit), "learner")
    writer.add_figure("eval/final", fig)

    writer.close()
