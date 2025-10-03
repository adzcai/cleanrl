import os
import random
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import minari
import numpy as np
import optax
import tyro
from jaxtyping import Array, Float
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.envs import env_tabular


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
    dataset_id: str = "D4RL/walker2d/expert-v0"
    """the id of the expert dataset"""
    num_expert_episodes: int = 8
    """the number of expert episodes to sample"""
    total_timesteps: int = 50
    """total timesteps of the experiments"""
    learning_rate: float = 0.5
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""



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

    # collect expert data
    dataset = minari.load_dataset(args.dataset_id)
    dataset.set_seed(seed=args.seed)
    episodes = dataset.sample_episodes(args.num_expert_episodes)
    expert_states, expert_actions = map(lambda ep: (ep.observations, ep.actions), episodes)

    env = dataset.recover_environment()

    # network
    w = jnp.zeros(env.D)

    # optimizer
    optim = optax.adamw(optax.exponential_decay(args.learning_rate, 100, 0.001))
    opt_state = optim.init(w)

    def update(w: optax.Params, opt_state: optax.OptState):
        def loss(w: Float[Array, " D"]):
            π = env.softmax_π(w)
            return -π.logits[expert_states, expert_actions].mean(), env.π_to_return(π)

        (l, value), grads = jax.value_and_grad(loss, has_aux=True)(w)
        updates, opt_state = optim.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return (w, opt_state), (value, l)

    (w_fit, _), (values, losses) = jax.lax.scan(lambda carry, _: update(*carry), (w, opt_state), length=args.total_timesteps)

    regrets = env.π_to_return(π_expert) - values

    for idx, (loss_val, regret) in enumerate(zip(losses, regrets)):
        writer.add_scalar("losses/bc_loss", loss_val.item(), idx)
        writer.add_scalar("losses/episodic_regret", regret.item(), idx)

    fig = env.draw(env.softmax_π(w_fit), "check behavior cloning")
    writer.add_figure("eval/final", fig)
    writer.close()
