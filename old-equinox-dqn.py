# %%
from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.random as rand
import equinox as eqx
import gymnax
from gymnax.environments.environment import Environment, EnvParams
from jaxtyping import Float, Integer, Bool, Array, PyTree
import optax
import math
import wandb
from functools import partial
from gymnax.visualize import Visualizer
from gymnax.wrappers import FlattenObservationWrapper, LogWrapper
import distrax


class Transition(NamedTuple):
    """A single transition."""

    state: Float[Array, "b ..."]
    action: Integer[Array, " b"]
    reward: Float[Array, " b"]
    done: Bool[Array, " b"]


class ReplayBuffer(NamedTuple):
    """A simple replay buffer containing transition data"""
    data: Transition
    index: Integer[Array, ""]

    @classmethod
    def init(cls, env: Environment, env_params: EnvParams, max_length: int):
        obs_space = env.observation_space(env_params)
        act_dtype = env.action_space(env_params).dtype

        return ReplayBuffer(
            Transition(
                state=jnp.empty((max_length, *obs_space.shape), dtype=obs_space.dtype),
                action=jnp.empty(max_length, dtype=act_dtype),
                reward=jnp.empty(max_length, dtype=jnp.float32),
                done=jnp.empty(max_length, dtype=jnp.bool),
            ),
            jnp.int32(0),
        )

    def add(self, txn: Transition) -> "ReplayBuffer":
        data = jax.tree_map(
            lambda ary, t: ary.at[self.index % self.data.action.size, ...].set(t),
            self.data,
            txn,
        )
        return ReplayBuffer(data, self.index + 1)

    def sample(self, n, *, key) -> Transition:
        idxs = rand.randint(key, (n,), 1, jnp.minimum(self.index, self.data.action.size))
        return jax.tree_map(
            lambda ary: ary[idxs - 1, ...],
            self.data,
        ), jax.tree_map(
            lambda ary: ary[idxs, ...],
            self.data,
        )


# %%
class Config(NamedTuple):
    batch_size: int
    discount: float
    n_updates: int
    update_every: int
    swap_target_model_every_n_updates: int
    buffer_size: int
    init_lr: int
    end_lr: int


config = Config(
    batch_size=32,
    discount=0.99,
    n_updates=10_000,
    update_every=100,
    swap_target_model_every_n_updates=20,
    buffer_size=50000,
    init_lr=0.01,
    end_lr=0.00001,
)

# %%
wandb.init(
    project="jax-rl",
    tags=["dqn"],
    config=config._asdict(),
)

# %%
def loss(params, target_params, static, buffer, *, key: rand.PRNGKey):
    model = eqx.combine(params, static)
    target_model = eqx.combine(target_params, static)

    def get_target(curr_step: Transition, next_step: Transition):
        """For a single transition"""
        next_value = jnp.max(target_model(next_step.state))
        target_value = next_step.reward + jnp.where(
            next_step.done, 0.0, config.discount * next_value
        )
        curr_value = model(curr_step.state)[curr_step.action]
        return (target_value - curr_value) ** 2

    curr_step, next_step = buffer.sample(config.batch_size, key=key)
    targets = jax.vmap(get_target)(curr_step, next_step)
    return jnp.mean(targets)


optim = optax.adamw(optax.linear_schedule(config.init_lr, config.end_lr, config.n_updates))

def step(params, target_params, static, opt_state, buffer, *, key):
    loss_value, grads = jax.value_and_grad(loss)(
        params, target_params, static, buffer, key=key
    )
    updates, opt_state = optim.update(
        grads, opt_state, params
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, updates, loss_value

def env_reset(rng, env, env_params):
    rng_reset, rng_act, rng_step = rand.split(rng, 3)
    obs, env_state = env.reset(rng_reset, env_params)
    # need initial step due to gymnax bug
    action = env.action_space(env_params).sample(rng_act)
    obs, env_state, _, _, _ = env.step(rng_step, env_state, action, env_params)
    return obs, env_state

# %%
class LoopCarry(NamedTuple):
    params: PyTree
    target_params: PyTree
    opt_state: optax.OptState
    buffer: ReplayBuffer
    n_steps: Integer[Array, ""]
    obs: Float[Array, "..."]
    env_state: gymnax.EnvState


def main(rng):
    env, env_params = gymnax.make("Catch-bsuite")
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    epsilon_schedule = optax.linear_schedule(
        1.0, 0.01, config.n_updates * config.update_every
    )

    @partial(jax.jit, donate_argnums=(0,))
    def interact(carry: LoopCarry, rng: rand.PRNGKey):
        rng, rng_act, rng_step, rng_reset, rng_update = rand.split(rng, 5)
        model = eqx.combine(carry.params, static)

        # pick an action
        values = model(carry.obs)
        epsilon = epsilon_schedule(carry.n_steps)
        act = distrax.EpsilonGreedy(values, epsilon).sample(seed=rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, carry.env_state, act, env_params
        )

        next_obs, next_env_state = jax.lax.cond(
            done,
            lambda: env_reset(rng_reset, env, env_params),
            lambda: (next_obs, next_env_state),
        )

        buffer = carry.buffer.add(
            Transition(
                obs,
                act,
                reward,
                done,
            )
        )

        # update
        count_updates, r = divmod(carry.n_steps, config.update_every)

        def update():
            params, opt_state, updates, loss_value = step(
                params=carry.params,
                target_params=carry.target_params,
                static=static,
                opt_state=carry.opt_state,
                buffer=buffer,
                key=rng_update,
            )
            jax.debug.callback(
                wandb.log,
                {
                    "train/loss": loss_value,
                    "train/updates": count_updates,
                    "train/epsilon": epsilon,
                    "train/opt_state": opt_state,
                } | {
                    f"train/gradients/{jax.tree_util.keystr(keys)}": leaf
                    for keys, leaf in jax.tree.leaves_with_path(updates)
                    if leaf is not None
                }
            )
            return params, opt_state

        params, opt_state = jax.lax.cond(
            r == 0,
            update,
            lambda: (
                carry.params,
                carry.opt_state,
            ),
        )

        target_params = jax.lax.cond(
            count_updates % config.swap_target_model_every_n_updates == 0,
            lambda: params,
            lambda: carry.target_params,
        )

        return LoopCarry(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            buffer=buffer,
            n_steps=carry.n_steps + 1,
            obs=next_obs,
            env_state=next_env_state,
        ), None

    rng, rng_model, rng_reset = rand.split(rng, 3)
    model = eqx.nn.MLP(
        math.prod(env.observation_space(env_params).shape),
        env.action_space(env_params).n,
        32,
        2,
        key=rng_model,
    )
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)
    buffer = ReplayBuffer.init(env, env_params, max_length=config.buffer_size)
    obs, env_state = env_reset(rng_reset, env, env_params)

    carry, losses = jax.lax.scan(
        interact,
        LoopCarry(
            params=params,
            opt_state=opt_state,
            target_params=params,
            buffer=buffer,
            n_steps=0,
            obs=obs,
            env_state=env_state,
        ),
        rand.split(rng, config.n_updates),
    )

    return carry, static


# %%
carry, static = jax.block_until_ready(main(rand.PRNGKey(15212)))

# %%

class VisualizeCarry(NamedTuple):
    done: Bool[Array, ""]
    obs: Float[Array, "..."]
    env_state: gymnax.EnvState

def visualize(params, static, max_steps: int, key: rand.PRNGKey):
    env, env_params = gymnax.make('Catch-bsuite')
    env = FlattenObservationWrapper(env)
    model = eqx.combine(params, static)

    def step(carry: VisualizeCarry, rng_step: rand.PRNGKey) -> VisualizeCarry:
        action = model(carry.obs).argmax()
        next_obs, next_env_state, reward, done, _ = env.step(
            rng_step, carry.env_state, action, env_params
        )
        jax.debug.print("{prev}", prev=rng_step, ordered=True)
        return VisualizeCarry(
            done=done,
            obs=next_obs,
            env_state=next_env_state,
        ), (carry.env_state, reward)
    
    rng, rng_reset = jax.random.split(key)
    obs, env_state = env_reset(rng_reset, env, env_params)
    carry, (env_states, rewards) = jax.lax.scan(
        lambda carry, rng: jax.lax.cond(
            carry.done,
            step,
            lambda carry, rng: (carry, (carry.env_state, 0.0)),
            carry,
            rng,
        ),
        VisualizeCarry(obs=obs, env_state=env_state, done=False),
        rand.split(rng, max_steps),
    )

    cum_rewards = jnp.cumsum(rewards)
    env_states = jax.tree.transpose(
        jax.tree.structure(env_state),
        None,
        jax.tree.map(lambda leaf: [leaf[i] for i in range(leaf.shape[0])], env_states)
    )

    vis = Visualizer(env, env_params, env_states, cum_rewards)
    vis.animate("./anim.gif")

    wandb.log_artifact("./anim.gif", type="animation", tags=["animation"])

    return carry


out = visualize(carry.params, static, 40, rand.PRNGKey(0))

# %%
import matplotlib.pyplot as plt

env, env_params = gymnax.make('Catch-bsuite')
env = FlattenObservationWrapper(env)
model = eqx.combine(carry.params, static)

rng = rand.PRNGKey(0)

rng, rng_reset = rand.split(rng)

obs, env_state = env.reset(rng_reset, env_params)

env_states, rewards = [], []

for _ in range(50):
    action = model(obs).argmax()
    rng, rng_step = rand.split(rng)
    next_obs, next_env_state, reward, done, _ = env.step(
        rng_step, env_state, action, env_params
    )

    env_states.append(env_state)
    rewards.append(reward)

    if done:
        break

    obs, env_state = next_obs, next_env_state

Visualizer(env, env_params, env_states, jnp.cumsum(jnp.array(rewards))).animate("anim.gif")


# %%
def visualize(model, key):
    env, env_params = gymnax.make('Catch-bsuite')
    env = FlattenObservationWrapper(env)

    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(key)
    obs, env_state = env.reset(rng_reset, env_params)
    t_counter = 0
    while True:
        state_seq.append(env_state)
        rng, rng_step = jax.random.split(rng)
        action = model(obs).argmax()
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        t_counter += 1
        if done or t_counter >= 50:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate("./anim.gif")

# %%



