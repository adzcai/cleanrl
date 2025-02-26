from typing import Any

from config import EnvConfig
import gymnax.environments.environment as ge
import gymnax.environments.spaces as spaces
import housemaze
import housemaze.env
import jax
import jax.numpy as jnp
import jax.random as jr
import navix as nx
from jaxtyping import Array, Key

from wrappers.auto_reset import auto_reset_wrapper
from wrappers.base import Environment, StepType, Timestep
from wrappers.flatten_observation import flatten_observation_wrapper
from wrappers.goal_wrapper import goal_wrapper
from wrappers.log import log_wrapper
from wrappers.multi_catch import make_multi_catch


def gymnax_wrapper(env: ge.Environment[ge.TEnvState, ge.TEnvParams], params: ge.TEnvParams):
    _, init_state = env.reset(jr.key(0), params)
    _, init_step = env.step(
        jr.key(0), init_state, env.action_space(params).sample(jr.key(0)), params
    )
    init_info = jax.tree.map(jnp.empty_like, init_step[-1])

    def reset(params: ge.TEnvParams, *, key: Key[Array, ""]):
        obs, state = env.reset(key, params)
        return Timestep.initial(obs, state, init_info)

    def step(state: ge.TEnvState, action: Any, params: ge.TEnvParams, *, key: Key[Array, ""]):
        obs, state, reward, done, info = env.step(key, state, action, params)
        return Timestep(
            obs=obs,
            state=state,
            reward=reward,
            step_type=jnp.where(done, StepType.TERMINATION, StepType.TRANSITION),
            info=info,
        )

    return Environment(
        _inner=env,  # technically a type error
        reset=reset,
        step=step,
        action_space=env.action_space,
        observation_space=env.observation_space,
        goal_space=lambda env_params: spaces.Discrete(0),
    )


def navix_wrapper(env: nx.Environment):
    def reset(params: None, *, key: Key[Array, ""]):
        timestep = env.reset(key)
        return timestep.observation, timestep

    def step(state: nx.Timestep, action: Any, params: None, *, key: Key[Array, ""]):
        timestep = env.step(state, action)
        step_type = jnp.select(
            [timestep.is_termination(), timestep.is_truncation()],
            [StepType.TERMINATION, StepType.TRUNCATION],
            default=StepType.TRANSITION,
        )
        return Timestep(
            obs=timestep.observation,
            state=timestep,
            reward=timestep.reward,
            step_type=step_type,
            info=timestep.info,
        )

    return Environment(
        _inner=env,
        reset=reset,
        step=step,
        action_space=lambda params: env.action_space,
        observation_space=lambda params: env.observation_space,
        goal_space=lambda params: spaces.Discrete(0),
    )


def make_env(env_config: EnvConfig, goal=True) -> tuple[Environment, Any]:
    env: Environment | None = None

    if env_config.env_source == "gymnax":
        env, params = gymnax.make(env_config.env_name, **env_config.env_kwargs)
        env = gymnax_wrapper(env)
        if goal:
            env = goal_wrapper(env)
    elif env_config.env_source == "brax":
        # env = brax.envs.get_environment(env_config.env_name, **env_config.env_kwargs)
        ...
    elif env_config.env_source == "navix":
        env = nx.make(env_config.env_name, **env_config.env_kwargs)
        env = navix_wrapper(env)

    elif env_config.env_source == "custom":
        if env_config.env_name == "MultiCatch":
            assert goal, "Multitask requires goal"
            env, params = make_multi_catch(**env_config.env_kwargs)
            env = auto_reset_wrapper(env)

    if env is None:
        raise ValueError(
            f"Unrecognized environment {env_config.env_name} in {env_config.env_source}"
        )

    env = flatten_observation_wrapper(env)
    env = log_wrapper(env)
    return env, params


# def housemaze_wrapper(env: housemaze.env.HouseMaze):
