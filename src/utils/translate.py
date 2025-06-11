from typing import Any

from utils.housemaze import housemaze_wrapper, new_housemaze, visualize_housemaze
from utils.structures import (
    GYMNAX_INSTALLED,
    NAVIX_INSTALLED,
    Environment,
    StepType,
    TimeStep,
)

if GYMNAX_INSTALLED:
    import gymnax.environments.environment as ge
    import gymnax.environments.spaces as spaces

if NAVIX_INSTALLED:
    import navix as nx

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Key, PyTree

from experiments.config import EnvConfig
from testing.dummy_env import Params, make_dummy_env
from utils.auto_reset import auto_reset_wrapper
from utils.flatten_observation import flatten_observation_wrapper
from utils.goal_wrapper import goal_wrapper
from utils.log import log_wrapper
from utils.multi_catch import make_multi_catch, visualize_catch

if GYMNAX_INSTALLED:

    def gymnax_wrapper(
        env: ge.Environment[ge.TEnvState, ge.TEnvParams], params: ge.TEnvParams
    ):
        _, init_state = env.reset(jr.key(0), params)
        _, init_step = env.step(
            jr.key(0), init_state, env.action_space(params).sample(jr.key(0)), params
        )
        init_info = jax.tree.map(jnp.empty_like, init_step[-1])

        def reset(params: ge.TEnvParams, *, key: Key[Array, ""]):
            obs, state = env.reset(key, params)
            return TimeStep.initial(obs, state, init_info)

        def step(
            state: ge.TEnvState,
            action: Any,
            params: ge.TEnvParams,
            *,
            key: Key[Array, ""],
        ):
            obs, state, reward, done, info = env.step(key, state, action, params)
            return TimeStep(
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


if NAVIX_INSTALLED:

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
            return TimeStep(
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

    if env_config.source == "gymnax":
        env, params = gymnax.make(env_config.name, **env_config.kwargs)
        env = gymnax_wrapper(env)
        env = flatten_observation_wrapper(env)
        if goal:
            env = goal_wrapper(env)
    elif env_config.source == "brax":
        # env = brax.envs.get_environment(env_config.env_name, **env_config.kwargs)
        ...
    elif env_config.source == "navix":
        env = nx.make(env_config.name, **env_config.kwargs)
        env = navix_wrapper(env)

    elif env_config.source == "custom":
        if env_config.name == "MultiCatch":
            assert goal, "Multitask requires goal"
            env, params = make_multi_catch(**env_config.kwargs)
            env = flatten_observation_wrapper(env)
            env = auto_reset_wrapper(env)

        elif env_config.name == "HouseMaze":
            env, params = new_housemaze()
            env = housemaze_wrapper(env)
            env = auto_reset_wrapper(env)

        elif env_config.name == "dummy":
            env, params = make_dummy_env(**env_config.kwargs)
            if goal:
                env = goal_wrapper(env)

    if env is None:
        raise ValueError(
            f"Unrecognized environment {env_config.name} in {env_config.source}"
        )

    env = log_wrapper(env)
    return env, params


SUPPORTED_VIDEO_ENVS = [
    "Catch-bsuite",
    "MultiCatch",
    "HouseMaze",
]


def visualize_env_state_frame(
    env_name: str, env_state: PyTree[Array], **kwargs
) -> Float[Array, " channel height width"]:
    """Visualize the environment.

    Returns:
        tuple[Float (horizon, channel, height, width), list[str]]: The recorded video;
            The sequence of labels to append to the titles.
    """
    if env_name in ["Catch-bsuite", "MultiCatch"]:
        return jax.jit(visualize_catch)(env_state, **kwargs)
    if env_name == "HouseMaze":
        return jax.jit(visualize_housemaze)(env_state)
    raise ValueError(f"Env {env_name} not recognized")
