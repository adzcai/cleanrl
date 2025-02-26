from typing import Any

import gymnax

from config import EnvConfig
from wrappers.auto_reset import auto_reset_wrapper
from wrappers.common import Environment
from wrappers.flatten_observation import flatten_observation_wrapper
from wrappers.goal_wrapper import goal_wrapper
from wrappers.gymnax_wrapper import gymnax_wrapper
from wrappers.log import log_wrapper
from wrappers.multi_catch import make_multi_catch


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
