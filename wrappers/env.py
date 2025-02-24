import dataclasses as dc
from dataclasses import dataclass
from typing import Literal

import gymnax

from wrappers.auto_reset import auto_reset_wrapper
from wrappers.common import Environment, TAction, TEnvParams, TEnvState, Timestep, TObs
from wrappers.flatten_observation import flatten_observation_wrapper
from wrappers.goal import goal_wrapper
from wrappers.log import LogWrapperState, log_wrapper
from wrappers.multi_catch import make_multi_catch


@dataclass(frozen=True)
class EnvConfig:
    env_name: str
    horizon: int  # also mcts max depth
    env_source: Literal["gymnax", "brax", "custom"] = "gymnax"
    env_kwargs: dict = dc.field(default_factory=dict)


def make_env(
    env_config: EnvConfig, num_episodes: int
) -> tuple[Environment[TObs, LogWrapperState[TEnvState], TAction, TEnvParams], TEnvParams]:
    env: Environment | None = None

    if env_config.env_source == "gymnax":
        gymnax_env, env_params = gymnax.make(env_config.env_name, **env_config.env_kwargs)
        env = Environment(
            reset=gymnax_env.reset,
            step=lambda *args, **kwargs: Timestep(*gymnax_env.step(*args, **kwargs)),
            action_space=gymnax_env.action_space,
            observation_space=gymnax_env.observation_space,
        )
        env = goal_wrapper(env)
    elif env_config.env_source == "brax":
        # env = brax.envs.get_environment(env_config.env_name, **env_config.env_kwargs)
        ...
    elif env_config.env_source == "custom":
        if env_config.env_name == "MultiCatch":
            env, env_params = make_multi_catch(**env_config.env_kwargs)
            env = auto_reset_wrapper(env)

    if env is None:
        raise ValueError(
            f"Unrecognized environment {env_config.env_name} in {env_config.env_source}"
        )

    env = flatten_observation_wrapper(env)
    env = log_wrapper(env, num_episodes=num_episodes)
    return env, env_params
