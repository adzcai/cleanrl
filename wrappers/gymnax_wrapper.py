from typing import Any

import gymnax.environments.environment as ge
import gymnax.environments.spaces as spaces
import jax.numpy as jnp
from jaxtyping import Array, Key

from wrappers.common import Environment, StepType, Timestep


def gymnax_wrapper(env: ge.Environment[ge.TEnvState, ge.TEnvParams]):
    def reset(params: ge.TEnvParams, *, key: Key[Array, ""]):
        obs, state = env.reset(key, params)
        return obs, state

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
