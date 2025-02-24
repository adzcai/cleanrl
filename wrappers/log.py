import dataclasses as dc
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, UInt

from wrappers.common import (
    Environment,
    TAction,
    TEnvParams,
    TEnvState,
    Timestep,
    TObs,
    WrapperState,
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass(frozen=True)
class LogWrapperState(WrapperState[TEnvState]):
    current_return: Float[Array, ""]
    current_length: Float[Array, ""]
    current_index: UInt[Array, ""]
    episode_returns: Float[Array, " num_episodes"]
    episode_lengths: UInt[Array, " num_episodes"]

    def average_return(self):
        return jnp.sum(self.episode_returns, axis=-1) / self.current_index


def log_wrapper(
    env: Environment[TObs, TEnvState, TAction, TEnvParams],
    num_episodes: int,
) -> Environment[TObs, LogWrapperState[TEnvState], TAction, TEnvParams]:
    """Log interactions and episode rewards."""

    def reset(key: Key[Array, ""], params: TEnvParams) -> tuple[TObs, TEnvState]:
        obs, env_state = env.reset(key, params)
        env_state = LogWrapperState(
            _env_state=env_state,
            current_return=jnp.zeros((), float),
            current_length=jnp.zeros((), jnp.uint32),
            current_index=jnp.zeros((), jnp.uint32),
            episode_returns=jnp.zeros(num_episodes),
            episode_lengths=jnp.zeros(num_episodes, jnp.uint32),
        )
        return obs, env_state

    def step(
        key: Key[Array, ""],
        state: LogWrapperState[TEnvState],
        action: TAction,
        params: TEnvParams,
    ) -> Timestep[TObs, LogWrapperState[TEnvState]]:
        timestep = env.step(key, state._env_state, action, params)

        env_state = jax.lax.cond(
            timestep.done,
            # start new episode if done
            lambda state, timestep: LogWrapperState(
                _env_state=timestep.env_state,
                current_return=jnp.zeros((), float),
                current_length=jnp.zeros((), jnp.uint32),
                current_index=state.current_index + 1,
                episode_returns=state.episode_returns.at[state.current_index].set(
                    state.current_return + timestep.reward
                ),
                episode_lengths=state.episode_lengths.at[state.current_index].set(
                    state.current_length + 1
                ),
            ),
            # otherwise update current state
            lambda state, timestep: dc.replace(
                state,
                _env_state=timestep.env_state,
                current_return=state.current_return + timestep.reward,
                current_length=state.current_length + 1,
            ),
            state,
            timestep,
        )

        return timestep._replace(env_state=env_state)

    return dc.replace(env, reset=reset, step=step)
