import jax
import jax.numpy as jnp
import gymnax.environments.environment as gymenv

from typing import TYPE_CHECKING
from jaxtyping import Real, UInt, Key, Float, Array
from wrappers.common import TObs, TBaseState, Timestep, Wrapper, WrapperState
import dataclasses as dc

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass(frozen=True)
class LogWrapperState(WrapperState[TBaseState]):
    current_return: Float[Array, ""]
    current_length: Float[Array, ""]
    current_index: UInt[Array, ""]
    episode_returns: Float[Array, " num_episodes"]
    episode_lengths: UInt[Array, " num_episodes"]

    def average_return(self):
        return jnp.sum(self.episode_returns, axis=-1) / self.current_index


class LogWrapper(Wrapper[TObs, TBaseState, LogWrapperState[TBaseState], gymenv.TEnvParams]):
    """Log interactions and episode rewards."""

    _num_episodes: int

    def __init__(
        self,
        env: gymenv.Environment[TBaseState, gymenv.TEnvParams],
        num_episodes: int,
    ):
        super().__init__(env)
        self._num_episodes = num_episodes

    def reset(self, key: Key[Array, ""], params: gymenv.TEnvParams) -> tuple[TObs, TBaseState]:
        obs, env_state = super().reset(key, params)
        env_state = LogWrapperState(
            current_return=jnp.zeros((), float),
            current_length=jnp.zeros((), jnp.uint32),
            current_index=jnp.zeros((), jnp.uint32),
            episode_returns=jnp.zeros(self._num_episodes),
            episode_lengths=jnp.zeros(self._num_episodes, jnp.uint32),
            _env_state=env_state,
        )
        return obs, env_state

    def step(
        self,
        key: Key[Array, ""],
        state: LogWrapperState[TBaseState],
        action: int | float | Real[Array, ""],
        params: gymenv.TEnvParams,
    ) -> Timestep[TObs, LogWrapperState[TBaseState]]:
        timestep = super().step(key, state, action, params)

        env_state = jax.lax.cond(
            timestep.done,
            lambda state, timestep: LogWrapperState(
                current_return=jnp.zeros((), float),
                current_length=jnp.zeros((), jnp.uint32),
                current_index=state.current_index + 1,
                episode_returns=state.episode_returns.at[state.current_index].set(
                    state.current_return + timestep.reward
                ),
                episode_lengths=state.episode_lengths.at[state.current_index].set(
                    state.current_length + 1
                ),
                _env_state=timestep.env_state,
            ),
            lambda state, timestep: dc.replace(
                state,
                current_return=state.current_return + timestep.reward,
                current_length=state.current_length + 1,
                _env_state=timestep.env_state,
            ),
            state,
            timestep,
        )

        return timestep._replace(env_state=env_state)
