# jax
import jax
import jax.numpy as jnp

import gymnax.environments.environment as gymenv

# typing
from jaxtyping import Bool, Real, UInt, Key, Float, Array
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Union

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

# util
import dataclasses as dc


@dataclass(frozen=True)
class LogWrapperState(Generic[gymenv.TEnvState]):
    current_return: Float[Array, ""]
    current_length: Float[Array, ""]
    current_index: UInt[Array, ""]
    episode_returns: Float[Array, " num_episodes"]
    episode_lengths: UInt[Array, " num_episodes"]
    _env_state: gymenv.TEnvState

    def average_return(self):
        return jnp.sum(self.episode_returns, axis=-1) / self.current_index

    def __getattr__(self, name):
        return getattr(self._env_state, name)


class Timestep(NamedTuple, Generic[gymenv.TEnvState]):
    obs: Float[Array, " obs_size"]
    env_state: gymenv.TEnvState
    reward: Float[Array, ""]
    done: Bool[Array, ""]
    info: dict[Any, Any]


class LogWrapper:
    """Log interactions and episode rewards."""

    def __init__(
        self,
        env: gymenv.Environment[gymenv.TEnvState, gymenv.TEnvParams],
        num_episodes: int,
    ):
        self._env = env
        self._num_episodes = num_episodes

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(
        self, key: Key[Array, ""], params: gymenv.TEnvParams
    ) -> tuple[Float[Array, " obs_size"], gymenv.TEnvState]:
        obs, env_state = self._env.reset(key, params)
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
        state: LogWrapperState[gymenv.TEnvState],
        action: Union[int, float, Real[Array, ""]],
        params: gymenv.TEnvParams,
    ) -> Timestep[LogWrapperState[gymenv.TEnvState]]:
        timestep = Timestep(*self._env.step(key, state._env_state, action, params))

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
