"""Base classes for gymnax environment wrappers."""

import gymnax.environments.environment as gymenv
from jaxtyping import Bool, Real, Key, Float, Array
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, Union

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

# generics to distinguish base environment state and wrapper state
# use gymenv.TEnvState for general
TBaseState = TypeVar("TBaseEnvState", bound=gymenv.EnvState)
TWrapperState = TypeVar("TWrapperState", bound="WrapperState")
TObs = TypeVar("Obs")


class Timestep(NamedTuple, Generic[TObs, gymenv.TEnvState]):
    obs: TObs
    env_state: gymenv.TEnvState
    reward: Float[Array, ""]
    done: Bool[Array, ""]
    info: dict[Any, Any]


@dataclass(frozen=True)
class WrapperState(Generic[TBaseState]):
    _env_state: TBaseState

    def __getattr__(self, name):
        return getattr(self._env_state, name)


class Wrapper(Generic[TObs, TBaseState, TWrapperState, gymenv.TEnvParams]):
    """Base class for environment wrappers."""

    _env: gymenv.Environment[TBaseState, gymenv.TEnvParams]

    def __init__(self, env: gymenv.Environment[TBaseState, gymenv.TEnvParams]):
        self._env = env

    def reset(self, key: Key[Array, ""], params: gymenv.TEnvParams) -> tuple[TObs, TBaseState]:
        return self._env.reset(key, params)

    def step(
        self,
        key: Key[Array, ""],
        state: TWrapperState,
        action: Union[int, float, Real[Array, ""]],
        params: gymenv.TEnvParams,
    ) -> Timestep[TObs, TWrapperState]:
        return Timestep(*self._env.step(key, state._env_state, action, params))

    def __getattr__(self, name):
        return getattr(self._env, name)
