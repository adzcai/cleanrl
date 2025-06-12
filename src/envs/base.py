from collections.abc import Callable
from typing import Generic

from utils import specs
from utils.structures import (
    ResetFn,
    StepFn,
    TAction,
    TEnvParams,
    TEnvState,
    TObs,
    dataclass,
)
from wrappers.base import Wrapper


@dataclass
class Environment(
    Wrapper["Environment"], Generic[TObs, TEnvState, TAction, TEnvParams]
):
    """An interface for an interactive environment.

    We allow the action, observation, and goal spaces to vary.
    """

    name: str
    reset: ResetFn[TObs, TEnvState, TEnvParams]
    step: StepFn[TObs, TEnvState, TAction, TEnvParams]
    action_space: Callable[[TEnvParams], specs.SpecTree]
    observation_space: Callable[[TEnvParams], specs.SpecTree]
    goal_space: Callable[[TEnvParams], specs.BoundedArray]

    @property
    def fullname(self):
        """Add all the wrapper names."""
        env = self
        name = self.name
        while getattr(env, "_inner", None) is not None:
            env = env._inner
            name = env.name + " > " + name
        return name
