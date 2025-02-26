"""Multitask version of Catch bsuite environment.

Source: github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py.
"""

import dataclasses as dc

import gymnax.environments.spaces as spaces
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer, Key

from log_util import dataclass
from wrappers.common import Environment, GoalObs, StepType, Timestep

Obs = Float[Array, " rows columns"]


@dataclass
class EnvState:
    ball_x: Integer[Array, ""]
    ball_y: Integer[Array, ""]
    paddle_x: Integer[Array, ""]
    ball_type: Integer[Array, ""]
    goal: Integer[Array, ""]


@dataclass
class EnvParams:
    rows: int = 10
    columns: int = 5
    num_goals: int = 2


def make_multi_catch(
    **kwargs,
) -> tuple[Environment[GoalObs[Obs], EnvState, Integer[Array, ""], EnvParams], EnvParams]:
    """JAX Compatible version of Catch bsuite environment."""

    def _get_obs(state: EnvState, params: EnvParams) -> GoalObs[Obs]:
        obs = (
            jnp.zeros((params.rows, params.columns))
            .at[state.ball_y, state.ball_x]
            .set(1.0)
            .at[params.rows - 1, state.paddle_x]
            .set(1.0)
        )
        return GoalObs(
            obs=obs,
            goal=state.goal,
        )

    def reset(params: EnvParams, *, key: Key[Array, ""]) -> tuple[GoalObs[Obs], EnvState]:
        """Randomly sample ball column, ball type, and goal."""
        key_ball, key_type, key_goal = jr.split(key, 3)
        state = EnvState(
            ball_x=jr.randint(key_ball, (), 0, params.columns, int),
            ball_y=jnp.zeros((), int),
            paddle_x=jnp.asarray(params.columns // 2, int),
            ball_type=jr.randint(key_type, (), 0, params.num_goals),
            goal=jr.randint(key_goal, (), 0, params.num_goals),
        )
        return _get_obs(state, params), state

    def step(
        state: EnvState,
        action: Integer[Array, ""],
        params: EnvParams,
        *,
        key: Key[Array, ""],
    ) -> Timestep[GoalObs[Obs], EnvState]:
        """Perform single timestep state transition."""

        dx = action - 1  # [-1, 0, 1] = left, no-op, right
        paddle_x = jnp.clip(state.paddle_x + dx, 0, params.columns - 1)
        new_state = dc.replace(
            state,
            ball_y=state.ball_y + 1,
            paddle_x=paddle_x,
        )

        done = new_state.ball_y == params.rows - 1
        missed = paddle_x != new_state.ball_x
        matched = state.goal == state.ball_type
        success = jnp.logical_xor(missed, matched)
        reward = done * jnp.where(success, 1.0, -1.0)

        # Check number of steps in episode termination condition
        return Timestep(
            obs=_get_obs(new_state, params),
            state=new_state,
            reward=reward,
            step_type=jnp.where(done, StepType.TERMINATION, StepType.TRANSITION),
            info={},
        )

    def action_space(params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(3)

    def observation_space(params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, (params.rows, params.columns))

    def goal_space(params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(2)

    return Environment(
        _inner=None,
        reset=reset,
        step=step,
        action_space=action_space,
        observation_space=observation_space,
        goal_space=goal_space,
    ), EnvParams(**kwargs)
