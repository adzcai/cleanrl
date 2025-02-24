"""Multitask version of Catch bsuite environment.

Source: github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py.
"""

import dataclasses as dc
from typing import TYPE_CHECKING

import gymnax.environments.spaces as spaces
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Key

from wrappers.common import Environment, GoalObs, Timestep

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

Obs = Float[Array, "rows columns"]


@dataclass(frozen=True)
class EnvState:
    ball_x: Integer[Array, ""]
    ball_y: Integer[Array, ""]
    paddle_x: Integer[Array, ""]
    ball_type: Integer[Array, ""]
    goal: Integer[Array, ""]


def make_multi_catch(
    rows: int = 5,
    columns: int = 5,
    num_goals: int = 2,
) -> tuple[Environment[GoalObs[Obs], EnvState, Integer[Array, ""], None], None]:
    """JAX Compatible version of Catch bsuite environment."""

    def _get_obs(state: EnvState) -> GoalObs[Obs]:
        obs = (
            jnp.zeros((rows, columns))
            .at[state.ball_y, state.ball_x]
            .set(1.0)
            .at[rows - 1, state.paddle_x]
            .set(1.0)
        )
        return GoalObs(
            obs=obs,
            goal=state.goal,
        )

    def reset(key: Key[Array, ""], params: None) -> tuple[GoalObs[Obs], EnvState]:
        """Randomly sample ball column, ball type, and goal."""
        key_ball, key_type, key_goal = jax.random.split(key, 3)
        state = EnvState(
            ball_x=jax.random.randint(key_ball, (), 0, columns, int),
            ball_y=jnp.zeros((), int),
            paddle_x=jnp.asarray(columns // 2, int),
            ball_type=jax.random.randint(key_type, (), 0, num_goals),
            goal=jax.random.randint(key_goal, (), 0, num_goals),
        )
        return _get_obs(state), state

    def step(
        key: Key[Array, ""],
        state: EnvState,
        action: Integer[Array, ""],
        params: None,
    ) -> Timestep[GoalObs[Obs], EnvState]:
        """Perform single timestep state transition."""

        dx = action - 1  # [-1, 0, 1] = left, no-op, right
        paddle_x = jnp.clip(state.paddle_x + dx, 0, columns - 1)
        new_state = dc.replace(
            state,
            ball_y=state.ball_y + 1,
            paddle_x=paddle_x,
        )

        done = new_state.ball_y == rows - 1
        missed = paddle_x != new_state.ball_x
        matched = state.goal == state.ball_type
        success = jnp.logical_xor(missed, matched)
        reward = done * jnp.where(success, 1.0, -1.0)

        # Check number of steps in episode termination condition
        return Timestep(
            _get_obs(new_state),
            new_state,
            reward,
            done,
            None,
        )

    def action_space(params: None) -> spaces.Discrete:
        return spaces.Discrete(3)

    def observation_space(params: None) -> spaces.Box:
        return spaces.Box(0, 1, (rows, columns))

    def goal_space(params: None) -> spaces.Discrete:
        return spaces.Discrete(2)

    return Environment(
        reset=reset,
        step=step,
        action_space=action_space,
        observation_space=observation_space,
        goal_space=goal_space,
    ), None
