"""Multitask version of Catch bsuite environment.

Source: github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py.
"""

import dataclasses as dc

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer, Key

from envs.base import Environment
from utils import specs
from utils.structures import GoalObs, StepType, TimeStep, dataclass

BaseObs = Float[Array, " rows columns"]


@dataclass
class EnvState:
    ball_x: Integer[Array, ""]
    ball_y: Integer[Array, ""]
    paddle_x: Integer[Array, ""]
    goal: Integer[Array, ""]


@dataclass
class EnvParams:
    rows: int = 10
    columns: int = 5
    num_goals: int = 2


def make_multi_catch(
    **kwargs,
) -> tuple[
    Environment[GoalObs[BaseObs], EnvState, Integer[Array, ""], EnvParams],
    EnvParams,
]:
    """JAX-compatible multitask version of Catch bsuite environment.

    If the goal is zero, reward is given for _missing_ the ball.
    Otherwise reward is given for _catching_ the ball.
    """

    def _get_obs(state: EnvState, params: EnvParams) -> GoalObs[BaseObs]:
        """Observe the state."""
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

    def reset(params: EnvParams, *, key: Key[Array, ""]) -> TimeStep[GoalObs[BaseObs], EnvState]:
        """Randomly sample ball column and goal."""
        key_ball, key_goal = jr.split(key)
        state = EnvState(
            ball_x=jr.randint(key_ball, (), 0, params.columns, int),
            ball_y=jnp.zeros((), int),
            paddle_x=jnp.asarray(params.columns // 2, int),
            goal=jr.randint(key_goal, (), 0, params.num_goals),
        )
        return TimeStep.initial(obs=_get_obs(state, params), state=state, info={})

    def step(
        env_state: EnvState,
        action: Integer[Array, ""],
        params: EnvParams,
        *,
        key=None,
    ) -> TimeStep[GoalObs[BaseObs], EnvState]:
        """Perform single timestep state transition."""

        dx = action - 1  # [-1, 0, 1] = left, no-op, right
        paddle_x = jnp.clip(env_state.paddle_x + dx, 0, params.columns - 1)
        next_state = dc.replace(
            env_state,
            ball_y=env_state.ball_y + 1,
            paddle_x=paddle_x,
        )

        terminal = next_state.ball_y == params.rows - 1
        missed = paddle_x != next_state.ball_x
        # if goal is zero, catch is success
        # if goal is nonzero, miss is success
        success = jnp.logical_xor(missed, env_state.goal == 0)
        reward = terminal * jnp.where(success, 1.0, -1.0)

        return TimeStep(
            obs=_get_obs(next_state, params),
            state=next_state,
            reward=reward,
            discount=1.0 - terminal,
            step_type=jnp.where(terminal, StepType.LAST, StepType.MID),
            info={},
        )

    def action_space(params: EnvParams):
        return specs.BoundedArray.discrete(3, name="action")

    def observation_space(params: EnvParams):
        return specs.BoundedArray(
            shape=(params.rows, params.columns),
            dtype=float,
            minimum=0,
            maximum=1,
            name="observation",
        )

    def goal_space(params: EnvParams):
        return specs.BoundedArray.discrete(params.num_goals, name="goal")

    return Environment(
        _inner=None,  # type: ignore
        name="MultiCatch",
        reset=reset,
        step=step,
        action_space=action_space,
        observation_space=observation_space,
        goal_space=goal_space,
    ), EnvParams(**kwargs)


def visualize_catch(
    env_state: EnvState,
) -> Float[Array, " channel height width"]:
    """Turn a sequence of Catch environment states into a wandb.Video matrix."""

    obs_shape = (10, 5)
    video_shape = (3, *obs_shape)

    video = (
        jnp.full(video_shape, 255, dtype=jnp.uint8)
        .at[:, env_state.ball_y, env_state.ball_x]
        .set(jnp.array([255, 0, 0], dtype=jnp.uint8))
        .at[:, obs_shape[0] - 1, env_state.paddle_x]
        .set(jnp.array([0, 255, 0], dtype=jnp.uint8))
    )
    return video
