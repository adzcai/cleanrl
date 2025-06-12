from typing import Any, NamedTuple

import housemaze.env as maze
import housemaze.levels as maze_levels
import housemaze.renderer as renderer
import housemaze.utils as maze_utils
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Key, PyTree

from envs.base import Environment
from utils import specs
from utils.structures import GoalObs, StepType, TimeStep


class HouseMazeObs(NamedTuple):
    """A categorical encoding of the observation."""

    image: Integer[Array, " height width"]
    """`image[row, column]` stores the type of object within that cell."""
    # state_features
    direction: Integer[Array, ""]
    """The agent's direction (0 = right < down < left < up < done = 4)"""
    row: Integer[Array, ""]
    """The agent's row coordinate in `image`."""
    column: Integer[Array, ""]
    """The agent's column coordinate in `image`."""
    prev_action: Integer[Array, ""]
    """The previous action."""


image_dict = maze_utils.load_image_dict()


def new_housemaze():
    """Create a new HouseMaze environment."""
    # initialize map
    char_to_key = dict(
        A="knife",
        B="fork",
        C="pan",
        D="pot",
        E="bowl",
        F="plates",
    )
    object_to_index = {key: idx for idx, key in enumerate(image_dict["keys"])}
    if False:
        # switch between multiple levels
        levels = [maze_levels.three_pairs_maze1, maze_levels.two_objects]
        map_init = [
            maze_utils.from_str(
                map_str, char_to_key=char_to_key, object_to_index=object_to_index
            )
            for map_str in levels
        ]
        map_init = jax.tree.map(lambda *x: jnp.stack(x), *map_init)
    else:
        map_init = maze_utils.from_str(
            maze_levels.two_objects,
            char_to_key=char_to_key,
            object_to_index=object_to_index,
        )
    map_init = maze.MapInit(*map_init)

    # create env params
    objects = jnp.array([object_to_index[v] for v in char_to_key.values()])
    env_params = maze.EnvParams(
        map_init=jax.tree.map(jnp.asarray, map_init),
        time_limit=50,
        objects=jnp.asarray(objects),
    )
    task_runner = maze.TaskRunner(task_objects=env_params.objects)
    env = maze.HouseMaze(
        task_runner=task_runner, num_categories=len(image_dict["keys"])
    )
    return env, env_params


def housemaze_wrapper(
    env: maze.HouseMaze,
) -> Environment[Any, maze.TimeStep, Integer[Array, ""], maze.EnvParams]:
    def _translate_obs(env_obs: maze.Observation, params: maze.EnvParams):
        row, column = env_obs.position
        # insert 0 and 1 as valid keys
        objects = jnp.concat([jnp.arange(2), params.objects])
        image = jnp.argmax(env_obs.image[:, :, jnp.newaxis] == objects, axis=-1)
        obs = HouseMazeObs(
            image=image.ravel(),  # flatten
            # "state_features": env_obs.state_features,
            direction=env_obs.direction,
            row=row,
            column=column,
            prev_action=env_obs.prev_action,
        )
        goal = jnp.argmax(env_obs.task_w)
        return GoalObs(obs=obs, goal=goal)

    def observation_space(params: maze.EnvParams) -> PyTree[specs.Array]:
        height, width = params.map_init.grid.shape[:-1]
        num_objects = len(params.objects) + 2  # for 0 and 1
        return HouseMazeObs(
            image=specs.BoundedArray(
                shape=(height * width,),
                dtype=int,
                minimum=0,
                maximum=num_objects - 1,
                name="image",
            ),
            # "state_features": spaces.Box(low=-jnp.inf, high=jnp.inf, dtype=float),
            direction=specs.BoundedArray.discrete(
                len(maze.DIR_TO_VEC), name="direction"
            ),
            row=specs.BoundedArray.discrete(height, name="row"),
            column=specs.BoundedArray.discrete(width, name="width"),
            # housemaze reset_action sets to num_actions + 1
            prev_action=specs.BoundedArray.discrete(
                env.num_actions(params) + 2, name="prev_action"
            ),
        )

    def reset(params: maze.EnvParams, *, key: Key[Array, ""]):
        env_timestep = env.reset(key, params)
        return TimeStep.initial(
            obs=_translate_obs(env_timestep.observation, params),
            state=env_timestep,
            info={},
        )

    def step(
        env_state: maze.TimeStep,
        action: Integer[Array, ""],
        params: maze.EnvParams,
        *,
        key: Key[Array, ""] | None = None,
    ):
        assert key is not None, "Key required for stochastic transitions."
        env_timestep = env.step(key, env_state, action, params)
        obs = env_timestep.observation
        step_type = jnp.select(
            [
                env_timestep.step_type == tp
                for tp in [maze.StepType.FIRST, maze.StepType.MID, maze.StepType.LAST]
            ],
            [StepType.FIRST, StepType.MID, StepType.LAST],
            default=-1,
        )
        return TimeStep(
            state=env_timestep,
            obs=_translate_obs(obs, params),
            # small negative reward to encourage short trajectories
            reward=jnp.where(step_type == StepType.MID, -0.1, env_timestep.reward),
            discount=env_timestep.discount,
            step_type=step_type,
            info={},
        )

    return Environment(
        _inner=None,  # type: ignore
        name="HouseMaze",
        reset=reset,
        step=step,
        action_space=lambda params: specs.BoundedArray.discrete(
            env.num_actions(params), name="action"
        ),
        observation_space=observation_space,
        goal_space=lambda params: specs.BoundedArray.discrete(
            len(params.objects), name="goal"
        ),
    )


def visualize_housemaze(
    timestep: maze.TimeStep,
) -> Float[Array, " channel height width"]:
    r, c = timestep.state.agent_pos
    video: Float[Array, " height width channel"] = jax.vmap(
        renderer.create_image_from_grid, in_axes=(0, 0, 0, None)
    )(
        timestep.state.grid,
        (r, c),
        timestep.state.agent_dir,
        image_dict,
    )
    return video.transpose((2, 0, 1))  # C H W
