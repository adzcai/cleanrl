from typing import Any, NamedTuple

import dm_env.specs as specs

from wrappers.base import (
    GYMNAX_INSTALLED,
    NAVIX_INSTALLED,
    Environment,
    GoalObs,
    StepType,
    Timestep,
)

if GYMNAX_INSTALLED:
    import gymnax.environments.environment as ge
    import gymnax.environments.spaces as spaces

if NAVIX_INSTALLED:
    import navix as nx

import housemaze.env as maze
import housemaze.levels as maze_levels
import housemaze.renderer as renderer
import housemaze.utils as maze_utils
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer, Key, PyTree

from config import EnvConfig
from wrappers.auto_reset import auto_reset_wrapper
from wrappers.flatten_observation import flatten_observation_wrapper
from wrappers.goal_wrapper import goal_wrapper
from wrappers.log import log_wrapper
from wrappers.multi_catch import make_multi_catch, visualize_catch

if GYMNAX_INSTALLED:

    def gymnax_wrapper(env: ge.Environment[ge.TEnvState, ge.TEnvParams], params: ge.TEnvParams):
        _, init_state = env.reset(jr.key(0), params)
        _, init_step = env.step(
            jr.key(0), init_state, env.action_space(params).sample(jr.key(0)), params
        )
        init_info = jax.tree.map(jnp.empty_like, init_step[-1])

        def reset(params: ge.TEnvParams, *, key: Key[Array, ""]):
            obs, state = env.reset(key, params)
            return Timestep.initial(obs, state, init_info)

        def step(
            state: ge.TEnvState,
            action: Any,
            params: ge.TEnvParams,
            *,
            key: Key[Array, ""],
        ):
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


if NAVIX_INSTALLED:

    def navix_wrapper(env: nx.Environment):
        def reset(params: None, *, key: Key[Array, ""]):
            timestep = env.reset(key)
            return timestep.observation, timestep

        def step(state: nx.Timestep, action: Any, params: None, *, key: Key[Array, ""]):
            timestep = env.step(state, action)
            step_type = jnp.select(
                [timestep.is_termination(), timestep.is_truncation()],
                [StepType.TERMINATION, StepType.TRUNCATION],
                default=StepType.TRANSITION,
            )
            return Timestep(
                obs=timestep.observation,
                state=timestep,
                reward=timestep.reward,
                step_type=step_type,
                info=timestep.info,
            )

        return Environment(
            _inner=env,
            reset=reset,
            step=step,
            action_space=lambda params: env.action_space,
            observation_space=lambda params: env.observation_space,
            goal_space=lambda params: spaces.Discrete(0),
        )


image_dict = maze_utils.load_image_dict()


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
            maze_utils.from_str(map_str, char_to_key=char_to_key, object_to_index=object_to_index)
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
        time_limit=jnp.asarray(50),
        objects=jnp.asarray(objects),
    )
    task_runner = maze.TaskRunner(task_objects=env_params.objects)
    env = maze.HouseMaze(task_runner=task_runner, num_categories=len(image_dict["keys"]))
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
            direction=specs.DiscreteArray(len(maze.DIR_TO_VEC), name="direction"),
            row=specs.DiscreteArray(height, name="row"),
            column=specs.DiscreteArray(width, name="width"),
            # housemaze reset_action sets to num_actions + 1
            prev_action=specs.DiscreteArray(env.num_actions(params) + 2, name="prev_action"),
        )

    def reset(params: maze.EnvParams, *, key: Key[Array, ""]):
        env_timestep = env.reset(key, params)
        return Timestep.initial(
            obs=_translate_obs(env_timestep.observation, params),
            state=env_timestep,
            info={},
        )

    def step(
        env_state: maze.TimeStep,
        action: Integer[Array, ""],
        params: maze.EnvParams,
        *,
        key: Key[Array, ""],
    ):
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
        return Timestep(
            state=env_timestep,
            obs=_translate_obs(obs, params),
            reward=env_timestep.reward,
            discount=env_timestep.discount,
            step_type=step_type,
            info={},
        )

    return Environment(
        _inner=None,
        name="HouseMaze",
        reset=reset,
        step=step,
        action_space=lambda params: specs.DiscreteArray(env.num_actions(params), name="action"),
        observation_space=observation_space,
        goal_space=lambda params: specs.DiscreteArray(len(params.objects), name="goal"),
    )


def visualize_housemaze(
    timestep: maze.TimeStep,
) -> Float[Array, " horizon channel height width"]:
    video: Float[Array, " horizon height width channel"] = jax.vmap(
        renderer.create_image_from_grid, in_axes=(0, 0, 0, None)
    )(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_dict,
    )
    return video.transpose((0, 3, 1, 2))  # C H W


def make_env(env_config: EnvConfig, goal=True) -> tuple[Environment, Any]:
    env: Environment | None = None

    if env_config.source == "gymnax":
        env, params = gymnax.make(env_config.name, **env_config.kwargs)
        env = gymnax_wrapper(env)
        env = flatten_observation_wrapper(env)
        if goal:
            env = goal_wrapper(env)
    elif env_config.source == "brax":
        # env = brax.envs.get_environment(env_config.env_name, **env_config.kwargs)
        ...
    elif env_config.source == "navix":
        env = nx.make(env_config.name, **env_config.kwargs)
        env = navix_wrapper(env)

    elif env_config.source == "custom":
        if env_config.name == "MultiCatch":
            assert goal, "Multitask requires goal"
            env, params = make_multi_catch(**env_config.kwargs)
            env = flatten_observation_wrapper(env)
            env = auto_reset_wrapper(env)

        elif env_config.name == "HouseMaze":
            env, params = new_housemaze()
            env = housemaze_wrapper(env)
            env = auto_reset_wrapper(env)

    if env is None:
        raise ValueError(f"Unrecognized environment {env_config.name} in {env_config.source}")

    env = log_wrapper(env)
    return env, params


def visualize(env_name: str, env_state: PyTree[Array], **kwargs):
    if env_name in ["Catch-bsuite", "MultiCatch"]:
        return visualize_catch(env_state, **kwargs)
    if env_name == "HouseMaze":
        return visualize_housemaze(env_state)
    raise ValueError(f"Env {env_name} not recognized")


def get_action_name(env_name: str, action: int):
    if env_name in ["Catch-bsuite", "MultiCatch"]:
        if action == 0:
            return "L"
        elif action == 1:
            return "N"
        elif action == 2:
            return "R"
        else:
            raise ValueError(f"Invalid action {action}")
    elif env_name == "HouseMaze":
        return ["➡️", "⬇️", "⬅️", "⬆️", "done", "NONE", "reset"][action]
    else:
        raise ValueError(f"Env {env_name} not recognized")
