from typing import Any

import gymnax.environments.environment as ge
import gymnax.environments.spaces as spaces
import housemaze.env as maze
import housemaze.levels as maze_levels
import housemaze.utils as maze_utils
import jax
import jax.numpy as jnp
import jax.random as jr
import navix as nx
from jaxtyping import Array, Key, Integer

from config import EnvConfig
from wrappers.auto_reset import auto_reset_wrapper
from wrappers.base import Environment, GoalObs, StepType, Timestep
from wrappers.flatten_observation import flatten_observation_wrapper
from wrappers.goal_wrapper import goal_wrapper
from wrappers.log import log_wrapper
from wrappers.multi_catch import make_multi_catch


def gymnax_wrapper(env: ge.Environment[ge.TEnvState, ge.TEnvParams], params: ge.TEnvParams):
    _, init_state = env.reset(jr.key(0), params)
    _, init_step = env.step(
        jr.key(0), init_state, env.action_space(params).sample(jr.key(0)), params
    )
    init_info = jax.tree.map(jnp.empty_like, init_step[-1])

    def reset(params: ge.TEnvParams, *, key: Key[Array, ""]):
        obs, state = env.reset(key, params)
        return Timestep.initial(obs, state, init_info)

    def step(state: ge.TEnvState, action: Any, params: ge.TEnvParams, *, key: Key[Array, ""]):
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


def new_housemaze():
    """Initialize HouseMaze environment."""
    # initialize map
    char_to_key = dict(
        A="knife",
        B="fork",
        C="pan",
        D="pot",
        E="bowl",
        F="plates",
    )
    image_dict = maze_utils.load_image_dict()
    object_to_index = {key: idx for idx, key in enumerate(image_dict["keys"])}
    levels = [maze_levels.two_objects, maze_levels.three_pairs_maze1]
    map_init = [
        maze_utils.from_str(map_str, char_to_key=char_to_key, object_to_index=object_to_index)
        for map_str in levels
    ]
    map_init = jax.tree.map(lambda *x: jnp.stack(x), *map_init)
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
    def _translate_obs(env_obs: maze.Observation):
        obs = {
            "image": env_obs.image,
            # "state_features": env_obs.state_features,
            "direction": env_obs.direction,
            "position": env_obs.position,
            "prev_action": env_obs.prev_action,
        }
        goal = jnp.argmax(env_obs.task_w)
        return GoalObs(obs=obs, goal=goal)

    def observation_space(params: maze.EnvParams) -> spaces.Space:
        """TODO inaccurate since image size changes per maze"""
        return spaces.Dict(
            {
                "image": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(None, None), dtype=int),
                # "state_features": spaces.Box(low=-jnp.inf, high=jnp.inf, dtype=float),
                "direction": spaces.Box(low=0, high=3, shape=(), dtype=int),
                "position": spaces.Box(low=0, high=jnp.inf, shape=(2,), dtype=int),
                "prev_action": spaces.Box(low=0, high=jnp.inf, shape=(), dtype=int),
            }
        )

    def reset(params: maze.EnvParams, *, key: Key[Array, ""]):
        env_timestep = env.reset(key, params)
        return Timestep.initial(
            obs=_translate_obs(env_timestep.observation),
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
            obs=_translate_obs(obs),
            reward=env_timestep.reward,
            discount=env_timestep.discount,
            step_type=step_type,
            info={},
        )

    return Environment(
        _inner=env,
        name="HouseMaze",
        reset=reset,
        step=step,
        action_space=lambda params: spaces.Discrete(env.num_actions(params)),
        observation_space=observation_space,
        goal_space=lambda params: spaces.Discrete(len(params.objects)),
    )


def make_env(env_config: EnvConfig, goal=True) -> tuple[Environment, Any]:
    env: Environment | None = None

    if env_config.source == "gymnax":
        env, params = gymnax.make(env_config.name, **env_config.kwargs)
        env = gymnax_wrapper(env)
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
            env = auto_reset_wrapper(env)

        elif env_config.name == "HouseMaze":
            env, params = new_housemaze()
            env = housemaze_wrapper(env)
            env = auto_reset_wrapper(env)

    if env is None:
        raise ValueError(f"Unrecognized environment {env_config.name} in {env_config.source}")

    env = flatten_observation_wrapper(env)
    env = log_wrapper(env)
    return env, params
