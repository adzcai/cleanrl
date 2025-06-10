from collections.abc import Callable
from typing import Annotated as Batched, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
import rlax

from experiments.config import TrainConfig
from utils.structures import GoalObs, Prediction, TAction, TObs, Transition


def bootstrap(
    predict: Callable[
        [Batched[GoalObs[TObs], " horizon"], Integer[Array, " horizon"]],
        tuple[Float[Array, " horizon"], Any],
    ],
    trajectory: Batched[Transition, " horizon"],
    config: TrainConfig,
) -> tuple[
    Float[Array, " horizon horizon-1"],
    Batched[Prediction, " horizon horizon"],
]:
    """Abstracted out for plotting.

    Note that we use `Timestep.discount` rather than `Timestep.step_type`.

    Returns:
        tuple[Prediction (horizon, horizon), Float (horizon, horizon-1)]: The rolled matrix of network predictions;
            the rolled matrix of bootstrapped returns along the imagined trajectories.
    """
    # see `loss_trajectory` for rolling details
    action_rolled = roll_into_matrix(trajectory.action)
    reward_rolled = roll_into_matrix(trajectory.timestep.reward)
    discount_rolled = roll_into_matrix(trajectory.timestep.discount)
    last_rolled = roll_into_matrix(trajectory.timestep.is_last)

    # i.e. pred.value_logits[i, j] is the array of predicted value logits at time i+j,
    # based on the observation at time i
    # each row of the rolled matrix corresponds to a different starting time
    # o0 | a0 a1 ...
    # o1 | a1 a2 ...
    value, aux = jax.vmap(predict)(trajectory.timestep.obs, action_rolled)
    bootstrapped_return: Float[Array, " horizon horizon-1"] = jnp.asarray(
        jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None))(
            reward_rolled[:, 1:],
            discount_rolled[:, 1:] * config.bootstrap.discount,
            value[:, 1:],
            config.bootstrap.lambda_gae,
        )
    )

    # ensure terminal states get zero value
    return jnp.where(last_rolled[:, :-1], 0.0, bootstrapped_return), aux


def roll_into_matrix(ary: Float[Array, " n *size"]) -> Float[Array, " n n *size"]:
    return jax.vmap(jnp.roll, in_axes=(None, 0, None))(
        ary, -jnp.arange(ary.shape[0]), 0
    )
