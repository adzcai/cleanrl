from collections.abc import Callable
from typing import Annotated as Batched
from typing import TypeVar

import jax
import jax.numpy as jnp
import rlax
from jaxtyping import Array, Float, Integer

from experiments.config import BootstrapConfig
from utils.structures import (
    TArrayTree,
    TEnvState,
    TObs,
    Transition,
)

TAux = TypeVar("TAux")


def bootstrap(
    predict_s: Callable[
        [TObs, Integer[Array, " horizon"]],
        tuple[Float[Array, " horizon"], TAux],
    ],
    txn_s: Batched[Transition[TObs, TEnvState], " horizon"],
    config: BootstrapConfig,
) -> tuple[Float[Array, " horizon-1 horizon-1"], TAux]:
    """Abstracted out for plotting.

    Note that we use `Timestep.discount` rather than `Timestep.step_type`.

    Returns:
        tuple[Prediction (horizon, horizon), Float (horizon, horizon-1)]: The rolled matrix of network predictions;
            the rolled matrix of bootstrapped returns along the imagined trajectories.
    """
    # see `loss_trajectory` for rolling details
    action_sh = roll_into_matrix(txn_s.action)
    reward_sh = roll_into_matrix(txn_s.time_step.reward)
    discount_sh = roll_into_matrix(txn_s.time_step.discount)
    last_sh = roll_into_matrix(txn_s.time_step.is_last)

    # i.e. pred.value_logits[i, j] is the array of predicted value logits at time i+j,
    # based on the observation at time i
    # each row of the rolled matrix corresponds to a different starting time
    # o0 | a0 a1 ...
    # o1 | a1 a2 ...
    # we remove the final row since there is only one transition (can't bootstrap)
    value, aux = jax.vmap(predict_s)(txn_s.time_step.obs, action_sh)
    bootstrapped_return: Float[Array, " horizon-1 horizon-1"] = jnp.asarray(
        jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None))(
            reward_sh[:-1, 1:],
            discount_sh[:-1, 1:] * config.discount,
            value[:-1, 1:],
            config.lambda_gae,
        )
    )

    # ensure terminal states get zero value
    return jnp.where(last_sh[:-1, :-1], 0.0, bootstrapped_return), aux


def roll_into_matrix(ary: Float[Array, " n *size"]) -> Float[Array, " n n *size"]:
    return jax.vmap(jnp.roll, in_axes=(None, 0, None))(
        ary, -jnp.arange(ary.shape[0]), 0
    )


def tree_slice(
    tree: TArrayTree, at: int | tuple[int | Integer[Array, ""], ...] | slice
) -> TArrayTree:
    """Slice each leaf of a pytree at the given index or slice."""
    return jax.tree.map(lambda x: x[at], tree)


def scale_gradient(x: Float[Array, " n"], factor: float):
    return x * factor + jax.lax.stop_gradient((1 - factor) * x)
