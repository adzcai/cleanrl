from collections.abc import Callable
from typing import Annotated as Batched
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import rlax
from jaxtyping import Array, Float, Integer

from cleanrl_utils.structures import (
    TArrayTree,
    TEnvState,
    TObs,
    Transition,
    dataclass,
)

TAux = TypeVar("TAux")


@dataclass
class BootstrapConfig:
    """For bootstrapping with a target network."""

    discount: float
    lambda_gae: float


def bootstrap(
    predict_s: Callable[
        [TObs, Integer[Array, " horizon"]],
        tuple[Float[Array, " horizon"], TAux],
    ],
    txn_s: Batched[Transition[TObs, TEnvState], " horizon"],
    config: BootstrapConfig,
) -> tuple[Float[Array, " horizon-1 horizon-1"], TAux]:
    """Abstracted out for testing.

    Note that we use `TimeStep.discount` rather than `TimeStep.step_type`.

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


def f_divergence(f_name: str, c: Float[Array, ""], dual: bool):
    if f_name == "chisq":
        if dual:
            return c * c / 4 + c
    if f_name == "kl_rev":
        if dual:
            return jnp.exp(c - 1)
    raise NotImplementedError(f"f {f_name} not recognized")


def roll_into_matrix(ary: Float[Array, " n *size"]) -> Float[Array, " n n *size"]:
    return jax.vmap(jnp.roll, in_axes=(None, 0, None))(ary, -jnp.arange(ary.shape[0]), 0)


def tree_slice(tree: TArrayTree, at: int | tuple[int | Integer[Array, ""], ...] | slice) -> TArrayTree:
    """Slice each leaf of a pytree at the given index or slice."""
    return jax.tree.map(lambda x: x[at], tree)


def scale_gradient(x: Float[Array, " n"], factor: float):
    return x * factor + jax.lax.stop_gradient((1 - factor) * x)


def get_weight_mask(net):
    """Replace network pytree with True at the weight matrices and False otherwise.

    Useful so that weight decay is applied only to the weight matrices.
    """

    def has_weight(x):
        return hasattr(x, "weight")

    def get_weights(net):
        return [leaf.weight for leaf in jax.tree.leaves(net, is_leaf=has_weight) if has_weight(leaf)]

    weight_mask = jax.tree.map(lambda _: False, net, is_leaf=lambda x: x is None)
    weight_mask = eqx.tree_at(get_weights, weight_mask, [True] * len(get_weights(net)))

    return weight_mask


def get_network_size(net):
    size, nbytes = zip(*[(x.size, x.nbytes) for x in jax.tree.leaves(net) if eqx.is_inexact_array(x)])

    return sum(size), sum(nbytes)
