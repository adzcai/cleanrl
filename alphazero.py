import argparse
from collections.abc import Callable
from typing import NamedTuple
import jax
import jax.random as rand
import jax.numpy as jnp
from jaxtyping import Float, Bool, Integer, PyTree, UInt, Array

NULL = -1

StateEmbedding = Float[Array, "..."]


class WorldModelOutput(NamedTuple):
    prior_logits: Float[Array, " a"]
    value: Float[Array, ""]
    reward: Float[Array, ""]
    embedding: StateEmbedding


Params = PyTree
StateIndex = Integer[Array, ""]
Action = UInt[Array, ""]
WorldModel = Callable[
    [Params, rand.PRNGKey, StateEmbedding, Action],
    WorldModelOutput,
]


class Config(argparse.Namespace):
    n_rollouts: int
    """Number of MCTS rollouts"""


class Tree(NamedTuple):
    state_visits: UInt[Array, " s"]
    state_parent: UInt[Array, " s"]
    state_value: Float[Array, " s"]
    state_parent_action: UInt[Array, " s"]
    state_embedding: Float[Array, "s ..."]

    edge_visits: UInt[Array, "s a"]
    edge_prior: Float[Array, "s a"]
    edge_reward: Float[Array, "s a"]
    edge_transition: UInt[Array, "s a"]


ActionChooser = Callable[[rand.PRNGKey, Tree, StateIndex, UInt[Array, ""]], Action]


def mcts(
    params: Params,
    rng: rand.PRNGKey,
    world_model: WorldModel,
    choose_action: ActionChooser,
    n_simulations: int,
    max_depth: int,
):
    def mcts_fn(i, carry: tuple[rand.PRNGKey, Tree]):
        rng, tree = carry
        rng, key_select, key_expand = rand.split(rng, 3)

        parent, action = select(
            rng=key_select,
            tree=tree,
            choose_action=choose_action,
            root_state=jnp.array(0, dtype=jnp.uint32),
            max_depth=max_depth,
        )

        leaf = tree.edge_transition[parent, action]
        # false branch only if max depth reached
        leaf = jnp.where(leaf == NULL, i + 1, leaf)
        tree = expand(
            params=params,
            rng=key_expand,
            tree=tree,
            world_model=world_model,
            parent_state=parent,
            action=action,
            leaf_state=leaf,
        )
        tree = backward(tree, leaf)
        return rng, tree

    tree = Tree()
    _, tree = jax.lax.fori_loop(0, n_simulations, mcts_fn, (rng, tree))
    return tree


class _SelectCarry(NamedTuple):
    rng: rand.PRNGKey
    curr_state: StateIndex
    action: Action
    next_state: StateIndex
    depth: UInt[Array, ""]
    running: Bool[Array, ""]


def select(
    rng: rand.PRNGKey,
    tree: Tree,
    choose_action: ActionChooser,
    root_state: StateIndex,
    max_depth: int,
):
    def select_fn(carry: _SelectCarry) -> _SelectCarry:
        state = carry.next_state
        rng, key = rand.split(carry.rng)
        action = choose_action(key, tree, state, carry.depth)
        next_state = tree.edge_transition[state, action]
        depth = carry.depth + 1
        running = jnp.logical_and(next_state != NULL, depth < max_depth)
        return _SelectCarry(
            rng=rng,
            curr_state=state,
            action=action,
            next_state=next_state,
            depth=depth,
            running=running,
        )

    init_carry = _SelectCarry(
        rng=rng,
        curr_state=NULL,
        action=NULL,
        next_state=root_state,
        depth=jnp.array(0, dtype=jnp.uint32),
        running=jnp.array(True),
    )

    out = jax.lax.while_loop(lambda carry: carry.running, select_fn, init_carry)
    return out.curr_state, out.action


def expand(
    params: PyTree,
    rng: rand.PRNGKey,
    tree: Tree,
    world_model: WorldModel,
    parent_state: StateIndex,
    action: Action,
    leaf_state: StateIndex,
):
    out = world_model(params, rng, tree.state_embedding[parent_state], action)

    # assumes leaf_state has not yet been visited
    return tree._replace(
        edge_prior=tree.edge_prior.at[leaf_state].set(out.prior_logits),
        state_value=tree.state_value.at[parent_state, action].set(out.value),
        edge_reward=tree.edge_reward.at[parent_state, action].set(out.reward),
        state_embedding=tree.state_embedding.at[leaf_state].set(out.embedding),
        state_parent=tree.state_parent.at[leaf_state].set(parent_state),
        state_parent_action=tree.state_parent_action.at[leaf_state].set(action),
        state_visits=tree.state_visits.at[leaf_state].add(1.0),
    )


class _BackwardCarry(NamedTuple):
    tree: Tree
    state: StateIndex
    value: Float[Array, ""]
    """The estimated value of `state`"""


def backward(
    tree: Tree, root_state: StateIndex, leaf_state: StateIndex, discount: float
):
    def backward_fn(carry: _BackwardCarry) -> _BackwardCarry:
        """Update the value of the parent of `carry.state`"""
        parent = tree.state_parent[carry.state]
        action = tree.state_parent_action[carry.state]
        reward = tree.edge_reward[parent, action]
        edge_value = reward + discount * carry.value
        count = tree.state_visits[parent]
        parent_value = (tree.state_value[parent] * count + edge_value) / (count + 1.0)
        return tree._replace(
            state_values=tree.state_value.at[parent].set(parent_value),
            state_visits=tree.state_visits.at[parent].add(1),
            edge_visits=tree.edge_visits.at[parent, action].add(1),
        )

    init_carry = _BackwardCarry(
        tree=tree, state=leaf_state, value=tree.state_value[leaf_state]
    )
    carry = jax.lax.while_loop(
        lambda carry: carry.state != root_state, backward_fn, init_carry
    )
    return carry.tree
