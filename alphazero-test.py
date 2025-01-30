import argparse
from collections.abc import Callable
from typing import NamedTuple, Optional
import jax
import jax.random as rand
import jax.numpy as jnp
from jaxtyping import Float, Bool, Integer, PyTree, UInt, Array

NULL = -1

StateEmbedding = Float[Array, "..."]
Params = PyTree
StateIndex = Integer[Array, ""]
Action = UInt[Array, ""]


class Config(argparse.Namespace):
    n_rollouts: int
    """Number of MCTS rollouts"""


class Tree(NamedTuple):
    state_visits: UInt[Array, "*batch state"]
    state_parent: UInt[Array, "*batch state"]
    state_value: Float[Array, "*batch state"]
    state_parent_action: UInt[Array, "*batch state"]
    state_embedding: Float[Array, "*batch state ..."]

    edge_visits: UInt[Array, "*batch state action"]
    edge_prior: Float[Array, "*batch state action"]
    edge_reward: Float[Array, "*batch state action"]
    edge_transition: UInt[Array, "*batch state action"]


class RecurrentActorCriticOutput(NamedTuple):
    prior_logits: Float[Array, " a"]
    value: Float[Array, ""]
    embedding: StateEmbedding


class WorldModelOutput(RecurrentActorCriticOutput):
    reward: Float[Array, ""]


WorldModel = Callable[
    [Params, rand.PRNGKey, StateEmbedding, Action],
    WorldModelOutput,
]


ActionChooser = Callable[[rand.PRNGKey, Tree, StateIndex, UInt[Array, ""]], Action]


def mcts(
    params: Params,
    rng: rand.PRNGKey,
    world_model: WorldModel,
    choose_action: ActionChooser,
    n_simulations: int,
    root: WorldModelOutput,
    max_depth: Optional[int] = None,
):
    root_state = jnp.array(0, dtype=jnp.uint32)

    if max_depth is None:
        max_depth = n_simulations

    def mcts_fn(i, carry: tuple[rand.PRNGKey, Tree]):
        rng, tree = carry
        rng, key_select, key_expand = rand.split(rng, 3)

        parent, action = select(
            rng=key_select,
            tree=tree,
            choose_action=choose_action,
            root_state=root_state,
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
        tree = backward(tree, root_state, leaf)
        return rng, tree

    tree_size = min(max_depth, n_simulations) + 1
    n_actions = root.prior_logits.size
    f_dtype = jnp.float32
    tree = Tree(
        state_visits=jnp.zeros(tree_size, dtype=jnp.uint32),
        state_parent=jnp.full(tree_size, NULL, dtype=jnp.int32),
        state_value=jnp.zeros(tree_size, dtype=f_dtype),
        state_parent_action=jnp.full(tree_size, NULL, dtype=jnp.int32),
        state_embedding=jnp.zeros_like(root.embedding),
        edge_visits=jnp.zeros((tree_size, n_actions), dtype=jnp.uint32),
        edge_prior=jnp.zeros((tree_size, n_actions), dtype=root.prior_logits.dtype),
        edge_reward=jnp.zeros(tree_size, dtype=f_dtype),
        edge_transition=jnp.full((tree_size, n_actions), NULL, dtype=jnp.int32),
    )

    tree = initialize_state(tree, root, root_state)

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
    tree = initialize_state(tree, out, leaf_state)
    return tree._replace(
        edge_reward=tree.edge_reward.at[parent_state, action].set(out.reward),
        state_parent=tree.state_parent.at[leaf_state].set(parent_state),
        state_parent_action=tree.state_parent_action.at[leaf_state].set(action),
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


def initialize_state(tree: Tree, update: RecurrentActorCriticOutput, state: StateIndex):
    return tree._replace(
        edge_prior=tree.edge_prior.at[state].set(update.prior_logits),
        state_value=tree.state_value.at[state].set(update.value),
        state_embedding=tree.state_embedding.at[state].set(update.embedding),
        state_visits=tree.state_visits.at[state].add(1.0),
    )


def make_train():
    ...




if __name__ == "__main__":
    ...