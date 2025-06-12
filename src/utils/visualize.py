import math
from typing import Annotated as Batched

import chex
import distrax
import jax
import matplotlib.pyplot as plt
import mctx
import numpy as np
from jaxtyping import Array, Bool, Float, PyTree
from matplotlib.figure import Figure

import wandb
from envs.housemaze_env import visualize_housemaze
from envs.multi_catch import visualize_catch
from experiments.config import ValueConfig
from utils.structures import StepType, Transition


def visualize_trajectory(
    env_name: str,
    value_cfg: ValueConfig,
    txn_s: Batched[Transition, "horizon"],
    boot_value_s: Float[Array, "horizon"],
    reward_logits_s: Float[Array, "horizon num_value_bins"],
    video: Float[Array, "horizon height width channels"],
):
    """Visualize the given trajectory."""

    init_mask = txn_s.time_step.is_first

    figs = dict(
        value_fig=value_figure(
            value_cfg.logits_to_value(txn_s.pred.value_logits),
            boot_value_s,
            ylabel="Value",
        ),
        policy_fig=policy_figure(
            env_name,
            distrax.Categorical(txn_s.pred.policy_logits).probs,  # type: ignore
            title="Predicted Action Probabilities",
        ),
        mcts_fig=policy_figure(
            env_name,
            txn_s.mcts_probs,
            title="MCTS Action Probabilities",
        ),
        reward_fig=value_figure(
            value_cfg.logits_to_value(reward_logits_s[:-1]),
            txn_s.time_step.reward[1:],
            ylabel="Reward",
        ),
        entropy_fig=entropy_figure(
            dict(
                reward=distrax.Categorical(logits=reward_logits_s).entropy(),
                value=distrax.Categorical(logits=txn_s.pred.value_logits).entropy(),
                policy=distrax.Categorical(logits=txn_s.pred.policy_logits).entropy(),
                mcts=distrax.Categorical(probs=txn_s.mcts_probs).entropy(),
            )
        ),
    )

    # add episode boundaries to all figures
    for fig in figs.values():
        add_episode_boundaries(fig.axes[0], init_mask)

    # add video
    if video is not None:
        figs["video_fig"] = video_figure(
            env_name,
            txn_s,
            video,
        )

    output = {}
    for k, fig in figs.items():
        output[f"visualize/{k}"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(output)


def value_figure(
    pred_value_s: Float[Array, "horizon"],
    boot_value_s: Float[Array, "horizon"],
    ylabel: str = "Value",
) -> Figure:
    """Plot the value or reward over the trajectory."""
    fig, ax = plt.subplots()
    # Set legend labels and title based on ylabel
    if ylabel.lower() == "reward":
        boot_label = "Actual Reward"
        title = "Reward Over Trajectory"
    else:
        boot_label = "Bootstrapped Value"
        title = "Value Over Trajectory"
    ax.plot(pred_value_s, label=f"Predicted {ylabel}", color="blue")
    ax.plot(boot_value_s, label=boot_label, linestyle="--")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def policy_figure(
    env_name: str, pred_probs_s: Float[Array, "horizon num_actions"], title: str
) -> Figure:
    """Plot the policy over the trajectory with improved colorbar and clearer axis labeling."""
    horizon, num_actions = pred_probs_s.shape
    # Dynamically set figure size for better readability
    fig_width = max(8, min(16, horizon * 0.5))
    fig_height = max(3, min(10, num_actions * 0.7))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        pred_probs_s.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=1,
        origin="lower",
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Action")
    ax.set_title(title)
    ax.set_yticks(range(num_actions))
    ax.set_yticklabels([get_action_name(env_name, i) for i in range(num_actions)])
    # ax.set_ylim(-0.5, num_actions - 0.5)
    # ax.set_xlim(-0.5, horizon - 0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")
    fig.tight_layout()
    return fig


def entropy_figure(entropy_dict: dict[str, Float[chex.Array, "horizon"]]) -> Figure:
    """Plot the entropy of various distributions over the trajectory."""
    fig, ax = plt.subplots()
    for key, entropy_s in entropy_dict.items():
        ax.plot(entropy_s, label=f"{key.capitalize()} Entropy")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy Over Trajectory")
    ax.legend()
    return fig


def video_figure(
    env_name: str,
    txn_s: Batched[Transition, " horizon"],
    video: Float[Array, " horizon 3 height width"],
) -> Figure:
    """Create a matplotlib figure with the trajectory video frames in a grid, with colored borders for step types."""
    horizon = txn_s.time_step.reward.shape[0]
    n_cols = math.ceil(math.sqrt(horizon))
    n_rows = math.ceil(horizon / n_cols)

    fig, ax_video = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        figsize=(n_cols * 2, n_rows * 2),
    )
    fig.subplots_adjust(hspace=0.5)

    for h in range(horizon):
        row = h // n_cols
        col = h % n_cols

        ax_frame = ax_video[row, col]
        step_type_val = txn_s.time_step.step_type[h].item()
        if step_type_val == StepType.FIRST:
            step_type = "F"
            border_color = "green"
        elif step_type_val == StepType.MID:
            step_type = "M"
            border_color = "blue"
        elif step_type_val == StepType.LAST:
            step_type = "L"
            border_color = "red"
        else:
            step_type = "?"
            border_color = "black"

        action_name = get_action_name(env_name, txn_s.action[h].item())
        ax_frame.imshow(np.permute_dims(video[h], (1, 2, 0)), aspect="auto")
        # Remove grid and ticks for cleaner look
        ax_frame.grid(False)
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        # Concise title
        ax_frame.set_title(f"{h}: {action_name} ({step_type})", fontsize=8)
        ax_frame.axis("off")
        # Set border color
        for spine in ax_frame.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
    # Hide unused axes
    for h in range(horizon, n_rows * n_cols):
        row = h // n_cols
        col = h % n_cols
        ax_video[row, col].axis("off")
    fig.tight_layout()
    fig.suptitle("Trajectory Video Frames", fontsize=12)
    return fig


def add_episode_boundaries(ax, mask_s: Bool[Array, "horizon"]):
    initial_idx = np.arange(mask_s.size)[mask_s]
    for i, idx in enumerate(initial_idx):
        ax.axvline(
            x=idx,
            color="purple",
            linestyle="--",
            label="initial" if i == 0 else None,
            alpha=0.5,
        )


def convert_tree_to_graph(
    tree: mctx.Tree, action_labels: list[str] | None = None, batch_index: int = 0
) -> "pygraphviz.AGraph":
    """Converts a search tree into a Graphviz graph.

    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.

    Returns:
      A Graphviz graph representation of `tree`.
    """
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = range(tree.num_actions)
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}."
        )

    def node_to_str(node_i, reward=0, discount=1):
        # ball_x = tree.embeddings.ball_x[batch_index, node_i].item()
        # ball_y = tree.embeddings.ball_y[batch_index, node_i].item()
        # paddle_y = tree.embeddings.paddle_y[batch_index, node_i].item()
        # paddle_x = tree.embeddings.paddle_x[batch_index, node_i].item()
        return (
            f"{node_i}\n"
            # f"Ball: {(ball_y, ball_x)}\n"
            # f"Paddle: {(paddle_y, paddle_x)}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"{action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
            f"p: {probs[a_i]:.2f}\n"
        )

    graph = pygraphviz.AGraph(directed=True)

    # Add root
    graph.add_node(0, label=node_to_str(node_i=0), color="green")
    # Add all other nodes and connect them up.
    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            children_i = tree.children_index[batch_index, node_i, a_i]
            if children_i >= 0:
                graph.add_node(
                    children_i,
                    label=node_to_str(
                        node_i=children_i,
                        reward=tree.children_rewards[batch_index, node_i, a_i],
                        discount=tree.children_discounts[batch_index, node_i, a_i],
                    ),
                    color="red",
                )
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    return graph


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
    elif env_name == "dummy":
        return f"{action}"
    else:
        raise ValueError(f"Env {env_name} not recognized")


def get_env_state_frame_label(env_name: str, env_state):
    """Label the frame in the generated video."""
    if env_name in ["Catch-bsuite", "MultiCatch"]:
        return f" task={env_state.goal}"
    if env_name == "HouseMaze":
        return f" task={env_state.state.task_w.argmax()}"
    return ""


def visualize_env_state_frame(
    env_name: str, env_state: PyTree[Array], **kwargs
) -> Float[Array, " channel height width"]:
    """Visualize a single frame of the environment."""
    if env_name in ["Catch-bsuite", "MultiCatch"]:
        return jax.jit(visualize_catch)(env_state, **kwargs)
    if env_name == "HouseMaze":
        return jax.jit(visualize_housemaze)(env_state)
    raise ValueError(f"Env {env_name} not recognized")


SUPPORTED_VIDEO_ENVS = [
    "Catch-bsuite",
    "MultiCatch",
    "HouseMaze",
]
