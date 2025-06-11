from typing import Annotated as Batched

import chex
import distrax
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import mctx
import rlax
from jaxtyping import Array, Float
from matplotlib.axes import Axes

# Add colorbar for reference (for p0 only)
from matplotlib.cm import ScalarMappable

# Add legend manually
from matplotlib.patches import Patch

import wandb
from experiments.config import TrainConfig, ValueConfig
from utils.log_utils import tree_slice
from utils.structures import StepType, Transition


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


def visualize(
    config: TrainConfig,
    txn_s: Batched[Transition, " horizon"],
    boot_return_s: Float[Array, " horizon-1"],
    video: Float[Array, " horizon 3 height width"] | None,
    prefix: str,
    reward_logits_s: Float[Array, " horizon num_value_bins"] | None = None,
    priorities: Float[Array, " num_envs"] | None = None,
) -> None:
    """Visualize a trajectory."""
    if not wandb.run or wandb.run.disabled:
        return

    fig_width = 4
    rows_per_env = 2
    if reward_logits_s is not None:
        rows_per_env += 1

    rows_per_env += 1

    bin_labels = jnp.linspace(
        value_cfg.min_value,
        value_cfg.max_value,
        value_cfg.num_value_bins,
    )
    bin_labels = [f"{v.item():.02f}" for v in bin_labels]
    bin_labels += [f"{label} (bs)" for label in bin_labels]

    num_envs, horizon = txn_s.time_step.reward.shape

    fig_stats, ax = plt.subplots(
        nrows=num_envs * rows_per_env,
        figsize=(2 * fig_width, 4 * num_envs * rows_per_env),
    )
    fig_stats.subplots_adjust(hspace=0.5)
    for i in range(num_envs):
        j = 0

        ax_stats: Axes = ax[rows_per_env * i + j]
        j += 1
        ax_stats.set_title(
            f"T{i}"
            + (f" priority {priorities[i]:.2f}" if priorities is not None else "")
        )
        plot_statistics(
            value_cfg,
            ax_stats,
            tree_slice(txn_s, i),
            boot_return_s[i],
        )

        ax_policy: Axes = ax[rows_per_env * i + j]
        j += 1
        ax_policy.set_title(f"Trajectory {i} Policy and MCTS")
        action_names = [get_action_name(env_name, i) for i in range(3)]
        action_names += [f"{name} (mc)" for name in action_names]
        plot_compare_dists(
            ax_policy,
            jax.nn.softmax(txn_s.pred.policy_logits[i], axis=-1),
            txn_s.mcts_probs[i],
            labels=action_names,
        )

        if reward_logits_s is not None:
            ax_reward: Axes = ax[rows_per_env * i + j]
            j += 1
            ax_reward.set_title(f"T{i} reward")
            plot_compare_dists(
                ax_reward,
                jax.nn.softmax(reward_logits_s[i], axis=-1),
                value_cfg.value_to_probs(txn_s.time_step.reward[i]),
                labels=bin_labels,
            )

        ax_value: Axes = ax[rows_per_env * i + j]
        j += 1
        ax_value.set_title(f"Trajectory {i} Value and Bootstrapped")
        plot_compare_dists(
            ax_value,
            jax.nn.softmax(txn_s.pred.value_logits[i, :-1], axis=-1),
            value_cfg.value_to_probs(boot_return_s[i]),
            labels=bin_labels,
        )

        # legend above first axes
        if i == 0:
            # move legend to right of plot
            box = ax_stats.get_position()
            ax_stats.set_position(
                (
                    box.x0,
                    box.y0,
                    box.width * 0.8,
                    box.height,
                )
            )
            ax_stats.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # use wandb.Image since otherwise wandb uses plotly,
    # which breaks the legends
    obj = {f"{prefix}/statistics": wandb.Image(fig_stats)}

    # separate figure for plotting video
    if video is not None:
        fig_video, ax_video = plt.subplots(
            nrows=num_envs,
            ncols=horizon,
            squeeze=False,
            figsize=(horizon * 2, num_envs * 3),
        )
        fig_video.subplots_adjust(
            hspace=0.5
        )  # Add more vertical space between subplots

        # wandb.Video(np.asarray(video), fps=10),

        for i in range(num_envs):
            for h in range(horizon):
                ax_stats: Axes = ax_video[i, h]

                step_type = txn_s.time_step.step_type[i, h].item()
                if step_type == StepType.FIRST:
                    step_type = "F"
                elif step_type == StepType.MID:
                    step_type = "M"
                elif step_type == StepType.LAST:
                    step_type = "L"

                a = get_action_name(env_name, txn_s.action[i, h].item())
                # c h w -> h w c
                ax_stats.imshow(jnp.permute_dims(video[i, h], (1, 2, 0)))
                ax_stats.grid(True)
                ax_stats.xaxis.set_major_locator(plt.MultipleLocator(1))  # type: ignore
                ax_stats.yaxis.set_major_locator(plt.MultipleLocator(1))  # type: ignore
                env_state = tree_slice(txn_s.time_step.state, (i, h))
                ax_stats.set_title(
                    f"{h=}\n"
                    f"{step_type}\n"
                    f"{a=}\n"
                    f"{get_env_state_frame_label(env_name, env_state)}"
                )
                ax_stats.axis("off")
        obj[f"{prefix}/trajectories"] = wandb.Image(fig_video)

    wandb.log(obj)
    plt.close(fig_stats)
    if video is not None:
        plt.close(fig_video)  # type: ignore


def plot_statistics(
    value_cfg: ValueConfig,
    ax: Axes,
    trajectory: Batched[Transition, " horizon"],
    bootstrapped_return: Float[Array, " horizon"],
):
    """Plot various trajectory statistics:

    - Beginning of episodes
    - Rewards
    - Predicted values vs bootstrapped values (td error)
    - Policy and value entropy and kl error
    """
    horizon = jnp.arange(trajectory.action.size)

    # misc
    initial_idx = horizon[trajectory.time_step.step_type == StepType.FIRST]
    for i, idx in enumerate(initial_idx):
        ax.axvline(
            x=idx,
            color="purple",
            linestyle="--",
            label="initial" if i == 0 else None,
            alpha=0.5,
        )

    # reward
    ax.plot(horizon, trajectory.time_step.reward, "r+", label="reward")

    # value
    online_value = value_cfg.logits_to_value(trajectory.pred.value_logits)
    ax.plot(horizon, online_value, "bo", label="online value")
    ax.plot(
        horizon[:-1],
        bootstrapped_return,
        "mo",
        label="bootstrapped returns",
        alpha=0.5,
    )
    ax.fill_between(
        horizon[:-1],
        bootstrapped_return,
        online_value[:-1],
        alpha=0.3,
        label="TD error",
    )

    # entropy
    policy_dist = distrax.Categorical(logits=trajectory.pred.policy_logits)
    mcts_dist = distrax.Categorical(probs=trajectory.mcts_probs)
    ax.plot(horizon, policy_dist.entropy(), "b:", label="policy entropy")
    ax.plot(horizon, mcts_dist.entropy(), "g:", label="MCTS entropy")

    value_dist = distrax.Categorical(logits=trajectory.pred.value_logits)
    ax.plot(horizon, value_dist.entropy(), "y:", label="value entropy")

    # loss
    ax.plot(
        horizon,
        mcts_dist.kl_divergence(policy_dist),
        "c--",
        label="policy / mcts kl",
    )
    bootstrapped_dist = distrax.Categorical(
        probs=value_cfg.value_to_probs(bootstrapped_return)
    )
    value_loss = bootstrapped_dist.kl_divergence(value_dist[:-1])  # type: ignore
    ax.plot(horizon[:-1], value_loss, "r--", label="value / bootstrap kl")

    ax.set_xticks(horizon, horizon)
    spread = value_cfg.max_value - value_cfg.min_value
    ax.set_ylim(value_cfg.min_value - spread / 10, value_cfg.max_value + spread / 10)


def plot_compare_dists(
    ax: Axes,
    p0: Float[Array, " horizon n"],
    p1: Float[Array, " horizon n"],
    labels: list[str],
):
    chex.assert_equal_shape([p0, p1])
    horizon, num_bins = p0.shape
    x = jnp.arange(horizon)

    # Plot p0 as blue, p1 as red, with alpha blending
    ax.imshow(
        p0.T,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        alpha=0.6,
        extent=(0, horizon, 0, num_bins),
        vmin=0,
        vmax=1,
    )
    ax.imshow(
        p1.T,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        alpha=0.4,
        extent=(0, horizon, 0, num_bins),
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_yticks(jnp.arange(num_bins))
    if labels is not None and len(labels) == num_bins:
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels(jnp.arange(num_bins))

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Bin")
    ax.set_title("Distribution (Blue: Policy, Red: MCTS)")

    sm = ScalarMappable(cmap="Blues", norm=mpl.colors.Normalize(vmin=0, vmax=1))  # type: ignore
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Probability")

    legend_handles = [
        Patch(facecolor="blue", edgecolor="blue", alpha=0.6, label="Policy"),
        Patch(facecolor="red", edgecolor="red", alpha=0.4, label="MCTS"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1, 0.5))


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
