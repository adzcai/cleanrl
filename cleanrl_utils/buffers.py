# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.
#
# Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from __future__ import annotations

import dataclasses as dc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Annotated as Batched
from typing import Any, Generic, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch as th
from gymnasium import spaces
from jaxtyping import Array, Float, Integer, Key, UInt

from cleanrl_utils.envs.utils import TArrayTree as TExperience
from cleanrl_utils.envs.utils import dataclass
from cleanrl_utils.jax_utils import exec_callback, tree_slice

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        )
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples | RolloutBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


@dataclass
class SumTreeState:
    nodes: Float[Array, " max_size"]
    max_priority: Float[Array, ""]  # new trajectories are assigned max priority
    step: UInt[Array, ""]  # calibrate every some updates


@dataclass
class BufferState(Generic[TExperience]):
    data: Batched[TExperience, " batch_size max_time"]
    pos: UInt[Array, ""]
    priority_state: SumTreeState


@dataclass
class Sample(Generic[TExperience]):
    experience: Batched[TExperience, " horizon"]
    idx: UInt[Array, ""]
    priority: Float[Array, ""]


@dataclass
class BufferConfig:
    """Configuration for the prioritized buffer."""

    batch_size: int
    max_length: int
    sample_length: int


@dataclass
class PrioritizedBuffer(BufferConfig, Generic[TExperience]):
    """A buffer that contains experiences and priority weights.

    We use "idx" to refer to the flattened index (across batches)
    and "pos" to refer to the time index.
    i.e. idx = batch * max_time + pos.
    """

    priority_tree: "SumTree"

    @classmethod
    def new(cls, batch_size: int, max_length: int, sample_length: int):
        return cls(
            batch_size=batch_size,
            max_length=max_length,
            sample_length=sample_length,
            priority_tree=SumTree.new(batch_size * max_length),
        )

    def init(self, experience: TExperience) -> BufferState[TExperience]:
        """Initialize buffer from a dummy experience."""
        data = jax.tree.map(
            lambda x: jnp.empty_like(x, shape=(self.batch_size, self.max_length, *x.shape)),
            experience,
        )

        return BufferState(
            data=data,
            pos=jnp.zeros((), jnp.uint32),
            priority_state=self.priority_tree.init(),
        )

    def add(
        self,
        state: BufferState[TExperience],
        experience: Batched[TExperience, " batch_size horizon"],
    ) -> BufferState[TExperience]:
        """Add a batch of trajectories.

        Args:
            state (BufferState[Experience]): The current buffer state.
            trajectories (Experience (batch_size, horizon)): The batch of trajectories to add.

        Returns:
            BufferState[Experience]: The updated buffer.
        """
        added_len = jax.tree.leaves(experience)[0].shape[1]
        pos_ary = (state.pos + jnp.arange(added_len)) % self.max_length
        data = jax.tree.map(lambda whole, new: whole.at[:, pos_ary].set(new), state.data, experience)
        state = dc.replace(state, data=data, pos=state.pos + added_len)
        # enable the trajectories starting up to this point
        # TODO currently incorrect for initial trajectory if too short
        state = self.set_priorities_by_pos(
            state,
            state.pos - self.sample_length - jnp.arange(added_len),
            state.priority_state.max_priority,
        )
        # when we wrap back to the front of the buffer,
        # disable the positions that now no longer can be sampled
        # (that would cross over into old data)
        state = self.set_priorities_by_pos(
            state,
            state.pos - 1 - jnp.arange(self.sample_length - 1),
            jnp.zeros(()),
        )
        return state

    def set_priorities_by_pos(
        self,
        state: BufferState[TExperience],
        pos: Float[Array, " horizon"],
        priorities: Float[Array, ""],
    ):
        pos = jnp.maximum(0, pos) % self.max_length
        pos = jnp.broadcast_to(pos, (self.batch_size, pos.size))
        idx = jax.vmap(self.pos_to_flat, in_axes=(1,))(pos)
        state = self.set_priorities(state, idx.ravel(), priorities[jnp.newaxis])
        return state

    def pos_to_flat(self, pos: UInt[Array, " batch_size"]) -> UInt[Array, " batch_size"]:
        # e.g. suppose max_time is 4 and batch_size is 3
        # then pos_to_flat([2 1 3]) = [2 1 3] + [0 4 8] = [2 5 11]
        return pos + self.max_length * jnp.arange(self.batch_size)

    def set_priorities(
        self,
        state: BufferState[TExperience],
        idx: UInt[Array, " batch_size"],
        priorities: Float[Array, " batch_size"],
    ) -> BufferState[TExperience]:
        """Set priorities for a batch of trajectories."""
        priority_state = self.priority_tree.update(state.priority_state, idx, priorities)
        return dc.replace(state, priority_state=priority_state)

    def sample(
        self, state: BufferState[TExperience], debug=False, *, key: Key[Array, ""]
    ) -> Batched[Sample[TExperience], " horizon"]:
        """Sample a trajectory from the buffer."""
        key_sample, key_fallback = jr.split(key)
        idx, priority = self.priority_tree.sample(state.priority_state, key_sample, debug=debug)
        batch, pos = divmod(idx, self.max_length)

        # ensure valid indices
        # sample from uniform if invalid
        pos_avail = self.available_pos(state)
        fallback_pos = jr.randint(key_fallback, pos.shape, 0, pos_avail)
        pos = jnp.where(pos < pos_avail, pos, fallback_pos)

        # take trajectories
        pos_ary = (pos + jnp.arange(self.sample_length)) % self.max_length
        return Sample(
            experience=tree_slice(state.data, (batch, pos_ary)),
            idx=idx,
            priority=priority,
        )

    def available_pos(self, state: BufferState[TExperience]) -> Integer[Array, ""]:
        """The number of available initial positions (along the horizon axis).

        i.e. (randint(0, available_pos) + range(horizon)) % max_time
        is entirely populated.
        """
        return jnp.where(
            state.pos < self.sample_length,
            0,
            jnp.minimum(state.pos, self.max_length) - self.sample_length + 1,
        )

    def num_available(self, state: BufferState[TExperience]) -> Integer[Array, ""]:
        """The number of possible trajectories (overlapping and across batch)."""
        size = self.batch_size * self.available_pos(state)
        # could also count nonzero in self.priority_tree.nodes[(1 << self.priority_tree.depth) - 1 :]
        return size


@dataclass
class SumTree:
    """Contains the priorities in a sum tree.

    Throughout the implementation,
    `index` indicates an index into the leaves and
    `idx` indicates an index into the flattened tree structure
    (offset by `self.leaf_idx`).
    """

    size: int
    depth: int
    calibrate_freq: int

    @classmethod
    def new(cls, size: int, calibrate_freq: int = 64):
        depth = int(jnp.ceil(jnp.log2(size)))  # root at zero
        return cls(
            size=size,
            depth=depth,
            calibrate_freq=calibrate_freq,
        )

    def init(self):
        nodes = jnp.zeros((1 << (self.depth + 1)) - 1)
        return SumTreeState(
            nodes=nodes,
            max_priority=jnp.ones((), float),
            step=jnp.zeros((), jnp.uint32),
        )

    def update(
        self,
        state: SumTreeState,
        indices: UInt[Array, " n"],
        priority: Float[Array, " n"],
    ) -> SumTreeState:
        """Set priorities at the given indices.

        To mitigate floating point errors,
        every calibrate_freq updates,
        recompute the tree.

        Args:
            indices (UInt (n,)): The indices to update.
            priority (Float (n,)): The new values.

        Returns:
            SumTree: The updated tree.
        """
        return jax.lax.cond(
            state.step % self.calibrate_freq == 0,
            self._recompute,
            self._update,
            state,
            indices,
            priority,
        )

    def _update(
        self,
        state: SumTreeState,
        indices: UInt[Array, " n"],
        priority: Float[Array, " n"],
    ) -> SumTreeState:
        # we only update using the first priority for each index
        # turn others into no-ops (add zero)
        unique, first = jnp.unique(indices, return_index=True, size=indices.size, fill_value=self.size)
        unique: Float[Array, " n_unique"]
        mask = unique < self.size
        idx = jnp.where(mask, self.leaf_idx + unique, 0)
        nodes = state.nodes
        delta = jnp.where(mask, priority[first] - nodes[idx], 0.0)

        for _ in range(self.depth + 1):  # include root
            # accumulate deltas
            nodes = nodes.at[idx].add(delta)
            idx = (idx - 1) >> 1

        return dc.replace(
            state,
            nodes=nodes,
            max_priority=jnp.maximum(jnp.max(priority), state.max_priority),
            step=state.step + 1,
        )

    def _recompute(
        self,
        state: SumTreeState,
        indices: UInt[Array, " n"],
        priority: Float[Array, " n"],
    ):
        """Sets the priorities and recomputes the entire tree."""
        nodes = state.nodes.at[indices + self.leaf_idx].set(priority)
        idx = jnp.arange(self.leaf_idx, nodes.size)
        for _ in range(self.depth):
            idx = (idx - 1) >> 1
            left_idx = (idx << 1) + 1
            right_idx = (idx << 1) + 2
            nodes = nodes.at[idx].set(nodes[left_idx] + nodes[right_idx])
        return dc.replace(
            state,
            nodes=nodes,
            max_priority=jnp.max(nodes[self.leaf_idx :]),
            step=state.step + 1,
        )

    def sample(self, state: SumTreeState, key: Key[Array, ""], debug=False) -> tuple[UInt[Array, ""], Float[Array, ""]]:
        """Sample a single index from the priorities.

        Raises:
            ValueError: If debug is enabled and an out-of-bounds index is returned.

        Returns:
            tuple[UInt (), Float ()]: The sampled index; The priority of the sampled item.
        """
        idx = jnp.zeros((), jnp.uint32)
        value = jr.uniform(key, maxval=state.nodes[idx])
        for _ in range(self.depth):
            left_idx = (idx << 1) + 1
            left_sum = state.nodes[left_idx]
            idx = jnp.where(value < left_sum, left_idx, left_idx + 1)
            value = jnp.where(value < left_sum, value, value - left_sum)

        if debug:

            @exec_callback
            def debug_callback(
                error=jnp.any(idx - self.leaf_idx >= self.size),
                top=state.nodes[0],
                total=state.nodes[self.leaf_idx :].sum(),
            ):
                if error:
                    raise IndexError(
                        "Invalid index encountered when sampling from the priority tree. "
                        "This is most likely due to accumulated floating-point errors. "
                        f"Difference was {top - total:.3e} "
                        f"(relative {(top - total) / top:.3e}). "
                        "Pass a smaller calibrate_freq to mitigate this."
                    )

        return idx - self.leaf_idx, state.nodes[idx]

    @property
    def leaf_idx(self):
        return (1 << self.depth) - 1

    def pprint(self, state: SumTreeState):
        """Pretty print the sum tree structure for debugging.

        Args:
            state (SumTreeState): The current state of the sum tree.
        """
        nodes = state.nodes
        depth = self.depth
        leaf_idx = self.leaf_idx

        # Width settings
        node_width = 7  # Width for each number
        total_cols = 2**depth

        for level in range(depth + 1):
            # Calculate indices for this level
            start_idx = (1 << level) - 1
            end_idx = (1 << (level + 1)) - 1
            level_nodes = nodes[start_idx:end_idx]

            # Calculate positions for this level
            positions = []
            segment_width = total_cols // (1 << level)
            for i in range(1 << level):
                pos = (segment_width // 2) + i * segment_width
                positions.append(pos)

            # Print level number and nodes
            line = [" " * node_width] * total_cols
            for pos, val in zip(positions, level_nodes):
                line[pos] = f"{val:7.2f}"

            print(f"Level {level}:", "".join(line))

        # Print separator and leaf values
        print("-" * (total_cols * node_width))
        print("Leaf values:", nodes[leaf_idx : leaf_idx + self.size])
