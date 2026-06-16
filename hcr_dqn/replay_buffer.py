"""AI work: Replay buffer implementation for DQN.

The replay buffer is where we store past transitions so the agent can learn
from shuffled experience instead of only the most recent few frames.
That breaks short-term correlations and is one of the stability tricks that
made DQN practical in the first place.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(slots=True)
class Transition:
    """One recorded interaction with the environment."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """A simple fixed-capacity replay buffer.

    This version is intentionally straightforward. Phase 1 is about making the
    training pipeline understandable and reliable before we add fancier ideas
    like prioritized replay.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")

        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Allow len(buffer) to tell us how much experience we have."""

        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition.

        We copy the arrays into float32 here so the buffer cannot be affected
        by later accidental in-place edits elsewhere in the code.
        """

        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Return a random minibatch in a training-friendly format."""

        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample batch of size {batch_size} from buffer "
                f"with only {len(self.buffer)} transitions."
            )

        batch = random.sample(self.buffer, batch_size)

        return {
            "states": np.stack([item.state for item in batch]).astype(np.float32),
            "actions": np.asarray([item.action for item in batch], dtype=np.int64),
            "rewards": np.asarray([item.reward for item in batch], dtype=np.float32),
            "next_states": np.stack([item.next_state for item in batch]).astype(np.float32),
            "dones": np.asarray([item.done for item in batch], dtype=np.float32),
        }
