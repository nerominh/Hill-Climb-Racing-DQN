"""AI work: Environment wrappers and observation utilities.

The Hill Climb Racing environment returns a Gymnasium Dict observation.
That is perfectly fine for Gym, but a plain MLP-based DQN is happiest when
the state is a single flat vector --> This file handles that translation.
"""

from __future__ import annotations

from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if __package__ in (None, ""):  # pragma: no cover - convenience for direct script usage
    from bootstrap import ensure_simulator_on_path
else:
    from .bootstrap import ensure_simulator_on_path


class FlattenObservation(gym.ObservationWrapper):
    # Turn the environment's Dict observation into one flat float vector.

    # We keep the flattening order explicit on purpose. That way the training
    # code and the human reading it both know exactly what each input dimension
    # means, instead of trusting an implicit or hidden flattening rule.


    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                "FlattenObservation expects a Dict observation space, "
                f"but received {type(env.observation_space).__name__}."
            )

        # This order is the contract for the neural network input.
        self.keys_in_order = (
            "chassis_position",
            "chassis_angle",
            "wheels_speed",
            "on_ground",
        )
        self.feature_names = (
            "chassis_x",
            "chassis_y",
            "chassis_angle_deg",
            "back_wheel_speed",
            "front_wheel_speed",
            "back_wheel_on_ground",
            "front_wheel_on_ground",
        )

        low_parts: list[np.ndarray] = []
        high_parts: list[np.ndarray] = []
        total_size = 0

        for key in self.keys_in_order:
            current_space = env.observation_space[key]

            if isinstance(current_space, spaces.MultiBinary):
                low = np.zeros(current_space.n, dtype=np.float32)
                high = np.ones(current_space.n, dtype=np.float32)
                shape_size = int(current_space.n)
            elif isinstance(current_space, spaces.Box):
                low = np.asarray(current_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(current_space.high, dtype=np.float32).reshape(-1)
                shape_size = int(np.prod(current_space.shape))
            else:
                raise TypeError(
                    f"Unsupported space type for key '{key}': "
                    f"{type(current_space).__name__}"
                )

            low_parts.append(low)
            high_parts.append(high)
            total_size += shape_size

        self.observation_space = spaces.Box(
            low=np.concatenate(low_parts).astype(np.float32),
            high=np.concatenate(high_parts).astype(np.float32),
            shape=(total_size,),
            dtype=np.float32,
        )

    def observation(self, observation: OrderedDict | dict) -> np.ndarray:
        """Convert one Dict observation into a single numeric vector."""

        flat_parts: list[np.ndarray] = []

        for key in self.keys_in_order:
            # Every piece is reshaped to 1D so the final concatenate is simple
            # and consistent no matter whether the original piece was scalar-like
            # or already vector-like.
            flat_parts.append(np.asarray(observation[key], dtype=np.float32).reshape(-1))

        return np.concatenate(flat_parts, dtype=np.float32)


def make_flat_env(config, render_mode: str | None = None) -> gym.Env:
    # Create the HCR environment with the observation wrapper already applied

    ensure_simulator_on_path()
    import hill_racing_env  # noqa: F401

    env = gym.make(config.env_id, render_mode=render_mode, **config.make_env_kwargs())
    return FlattenObservation(env)
