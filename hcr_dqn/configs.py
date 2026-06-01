"""Configuration objects for the DQN training scaffold.

The point of this file is to keep all of the "magic numbers" in one place.
When you start tuning later, this becomes the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping


@dataclass(slots=True)
class DQNConfig:
    # Small configuration bundle

    env_id: str = "hill_racing_env/HillRacing-v0"

    # We start with discrete actions because vanilla DQN is built for them.
    action_space: str = "discrete_3"
    reward_function: str = "distance"
    reward_type: str = "soft"
    max_steps: int = 1200
    original_noise: bool = False

    # Training schedule.
    num_episodes: int = 300
    max_episode_steps: int = 3000
    warmup_steps: int = 2_000
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    evaluation_frequency: int = 25
    evaluation_episodes: int = 5
    validation_seed_start: int = 1_000
    final_evaluation_episodes: int = 30
    final_evaluation_seed_start: int = 10_000

    # Core DQN hyperparameters.
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_buffer_capacity: int = 50_000
    hidden_sizes: tuple[int, int] = (128, 128)

    # Exploration schedule.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    # Reproducibility and output structure.
    seed: int = 7
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    run_name: str = "momentum_sensitive_dqn_seed7" # Change this for the right variation and seed
    agent_variant: str = "momentum_sensitive" # or "vanilla"

    # My variation settings: small reward shaping terms for smoother forward motion.
    momentum_bonus_scale: float = 0.05
    momentum_stall_penalty: float = 0.02
    momentum_oscillation_penalty: float = 0.02
    momentum_forward_streak_required: int = 3
    momentum_stall_patience: int = 30
    momentum_progress_clip: float = 5.0
    momentum_angle_limit_deg: float = 45.0
    momentum_angle_penalty_scale: float = 0.001
    momentum_air_penalty: float = 0.01
    momentum_back_wheel_penalty: float = 0.005

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "DQNConfig":
        """Rebuild a config object from saved checkpoint metadata.

        Older checkpoints might not contain every field we have today, so this
        method starts from the current defaults and only overwrites known keys.
        """

        config = cls()
        valid_field_names = {field_info.name for field_info in fields(cls)}

        for key, value in values.items():
            if key not in valid_field_names:
                continue

            if key == "hidden_sizes" and isinstance(value, list):
                value = tuple(value)
            elif key == "project_root" and isinstance(value, str):
                value = Path(value)

            setattr(config, key, value)

        return config

    @property
    def output_dir(self) -> Path:
        """Top-level folder for everything produced by a training run."""

        return self.project_root / "runs" / self.run_name

    @property
    def checkpoint_dir(self) -> Path:
        """Where trained model weights are stored."""

        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        """Where CSV training logs are stored."""

        return self.output_dir / "logs"

    @property
    def plot_dir(self) -> Path:
        """Where evaluation plots can be written later."""

        return self.output_dir / "plots"

    def make_env_kwargs(self) -> dict[str, object]:
        """Return the environment settings in the exact shape gym.make expects."""

        return {
            "action_space": self.action_space,
            "reward_function": self.reward_function,
            "reward_type": self.reward_type,
            "max_steps": self.max_steps,
            "original_noise": self.original_noise,
        }

    def validation_seed_list(self, episodes: int | None = None) -> list[int]:
        """Small held-out seed set for model selection during training."""

        count = episodes or self.evaluation_episodes
        return [self.validation_seed_start + index for index in range(count)]

    def final_evaluation_seed_list(
        self,
        episodes: int | None = None,
        seed_start: int | None = None,
    ) -> list[int]:
        """Larger held-out seed set for final reporting.

        These seeds are intentionally far away from the training and validation
        seeds so the final score is measured on genuinely different terrain.
        """

        count = episodes or self.final_evaluation_episodes
        base_seed = self.final_evaluation_seed_start if seed_start is None else seed_start
        return [base_seed + index for index in range(count)]
