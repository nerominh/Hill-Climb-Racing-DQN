"""Phase 1 training script for vanilla DQN on Hill Climb Racing.

This script is intentionally written as a readable baseline instead of a
framework. The goal is that you can step through it line by line and still
understand the whole training story.
"""

from __future__ import annotations

import csv
from pathlib import Path
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local setup
    raise ModuleNotFoundError(
        "PyTorch is required for training. Install torch in the project "
        "environment before running train_dqn.py."
    ) from exc

if __package__ in (None, ""):  # pragma: no cover - convenience for direct script usage
    from configs import DQNConfig # Modification from the configs.py will automatically be reflected here --> I can change the training settings in one place and have them apply to both training and evaluation
    from dqn_agent import AntiStallMomentumDQNAgent, DQNAgent, MomentumSensitiveDQNAgent
    from env_wrappers import make_flat_env
    from evaluate_dqn import evaluate_agent
    from replay_buffer import ReplayBuffer
else:
    from .configs import DQNConfig
    from .dqn_agent import AntiStallMomentumDQNAgent, DQNAgent, MomentumSensitiveDQNAgent
    from .env_wrappers import make_flat_env
    from .evaluate_dqn import evaluate_agent
    from .replay_buffer import ReplayBuffer


def seed_everything(seed: int) -> None:
    # Set all the obvious seeds so runs are easier to reproduce
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs(config: DQNConfig) -> None:
    #Create the run folders before we try to write logs or checkpoints
    for directory in (config.output_dir, config.checkpoint_dir, config.log_dir, config.plot_dir):
        directory.mkdir(parents=True, exist_ok=True)


def write_training_row(csv_writer, row: dict[str, float | int]) -> None:
    # Write one training summary row to the log file
    csv_writer.writerow(row)


def write_validation_row(csv_writer, row: dict[str, float | int | str]) -> None:
    # Write one validation-evaluation row so checkpoint selection can be audited later
    csv_writer.writerow(row)


def print_run_header(config: DQNConfig) -> None:
    # Print the main settings once so the run is easy to follow

    print("Starting DQN training")
    print(f"Run name: {config.run_name}")
    print(f"Agent variant: {config.agent_variant}")
    print(f"TD target mode: {config.td_target_mode}")
    print(f"Environment: {config.env_id}")
    print(
        "Env config: "
        f"action_space={config.action_space}, "
        f"reward_function={config.reward_function}, "
        f"reward_type={config.reward_type}, "
        f"max_steps={config.max_steps}, "
        f"original_noise={config.original_noise}"
    )
    print(
        "Training config: "
        f"episodes={config.num_episodes}, "
        f"batch_size={config.batch_size}, "
        f"gamma={config.gamma}, "
        f"lr={config.learning_rate}, "
        f"warmup_steps={config.warmup_steps}, "
        f"target_update_frequency={config.target_update_frequency}, "
        f"validation_episodes={config.evaluation_episodes}, "
        f"validation_seed_start={config.validation_seed_start}"
    )
    print(f"Logs will be written to: {config.log_dir}")
    print(f"Checkpoints will be written to: {config.checkpoint_dir}")
    print("-" * 80)


def print_episode_summary(
    episode: int,
    config: DQNConfig,
    global_step: int,
    episode_return: float,
    final_score: float,
    step_count: int,
    epsilon: float,
    mean_loss: float,
    eval_metrics: dict[str, float],
    saved_best: bool,
) -> None:
    # Print one compact summary line for the finished episode
    summary = (
        f"Episode {episode:4d}/{config.num_episodes} | "
        f"steps={global_step:6d} | "
        f"return={episode_return:9.3f} | "
        f"score={final_score:7.3f} | "
        f"len={step_count:5d} | "
        f"epsilon={epsilon:7.4f} | "
        f"loss={mean_loss:10.6f}"
    )

    if episode % config.evaluation_frequency == 0:
        summary += (
            " | "
            f"eval_return={eval_metrics['mean_return']:9.3f} | "
            f"eval_score={eval_metrics['mean_score']:7.3f} | "
            f"eval_len={eval_metrics['mean_length']:7.2f}"
        )
        if saved_best:
            summary += " | saved_best_checkpoint=yes"

    print(summary, flush=True)


def resolve_agent_class(config: DQNConfig):
    # agent_variant only chooses the reward style.
    # td_target_mode separately chooses whether the Bellman target is DQN or DoubleDQN.
    if config.agent_variant == "vanilla":
        return DQNAgent

    if config.agent_variant == "momentum_sensitive":
        return MomentumSensitiveDQNAgent

    if config.agent_variant == "antistall_momentum":
        return AntiStallMomentumDQNAgent

    raise ValueError(
        f"Unknown agent_variant: {config.agent_variant}. "
        "Expected 'vanilla', 'momentum_sensitive', or 'antistall_momentum'."
    )


def train(config: DQNConfig | None = None) -> Path:
    # Main train function, train the selected DQN agent and return the best checkpoint path

    config = config or DQNConfig()
    agent_cls = resolve_agent_class(config)
    seed_everything(config.seed)
    ensure_output_dirs(config)
    print_run_header(config)

    env = make_flat_env(config)
    replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = agent_cls(state_dim=state_dim, action_dim=action_dim, config=config)

    log_path = config.log_dir / "training_metrics.csv"
    validation_log_path = config.log_dir / "validation_metrics.csv"
    best_checkpoint_path = config.checkpoint_dir / "best_model.pt"
    best_mean_score = float("-inf")

    global_step = 0

    with (
        log_path.open("w", newline="", encoding="utf-8") as training_handle,
        validation_log_path.open("w", newline="", encoding="utf-8") as validation_handle,
    ):
        writer = csv.DictWriter(
            training_handle,
            fieldnames=[
                "episode",
                "global_step",
                "episode_return",
                "episode_score",
                "episode_length",
                "epsilon",
                "mean_loss",
                "eval_mean_return",
                "eval_mean_score",
                "eval_mean_length",
            ],
        )
        writer.writeheader()
        validation_writer = csv.DictWriter(
            validation_handle,
            fieldnames=[
                "episode",
                "global_step",
                "validation_episodes",
                "first_seed",
                "last_seed",
                "mean_return",
                "std_return",
                "mean_score",
                "std_score",
                "mean_length",
                "std_length",
                "epsilon",
                "saved_best_checkpoint",
            ],
        )
        validation_writer.writeheader()

        try:
            for episode in range(1, config.num_episodes + 1):
                state, _ = env.reset(seed=config.seed + episode)
                if hasattr(agent, "reset_episode_reward_state"):
                    agent.reset_episode_reward_state()
                episode_return = 0.0
                episode_losses: list[float] = []
                done = False
                truncated = False
                step_count = 0
                final_score = 0.0

                while not done and not truncated and step_count < config.max_episode_steps:
                    action = agent.select_action(state, explore=True)
                    next_state, reward, done, truncated, info = env.step(action)
                    learning_reward = (
                        agent.shape_reward(reward, action, info, state, next_state)
                        if hasattr(agent, "shape_reward")
                        else reward
                    )

                    replay_buffer.push(
                        state=state,
                        action=action,
                        reward=learning_reward,
                        next_state=next_state,
                        done=(done or truncated),
                    )

                    state = next_state
                    episode_return += reward
                    final_score = float(info.get("score", 0.0))
                    step_count += 1
                    global_step += 1

                    # We wait until the buffer has enough variety before learning.
                    if (
                        len(replay_buffer) >= max(config.batch_size, config.warmup_steps)
                        and global_step % config.train_frequency == 0
                    ):
                        batch = replay_buffer.sample(config.batch_size)
                        loss = agent.train_step(batch)
                        episode_losses.append(loss)

                    if global_step % config.target_update_frequency == 0:
                        agent.update_target_network()

                agent.decay_epsilon()

                eval_metrics = {
                    "mean_return": 0.0,
                    "mean_score": 0.0,
                    "mean_length": 0.0,
                }
                saved_best = False
                if episode % config.evaluation_frequency == 0:
                    eval_metrics = evaluate_agent(agent, config, mode="validation")
                    validation_seeds = eval_metrics.get("seeds", [])

                    if eval_metrics["mean_score"] > best_mean_score:
                        best_mean_score = eval_metrics["mean_score"]
                        agent.save(best_checkpoint_path)
                        saved_best = True
                        print(
                            "New best checkpoint saved: "
                            f"episode={episode}, eval_mean_score={best_mean_score:.3f}, "
                            f"path={best_checkpoint_path}",
                            flush=True,
                        )

                    write_validation_row(
                        validation_writer,
                        {
                            "episode": episode,
                            "global_step": global_step,
                            "validation_episodes": int(eval_metrics.get("num_episodes", 0)),
                            "first_seed": validation_seeds[0] if validation_seeds else "",
                            "last_seed": validation_seeds[-1] if validation_seeds else "",
                            "mean_return": round(float(eval_metrics["mean_return"]), 4),
                            "std_return": round(float(eval_metrics["std_return"]), 4),
                            "mean_score": round(float(eval_metrics["mean_score"]), 4),
                            "std_score": round(float(eval_metrics["std_score"]), 4),
                            "mean_length": round(float(eval_metrics["mean_length"]), 4),
                            "std_length": round(float(eval_metrics["std_length"]), 4),
                            "epsilon": round(agent.epsilon, 6),
                            "saved_best_checkpoint": "yes" if saved_best else "no",
                        },
                    )

                mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

                write_training_row(
                    writer,
                    {
                        "episode": episode,
                        "global_step": global_step,
                        "episode_return": round(episode_return, 4),
                        "episode_score": round(final_score, 4),
                        "episode_length": step_count,
                        "epsilon": round(agent.epsilon, 6),
                        "mean_loss": round(mean_loss, 6),
                        "eval_mean_return": round(eval_metrics["mean_return"], 4),
                        "eval_mean_score": round(eval_metrics["mean_score"], 4),
                        "eval_mean_length": round(eval_metrics["mean_length"], 4),
                    },
                )

                print_episode_summary(
                    episode=episode,
                    config=config,
                    global_step=global_step,
                    episode_return=episode_return,
                    final_score=final_score,
                    step_count=step_count,
                    epsilon=agent.epsilon,
                    mean_loss=mean_loss,
                    eval_metrics=eval_metrics,
                    saved_best=saved_best,
                )

                training_handle.flush()
                validation_handle.flush()
        finally:
            env.close()

    # If no evaluation ever beat the initial sentinel, at least save the final model.
    if not best_checkpoint_path.exists():
        agent.save(best_checkpoint_path)

    return best_checkpoint_path


if __name__ == "__main__":
    checkpoint_path = train()
    print(f"Training finished. Best checkpoint saved to: {checkpoint_path}")
