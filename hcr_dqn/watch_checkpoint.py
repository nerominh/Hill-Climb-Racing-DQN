"""Visual runner for watching a saved DQN checkpoint play the game.
I have let this be AI generated so that I don't have to maintain it. 
It's not really part of the core project, and it's mostly just boilerplate for rendering and handling window events, which is a bit outside my usual wheelhouse. 
The important thing is that it can load a checkpoint and run episodes with the trained agent, so I can watch how it's doing in the game.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pygame

if __package__ in (None, ""):  # pragma: no cover - convenience for direct script usage
    from configs import DQNConfig
    from dqn_agent import DQNAgent
    from env_wrappers import make_flat_env
else:
    from .configs import DQNConfig
    from .dqn_agent import DQNAgent
    from .env_wrappers import make_flat_env


def watch_checkpoint(
    checkpoint_path: Path,
    episodes: int,
    run_name: str | None = None,
    seed: int | None = None,
    step_delay: float = 0.02,
) -> None:
    """Load a checkpoint and render greedy episodes in a Pygame window."""

    config = DQNConfig()
    if run_name is not None:
        config.run_name = run_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}. "
            "Train the agent first or pass a different checkpoint path."
        )

    env = make_flat_env(config, render_mode="human")
    try:
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)

        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)
        agent.load(checkpoint_path)

        base_seed = config.seed if seed is None else seed
        """
        One checkpoint file stores one saved model snapshot.
        The `episodes` argument does not mean "how many checkpoints to load" -
        it means "how many separate evaluation playthroughs to run" using this
        same checkpoint so we can judge the policy more reliably.
        """
        for episode_index in range(episodes):
            state, _ = env.reset(seed=base_seed + episode_index)
            env.render()
            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0
            final_score = 0.0

            while not done and not truncated and step_count < config.max_episode_steps:
                if _handle_window_events():
                    print("Closed visual evaluation window.", flush=True)
                    return

                action = agent.select_action(state, explore=False)
                next_state, reward, done, truncated, info = env.step(action)
                env.render()

                state = next_state
                total_reward += reward
                final_score = float(info.get("score", 0.0))
                step_count += 1

                if step_delay > 0.0:
                    time.sleep(step_delay)

            outcome = "terminated" if done else "truncated"
            print(
                f"Episode {episode_index + 1}/{episodes} | "
                f"return={total_reward:.3f} | "
                f"score={final_score:.3f} | "
                f"length={step_count} | "
                f"outcome={outcome}",
                flush=True,
            )
    finally:
        env.close()


def _handle_window_events() -> bool:
    """Process Pygame window events and return True when the user wants to quit."""

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def parse_args() -> argparse.Namespace: # Section to parse CLI options for visual evaluation.
    """Parse a few small CLI options for visual evaluation."""

    parser = argparse.ArgumentParser(
        description="Render a saved DQN checkpoint playing Hill Climb Racing.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name under runs/. Defaults to DQNConfig.run_name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Defaults to runs/<run_name>/checkpoints/best_model.pt.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for the rendered episodes. Basically for the environment/terrain generation. Defaults to DQNConfig.seed.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.02,
        help="Extra seconds to sleep after each step so the run is easier to watch.",
    )
    return parser.parse_args()


def main() -> None:
    # CLI entry point

    args = parse_args()
    config = DQNConfig()
    if args.run_name is not None:
        config.run_name = args.run_name

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else (
        config.checkpoint_dir / "best_model.pt"
    )

    watch_checkpoint(
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        run_name=args.run_name,
        seed=args.seed,
        step_delay=args.step_delay,
    )


if __name__ == "__main__":
    main()
