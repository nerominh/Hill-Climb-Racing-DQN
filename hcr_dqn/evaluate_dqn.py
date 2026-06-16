"""AI work:Evaluation helpers for the DQN scaffold.

Training tells us whether the loss is going down.
Evaluation tells us whether the agent is actually becoming a better driver.
"""

from __future__ import annotations

from statistics import mean, pstdev

if __package__ in (None, ""):  # pragma: no cover - convenience for direct script usage
    from env_wrappers import make_flat_env
else:
    from .env_wrappers import make_flat_env


def build_evaluation_seed_list(
    config,
    mode: str = "validation",
    num_episodes: int | None = None,
    seed_start: int | None = None,
) -> list[int]:
    """Choose which terrain seeds evaluation should use.

    Validation is for quick, repeatable checkpoint selection during training.
    Final evaluation is a larger held-out sweep for reporting.
    """

    if mode == "final":
        return config.final_evaluation_seed_list(
            episodes=num_episodes,
            seed_start=seed_start,
        )

    return config.validation_seed_list(episodes=num_episodes)


def evaluate_agent(
    agent,
    config,
    num_episodes: int | None = None,
    mode: str = "validation",
    seed_start: int | None = None,
    return_episode_metrics: bool = False,
) -> dict[str, float | int | list[dict[str, float | int]] | list[int]]:
    """Run a few greedy episodes and summarize the results.

    Exploration is turned off during evaluation because we want to measure the
    policy the network has actually learned, not the random actions injected by
    epsilon-greedy training.
    """

    evaluation_seeds = build_evaluation_seed_list(
        config=config,
        mode=mode,
        num_episodes=num_episodes,
        seed_start=seed_start,
    )
    env = make_flat_env(config)

    episode_returns: list[float] = []
    episode_scores: list[float] = []
    episode_lengths: list[int] = []
    episode_metrics: list[dict[str, float | int]] = []

    try:
        for episode_index, episode_seed in enumerate(evaluation_seeds, start=1):
            state, _ = env.reset(seed=episode_seed)
            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0
            final_score = 0.0

            while not done and not truncated and step_count < config.max_episode_steps:
                action = agent.select_action(state, explore=False)
                next_state, reward, done, truncated, info = env.step(action)

                total_reward += reward
                final_score = float(info.get("score", 0.0))
                state = next_state
                step_count += 1

            episode_returns.append(total_reward)
            episode_scores.append(final_score)
            episode_lengths.append(step_count)

            if return_episode_metrics:
                episode_metrics.append(
                    {
                        "episode": episode_index,
                        "seed": episode_seed,
                        "episode_return": float(total_reward),
                        "episode_score": float(final_score),
                        "episode_length": int(step_count),
                    }
                )
    finally:
        env.close()

    results: dict[str, float | int | list[dict[str, float | int]] | list[int]] = {
        "mode": mode,
        "num_episodes": len(evaluation_seeds),
        "seeds": evaluation_seeds,
        "mean_return": float(mean(episode_returns)) if episode_returns else 0.0,
        "std_return": float(pstdev(episode_returns)) if len(episode_returns) > 1 else 0.0,
        "mean_score": float(mean(episode_scores)) if episode_scores else 0.0,
        "std_score": float(pstdev(episode_scores)) if len(episode_scores) > 1 else 0.0,
        "mean_length": float(mean(episode_lengths)) if episode_lengths else 0.0,
        "std_length": float(pstdev(episode_lengths)) if len(episode_lengths) > 1 else 0.0,
    }

    if return_episode_metrics:
        results["episode_metrics"] = episode_metrics

    return results
