"""Small command-line entry point for evaluating a saved DQN checkpoint."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - convenience for direct script usage
    from configs import DQNConfig
    from dqn_agent import DQNAgent, load_checkpoint_payload
    from env_wrappers import make_flat_env
    from evaluate_dqn import evaluate_agent
else:
    from .configs import DQNConfig
    from .dqn_agent import DQNAgent, load_checkpoint_payload
    from .env_wrappers import make_flat_env
    from .evaluate_dqn import evaluate_agent


def parse_args() -> argparse.Namespace: # This function is to parse CLI options for visual evaluation. It allows me to specify the run name, checkpoint path, number of episodes to evaluate, and output CSV path for saving the evaluation summary
    """Parse small CLI options for evaluating a saved checkpoint."""

    parser = argparse.ArgumentParser(
        description="Evaluate a saved DQN checkpoint and track the summary.",
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
        "--mode",
        type=str,
        choices=("validation", "final"),
        default="final",
        help="Choose quick validation evaluation or a larger held-out final evaluation. Defaults to final.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional number of evaluation episodes. Defaults to the count for the chosen evaluation mode.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Optional manual first seed for final evaluation. Ignored for validation mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for saving the evaluation summary.",
    )
    return parser.parse_args()


def append_evaluation_summary(
    csv_path: Path,
    mode: str,
    run_name: str,
    checkpoint_path: Path,
    metrics: dict[str, float | int | list[int] | list[dict[str, float | int]]],
) -> None:
    """Replace the latest summary for this run/mode so reruns stay tidy."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "mode",
        "run_name",
        "checkpoint_path",
        "evaluation_episodes",
        "first_seed",
        "last_seed",
        "mean_return",
        "std_return",
        "mean_score",
        "std_score",
        "mean_length",
        "std_length",
    ]
    existing_rows: list[dict[str, str]] = []

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = [
                row
                for row in csv.DictReader(handle)
                if not (
                    row.get("mode", "") == mode
                    and row.get("run_name", "") == run_name
                )
            ]

    seeds = metrics.get("seeds", [])
    existing_rows.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "run_name": run_name,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_episodes": str(metrics["num_episodes"]),
            "first_seed": str(seeds[0]) if seeds else "",
            "last_seed": str(seeds[-1]) if seeds else "",
            "mean_return": f"{round(float(metrics['mean_return']), 4):.4f}",
            "std_return": f"{round(float(metrics['std_return']), 4):.4f}",
            "mean_score": f"{round(float(metrics['mean_score']), 4):.4f}",
            "std_score": f"{round(float(metrics['std_score']), 4):.4f}",
            "mean_length": f"{round(float(metrics['mean_length']), 4):.4f}",
            "std_length": f"{round(float(metrics['std_length']), 4):.4f}",
        }
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(existing_rows)


def append_evaluation_details(
    csv_path: Path,
    mode: str,
    run_name: str,
    checkpoint_path: Path,
    episode_metrics: list[dict[str, float | int]],
) -> None:
    """Replace the latest per-episode details for this run/mode on rerun."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "mode",
        "run_name",
        "checkpoint_path",
        "episode",
        "seed",
        "episode_return",
        "episode_score",
        "episode_length",
    ]
    existing_rows: list[dict[str, str]] = []

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = [
                row
                for row in csv.DictReader(handle)
                if not (
                    row.get("mode", "") == mode
                    and row.get("run_name", "") == run_name
                )
            ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        timestamp = datetime.now().isoformat(timespec="seconds")
        new_rows: list[dict[str, str]] = []
        for row in episode_metrics:
            new_rows.append(
                {
                    "timestamp": timestamp,
                    "mode": mode,
                    "run_name": run_name,
                    "checkpoint_path": str(checkpoint_path),
                    "episode": str(row["episode"]),
                    "seed": str(row["seed"]),
                    "episode_return": f"{round(float(row['episode_return']), 4):.4f}",
                    "episode_score": f"{round(float(row['episode_score']), 4):.4f}",
                    "episode_length": str(int(row["episode_length"])),
                }
            )

        writer.writerows(existing_rows + new_rows)


def load_config_from_checkpoint(checkpoint_path: Path, run_name_override: str | None = None) -> DQNConfig:
    """Use saved checkpoint metadata so evaluation matches the trained model setup."""

    current_defaults = DQNConfig()
    checkpoint = load_checkpoint_payload(checkpoint_path, device="cpu")
    checkpoint_config = checkpoint.get("config", {})

    if isinstance(checkpoint_config, dict):
        config = DQNConfig.from_dict(checkpoint_config)
    else:
        config = DQNConfig()

    if run_name_override is not None:
        config.run_name = run_name_override

    # Keep outputs inside the current workspace even if the checkpoint was
    # created from an earlier copy of the project in a different location.
    config.project_root = current_defaults.project_root

    return config


def main() -> None:
    """Load a saved checkpoint, print a summary, and track it in a CSV log."""

    args = parse_args()

    bootstrap_config = DQNConfig()
    run_name = args.run_name or bootstrap_config.run_name
    checkpoint_path = args.checkpoint or (bootstrap_config.project_root / "runs" / run_name / "checkpoints" / "best_model.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}. "
            "Train the agent first so there is something to evaluate."
        )

    config = load_config_from_checkpoint(checkpoint_path, run_name_override=run_name)
    output_path = args.output or (config.log_dir / "evaluation_summaries.csv")
    details_output_path = config.log_dir / "evaluation_episode_details.csv"

    if args.mode == "final":
        episodes = args.episodes or config.final_evaluation_episodes
    else:
        episodes = args.episodes or config.evaluation_episodes

    env = make_flat_env(config)
    try:
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.n)
    finally:
        env.close()

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    agent.load(checkpoint_path)

    metrics = evaluate_agent(
        agent,
        config,
        num_episodes=episodes,
        mode=args.mode,
        seed_start=args.seed_start,
        return_episode_metrics=True,
    )
    append_evaluation_summary(
        csv_path=output_path,
        mode=args.mode,
        run_name=config.run_name,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    append_evaluation_details(
        csv_path=details_output_path,
        mode=args.mode,
        run_name=config.run_name,
        checkpoint_path=checkpoint_path,
        episode_metrics=metrics.get("episode_metrics", []),
    )

    print("Evaluation summary")
    print(f"Mode: {args.mode}")
    print(f"Run name: {config.run_name}")
    print(f"Checkpoint: {checkpoint_path}")
    seeds = metrics.get("seeds", [])
    if seeds:
        print(f"Seeds used: {seeds[0]} to {seeds[-1]} ({len(seeds)} episodes)")
    print(f"Mean return (average total reward): {metrics['mean_return']:.3f}")
    print(f"Return standard deviation: {metrics['std_return']:.3f}")
    print(f"Mean score (average episode score, from the game): {metrics['mean_score']:.3f}")
    print(f"Score standard deviation: {metrics['std_score']:.3f}")
    print(f"Mean length (average episode length in steps): {metrics['mean_length']:.3f}")
    print(f"Length standard deviation: {metrics['std_length']:.3f}")
    print("Per-episode results")
    for row in metrics.get("episode_metrics", []):
        print(
            f"Episode {int(row['episode']):2d} | "
            f"seed={int(row['seed']):5d} | "
            f"return={float(row['episode_return']):9.3f} | "
            f"score={float(row['episode_score']):7.3f} | "
            f"len={int(row['episode_length']):5d}"
        )
    print(f"Saved summary to: {output_path}")
    print(f"Saved detailed episode results to: {details_output_path}")


if __name__ == "__main__":
    main()
