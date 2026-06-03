"""Generate report-ready plots from the saved final evaluation summaries.

This script is intentionally focused on the data you already have:
- one run folder per seed
- one evaluation summary CSV inside each run
- multiple runs belonging to the same variant

The main job here is to answer:
"Across the 5 seeds for each variation, what is the average final performance,
and how much does it vary?"
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev, variance

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local setup
    raise ModuleNotFoundError(
        "matplotlib is required for plotting. Install matplotlib in your "
        "project environment before running generate_experiment_plots.py."
    ) from exc


RUN_NAME_PATTERN = re.compile(r"^(?P<variant>.+)_seed(?P<seed>-?\d+)$")

# A small fixed palette keeps the plots consistent across reruns.
PLOT_COLORS = [
    "#1F4E79",
    "#D97706",
    "#2F855A",
    "#B91C1C",
    "#6B7280",
    "#0F766E",
]


@dataclass
class RunSummary:
    """One final evaluation result for one trained seed run."""

    variant: str
    run_name: str
    seed: int | None
    evaluation_episodes: int
    first_seed: int | None
    last_seed: int | None
    mean_return: float
    std_return: float
    mean_score: float
    std_score: float
    mean_length: float
    std_length: float
    timestamp: str


@dataclass
class TrainingSeries:
    """One metric series from one seed run's training log."""

    variant: str
    run_name: str
    seed: int | None
    episodes: list[int]
    values: list[float]


def parse_args() -> argparse.Namespace:
    """Keep the CLI small and practical for report work."""

    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Generate report-ready plots from evaluation summaries.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=project_root / "runs",
        help="Folder containing one subfolder per seed run.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=project_root / "plots",
        help="Folder where all generated plots and CSV summaries will be saved.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="final",
        choices=("final", "validation"),
        help="Which evaluation mode to read from evaluation_summaries.csv.",
    )
    parser.add_argument(
        "--expected-runs-per-variant",
        type=int,
        default=5,
        help="Used only for warnings so you notice missing seed runs.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional explicit variant names to include, such as vanilla_dqn momentum_sensitive_dqn.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=20,
        help="Trailing moving-average window for the learning curve plot.",
    )
    return parser.parse_args()


def infer_variant_and_seed(run_name: str) -> tuple[str, int | None]:
    """Split a run name like vanilla_dqn_seed42 into variant and seed."""

    match = RUN_NAME_PATTERN.match(run_name)
    if match is None:
        return run_name, None

    return match.group("variant"), int(match.group("seed"))


def safe_int(value: str | None) -> int | None:
    """Convert CSV text to int while gracefully handling blanks."""

    if value in (None, ""):
        return None
    return int(float(value))


def safe_float(value: str | None) -> float:
    """Convert CSV text to float with a predictable fallback."""

    if value in (None, ""):
        return 0.0
    return float(value)


def choose_latest_summary_row(summary_csv: Path, mode: str) -> dict[str, str] | None:
    """Read one run's CSV and keep the latest row for the chosen mode.

    The file can contain multiple appended evaluation rows if you reran the
    evaluation command. In that case we want the newest matching entry.
    """

    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    matching_rows = [row for row in rows if row.get("mode", "") == mode]
    if not matching_rows:
        return None

    return matching_rows[-1]


def load_run_summaries(
    runs_dir: Path,
    mode: str,
    allowed_variants: set[str] | None = None,
) -> list[RunSummary]:
    """Collect one final summary row from every usable run folder."""

    if not runs_dir.exists():
        raise FileNotFoundError(
            f"Could not find runs directory at {runs_dir}. "
            "Train and evaluate the models first."
        )

    summaries: list[RunSummary] = []

    for run_dir in sorted(child for child in runs_dir.iterdir() if child.is_dir()):
        summary_csv = run_dir / "logs" / "evaluation_summaries.csv"
        if not summary_csv.exists():
            continue

        chosen_row = choose_latest_summary_row(summary_csv, mode=mode)
        if chosen_row is None:
            continue

        run_name = chosen_row.get("run_name", run_dir.name) or run_dir.name
        variant, seed = infer_variant_and_seed(run_name)

        if allowed_variants is not None and variant not in allowed_variants:
            continue

        summaries.append(
            RunSummary(
                variant=variant,
                run_name=run_name,
                seed=seed,
                evaluation_episodes=safe_int(chosen_row.get("evaluation_episodes")) or 0,
                first_seed=safe_int(chosen_row.get("first_seed")),
                last_seed=safe_int(chosen_row.get("last_seed")),
                mean_return=safe_float(chosen_row.get("mean_return")),
                std_return=safe_float(chosen_row.get("std_return")),
                mean_score=safe_float(chosen_row.get("mean_score")),
                std_score=safe_float(chosen_row.get("std_score")),
                mean_length=safe_float(chosen_row.get("mean_length")),
                std_length=safe_float(chosen_row.get("std_length")),
                timestamp=chosen_row.get("timestamp", ""),
            )
        )

    if not summaries:
        raise FileNotFoundError(
            "No matching evaluation summaries were found. "
            "Make sure run_evaluation.py has been executed for the requested runs."
        )

    return summaries


def load_training_series(
    runs_dir: Path,
    metric_name: str,
    allowed_variants: set[str] | None = None,
) -> dict[str, list[TrainingSeries]]:
    """Read one training metric series from every usable run folder."""

    if not runs_dir.exists():
        raise FileNotFoundError(
            f"Could not find runs directory at {runs_dir}. "
            "Train the models first so training_metrics.csv exists."
        )

    grouped_series: dict[str, list[TrainingSeries]] = defaultdict(list)

    for run_dir in sorted(child for child in runs_dir.iterdir() if child.is_dir()):
        training_csv = run_dir / "logs" / "training_metrics.csv"
        if not training_csv.exists():
            continue

        run_name = run_dir.name
        variant, seed = infer_variant_and_seed(run_name)

        if allowed_variants is not None and variant not in allowed_variants:
            continue

        episodes: list[int] = []
        values: list[float] = []

        with training_csv.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                episode = safe_int(row.get("episode"))
                if episode is None:
                    continue

                episodes.append(episode)
                values.append(safe_float(row.get(metric_name)))

        if episodes:
            grouped_series[variant].append(
                TrainingSeries(
                    variant=variant,
                    run_name=run_name,
                    seed=seed,
                    episodes=episodes,
                    values=values,
                )
            )

    if not grouped_series:
        raise FileNotFoundError(
            "No matching training metric series were found. "
            "Make sure training_metrics.csv exists for the requested runs."
        )

    for variant_runs in grouped_series.values():
        variant_runs.sort(key=lambda item: (item.seed is None, item.seed))

    return dict(grouped_series)


def group_summaries_by_variant(
    summaries: list[RunSummary],
) -> dict[str, list[RunSummary]]:
    """Group seed runs under their shared method name."""

    grouped: dict[str, list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.variant].append(summary)

    for variant_runs in grouped.values():
        variant_runs.sort(key=lambda item: (item.seed is None, item.seed))

    return dict(grouped)


def sample_std(values: list[float]) -> float:
    """Use sample standard deviation across seeds for report error bars."""

    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def sample_variance(values: list[float]) -> float:
    """Variance across seeds is useful for the experiment tracker tables."""

    if len(values) < 2:
        return 0.0
    return float(variance(values))


def moving_average(values: list[float], window: int) -> list[float]:
    """Smooth a noisy series with a simple trailing moving average."""

    if window <= 1 or not values:
        return list(values)

    smoothed_values: list[float] = []
    for index in range(len(values)):
        start_index = max(0, index - window + 1)
        current_window = values[start_index : index + 1]
        smoothed_values.append(sum(current_window) / len(current_window))

    return smoothed_values


def aggregate_training_series(
    grouped_series: dict[str, list[TrainingSeries]],
    smoothing_window: int,
) -> dict[str, dict[str, list[float] | int]]:
    """Aggregate per-seed training curves into mean and std per episode."""

    aggregated_curves: dict[str, dict[str, list[float] | int]] = {}

    for variant in ordered_variant_names(grouped_series):
        runs = grouped_series[variant]
        episode_to_values: dict[int, list[float]] = defaultdict(list)

        for run in runs:
            for episode, value in zip(run.episodes, run.values):
                episode_to_values[episode].append(value)

        ordered_episodes = sorted(episode_to_values.keys())
        mean_values = [float(mean(episode_to_values[episode])) for episode in ordered_episodes]
        std_values = [sample_std(episode_to_values[episode]) for episode in ordered_episodes]

        aggregated_curves[variant] = {
            "episodes": ordered_episodes,
            "mean": moving_average(mean_values, smoothing_window),
            "std": moving_average(std_values, smoothing_window),
            "num_runs": len(runs),
        }

    return aggregated_curves


def build_aggregate_rows(
    grouped_runs: dict[str, list[RunSummary]],
) -> list[dict[str, float | int | str]]:
    """Turn per-seed summaries into one aggregate row per variant."""

    aggregate_rows: list[dict[str, float | int | str]] = []

    for variant in ordered_variant_names(grouped_runs):
        runs = grouped_runs[variant]
        score_values = [run.mean_score for run in runs]
        return_values = [run.mean_return for run in runs]
        length_values = [run.mean_length for run in runs]

        aggregate_rows.append(
            {
                "variant": variant,
                "display_name": pretty_variant_name(variant),
                "num_runs": len(runs),
                "mean_score": float(mean(score_values)),
                "score_std": sample_std(score_values),
                "score_variance": sample_variance(score_values),
                "mean_return": float(mean(return_values)),
                "return_std": sample_std(return_values),
                "return_variance": sample_variance(return_values),
                "mean_length": float(mean(length_values)),
                "length_std": sample_std(length_values),
                "length_variance": sample_variance(length_values),
            }
        )

    return aggregate_rows


def ordered_variant_names(grouped_runs: dict[str, list[RunSummary]]) -> list[str]:
    """Keep vanilla first when it exists, then sort the rest alphabetically."""

    variants = sorted(grouped_runs.keys())
    if "vanilla_dqn" in variants:
        variants.remove("vanilla_dqn")
        variants.insert(0, "vanilla_dqn")
    return variants


def pretty_variant_name(variant: str) -> str:
    """Make variant names look like report labels instead of folder names."""

    words = variant.split("_")
    pretty_words: list[str] = []

    for word in words:
        if word.upper() == "DQN":
            pretty_words.append("DQN")
        else:
            pretty_words.append(word.capitalize())

    return " ".join(pretty_words)


def ensure_plots_dir(plots_dir: Path) -> None:
    """Create the root plots folder once before saving anything."""

    plots_dir.mkdir(parents=True, exist_ok=True)


def write_per_run_csv(plots_dir: Path, summaries: list[RunSummary]) -> Path:
    """Save the exact per-seed numbers that feed the plots."""

    output_path = plots_dir / "per_run_final_evaluation_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "run_name",
                "seed",
                "evaluation_episodes",
                "first_seed",
                "last_seed",
                "mean_score",
                "std_score",
                "mean_return",
                "std_return",
                "mean_length",
                "std_length",
                "timestamp",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)

    return output_path


def write_aggregate_csv(
    plots_dir: Path,
    aggregate_rows: list[dict[str, float | int | str]],
) -> Path:
    """Save the across-seed summary that the report can quote directly."""

    output_path = plots_dir / "aggregate_final_evaluation_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "display_name",
                "num_runs",
                "mean_score",
                "score_std",
                "score_variance",
                "mean_return",
                "return_std",
                "return_variance",
                "mean_length",
                "length_std",
                "length_variance",
            ],
        )
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    return output_path


def build_variant_color_map(variant_names: list[str]) -> dict[str, str]:
    """Reuse the same color for a variant across every figure."""

    color_map: dict[str, str] = {}
    for index, variant in enumerate(variant_names):
        color_map[variant] = PLOT_COLORS[index % len(PLOT_COLORS)]
    return color_map


def plot_bar_chart(
    aggregate_rows: list[dict[str, float | int | str]],
    metric_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    color_map: dict[str, str],
) -> None:
    """Create a clean comparison chart with across-seed error bars."""

    display_names = [str(row["display_name"]) for row in aggregate_rows]
    values = [float(row[metric_key]) for row in aggregate_rows]
    errors = [float(row[std_key]) for row in aggregate_rows]
    colors = [color_map[str(row["variant"])] for row in aggregate_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        display_names,
        values,
        yerr=errors,
        capsize=8,
        color=colors,
        edgecolor="#1F2937",
        linewidth=1.0,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(
    aggregated_curves: dict[str, dict[str, list[float] | int]],
    output_path: Path,
    color_map: dict[str, str],
    smoothing_window: int,
) -> None:
    """Plot the mean training score over episodes with across-seed std bands."""

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for variant in ordered_variant_names(aggregated_curves):
        curve = aggregated_curves[variant]
        episodes = list(curve["episodes"])
        mean_values = list(curve["mean"])
        std_values = list(curve["std"])

        lower_band = [mean_value - std_value for mean_value, std_value in zip(mean_values, std_values)]
        upper_band = [mean_value + std_value for mean_value, std_value in zip(mean_values, std_values)]
        display_name = pretty_variant_name(variant)
        color = color_map[variant]

        ax.plot(
            episodes,
            mean_values,
            color=color,
            linewidth=2.2,
            label=f"{display_name} (n={curve['num_runs']})",
        )
        ax.fill_between(
            episodes,
            lower_band,
            upper_band,
            color=color,
            alpha=0.18,
        )

    ax.set_title("Learning Curves: Mean Episode Score Across Training Seeds", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean episode score")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    # The smoothing note helps the reader understand why the curves look calmer
    # than the raw episode-by-episode scores in the CSV files.
    ax.text(
        0.99,
        0.02,
        f"Shaded band = across-seed std, smoothing window = {smoothing_window}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_boxplot(
    grouped_runs: dict[str, list[RunSummary]],
    output_path: Path,
    color_map: dict[str, str],
) -> None:
    """Show the final-score distribution across seeds for each variant.

    This directly answers the reliability question, which matters a lot in a
    multi-seed reinforcement learning project.
    """

    ordered_variants = ordered_variant_names(grouped_runs)
    display_names = [pretty_variant_name(variant) for variant in ordered_variants]
    score_lists = [
        [run.mean_score for run in grouped_runs[variant]]
        for variant in ordered_variants
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    boxplot = ax.boxplot(
        score_lists,
        labels=display_names,
        patch_artist=True,
        widths=0.55,
    )

    for patch, variant in zip(boxplot["boxes"], ordered_variants):
        patch.set_facecolor(color_map[variant])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#1F2937")
        patch.set_linewidth(1.0)

    for element_name in ("whiskers", "caps", "medians"):
        for item in boxplot[element_name]:
            item.set_color("#1F2937")
            item.set_linewidth(1.0)

    # Overlay the actual seed results so the reader can see how many runs exist.
    for x_position, variant in enumerate(ordered_variants, start=1):
        for offset_index, run in enumerate(grouped_runs[variant]):
            horizontal_offset = ((offset_index % 5) - 2) * 0.04
            ax.scatter(
                x_position + horizontal_offset,
                run.mean_score,
                color="#111827",
                s=35,
                zorder=3,
            )

    ax.set_title("Final Evaluation Score Distribution Across Seeds", fontsize=14, fontweight="bold")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Final mean score")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_run_warnings(
    grouped_runs: dict[str, list[RunSummary]],
    expected_runs_per_variant: int,
) -> None:
    """Warn when a variant is missing some of its planned seed runs."""

    for variant in ordered_variant_names(grouped_runs):
        run_count = len(grouped_runs[variant])
        if run_count != expected_runs_per_variant:
            print(
                f"Warning: variant '{variant}' has {run_count} run(s), "
                f"but {expected_runs_per_variant} were expected.",
                flush=True,
            )


def main() -> None:
    """Read saved summaries, generate plots, and save the aggregate numbers."""

    args = parse_args()
    allowed_variants = set(args.variants) if args.variants else None

    ensure_plots_dir(args.plots_dir)

    summaries = load_run_summaries(
        runs_dir=args.runs_dir,
        mode=args.mode,
        allowed_variants=allowed_variants,
    )
    training_series = load_training_series(
        runs_dir=args.runs_dir,
        metric_name="episode_score",
        allowed_variants=allowed_variants,
    )
    grouped_runs = group_summaries_by_variant(summaries)
    aggregate_rows = build_aggregate_rows(grouped_runs)
    variant_names = ordered_variant_names(grouped_runs)
    color_map = build_variant_color_map(variant_names)
    aggregated_learning_curves = aggregate_training_series(
        grouped_series=training_series,
        smoothing_window=args.smoothing_window,
    )

    print_run_warnings(
        grouped_runs=grouped_runs,
        expected_runs_per_variant=args.expected_runs_per_variant,
    )

    per_run_csv = write_per_run_csv(args.plots_dir, summaries)
    aggregate_csv = write_aggregate_csv(args.plots_dir, aggregate_rows)

    plot_learning_curves(
        aggregated_curves=aggregated_learning_curves,
        output_path=args.plots_dir / "plot_1_learning_curves_mean_episode_score.png",
        color_map=color_map,
        smoothing_window=args.smoothing_window,
    )
    plot_bar_chart(
        aggregate_rows=aggregate_rows,
        metric_key="mean_score",
        std_key="score_std",
        ylabel="Final mean score",
        title="Final Score Comparison Across Variants",
        output_path=args.plots_dir / "plot_2_final_score_comparison_bar_chart.png",
        color_map=color_map,
    )
    plot_bar_chart(
        aggregate_rows=aggregate_rows,
        metric_key="mean_return",
        std_key="return_std",
        ylabel="Final mean return",
        title="Final Return Comparison Across Variants",
        output_path=args.plots_dir / "plot_3_final_return_comparison_bar_chart.png",
        color_map=color_map,
    )
    plot_bar_chart(
        aggregate_rows=aggregate_rows,
        metric_key="mean_length",
        std_key="length_std",
        ylabel="Final mean episode length",
        title="Final Episode Length Comparison Across Variants",
        output_path=args.plots_dir / "plot_4_final_episode_length_comparison_bar_chart.png",
        color_map=color_map,
    )
    plot_score_boxplot(
        grouped_runs=grouped_runs,
        output_path=args.plots_dir / "plot_5_seed_variance_box_plot_final_score.png",
        color_map=color_map,
    )

    print("Generated plot files:", flush=True)
    print(f"- {args.plots_dir / 'plot_1_learning_curves_mean_episode_score.png'}", flush=True)
    print(f"- {args.plots_dir / 'plot_2_final_score_comparison_bar_chart.png'}", flush=True)
    print(f"- {args.plots_dir / 'plot_3_final_return_comparison_bar_chart.png'}", flush=True)
    print(f"- {args.plots_dir / 'plot_4_final_episode_length_comparison_bar_chart.png'}", flush=True)
    print(f"- {args.plots_dir / 'plot_5_seed_variance_box_plot_final_score.png'}", flush=True)
    print("Generated data files:", flush=True)
    print(f"- {per_run_csv}", flush=True)
    print(f"- {aggregate_csv}", flush=True)


if __name__ == "__main__":
    main()
