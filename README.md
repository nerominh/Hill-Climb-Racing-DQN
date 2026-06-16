# Hill Climb Racing DQN

A reinforcement learning project that trains and compares DQN-style agents on a physics-based Hill Climb Racing Gymnasium environment.

This repository combines:

1. A reused Hill Climb Racing Gymnasium simulator from [`alexzh3/hillclimbracing`](https://github.com/alexzh3/hillclimbracing), vendored locally in [`hillclimbracing`](./hillclimbracing)
2. A separate project-specific DQN package in [`hcr_dqn`](./hcr_dqn)
3. Multi-seed experiment logs, generated plots, and project documentation for comparing DQN variants

The simulator is treated as the environment layer. My reinforcement-learning work lives around it in `hcr_dqn`, especially the Q-network, agent logic, training/evaluation flow, and experiment tracking.

## Demo

The original environment provides the Hill Climb Racing physics task used by this project:

<p align="center">
  <img src="./hillclimbracing/hill_racing_env/envs/pictures/hcr_demo.gif" alt="Hill Climb Racing Gymnasium environment demo" width="720">
</p>

## My DQN Experiment Showcase

The project compares vanilla DQN, DoubleDQN, momentum-sensitive reward shaping, and an anti-stall reward branch across multiple seeds. The plots below are generated from the saved CSV logs using [`hcr_dqn/generate_experiment_plots.py`](./hcr_dqn/generate_experiment_plots.py).

### Learning Curves

<p align="center">
  <img src="./plots/plot_1_learning_curves_mean_episode_score.png" alt="Mean training episode score learning curves across DQN variants" width="780">
</p>

### Final Score Comparison

<p align="center">
  <img src="./plots/plot_2_final_score_comparison_bar_chart.png" alt="Final held-out score comparison across DQN variants" width="720">
</p>

### Final Score Distribution Across Seeds

<p align="center">
  <img src="./plots/plot_6_final_evaluation_score_distribution_across_seeds.png" alt="Final score distribution across training seeds" width="780">
</p>

Current summary from [`plots/aggregate_final_evaluation_metrics.csv`](./plots/aggregate_final_evaluation_metrics.csv):

| Method | Runs | Mean score | Score std | Mean return | Mean length | Takeaway |
|---|---:|---:|---:|---:|---:|---|
| Momentum DQN | 5 | 529.33 | 64.10 | 2322.02 | 2248.01 | Best current mean score and return |
| Momentum Double | 5 | 516.55 | 58.04 | 2017.76 | 2380.01 | Most stable final score and longest episodes |
| Vanilla Double | 5 | 482.42 | 121.38 | 1786.62 | 2202.62 | Improves over vanilla DQN |
| Anti-stall DQN | 5 | 424.98 | 125.59 | 1738.90 | 1938.77 | Tested but discontinued |
| Vanilla DQN | 5 | 410.03 | 125.32 | 1670.44 | 1912.07 | Baseline reference |

## What This Repository Adds

Compared with the original `hillclimbracing` environment package, this repository adds a DQN-focused research layer:

- [`hcr_dqn/q_network.py`](./hcr_dqn/q_network.py)
  Defines the PyTorch Q-network used by the agents.

- [`hcr_dqn/dqn_agent.py`](./hcr_dqn/dqn_agent.py)
  Implements vanilla DQN learning, epsilon-greedy action selection, target-network updates, checkpoint save/load, the DoubleDQN target option, and reward-shaping variants.

- [`hcr_dqn/env_wrappers.py`](./hcr_dqn/env_wrappers.py)
  Converts the environment's dictionary observation into a flat 7-value vector for the MLP.

- [`hcr_dqn/train_dqn.py`](./hcr_dqn/train_dqn.py)
  Runs training, logging, validation evaluation, and best-checkpoint selection.

- [`hcr_dqn/run_evaluation.py`](./hcr_dqn/run_evaluation.py)
  Evaluates trained checkpoints on held-out seeds and writes final CSV results.

- [`hcr_dqn/watch_checkpoint.py`](./hcr_dqn/watch_checkpoint.py)
  Opens a rendered playback window for visually inspecting trained checkpoints.

- [`hcr_dqn/generate_experiment_plots.py`](./hcr_dqn/generate_experiment_plots.py)
  Generates report-ready comparison plots from training and evaluation CSV files.

- Project documentation
  [`AI_HUMAN_WORK_DIVISION.md`](./AI_HUMAN_WORK_DIVISION.md), [`PROJECT_PHASE_TRACKER.md`](./PROJECT_PHASE_TRACKER.md), and [`EXPERIMENT_RESULTS_TRACKER.md`](./EXPERIMENT_RESULTS_TRACKER.md) record ownership, methodology, experiment results, and reporting notes.

## Project Design

The repository keeps a clear boundary between the simulator and the DQN work:

| Layer | Folder | Role |
|---|---|---|
| Environment | [`hillclimbracing`](./hillclimbracing) | Reused Gymnasium simulator, physics, terrain, rendering, and original reward/action definitions |
| RL implementation | [`hcr_dqn`](./hcr_dqn) | DQN agents, Q-network, wrappers, training, evaluation, plots, and checkpoint tools |
| Experiments | [`runs`](./runs), [`plots`](./plots) | Saved checkpoints, CSV logs, final evaluation metrics, and generated figures |
| Documentation | Markdown files in the repo root | Project plan, result tracker, and AI/human work division |

## Environment Facts Used By DQN

The reused simulator registers this Gymnasium environment:

```python
"hill_racing_env/HillRacing-v0"
```

The current DQN setup uses:

| Setting | Value |
|---|---|
| `action_space` | `discrete_3` |
| `reward_function` | `distance` |
| `reward_type` | `soft` |
| `max_steps` | `1200` |
| `original_noise` | `False` |

The discrete actions are:

| Action | Meaning |
|---:|---|
| `0` | Idle |
| `1` | Gas |
| `2` | Reverse |

The original observation is a Gymnasium `Dict`. The DQN wrapper flattens it into:

```text
chassis_x,
chassis_y,
chassis_angle_deg,
back_wheel_speed,
front_wheel_speed,
back_wheel_on_ground,
front_wheel_on_ground
```

## Repository Structure

```text
Hill Climb Racing AI/
├── README.md
├── LICENSE
├── AI_HUMAN_WORK_DIVISION.md
├── EXPERIMENT_RESULTS_TRACKER.md
├── PROJECT_PHASE_TRACKER.md
├── REPORT_DRAFT.md
├── quick_test.py
├── hcr_dqn/
│   ├── bootstrap.py
│   ├── configs.py
│   ├── dqn_agent.py
│   ├── env_wrappers.py
│   ├── evaluate_dqn.py
│   ├── generate_experiment_plots.py
│   ├── q_network.py
│   ├── replay_buffer.py
│   ├── run_evaluation.py
│   ├── train_dqn.py
│   └── watch_checkpoint.py
├── hillclimbracing/
│   ├── LICENSE
│   ├── README.md
│   ├── pyproject.toml
│   ├── hill_racing_env/
│   └── tests/
├── plots/
└── runs/
```

## Setup

This project is easiest to run in a Python 3.10 Conda environment.

### 1. Create and activate the environment

```bash
conda create -n hcr python=3.10 -y
conda activate hcr
```

### 2. Install the local simulator and DQN dependencies

From the repository root:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./hillclimbracing
python -m pip install torch pytest
```

Core simulator dependencies come from [`hillclimbracing/pyproject.toml`](./hillclimbracing/pyproject.toml):

- `gymnasium`
- `pygame`
- `box2d-py`
- `numpy`
- `noise`

The DQN code additionally requires:

- `torch`

If the `hill-climb-play` command exists but fails with `ModuleNotFoundError: No module named 'hill_racing_env'`, reinstall the local simulator inside the active environment:

```bash
python -m pip install -e ./hillclimbracing
```

### 3. Headless server note

If running through SSH or another headless Linux environment, set dummy SDL drivers before non-visual environment interaction:

```bash
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
```

Human rendering usually requires a real display, X forwarding, or `xvfb`.

## Quick Start

### Smoke test the environment import

```bash
python quick_test.py
```

### Play the original environment manually

```bash
hill-climb-play
```

This command comes from the reused `hillclimbracing` package after it is installed with `pip install -e ./hillclimbracing`.

### Train a DQN agent

Training reads settings from [`hcr_dqn/configs.py`](./hcr_dqn/configs.py).

```bash
python hcr_dqn/train_dqn.py
```

Change these fields in `DQNConfig` to select an experiment:

- `run_name`
- `seed`
- `agent_variant`
- `td_target_mode`
- `reward_function`
- `reward_type`
- `num_episodes`

Use `agent_variant` for the reward/behavior style:

- `vanilla`
- `momentum_sensitive`
- `antistall_momentum`

Use `td_target_mode` for the target rule:

- `dqn`
- `double_dqn`

For example:

```python
agent_variant = "momentum_sensitive"
td_target_mode = "double_dqn"
```

This runs the momentum-sensitive reward with the DoubleDQN target calculation.

### Evaluate a checkpoint

```bash
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7
```

Useful variants:

```bash
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7 --mode validation
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7 --episodes 10
python -m hcr_dqn.run_evaluation --checkpoint runs/momentum_sensitive_dqn_seed7/checkpoints/best_model.pt
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7 --output runs/momentum_sensitive_dqn_seed7/logs/final_eval_custom.csv
```

Default output files:

```text
runs/<run_name>/logs/evaluation_summaries.csv
runs/<run_name>/logs/evaluation_episode_details.csv
```

### Watch a trained checkpoint

```bash
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7
```

Useful variants:

```bash
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7 --episodes 1
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7 --seed 2026
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7 --step-delay 0.01
python -m hcr_dqn.watch_checkpoint --checkpoint runs/momentum_sensitive_dqn_seed7/checkpoints/best_model.pt
```

### Regenerate experiment plots

```bash
python -m hcr_dqn.generate_experiment_plots
```

Generated figures and aggregate CSV files are written to [`plots`](./plots).

## Experiment Methods

| Method | Main idea | Implementation |
|---|---|---|
| `vanilla_dqn` | Baseline DQN with target network and replay buffer | `DQNAgent` |
| `momentum_sensitive_dqn` | Adds reward shaping for forward momentum and stability | `MomentumSensitiveDQNAgent` |
| `vanilla_double_dqn` | Uses DoubleDQN target selection/evaluation split | `td_target_mode = "double_dqn"` |
| `momentum_sensitive_double_dqn` | Combines momentum reward shaping with DoubleDQN targets | `agent_variant = "momentum_sensitive"`, `td_target_mode = "double_dqn"` |
| `antistall_momentum_dqn` | Tests an explicit stuck-recovery reward penalty | `AntiStallMomentumDQNAgent` |

The current result tracker treats `antistall_momentum_dqn` as a completed but discontinued branch because it did not improve the previous momentum-based methods.

## Training Outputs

Each run writes artifacts under:

```text
runs/<run_name>/
```

Typical files:

- `checkpoints/best_model.pt`
- `logs/training_metrics.csv`
- `logs/validation_metrics.csv`
- `logs/evaluation_summaries.csv`
- `logs/evaluation_episode_details.csv`

The root [`plots`](./plots) folder stores cross-run comparison outputs generated from those logs.

## Documentation Files

- [`AI_HUMAN_WORK_DIVISION.md`](./AI_HUMAN_WORK_DIVISION.md)
  Records reused code, AI-assisted tooling, and human-owned implementation work.

- [`PROJECT_PHASE_TRACKER.md`](./PROJECT_PHASE_TRACKER.md)
  Tracks project phases, protocol notes, and code understanding.

- [`EXPERIMENT_RESULTS_TRACKER.md`](./EXPERIMENT_RESULTS_TRACKER.md)
  Records experiment numbers, seed-level notes, plot interpretation, and final comparison summaries.

- [`REPORT_DRAFT.md`](./REPORT_DRAFT.md)
  Draft report material.

## Acknowledgement

This repository directly acknowledges and builds on the original `hillclimbracing` environment project.

Credit is due to:

- Alex Zheng for the original `hillclimbracing` Gymnasium environment package
- The original thesis-driven environment work described in the upstream project
- Code Bullet for the original Hill Climb Racing AI JavaScript inspiration
- Farama Gymnasium for the environment API
- Box2D for the physics engine
- Pygame for rendering and interaction support

The original environment remains an important part of this repository and is not presented as newly authored from scratch here. The top-level DQN code, experiment structure, generated plots, and project documents are additions built around that reused simulator.

## License

The original `hillclimbracing` package included in this repository is licensed under the GNU General Public License v3.0 only.

Because this repository distributes that GPL-licensed code together with top-level additions and modifications as a single combined codebase, the repository should be treated as distributed under the GNU General Public License v3.0 as well.

That means:

- the vendored simulator code in [`hillclimbracing`](./hillclimbracing) remains under GPL-3.0-only
- the new top-level DQN code and repository-level additions are also distributed under GPL-3.0-only in this combined repository
- redistribution of this combined repository should preserve the GPL license text and the original attribution notices

Full license text is provided in:

- [`LICENSE`](./LICENSE)
- [`hillclimbracing/LICENSE`](./hillclimbracing/LICENSE)

This README is an informational project summary and not legal advice.

## Original Source References

- Upstream environment project: <https://github.com/alexzh3/hillclimbracing>
- Inspiration game project by Code Bullet: <https://github.com/Code-Bullet/Hill-Climb-Racing-AI>
