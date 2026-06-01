# Hill Climb Racing DQN

This repository combines two layers of work into a single research and development codebase:

1. A reused and vendored Hill Climb Racing Gymnasium environment in [`hillclimbracing`](./hillclimbracing)
2. A separate DQN training and evaluation package in [`hcr_dqn`](./hcr_dqn) built on top of that environment

The main goal of this repository is to study value-based reinforcement learning for a physics-driven Hill Climb Racing task. The environment package provides the simulator, while the top-level project adds DQN-oriented training code, experiment tracking, evaluation utilities, and project documentation.

## Repository Purpose

This is not only the original environment package and not only a standalone DQN script collection. It is a combined project that:

- reuses the original Hill Climb Racing simulator as the experimental environment
- adds a DQN pipeline for discrete-control reinforcement learning
- separates simulator code from RL code so the contribution boundary stays clear
- supports training, evaluation, checkpoint watching, and experiment logging

In practice, the `hillclimbracing` folder is the simulation dependency and the `hcr_dqn` folder is the project-specific reinforcement learning layer.

## Relationship To The Original `hillclimbracing` Project

The environment code in [`hillclimbracing`](./hillclimbracing) is derived from the original open-source project:

- Original project: `alexzh3/hillclimbracing`
- Original package name: `hill-climb-racing-env`
- Original environment ID: `hill_racing_env/HillRacing-v0`

That original project provides:

- the Box2D-based Hill Climb Racing simulator
- Gymnasium environment registration
- terrain generation
- reward configurations
- rendering and human play support
- included PPO baseline assets and tests

This repository keeps that environment available locally, then builds new top-level DQN work around it instead of rewriting the simulator from scratch.

## What Is New In This Repository

Compared with the original environment package, this repository adds project-level work focused on DQN experimentation and coursework-style reporting:

- `hcr_dqn/bootstrap.py`
  Ensures the simulator package can be imported from the local repository layout.

- `hcr_dqn/env_wrappers.py`
  Converts the environment's dictionary observation into a flat numeric vector suitable for an MLP-based DQN.

- `hcr_dqn/replay_buffer.py`
  Stores transition data for experience replay.

- `hcr_dqn/q_network.py`
  Defines the Q-network used by the DQN agent.

- `hcr_dqn/dqn_agent.py`
  Implements the main learning logic, action selection, target network updates, checkpoint save/load behavior, and the momentum-sensitive agent variation currently present in this repository.

- `hcr_dqn/train_dqn.py`
  Runs end-to-end DQN training, logging, periodic evaluation, and checkpoint selection.

- `hcr_dqn/evaluate_dqn.py`
  Evaluates trained agents on held-out seeds.

- `hcr_dqn/run_evaluation.py`
  Provides an evaluation entry point for checkpoint-based reporting.

- `hcr_dqn/watch_checkpoint.py`
  Helps visually inspect trained checkpoints.

- Top-level planning and experiment documents
  Files such as [`HCR_DQN_Project_Plan.md`](./HCR_DQN_Project_Plan.md), [`PROJECT_PHASE_TRACKER.md`](./PROJECT_PHASE_TRACKER.md), [`EXPERIMENT_RESULTS_TRACKER.md`](./EXPERIMENT_RESULTS_TRACKER.md), and [`AI_HUMAN_WORK_DIVISION.md`](./AI_HUMAN_WORK_DIVISION.md) document the project direction, experiments, and ownership boundaries.

## Project Highlights

- Uses a physics-based Hill Climb Racing environment backed by Box2D and Pygame
- Treats the simulator as a reusable dependency instead of mixing all code into one package
- Focuses on DQN-style value-based learning over the environment's discrete action space
- Includes evaluation and checkpoint inspection tooling
- Keeps research notes and experiment tracking in-repo
- Preserves the original environment package, attribution, and licensing

## Repository Structure

```text
Hill Climb Racing AI/
├── README.md
├── LICENSE
├── AI_HUMAN_WORK_DIVISION.md
├── EXPERIMENT_RESULTS_TRACKER.md
├── HCR_DQN_Project_Plan.md
├── PROJECT_PHASE_TRACKER.md
├── comprehensive_run_summaries.csv
├── quick_test.py
├── hcr_dqn/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── configs.py
│   ├── dqn_agent.py
│   ├── env_wrappers.py
│   ├── evaluate_dqn.py
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
└── runs/
```

## Environment Facts Used By The DQN Code

The local simulator package exposes the Gymnasium environment:

```python
"hill_racing_env/HillRacing-v0"
```

The DQN code is built around the environment's discrete action setup:

- `action_space="discrete_3"`
- observation space is a `Dict`
- the wrapper flattens the observation into:
  `chassis_x, chassis_y, chassis_angle_deg, back_wheel_speed, front_wheel_speed, back_wheel_on_ground, front_wheel_on_ground`

This is why the RL code lives outside the simulator package: the environment remains reusable, while the DQN layer can evolve independently.

## Setup

This repository is easiest to run in a fresh Python 3.10 Conda environment.

### 1. Create and activate an environment

```bash
conda create -n hcr python=3.10 -y
conda activate hcr
```

### 2. Install system prerequisites

`box2d-py` commonly needs `swig` available at build time.

Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential swig
```

### 3. Install Python dependencies

From the repository root:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ./hillclimbracing
pip install torch pytest
```

Core dependencies come from [`hillclimbracing/pyproject.toml`](./hillclimbracing/pyproject.toml):

- `gymnasium`
- `pygame`
- `box2d-py`
- `numpy`
- `noise`

The DQN training code additionally requires:

- `torch`

### 4. Headless server note

If you run this project over SSH on a headless Linux server, exporting dummy SDL drivers is often necessary before using `pygame`:

```bash
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
```

This is especially useful for testing and non-visual environment interaction. Human rendering usually requires a real display, X forwarding, or `xvfb`.

## Quick Start

### Smoke test the environment import

```bash
python quick_test.py
```

### Run DQN training

Training currently does not expose CLI flags. It uses the values stored in [`hcr_dqn/configs.py`](./hcr_dqn/configs.py), so edit `DQNConfig` there when you want to change the run name, seed, agent variant, reward settings, or other hyperparameters.

```bash
python hcr_dqn/train_dqn.py
```

### Evaluate a saved checkpoint

```bash
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7
```

### Watch a checkpoint

```bash
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7
```

## Command-Line Interface

The repository currently exposes CLI argument parsing for:

- `python -m hcr_dqn.run_evaluation`
- `python -m hcr_dqn.watch_checkpoint`

The training entry point:

- `python hcr_dqn/train_dqn.py`

does not currently parse command-line arguments. It reads configuration directly from `DQNConfig` in [`hcr_dqn/configs.py`](./hcr_dqn/configs.py).

### `run_evaluation`

Use this command to evaluate a saved checkpoint and write summary CSV files.

Example:

```bash
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7
```

What it does:

- loads the checkpoint for the requested run
- evaluates the trained agent
- prints aggregate metrics to the terminal
- appends a summary CSV row
- appends per-episode evaluation rows

Default checkpoint path:

```text
runs/<run_name>/checkpoints/best_model.pt
```

Default output files:

```text
runs/<run_name>/logs/evaluation_summaries.csv
runs/<run_name>/logs/evaluation_episode_details.csv
```

Supported arguments:

- `--run-name <name>`
  Run folder name under `runs/`. If omitted, the code falls back to `DQNConfig.run_name`.

- `--checkpoint <path>`
  Explicit path to a checkpoint file. Use this if you do not want the default `runs/<run_name>/checkpoints/best_model.pt` path.

- `--mode {validation,final}`
  Chooses the evaluation regime.
  `validation` uses the smaller held-out validation setup.
  `final` uses the larger final evaluation setup.
  Default: `final`

- `--episodes <int>`
  Overrides the number of evaluation episodes. If omitted, the script uses the count configured for the chosen mode.

- `--seed-start <int>`
  Overrides the first seed used for final evaluation. The help text notes that this is ignored in validation mode.

- `--output <path>`
  Custom CSV path for the summary file.

Useful examples:

```bash
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7 --mode validation
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7 --episodes 10
python -m hcr_dqn.run_evaluation --checkpoint runs/momentum_sensitive_dqn_seed7/checkpoints/best_model.pt
python -m hcr_dqn.run_evaluation --run-name momentum_sensitive_dqn_seed7 --output runs/momentum_sensitive_dqn_seed7/logs/final_eval_custom.csv
```

### `watch_checkpoint`

Use this command to open a rendered window and watch a trained checkpoint play greedily.

Example:

```bash
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7
```

Default checkpoint path:

```text
runs/<run_name>/checkpoints/best_model.pt
```

Supported arguments:

- `--run-name <name>`
  Run folder name under `runs/`. If omitted, the script falls back to `DQNConfig.run_name`.

- `--checkpoint <path>`
  Explicit checkpoint file path.

- `--episodes <int>`
  Number of rendered episodes to watch.
  Default: `3`

- `--seed <int>`
  Base seed for the rendered episodes. If omitted, the script uses `DQNConfig.seed`.

- `--step-delay <float>`
  Extra seconds to sleep after each environment step so playback is easier to watch.
  Default: `0.02`

Useful examples:

```bash
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7 --episodes 1
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7 --seed 2026
python -m hcr_dqn.watch_checkpoint --run-name momentum_sensitive_dqn_seed7 --step-delay 0.01
python -m hcr_dqn.watch_checkpoint --checkpoint runs/momentum_sensitive_dqn_seed7/checkpoints/best_model.pt
```

### `train_dqn`

Training currently works as a script entry point, not a parsed CLI.

Run it with:

```bash
python hcr_dqn/train_dqn.py
```

To change behavior, edit the fields in [`hcr_dqn/configs.py`](./hcr_dqn/configs.py), especially:

- `run_name`
- `agent_variant`
- `seed`
- `reward_function`
- `reward_type`
- `action_space`
- `num_episodes`
- `evaluation_frequency`

If you want fully scriptable experiment runs later, a natural next improvement would be to add an `argparse` interface to `train_dqn.py` so run names, seeds, and reward settings can be changed from the command line.

## Training Outputs

Training artifacts are written under [`runs`](./runs). A run typically contains:

- checkpoints
- CSV logs
- evaluation summaries
- plots or later analysis outputs

The configuration object in [`hcr_dqn/configs.py`](./hcr_dqn/configs.py) controls run naming, output paths, environment settings, and core DQN hyperparameters.

## Documentation Files

This repository includes project-management and reporting artifacts alongside the code:

- [`HCR_DQN_Project_Plan.md`](./HCR_DQN_Project_Plan.md)
  Repository-grounded project framing and implementation roadmap.

- [`PROJECT_PHASE_TRACKER.md`](./PROJECT_PHASE_TRACKER.md)
  Ongoing progress and structure notes.

- [`EXPERIMENT_RESULTS_TRACKER.md`](./EXPERIMENT_RESULTS_TRACKER.md)
  Seed-level experiment tracking, metric interpretation, and result summaries.

- [`AI_HUMAN_WORK_DIVISION.md`](./AI_HUMAN_WORK_DIVISION.md)
  Ownership notes describing reused code, AI-assisted scaffolding, and human-authored algorithm work.

## Acknowledgement

This repository directly acknowledges and builds upon the original `hillclimbracing` environment project.

Credit is due to:

- Alex Zheng for the original `hillclimbracing` Gymnasium environment package
- The original thesis-driven environment work described in the upstream project
- Code Bullet for the original Hill Climb Racing AI JavaScript inspiration
- Farama Gymnasium for the environment API
- Box2D for the physics engine
- Pygame for rendering and interaction support

The original environment remains an important part of this repository and is not being presented as newly authored from scratch here. The top-level DQN code, experiment structure, and project documents are additions built around that reused simulator.

## Modification Notice

This repository contains original and modified material beyond the upstream `hillclimbracing` package, including:

- a separate DQN package under `hcr_dqn`
- local repository integration helpers
- experiment runners and evaluation scripts
- new project-level documentation
- experiment tracking artifacts and run outputs

These additions make the repository a combined derivative work built on top of GPL-licensed source material.

## License

### Summary

The original `hillclimbracing` package included in this repository is licensed under the GNU General Public License v3.0 only.

Because this repository distributes that GPL-licensed code together with top-level additions and modifications as a single combined codebase, the repository should be treated as distributed under the GNU General Public License v3.0 as well.

That means:

- the vendored simulator code in [`hillclimbracing`](./hillclimbracing) remains under GPL-3.0-only
- the new top-level DQN code and repository-level additions are also distributed under GPL-3.0-only in this combined repository
- redistribution of this combined repository should preserve the GPL license text and the original attribution notices

### Full License Text

The full GNU GPL v3 license text is provided in:

- [`LICENSE`](./LICENSE)
- [`hillclimbracing/LICENSE`](./hillclimbracing/LICENSE)

The root `LICENSE` file is included so the combined repository clearly carries the governing license at the top level, while the original copy from the vendored project is preserved in place.

### Practical License Interpretation For This Repository

For this repository as currently organized:

- if you share the repository, share the source code and retain the GPL notice
- if you modify files and redistribute them, mark the modified version clearly
- if you reuse the combined codebase, keep the same GPL licensing obligations in mind

This README is an informational project summary and not legal advice. When in doubt, consult the full GPL text.

## Original Source References

- Upstream environment project: <https://github.com/alexzh3/hillclimbracing>
- Inspiration game project by Code Bullet: <https://github.com/Code-Bullet/Hill-Climb-Racing-AI>
