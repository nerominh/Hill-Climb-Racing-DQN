# AI and Human Work Division for `hcr_dqn`

This file records how the `hcr_dqn` work is divided between:

- reused external environment code
- the AI-assisted baseline scaffold
- my own human implementation work

The goal is to keep the division clear enough for the project statement, report writing, and future verification.

---

## 1. Overall Approach

This project follows the approach:

- reuse the open-source Hill Climb Racing Gymnasium simulator from GitHub
- keep the simulator package separate from the RL implementation
- start from an AI-assisted vanilla DQN training scaffold
- implement my own DQN understanding, algorithm edits, and reward-shaping variations mainly in the DQN core files

This means the baseline scaffold can be AI-assisted, but the project still contains meaningful human implementation work in the files that control the learned policy and the actual DQN update.

---

## 2. Environment Used From The GitHub HCR Gymnasium Project

### Reused external environment code

The folder below is reused external simulator code:

- `hillclimbracing/`

This folder contains the Hill Climb Racing Gymnasium environment pulled from the upstream GitHub project `alexzh3/hillclimbracing`. It is not counted as my own RL implementation.

Important environment facts:

| Item | Value used in this project |
|---|---|
| Upstream package folder | `hillclimbracing/` |
| Import/registration package | `hill_racing_env` |
| Gymnasium environment ID | `hill_racing_env/HillRacing-v0` |
| Environment class | `hill_racing_env.envs.hill_racing.HillRacingEnv` |
| Physics/rendering stack | Box2D and Pygame |
| Registration file | `hillclimbracing/hill_racing_env/__init__.py` |
| Main environment file | `hillclimbracing/hill_racing_env/envs/hill_racing.py` |

The upstream environment provides:

- the Box2D physics world
- the car, wheels, and rider simulation
- procedural hill terrain generation
- Gymnasium `reset(...)`, `step(...)`, and optional human rendering
- the original reward choices and action-space choices
- environment information such as score, position, stuck steps, airtime, and wheel contact

My DQN code does not rewrite this simulator. Instead, `hcr_dqn/env_wrappers.py` imports and creates the environment through:

```python
gym.make(config.env_id, render_mode=render_mode, **config.make_env_kwargs())
```

The current `DQNConfig` uses:

| Config field | Current project setting | Meaning |
|---|---|---|
| `env_id` | `hill_racing_env/HillRacing-v0` | The registered Gymnasium environment |
| `action_space` | `discrete_3` | Three actions for DQN |
| `reward_function` | `distance` | Environment reward is based mainly on forward distance progress |
| `reward_type` | `soft` | Smaller idle/reverse penalties than the aggressive setting |
| `max_steps` | `1200` | Environment stuck/truncation threshold |
| `original_noise` | `False` | Uses the non-original terrain noise setting from the environment code |

In the reused environment, the `discrete_3` action meanings are:

| Action index | Environment behavior |
|---|---|
| `0` | Idle / motor off |
| `1` | Gas / motor forward |
| `2` | Reverse / motor backward |

The original environment observation is a Gymnasium `Dict` with:

- `chassis_position`
- `chassis_angle`
- `wheels_speed`
- `on_ground`

Because the DQN network is a plain MLP, the project-specific wrapper `FlattenObservation` converts that dictionary into a 7-value flat vector:

| Flat index | Feature name |
|---|---|
| `0` | `chassis_x` |
| `1` | `chassis_y` |
| `2` | `chassis_angle_deg` |
| `3` | `back_wheel_speed` |
| `4` | `front_wheel_speed` |
| `5` | `back_wheel_on_ground` |
| `6` | `front_wheel_on_ground` |

This environment-wrapper layer is important because it connects the reused Gymnasium environment to my PyTorch DQN implementation while keeping the simulator itself separate.

---

## 3. High-Level Division

### Reused external code

The following files and folders are reused environment code:

- `hillclimbracing/`
- `hillclimbracing/hill_racing_env/`

These files provide the simulator. They are not counted as my own RL implementation.

### AI-assisted baseline scaffold

The following files are treated as part of the AI-assisted vanilla DQN scaffold:

- `hcr_dqn/bootstrap.py`
- `hcr_dqn/configs.py`
- `hcr_dqn/env_wrappers.py`
- `hcr_dqn/replay_buffer.py`
- `hcr_dqn/evaluate_dqn.py`
- `hcr_dqn/train_dqn.py`
- `hcr_dqn/run_evaluation.py`
- `hcr_dqn/watch_checkpoint.py`
- `hcr_dqn/generate_experiment_plots.py`

These provide the baseline training pipeline, environment setup, evaluation entry points, logging, checkpoint support, plotting, and utility structure.

`hcr_dqn/generate_experiment_plots.py` should be treated as an AI-assisted reporting tool. It helps generate comparison figures from the experiment CSV files, but it is not part of my own DQN algorithm implementation.

### Human-owned implementation work

The following files are still the main files where I contributed my own implementation work:

- `hcr_dqn/q_network.py`
- `hcr_dqn/dqn_agent.py`

These are the most important human contribution areas because they define:

- the Q-network architecture used by the agent
- the online and target Q-network setup
- epsilon-greedy action selection
- the Bellman update and loss computation
- the target-network update behavior
- checkpoint saving and loading behavior
- the DoubleDQN target calculation option
- the Momentum-Sensitive DQN reward-shaping variation
- the Anti-Stall Momentum DQN reward-shaping variation

The latest status is that the human-owned work is no longer only the vanilla DQN core. It now also includes the implemented algorithmic and reward-shaping changes in `dqn_agent.py`, while `q_network.py` remains the shared MLP architecture used by all current DQN variants.

---

## 4. File-by-File Ownership Record

| File | Primary role | Ownership record |
|---|---|---|
| `hillclimbracing/` | GitHub HCR Gymnasium simulator | Reused external environment code |
| `hcr_dqn/bootstrap.py` | Import/bootstrap helper | AI-assisted baseline scaffold |
| `hcr_dqn/configs.py` | Hyperparameters, run names, environment kwargs, and variation settings | AI-assisted scaffold with experiment-specific configuration edits |
| `hcr_dqn/env_wrappers.py` | Converts Dict observations into the 7-value DQN input vector | AI-assisted baseline scaffold |
| `hcr_dqn/replay_buffer.py` | Experience replay storage | AI-assisted baseline scaffold |
| `hcr_dqn/q_network.py` | Q-network definition | Human contribution for the DQN network implementation |
| `hcr_dqn/dqn_agent.py` | Core DQN, DoubleDQN target rule, and reward-shaping agents | Main human implementation contribution |
| `hcr_dqn/evaluate_dqn.py` | Evaluation helper | AI-assisted baseline scaffold |
| `hcr_dqn/train_dqn.py` | Training loop and agent-variant selection | AI-assisted scaffold that calls the human-owned agent logic |
| `hcr_dqn/run_evaluation.py` | CLI evaluation runner | AI-assisted baseline scaffold |
| `hcr_dqn/watch_checkpoint.py` | Visual checkpoint runner | AI-assisted baseline scaffold |
| `hcr_dqn/generate_experiment_plots.py` | Generates comparison plots from run CSV files | AI-assisted plotting and reporting tool |

---

## 5. Development Workflow Record

This section explains the intended workflow order of the files and who primarily owned each stage.

### Stage 1: Environment reuse and project structure

Created first:

- reused simulator package under `hillclimbracing/`
- separate RL package under `hcr_dqn/`

Primary ownership:

- simulator: reused external code from the GitHub HCR Gymnasium project
- package separation and scaffold setup: AI-assisted

Reason:

- the project needed a clean separation between reused environment code and the RL code that belongs to this project

### Stage 2: Vanilla DQN baseline scaffold

Created next:

- `bootstrap.py`
- `configs.py`
- `env_wrappers.py`
- `replay_buffer.py`
- `evaluate_dqn.py`
- `train_dqn.py`

Primary ownership:

- AI-assisted baseline scaffold

Reason:

- these files establish the standard training pipeline and make the environment usable for a vanilla DQN baseline

### Stage 3: Human implementation focus for vanilla DQN core

Main human contribution stage:

- `q_network.py`
- `dqn_agent.py`

Primary ownership:

- human implementation work

Reason:

- these files represent the central algorithm logic of the vanilla DQN baseline
- they are the most appropriate place to show understanding of how DQN actually works
- they directly control the learned policy behavior

### Stage 4: Human-owned algorithm and reward variations

Implemented after the vanilla baseline:

- DoubleDQN target calculation in `dqn_agent.py`
- Momentum-Sensitive DQN reward shaping in `dqn_agent.py`
- Anti-Stall Momentum DQN reward shaping in `dqn_agent.py`
- variation settings in `configs.py`

Primary ownership:

- human implementation and experiment design, especially in `dqn_agent.py`

Reason:

- these changes modify the actual learning method or the reward stored in replay memory
- they are not only surrounding utilities
- they directly test my own project ideas about overestimation, momentum, unstable motion, and stuck recovery

Current result status:

- `momentum_sensitive_dqn` is the strongest completed method by final mean score and return in the current tracker.
- `momentum_sensitive_double_dqn` is the most stable completed method by final-score standard deviation.
- `antistall_momentum_dqn` was implemented and tested, but it performed worse than the previous momentum-based methods and should be treated as a discontinued branch.

### Stage 5: Experiment execution and inspection

Created and used after the baseline was working:

- `run_evaluation.py`
- `watch_checkpoint.py`
- `generate_experiment_plots.py`
- generated logs under `runs/<run_name>/logs/`
- generated comparison plots under `plots/`

Primary ownership:

- AI-assisted support tooling plus human interpretation of results

Reason:

- these files help inspect the trained model, track results, generate report figures, and visualize behavior, but they are not the main algorithm contribution
- `generate_experiment_plots.py` is included here because it was provided as an AI-assisted plot-generation tool for turning training and evaluation CSV outputs into report-ready figures

---

## 6. Practical Rule for Future Work

To keep the project statement honest and consistent, I will use this rule:

- AI can assist with baseline scaffolding, utilities, debugging support, plotting, and code explanation.
- I should personally implement or substantially modify my own algorithmic variations.
- If a file contains my own idea or experiment-specific change, that contribution should be recorded explicitly.
- The reused `hillclimbracing/` simulator should be described as external environment code, not as my own implementation.
- My own implementation work should be described mainly through `q_network.py`, `dqn_agent.py`, and the experiment decisions that select and compare the variants.

---

## 7. Human-Owned Variation Areas

The main human-owned variation areas are:

- `hcr_dqn/dqn_agent.py`
  - vanilla DQN learning logic
  - DoubleDQN next-action selection and target-network evaluation
  - momentum-sensitive reward shaping
  - anti-stall reward shaping

- `hcr_dqn/q_network.py`
  - MLP architecture shared by the current DQN variants
  - input/output dimension validation
  - forward pass from flattened HCR observation to action Q-values

- `hcr_dqn/configs.py`
  - experiment-specific run naming
  - `agent_variant` selection
  - `td_target_mode` selection
  - reward-shaping hyperparameters

These are strong human contribution areas because they affect the actual method and experiment design rather than only the surrounding utilities.

---

## 8. Latest Update Record

Date: 2026-06-16

Latest ownership update:

- My own work is still centered in `hcr_dqn/q_network.py` and `hcr_dqn/dqn_agent.py`.
- `q_network.py` defines the shared feedforward Q-network used by the current DQN experiments.
- `dqn_agent.py` now includes the vanilla DQN agent, the DoubleDQN target option, the Momentum-Sensitive DQN reward-shaping agent, and the Anti-Stall Momentum DQN reward-shaping agent.
- The Anti-Stall Momentum DQN branch is implemented and tested, but the experiment tracker currently says it should not continue into `antistall_momentum_double_dqn` unless the reward idea is redesigned.
- The HCR environment itself remains reused code from the GitHub Gymnasium project under `hillclimbracing/`; my DQN work uses it through `gym.make("hill_racing_env/HillRacing-v0", ...)`.
- `hcr_dqn/generate_experiment_plots.py` is recorded as an AI-assisted plotting/reporting tool, separate from my human-owned DQN implementation in `q_network.py` and `dqn_agent.py`.

Current completed method story:

| Method | Implementation location | Status |
|---|---|---|
| `vanilla_dqn` | `q_network.py`, `dqn_agent.py` | Baseline completed |
| `momentum_sensitive_dqn` | `dqn_agent.py` reward shaping | Completed; strongest current mean score/return |
| `vanilla_double_dqn` | `dqn_agent.py` target calculation | Completed; improves over vanilla DQN |
| `momentum_sensitive_double_dqn` | `dqn_agent.py` target calculation plus reward shaping | Completed; most stable final score |
| `antistall_momentum_dqn` | `dqn_agent.py` anti-stall reward shaping | Completed but discontinued |
| `antistall_momentum_double_dqn` | Not implemented as a completed run | Cancelled for now |

---

## 9. Statement-Ready Summary

Short version for later report use:

> This project reused the open-source Hill Climb Racing Gymnasium simulator from the `hillclimbracing` GitHub project as the environment layer. The reused simulator is kept under `hillclimbracing/` and registered as `hill_racing_env/HillRacing-v0`. Around that environment, the project builds a separate `hcr_dqn` reinforcement learning package. The baseline training, logging, evaluation, plotting, and utility structure were AI-assisted, including the `generate_experiment_plots.py` tool used to produce report figures from CSV outputs. My main human implementation work is concentrated in `hcr_dqn/q_network.py` and `hcr_dqn/dqn_agent.py`. These files define the Q-network, vanilla DQN learning logic, DoubleDQN target option, and my reward-shaping variants. The latest implemented branch, `antistall_momentum_dqn`, was tested but did not improve the previous momentum-based methods, so it is recorded as a discontinued experiment rather than the new base method.

---

## 10. Update Rule

Whenever I make a new variation myself, I should append:

- date
- file changed
- what I implemented
- whether it was a baseline edit or a new variation
- whether the result became a recommended method or a discontinued branch

This keeps the ownership trail clear for the final submission.
