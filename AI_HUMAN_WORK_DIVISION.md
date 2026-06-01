# AI and Human Work Division for `hcr_dqn`

This file records how the `hcr_dqn` work is divided between:

- the AI-assisted baseline implementation
- my own human implementation work

The goal is to keep the division clear enough for the project statement, report writing, and future verification.

---

## 1. Overall Approach

This project follows the approach:

- start from an AI-assisted implementation of the standard vanilla DQN pipeline
- keep the simulator package separate from the RL implementation
- implement my own understanding, edits, and variations on top of that baseline

This means the baseline scaffold can be AI-assisted, but the project must still contain meaningful human implementation work, especially for my own ideas and algorithm variations.

---

## 2. High-Level Division

### Reused external code

- `hillclimbracing/`
- `hill_racing_env/` inside the simulator package

These files are reused environment code and are not counted as my own RL implementation.

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

These provide the baseline training pipeline, evaluation entry points, logging, and utility structure.

### Human-owned implementation work

The following files are the main files where I contributed my own implementation work for the vanilla DQN:

- `hcr_dqn/q_network.py`
- `hcr_dqn/dqn_agent.py`

These are the most important human contribution areas in the current vanilla DQN baseline because they define:

- the Q-network architecture
- the DQN learning logic
- epsilon-greedy action selection
- target network updates
- checkpoint loading and saving behavior

---

## 3. File-by-File Ownership Record

| File | Primary role | Ownership record |
|---|---|---|
| `hcr_dqn/bootstrap.py` | Import/bootstrap helper | AI-assisted baseline scaffold |
| `hcr_dqn/configs.py` | Hyperparameters and output paths | AI-assisted baseline scaffold |
| `hcr_dqn/env_wrappers.py` | Observation flattening | AI-assisted baseline scaffold |
| `hcr_dqn/replay_buffer.py` | Experience replay storage | AI-assisted baseline scaffold |
| `hcr_dqn/q_network.py` | Q-network definition | Human contribution for vanilla DQN implementation |
| `hcr_dqn/dqn_agent.py` | Core DQN learning logic | Human contribution for vanilla DQN implementation |
| `hcr_dqn/evaluate_dqn.py` | Evaluation helper | AI-assisted baseline scaffold |
| `hcr_dqn/train_dqn.py` | Training loop | AI-assisted baseline scaffold |
| `hcr_dqn/run_evaluation.py` | CLI evaluation runner | AI-assisted baseline scaffold |
| `hcr_dqn/watch_checkpoint.py` | Visual checkpoint runner | AI-assisted baseline scaffold |

---

## 4. Development Workflow Record

This section explains the intended workflow order of the files and who primarily owned each stage.

### Stage 1: Environment reuse and project structure

Created first:

- reused simulator package under `hillclimbracing/`
- separate RL package under `hcr_dqn/`

Primary ownership:

- simulator: reused external code
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

### Stage 4: Experiment execution and inspection

Created after the baseline was working:

- `run_evaluation.py`
- `watch_checkpoint.py`

Primary ownership:

- AI-assisted support tooling

Reason:

- these files help inspect the trained model, track results, and visualize behavior, but they are not the main algorithm contribution

### Stage 5: Human-owned future variation work

Planned future human implementation areas:

- algorithm variations in `dqn_agent.py`
- architecture variations in `q_network.py`
- experiment-specific tuning in `configs.py`
- result interpretation in the trackers and report

Primary ownership:

- human implementation and experiment design

Reason:

- the professor’s requirement emphasizes that my own ideas and variations should be implemented by me

---

## 5. Practical Rule for Future Work

To keep the project statement honest and consistent, I will use this rule:

- AI can assist with baseline scaffolding, utilities, debugging support, and code explanation
- I should personally implement or substantially modify my own algorithmic variations
- if a file contains my own idea or experiment-specific change, that contribution should be recorded explicitly

---

## 6. Suggested Human-Owned Variation Areas

The best places for my own next implementation work are:

- `hcr_dqn/dqn_agent.py`
  - Double DQN target calculation
  - alternative exploration logic
  - different target update strategy

- `hcr_dqn/q_network.py`
  - larger network
  - different hidden layout
  - Dueling DQN architecture

- `hcr_dqn/configs.py`
  - experiment-specific hyperparameter schedules
  - run naming for comparison studies

These are strong human contribution areas because they affect the actual method rather than only the surrounding utilities.

---

## 7. Statement-Ready Summary

Short version for later report use:

> The project reused the provided Hill Climb Racing simulator environment and built a separate `hcr_dqn` reinforcement learning pipeline around it. The baseline vanilla DQN scaffold was AI-assisted, while the core vanilla DQN implementation work contributed by the student focused on `q_network.py` and `dqn_agent.py`, which define the Q-network and main DQN learning logic. Future algorithmic variations and experiment-specific modifications are intended to be implemented directly by the student.

---

## 8. Update Rule

Whenever I make a new variation myself, I should append:

- date
- file changed
- what I implemented
- whether it was a baseline edit or a new variation

This keeps the ownership trail clear for the final submission.
