# Hill Climb Racing AI Project Phase Tracker

This file is both:

- a project log
- a step-by-step working guide

Use it to track what changed, why it changed, what theory is being used, how to understand the code, how to run each phase, and what findings to write down after experiments.

---

## 1. Project Overview

### Goal of this repository
- Use the Hill Climb Racing environment as the simulator
- Keep the simulator code separate from your own reinforcement learning code
- Build a DQN pipeline step by step
- Run experiments and record results clearly enough for a report

### Main code areas

| Folder or file | Role |
|---|---|
| `hillclimbracing/` | Reused simulator package |
| `hcr_dqn/` | Your RL code |
| `HCR_DQN_Project_Plan.md` | Main project plan |
| `PROJECT_PHASE_TRACKER.md` | This file, which tracks progress and explains what to do next |
| `quick_test.py` | Simple environment smoke test |
| `runs/` | Output folder created by training runs |

### Big design decision
- `hillclimbracing/` is the environment layer
- `hcr_dqn/` is the RL layer

That separation is important because it helps you explain in your report what was reused and what was your own implementation.

---

## 2. How To Use This File

For every phase:

1. Read the phase goal
2. Follow the step-by-step task list
3. Use the code walkthrough to understand the files involved
4. Run the commands listed for that phase
5. Fill in the findings table after you actually run experiments

If a run fails, still record it here. Failed runs are useful evidence.

---

## 3. Reinforcement Learning Theory Used So Far

This section explains the theory behind the Phase 1 implementation in plain language.

### 3.1 What problem DQN is solving
- The agent sees a state
- It chooses an action
- The environment returns a reward and a new state
- The agent tries to learn which actions lead to better long-term reward

### 3.2 Q-learning idea
- DQN is based on Q-learning
- A Q-value means: "How good is it to take action `a` in state `s`?"
- The agent wants to approximate `Q(s, a)`

### 3.3 Bellman target
- The target idea is:
- good actions now should lead to good future states later

The core update is:

```text
Q(s, a) <- r + gamma * max_a' Q(s', a')
```

Meaning:
- `r` is the reward you got now
- `gamma` controls how much future reward matters
- `max_a' Q(s', a')` is the best future value from the next state

### 3.4 Why we use a neural network
- The HCR state space is too large for a simple Q-table
- Instead of storing a number for every state-action pair, we train a neural network to predict Q-values
- Input: flattened environment observation
- Output: one Q-value for each discrete action

### 3.5 Why replay buffer exists
- Consecutive transitions are highly correlated
- Learning directly from them in order can make training unstable
- A replay buffer stores old transitions
- We later sample a random minibatch from memory

This helps:
- break short-term correlations
- reuse old experience
- improve sample efficiency

### 3.6 Why target network exists
- If the same network both predicts and defines the training target, the target keeps moving too fast
- So DQN uses:
- an online network for learning
- a target network for more stable targets

The target network is updated less often.

### 3.7 Why epsilon-greedy exists
- If the agent only chooses the current best action, it may never discover better behavior
- So with probability `epsilon`, it explores by acting randomly
- Otherwise it exploits the best predicted action

At the start:
- epsilon is high

Later:
- epsilon decays
- the agent becomes more greedy

### 3.8 Why we flatten the HCR observation
- The HCR environment returns a Gymnasium `Dict`
- A simple MLP-based DQN works more naturally with a single vector
- So we transform:

```text
{
  chassis_position,
  chassis_angle,
  wheels_speed,
  on_ground
}
```

into:

```text
[chassis_x, chassis_y, angle, back_wheel_speed, front_wheel_speed, back_on_ground, front_on_ground]
```

This gives the network a fixed numeric input size of `7`.

---

## 4. Repository Map

### Simulator side

| Path | Purpose |
|---|---|
| `hillclimbracing/hill_racing_env/` | Environment package you reuse |
| `hillclimbracing/tests/` | Tests for the simulator package |
| `hillclimbracing/pyproject.toml` | Packaging info for the simulator package |

### RL side

| Path | Purpose |
|---|---|
| `hcr_dqn/bootstrap.py` | Makes sure the simulator package can be imported |
| `hcr_dqn/configs.py` | Stores hyperparameters and output paths |
| `hcr_dqn/env_wrappers.py` | Flattens HCR observations for DQN |
| `hcr_dqn/replay_buffer.py` | Stores transitions and samples minibatches |
| `hcr_dqn/q_network.py` | Defines the Q-network |
| `hcr_dqn/dqn_agent.py` | Encapsulates DQN logic |
| `hcr_dqn/evaluate_dqn.py` | Runs evaluation episodes |
| `hcr_dqn/train_dqn.py` | Runs the training loop |
| `hcr_dqn/run_evaluation.py` | Loads a saved model and prints metrics |

---

# Phase 0

## Phase goal
- Understand the environment and verify that the simulator works

## Status
- Partially done

## What you should do step by step

1. Confirm the environment imports correctly
2. Run the random smoke test
3. Inspect one observation manually
4. Play the game in human mode if possible
5. Confirm the environment ID and action space

## Commands to run

From the repository root:

```powershell
.\hillclimbracing\.venv\Scripts\python.exe quick_test.py
```

If you want to inspect an observation:

```powershell
.\hillclimbracing\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'hillclimbracing'); import gymnasium as gym, hill_racing_env; env = gym.make('hill_racing_env/HillRacing-v0'); obs, info = env.reset(seed=42); print(obs); env.close()"
```

## What output to expect
- `quick_test.py` should open the environment if rendering works in your setup
- The observation print should show a dictionary with keys like:
- `chassis_position`
- `chassis_angle`
- `wheels_speed`
- `on_ground`

## Files changed in this phase

| File | Change type | Purpose | Notes |
|---|---|---|---|
| `HCR_DQN_Project_Plan.md` | Edited | Align plan with the actual repository | Corrected assumptions from the earlier plan |
| `quick_test.py` | Existing file | Quick environment smoke test | Used to confirm the simulator loop works |

## Findings

### Environment facts confirmed
- Environment ID: `hill_racing_env/HillRacing-v0`
- Observation type: Gymnasium `Dict`
- Recommended DQN starting action space: `discrete_3`
- RL code should stay separate from simulator code

### Notes to fill later
- What did the raw observation look like?
- What did random play look like?
- Did human play make the reward behavior easier to understand?

---

# Phase 1

## Phase goal
- Build a complete vanilla DQN scaffold outside the simulator package
- Make every part readable enough that you can extend it confidently

## Status
- Implemented

## Phase 1 step-by-step checklist

Follow these in order.

### Step 1. Understand the import setup
- Read `hcr_dqn/bootstrap.py`
- Confirm how the RL code finds the simulator package

What to understand:
- If `hill_racing_env` is already installed, the RL code uses the installed package
- If not, it falls back to the local `hillclimbracing/` folder

### Step 2. Understand the experiment configuration
- Read `hcr_dqn/configs.py`
- Find the fields controlling:
- environment settings
- training schedule
- DQN hyperparameters
- output paths

What to understand:
- `DQNConfig` is the single source of truth for the Phase 1 run
- Changing the config changes the whole run behavior

### Step 3. Understand the observation wrapper
- Read `hcr_dqn/env_wrappers.py`
- Focus on the order of the flattened features

What to understand:
- The environment returns a `Dict`
- The neural network expects a vector
- The wrapper defines the exact semantic meaning of each input feature

### Step 4. Understand the replay buffer
- Read `hcr_dqn/replay_buffer.py`
- Focus on `push()` and `sample()`

What to understand:
- `push()` stores transitions
- `sample()` returns a random minibatch for training
- This is where experience replay happens

### Step 5. Understand the Q-network
- Read `hcr_dqn/q_network.py`
- Focus on how input size and output size are used

What to understand:
- Input size is `7`
- Output size is the number of discrete actions, which is `3`
- The final output is one Q-value per action

### Step 6. Understand the DQN agent
- Read `hcr_dqn/dqn_agent.py`
- Focus on:
- `select_action()`
- `train_step()`
- `update_target_network()`
- `decay_epsilon()`

What to understand:
- `select_action()` is the exploration policy
- `train_step()` is the Bellman update
- `update_target_network()` is the stability mechanism
- `decay_epsilon()` changes exploration over time

### Step 7. Understand training orchestration
- Read `hcr_dqn/train_dqn.py`
- Follow the code in this order:
- `seed_everything()`
- `ensure_output_dirs()`
- `train()`

What to understand:
- the training loop creates the environment
- the training loop creates the agent and replay buffer
- each episode collects transitions
- learning only starts after enough experience exists
- evaluation runs periodically
- the best checkpoint is saved

### Step 8. Understand evaluation
- Read `hcr_dqn/evaluate_dqn.py`
- Read `hcr_dqn/run_evaluation.py`

What to understand:
- evaluation turns exploration off
- evaluation measures policy quality more cleanly than training episodes
- there are now two evaluation layers:
- validation evaluation for checkpoint selection during training
- final evaluation for report-quality held-out testing
- `run_evaluation.py` is the simplest script to use after training

---

## Phase 1 file-by-file explanation

### `hcr_dqn/bootstrap.py`

#### Purpose
- Keep RL code physically separate from simulator code
- Still allow the RL code to import `hill_racing_env`

#### Underlying idea
- Software architecture, not RL theory
- The file supports two workflows:
- installed package workflow
- local repository workflow

#### How to read it
- `REPO_ROOT` and `SIMULATOR_ROOT` define path locations
- `ensure_simulator_on_path()` first checks if `hill_racing_env` is already importable
- If not, it adds the local simulator folder to `sys.path`

#### Why it matters
- It keeps your RL project independent
- It avoids mixing your RL files into the environment package

---

### `hcr_dqn/configs.py`

#### Purpose
- Store all important settings in one place

#### Underlying theory
- Experiment control and reproducibility
- Good RL practice is not just about algorithms
- It is also about being able to repeat a run with the same settings

#### Main sections to understand

| Section | Meaning |
|---|---|
| Environment fields | Which simulator configuration to use |
| Training schedule | How long and how often learning happens |
| DQN hyperparameters | Gamma, learning rate, batch size, etc. |
| Exploration settings | Epsilon schedule |
| Output paths | Where logs and checkpoints go |

#### Key settings in this file
- `action_space="discrete_3"`
- `reward_function="distance"`
- `reward_type="soft"`
- `gamma=0.99`
- `batch_size=64`
- `hidden_sizes=(128, 128)`
- `run_name="phase1_baseline"`

#### How to use it
- Start by leaving defaults alone
- Later, change one setting at a time and record it in this tracker

---

### `hcr_dqn/env_wrappers.py`

#### Purpose
- Convert the environment observation into a vector suitable for an MLP

#### Underlying theory
- Representation preprocessing
- A learning algorithm is only as good as the data format it receives

#### Main idea
- Dict observations are flexible for environments
- Flat vectors are simpler for vanilla DQN

#### Key code sections

| Part | What it does |
|---|---|
| `keys_in_order` | Defines the exact flattening order |
| `feature_names` | Gives a human-readable meaning to each input |
| `observation_space` rebuild | Creates the new vector-shaped Gym space |
| `observation()` | Actually flattens one observation |
| `make_flat_env()` | Creates the environment and applies the wrapper |

#### Output of the wrapper

```text
[chassis_x, chassis_y, chassis_angle_deg, back_wheel_speed, front_wheel_speed, back_wheel_on_ground, front_wheel_on_ground]
```

#### How to understand this file
- This file is the translation layer between the simulator and the neural network

---

### `hcr_dqn/replay_buffer.py`

#### Purpose
- Store transitions and return random minibatches

#### Underlying theory
- Experience replay

#### Main concepts
- `Transition` is one memory item
- `ReplayBuffer` stores many memory items
- `push()` appends data
- `sample()` returns a random batch

#### Why this matters
- Without replay, the model would learn from highly correlated consecutive states
- With replay, the gradient updates are more stable

#### How to understand this file
- Think of it as the agent's notebook of past experiences

---

### `hcr_dqn/q_network.py`

#### Purpose
- Map the state vector to action values

#### Underlying theory
- Function approximation for Q-values

#### Main structure
- Input layer: takes the 7-feature state
- Hidden layers: learn useful internal patterns
- Output layer: one value for each action

#### Why the architecture is simple
- Phase 1 is about a clean baseline
- The environment state is already structured numeric data
- There is no need for convolutions here

#### How to understand this file
- The network answers the question:
- "How good is action 0, 1, or 2 in this state?"

---

### `hcr_dqn/dqn_agent.py`

#### Purpose
- Hold the learning logic in one place

#### Underlying theory used here
- epsilon-greedy exploration
- target network stabilization
- Bellman regression

#### Most important methods

| Method | Meaning |
|---|---|
| `select_action()` | Chooses random or greedy action |
| `train_step()` | Learns from one minibatch |
| `update_target_network()` | Copies online weights to target network |
| `decay_epsilon()` | Reduces exploration over time |
| `save()` | Writes checkpoint |
| `load()` | Restores checkpoint |

#### How to understand `train_step()`
- Convert arrays to tensors
- Predict current Q-values for chosen actions
- Compute target Q-values from the target network
- Compute loss
- Run backpropagation
- Update weights

This is the mathematical center of Phase 1.

---

### `hcr_dqn/evaluate_dqn.py`

#### Purpose
- Measure what the learned policy does when not exploring randomly

#### Underlying theory
- Distinguish learning-time behavior from test-time behavior

#### What it measures
- mean return
- mean score
- mean episode length

#### Why it matters
- Training episodes contain exploration noise
- Evaluation episodes should reflect the actual learned policy

---

### `hcr_dqn/train_dqn.py`

#### Purpose
- Run the full training pipeline

#### Underlying theory used here
- seeded reproducibility
- online interaction with the environment
- replay-based minibatch training
- periodic target syncing
- periodic greedy evaluation
- model checkpointing

#### Major sections of the file

| Section | What it does |
|---|---|
| `seed_everything()` | Makes runs more reproducible |
| `ensure_output_dirs()` | Creates output folders |
| `write_training_row()` | Writes metrics to CSV |
| `train()` | Executes the full training loop |

#### How the training loop works

1. Load config
2. Seed randomness
3. Create output folders
4. Create wrapped environment
5. Create replay buffer
6. Create DQN agent
7. Start episodes
8. For each step:
- pick action
- step environment
- store transition
- learn if enough data exists
- update target network when scheduled
9. Decay epsilon after each episode
10. Evaluate every `evaluation_frequency` episodes
11. Save the best checkpoint
12. Write logs to CSV

#### How to understand this file
- This file is the control room
- It does not define every algorithm detail itself
- Instead, it coordinates the other modules

---

### `hcr_dqn/run_evaluation.py`

#### Purpose
- Load the saved checkpoint and print evaluation metrics

#### Underlying theory
- Post-training policy assessment

#### What it does
- loads default config
- finds `best_model.pt`
- rebuilds environment shape information
- loads the trained agent
- runs evaluation
- prints summary metrics

#### How to understand this file
- It is the quickest way to answer:
- "How good is my saved model right now?"

---

## Files changed in this phase

| File | Change type | Purpose | Notes |
|---|---|---|---|
| `hcr_dqn/__init__.py` | Created | Package marker for RL code | Organizes imports |
| `hcr_dqn/bootstrap.py` | Created | Import helper between RL and simulator layers | Supports installed package or local source |
| `hcr_dqn/configs.py` | Created | Central config for Phase 1 | Keeps settings in one place |
| `hcr_dqn/env_wrappers.py` | Created | Observation flattening layer | Converts Dict state to 7D vector |
| `hcr_dqn/replay_buffer.py` | Created | Replay memory | Supports randomized minibatch learning |
| `hcr_dqn/q_network.py` | Created | MLP Q-network | Baseline function approximator |
| `hcr_dqn/dqn_agent.py` | Created | Main DQN logic | Holds policy, target net, optimizer, update rule |
| `hcr_dqn/evaluate_dqn.py` | Created | Evaluation helper | Measures greedy policy performance |
| `hcr_dqn/train_dqn.py` | Created | Main training script | Produces logs and checkpoints |
| `hcr_dqn/run_evaluation.py` | Created | Quick evaluation entry point | Prints saved-model metrics |
| `hillclimbracing/pyproject.toml` | Edited | Packaging file for simulator | Kept simulator package independent of RL layer |
| `hillclimbracing/hcr_dqn/` | Removed | Old in-package RL location | Deleted to keep architecture clean |

---

## Phase 1 verification already done

| Check | Result | Notes |
|---|---|---|
| Python syntax check for RL files | Passed | Files compile cleanly |
| Import test for wrapper utilities | Passed | Non-PyTorch parts load correctly |
| Wrapped environment shape check | Passed | Observation shape `(7,)`, action count `3` |
| Training run | Not run yet | PyTorch was not installed locally at the time |

---

## 5. How To Run Phase 1

This section is the exact runbook for using the current code.


## 5.3 Optional environment import check

Run:

```powershell
.\hillclimbracing\.venv\Scripts\python.exe -c "from hcr_dqn.env_wrappers import make_flat_env; from hcr_dqn.configs import DQNConfig; env = make_flat_env(DQNConfig()); print(env.observation_space.shape, env.action_space.n); env.close()"
```

### Expected output

```text
(7,) 3
```

Meaning:
- the flattened state has 7 features
- the action space has 3 discrete actions

## 5.4 Start training

From the repository root, run:

```powershell
.\hillclimbracing\.venv\Scripts\python.exe -m hcr_dqn.train_dqn
```


## 5.5 Evaluate the saved model

After training finishes:

```powershell
python -m hcr_dqn.run_evaluation
```

This now defaults to a final-style evaluation:
- 30 held-out episodes
- seeds `10000` to `10029`
- greedy policy with no exploration
- per-episode results plus summary statistics

If you want the smaller validation-style check instead, run:

```powershell
python -m hcr_dqn.run_evaluation --mode validation
```

### Expected output

It should print something like:

```text
Evaluation summary
Mode: final
Seeds used: 10000 to 10029 (30 episodes)
Mean return (average total reward): ...
Return standard deviation: ...
Mean score (average episode score, from the game): ...
Score standard deviation: ...
Mean length (average episode length in steps): ...
Length standard deviation: ...
```

If you see an error saying the checkpoint was not found, it usually means training never completed or the output path changed.

### How to interpret the evaluation summary

- `Mean return` is the average total reward collected across the evaluation episodes.
- `Mean score` is the average final game score reported by the environment.
- `Mean length` is the average number of environment steps the agent survived before the episode ended.
- the standard deviations tell you how stable or unstable the policy is across different terrains

These three numbers answer different questions:
- Return tells you how well the policy matches the reward signal.
- Score tells you how well the agent is doing in game terms.
- Length tells you whether the agent is surviving for a long time or dying early.

The seed choice also matters:
- validation mode uses a small held-out seed set for fast checkpoint comparison
- final mode uses a larger, different held-out seed set so report numbers are not based on the same terrains used during training selection

### Current evaluation notes from the first saved baseline

Recorded evaluation summary:

```text
Mean return (average total reward): 2816.031
Mean score (average episode score, from the game): 555.600
Mean length (average episode length in steps): 2374.600
```

Interpretation:
- The agent is surviving for a long time on average, because the mean episode length is high.
- The score is moderate rather than obviously strong, so the policy may be learning to stay alive without driving very efficiently.
- This is a classic "stable but conservative" pattern: the car is probably not instantly crashing, but it may hesitate, crawl, or fail to convert long survival into high score.
- The return is useful mainly for comparison against other runs, because reward is environment-defined and should not be treated as an absolute game-performance number.

Recommended next checks:
- Watch several greedy evaluation episodes visually to see whether the car is moving confidently or just surviving.
- Increase evaluation episodes from 5 to 20 or more before drawing strong conclusions from the averages.
- Compare this checkpoint against a random or earlier checkpoint to confirm that score is genuinely improving.
- Inspect per-episode metrics, not just means, to see whether performance is consistent or highly variable.
- If the behavior looks too cautious, tune one thing at a time such as reward design, exploration schedule, training length, or action-space choices.

### Watching the best checkpoint visually

The file `hcr_dqn/watch_checkpoint.py` exists to answer a simple question:

"After training finishes, what does the saved best model actually look like when it drives?"

That script does exactly this:
- it loads `runs/phase1_baseline/checkpoints/best_model.pt` by default
- it creates the environment with `render_mode="human"`
- it loads the trained DQN weights into the agent
- it runs greedy evaluation with `explore=False`, so the model uses what it learned instead of random exploration
- it opens the Pygame window so you can visually inspect the agent's behavior

In other words, the workflow is:
1. Train the model.
2. Save the best checkpoint during evaluation.
3. Run `watch_checkpoint.py`.
4. Watch whether the best saved policy actually drives well.

Use:

```powershell
.\hillclimbracing\.venv\Scripts\python.exe -m hcr_dqn.watch_checkpoint
```

Useful optional arguments:

```powershell
.\hillclimbracing\.venv\Scripts\python.exe -m hcr_dqn.watch_checkpoint --episodes 5 --step-delay 0.03
```

Why this matters:
- Numerical evaluation tells you how the policy scores on average.
- Visual evaluation tells you what kind of behavior produced those numbers.
- This is especially useful when the metrics suggest "stable but conservative" behavior and you want to verify whether the car is hesitating, crawling, or driving confidently.

---

## 6. Where To Find Outputs

The default run name is:

```text
phase1_baseline
```

So the default output folder is:

```text
runs/phase1_baseline/
```

### Output structure

| Path | What you should find there |
|---|---|
| `runs/phase1_baseline/checkpoints/` | Saved model files |
| `runs/phase1_baseline/logs/` | CSV training logs |
| `runs/phase1_baseline/plots/` | Plot files if you add plotting later |

### Most important files

| File | Meaning |
|---|---|
| `runs/phase1_baseline/checkpoints/best_model.pt` | Best saved checkpoint according to evaluation score |
| `runs/phase1_baseline/logs/training_metrics.csv` | Per-episode metrics written during training |

---

## 7. How To Read The Training Log

The main log file is:

```text
runs/phase1_baseline/logs/training_metrics.csv
```

### Columns explained

| Column | Meaning |
|---|---|
| `episode` | Which training episode this row corresponds to |
| `global_step` | Total environment steps taken so far |
| `episode_return` | Sum of rewards in that episode |
| `episode_score` | Final HCR score from the environment info dictionary |
| `episode_length` | Number of steps in the episode |
| `epsilon` | Exploration rate after decay |
| `mean_loss` | Average training loss collected during that episode |
| `eval_mean_return` | Evaluation return, only populated on evaluation episodes |
| `eval_mean_score` | Evaluation score, only populated on evaluation episodes |
| `eval_mean_length` | Evaluation episode length, only populated on evaluation episodes |

### What trends to look for

Good signs:
- `episode_score` starts increasing over time
- `eval_mean_score` becomes larger than random performance
- `epsilon` gradually decreases
- `mean_loss` is finite and not exploding

Possible warning signs:
- scores stay near zero for all episodes
- `mean_loss` becomes `nan`
- training ends but no checkpoint is saved
- evaluation stays flat for a long time

---

## 8. What You Should Record After A Run

Fill this in after each real experiment.

### Run information
- Date:
- Run name:
- Python interpreter used:
- PyTorch version:
- Seed:
- Number of episodes:
- Reward function:
- Reward type:
- Action space:

### Training behavior
- Did the run start successfully?: Yes
- Did the log file appear?: Yes
- Did the checkpoint file appear?: Yes
- Did evaluation run without errors?: yes

### Performance observations
- Best score observed: 684
- Mean evaluation score: 555.600
- Did it outperform random behavior?: Yes it should
- Did the agent learn anything visible?: From the visualization, the agent behavior is very good, survive verylong with 1 episide even exceed the max_episode_steps
- Did it mainly survive longer, or did it also convert survival into higher score?:
- Was the behavior confident, hesitant, or unstable when watched visually?: Confidently handle the situation

### Problems encountered
- Problem 1:
- Problem 2:
- Problem 3: 

### Possible next actions
- Tune hyperparameters
- Add console logging
- Add plotting
- Try longer training
- Move to Phase 2 experiments

---

# Phase 2

## Phase goal
- Get vanilla DQN training reliably and produce a baseline learning curve

## Status
- Not started yet

## Step-by-step plan

1. Install PyTorch if it is still missing
2. Run the default Phase 1 training once
3. Inspect `training_metrics.csv`
4. Run checkpoint evaluation
5. Decide whether the baseline is learning enough
6. If not, tune only one hyperparameter at a time

## Files likely to change

| File | Likely reason |
|---|---|
| `hcr_dqn/configs.py` | Hyperparameter tuning |
| `hcr_dqn/train_dqn.py` | Better logging or training-loop fixes |
| `hcr_dqn/dqn_agent.py` | Learning-rule adjustments |
| `runs/...` | New logs and checkpoints |

## Findings

### Baseline run details
- Date: 2026-05-26
- Run name: `phase1_baseline`
- Seed(s): `42` for the baseline config
- Episode count: 300 in the current default config
- Max episode steps: 3000

### Baseline results
- Mean training return: See `training_metrics.csv`
- Mean evaluation return: 2816.031
- Mean evaluation score: 555.600
- Mean evaluation length: 2374.600
- Best checkpoint: `runs/phase1_baseline/checkpoints/best_model.pt`

### Behavioral observations
- What the agent seemed to learn first: Likely basic survival and staying upright for longer stretches
- Common failure pattern: Not confirmed visually yet; current metrics suggest possible cautious or inefficient driving
- Whether score improved over random baseline: Not verified yet because no direct random-policy comparison was recorded
- Additional interpretation: The current checkpoint appears more stable than naive behavior, but the score is not high enough yet to conclude strong driving performance
- Additional interpretation: The combination of long episode length and only moderate score suggests the policy may be optimizing survival more than efficient forward progress

### Questions raised
- Question 1: Does the visual behavior match the "survives but drives conservatively" hypothesis?
- Question 2: Are the evaluation averages stable across 20 to 30 episodes, or are they inflated by a few good runs?
- Question 3: Which single hyperparameter change most improves score without sacrificing stability?

---

# Phase 3

## Phase goal
- Add comparison variants such as Double DQN and Dueling DQN

## Status
- Not started yet

## Step-by-step plan

1. Freeze a working vanilla DQN baseline
2. Create one variant at a time
3. Keep the rest of the pipeline unchanged
4. Compare each variant against the same baseline settings
5. Record multi-run evidence before drawing conclusions

## Files likely to change

| File | Likely reason |
|---|---|
| `hcr_dqn/dqn_agent.py` | Double DQN or other update-rule variants |
| `hcr_dqn/q_network.py` | Dueling network architecture |
| `hcr_dqn/configs.py` | Variant-specific settings |
| `hcr_dqn/train_dqn.py` | Logging extra comparison metadata |

## Findings

### Variant list
- Variant 1:
- Variant 2:
- Variant 3:

### Results summary
- Which variant trained fastest:
- Which variant reached the best score:
- Which variant was most stable:

### Interpretation
- Why one variant may have helped:
- Why one variant may have failed:
- What should be tested next:

---

# Phase 4

## Phase goal
- Run proper multi-seed experiments and prepare report-quality results

## Status
- Not started yet

## Step-by-step plan

1. Finalize experiment settings
2. Run each model with multiple seeds
3. Save logs for every run
4. Compute summary statistics
5. Produce plots and tables
6. Write down interpretations immediately

## Findings

### Experiment matrix

| Experiment name | Model | Reward setting | Seed(s) | Notes |
|---|---|---|---|---|
| `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

### Final quantitative results

| Model | Mean score | Std dev | Mean return | Notes |
|---|---|---|---|---|
| `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

### Plot checklist
- [ ] Learning curves saved
- [ ] Final comparison chart saved
- [ ] Seed summary table saved
- [ ] Best checkpoint names recorded

### Interpretation
- Most important outcome:
- Most surprising result:
- Biggest limitation:

---

# Phase 5

## Phase goal
- Write up the report and package the project clearly

## Status
- Not started yet

## Step-by-step plan

1. Freeze final code state
2. Organize tables and plots
3. Write the methods section from the real code
4. Write the results section from recorded findings
5. Clearly separate reused simulator code from your own RL code
6. Prepare final submission checklist

## Findings

### Report progress
- Introduction drafted:
- Methodology drafted:
- Results section drafted:
- Discussion drafted:
- Conclusion drafted:

### Submission checklist
- [ ] Code cleaned and organized
- [ ] Experiment results recorded
- [ ] Figures inserted into report
- [ ] References finalized
- [ ] Statement of reused code prepared

### Final project reflection
- What worked best:
- What I would improve next:
- What I learned from the project:

---

## 9. Ongoing File History

Use this as the quick global log when you do not want to search through the phase tables.

| Date | File | What changed | Why it changed |
|---|---|---|---|
| 2026-05-21 | `HCR_DQN_Project_Plan.md` | Plan revised | Align the project plan with the real codebase |
| 2026-05-25 | `hcr_dqn/` package | Created | Build the full Phase 1 RL scaffold |
| 2026-05-25 | `hillclimbracing/hcr_dqn/` | Removed | Keep RL fully separate from the simulator package |
| 2026-05-25 | `PROJECT_PHASE_TRACKER.md` | Created | Track all phases, file changes, purposes, and findings |
| 2026-05-25 | `PROJECT_PHASE_TRACKER.md` | Expanded | Turn it into a working manual with theory, code walkthroughs, and run instructions |

---

## 10. Notes For Future Me

- Keep simulator changes separate from RL changes whenever possible
- Log results immediately after each run so nothing is forgotten
- If a run fails, still record it here because failed runs are useful evidence
- If you change a hyperparameter, write down exactly what changed
- If you add a new algorithm, explain what theory changed and why
- Keep using plain language so later report writing is easier
