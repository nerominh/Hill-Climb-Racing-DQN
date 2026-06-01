# Repository-Grounded Project Plan: DQN on Hill Climb Racing
### COMP2050 AI Programming Project

---

## 1. What This Repository Already Gives You

This repository is **not yet a DQN project**. It is a **Gymnasium-compatible Hill Climb Racing environment package** that you can use as the simulation layer for your own reinforcement learning experiments.

Right now, the repo already includes:
- A custom Gymnasium environment: `hill_racing_env/HillRacing-v0`
- Physics and gameplay implemented with **Box2D**
- Rendering and human play with **Pygame**
- Procedural terrain generation with **Perlin noise**
- Multiple action-space and reward-function configurations
- Tests for environment creation, reset/step behavior, spaces, and custom noise
- Pretrained **PPO** baseline checkpoints for comparison

What it does **not** yet include:
- A DQN agent
- Replay buffer code
- Neural network code
- Training loop code
- Evaluation/plotting pipeline for your report
- Experiment runners for seeds / ablations / statistics

That means your project should be framed as:

> Use the existing HCR environment package as the simulator, then build and evaluate your own DQN-based agents on top of it.

---

## 2. Correct Environment Facts From This Codebase

Use these facts in your report and implementation.

### Environment ID
```python
"hill_racing_env/HillRacing-v0"
```

### Default configuration
- `action_space="discrete_3"`
- `reward_function="distance"`
- `reward_type="aggressive"`
- `max_steps=1200`
- `original_noise=False`

### Observation space
The environment returns a `Dict` observation with 4 keys:

| Key | Shape | Meaning |
|-----|-------|---------|
| `chassis_position` | `(2,)` | car x/y position |
| `chassis_angle` | `(1,)` | chassis angle in degrees |
| `wheels_speed` | `(2,)` | angular speed of the two wheels |
| `on_ground` | `(2,)` | whether each wheel is touching the ground |

This means a DQN implementation will likely need to **flatten** the dict observation into a 7-dimensional numeric vector before feeding it to a neural network.

### Available action spaces
- `"discrete_3"`: idle / gas / reverse
- `"continuous"`: 1D motor wheel speed in `[-13, 13]`

For a **DQN** project, start with:
```python
action_space="discrete_3"
```
Because vanilla DQN assumes a discrete action space.

### Available reward functions
- `"distance"`
- `"action"`
- `"wheel_speed"`
- `"airtime_distance"`
- `"airtime_wheel_speed"`

### Reward intensity modes
- `"aggressive"`
- `"soft"`

### Episode ending conditions
An episode ends when:
- the driver dies
- the agent gets stuck for too long (`max_steps` without enough progress)
- the score reaches the max target (`1000`)

Death and getting stuck both produce a `-100` reward.

---

## 3. Codebase Walkthrough

### 3.1 Top-level files

#### `HCR_DQN_Project_Plan.md`
Your planning document. This needed correction because the original version mixed good research ideas with a few assumptions that do not match the local codebase.

#### `quick_test.py`
A smoke test that imports the package, creates the environment in human-render mode, samples random actions, and resets when an episode ends.

---

### 3.2 Package setup

#### `hillclimbracing/pyproject.toml`
Defines the Python package and dependencies:
- core: `gymnasium`, `pygame`, `box2d-py`, `numpy`, `noise`
- optional training extra: `stable-baselines3`
- optional test extra: `pytest`

It also exposes a CLI script:
```bash
hill-climb-play
```
which launches the human-play mode.

#### `hillclimbracing/hill_racing_env/__init__.py`
Registers the Gymnasium environment:
```python
id="hill_racing_env/HillRacing-v0"
entry_point="hill_racing_env.envs:HillRacingEnv"
```
This is why simply importing `hill_racing_env` makes `gym.make(...)` work.

#### `hillclimbracing/hill_racing_env/envs/__init__.py`
Re-exports the main public classes such as `HillRacingEnv`, `Agent`, `Car`, `Ground`, `Person`, and `Wheel`.

---

### 3.3 Main environment logic

#### `hillclimbracing/hill_racing_env/envs/hill_racing.py`
This is the core of the project.

Responsibilities of `HillRacingEnv`:
- creates and owns the Box2D world
- defines Gymnasium action and observation spaces
- generates terrain and the playable agent on reset
- applies actions to the car
- advances physics in `step()`
- computes rewards
- decides termination and truncation
- renders the scene with Pygame

Important internal methods:
- `_generate_ground()` builds a valid terrain instance
- `_generate_agent()` spawns the car + wheels + ragdoll driver
- `_execute_action()` maps Gym actions to car motor controls
- `_get_reward()` contains the reward shaping logic for all reward modes
- `_get_obs()` builds the Gym observation dict
- `reset()` reconstructs the world and returns the first observation
- `step()` runs one simulation step and returns `(obs, reward, terminated, truncated, info)`

Important design note:
The environment itself currently contains a lot of the task logic directly inside one file, especially reward and termination handling. That is fine for using it as a simulator, but if you later want cleaner experimentation, you may eventually wrap it instead of editing it heavily.

---

### 3.4 Agent and physics objects

#### `hillclimbracing/hill_racing_env/envs/agent.py`
`Agent` is a gameplay wrapper around the physical car. It tracks:
- whether the run is dead
- score
- airtime counters
- camera-follow behavior through `panX`

It owns a `Car` instance and updates high-level gameplay state after physics steps.

#### `hillclimbracing/hill_racing_env/envs/car.py`
Builds the main vehicle:
- chassis body
- two wheels
- ragdoll driver
- Box2D joints connecting these parts

It also implements the control interface used by the environment:
- `motor_on(forward=True/False)`
- `motor_off()`
- `set_motor_wheel_speed(...)`

This is the layer that translates RL actions into physical wheel torque / speed.

#### `hillclimbracing/hill_racing_env/envs/wheels.py`
Defines each wheel as a Box2D body plus joints connecting it to the chassis. Also tracks whether the wheel is touching the ground.

#### `hillclimbracing/hill_racing_env/envs/person.py`
Defines the ragdoll driver using a `Head` and `Torso` connected with joints. If the head collides with the ground, the episode ends.

---

### 3.5 Terrain generation

#### `hillclimbracing/hill_racing_env/envs/ground.py`
Generates procedural terrain and installs it into the Box2D world.

Key ideas:
- terrain is a sequence of edge segments
- shape is produced using Perlin noise
- `original_noise=False` uses the `noise` library version and starts with a flat launch section
- `original_noise=True` uses the custom Processing-style Perlin implementation
- overly steep terrain is rejected and regenerated

This file is important for research because terrain generation affects difficulty, generalization, and reproducibility.

#### `hillclimbracing/hill_racing_env/envs/perlin.py`
Contains a custom Perlin-noise implementation ported from Processing. It is mostly used when `original_noise=True`.

---

### 3.6 Human play mode

#### `hillclimbracing/hill_racing_env/envs/hill_racing_human.py`
Standalone keyboard-playable version of the environment loop. It is useful for:
- understanding the physics and feel of the environment
- seeing how motor controls affect stability
- debugging reward intuition before training agents

---

### 3.7 Baseline assets

#### `hillclimbracing/hill_racing_env/envs/baseline_models/`
Contains pretrained PPO checkpoints from the thesis work that originally used this environment.

These are helpful as:
- sanity-check baselines
- comparison points in your report
- proof that the environment is trainable

But they are **not** part of your DQN implementation.

---

### 3.8 Tests

#### `hillclimbracing/tests/test_env.py`
Checks environment creation, reset/step contracts, deterministic reset with seeds, and reward output types across configurations.

#### `hillclimbracing/tests/test_spaces.py`
Checks that observations conform to the declared Gym spaces and that both action-space variants behave as expected.

#### `hillclimbracing/tests/test_perlin.py`
Checks the custom Perlin helper functions.

#### `hillclimbracing/tests/conftest.py`
Provides shared test fixtures, including headless setup for Pygame.

---

## 4. How Data Flows Through One Environment Step

This is the best mental model for the code:

1. Your RL agent selects an action.
2. `HillRacingEnv.step(action)` receives it.
3. `_execute_action()` converts that action into car motor behavior.
4. Box2D advances the world by one physics step.
5. `Agent.update()` refreshes score, death state, airtime, and camera position.
6. `HillRacingEnv` checks for:
   - death
   - stuck condition
   - score cap
7. If still alive, `_get_reward()` computes the reward.
8. `_get_obs()` packs the latest state into the observation dict.
9. `step()` returns observation, reward, done flags, and debug info.

That means the best extension point for your DQN project is **outside** this package: write a training script that repeatedly calls `reset()` and `step()` and learns from the returned transitions.

---

## 5. What Should Change in the Project Direction

The earlier plan had the right research spirit, but a few details needed correction:

### Keep
- DQN as the main algorithm
- Double DQN as a natural first variant
- Dueling DQN as a second architecture variant
- Experimental comparison across seeds
- Report-style evaluation with plots and discussion

### Adjust
- Use the **actual environment ID**: `hill_racing_env/HillRacing-v0`
- Use the **actual observation space**, not terrain-height vectors
- Recognize that the repo already contains reward/action variants from prior PPO work
- Treat this repo as the **environment dependency**, not as the full project implementation
- Start with **discrete actions only** for DQN

### De-prioritize for now
- Implementing continuous-control methods inside the first DQN milestone
- Modifying the environment internals unless needed
- Overloading the project with too many algorithm variants before you have a stable baseline

---

## 6. Recommended Project Plan

This is the plan I would recommend for this exact repository.

### Phase 0 - Verify and Understand the Environment
Goal: make sure the simulator is stable and you understand the signals.

- [ ] Install the package locally
- [ ] Run `quick_test.py`
- [ ] Run human mode with `hill-climb-play`
- [ ] Print one sample observation after `reset()`
- [ ] Confirm how reward changes for idle / gas / reverse under `distance` reward
- [ ] Decide your baseline environment config

**Recommended baseline config for the paper:**
```python
action_space="discrete_3"
reward_function="distance"
reward_type="soft"
original_noise=False
```

Why `soft` first: it is usually a friendlier starting point than aggressive penalties when bootstrapping a new value-based agent.

### Phase 1 - Build a Thin RL Project Around the Environment
Goal: create your own training code without changing the simulator.

Suggested new structure:

```text
project/
├── env_wrappers.py         # flatten dict observations, maybe normalize
├── replay_buffer.py
├── q_network.py
├── dqn_agent.py
├── train_dqn.py
├── evaluate_dqn.py
├── configs.py
├── results/
└── plots/
```

Tasks:
- [ ] Create an observation wrapper that converts the dict observation into a flat vector
- [ ] Add a small config file for hyperparameters
- [ ] Implement replay buffer
- [ ] Implement vanilla DQN network and target network
- [ ] Implement epsilon-greedy action selection
- [ ] Write the training loop
- [ ] Save checkpoints and logs

### Phase 2 - Get Vanilla DQN Working Reliably
Goal: produce one learning curve that clearly beats random behavior.

- [ ] Train on `discrete_3`
- [ ] Track episode return, score, episode length, epsilon
- [ ] Evaluate every N episodes with exploration off
- [ ] Save the best checkpoint
- [ ] Compare against a random-action baseline

Success criterion:
- the DQN agent consistently achieves a better mean score than random play

### Phase 3 - Add 2 Strong Variants
Goal: satisfy the originality and comparison parts of the assignment without exploding scope.

Recommended variants:
- [ ] **Double DQN**
- [ ] **Dueling DQN**

Optional third comparison axis:
- [ ] compare `reward_type="soft"` vs `reward_type="aggressive"`
- [ ] compare `reward_function="distance"` vs `reward_function="action"`

This is better than inventing many custom reward functions from scratch at the start, because the environment already exposes structured reward variants you can study.

### Phase 4 - Run Proper Experiments
Goal: produce report-quality results.

- [ ] Use 3 to 5 random seeds per configuration
- [ ] Store per-episode metrics to CSV
- [ ] Plot learning curves with mean ± std
- [ ] Report final evaluation score across seeds
- [ ] Include at least one qualitative rollout screenshot or short video

Minimum experiment set I recommend:
1. Random policy baseline
2. Vanilla DQN
3. Double DQN
4. Dueling DQN
5. Best DQN variant under a second reward setting

### Phase 5 - Write the Report
Goal: make the contribution easy to defend.

Core report message:
- this repository supplied the HCR environment
- you built the DQN training and evaluation pipeline yourself
- you compared meaningful DQN variants on the same simulator
- you analyzed how reward/action design affects value-based learning

---

## 7. Best Research Angle for This Repo

If you want a clean, defensible project story, use this:

**Main question:**
How well can DQN-style discrete-control agents learn to drive in a physics-based Hill Climb Racing environment, and which DQN variant is most effective under different reward settings?

That question fits the repository very well because:
- the environment already supports discrete actions
- the reward settings are easy to vary systematically
- PPO baselines suggest the task is learnable
- you can contribute a value-based perspective on top of prior PPO-oriented work

---

## 8. Concrete Implementation Advice

### Observation preprocessing
Because the environment returns a `Dict`, flatten it consistently in a fixed key order, for example:
```python
[chassis_x, chassis_y, angle, wheel_speed_0, wheel_speed_1, on_ground_0, on_ground_1]
```

### First baseline
Do **not** start with custom reward shaping plus Dueling plus Double all at once.
Get this working first:
- discrete actions
- default observation flattening
- vanilla DQN
- one reward setting

### Evaluation metric
Your best primary metric is probably:
- **final score** from `info["score"]`

Also log:
- episode return
- episode length
- termination type if you expose it

### Reproducibility
Use fixed seeds for:
- Python random
- NumPy
- PyTorch
- environment reset seed

### Comparison to PPO
Use the pretrained PPO models only as background context or an aspirational baseline, not as a direct fairness claim unless your setup matches closely.

---

## 9. Risks and How to Handle Them

### Risk 1: DQN learns slowly or unstably
Response:
- start with `reward_type="soft"`
- use target network updates carefully
- try observation normalization
- lower scope before adding more variants

### Risk 2: Dict observations complicate training
Response:
- flatten them with a simple wrapper
- keep the first network fully connected

### Risk 3: The environment is harder than CartPole-level tasks
Response:
- reduce ambition early
- prove learning on a simpler configuration first
- use variant comparisons only after baseline success

### Risk 4: Too much time goes into environment modifications
Response:
- avoid editing simulator internals unless debugging reveals a real blocker
- build most of your project in new files outside the package

---

## 10. Final Recommended Deliverables

By the end of the project, aim to have:

- [ ] A working DQN implementation for `hill_racing_env/HillRacing-v0`
- [ ] At least two DQN variants (preferably Double and Dueling)
- [ ] A reproducible experiment runner
- [ ] CSV logs and plots
- [ ] A short evaluation script / demo rollout
- [ ] A report clearly separating reused environment code from your own RL code

---

## 11. Short Version: What This Repo Is

If you want the one-sentence summary:

> This codebase is a custom Hill Climb Racing simulator packaged as a Gymnasium environment; your project is to build the DQN research code on top of it, not to treat the existing repo as the finished RL training system.
