- Bài toán là gì, vấn đề là gì
- Baseline Model: suwr dungj thuật toán nào
- Variation/technique mình dùng là gì? Nó có impat j đến cái baseline

# Experiment Results Tracker

This file is for recording all experiment runs, especially multi-seed runs used to study mean and variance.

Use this together with `PROJECT_PHASE_TRACKER.md`:
- `PROJECT_PHASE_TRACKER.md` explains the project, code structure, and progress
- `EXPERIMENT_RESULTS_TRACKER.md` records actual experiment results

---
### Training Log Columns explained

| Column | Meaning |
|---|---|
| `episode` | Which training episode this row corresponds to |
| `global_step` | Total environment steps taken so far. A step is a single cycle of interaction between AI agent and its environment. |
| `episode_return` | Total reward collected during one training episode. This is calculated as `reward_1 + reward_2 + ... + reward_T`. In this project, reward comes from the environment reward function, not directly from the game score. |
| `episode_score` | Final HCR game score at the end of that training episode. This comes from `info["score"]` and is based on forward progress, using the car's farthest horizontal distance reached from the spawn point. It is not just the episode length. |
| `episode_length` | Number of environment steps taken in that episode before it ended by termination or truncation. This is literally how many `env.step(...)` calls happened in that episode. |
| `epsilon` | Exploration rate after decay. The higher the exploration rate, the more the agent is taking random actions during training instead of following its learned best action. |
| `mean_loss` | Average training loss collected during that episode |
| `eval_mean_return` | Mean evaluation return across the evaluation episodes. This is the average of total rewards from greedy evaluation runs, so `mean(total_reward_per_eval_episode)`. Only populated on evaluation episodes. |
| `eval_mean_score` | Mean evaluation score across the evaluation episodes. This is the average of the final `info["score"]` values from the evaluation runs. Only populated on evaluation episodes. |
| `eval_mean_length` | Mean evaluation episode length across the evaluation episodes. This is the average number of steps survived in the evaluation runs. Only populated on evaluation episodes. |

### How to interpret the three most important training columns

- `episode_return` tells you how much reward the agent collected according to the RL reward function.
- `episode_score` tells you how much actual game progress the car achieved.
- `episode_length` tells you how long the episode lasted in steps.

These three numbers are related, but they are not the same thing.

Example interpretation:

- high `episode_length` + low `episode_score` can mean the agent survived for a long time but did not move forward efficiently
- high `episode_score` usually means the car reached a farther distance
- high `episode_return` means the learned policy matched the reward function well, but that still needs to be compared with score to judge whether the reward design is helping the real task

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

## 1. How To Use This File

For each algorithm or variation:
- choose one method name
- keep the hyperparameters fixed
- run multiple seeds
- record one summary row per seed
- compute the mean and variance across seeds

One run should mean:
- same method
- same hyperparameters
- different random seed
- different `run_name`

Example:
- `vanilla_dqn_seed42`
- `vanilla_dqn_seed123`
- `vanilla_dqn_seed999`

---

## 2. Recommended Main Metrics

Primary metric:
- `eval_mean_score`

Supporting metrics:
- `eval_mean_return`
- `eval_mean_length`

Optional extra metrics:
- best training score observed
- final episode score
- visible behavior notes

Why:
- score is the most intuitive game-performance metric
- return shows alignment with the reward signal
- length helps distinguish survival from effective progress

---

## 3. Seed Plan

Suggested baseline seeds:
- `42`
- `123`
- `999`
- `2026`
- `7`



---

## 4. Experiment Registry

Record each method here before running it.

| Experiment ID | Method Name | Main Change | Fixed Settings Confirmed? | Number of Seeds Planned | Status |
|---|---|---|---|---:|---|
| EXP-001 | vanilla_dqn | Baseline | Yes | 5 | Planned |
| EXP-002 | my_variation | Fill this in | Yes | 5 | Planned |

---

## 5. Per-Run Results
### EXP-001

| Experiment ID | Method | Seed | Run Name | Eval Mean Score | Eval Mean Return | Eval Mean Length | Notes  |
|---|---|---:|---|---:|---:|---:|---|
| EXP-001 | vanilla_dqn | 7 | `vanilla_dqn_seed7` | | | | |
| EXP-001 | vanilla_dqn | 42 | `vanilla_dqn_seed42` | | | | |
| EXP-001 | vanilla_dqn | 123 | `vanilla_dqn_seed123` | | | | |
| EXP-001 | vanilla_dqn | 999 | `vanilla_dqn_seed999` | | | | |
| EXP-001 | vanilla_dqn | 2026 | `vanilla_dqn_seed2026` | | | |At the beggining, the agent will always flip 2 anti-clockwise full rounds, barely keeping the head from the ground. But all three episode the agent was able to survive |


---

## 6. Aggregate Results By Method

After all seeds for one method are complete, summarize them here.

| Experiment ID | Method | Num Runs | Mean Score | Score Variance | Score Std Dev | Mean Return | Return Variance | Return Std Dev | Mean Length | Length Variance | Length Std Dev |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-001 | vanilla_dqn | 3 |  |  |  |  |  |  |  |  |  |
| EXP-002 | my_variation |  |  |  |  |  |  |  |  |  |  |

Interpretation:
- higher mean score is better on average
- lower variance or std dev means the method is more stable across seeds
- if score goes up but variance also goes way up, the method may be stronger but less reliable

---

## 7. Run Configuration Record
### EXP-001: vanilla_dqn

Method description:
- Vanilla DQN baseline

Config values:
- `run_name` pattern: vanilla_dqn_seed_(num_seed)
- `seed` list: 42, 123, 999
- `num_episodes`: 300
- `max_episode_steps`: 3000
- `reward_function`: distance
- `reward_type`: soft (why soft?)
- `action_space`: discrete_3: idle / gas / reverse
- `learning_rate`: 1e-3
- `gamma`: 0.99
- `batch_size`: 64
- `epsilon_start`: 1.0
- `epsilon_end`: 0.05
- `epsilon_decay`: 0.995
- `target_update_frequency`: 1_000
- `evaluation_frequency`: 25
- `evaluation_episodes`: 5

What changed relative to baseline:
- None

Comment:
- The TD loss did not monotonically decrease, which is expected for DQN with moving targets. Behavioral metrics improved on average, but evaluation remained high-variance, so the policy cannot yet be considered consistently robust.

### EXP-002: my_variation

Method description:
- Fill this in

Config values:
- `run_name` pattern:
- `seed` list:
- `num_episodes`:
- `max_episode_steps`:
- `reward_function`:
- `reward_type`:
- `action_space`:
- `learning_rate`:
- `gamma`:
- `batch_size`:
- `hidden_sizes`:
- `epsilon_start`:
- `epsilon_end`:
- `epsilon_decay`:
- `target_update_frequency`:
- `evaluation_frequency`:
- `evaluation_episodes`:

What changed relative to baseline:
- Fill this in

---

## 8. Behavior Notes

Use this section for qualitative observations from `watch_checkpoint.py`.

### vanilla_dqn

- Did the car move confidently?
- Did it hesitate?
- Did it survive long but score slowly?
- Common failure mode:
- Best-looking seed:
- Worst-looking seed:

### my_variation

- Did the variation look more stable?
- Did it look faster or more aggressive?
- Did it crash more often?
- Common failure mode:
- Best-looking seed:
- Worst-looking seed:

---

## 9. Final Comparison Summary

Write the final takeaway here after all experiments are done.

Questions to answer:
- Which method had the highest mean score?
- Which method had the lowest variance?
- Was your variation better than vanilla DQN?
- Was it better on average, more stable, or both?
- Did the visual behavior match the numeric results?

Final summary:
- 

---

