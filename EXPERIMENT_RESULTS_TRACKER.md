- Bài toán là gì, vấn đề là gì
- Baseline Model: suwr dungj thuật toán nào
- Variation/technique mình dùng là gì? Nó có impat j đến cái baseline

# Experiment Results Tracker

This file is for recording all experiment runs, especially multi-seed runs used to study mean and variance.

Use this together with `PROJECT_PHASE_TRACKER.md`:
- `PROJECT_PHASE_TRACKER.md` explains the project, code structure, and progress
- `EXPERIMENT_RESULTS_TRACKER.md` records actual experiment results

Division of responsibility:
- use `PROJECT_PHASE_TRACKER.md` for phase goals, theory, code walkthroughs, setup instructions, and evaluation protocol
- use `EXPERIMENT_RESULTS_TRACKER.md` for actual numbers, seed-by-seed results, behavior observations, failed runs, and final report tables

Recommended workflow:
1. Read the current phase and run instructions in `PROJECT_PHASE_TRACKER.md`
2. Run training or evaluation
3. Collect outputs from `runs/<run_name>/logs/`
4. Record the results here
5. Go back to `PROJECT_PHASE_TRACKER.md` only to update high-level conclusions or phase status

Important:
- if the two files ever disagree on a metric value, treat `EXPERIMENT_RESULTS_TRACKER.md` as the source of truth for experiment results
- use the evaluation protocol defined in `PROJECT_PHASE_TRACKER.md` when deciding what counts as validation versus final reporting

---
### Main run artifacts to read before filling this file

- `runs/<run_name>/logs/training_metrics.csv`
- `runs/<run_name>/logs/validation_metrics.csv`
- `runs/<run_name>/logs/evaluation_summaries.csv`
- `runs/<run_name>/logs/evaluation_episode_details.csv`
- `runs/<run_name>/checkpoints/best_model.pt`

The meanings of these files and the code that created them are explained in `PROJECT_PHASE_TRACKER.md`.

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

Important note:
- these `eval_mean_*` columns come from validation evaluation during training
- they are useful for checkpoint selection and learning-curve interpretation
- they should not replace the held-out final evaluation when you prepare report-quality results
- for final reporting, use `evaluation_summaries.csv` and `evaluation_episode_details.csv`

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

For final report tables, use the held-out final evaluation outputs from `run_evaluation.py`.

Primary metric:
- `mean_score` from `evaluation_summaries.csv`

Supporting metrics:
- `mean_return` from `evaluation_summaries.csv`
- `mean_length` from `evaluation_summaries.csv`
- `std_score`
- `std_return`
- `std_length`

Optional extra metrics:
- best training score observed
- final episode score
- visible behavior notes

Why:
- score is the most intuitive game-performance metric
- return shows alignment with the reward signal
- length helps distinguish survival from effective progress

For final report tables:
- prefer the held-out final evaluation metrics saved by `run_evaluation.py`
- use the training-log `eval_mean_*` values mainly for training diagnostics and checkpoint tracking

---

## Research Flow And Next Reward Fix

Use this section to keep the experiment story consistent in the report.

Planned progression:
1. Start with `vanilla_dqn` as the baseline.
2. Try `momentum_sensitive_dqn` as a reward-shaping variation on top of the same core DQN update.
3. After comparing those two, note the deeper mathematical bottleneck in standard DQN itself.
4. Try plain `double_dqn` to address that bottleneck directly.
5. Combine the DoubleDQN target rule with the custom momentum-sensitive reward shaping to produce `momentum_sensitive_double_dqn`.
6. Because `momentum_sensitive_dqn` is still the best completed method by mean score and return, apply the new anti-stall reward modification there first as `antistall_momentum_dqn`.
7. If `antistall_momentum_dqn` improves stuck recovery, then test the same reward idea with DoubleDQN as `antistall_momentum_double_dqn`.

Reason for adding `double_dqn`:
- Both `vanilla_dqn` and `momentum_sensitive_dqn` still rely on the standard DQN target.
- With function approximation, Q-values can be unevenly overestimated or underestimated during learning.
- In standard DQN, the max operator uses the same estimates to both select the next action and evaluate that action's value.
- That coupling can reinforce overestimated actions and make learning less stable.
- `double_dqn` separates those two jobs:
- the online network selects the action
- the target network evaluates the selected action
- This made `double_dqn` the clean next experiment at that stage because it tested whether the instability came partly from the DQN target rule itself, not only from reward design.

Reporting takeaway:
- `momentum_sensitive_dqn` answers: "What happens if I keep DQN but change the learning reward?"
- `double_dqn` answers: "What happens if I keep the task mostly the same but fix the overestimation bias in the update rule?"
- `momentum_sensitive_double_dqn` answers: "What happens if I combine the algorithmic fix with my custom reward design?"
- `antistall_momentum_dqn` answers: "What happens if I keep the best reward-shaped DQN base and directly punish the stuck/no-recovery behavior?"
- `antistall_momentum_double_dqn` should only be used later to answer: "Does the same anti-stall reward also help when the target rule is DoubleDQN?"

---

## Plot Guide

The plots are generated by:

```powershell
python -m hcr_dqn.generate_experiment_plots
```

They are saved in the root `plots/` folder, parallel to `runs/`.

Use this section when writing the Results and Discussion parts of the report.

### Plot 1: Learning curves

File:
- `plots/plot_1_learning_curves_mean_episode_score.png`

Data source:
- `runs/<run_name>/logs/training_metrics.csv`
- metric used: `episode_score`

What the plot shows:
- one line per variant, such as `vanilla_dqn` and `momentum_sensitive_dqn`
- each line is the mean episode score across the available seeds
- the shaded band is the across-seed standard deviation
- the curve is smoothed with a trailing moving-average window of 20 episodes by default

How to read it:
- higher curves mean better average training-time game progress
- a steeper early rise means the method learns faster
- a higher late-stage plateau means the method finishes training at a stronger level
- a narrow shaded band means the method is more consistent across seeds
- a wide shaded band means the method is more sensitive to seed choice

What this plot is good for:
- comparing learning speed
- comparing training stability
- showing whether one method consistently overtakes another during training

Important caution:
- this is a training curve, not the final held-out test result
- use it together with the final evaluation plots, not instead of them

Current comparative interpretation for `vanilla_dqn`, `momentum_sensitive_dqn`, `vanilla_double_dqn`, and `momentum_sensitive_double_dqn`:

1. Early-stage sample efficiency
- Observation: the reward-shaped methods remain stronger than the unshaped baseline. `momentum_sensitive_dqn` has the best final held-out score, and `momentum_sensitive_double_dqn` is close behind.
- Interpretation: the shaped reward helps the agent find forward-driving behavior, but the visual behavior still matters because both momentum-based methods can get stuck and fail to recover.

2. Late-stage training trend
- Observation: after retraining with the updated validation setup, `vanilla_double_dqn` beats `vanilla_dqn`, and `momentum_sensitive_double_dqn` beats plain `vanilla_double_dqn`.
- Interpretation: the current comparison suggests four different behaviors:
- `vanilla_dqn` is the baseline reference and remains usable, but it now has the lowest final mean score.
- `vanilla_double_dqn` benefits from the target-rule change and improves over vanilla DQN.
- `momentum_sensitive_dqn` is the strongest completed method by mean score and return.
- `momentum_sensitive_double_dqn` is the most stable by final-score standard deviation and has the longest mean episode length, but its long survival can include stuck behavior with little useful recovery.

3. Inter-seed variance
- Observation: `momentum_sensitive_double_dqn` has the lowest across-seed final-score standard deviation, while `momentum_sensitive_dqn` has the highest final mean score.
- Interpretation: combining DoubleDQN with the momentum-sensitive reward improves score consistency, but it does not fully solve the stuck-policy failure mode.

4. What this training curve cannot prove
- Observation: this plot is still based on training episode score, not the held-out final evaluation.
- Interpretation: a method can look strong here and still lose on held-out evaluation, or vice versa. Use this plot together with the final bar charts and box plot before making claims.

Important reporting note:
- Use this interpretation when discussing `plots/plot_1_learning_curves_mean_episode_score.png`.
- Do not treat this section as a replacement for final reporting.
- Final performance claims should still be grounded in the held-out results recorded from `evaluation_summaries.csv` and `evaluation_episode_details.csv`.

### Plot 6: Validation learning curves

File:
- `plots/plot_6_validation_learning_curves_mean_score.png`

Data source:
- `runs/<run_name>/logs/validation_metrics.csv`
- legacy fallback for older runs: validation checkpoints reconstructed from `training_metrics.csv`
- metric used: held-out validation `mean_score` during training

What the plot shows:
- one line per variant
- each point is a greedy validation score measured on the fixed validation seed block during training
- the shaded band is the across-seed standard deviation at that checkpoint

Why this plot matters more than raw training score for checkpoint selection:
- it removes exploration noise
- it uses a held-out seed block instead of the same trajectory that generated the replay data
- it is a better view of which checkpoints are actually improving on repeatable terrains

Reporting note:
- this is the preferred plot when discussing checkpoint-selection quality and validation stability
- `plot_1` is still useful for training dynamics, but `plot_6` is the stronger validation-based plot

### Plot 2: Final score comparison bar chart

File:
- `plots/plot_2_final_score_comparison_bar_chart.png`

Data source:
- `runs/<run_name>/logs/evaluation_summaries.csv`
- metric used: `mean_score` from the final held-out evaluation

What the plot shows:
- one bar per variant
- bar height is the mean final score across the seed runs
- error bars are the standard deviation across seed runs

How to read it:
- taller bars mean better final game performance on held-out terrains
- shorter error bars mean the method is more reliable across seeds
- if two bars are close but one has much smaller error bars, the more stable method may be preferable

What this plot is good for:
- the cleanest side-by-side final comparison
- a fast visual answer to the question, "Which method performed best overall?"

Current interpretation:
- `momentum_sensitive_dqn` currently has the highest final mean score at about `529.33`.
- `momentum_sensitive_double_dqn` is very close at about `516.55`.
- `vanilla_double_dqn` is next at about `482.42`.
- `vanilla_dqn` is lower at about `410.03`.
- The combined `momentum_sensitive_double_dqn` result is important: it improves over plain DoubleDQN and is more consistent across seeds, but it still does not beat `momentum_sensitive_dqn` on mean score.
- `momentum_sensitive_double_dqn` has the smallest across-seed final-score standard deviation at about `58.04`.

### Plot 3: Final return comparison bar chart

File:
- `plots/final_return_comparison_bar_chart.png`

Data source:
- `runs/<run_name>/logs/evaluation_summaries.csv`
- metric used: `mean_return`

What the plot shows:
- one bar per variant
- bar height is the mean final return across the seed runs
- error bars are the standard deviation across seed runs

How to read it:
- higher bars mean better performance according to the learning reward
- this helps you judge whether a reward-shaped method is doing better under its own reward signal
- compare this plot with the final score bar chart to see whether better reward return also translated into better game performance

What this plot is good for:
- checking reward alignment
- supporting discussion about whether reward shaping improved the intended objective or only the internal reward

Important caution:
- return is an RL metric, not the same as the game score
- a method can improve return without improving score by the same amount

Current interpretation:
- `momentum_sensitive_dqn` has the highest final mean return at about `2322.02`.
- `momentum_sensitive_double_dqn` is next at about `2017.76`.
- `vanilla_double_dqn` follows at about `1786.62`.
- `vanilla_dqn` is lower at about `1670.44`.
- The refreshed result suggests the shaped reward is no longer only improving score; it also improves final held-out return under the current evaluation.
- The combined method has lower return than `momentum_sensitive_dqn`, which suggests that the DoubleDQN target plus the current shaping does not automatically improve reward optimization.
- `vanilla_double_dqn` improves over vanilla DQN on mean return, but its return standard deviation is still large, so the return improvement is not equally reliable for every seed.

### Plot 4: Final episode length comparison bar chart

File:
- `plots/final_episode_length_comparison_bar_chart.png`

Data source:
- `runs/<run_name>/logs/evaluation_summaries.csv`
- metric used: `mean_length`

What the plot shows:
- one bar per variant
- bar height is the mean final episode length across seeds
- error bars are the standard deviation across seeds

How to read it:
- taller bars mean the agent survives for more steps on average
- compare this plot with the final score plot
- if episode length is high but score is only moderate, the agent may be surviving without driving efficiently
- if both length and score are high, the agent is likely surviving and converting survival into progress

What this plot is good for:
- separating "stays alive" from "actually drives well"
- supporting discussion about conservative behavior versus efficient forward motion

Current interpretation:
- `momentum_sensitive_double_dqn` has the highest mean episode length at about `2380.01`.
- `momentum_sensitive_dqn` is next at about `2248.01`.
- `vanilla_double_dqn` is close at about `2202.62`.
- `vanilla_dqn` is lower at about `1912.07`.
- This suggests the combined method is especially good at staying alive, but survival is not the same as solving the terrain.
- The visual stuck behavior is important because a high mean episode length can hide cases where the car survives but stops making useful progress.

### Plot 5: Seed variance box plot for final score

File:
- `plots/plot_5_seed_variance_box_plot_final_score.png`

Data source:
- `runs/<run_name>/logs/evaluation_summaries.csv`
- metric used: per-seed `mean_score`

What the plot shows:
- one box per variant
- each box summarizes the distribution of final scores across seeds
- the scattered points show the actual seed-level values

How to read it:
- the median line inside the box shows the middle result
- the box height shows the interquartile range, which is the middle spread of results
- shorter boxes usually mean more stable seed-to-seed behavior
- taller boxes or more spread-out points mean more variance across seeds

What this plot is good for:
- showing reliability
- supporting claims about robustness, not just average performance

Why this plot matters in this project:
- reinforcement learning can be very seed-sensitive
- this plot helps justify whether a gain is consistently repeatable or only driven by one or two lucky runs

Current interpretation:
- `momentum_sensitive_dqn` has the highest final-score mean, while `momentum_sensitive_double_dqn` has the lowest across-seed score standard deviation.
- `momentum_sensitive_double_dqn` is numerically consistent, but watched behavior still shows instability and stuck cases, especially around the seed-2026 behavior notes.
- `vanilla_double_dqn` improves over `vanilla_dqn` on average, but still has weaker seeds such as `seed42` and `seed2026` compared with stronger seeds such as `seed7` and `seed999`.
- `vanilla_dqn` now has the lowest mean final score and includes one especially weak seed, `seed123`.
- This supports the updated conclusion that reward shaping is still the strongest direction, but the next reward design should directly penalize getting stuck instead of only rewarding smooth momentum.

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
| EXP-001 | vanilla_dqn | Baseline | Yes | 5 | Completed |
| EXP-002 | momentum_sensitive_dqn | Reward shaping for smoother forward motion and less unstable flipping/balancing | Yes | 5 | Completed |
| EXP-003 | vanilla_double_dqn | Replace the standard DQN max target with decoupled online-action selection and target-network evaluation | Yes | 5 | Completed |
| EXP-004 | momentum_sensitive_double_dqn | DoubleDQN target rule plus the custom Momentum-Sensitive reward function | Yes | 5 | Completed |
| EXP-005 | antistall_momentum_dqn | Next reward modification applied to the best current base method: strongly penalize being stuck while not using gas or trying to recover | Code implemented | 5 | Recommended next |
| EXP-006 | antistall_momentum_double_dqn | Later follow-up: apply the same anti-stall reward idea on top of Momentum-Sensitive DoubleDQN if EXP-005 improves stuck recovery | Design pending | 5 | Future follow-up |

---

## 5. Per-Run Results
### EXP-001

| Experiment ID | Method | Seed | Run Name | Final Mean Score | Final Mean Return | Final Mean Length | Notes  |
|---|---|---:|---|---:|---:|---:|---|
| EXP-001 | vanilla_dqn | 7 | `vanilla_dqn_seed7` | 555.6000 | 2594.0029 | 2146.7333 | Strongest vanilla seed by final score |
| EXP-001 | vanilla_dqn | 42 | `vanilla_dqn_seed42` | 438.6333 | 2131.1302 | 2804.6333 | Very long survival, but not always converted into proportional score |
| EXP-001 | vanilla_dqn | 123 | `vanilla_dqn_seed123` | 226.4333 | 849.8268 | 762.6000 | Weakest vanilla seed, with short average survival |
| EXP-001 | vanilla_dqn | 999 | `vanilla_dqn_seed999` | 355.3000 | 718.3840 | 1579.5000 | High per-episode variance during final evaluation |
| EXP-001 | vanilla_dqn | 2026 | `vanilla_dqn_seed2026` | 474.1667 | 2058.8636 | 2266.8667 | Visual note: at the beginning, the agent often flips two anti-clockwise full rounds while barely keeping the head from the ground, but it can still survive |

### EXP-002

| Experiment ID | Method | Seed | Run Name | Final Mean Score | Final Mean Return | Final Mean Length | Notes |
|---|---|---:|---|---:|---:|---:|---|
| EXP-002 | momentum_sensitive_dqn | 7 | `momentum_sensitive_dqn_seed7` | 583.8333 | 2703.9485 | 2225.3667 | Strong final score and return |
| EXP-002 | momentum_sensitive_dqn | 42 | `momentum_sensitive_dqn_seed42` | 507.7333 | 1479.3446 | 2638.0333 | Visually a little more stable, but still aggressive on some terrain |
| EXP-002 | momentum_sensitive_dqn | 123 | `momentum_sensitive_dqn_seed123` | 431.1333 | 1781.5653 | 1489.9667 | Weakest momentum-sensitive seed by final score |
| EXP-002 | momentum_sensitive_dqn | 999 | `momentum_sensitive_dqn_seed999` | 586.7000 | 2847.9477 | 2446.1667 | Strongest momentum-sensitive seed by final score |
| EXP-002 | momentum_sensitive_dqn | 2026 | `momentum_sensitive_dqn_seed2026` | 537.2333 | 2797.3048 | 2440.5000 | Strong numeric result, but visually the agent can still flip backwards or get stuck on elevated terrain without recovering |

### EXP-003

| Experiment ID | Method | Seed | Run Name | Final Mean Score | Final Mean Return | Final Mean Length | Notes |
|---|---|---:|---|---:|---:|---:|---|
| EXP-003 | vanilla_double_dqn | 7 | `vanilla_double_dqn_seed7` | 579.3333 | 2594.0988 | 2350.2333 | Strong DoubleDQN seed with good score and survival |
| EXP-003 | vanilla_double_dqn | 42 | `vanilla_double_dqn_seed42` | 363.1667 | 575.2698 | 1880.4333 | Weaker DoubleDQN seed; survival is moderate but reward return is low |
| EXP-003 | vanilla_double_dqn | 123 | `vanilla_double_dqn_seed123` | 501.4000 | 2330.1108 | 2007.4333 | Solid seed; shows DoubleDQN can learn a high-scoring policy in this setup |
| EXP-003 | vanilla_double_dqn | 999 | `vanilla_double_dqn_seed999` | 616.3000 | 3137.3704 | 2641.0000 | Strongest DoubleDQN seed by final score |
| EXP-003 | vanilla_double_dqn | 2026 | `vanilla_double_dqn_seed2026` | 351.9000 | 296.2373 | 2134.0000 | Weak return despite decent survival, which pulls down the method mean |

### EXP-004

| Experiment ID | Method | Seed | Run Name | Final Mean Score | Final Mean Return | Final Mean Length | Notes |
|---|---|---:|---|---:|---:|---:|---|
| EXP-004 | momentum_sensitive_double_dqn | 7 | `momentum_sensitive_double_dqn_seed7` | 494.7667 | 2014.5159 | 2055.5000 | Solid score, but not clearly better than the best single-change methods |
| EXP-004 | momentum_sensitive_double_dqn | 42 | `momentum_sensitive_double_dqn_seed42` | 483.1333 | 1626.1511 | 2782.9333 | Long survival, but lower return suggests inefficient or stuck behavior in some episodes |
| EXP-004 | momentum_sensitive_double_dqn | 123 | `momentum_sensitive_double_dqn_seed123` | 537.3000 | 2290.5845 | 1889.3667 | Good score despite shorter average survival than several other combined-method seeds |
| EXP-004 | momentum_sensitive_double_dqn | 999 | `momentum_sensitive_double_dqn_seed999` | 460.1000 | 760.8665 | 2236.8000 | Weakest combined-method seed by score and return; likely survives without enough useful progress |
| EXP-004 | momentum_sensitive_double_dqn | 2026 | `momentum_sensitive_double_dqn_seed2026` | 607.4667 | 3396.6630 | 2935.4667 | Strongest numeric seed, but visual notes still show instability and stuck behavior with little improvement |


---

## 6. Aggregate Results By Method

After all seeds for one method are complete, summarize them here.

| Experiment ID | Method | Num Runs | Final Mean Score | Score Variance | Score Std Dev | Final Mean Return | Return Variance | Return Std Dev | Final Mean Length | Length Variance | Length Std Dev |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-001 | vanilla_dqn | 5 | 410.0267 | 15706.3528 | 125.3250 | 1670.4415 | 698973.3587 | 836.0463 | 1912.0667 | 602375.2232 | 776.1284 |
| EXP-002 | momentum_sensitive_dqn | 5 | 529.3266 | 4108.3495 | 64.0964 | 2322.0222 | 412639.5196 | 642.3702 | 2248.0067 | 200894.7540 | 448.2128 |
| EXP-003 | vanilla_double_dqn | 5 | 482.4200 | 14733.2756 | 121.3807 | 1786.6174 | 1615135.2012 | 1270.8797 | 2202.6200 | 90144.3832 | 300.2405 |
| EXP-004 | momentum_sensitive_double_dqn | 5 | 516.5533 | 3368.5498 | 58.0392 | 2017.7562 | 927239.0041 | 962.9325 | 2380.0133 | 209356.5058 | 457.5549 |
| EXP-005 | antistall_momentum_dqn | 5 |  |  |  |  |  |  |  |  |  |
| EXP-006 | antistall_momentum_double_dqn | 5 |  |  |  |  |  |  |  |  |  |

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
- `seed` list: 42, 123, 999, 2026, 7
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
- `evaluation_episodes`: 10
- final held-out evaluation: 30 episodes on seeds `10000` to `10029` by default

What changed relative to baseline:
- None

Comment:
- The TD loss did not monotonically decrease, which is expected for DQN with moving targets. Behavioral metrics improved on average, but evaluation remained high-variance, so the policy cannot yet be considered consistently robust.
- Other agents may have the action of balancing on the back wheels quite frequently. 

Evaluation comments to reuse in the report:
- `vanilla_dqn` learns more slowly in the early episodes, but its training curve is more stable in the long run and it achieves the stronger late-stage performance.
- The learning trajectory is noisy, yet it is closer to monotonic improvement than the momentum-sensitive variant.
- The large late-stage variance suggests that the baseline is still seed-sensitive, but the mean trend indicates that it is capable of converging to stronger policies by the end of training.
- In comparative terms, the baseline appears weaker in early sample efficiency but stronger in asymptotic performance and late-stage reliability.

### EXP-002: Momentum-Sensitive DQN

Method description:
- Momentum-Sensitive DQN keeps the vanilla DQN algorithm, replay buffer, target network, epsilon-greedy exploration, and Q-network architecture the same, but changes the reward used for learning.
- The main idea is to encourage stable forward driving instead of unstable survival behavior.
- This variation is trying to reduce behaviors observed in the vanilla baseline such as repeated flipping at the start of an episode, balancing too often on the back wheels, oscillating between opposite actions, staying alive without making smooth progress, and spending too long stalled with little forward momentum.
- To do this, the learning reward gives a small bonus when the car keeps forward motion consistently, and small penalties when the car becomes unstable or unproductive.

Config values:
- `run_name` pattern: momentum_sensitive_dqn_seed_(num_seed)
- `seed` list: 42, 123, 999, 2026, 7
- `num_episodes`: 300
- `max_episode_steps`: 3000
- `reward_function`: distance
- `reward_type`: soft
- `action_space`: discrete_3: idle / gas / reverse
- `learning_rate`: 1e-3
- `gamma`: 0.99
- `batch_size`: 64
- `hidden_sizes`: (128, 128)
- `epsilon_start`: 1.0
- `epsilon_end`: 0.05
- `epsilon_decay`: 0.995
- `target_update_frequency`: 1_000
- `evaluation_frequency`: 25
- `evaluation_episodes`: 10
- final held-out evaluation: 30 episodes on seeds `10000` to `10029` by default
- `momentum_bonus_scale`: 0.05
- `momentum_stall_penalty`: 0.02
- `momentum_oscillation_penalty`: 0.02
- `momentum_forward_streak_required`: 3
- `momentum_stall_patience`: 30
- `momentum_progress_clip`: 5.0
- `momentum_angle_limit_deg`: 45.0
- `momentum_angle_penalty_scale`: 0.001
- `momentum_air_penalty`: 0.01
- `momentum_back_wheel_penalty`: 0.005

What changed relative to baseline:
- Vanilla DQN baseline:
  The replay buffer stores the original environment reward directly.
- Momentum-Sensitive DQN variation:
  The replay buffer stores a shaped learning reward instead of the raw environment reward.
- Added a forward-momentum bonus:
  When the car keeps making forward progress for several consecutive steps, the learning reward is increased slightly.
- Added a stall penalty:
  When the car spends too many steps without useful progress, the learning reward is reduced slightly.
- Added an oscillation penalty:
  When the agent keeps switching between opposite drive actions without progress, the learning reward is reduced slightly.
- Added a chassis-angle penalty:
  When the car tilts too far, the learning reward is reduced to discourage unstable flipping.
- Added an airborne penalty:
  When both wheels leave the ground, the learning reward is reduced slightly to discourage uncontrolled motion.
- Added a back-wheel balancing penalty:
  When the car relies too much on back-wheel-only balance, the learning reward is reduced slightly.
- Kept the core DQN structure unchanged:
  same Q-network, same Bellman update, same optimizer, same epsilon-greedy policy, same replay buffer structure, and same target network update logic.

Evaluation comments to reuse in the report:
- In the refreshed final evaluation, `momentum_sensitive_dqn` is the strongest completed method numerically, with the highest mean score, highest mean return, highest mean length, and lowest across-seed final-score standard deviation.
- The shaped reward appears to improve both forward progress and reward alignment under the updated validation and final-evaluation setup.
- The main weakness is qualitative rather than hidden by the aggregate score: the agent can get stuck on more elevated terrain and does not do anything useful to solve the situation.
- The method is also visually aggressive in some seeds, so the strong mean result should be reported together with the remaining failure modes: flipping backward, overshooting hills, and stalling on difficult elevated terrain.
- Because this variation still uses the standard DQN target, it does not remove the selection-evaluation coupling that can cause overoptimistic Q-targets. That is the main reason the next variation should combine this reward function with DoubleDQN.

### EXP-003: Vanilla DoubleDQN

Method description:
- `vanilla_double_dqn` keeps the replay buffer, epsilon-greedy exploration, target-network structure, and Q-network architecture from the vanilla baseline, but changes how the temporal-difference target is computed.
- The goal is to reduce overestimation bias in standard DQN.
- Instead of using one max over a single estimator for both choosing and valuing the next action, `double_dqn` decouples the two steps:
- the online network selects the best next action
- the target network evaluates that selected action
- This method is being tried after `momentum_sensitive_dqn` because it addresses a deeper mathematical issue in DQN itself rather than only changing the reward signal.

Config values:
- `run_name` pattern: vanilla_double_dqn_seed_(num_seed)
- `seed` list: 42, 123, 999, 2026, 7
- `num_episodes`: 300
- `max_episode_steps`: 3000
- `reward_function`: distance
- `reward_type`: soft
- `action_space`: discrete_3: idle / gas / reverse
- `learning_rate`: 1e-3
- `gamma`: 0.99
- `batch_size`: 64
- `hidden_sizes`: (128, 128)
- `epsilon_start`: 1.0
- `epsilon_end`: 0.05
- `epsilon_decay`: 0.995
- `target_update_frequency`: 1_000
- `evaluation_frequency`: 25
- `evaluation_episodes`: 10 for validation during training in these refreshed completed runs
- final held-out evaluation: 30 episodes on seeds `10000` to `10029` by default

What changed relative to baseline:
- Vanilla DQN target:
  `r + gamma * max_a Q_target(s', a)` for non-terminal transitions
- DoubleDQN target:
  `r + gamma * Q_target(s', argmax_a Q_online(s', a))` for non-terminal transitions
- Kept the rest of the pipeline as close as possible to `vanilla_dqn` so the comparison isolates the target-update change.

Main reason for this experiment:
- The first two methods test baseline DQN behavior and reward shaping behavior.
- This third method tests whether the standard DQN max target is itself a source of instability or overoptimism in Hill Climb Racing.
- If `double_dqn` improves final score, stability, or seed consistency, then the next variation should probably build on DoubleDQN instead of standard DQN.

Observed result summary:
- Final mean score: about `482.42`, which is above `vanilla_dqn` at `410.03` but below `momentum_sensitive_dqn` at `529.33`.
- Final score standard deviation: about `121.38`, which is much higher than `momentum_sensitive_dqn` at `64.10` and slightly lower than `vanilla_dqn` at `125.32`.
- Final mean return: about `1786.62`, which is above `vanilla_dqn` but below `momentum_sensitive_dqn`.
- Final mean episode length: about `2202.62`, which is clearly above `vanilla_dqn` and close to `momentum_sensitive_dqn`.

Interpretation to reuse in the report:
- Plain DoubleDQN now improves over the vanilla baseline after retraining with the updated validation setup.
- The improvement is not enough to beat `momentum_sensitive_dqn`, so the best completed result still comes from reward shaping rather than the target-rule change alone.
- Some DoubleDQN seeds are strong, especially `seed7` and `seed999`, while `seed42` and `seed2026` are weaker. This means DoubleDQN is useful but still seed-sensitive.
- This result justified testing `momentum_sensitive_double_dqn`: DoubleDQN target calculation plus the same custom reward function used by `momentum_sensitive_dqn`.
- After running that combined method, the next open question is no longer whether to combine DoubleDQN and momentum shaping. The next open question is how to fix the repeated stuck-recovery failure mode.


### EXP-004: Momentum-Sensitive DoubleDQN

Method description:
- `momentum_sensitive_double_dqn` combines the two useful ideas from the previous experiments.
- It keeps the DoubleDQN target rule from `vanilla_double_dqn`.
- It also uses the custom momentum-sensitive reward function from `momentum_sensitive_dqn`.
- The goal is to test whether the stronger behavior incentive from reward shaping can work better when the bootstrap target is less overestimation-prone.

Config values:
- `run_name` pattern: momentum_sensitive_double_dqn_seed_(num_seed)
- `seed` list: 42, 123, 999, 2026, 7
- `agent_variant`: momentum_sensitive
- `td_target_mode`: double_dqn
- `num_episodes`: 300
- `max_episode_steps`: 3000
- `reward_function`: distance
- `reward_type`: soft
- `action_space`: discrete_3: idle / gas / reverse
- `learning_rate`: 1e-3
- `gamma`: 0.99
- `batch_size`: 64
- `hidden_sizes`: (128, 128)
- `epsilon_start`: 1.0
- `epsilon_end`: 0.05
- `epsilon_decay`: 0.995
- `target_update_frequency`: 1_000
- `evaluation_frequency`: 25
- `evaluation_episodes`: 10
- final held-out evaluation: 30 episodes on seeds `10000` to `10029` by default
- reward-shaping hyperparameters: same as `momentum_sensitive_dqn`

What changed relative to plain DoubleDQN:
- Plain DoubleDQN:
  Stores the original environment reward and uses the DoubleDQN target.
- Momentum-Sensitive DoubleDQN:
  Stores the shaped momentum-sensitive reward and uses the DoubleDQN target.

Current status:
- completed for 5 seeds
- final held-out evaluation used 30 episodes on seeds `10000` to `10029`

Observed result summary:
- Final mean score: about `516.55`, which is below `momentum_sensitive_dqn` at `529.33` but above `vanilla_double_dqn` at `482.42` and `vanilla_dqn` at `410.03`.
- Final score standard deviation: about `58.04`, the lowest among the completed methods.
- Final mean return: about `2017.76`, below `momentum_sensitive_dqn` but above both unshaped methods.
- Final mean episode length: about `2380.01`, the highest among the completed methods.

Interpretation to reuse in the report:
- Combining DoubleDQN with the momentum-sensitive reward improved over plain DoubleDQN and produced the most consistent final scores across seeds.
- It did not beat `momentum_sensitive_dqn` on mean score or mean return, so the combination did not automatically improve the best reward-shaped policy.
- The high mean episode length is not purely good. It matches the visual observation that the agent can survive for a long time while getting stuck and doing little to recover, so even though the episode length is long, the agents are practically doing nothing.
- Seed `2026` is especially important to discuss carefully: it has the strongest numeric result in the final table, but visual notes still show unstable behavior and stuck states with little to no improvement.
- The main remaining weakness is no longer only flipping or early crash behavior. The repeated failure mode is stalled survival: the agent gets stuck, does not increase gas enough, and does not actively solve the situation.


### EXP-005: Anti-Stall Momentum DQN

Method description:
- This is the recommended next experiment because `momentum_sensitive_dqn` is still the strongest completed method by final mean score and final mean return.
- Keep the standard DQN target rule and the momentum-sensitive reward terms.
- Add a stronger stuck-recovery reward modification that penalizes long periods of little or no progress, especially when the selected action is idle or reverse instead of gas.
- This isolates the reward change more cleanly than applying it to DoubleDQN first. If the result improves, the same anti-stall idea can later be combined with DoubleDQN.

Planned config values:
- `run_name` pattern: antistall_momentum_dqn_seed_(num_seed)
- `seed` list: 42, 123, 999, 2026, 7
- `agent_variant`: antistall_momentum
- `td_target_mode`: dqn
- base reward-shaping hyperparameters: same as `momentum_sensitive_dqn`
- new stuck-recovery settings implemented in `configs.py`:
  - `anti_stall_patience`: `20` low-progress steps before the stronger penalty begins
  - `anti_stall_progress_threshold`: `0.01` minimum progress needed to count as not stuck
  - `anti_stall_idle_penalty`: `0.08` extra penalty when stuck and action is idle (`0`)
  - `anti_stall_reverse_penalty`: `0.05` extra penalty when stuck and action is reverse (`2`)
  - `anti_stall_gas_penalty`: `0.005` very small penalty when stuck and gas still fails to create progress
  - `anti_stall_penalty_growth`: `0.002` gradual penalty increase during long stuck streaks
  - `anti_stall_penalty_cap`: `0.25` cap so the reward does not become too extreme
  - `anti_stall_gas_recovery_bonus`: `0.03` small bonus when the agent was stuck and gas (`1`) actually creates progress

Design caution:
- Do not simply reward gas every time. On steep terrain, the car sometimes needs braking or reversing to rebalance.
- The penalty should activate only after repeated low-progress steps, not during short normal slowdowns.
- A good rule is to penalize "stuck plus not trying to recover" more than "temporarily slow."
- Track a new diagnostic during evaluation if possible: number of stuck steps, longest stuck streak, and gas action rate during stuck streaks.

Expected result:
- This variation should reduce the common behavior where the agent survives but sits stuck without improving.
- The key metric to watch is not only final mean score. Also compare mean length against score and inspect whether long episodes still contain stalled behavior.
- If this improves `momentum_sensitive_dqn`, then EXP-006 can test `antistall_momentum_double_dqn` to see whether the anti-stall reward and DoubleDQN target rule work better together.

---

## 8. Behavior Notes

Use this section for qualitative observations from `watch_checkpoint.py`.

### vanilla_dqn

- Did the car move confidently?
  - Sometimes, but the behavior is less consistently strong than the two modified methods in the refreshed final evaluation.
- Did it hesitate?
  - Yes. Some seeds survive but do not convert survival into strong forward progress.
- Did it survive long but score slowly?
  - Yes, especially `vanilla_dqn_seed42`, which has very long average survival but only moderate final score.
- Common failure mode: unstable starts, slow progress, and high seed sensitivity
- Best-looking seed: Seed 7 by final score
- Worst-looking seed: Seed 123 by final score

### momentum_sensitive_dqn

- Did the variation look more stable?
  - For momentum_sensitive_dqn_seed2026: Some of the forward motion are too aggressive and sometimes cause the car to flip backwards
  - momentum_sensitive_dqn_seed42: A little bit more stable, but still quite aggressive on some terrain
  - Additional observation: `momentum_sensitive_dqn` can get stuck on more elevated terrain and does not do anything useful to solve the situation.
- Did it look faster or more aggressive?
  - In general: faster and more aggressive, but not fully reliable because there is a higher chance that it will flip both forward and backward
- Did it crash more often?
  - Yes, sometimes in the very beginning and it overshoots the hill quite a bit. The speed is good but the reliability is low
- Common failure mode: terminated episodes, backward flips, overshooting hills, or getting stuck on elevated terrain
- Best-looking seed: Seed 42 visually; Seed 999 by final held-out score
- Worst-looking seed: Seed 2026 visually; Seed 123 by final held-out score

### vanilla_double_dqn

- Did the variation improve over vanilla DQN?
  - Yes, in the refreshed final held-out metrics it improves over vanilla DQN in mean score, mean return, and mean episode length.
- Did it beat momentum-sensitive DQN?
  - No. It is second-best by final mean score, final mean return, and final mean episode length.
- Common failure mode: seed sensitivity, especially weaker final return for `seed42` and `seed2026`
- Best-looking seed by final held-out score: Seed 999
- Worst-looking seed by final held-out score: Seed 2026

### momentum_sensitive_double_dqn

- Did the combined variation improve numerically?
  - Yes. It improves over plain `vanilla_double_dqn` in final mean score, final mean return, and final mean episode length.
  - It also has the lowest final-score standard deviation among the completed methods, so the score result is numerically consistent across seeds.
- Did it solve the behavior problem?
  - No. Some agents, including behavior observed around seed `2026`, are still unstable and still get stuck with little to no improvement.
- Did it look fast?
  - The speed in general is quite good, but the consistency is not there.
- Common failure mode: the agent gets stuck and usually does not do anything useful to recover.
- Best numeric seed: Seed 2026 by final score and return
- Weakest numeric seed: Seed 999 by final score and return
- Report caution:
  - The high mean episode length can make the method look strong, but long survival can include stuck behavior rather than active terrain solving.

### antistall_momentum_dqn

- This is the recommended next variation.
- Reason for using this base first: `momentum_sensitive_dqn` has the best completed mean score and mean return, so the anti-stall reward should be tested there before adding another algorithmic change.
- Main behavior target: reduce stuck states where the agent is not trying to increase gas.
- The reward should strongly penalize repeated no-progress steps when the agent keeps choosing idle or reverse, and only give a small recovery bonus for gas when it actually helps progress.
- Current implementation note:
  - Code mode is available in `hcr_dqn` as `agent_variant = antistall_momentum`.
  - Default run name pattern is `antistall_momentum_dqn_seed7`.

### antistall_momentum_double_dqn

- This should be treated as a later follow-up, not the immediate next experiment.
- Only run this if `antistall_momentum_dqn` improves stuck recovery, because then it is worth testing whether the same reward change also benefits the DoubleDQN target rule.

---

## 9. Final Comparison Summary

Write the final takeaway here after all experiments are done.

Questions to answer:
- Which method had the highest mean score?
- Which method had the lowest variance?
- Was your variation better than vanilla DQN?
- Did `double_dqn` improve stability or final score enough to become the new base algorithm?
- Was it better on average, more stable, or both?
- Did the visual behavior match the numeric results?

Final summary:
- Based on the refreshed held-out final evaluation, `momentum_sensitive_dqn` is still the best completed method by mean score (`529.33`) and mean return (`2322.02`).
- `momentum_sensitive_double_dqn` is the second-best method by mean score (`516.55`) and has the lowest final-score standard deviation (`58.04`), so combining DoubleDQN with the custom reward improved consistency.
- The combined method has the longest mean episode length (`2380.01`), but the visual behavior shows why length alone is not enough: the agent can survive while getting stuck and not doing anything useful.
- The main remaining problem is stalled recovery. The speed is generally good, but consistency is not there, and agents can still get stuck with little to no improvement.
- The recommended next experiment is `antistall_momentum_dqn`: keep the best-performing `momentum_sensitive_dqn` base, but add a stronger penalty for repeated no-progress steps when the agent is idle/reverse or not attempting gas-based recovery.
- The later follow-up is `antistall_momentum_double_dqn`, but it should come after EXP-005 so the reward modification can be evaluated cleanly before mixing it with the DoubleDQN target rule.

---
