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

## Research Flow And Why DoubleDQN Is Next

Use this section to keep the experiment story consistent in the report.

Planned progression:
1. Start with `vanilla_dqn` as the baseline.
2. Try `momentum_sensitive_dqn` as a reward-shaping variation on top of the same core DQN update.
3. After comparing those two, note the deeper mathematical bottleneck in standard DQN itself.
4. Try plain `double_dqn` next to address that bottleneck directly.
5. Only after plain `double_dqn` is understood, design a new DoubleDQN-based variation later.

Reason for adding `double_dqn`:
- Both `vanilla_dqn` and `momentum_sensitive_dqn` still rely on the standard DQN target.
- With function approximation, Q-values can be unevenly overestimated or underestimated during learning.
- In standard DQN, the max operator uses the same estimates to both select the next action and evaluate that action's value.
- That coupling can reinforce overestimated actions and make learning less stable.
- `double_dqn` separates those two jobs:
- the online network selects the action
- the target network evaluates the selected action
- This makes `double_dqn` the clean next experiment because it tests whether the instability comes partly from the DQN target rule itself, not only from reward design.

Reporting takeaway:
- `momentum_sensitive_dqn` answers: "What happens if I keep DQN but change the learning reward?"
- `double_dqn` answers: "What happens if I keep the task mostly the same but fix the overestimation bias in the update rule?"
- A future DoubleDQN-based variation will answer: "What happens if I combine the algorithmic fix with a new design choice?"

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

Current comparative interpretation for `vanilla_dqn` and `momentum_sensitive_dqn`:

- This is a well-constructed evaluation plot. Presenting the mean together with the across-seed standard deviation for `n = 5` gives an academically honest view of the training dynamics, which is especially important in reinforcement learning because performance can vary substantially across random seeds.

1. Early-stage sample efficiency
- Observation: `momentum_sensitive_dqn` shows a clearly stronger initial learning rate. Between episodes 0 and 100, its mean training score rises much faster than `vanilla_dqn`, reaching roughly 150 while the vanilla baseline still struggles to move much beyond 75.
- Interpretation: the momentum-sensitive reward shaping appears to accelerate early policy discovery. By rewarding more consistent forward motion, it likely helps the agent leave the near-random exploration regime earlier than the baseline.

2. Asymptotic performance and policy collapse
- Observation: the two trajectories separate in the opposite direction later in training. `momentum_sensitive_dqn` appears to peak around episode 180 at roughly 220 mean score, then degrades and ends closer to 150 by episode 300. In contrast, `vanilla_dqn` improves more slowly but more steadily, and appears to finish around 280 to 300 by the end of training.
- Interpretation: this pattern is consistent with late-stage policy collapse or catastrophic forgetting in `momentum_sensitive_dqn`. The shaping helps early exploitation, but may destabilize learning once the state distribution becomes harder and vehicle speed increases. The agent may over-specialize to a narrow high-momentum driving style and then fail when it encounters difficult terrain or unstable body states.

3. Inter-seed variance
- Observation: both methods show wide standard deviation bands, but the late-stage variance of `momentum_sensitive_dqn` is particularly extreme. Its lower band can fall near zero or below, while its upper band reaches much higher values.
- Interpretation: this suggests high sensitivity to initialization and early exploration. For `vanilla_dqn`, the widening band in the later episodes may mean that one or two seeds learn very strong policies while other seeds remain mediocre, creating a high mean with large spread. For `momentum_sensitive_dqn`, the very low lower bound suggests that at least one seed may suffer a severe collapse that drags the mean downward.

4. Algorithmic hypothesis
- A plausible explanation is that the momentum-sensitive shaping improves the short-term exploration and exploitation balance, but worsens long-term value instability. Standard DQN already has a tendency to overestimate action values through the `max_a Q(s', a)` target. If the momentum-sensitive reward inflates the temporal-difference target during sustained forward movement, that bias may become stronger at the same time the car is entering faster and more unstable physics states, such as aggressive hill launches or chassis flips.

Important reporting note:
- Use this interpretation when discussing `plots/plot_1_learning_curves_mean_episode_score.png`.
- Do not treat this section as a replacement for final reporting.
- Final performance claims should still be grounded in the held-out results recorded from `evaluation_summaries.csv` and `evaluation_episode_details.csv`.

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

### Plot 5: Seed variance box plot for final score

File:
- `plots/plot_9_seed_variance_box_plot_final_score.png`

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
| EXP-002 | momentum_sensitive_dqn | Reward shaping for smoother forward motion and less unstable flipping/balancing | Yes | 5 | Planned |
| EXP-003 | double_dqn | Replace the standard DQN max target with decoupled online-action selection and target-network evaluation | Yes | 5 | Planned |
| EXP-004 | double_dqn_variation_tbd | Future variation on top of DoubleDQN after the plain DoubleDQN baseline is understood | No, design pending | 5 | Backlog |

---

## 5. Per-Run Results
### EXP-001

| Experiment ID | Method | Seed | Run Name | Final Mean Score | Final Mean Return | Final Mean Length | Notes  |
|---|---|---:|---|---:|---:|---:|---|
| EXP-001 | vanilla_dqn | 7 | `vanilla_dqn_seed7` | | | | |
| EXP-001 | vanilla_dqn | 42 | `vanilla_dqn_seed42` | | | | |
| EXP-001 | vanilla_dqn | 123 | `vanilla_dqn_seed123` | | | | |
| EXP-001 | vanilla_dqn | 999 | `vanilla_dqn_seed999` | | | | |
| EXP-001 | vanilla_dqn | 2026 | `vanilla_dqn_seed2026` | | | |At the beggining, the agent will always flip 2 anti-clockwise full rounds, barely keeping the head from the ground. But all three episode the agent was able to survive |


---

## 6. Aggregate Results By Method

After all seeds for one method are complete, summarize them here.

| Experiment ID | Method | Num Runs | Final Mean Score | Score Variance | Score Std Dev | Final Mean Return | Return Variance | Return Std Dev | Final Mean Length | Length Variance | Length Std Dev |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-001 | vanilla_dqn | 5 |  |  |  |  |  |  |  |  |  |
| EXP-002 | momentum_sensitive_dqn | 5 |  |  |  |  |  |  |  |  |  |
| EXP-003 | double_dqn | 5 |  |  |  |  |  |  |  |  |  |
| EXP-004 | double_dqn_variation_tbd | 5 |  |  |  |  |  |  |  |  |  |

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
- `evaluation_episodes`: 5 for validation during training
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
- `evaluation_episodes`: 5 for validation during training
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
- `momentum_sensitive_dqn` appears to improve early-stage sample efficiency. Its score rises faster in the first part of training, which suggests that the shaped reward helps the agent find forward-driving behavior more quickly.
- The same method appears to peak earlier and then degrade later, which is consistent with late-stage instability, policy collapse, or catastrophic forgetting.
- Its wide late-stage variance indicates that the reward shaping is not reliably improving all seeds in the same way. Some runs may benefit strongly, while others fail badly enough to pull the overall mean down.
- A reasonable hypothesis is that the shaping terms encourage aggressive momentum exploitation early on, but may also amplify value overestimation or over-specialization once the terrain and body dynamics become more difficult.
- Because this variation still uses the standard DQN target, it does not remove the selection-evaluation coupling that can cause overoptimistic Q-targets. That is the main reason `double_dqn` is the next planned method.

### EXP-003: DoubleDQN

Method description:
- DoubleDQN keeps the replay buffer, epsilon-greedy exploration, target-network structure, and Q-network architecture from the vanilla baseline, but changes how the temporal-difference target is computed.
- The goal is to reduce overestimation bias in standard DQN.
- Instead of using one max over a single estimator for both choosing and valuing the next action, `double_dqn` decouples the two steps:
- the online network selects the best next action
- the target network evaluates that selected action
- This method is being tried after `momentum_sensitive_dqn` because it addresses a deeper mathematical issue in DQN itself rather than only changing the reward signal.

Config values:
- `run_name` pattern: double_dqn_seed_(num_seed)
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
- `evaluation_episodes`: 5 for validation during training
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

Expected evaluation questions:
- Does `double_dqn` reduce late-stage collapse relative to `momentum_sensitive_dqn`?
- Does it improve seed-to-seed stability relative to `vanilla_dqn`?
- Does it preserve or improve final held-out score?

### EXP-004: DoubleDQN-Based Variation (TBD)

Method description:
- This placeholder is reserved for a future variation built on top of `double_dqn`.
- The exact variation is intentionally not fixed yet.
- It should only be defined after the plain `double_dqn` baseline is implemented and evaluated.

Provisional plan:
- keep the DoubleDQN target rule
- decide later whether the added variation should be reward shaping, architecture change, exploration change, or another stability mechanism
- compare it against plain `double_dqn`, not only against `vanilla_dqn`

Current status:
- algorithm variation not chosen yet
- do not fill final config details until `EXP-003` results are available

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

### momentum_sensitive_dqn

- Did the variation look more stable?
  - For momentum_sensitive_dqn_seed2026: Some of the forward motion are too aggressive and sometimes cause the car to flip backwards
  - momentum_sensitive_dqn_seed42: A little bit more stable but is quite aggressive on the some
- Did it look faster or more aggressive?
  - In general: Faster and more aggressive, but I dont think it is reliable since there is a higher chance that it would flip (both to the front and back)
- Did it crash more often?
  - Yes, sometimes in the very beginning and it overshoots the hill quite a bit. The speed is good but the reliability is low
- Common failure mode: terminated (The agent is dead)
- Best-looking seed: Seed 42
- Worst-looking seed: Seed 2026

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
- 

---
