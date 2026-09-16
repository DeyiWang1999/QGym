# Evaluate ZO checkpoints with the PPO evaluation protocol

From the repository root on the machine holding the checkpoints:

```sh
python RL/PPO/ZO_eval.py RL/PPO/reentrant_9
```

This requests `original_000000.pt`, `original_000010.pt`, ..., `original_000100.pt`.
All requested files and their saved seed metadata are checked before any rollout.
Add `--check-only` to perform only these checks.

A run with `total_iterations: 100` saves originals numbered 0–99. For that run:

```sh
python RL/PPO/ZO_eval.py RL/PPO/reentrant_9 --indices 0 10 20 30 40 50 60 70 80 90 --include-final
```

`--include-final` evaluates `final_policy.pt` as a separately named result; it
does not rename it to `original_000100.pt`. The final state dict has no metadata,
so it must belong to the same run as `original_000000.pt` in that directory.

Protocol:

- Exactly 100 reset environments, the checkpoint's `env_config.test_T` simulator events per environment
  (30 million events per policy), starting at zero queues/time as in PPO.
- Uses `RL/utils/eval.py`'s actual PPO rollout and metric aggregation, with
  the base PPO simulator, not ZO's training simulator or external-arrival horizon.
- CPU process workers: up to 48, limited by available CPUs. There are 100
  trajectories, not necessarily 100 simultaneous processes.
- Test simulator seeds default to 3003–3102. `--test-seed` changes the first seed.
  The same seeds are reset for every checkpoint. Seeds overlapping saved ZO
  comparison trajectories, configured comparison trajectories, the initial
  training environment, or the 100 normalization trajectories are rejected.
- Sample actions from the ZO actor and preserve its saved normalization.
  Action sampling is explicitly seeded per trajectory with its environment seed;
  these are separate Torch and NumPy generators. PPO currently does not seed its
  action generator, so historical PPO evaluations are not bitwise reproducible.
  PPO training behavior is unchanged by the new optional evaluation setting.
- Current PPO configs use test seeds 42–141 and training seeds 3003–3052.
  Current ZO configs use optimization seeds 42–51 and normalization seeds 42–141.
  Therefore exact PPO test seeds and disjoint ZO test seeds cannot both be used
  with these configs. The default 3003–3102 is held out from ZO, but overlaps PPO
  training. For a common test set held out from both current training setups,
  use `--test-seed 100003` (and verify any other runs used in the comparison).

Results are saved after each completed policy to `ppo_evaluation.json` in the
checkpoint directory, including seeds, excluded seeds, and all eight PPO callback
metrics. `total_queue_mean` and `total_queue_std` describe the time-averaged total
queue length across trajectories. `queue_std_across_queues` has PPO's different
meaning: spread across the 27 per-queue means, not uncertainty of total queue length.
The callback also prints its holding-cost metrics; these are distinct accumulators
from the queue-length metrics returned by PPO's callback.
Use `--output PATH` for a different results file; an existing output is replaced.

Load only trusted checkpoint files. Full checkpoints contain Python objects.

## Ranking check over longer trajectories

Add `--ranking-check` to either command above. For example, for a 100-iteration run:

```sh
python RL/PPO/ZO_eval.py RL/PPO/reentrant_9 --indices 0 10 20 30 40 50 60 70 80 90 --include-final --ranking-check
```

The network name and base horizon come from each checkpoint's saved `env_config`.
Each of the 100 environments runs continuously for `5 * test_T` simulator events,
with snapshots at `test_T`, `2 * test_T`, `3 * test_T`, `4 * test_T`, and `5 * test_T`.
For `test_T=300000`, these are 300,000, 600,000, 900,000, 1,200,000, and 1,500,000.
Pass another network's checkpoint directory to evaluate that network. Editing a
local YAML after training does not change the configuration embedded in a checkpoint.
The PPO-compatible evaluator currently supports `num_pool=1` networks.
At each snapshot, `total_queue_mean` and `total_queue_std` are the mean and sample
standard deviation across the 100 trajectories' time-weighted average **total**
queue lengths. Each average includes all events from the start through that
snapshot; these are not separate 300,000-event windows. Neither the environment
nor its random streams are reset between snapshots. The 48-worker limit and
seed separation checks remain in force.

Results default to `ppo_ranking_evaluation.json`, preserving ordinary evaluation
output. Each policy has a `milestones` list containing `events`, `total_queue_mean`,
and `total_queue_std`; its usual `metrics` entry describes the final horizon.
Snapshots are collected from workers and written after each policy completes.
Without `--ranking-check`, the horizon is the checkpoint's unmodified `test_T`.
Each policy result records its network, `base_test_T`, and actual event horizon.
