# Zeroth-order queueing policy training

Run from the repository root with the existing QGym Python dependencies:

```sh
python RL/PPO/ZO_train.py ZO reentrant_2
python -m unittest RL.PPO.test_ZO_trainer
```

Configuration lives in `RL/policy_configs/ZO.yaml`. Both positional arguments
accept config names or absolute YAML paths. `--output-dir` selects a new output
directory; otherwise the directory is `RL/PPO/<network_name>`, for example
`RL/PPO/reentrant_2`.

Set `pretrain_evaluation.enabled: false` to skip normalization fitting and cached
statistics entirely. Queue inputs remain raw, with identity statistics (mean 0,
std 1) saved in checkpoints, including `initial_policy.pt`. Behavioral cloning
still follows its own switch. The evaluation switch defaults to true if omitted.

When enabled, if `RL/PPO/pretrain_evaluation.json` exists, its `mean_queue_length` and
`std_queue_length` are loaded instead of running the 100-environment pretraining
evaluation. The lookup is independent of the working directory. Statistics are
copied into the run directory and saved in the initial policy checkpoint.
Behavioral cloning still follows `behavior_cloning.enabled`; the JSON contains
statistics, not trained weights. Use statistics appropriate to the chosen network.
Invalid or nonfinite cached statistics raise an error.

Enable `training.policy_diagnostics.enabled` and set
`training.policy_diagnostics.num_samples` to Q to record diagnostics in each
`history.jsonl` entry. Each draw chooses an original-policy trajectory uniformly,
then a pre-action event state uniformly within it, with replacement (not weighted
by event duration). Streaming reservoirs keep memory bounded by Q queue vectors;
sampling uses independent random streams and does not alter training draws.
All actors receive the same complete queue vectors; their masked probabilities
are compared separately for each of S expanded server rows.

Diagnostics report the count and fraction of the Q*S server states with at most
one compatible nonempty queue, and separate zero/one counts. For each tested
perturbation (identified by partition), they report mean, maximum, and per-server
mean total variation distance: `0.5 * sum(abs(perturbed_probs - original_probs))`.
Distance 0 means identical distributions and 1 means disjoint distributions;
`similarity` is one minus the mean distance. All distance and similarity metrics
exclude server states with at most one compatible nonempty queue, including idle
servers. `similarity_per_server_states` records the number included. Metrics are
null when no eligible states exist (also per server); the retained
`mean_total_variation_with_choice` field equals `mean_total_variation`.
Comparisons use the actual perturbed actors before
accepted updates are combined.

Set `behavior_cloning.enabled: false` to skip behavioral cloning (default: true,
including configs that omit the switch). The randomly initialized actor is
still saved as `initial_policy.pt`, and its partition maxima are fixed for the
whole run. A timestamped message reports that cloning is disabled. With no
cloning, all-zero bias partitions remain unperturbed in split-layer mode.

The actor uses WC PPO's hidden widths, Tanh activations and orthogonal gains
(sqrt(2) for hidden weights, 0.01 for output weights, zero biases). It has no
critic, reward scaling, or parameter normalization. Queue inputs use PPO WC's
scalar observation standardization, fitted once after optional BC (see below).
BC uses 100,000 uniform integer queue vectors in [0,100] and a teacher with
scores `queue_length[j] * service_rate[s, j]` for compatible queues, zero
otherwise. Scores are divided by their per-server sum, without softmax or
clipping. Pool replicas use their corresponding server's service rates.
Zero-score rows use uniform probabilities over compatible nonempty queues,
or all compatible queues if the server is idle. Training uses MSE,
Adam at 0.0003, one pass, batches of 100. Sampling batches directly avoids the
callback's actor-critic dependencies. Actor seeds are explicit; initialization
has the same distribution as PPO, not identical RNG consumption from its critic.

## Choices and defaults beyond the requested algorithm

- Default mode: vanilla; 100 iterations; perturbation ratios 0.1 down to 0.01;
  update ratio 1; 100 trajectories of 10,000 external arrivals each.
- `logarithmic` means linear interpolation of log(ratio), i.e. geometric
  decay. Linear and cosine schedules are also available. A one-iteration run
  uses the initial ratio.
- Immediately after behavioral cloning, `para_max` is computed once as the
  maximum absolute value of the initial policy parameters in each assigned
  partition and remains fixed throughout training. That partition's
  distance is the scheduled ratio times `para_max`; accepted updates use this
  same distance times `update_ratio`. Vanilla uses the whole actor, and
  split-layer uses each individual weight/bias tensor. An all-zero partition
  has zero distance and remains unchanged. Distances change only with the
  scheduled ratio, regardless of subsequent parameter updates. History records the
  ratio, per-partition maxima, and actual distances.
- Seeds: model/perturbations 100, train environment 3003, evaluation 42 onward.
  Evaluation seeds are reused across iterations; state, including residual
  service work, arrival clocks and absolute time, comes from the original policy.
- The first evaluation uses the environment's configured queue vector and
  independently sampled residual work and arrival clocks for each trajectory.
  Thus full initial states differ even when queue counts match. Set
  `training.initial_queues` to one vector per trajectory to customize counts too.
- Evaluation always samples actions from probabilities, even if `env.randomize`
  is false in a custom config. Each trajectory has a separate Torch generator
  seeded by its evaluation seed, shared across baseline/candidate comparisons
  for reproducibility. Logit masking ensures empty
  queues cannot be selected when a compatible nonempty queue exists, including
  under extreme logits. Idle servers retain WC's compatible-queue fallback.
- Partitions are contiguous in flattened PyTorch parameter order. Each has
  floor(total/count) parameters, with the entire remainder added to the last:
  99 parameters / 5 gives 19, 19, 19, 19, 23. Counts must be between 1 and the
  parameter total. Split-layer uses each weight or bias tensor as one partition.
- All candidates compare against the same unchanged baseline. Strict improvement
  accepts a partition; ties reject it. Accepted updates are combined without
  another evaluation of the combination, using ratio times the tested direction.
- CPU workers use spawn and one Torch thread each, bounded by configured CPUs,
  available CPUs, number of jobs, and 48. Server pool rows are expanded in the
  simulator consistently with PPO's expanded actor output.
- Worker inputs and baseline ending states use NumPy arrays for transfer, with
  service jobs stacked into one array per queue instead of sharing individual
  Torch tensors. Queue order, dtypes, simulation time, and all residual clocks
  are preserved. Only original-policy trajectories return ending states;
  perturbed trajectories return scores only. Each original ending state still
  becomes its corresponding trajectory's initial state in the next iteration.
- Arrival/service streams are separated by source event and queue. A local
  evaluation adapter prevents internal routing from resetting external arrival
  clocks (the base simulator does this). This is necessary for paired external
  arrival histories. ZO uses float64 service rates, evaluation time, and event
  times so consumed service work also stays in float64, and to avoid rounding past
  nearly simultaneous events and creating negative residual clocks; the shared
  simulator keeps its existing float32 default for other callers. Time-dependent
  rate functions still follow the existing loader's sampling convention.
  Zero-change arrival rows (dummy events with tiny rates in some network
  configs) are not counted as external arrivals; their clocks are disabled
  after firing because they never add jobs.
- The objective is the time integral of the **sum** of queue lengths divided by
  trajectory elapsed time, then the arithmetic mean across trajectories. Holding
  cost weights are not used. The event that reaches the arrival limit is included.
- After optional behavioral cloning, pre-training evaluation matches PPO WC:
  100 fresh base-simulator environments with consecutive seeds starting at
  `env.test_seed`, each reset and run for the environment's `test_T` simulator
  events (300,000 for reentrant_9). This stage uses the PPO simulator path,
  without ZO's paired-clock adapters. The CPU cap still follows `max_cpus`.
  Per-queue time averages are averaged across trajectories; their scalar mean
  and sample standard deviation normalize inputs as `(queues - mean)/(std + 1e-8)`.
  These statistics stay fixed throughout training, as in WC's default config,
  and are saved as buffers in every policy checkpoint. Raw queues still determine
  action feasibility. `pretrain_evaluation.json` records the statistics and budget.
  This stage does not advance the ZO comparison states or change their configured
  trajectory count and external-arrival horizon.
- `initial_policy.pt` is first saved after optional cloning, then updated with
  the fitted normalization before any zeroth-order updates, for reuse with
  `policy.load_state_dict(torch.load(path, weights_only=True))` on a matching
  `ZOPolicy` architecture. Older checkpoints without normalization buffers need
  explicit migration before loading into this version.
- Every pre-update original actor is saved as `original_000000.pt`, etc., with
  config, initial evaluation states, and resolved seeds. `final_policy.pt` saves
  the last updated actor; `history.jsonl` records scores and acceptance decisions.
  Existing output directories are rejected to avoid overwriting a prior run.

Checkpoints are PyTorch files, not SB3 policy archives. Original checkpoints
include simulator states and are intended to be loaded only from trusted runs.
