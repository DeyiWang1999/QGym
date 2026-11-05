# Zeroth-Order Training on QGym

This repository is built on the original [QGym repository](https://github.com/namkoong-lab/QGym), which provides simulation and benchmarking tools for queueing network controllers. This README focuses on zeroth-order (ZO) policy training and evaluation, with instructions for the PPO baseline.

## Python environment

In your Python environment, install the dependencies with:

```bash
python -m pip install torch==2.2.2 numpy==1.26.4 scipy==1.11.4 gym==0.26.2 gymnasium==0.29.1 stable-baselines3==2.3.2 shimmy==1.3.0 PyYAML==6.0.1 matplotlib==3.8.4 tqdm==4.66.4 cvxpy==1.4.3 pathos==0.3.2
```

Run the commands below from the repository root.

## ZO training

### Run from the command line

```bash
python RL/PPO/ZO_train.py ZO reentrant_2 --output-dir RL/PPO/reentrant_2_zo_run1
```

The positional arguments select the policy configuration (`RL/policy_configs/ZO.yaml`) and environment configuration (`configs/env/reentrant_2.yaml`). Both accept names with or without `.yaml`; absolute YAML paths are also supported. If omitted, these arguments default to `ZO` and `reentrant_2`.

`--output-dir` selects the output directory, relative to the current working directory or as an absolute path. Without this option, results go to `RL/PPO/<env_config.name>/`. **The output directory must not already exist**, so choose a new directory for each run. ZO training runs on CPU.

The trainer optionally initializes the actor through behavior cloning, then compares the current actor with perturbed candidates using paired simulation trajectories. A candidate improves the objective when it reduces average holding cost per unit simulated time, averaged across trajectories. Accepted directions update the actor by a fraction of the perturbation. Simulator states from the unperturbed actor continue into the next iteration.

### Configure the training process

Edit [RL/policy_configs/ZO.yaml](RL/policy_configs/ZO.yaml) before launching a run. The environment YAML separately defines the network, service and arrival parameters, holding costs, initial queues, and evaluation horizon `test_T`.

| Configuration | Effect |
| --- | --- |
| `env.device` | Must be `cpu` for ZO. |
| `env.env_temp` | Temperature passed to the simulator. |
| `env.model_seed` | Seeds actor initialization and behavior-cloning queue samples. |
| `env.train_seed` | Seeds the training environment and supplies the starting seed for default comparison trajectories. |
| `env.test_seed` | Starting seed for optional pre-training normalization evaluation. |
| `env.randomize` | Actor action-selection setting; training comparisons and evaluation explicitly sample actions regardless of this flag. |
| `model.scale` | Controls actor hidden-layer widths. The ZO implementation uses a work-conserving (WC) actor without a critic; keep `model.policy_name: WC`. |
| `behavior_cloning.enabled` | Enables initialization by fitting action probabilities proportional to queue length times service rate over compatible queues. Set to `false` to start from the initialized actor. |
| `behavior_cloning.num_samples`, `batch_size`, `epochs`, `learning_rate` | Control the amount of synthetic queue data and the Adam optimization used for behavior cloning. |
| `pretrain_evaluation.enabled` | When `false`, uses raw queue inputs. When `true`, loads normalization statistics from `RL/PPO/pretrain_evaluation.json` if present; otherwise estimates them using 100 trajectories of `test_T` events. Ensure any shared cache is appropriate for the run. |
| `training.mode` | Selects the perturbation scheme described below. |
| `training.total_iterations` | Number of ZO update iterations. |
| `training.initial_perturbation_ratio`, `ending_perturbation_ratio` | Start and end perturbation ratios, scaled by fixed parameter magnitudes measured after initialization/behavior cloning. Zero-scale parts inherit the preceding part's scale; leading zero-scale parts remain unchanged. |
| `training.perturbation_scheduler` | `logarithmic` (geometric decay), `linear`, or `cosine` schedule between the two ratios. |
| `training.perturbation_seed` | Seeds the random sign perturbation directions. |
| `training.partition_count` | Number of parameter-vector partitions in `partitioned` mode; the final partition contains the remainder. |
| `training.evaluation_trajectories` | Number of paired trajectories used to compare each candidate with the current actor. These trajectories select training updates, so they are not held-out tests. |
| `training.evaluation_length` | Number of simulator events per comparison trajectory per iteration. |
| `training.evaluation_seeds` | Explicit list of distinct seeds, one per trajectory. With `null`, the implementation uses consecutive seeds starting at `env.train_seed` (the YAML comment currently says `env.test_seed`). |
| `training.initial_queues` | Optional list of initial queue vectors, one per trajectory. With `null`, uses the environment YAML's `init_queues`. |
| `training.update_ratio` | Fraction of an accepted perturbation applied to the actor. |
| `training.max_cpus` | Upper bound on worker processes, also limited by available CPUs, comparison jobs, and a hard cap of 48. |
| `training.policy_diagnostics.enabled`, `num_samples` | Enable policy comparisons on sampled queue states and control the number of sampled states per iteration. Results are included in the history. |

The supported modes are:

- `vanilla`: perturb the entire parameter vector as one candidate with one scale.
- `partitioned`: perturb each contiguous parameter partition separately and accept improvements independently.
- `split_layer`: perturb each parameter tensor separately (weights and biases form separate parts), with independent comparisons and acceptance.
- `split_vanilla`: use tensor-specific scales but combine all perturbations into one candidate and accept or reject them together.

The checked-in configuration uses `split_vanilla`, 100 iterations, 10 comparison trajectories, and 135,000 events per trajectory, with behavior cloning enabled and pre-training normalization evaluation disabled. Adjust the horizon and iteration count for your environment and compute budget.

### Generated outputs

The output directory contains:

| File | Contents |
| --- | --- |
| `initial_policy.pt` | Actor state dictionary after optional behavior cloning and normalization setup. |
| `original_000000.pt`, `original_000001.pt`, ... | Checkpoint saved **before** each iteration's update, containing actor weights, policy/environment configurations, comparison seeds, and simulator states. |
| `history.jsonl` | One JSON record per completed iteration: perturbation ratio, parameter scales and distances, active partitions, baseline/candidate scores, acceptance decisions, and optional policy diagnostics. Lower scores are better. |
| `final_policy.pt` | Actor state dictionary after the final update. |
| `pretrain_evaluation.json` | Normalization statistics, written only when pre-training evaluation is enabled, whether computed or loaded from the shared cache. |

Progress and iteration results are also printed to the terminal. With 100 iterations, checkpoints range from `original_000000.pt` through `original_000099.pt`; the final updated actor is in `final_policy.pt`.

### Evaluate saved ZO policies

For the 100-iteration run above, evaluate every tenth checkpoint and the final actor with:

```bash
python RL/PPO/ZO_eval.py RL/PPO/reentrant_2_zo_run1 --indices 0 10 20 30 40 50 60 70 80 90 --include-final --test-seed 142
```

Supply explicit `--indices`: the script's default list includes index 100, which is absent from a 100-iteration run. To evaluate only the initial checkpoint and final actor:

```bash
python RL/PPO/ZO_eval.py RL/PPO/reentrant_2_zo_run1 --indices 0 --include-final --test-seed 142
```

Evaluation uses 400 environments with consecutive seeds starting at `--test-seed`, stochastic actions, and `test_T` events per environment from the saved environment configuration. It resets environments for evaluation and currently requires `num_pool: 1`. Choose test seeds disjoint from training and normalization seeds; the current script does not enforce overlap rejection.

Results are written to `evaluation.json` inside the checkpoint directory, including queue, simulated-time, total-queue, and holding-cost statistics. Use `--output path/to/results.json` to choose another file (its parent directory must exist).

Additional options:

- `--check-only`: load and validate selected checkpoints without running rollouts; it does not currently enforce seed separation.
- `--ranking-check`: evaluate for `5 * test_T` events and record cumulative queue statistics at each `test_T` milestone. The default output becomes `ranking_evaluation.json`.
- `--include-final`: load `final_policy.pt` using configuration metadata from `original_000000.pt`; keep both files in the checkpoint directory.

## PPO training

Run the work-conserving PPO baseline with:

```bash
python RL/PPO/train.py WC reentrant_2
```

The first argument selects `RL/policy_configs/WC.yaml`; the second selects `configs/env/reentrant_2.yaml` and must omit `.yaml`. Other available policy configurations are `vanilla` and `vanilla_bc`:

```bash
python RL/PPO/train.py vanilla reentrant_2
python RL/PPO/train.py vanilla_bc reentrant_2
```

Edit the selected policy YAML to configure the actor architecture, behavior cloning, normalization, learning rates, discount factor, GAE, minibatch size, and PPO optimization epochs. The training budget is `training.num_epochs * training.episode_steps * training.actors` total timesteps. The script performs pre-training evaluation and periodic evaluation using the environment's `test_T`; its evaluation batch size is hard-coded to 400 rather than taken from `training.test_batch`.

Outputs go to `RL/PPO/<env_config_name>_PPO/`, including evaluated policies (`policy_eval_000000.pkl`, etc.) and `test_cost_list.json` after training completes. All policy variants for the same environment share this directory, so preserve or rename previous results before running another variant.

## Rule-based benchmarks

For rule-based benchmarks such as c-mu, c-mu-q, MaxPressure, and fluid policies, follow the configuration and execution instructions in the [original QGym repository](https://github.com/namkoong-lab/QGym#how-to-run-experiments). Refer to that repository for the simulator API and general queueing-network setup documentation as well.
