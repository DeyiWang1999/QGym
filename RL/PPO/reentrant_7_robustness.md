# Reentrant-7 arrival-rate robustness

From the repository root, run:

```sh
python RL/PPO/reentrant_7_robustness.py
```

The runner evaluates `cmu`, `cmuq`, `reentrant_7_PPO/policy_eval_000002.pkl`,
and `reentrant_7_split/original_000090.pt` at each of 6/140, 7/140, 8/140,
and 9.5/140. Only entries 0 and 2 of the arrival vector change. The original
NPY file is never written; all other entries retain their original values.

Each combination uses 400 trajectories with environment seeds 142–541 and
100,000 events per trajectory, for 6,400 trajectories total. Combinations
run sequentially, each using at most 48 CPU workers with one Torch thread
per worker. Numerical-library thread counts are also limited to one.

Evaluation reuses `parallel_eval` and its original metric definitions.
Traditional policies use the same priorities, padded linear assignment,
and rounded actions as `Trainer.test_epoch`. PPO retains its saved policy
and normalization; split uses the checkpoint actor and normalization.
Both PPO and split action sampling are seeded with the environment seed for
each trajectory, making their stochastic action sampling reproducible across
repeated runs. This override applies only to this robustness runner; the
standalone evaluation scripts are unchanged.

Outputs default to `RL/PPO/reentrant_7_robustness.json` and the matching CSV.
Both are updated after each completed combination. JSON includes full arrival
vectors, seeds, checkpoint paths, horizon, and all ten original metrics.
CSV has one row per combination, including total time-averaged queue mean/std
and total time-averaged holding-cost mean/std. These standard deviations
describe variation across trajectories, not standard errors.

Use `--output path/to/results.json` to choose another destination. Running again
starts a fresh evaluation and replaces the report as combinations finish.

For a quick execution check covering all 16 combinations:

```sh
python RL/PPO/reentrant_7_robustness.py --smoke-test
```

This uses two seeds and 20 events per trajectory and writes separate
`reentrant_7_robustness_smoke.json` / `.csv` files. Smoke results are not
robustness estimates; use the default invocation for the actual experiment.
