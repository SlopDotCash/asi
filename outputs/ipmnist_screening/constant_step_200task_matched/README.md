# Constant body step on paired 200-task development lives

The [public plan](https://github.com/SlopDotCash/asi/pull/3841) fixed the two configurations, three development seeds, horizon, primary comparison, four secondary windows, and resource accounting before the first run. Each score below averages pre-update accuracy over 200 tasks of 5,000 examples. The arms were rerun in separate processes against the same source, dataset and CPU runtime for a total of six million online steps.

Retain step 0.02 as the preferred development setting under this protocol. The mean paired gain was **+0.004225**, with 3/3 positive seeds. The result is permanently nonpromoting and concerns the tested pair at this horizon.

| Development seed | Step 0.005625 | Step 0.02 | Difference |
| --- | ---: | ---: | ---: |
| 311012 | 0.973184 | 0.977418 | +0.004234 |
| 311013 | 0.973021 | 0.977331 | +0.004310 |
| 311014 | 0.973468 | 0.977600 | +0.004132 |

| Candidate minus control | Tasks 1–50 | Tasks 51–100 | Tasks 101–150 | Tasks 151–200 |
| --- | ---: | ---: | ---: | ---: |
| 311012 | +0.008820 | +0.003480 | +0.002628 | +0.002008 |
| 311013 | +0.008648 | +0.003440 | +0.002592 | +0.002560 |
| 311014 | +0.008604 | +0.002888 | +0.002780 | +0.002256 |
| Mean | +0.008691 | +0.003269 | +0.002667 | +0.002275 |

Both arms used zero weight decay. The two differing resolved hyperparameters were `step_size` and `anneal_floor`. Within each arm they were equal, yielding a constant 0.005625 or 0.02 body step. The retained one-seed result at 311011 remains historical context and is excluded from every table and average here.

[Resource receipts](resources.tsv) give wall time, user and system CPU, maximum RSS in bytes and the runner's internal elapsed time for each shard. Total process wall time was 4,508.8 seconds. Wall time includes interpreter start, data loading and compilation. Other workloads shared the machine during the experiment. [Paired values](paired.tsv) retain nine decimal places. The [maintained merge](summary.json) records the input hashes and full benchmark configuration.

## Reproduction

Use a clean worktree of [the public protocol commit](https://github.com/SlopDotCash/asi/commit/0ba0d77de67ee3035f090bbaa9f6567b9c0b4393). Both the run and the v2 merge bind the exact Git commit. The research environment was installed with Python 3.12 and the repository lock. For each seed run the control then the candidate, reversing the order on seed 311013. In the following shell commands set `arm` to one of the two registry IDs from [the plan](PLAN.md), set `seed`, and replace the cache and fresh output paths.

```sh
uv sync --locked --no-dev --extra research --python 3.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 JAX_PLATFORMS=cpu
/usr/bin/time -l .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$arm" --seed "$seed" --n-tasks 200 --task-length 5000 \
  --noise-mode step --data-home "$cache" --out "$fresh_output"
```

To verify the retained summary, copy the six JSON shards unchanged into `outputs/ipmnist_screening/constant_step_200task_matched/` in the clean protocol worktree. Leave these copies untracked. Run the merge from the repository root with the same runtime settings and use a new output file.

```sh
d=outputs/ipmnist_screening/constant_step_200task_matched
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards "$d"/control-*.json "$d"/candidate-*.json \
  --control-name rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0 \
  --output "$fresh_summary"
```

Replaying the maintained merge against the six retained shards reproduces all summary fields except the new creation timestamp.
