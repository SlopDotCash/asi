# Fixed rank-eight covariance on 12-task development lives

Stop advancing this fixed basis under the planned accuracy criterion. On the three matched development seeds, the fixed orthonormal projection changed pre-update accuracy by **-0.144683** against the dense parent (paired sample SD 0.019593). The [public protocol](https://github.com/SlopDotCash/asi/pull/3845) specified a mean loss of at most 0.002 and an individual loss of at most 0.004 before the first run. All nine fresh shards used the same source, CPU runtime and dataset, for a total of 540,000 online updates.

| Development seed | Dense parent | Fixed rank 8 | Rank 8 minus parent | Contemporary dense lead |
| --- | ---: | ---: | ---: | ---: |
| 191171 | 0.929933 | 0.806233 | -0.123700 | 0.932500 |
| 191172 | 0.927267 | 0.764767 | -0.162500 | 0.930567 |
| 191173 | 0.927500 | 0.779650 | -0.147850 | 0.930267 |
| Mean | 0.928233 | 0.783550 | -0.144683 | 0.931111 |
| Seed sample SD | 0.001477 | 0.021007 | 0.019593 | 0.001212 |

The projected and matched parent arms differed only in their covariance setting. The groups in the fixed basis share a readout direction across their hidden features. The experiment measures this specific basis and horizon. Adaptive low-rank directions remain an independent question. The current lead provides context under different body settings. This evidence is permanently nonpromoting.

## Resource measurements

Each head uses the same stored 151-by-10 readout. The dense parent stores 22,801 float32 covariance entries (91,204 bytes), and the rank-eight arm stores 64 entries (256 bytes). The fixed grouping is computed from coordinate indices. In addition to the covariance update, each example pools the 150 features once and lifts the gain once. At each update the candidate computes an 8-by-8 Joseph-form square root using QR.

| Arm | Wall seconds summed across three processes | Mean peak RSS (MiB) | Largest peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| parent | 168.21 | 2603.8 | 2619.4 |
| rank8 | 199.37 | 2516.7 | 2602.8 |
| lead | 179.32 | 2499.0 | 2606.3 |

[Per-process receipts](resources.tsv) include interpreter startup, JAX compilation, data loading and the 12-task run. Wall time on a shared host is descriptive. Each example used one update with the registered protocol's body, observation schedule and readout.

## Reproduce

Use a clean worktree of the [implementation and plan commit](https://github.com/SlopDotCash/asi/commit/da095ff8c8bfcd5f20d3dcf7aab5b9bdce47521b) and the locked Python 3.12 research environment. For each of the three seeds in [the plan](PLAN.md), run its three registered IDs in cyclic order. The recorded order was parent, rank8, lead on seed 191171, rotated left by one on each subsequent seed.

```sh
uv sync --locked --extra research --extra dev --python 3.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 JAX_PLATFORMS=cpu
/usr/bin/time -l .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$arm" --seed "$seed" --n-tasks 12 --task-length 5000 \
  --noise-mode step --data-home "$cache" --out "$fresh_shard"
```

To check the [summary](summary.json), copy the nine retained shard files into their original relative directory in the clean plan worktree as untracked files. Use a fresh output path under the same runtime. The six-task slope window is a secondary merge field; the decision uses the planned mean over all 12 tasks.

```sh
d=outputs/ipmnist_screening/projected_rank8_matched
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards "$d"/parent-*.json "$d"/rank8-*.json "$d"/lead-*.json \
  --control-name rls_head_resid_identmap50_r_idet003_s0031640625_r02 --slope-window 6 --output "$fresh_summary"
```

The maintained merge ran twice and reproduced every field except the new creation timestamp. All nine manifest byte lengths and SHA-256 digests independently matched the retained shards.
