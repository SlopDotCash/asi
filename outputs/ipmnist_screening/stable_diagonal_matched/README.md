# Stable diagonal covariance development screen

Run the two stable diagonal variants named in the campaign index on 60 tasks of 5,000 steps, step noise, fresh processes and development seeds 0, 1 and 2. Pair each with the full-matrix r02 parent to isolate covariance structure. Also rerun the campaign's leading constant-step zero-decay dense arm to evaluate practical opportunity cost; its other settings differ, so differences against it do not isolate the head. Keep the data, arm-independent stream, source, runtime and CPU thread settings fixed. Report average pre-update accuracy and differences on all seeds, per-shard process time and peak resident memory. The primary question is whether global excitation pooling improves the stable diagonal control versus independent denominators. A consistent loss versus the matched full parent ends this screen. Even a win would require longer matched confirmation before any further development. This subsystem screen is permanently nonpromoting.

The campaign index requested fresh matched measurements for both stable controls. The protocol above was recorded locally before this run. It used previously consumed development seeds.

## Results

Each entry is mean per-task pre-update accuracy. Paired differences use seed-matched observations.

| Seed | Dense parent | Diagonal | Pooled diagonal | Campaign lead | Pooled minus diagonal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.948713 | 0.948047 | 0.947660 | 0.964383 | -0.000387 |
| 1 | 0.948657 | 0.947433 | 0.947527 | 0.964147 | 0.000093 |
| 2 | 0.948003 | 0.947217 | 0.946920 | 0.963953 | -0.000297 |

| Matched contrast | Mean difference | Positive seeds |
| --- | ---: | ---: |
| pooled minus diag | -0.000197 | 1/3 |
| diag minus parent | -0.000892 | 0/3 |
| pooled minus parent | -0.001089 | 0/3 |
| parent minus lead | -0.015703 | 0/3 |

Both stable heads lost to the full-matrix parent on all three seeds. Pooling improved the diagonal head on one seed. The screen therefore ends without promotion. This result applies to the tested head variants and horizon; it does not resolve the separate low-rank or blockwise designs.

The maintained merge's summary binds the input shard bytes and the source, runtime and dataset provenance. The resource table records complete process time (including interpreter start, loading and JIT compilation), process CPU time and peak RSS from macOS time -l. Runs were serial on a shared host. These measurements describe this run; they are unsuitable for a portable speed ranking.

## Reproduction

Start from the commit identified by the shards' source provenance, with Python 3.12 and the locked research environment. Use a fresh output directory outside the checkout for runs. From the checkout root, run the following for seeds 0, 1 and 2 in each configuration in the table. The examples assume shell variables for the arm, seed, data cache and destination.

```sh
uv sync --locked --no-dev --extra research --python 3.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 JAX_PLATFORMS=cpu
/usr/bin/time -l .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$arm" --seed "$seed" --n-tasks 60 --task-length 5000 \
  --data-home "$data_cache" --out "$output_file"
```

| Arm | Registry configuration |
| --- | --- |
| parent | `rls_head_resid_identmap50_r_idet003_s0031640625_r02` |
| diag | `rls_head_resid_identmap50_r_idet003_s0031640625_r02_diaghead_stable` |
| pooled | `rls_head_resid_identmap50_r_idet003_s0031640625_r02_gdiag_stable` |
| lead | `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` |

Recompute the summary from the checkout root using the published paths. Select a fresh output path.

```sh
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/ipmnist_screening/stable_diagonal_matched/parent-0.json \
    outputs/ipmnist_screening/stable_diagonal_matched/diag-0.json \
    outputs/ipmnist_screening/stable_diagonal_matched/pooled-0.json \
    outputs/ipmnist_screening/stable_diagonal_matched/lead-0.json \
    outputs/ipmnist_screening/stable_diagonal_matched/parent-1.json \
    outputs/ipmnist_screening/stable_diagonal_matched/diag-1.json \
    outputs/ipmnist_screening/stable_diagonal_matched/pooled-1.json \
    outputs/ipmnist_screening/stable_diagonal_matched/lead-1.json \
    outputs/ipmnist_screening/stable_diagonal_matched/parent-2.json \
    outputs/ipmnist_screening/stable_diagonal_matched/diag-2.json \
    outputs/ipmnist_screening/stable_diagonal_matched/pooled-2.json \
    outputs/ipmnist_screening/stable_diagonal_matched/lead-2.json \
  --control-name rls_head_resid_identmap50_r_idet003_s0031640625_r02 --output "$new_summary"
```
