# Local hillclimb checks, 2026-11-07

These paired development checks are permanently nonpromoting. The IPMNIST
screens measure pre-update accuracy. The reference-life checks measure raw
cumulative reward over 4,000 Switching events and 5,000 RiverSwim
events. The local reference checks did
not run the random and oracle normalization arms, so no local scorecard or
qualification verdict follows. Resource receipts distinguish persistent
numeric state from peak memory and timing telemetry.

## IPMNIST 12-task screen

All nine lives completed. Every life used 12 tasks × 5,000 examples on
the same arm-paired schedules for seeds 311001–311003.

| Body step at zero weight decay | Mean pre-update accuracy |
|---|---:|
| Constant 0.02 | 0.93212220 |
| Constant 0.005625 | 0.91584997 |
| 0.005625 annealed to half | 0.90676664 |

The paired gain against the stronger constant control was +0.01627223
on average, positive on all three seeds. The paired gain against the
annealed control was +0.02535556, also positive on all three seeds. A
separate initial state measurement counted 2,451,845 numeric array bytes
for each arm, including model parameters, learner state and the noise key.
See [raw shards and summaries](ipmnist/).

The PR's retained 200-task analysis also recomputed byte-identically from
its 40 shards. It compared 0.02 against the annealed arm, with a paired gain
of +0.007665, 20/20 positive, and 95% CI [+0.007588, +0.007741].
See the [recomputed verdict](ipmnist/retained-3646-recomputed.txt).

## IPMNIST 200-task pair

Both arms completed 200 tasks × 5,000 examples on fresh paired seed
311011. Constant step 0.02 averaged 0.97729498 online accuracy, and
constant step 0.005625 averaged 0.97336498, a gain of 0.00393000. The
four 50-task block gains were +0.00812800, +0.00323200, +0.00243600 and
+0.00192400. The gain narrowed across the life and remained positive in
the final block. This is a one-seed check. See the
[paired shards and summary](ipmnist-200/).

## SwitchingTwoState 4,000-event comparison

All 12 lives completed on paired seeds 118331–118333.

| Arm | Mean cumulative reward | Initial and final numeric bytes |
|---|---:|---:|
| Prototype step 0.20 | 3554.00 | 217 |
| Prototype step 0.10 | 3406.67 | 217 |
| Prototype step 0.05 | 3030.33 | 217 |
| Differential SARSA | 3231.33 | 108 |

Prototype step 0.20 won all three pairs against each comparator, with
mean gains of +147.33 over step 0.10, +523.67 over step 0.05, and +322.67
over differential SARSA. See the [paired summary](reference/switching-summary.json).

## RiverSwim 5,000-event smoke

All 12 lives completed on paired seeds 118331–118333. Both optimistic
initializations improved on their own default in all three pairs and
reached the high-reward end in all three lives.

| Arm | Mean cumulative reward | Initial and final numeric bytes |
|---|---:|---:|
| Discounted SARSA, Q0 = 10 | 3006.42 | 220 |
| Discounted SARSA, default | 1513.39 | 220 |
| Differential SARSA, initial average reward = 1.0 | 2995.47 | 204 |
| Differential SARSA, default | 22.68 | 204 |

The head-to-head reward differences for discounted minus differential
optimism were +79.905, −39.050 and −8.005. The two optimistic arms remain
separate candidates for a future matched full-horizon control panel.
See the [raw-reward summary](reference/riverswim-summary.json).

## Reproduction

`inputs.txt` binds the source trees, locked dependencies, and downloaded
archives. IPMNIST ran at PR [#3646](https://github.com/SlopDotCash/asi/pull/3646)
head `c8c1050e4d69a41ddf1950cd49ca1cae0e331d08`. The reference checks ran from
a detached composition of PRs [#3664](https://github.com/SlopDotCash/asi/pull/3664)
(including [#3656](https://github.com/SlopDotCash/asi/pull/3656)),
[#3655](https://github.com/SlopDotCash/asi/pull/3655), and
[#3652](https://github.com/SlopDotCash/asi/pull/3652), merged in that order.
The locally composed commit IDs are provenance records. Verify the tree
hash after reconstructing it with the PR heads below. Metadata such as
creation time, wall time, and the local commit ID can vary across reruns.

| PR | Pinned head |
|---|---|
| #3664 | `7d586c9a3690d9c17167fbe69419ba3321697d2e` |
| #3655 | `13e08153f0cb5d45727f76f478c304490a4e3b9b` |
| #3652 | `05c9b46b257f10f3cabe8d69154601cc1ce2894b` |

From an ASI clone, fetch the PR heads and create separate source worktrees.

```bash
git fetch https://github.com/SlopDotCash/asi.git \
    refs/pull/3646/head refs/pull/3664/head refs/pull/3655/head refs/pull/3652/head
git worktree add --detach "$ip_source" c8c1050e4d69a41ddf1950cd49ca1cae0e331d08
git worktree add --detach "$ref_source" 7d586c9a3690d9c17167fbe69419ba3321697d2e
git -C "$ref_source" merge --no-ff --no-edit 13e08153f0cb5d45727f76f478c304490a4e3b9b
git -C "$ref_source" merge --no-ff --no-edit 05c9b46b257f10f3cabe8d69154601cc1ce2894b
git -C "$ref_source" rev-parse 'HEAD^{tree}'
```

The run used Python 3.12.14, JAX/JAXLIB 0.11.0, and NumPy 2.5.1 on macOS
arm64 CPU. In each source checkout install the locked environment with
`uv sync --locked --python 3.12 --extra dev`. Set the following variables
from that checkout.

```bash
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
```

`run.py` consolidates the thin local wrappers around the unchanged runners
from the pinned worktrees. Use the IPMNIST checkout for the first two
lanes and the composed reference checkout for the last two. Set `bundle`
to the absolute path of this directory, and `rerun` to a fresh destination.

```bash
mkdir -p "$rerun/data"
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz \
    -o "$rerun/data/train-images.gz"
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz \
    -o "$rerun/data/train-labels.gz"
for seed in 311001 311002 311003; do
  for arm in s02_anneal_f1t1000_wd0 s005625_anneal_f1t1000_wd0 s005625_anneal_f2t1000_wd0; do
    .venv/bin/python "$bundle/run.py" ipmnist "$arm" "$seed" \
        --data "$rerun/data" --out "$rerun/ipmnist"
  done
done
for arm in s02_anneal_f1t1000_wd0 s005625_anneal_f1t1000_wd0; do
  .venv/bin/python "$bundle/run.py" ipmnist-200 "$arm" 311011 \
      --data "$rerun/data" --out "$rerun/ipmnist-200"
done
```

The IDX archive is an alternate retrieval path for the same training arrays.
The runner compares both normalized array hashes against the retained #3646
receipt before any run.

```bash
for seed in 118331 118332 118333; do
  for arm in prototype_step020 prototype_step010 prototype differential_sarsa; do
    .venv/bin/python "$bundle/run.py" switching "$arm" "$seed" --out "$rerun/reference"
  done
  for arm in sarsa_q0 differential_sarsa_rbar0 sarsa differential_sarsa; do
    .venv/bin/python "$bundle/run.py" riverswim "$arm" "$seed" --out "$rerun/reference"
  done
done
```

For each lane, run `python3.12 "$bundle/compare.py" <lane>` to recompute
the descriptive summary of the saved shards. Add `--root "$rerun"` to
summarize a rerun. The commands launch a fresh process per arm and seed.
The prerecords, raw shards and recomputed summaries are grouped by lane.
