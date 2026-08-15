# Forager benchmark

This integration evaluates Alberta agents in the partially observable,
continuing-control testbed introduced by *Forager: a lightweight testbed for
continual learning with partial observability in reinforcement learning*
([arXiv:2605.01131](https://arxiv.org/abs/2605.01131)).

The paper contributes a testbed and evaluation protocol, not a new agent
algorithm. Alberta uses the authors' official JAX environment at runtime and
keeps the environment state opaque to learning policies.

Companion documents for this lane:

- `docs/runbooks/foragax-open-screen.md` — the strict OCI harness for the frozen
  nonpromoting FOV development screens.
- `docs/archive/forager-comparator-audit.md` — a read-only audit of the frozen
  matched-current campaign (see the
  matched-current section below).
- `docs/archive/historical-forager-reconstruction.md` — the separate reconstructed
  paper-era NumPy environment family.
- [`docs/status.md`](docs/status.md), especially Step 6 — the canonical
  evidence-status statement; no Forager performance result is registered in
  the scientific evidence manifest.

## Install

Use the optional dependency group:

```bash
.venv/bin/python -m pip install -e '.[forager]'
```

This installs `continual-foragax==0.55.0` (imported as `foragax`) and
`gymnax==0.0.9`. The unrelated PyPI distribution named `foragax` is not used.

The environment repository currently publishes no license file or license
metadata. Its source is therefore not vendored into this Apache-2.0 repository;
the integration is an optional runtime dependency. Version 0.55.0 was also
released after the arXiv v1 submission. Results must be labelled a
**Foragax 0.55.0 reproduction**, not an exact reconstruction of the
submission-time environment.

## Quick run

Run Alberta and a uniform-random lower control on one short paired seed:

```bash
alberta-forager-benchmark \
  --preset relearning \
  --steps 10000 \
  --seeds 0 \
  --agent alberta \
  --agent random \
  --output outputs/forager/smoke.json
```

Both methods use bounded-memory `jax.lax.scan` runners. The output contains
per-seed curves, bootstrap summaries, paired differences, exact environment and
agent configurations, package/device/git provenance, timing split into setup,
compilation, and execution, and a payload hash.

The Alberta method in this benchmark is `alberta_horde_ac`: a nonlinear Horde
actor-critic with Autostep and ObGD. It learns once from each transition without
replay or task-boundary signals. Its state contains the flattened visible
image, optional public cue, visible-image channel means, previous action,
scaled previous reward, and a biased multi-timescale reward-trace bank.
Those extra channel means and traces make it an Alberta method, not a
paper-faithful DQN/PPO baseline.

The stationary field-of-view study also exposes
`alberta_causal_map`, a learned cognitive-map/world-model candidate:

```bash
alberta-forager-benchmark \
  --preset field_of_view \
  --steps 10000 \
  --seeds 1000000,1000001 \
  --agent causal-map
```

This variant assigns the initial location an arbitrary origin, dead-reckons
only from its own actions under the task's public 15×15 toroidal movement
semantics, and projects raw egocentric one-hot observations into a fixed-size
relative map. It learns reward signs and magnitudes per observed channel from
its own collections. Respawn scheduling begins from one channel-agnostic retry
prior, backs off after an estimated-ready cell is observed empty, and learns
per-channel collection-to-reappearance intervals. It retains separate exact
consecutive-visibility diagnostics, schedules from all interval-censored
samples to avoid policy-dependent exact-sample selection bias, and corrects
early estimates through observed-empty retry backoff. Known learned-negative
cells are excluded from targets and immediate moves whenever a safe move
exists.

State schema `alberta.forager_causal_map_state.v5` stores each lower and upper
endpoint mean as an exact int32 floor quotient plus a non-negative remainder:
`sum = floor * count + remainder`. The update streams visible samples without
x64, and scheduling begins at `upper_floor + (upper_remainder > 0)`, the exact
outward integer ceiling. Exact delays retain their own quotient/remainder sum;
their float32 mean is derived canonically from it after every streamed Welford
update, with M2 recentered on that canonical mean. This avoids both raw-second-
moment cancellation and batch/checkpoint-dependent rounding. Exact-delay
variance and a safety factor of at least float32 one may raise that schedule.
`maximum_respawn_delay` remains an explicit operational ceiling: if the learned
outward bound exceeds it, the schedule is capped and is no longer a formal
upper bound beyond that ceiling.
Checkpoints bind both the `threefry2x32` implementation and JAX's
`jax_threefry_partitionable` mode in immutable state, so restoring,
serializing, or stepping after split-mode drift fails closed. The compiled
runner also roots environment randomness with an explicit `threefry2x32`
typed key; changing JAX's default PRNG implementation cannot silently change
the environment trajectory. Pure entrypoints reject non-real observations,
non-finite/non-binary images, invalid rewards, and bool, negative, out-of-range,
or non-scalar seeds before lossy conversion. Cognitive maps are capped at 4096
cells so pathological custom shapes cannot trigger near-int32 planner bounds
or unbounded device allocation.

The policy does not receive `ForagerAgentContext`, global position, object or
reward grids, biome/task labels, evaluator `info`, or environment time. Its
finite state is serializable and its single-seed, `vmap`, and strict
`lax.map` runners use the same pure transition. The implementation is specific
to the stationary `ForagaxTwoBiomeLarge-v1` color-observation task; selecting it
for relearning, Big-v5, delayed rewards, or custom environment kwargs fails
closed. It is a candidate method, not a SOTA claim without a frozen disjoint
tuning record and matched-seed confirmation against the strongest admissible
learning baseline.

## Resumable Alberta variant matrices

`.venv/bin/python -m alberta_framework.benchmarks.forager_matrix` runs the strict,
resumable Alberta-only matrix harness. Manifest schema `2.2` remains frozen to
the original `alberta_horde_ac` and `alberta_causal_map` kinds. Schema `2.3`
adds `alberta_rtu_rtrl`, backed by `RTURTRLForagerConfig`; its run records use
the distinct implementation label `alberta_rtu_rtrl_ac` so it cannot be
relabelled as the fixed-GRU Horde variant. Schema `2.4` is required for the
RTU core's opt-in adaptive-ObGD fields (`adaptive_obgd`, `beta2`, and
`epsilon`). Schema `2.3` rejects those fields even when they spell their
defaults, preserving its canonical configuration hashes.

An RTU entry may supply partial, strictly typed nested overrides. Omitted
values normalize to the complete configuration before hashing:

```json
{
  "rtu": {
    "kind": "alberta_rtu_rtrl",
    "selection_group": "policy",
    "config": {
      "core": {
        "hidden_size": 8,
        "encoder_width": 8,
        "output_width": 8
      },
      "freeze_after_steps": null,
      "features": {
        "reward_trace_decays": [0.9, 0.99, 0.999]
      }
    }
  }
}
```

This object is the `variants` map inside a full schema-`2.3` manifest. Both
`strict` (`lax.map`) and `vmap` seed batches use the existing compiled RTU
runner. With `metric_evidence_mode: "raw_reward_npz_v2"`, each seed produces a
canonical reward/biome-regret sidecar; resume verification checks its digest,
ZIP/NPY encoding, array inventory, and independently recomputes every reported
metric. The matrix configuration, RTU algorithm metadata, disjoint agent RNG
namespace, source snapshot, batches, summaries, and selection ranking are all
hash-bound. Host and snapshot-subprocess executions remain explicitly
unsealed; neither schema `2.3` nor `2.4` promotes scientific evidence by
itself.

An adaptive variant changes the core block to schema `2.4`, for example
`{"adaptive_obgd": true, "beta2": 0.999, "epsilon": 1e-8}`. Its raw actor
and critic second-moment trees persist across episode boundaries and count
toward the matrix's per-batch RTU memory bound. Exact default adaptive
coefficients are omitted from normalized configuration output; the opt-in flag
remains hash-bound.

A retained host/snapshot tuning report is historical development evidence
only and cannot authorize evaluation, even if its source snapshot, installed
wheel, host inventory, and candidate OCI build metadata verify. The
`validate_verifier_issued_tuning_envelope` API defines the fail-closed boundary
for a future immutable OCI evaluation adapter, but the host matrix runner does
not call it and rejects an envelope embedded in a matrix manifest. A usable
envelope must come from an external trust/revocation verifier and jointly bind
the tuning report file and payload, raw metric evidence, source tree and
archive, read-only content-addressed OCI source mount, runtime profile, and
environment RNG schedule to the evaluation executor. Candidate OCI metadata
and normalized runtime fields are not execution attestation.

`core.rtrl_taylor_correction` is an opt-in, hash-bound approximation and is
disabled by default. It corrects only the parameter-diagonal component;
simultaneous multi-parameter motion can retain first-order mixed-Hessian
staleness and can worsen the sensitivity error. It does not make the
moving-parameter sensitivity exact. Matrix validation applies a tighter
persistent-state resource bound when this additional trace bank is enabled.

The repository's broader `PrototypeAgent` is deliberately excluded from this
initial runner. Its primitive-action credit and option lifecycle now have
focused regressions, but no Prototype configuration has been tuned or frozen
for Forager, and enabling its hand-specified options would not test the
learned-state/feature-lifecycle gap that motivates this lane. It should enter
only as a separately declared method after a causal observation contract,
resource budget, development sweep, and held-out protocol are fixed.

## Paper protocols

The packaged presets and primary statistics are:

| Study | Foragax ID | Horizon | Tuning | Primary statistic |
|---|---|---:|---:|---|
| Field of view | `ForagaxTwoBiomeLarge-v1` | 500,000 × 30 seeds | 10,000 × 5 disjoint seeds | Mean over final 10% of unadjusted EMA curve (decay 0.999, sampled every 100 rewards) |
| Relearning | `ForagaxSquareWaveTwoBiome-v11` | 10,000,000 × 30 seeds | 1,000,000 × 10 disjoint seeds | Lifetime mean of adjusted EMA, α=0.001 |
| Unending tasks | `ForagaxBig-v5` | 10,000,000 × 30 seeds | 1,000,000 × 10 disjoint seeds | Final adjusted EMA, α=1e-5 |

SquareWave has a 500,000-step waveform and changes reward regime every
250,000 steps. A frozen-at-5M run is a separate relearning ablation, not the
default continuously learning condition:

```bash
alberta-forager-benchmark \
  --preset relearning \
  --paper-protocol \
  --agent alberta \
  --freeze-after-steps 5000000 \
  --output outputs/forager/relearning_alberta_frozen_5m.json
```

Evaluation commands for the continuously learning conditions are:

```bash
alberta-forager-benchmark \
  --preset field_of_view --paper-protocol \
  --agent alberta --agent random \
  --output outputs/forager/fov9.json

alberta-forager-benchmark \
  --preset relearning --paper-protocol \
  --agent alberta --agent random \
  --output outputs/forager/relearning.json

alberta-forager-benchmark \
  --preset unending --paper-protocol \
  --agent alberta --agent random \
  --output outputs/forager/unending.json
```

`--paper-protocol` applies the evaluation horizon and seeds 0–29. It does not
claim that Alberta's hyperparameters have completed the paper's separate
tuning stage. Artifacts record this explicitly as
`tuning_stage_executed: false` and report individual evaluation-conformance
checks. Run tuning on the disjoint seed offset 1,000,000 before promoting a
full-protocol result.

Inspect a protocol or the complete baseline manifest without running:

```bash
alberta-forager-benchmark --preset relearning --protocol-only
alberta-forager-benchmark --preset relearning --list-paper-baselines
```

## Matched literature comparison workflow

The paper compares DQN, plasticity interventions, PPO, reward-trace methods,
DRQN, and RTU-PPO. These architectures are not silently approximated with
Alberta's MLP actor-critic. Run them from the authors' MIT-licensed
[`continual-foragax-agents`](https://github.com/steventango/continual-foragax-agents)
repository, then import each seed archive:

```bash
alberta-forager-benchmark \
  --preset relearning \
  --steps 10000000 \
  --seeds 0,1 \
  --agent alberta \
  --reference-npz RTU-PPO:0:/path/to/rtu/data/0.npz \
  --reference-npz RTU-PPO:1:/path/to/rtu/data/1.npz \
  --reference-source-repository https://github.com/steventango/continual-foragax-agents \
  --reference-source-commit 6c3175729377e634460ed41621fed7de06432cf8 \
  --reference-config RTU-PPO:experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/RealTimeActorCriticMLP.json \
  --output outputs/forager/alberta_vs_rtu_unattested_diagnostic.json
```

Direct `--reference-npz` imports are hashed diagnostics but are unattested and
excluded from paired comparisons. Supplying repository, commit, config, or the
original `<seed>.npz` filename improves provenance but does not change that
trust class. `--attest-reference-protocol` is deprecated and always rejected;
attestation cannot be asserted manually.

An official import can be attested only through a completed, externally
endorsed schema-`1.4` single- or batch-run manifest:

```bash
alberta-forager-benchmark \
  --preset relearning \
  --steps 10000000 \
  --seeds 0,1 \
  --agent alberta \
  --reference-manifest /path/to/completed-schema-1.4-batch/manifest.json \
  --output outputs/forager/alberta_vs_verified_rtu.json
```

The importer re-verifies the endorsement, manifest and payload hashes,
source/config/environment and agent-access bindings, immutable output tree,
logs, and every referenced archive. Schemas `1.1`–`1.3` are archival only and
are rejected for verified use. Paired comparisons additionally require
identical seed sets, environment contracts, interaction budgets, metric
transformations (EMA decay/bias correction and final window), and internally
consistent method configs; mixed or duplicate runs fail closed.
Use one repeated `--reference-config NAME:CONFIG_PATH` per method when
importing several baselines in the same artifact.
Repository and commit defaults are preset-specific: the paper-era FOV study
uses `steventango/forager-agents@696b3a06…`, while relearning and Big-v5 use
`steventango/continual-foragax-agents@6c317572…`.

The historical FOV repository records collector data in SQLite and runs the
pre-Foragax NumPy environment. Those historical curves are unpaired orientation
evidence for a current Foragax 0.55.0 Alberta run, even when their statistic is
recomputed exactly. A paired FOV claim requires rerunning every compared method
and Alberta in the same executable environment; relabelling a historical run as
a Foragax result is invalid.

Official Search baselines use privileged global state and must be labelled so
they cannot be confused with admissible learning agents. The historical FOV
repository does not write raw NPZ rewards: its SQLite `reward` rows are already
EMA-smoothed and subsampled. The raw-NPZ importer rejects that repository
instead of double-smoothing the curve or inventing an interaction count.

Historical FOV databases have a separate fail-closed API:

```python
from pathlib import Path

from alberta_framework.benchmarks import (
    LegacyFOVSQLiteRunSpec,
    import_legacy_fov_sqlite,
)

run = import_legacy_fov_sqlite(
    LegacyFOVSQLiteRunSpec(
        agent="DQN",
        path=Path("/results/ForagerTwoBiomeLarge/DQN-9/results.db"),
        config_path=Path("/checkout/experiments/forager-two-biome-large/"
                         "ForagerTwoBiomeLarge/DQN-9.json"),
        run_index=0,
        stored_seed=0,
        expected_config_agent="DQN-9",
        expected_aperture_size=9,
        expected_stored_seeds=tuple(range(30)),
        # Pin these when publishing an artifact:
        expected_database_sha256="<64 lowercase hex characters>",
        expected_config_sha256="<64 lowercase hex characters>",
    )
)
```

The importer checks SQLite integrity and the exact PyExpUtils v2 schema,
matches the database hyperparameter row to the flattened config JSON, checks
the declared seed set, and requires one finite unique value at every frame
`0, 100, ..., 499900`. It records separate run-index, stored-seed, and
effective-seed fields plus database, config, and flattened-config hashes.
Only `fov_last_10pct_ema_auc` is finite: it is the mean of the final 10% of
the stored EMA curve. The importer never smooths the values again. The runtime
is labelled `historical_numpy_forager`, source is `official_fov_sqlite`, and
paired comparison with a current Foragax result is rejected explicitly.

No exact raw seed results or numeric tables accompany the paper. The CLI
therefore includes clearly labelled, figure-digitized central estimates for
orientation only. Notable relearning estimates are RTU-PPO ≈1.30, DQN Simple
Memory ≈0.97, DQN+cReLU ≈0.95, and privileged Search Oracle ≈1.59 on lifetime
mean EMA. Big-v5 estimates are PPO ≈0.089, RTU-PPO ≈0.118, and privileged
Search Oracle ≈0.214 on final EMA. These are not acceptance thresholds and
cannot replace seed-level bootstrap comparisons.

The paper-time Big-v5 “PPO Simple Memory” config is ambiguous: it nested its
reward-trace flag where the checked-in code did not read it. The manifest and
digitized target retain that warning.

## Matched-current campaign (frozen v1)

The matched-current pipeline is the lane's only path toward an eventual
matched-panel comparison on the exact stationary task
(`ForagaxTwoBiomeLarge-v1`, aperture 9, color, 499,712 steps). Its stages, in
order, are:

1. **Qualification** — `alberta-forager-matched-qualification`
   (`forager_matched_qualification.py`) binds sources, configurations, the
   OCI runtime, and per-candidate probes.
2. **Open tuning** — `alberta-forager-matched-campaign`
   (`forager_matched_campaign.py`, subcommands `prepare`/`run`/`verify`/
   `status`) executes the 21 inferential candidates on ten fresh tuning seeds
   (2,300,001–2,300,010; 210 cells).
3. **Seal and held-out evaluation** — `forager_matched_seal.py` freezes the
   selection; `forager_matched_sealed_evaluation_campaign.py` runs the
   6-candidate × 30-seed held-out grid; `forager_matched_final_analysis.py`
   and `forager_matched_statistics.py` produce the final analysis.

Execution status in this tree:

- Current source uses candidate-universe schema v2 (SHA-256 `6a9315cb…`) and
  has no renewed qualification or open-campaign artifact. It requires a fresh
  qualification and a new output namespace before execution; the `2c3b214c`
  roots below are immutable historical v1 artifacts and are source-incompatible
  with the current builder.
- Historical v1 qualification **completed**:
  `outputs/forager/matched_current_qualification_2c3b214c_v1` (candidate
  universe SHA-256 `2c3b214c…`, status
  `structurally_qualified_external_trust_resolution_required`,
  classification `content_only_unendorsed_nonpromoting`).
- Historical v1 open tuning is **prepared but unexecuted**:
  `outputs/forager/matched_current_open_tuning_2c3b214c_v1` holds the frozen
  manifests, but its `runs/` and `completions/` directories are empty — zero
  of the 210 cells have run.
- The sealed stages are **implemented and tested but never executed**: no
  seal bundle, sealed-evaluation artifact, or final-analysis bundle exists
  under `outputs/`. The sealed campaign is exposed as
  `alberta-forager-matched-sealed-evaluation`; seal and final analysis remain
  module-only.

Every in-tree authority is `content_only_unendorsed_v1`. The external trust
resolver that authority-bearing paths require does not exist in this
repository, the fixed-action RNG-parity receipt
(`outputs/forager/rng_parity_live_qualification_v1_execution/receipt.json`)
remains `content_complete_external_executor_receipt_unverified` with
`promotion_authorized: false`, and no `forager_matched` claim is registered
in the evidence manifest. In-tree code alone therefore cannot produce a
promoted matched-current result. `docs/archive/forager-comparator-audit.md`
records comparator provenance and the frozen protocol's statistical scope limits.

## Development receipts

`outputs/forager/rtu_rtrl_500k_dev4/receipt.v1.json` preserves an exact
completed four-seed, 500,000-step RTU-RTRL GPU development run. Its
`fov_last_10pct_ema_auc` mean is 1.550 with sample SD 0.324 and range
1.167–1.936. The receipt pins its captured output and run-time direct-source
hashes and is tested for exact bytes and recomputed summary statistics.
`outputs/forager/rtu_rtrl_500k_dev4/capture-correction.v1.json` corrects the
original receipt's mislabeled seven-line capture digest while preserving the
original receipt byte-for-byte.

It is deliberately nonpromoting. The seeds are consumed open-development
seeds, the variant was not selected under a preregistered protocol, no
admissible paired baseline or held-out interval exists, and the source hashes
do not cover the complete import/runtime closure. It must not be compared with
paper curves as if it were a matched SOTA result.

An unsealed four-seed DQN development comparator does exist. Its corrected
receipt is
`outputs/forager/dqn_fov_500k_dev_seeds2000001_2000004_reconciled/receipt.v1.json`;
it supersedes, without modifying, the original read-only
`DEVELOPMENT_MANIFEST.json`. Recomputed through the public importer, DQN's FOV
metric mean is `1.2190922828452653` (sample SD
`0.19442361562596436`) versus RTU's `1.5499997668875873` (sample SD
`0.32388877090181417`). The descriptive paired RTU-minus-DQN difference is
`+0.3309074840423221` (sample SD `0.2159564083561187`), positive for 4/4
consumed seeds.

That comparison is permanently nonpromoting. The DQN comparator was configured
after RTU output was available; the runtime envelopes, representations,
parameter/replay/memory budgets, and update work are unmatched; RTU has no raw
reward trace; the four seeds were open and unregistered; and this DQN
configuration is not an exact frozen paper baseline. It supports no
inferential, causal, speed, SOTA, or Alberta Plan claim.

## Fairness boundary

Learning policies receive only the visible image, public Big-v5 cue, and their
own causal action/reward history. They never receive position, environment
state, biome ID/regret, hidden time/switch labels, or `info["rewards"]`.
The runner passes full context only to policies explicitly marked privileged.

`OracleSearchForagerAgent` is an in-tree, blocking-aware privileged diagnostic.
It is intentionally named `privileged_blocking_search` in results: it is not
the paper's exact Search Oracle and not a mathematical upper bound. Import the
official Search Oracle archives for paper comparisons.

The audited upstream `rtu_ppo.py` reuses one derived key for policy action
sampling and the corresponding environment transition. Exact-upstream PPO and
RTU-PPO are therefore descriptive shared-key reproductions, not admissible
paired baselines for Alberta's dedicated environment-key runners.
`forager_rtu_ppo_rng_isolation.py` defines a separately labelled correction
for the matched-current suite. It accepts only upstream source SHA-256
`e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24`
at commit `9710f60fa30da5badc451ad7ce3ff296d5070830`, applies eight exact
single-occurrence byte replacements, validates the derived AST, and freezes
the derived source SHA-256 as
`70bbdd0943d82570c1dc0d28494cf93f9c1b208ef67b3a547585fe5897cdf409`.
The derived runner keeps the canonical reset/transition environment split
chain and roots all agent randomness under a separate fixed namespace. It must
be named an **isolated-RNG reproduction**, never unaltered upstream code.

The derivation alone does not create scientific evidence. A matched executor
must bind both source identities read-only, prove the fixed-action
environment-key schedule, use one qualified runtime profile, preserve native
artifacts and canonical raw traces, and validate the complete candidate/seed
receipt before admitting a paired comparison.

Foragax 0.55.0 has known differences from the paper prose, including generated
reward periods and normalization. The executable 0.55.0 semantics define this
reproduction. Exact-mode runs verify the installed package payload against the
audited 0.55.0 wheel and record the release wheel hash plus both expected and
observed package-tree hashes.

## Validation

Run the focused integration suite:

```bash
.venv/bin/python -m pytest -q -o addopts="" \
  tests/test_forager_benchmark.py \
  tests/test_causal_map_forager.py \
  tests/test_forager_rtu_rtrl.py \
  tests/test_forager_matrix.py
```

It checks causal state construction, disjoint agent/environment RNG namespaces,
privilege isolation, host/scan lifecycle parity, bounded compiled runners,
causal-map reward/respawn learning, state serialization, chunk and batch
parity, negative-cell avoidance, official NPZ metric import, strict paired
statistics, task-specific protocols, and the installed official environment
API. The matched-current campaign machinery has its own contract suite in
`tests/test_forager_matched_*.py`; passing tests are mechanism validation
only and never promote evidence.
