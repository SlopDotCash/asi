# IPMNIST campaign index

Last reconciled 2026-11-07.

This is the mutable index for the active IPMNIST development-screening and
development-confirmation campaign. It is **permanently nonpromoting**. It
tests one plasticity/conditioning subsystem and does not support an integrated
ASI-agent, robotics, scientific-evidence, state-of-the-art, or Alberta Plan
completion claim.

Maintained ceiling, frontier, and rule-discovery reproduction commands live in
the [IPMNIST campaign-tools runbook](../runbooks/ipmnist-maintained-tools.md).
Programs stored beside historical `outputs/` artifacts remain provenance
records and are not the maintained implementation.

## Current stored record

The lead candidate is
`rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0`. It uses a
constant body step of 0.02 and zero weight decay. In the retained 200-task
confirmation it gained **0.007665** mean paired online accuracy over the
annealed zero-decay arm on **20/20** seeds (95% CI [0.007588, 0.007741]).
The comparator, per-seed results, and predeclared gates are in the
[200-task analysis](../../outputs/ipmnist_screening/wd0_const_step_200task/ANALYSIS.txt).

The [constant-step 0.005625 arm](../../outputs/ipmnist_screening/wd0_const200/ANALYSIS.txt)
also beat the annealed zero-decay comparator in a separate 200-task cohort.
The retained analyses used different seeds. The [fresh 12-task screen](../../outputs/hillclimb_local/2026-11-07/ipmnist/SUMMARY.json)
paired both constant steps directly. Step 0.02 gained **0.016272** mean
pre-update accuracy and led on all three seeds. This short screen supports
the candidate for the full-length comparison, with the same dataset bytes
and separate resource receipts for each arm. A [fresh 200-task pair](../../outputs/hillclimb_local/2026-11-07/ipmnist-200/SUMMARY.json)
on seed 311011 also favored step 0.02, with mean accuracy 0.977295 versus
0.973365 (+0.003930). Its four consecutive 50-task block gains were
+0.008128, +0.003232, +0.002436, and +0.001924. The one-seed comparison
establishes direction on that life. A matched multi-seed 200-task comparison
against constant step 0.005625 remains open.

The older cross-runner
[`proxy_validation.json`](../../outputs/ipmnist_screening/proxy_validation.json)
records `proxy_validated=false`: three UPGD control prefixes do not match their
batched full-horizon references bitwise. That finding does not erase the newer
same-runner paired RLS description, but it prevents a general cross-harness
bitwise-reproduction claim.

## Record authority and supersession

The JSON summaries own measurements. This index owns which stored summary is
current. The documents under `outputs/ipmnist_screening/` are append-only
chronological records and must not be rewritten merely because later work
superseded their status language.

| Record | How to read it now |
|---|---|
| [`summary_rls_head_confirm.json`](../../outputs/ipmnist_screening/summary_rls_head_confirm.json) | Earlier paired RLS confirmation against the UPGD-derived control |
| [`summary_rls_head.json`](../../outputs/ipmnist_screening/summary_rls_head.json) | RLS screening record that selected the confirmed arm |
| [`RUNBOOK.md`](../../outputs/ipmnist_screening/RUNBOOK.md) | Chronological operations log; earlier “queued” states can be superseded by later sections |
| [`FINAL_REPORT.md`](../../outputs/ipmnist_screening/FINAL_REPORT.md) | Historical accumulated report; its “final best” language predates RLS confirmation |
| [`publication_runs/RESULTS.md`](../../outputs/ipmnist_screening/publication_runs/RESULTS.md) | Historical pre-RLS publication-run record, not the current campaign leaderboard |
| [`CEILING_ANALYSIS.md`](../../outputs/ipmnist_screening/CEILING_ANALYSIS.md) | Pre-RLS diagnostic analysis; its “record” and “measured state of the art” language is historical |
| [`SOTA_LANDSCAPE_2026.md`](../../outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md) | Bounded 2026-08-02 web survey, not a systematic or current external ranking; its BiMU sentence omits the higher MESU entry from the same paper table |
| [`AUDIT.md`](../../outputs/ipmnist_screening/AUDIT.md) | Historical audit of the then-existing campaign records and cross-runner proxy discrepancy |

Use “lead development candidate” for the 0.02 arm and “local reproduced
UPGD-W reference” for the stored published-configuration comparator. Historical
`BEATS-SOTA` labels mean only that an arm cleared that local reference; they are
not external SOTA findings.

Use the mutable [protocol-aware comparison landscape](sota-landscape.md) for
the current external review and the correction to the dated output survey.

## Historical identmap development waves

These retained 60-task waves use seeds 0–2. The original shard and summary
bytes carry their historical source and dataset identities. The current
registry can validate their configurations and recompute their summaries.
Any new experiment must measure the current control under its own matched
protocol. The historical waves do not change the lead development candidate.

| PR | Retained summary |
|---|---|
| #3106 | [Ungated body with rematches at 50/200/2000](../../outputs/ipmnist_screening/wave_identmap50_nogate/summary.json) |
| #3107 | [Ungated body with rematches at 200/500/2000](../../outputs/ipmnist_screening/wave_identmap200_nogate/summary.json) |
| #3109 | [Ungated body with a 0.03 identifier trigger](../../outputs/ipmnist_screening/wave_identmap50_nogate_idet003/summary.json) |
| #3110 | [Ungated 0.03 identifier trigger and body step 0.0075](../../outputs/ipmnist_screening/wave_identmap50_nogate_idet003_s0075/summary.json) |
| #3111 | [Ungated 0.03 identifier trigger and body step 0.005625](../../outputs/ipmnist_screening/wave_identmap50_nogate_idet003_s005625/summary.json) |
| #3114 | [Gated body momentum 0.9](../../outputs/ipmnist_screening/wave_identmap50_r_mom9/summary.json) and [0.99](../../outputs/ipmnist_screening/wave_identmap50_r_mom99/summary.json) |
| #3116 | [Gated body with a 0.02 identifier trigger](../../outputs/ipmnist_screening/wave_identmap50_r_idet002/summary.json) |
| #3118 | [Gated body with a 0.03 identifier trigger and body step 0.00421875](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s00421875/summary.json) |
| #3119 | [Gated body with a 0.04 identifier trigger and body step 0.00421875](../../outputs/ipmnist_screening/wave_identmap50_r_idet004_s00421875/summary.json) |
| #3120 | [Gated body with a 0.03 identifier trigger and body step 0.0031640625](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625/summary.json) |
| #3121 | [Gated body with a 0.03 identifier trigger and body step 0.002373046875](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s002373046875/summary.json) |
| #3122 | [Gated body step 0.0031640625 and rls ridge 0.1](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r01/summary.json) |
| #3123 | [Gated body step 0.0031640625 and rls ridge 0.03](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02/summary.json) |
| #3124 | [Gated rls ridge 0.03 with head reset fraction 0.03](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_preset003/summary.json) |
| #3125 | [Gated r02 body L2-to-initialization](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_l2init/summary.json) |
| #3126 | [Gated r02 decayed SGD readout](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_sgdhead/summary.json) |
| #3127 | [Historical per-coordinate diagonal head](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_diaghead/summary.json) |
| #3128 | [Historical block-local head](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_blockhead/summary.json) |
| #3129 | [Historical pooled diagonal head](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_gdiag/summary.json) |
| #3131 | [Historical low-rank experiment with an indefinite covariance recurrence](../../outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_lowrank/summary.json) |
| #3144 | [Gradual transitions with a 64-step ramp](../../outputs/ipmnist_screening/gradual64_r2/summary.json) |

The original `diaghead` and `gdiag` arm IDs are available for artifact replay
with their bound source commits. New runs use the `diaghead_stable` and
`gdiag_stable` suffixes. These controls update the diagonal covariance with
positive terms and use scaled excitation to avoid float32 overflow. Both
stable variants are unmeasured and require fresh matched runs.

The archived `blockhead` ID uses block-local innovation denominators. Its
new `blockhead_stable` control projects the dense update onto four covariance
blocks using one global innovation denominator. It stores square root factors
and updates them with Joseph-form QR, requiring O(G*b^2) state and O(G*b^3)
arithmetic for G blocks of width b. This variant is unmeasured and needs a
fresh matched run with resource accounting. Its scores cannot be inferred
from the archived local-denominator experiment.

The #3131 low-rank records are retained for source-bound artifact replay. Dropping
an old downdate's off-diagonal entries while retaining its diagonal can make the
covariance indefinite. With ridge 0.03 and eight stored directions, an eleven-step
nonnegative-feature witness gives innovation −680.828 before a square root. The
historical accuracy comparison therefore cannot support its claimed conclusion
about low-rank head capacity. A live low-rank control requires a covariance design
that preserves positive semidefiniteness and a fresh matched experiment.

## Artifact-schema boundary

The append-only campaign contains v1, v2, and v3 records. New `run` invocations
write v2 shards for step and pool noise and v3 shards for gradual transitions.
The live merge command recomputes summaries within a single shard schema. It
requires the declared control and the same observed seed set for every arm.
A v1 merge carries a legacy description. V2 and v3 add source, dataset,
runtime, arm, and input-shard bindings.

V2 binds the exact post-cast array bytes, arm fields and callback descriptors,
a package-wide Python disk snapshot plus available checkout lock documents,
selected runtime/JAX configuration, the RNG contract, invocation, and
input-shard bytes. Its content digest is an unkeyed self-hash: it detects
accidental inconsistency and supports exact merge comparison, but it is not a
signature, execution receipt, MNIST-semantic attestation, or proof that the
recorded run occurred. It also does not attest that the complete loaded module
and global state matches disk or that the active environment matches the bound
lock document. The entire campaign remains permanently nonpromoting.

The seed contract is exact for each supplied arm, but v2 does not declare or
prove that a wider candidate-arm universe is complete. V2 summaries record each
shard path exactly as supplied to the merge command and recheck the file binding
before publication. Relative paths therefore remain working-directory-dependent,
while absolute paths remain machine-location-dependent; neither form is a portable
artifact locator. The merge host is not attested, and mixed invocation or
progress-logging settings make aggregated wall-clock totals diagnostic only,
not a speed comparison.

New runs must target new output paths; the current writer refuses replacement
of an existing shard or summary. Use the live CLI help for the exact syntax:

```bash
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run --help
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge --help
```

## Historical path names

Some append-only records retain paths that existed when they were written:

- `CONTINUAL_LEARNING_THEORY.md` is now
  [`docs/research/ipmnist-theory.md`](ipmnist-theory.md).
- `CONTINUAL_LEARNING_EVIDENCE.md` is now
  [`docs/evidence/methodology.md`](../evidence/methodology.md).
- `NEW_DIRECTIONS.md` was retired. Its stored diagnostics remain historical
  development records; the [negative-results ledger](../evidence/negative-results.md)
  owns any concluded interpretation.

When a later summary supersedes the RLS confirmation, append the new measurement
under `outputs/ipmnist_screening/` and update this index. Do not edit an old
output record in place.
