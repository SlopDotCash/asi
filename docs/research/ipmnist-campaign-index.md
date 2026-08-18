# IPMNIST campaign index

This is the mutable index for the active IPMNIST development-screening and
development-confirmation campaign. It is **permanently nonpromoting**. It
tests one plasticity/conditioning subsystem and does not support an integrated
ASI-agent, robotics, scientific-evidence, state-of-the-art, or Alberta Plan
completion claim.

## Current stored record

The latest stored paired confirmation names
`rls_head_resid_l1_preset005` as the current development leader over the paired
`sigma0_shiftnorm_d099` control. Read the means, uncertainty, seed coverage, and
per-seed differences directly from
[`summary_rls_head_confirm.json`](../../outputs/ipmnist_screening/summary_rls_head_confirm.json).
The [mechanistic synthesis](ipmnist-theory.md) explains the arm and its bounded
interpretation.

This is a descriptive same-runner comparison. The summary declares
`development_only=true` and `scientific_promotion_allowed=false`; it is not
bound to the current source/runtime or an independently authenticated execution
lifecycle. Any new A/B must capture its current implementation identity and
remeasure its control in the same runner instead of treating the stored control
mean as a live guarantee.

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
| [`summary_rls_head_confirm.json`](../../outputs/ipmnist_screening/summary_rls_head_confirm.json) | Current paired development confirmation |
| [`summary_rls_head.json`](../../outputs/ipmnist_screening/summary_rls_head.json) | RLS screening record that selected the confirmed arm |
| [`RUNBOOK.md`](../../outputs/ipmnist_screening/RUNBOOK.md) | Chronological operations log; earlier “queued” states can be superseded by later sections |
| [`FINAL_REPORT.md`](../../outputs/ipmnist_screening/FINAL_REPORT.md) | Historical accumulated report; its “final best” language predates RLS confirmation |
| [`publication_runs/RESULTS.md`](../../outputs/ipmnist_screening/publication_runs/RESULTS.md) | Historical pre-RLS publication-run record, not the current campaign leaderboard |
| [`CEILING_ANALYSIS.md`](../../outputs/ipmnist_screening/CEILING_ANALYSIS.md) | Pre-RLS diagnostic analysis; its “record” and “measured state of the art” language is historical |
| [`SOTA_LANDSCAPE_2026.md`](../../outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md) | Bounded 2026-08-02 web survey, not a systematic or current external ranking; its BiMU sentence omits the higher MESU entry from the same paper table |
| [`AUDIT.md`](../../outputs/ipmnist_screening/AUDIT.md) | Historical audit of the then-existing campaign records and cross-runner proxy discrepancy |

Use “current development leader” for the latest arm and “local reproduced
UPGD-W reference” for the stored published-configuration comparator. Historical
`BEATS-SOTA` labels mean only that an arm cleared that local reference; they are
not external SOTA findings.

Use the mutable [protocol-aware comparison landscape](sota-landscape.md) for
the current external review and the correction to the dated output survey.

## Artifact-schema boundary

Every stored screening shard and summary in this append-only campaign is a v1
record. The runbook's historical v1 validation and merge examples are therefore
not commands for the current implementation. Strict merge now accepts only
`alberta.ipmnist_screening.shard.v2`, requires an explicit `--expected-seeds`
set for every arm, requires the declared control to be present, and writes a
`summary.v2` that rebinds and recomputes all input shards. V1 shards cannot be
strictly merged; `--legacy-v1-quarantine` is available only to parse them for
the nonpromoting proxy audit.

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
prove that a wider candidate-arm universe is complete. Summaries use canonical
absolute shard paths and are therefore machine-location-bound. Their merge host
is not attested, and mixed invocation/progress-logging settings make aggregated
wall-clock totals diagnostic only, not a speed comparison.

New runs must target new output paths; the current writer refuses replacement
of an existing shard or summary. Use the live CLI help for the exact v2 syntax:

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
