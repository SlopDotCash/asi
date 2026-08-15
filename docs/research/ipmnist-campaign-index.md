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
| [`SOTA_LANDSCAPE_2026.md`](../../outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md) | Bounded 2026-08-02 web survey, not a systematic or current external ranking |
| [`AUDIT.md`](../../outputs/ipmnist_screening/AUDIT.md) | Historical audit of the then-existing campaign records and cross-runner proxy discrepancy |

Use “current development leader” for the latest arm and “local reproduced
UPGD-W reference” for the stored published-configuration comparator. Historical
`BEATS-SOTA` labels mean only that an arm cleared that local reference; they are
not external SOTA findings.

## Artifact-schema boundary

Every stored screening shard and summary in this append-only campaign is a v1
record. The implementation keeps those historical files readable and permits a
homogeneous v1-only merge into a v1 summary. It refuses to mix v1 and v2 shards,
and v1 has none of the current source, data, or derivation bindings. Treat every
v1 command and result in the append-only runbook as a historical development
operation, never as promotion-grade evidence.

New CLI runs write `alberta.ipmnist_screening.shard.v2`. Before execution, the
CLI requires a clean Git checkout and binds the commit/tree, every tracked file
under `alberta_framework/`, `pyproject.toml`, `uv.lock`, the canonical
materialized MNIST arrays, and selected Python/JAX/package/device/process
configuration. It verifies the source, runtime, and data bindings again before
exclusive publication. A v2 merge requires one common source, dataset,
runtime, protocol, and noise contract, a present control arm, and unique
arm/seed shards; the output records the exact bytes and paths of its inputs.
The CLI verifies those inputs and the derivation source/runtime again before
writing the v2 summary.

These are strong consistency checks, not authentication. The JSON has no
signature or independently issued execution receipt; the active environment is
not proven to match `uv.lock`, and already-loaded code or mutable process state
is not independently attested. The source selection and materialized bytes do
identify the canonical MNIST input expected by the CLI, but a library caller
can still construct self-declared mappings. There is no standalone strict
summary reload/reconstruction API in this schema version. Input paths are
stored as supplied and may therefore depend on the derivation working
directory. Wall-clock totals remain operational diagnostics, not a cross-host
speed comparison. The entire campaign remains permanently nonpromoting.

V2 validates the shards actually supplied; it does not declare or prove that a
wider candidate-arm universe or preregistered seed set is complete. Comparisons
are paired on the seeds shared with the named control, and confirmation
eligibility requires at least two shared seeds with positive per-seed effects.

All new runs must target new output paths; publication refuses replacement of
an existing shard or summary. Use the live CLI help for the exact syntax:

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
