# Preregistration — 200-task, 20-seed confirmation of `rls_head_resid_identmap50_r_idet003_s005625`

Committed before any run. Development screening diagnostic; permanently nonpromoting.

## Question
Does the merged screen best `rls_head_resid_identmap50_r_idet003_s005625` (PR #3117, 0.93469 at
60 tasks, seeds 0-2) hold at the full 200-task horizon against the confirmed development best
`rls_head_resid_identmap50_r` (0.916569 ± 0.001119, `../identmap_star_confirm_r1/`)?

## Protocol (follows `identmap_star_confirm_r1/`)
- Both arms re-run on one runner at one source commit (main fd25c476 + this folder only; no source
  change). 200 tasks x 5000 steps, default step noise, seeds 0-19, `--progress-every 20`.
- One shard per (arm, seed) from `jobs.txt` via idempotent `worker.sh`; at most 4 workers, one thread
  each (OMP/OPENBLAS/MKL=1 plus single-thread XLA CPU flags, verified value-neutral on this host in the
  Autostep screen diagnostics). Shards are never edited; `summary.json` comes from `merge`.
- Seeds 0-19 are development seeds already consumed by the reference confirmation. No tuning; no
  retries with changed settings.

## Reference reproduction
The stored reference shards (`identmap_star_confirm_r1/shards/`) came from Darwin arm64. All 20
reference seeds are re-run here and the per-seed offset (local - stored) is reported. Pairing uses
the LOCAL reference (same-runner rule). An offset never moves a bar.

## Bars — FROZEN (lane precedent, `identmap_star_r1/PREREGISTRATION.md`)
- CONFIRMED: paired mean >= +0.002 vs local `rls_head_resid_identmap50_r` AND all 20 seeds improve.
- INCONCLUSIVE: paired mean +0.0005..+0.002, or >= +0.002 without all 20 seeds improving.
- REJECTED: paired mean < +0.0005.
Always reported: per-arm mean ± SD, paired mean/SD, 95% t-CI (df=19), seeds improved, weakest seed,
late-window (tasks 181-200) paired diff, and 60->200 shrinkage = paired diff over the first 60 tasks of
these shards / paired diff over 200 tasks; the 60-task prefix is also checked against stored 60-task
shards where they exist.

## Stop rules
- A crashed or partial shard (our own output only) is re-run unchanged from scratch and listed in RESULTS.md.
- A NaN/collapsed shard is recorded as-is, not re-run.
- No new shard starts while a cargo-build-sbf process runs or while available RAM < ~3 GB.
- No outcome changes the bar, the seeds or the arms.
