# Preregistration: 200-task, 20-seed confirmation of `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`

This file is committed before any run. Development screening diagnostic; permanently nonpromoting.

## Question
Does the within-task step anneal (`..._anneal_f2t1000`: +0.00226 over 10 seeds at 60 tasks, 10/10,
`wave_identmap50_r_idet003_s005625_anneal/`) hold at the full 200-task horizon against its parent,
the merged development best `rls_head_resid_identmap50_r_idet003_s005625`?

## Protocol (follows `idet003_s005625_confirm_r1/`)
- Source: commit 3c38b025 (the anneal arm) plus this folder only. There is no further source change.
- Runs: 200 tasks x 5000 steps, default step noise, seeds 0-19, `--progress-every 20`, one shard per
  (arm, seed), idempotent `worker.sh`.
- Compute: at most 4 workers, one thread each (OMP/OPENBLAS/MKL=1 plus single-thread XLA CPU flags).
  No new shard starts while cargo-build-sbf, pytest or other JAX jobs run, or while available RAM
  is below about 3 GB.
- Results: shards are never edited.
- No tuning, and no retries with changed settings.

## Seeds, and the 60-task holdout
The lane's confirmation convention (`identmap_star_confirm_r1`, `idet003_s005625_confirm_r1`) uses
development seeds 0-19 at 200 tasks, so this confirmation uses seeds 0-19.
- **Holdout status:** seeds 10-19 were held out of the 60-task anneal screen. They are run here
  under a different protocol (200 tasks), so the 60-task screen's holdout cells (seeds 10-19 at 60
  tasks) remain unrun.
- **Consequence:** seeds 10-19 are no longer unseen for this arm at the 200-task horizon. This is
  stated so that Lantern can object before or after the run.

## Parent shards
- **Rule:** the 20 parent shards in `idet003_s005625_confirm_r1/shards/` (main fd25c476, this host)
  are reused ONLY if all of the following hold:
  1. the parent's traced step (jaxpr) at 3c38b025 is identical to its jaxpr at fd25c476;
  2. the venv package freeze matches the one that produced those shards; and
  3. the first-60-task prefix of those 200-task parent shards equals, bit for bit, the 60-task parent
     shards re-run at 3c38b025 (anneal wave, seeds 0-9).
- **Otherwise:** all 20 parent seeds are re-run here.
- **Reporting:** which path was taken, and why, goes in RESULTS.md.

## Bars, frozen (the lane precedent, identical to `idet003_s005625_confirm_r1`)
| Outcome | Rule |
|---|---|
| CONFIRMED | paired mean ≥ +0.002 vs the parent AND all 20 seeds improve |
| INCONCLUSIVE | paired mean in [+0.0005, +0.002), or ≥ +0.002 without all 20 seeds improving |
| REJECTED | paired mean < +0.0005 |

Always reported:
- per-arm mean ± SD;
- paired mean, SD and 95% t-CI (df = 19);
- seeds improved and the weakest seed;
- the paired diff over the first 60 tasks, and the 60→200 ratio;
- the late-window (tasks 181-200) paired diff.

## Stop rules
- A crashed or partial shard (our own output only) is re-run unchanged from scratch and listed in
  RESULTS.md.
- A NaN or collapsed shard is recorded as is and not re-run.
- No outcome changes the bar, the seeds or the arms.
