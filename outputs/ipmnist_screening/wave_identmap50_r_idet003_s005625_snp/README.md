# Shrink & Perturb at the first identmap remap: 60-task screen (NEGATIVE)

Status: `development_screening_diagnostic`, nonpromoting. The preregistration is in
[PREREGISTRATION.md](PREREGISTRATION.md) and was committed before any code or run.

## Arm
`rls_head_resid_identmap50_r_idet003_s005625_snp_l08s001` is the parent
`rls_head_resid_identmap50_r_idet003_s005625` with one change: once per detected task, at the
identifier's first remap (`ref_frozen and since_shift == ident_match_at`), each residual body
tensor `w1, b1, w2, b2` becomes `0.8 * theta + 0.01 * fresh`. Here
`fresh ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, keyed by `fold_in(step_key, 0x5A9D)`
(Ash & Adams 2020). The refinement matches at 200 and 2000 do not trigger it.

Off path, as the code implements it:
- With the `snp_shrink` key absent (the parent's registration), no S&P code is traced.
- With the key present at `shrink = 1, sigma = 0`, the live path runs at every event and is
  bitwise the parent. This is pinned by `TestRLSHeadIdentMapShrinkPerturb` on a stream where
  the remap fires. Mutating `sigma * fresh` to `fresh` fails that pin, plus 2 helper tests.

## Tuning (seed 0, 3 tasks; `tune/`)
| cell | mean acc | vs parent 0.827333 |
|---|---|---|
| A: shrink 0.8, sigma 0.01 (`snp_l08s001`) | 0.821733 | -0.00560 |
| B: shrink 0.5, sigma 0.1 (`snp_l05s01`) | 0.807067 | -0.02027 |

A advanced to the screen, as preregistered: the higher cell advances even when both are below
the parent.

## Screen (seeds 0-2, 60 tasks x 5000; `shards/`)
| seed | S&P l08s001 | parent (local) | paired diff | parent (stored #3117) | control (local) | control (stored) |
|---|---|---|---|---|---|---|
| 0 | 0.92089 | 0.93429 | -0.01340 | 0.93446 | 0.77748 | 0.77748 (bit-exact) |
| 1 | 0.92074 | 0.93457 | -0.01383 | 0.93467 | 0.77856 | 0.77856 (bit-exact) |
| 2 | 0.92176 | 0.93467 | -0.01291 | 0.93495 | 0.77724 | 0.77724 (bit-exact) |

Mean paired difference -0.01338 (sd 0.00046), 95% t-CI [-0.01452, -0.01224] (df = 2); **0 of 3 seeds improved**. S&P mean 0.92113, local parent mean 0.93451.

The paired difference is the S&P arm minus the parent re-run locally on this box, and that is
what the bar is judged on. The stored parent shards from #3117 (Python 3.14.4, kernel 7.0) do
not bit-reproduce here (Python 3.13.5). The local parent sits 0.00010-0.00027 below them, the
same offset as in the Autostep wave (PR #3569), and matches that wave's local parent exactly.
The `upgd_w_control` re-measure is bit-exact against its stored shards on all three seeds
(mean 0.77776).

Life profile of the paired difference (mean over seeds):
- task 0: +0.00000 (no remap yet);
- tasks 1-19: -0.01262;
- tasks 20-39: -0.01377;
- tasks 40-59: -0.01438.

The loss starts at the first remap and grows slightly across life.

## Decision
The preregistered bar (mean paired >= +0.002 with at least 2/3 seeds improved) **fails**:
-0.01338, 0/3 seeds. Seeds 3-9 were not run, and held-out seeds 10-19 were not touched.
This agrees with both stated priors:
- ledger #3: perturbation noise is not additive to good conditioning;
- #3125: L2-Init lost -0.01117, because pulling the body toward its init erodes a matured
  feature map.

Even a mild, once-per-task shrink of the body at the remap costs about 1.3 points. In this family
the body's learned features transfer across permutations once the identifier has realigned the
input, so softening them only forces a re-fit.

## Run notes
- **Parent re-runs:** the local parent seeds 0 and 1 first aborted with "screening source
  provenance changed during execution", because a job-list commit moved HEAD mid-run. Seed 2 was
  killed deliberately when available RAM fell to about 2.2 GB (a concurrent unrelated process).
  All three parent seeds were then re-run from scratch on the final source tree (`shards/` holds
  only the complete re-runs). No commits were made while any later run was active.
- **Runtime:** 4 one-thread workers at most, gated on MemAvailable >= 3 GB and no
  cargo-build-sbf. Logs are in `logs/`, the worker in `worker.sh`, and the job lists in
  `jobs_wave*.txt`.
- **Summary:** `summary.json` is the `merge` output over the 9 screen shards (control
  `upgd_w_control`).
