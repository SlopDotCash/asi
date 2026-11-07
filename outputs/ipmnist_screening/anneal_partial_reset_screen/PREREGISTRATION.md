# Preregistration: partial anneal reset on anneal_f2t1000 (development screen)

Written before any run of the new arms. Base commit 3c38b025. Parent: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`.

## Hypothesis
The parent restarts its within-task body-step anneal at `s = 1` on every detected shift, so the step jumps back to 0.005625 at every boundary, and on any false or duplicate trigger as well. If we restart the anneal clock at `floor(f * s)` instead, the step after a shift is lower (with s ≈ 5000 at a true boundary, f=0.25 starts at 0.005625/sqrt(2.25) ≈ 0.00375 and f=0.5 at 0.005625/sqrt(3.5) ≈ 0.00301, both close to the constant-step optimum of 0.0031640625 from #3121/#3122). A misfire then also costs less. The anneal-peak negative pointed this way: a higher post-shift step helped tasks 1-15 and lost more in every later block.

## Mechanism, exactly as the code implements it
- New optional key `anneal_reset_frac` = f. It is valid only together with `anneal_floor` and requires 0 ≤ f < 1. Otherwise it raises ValueError (NaN is already rejected by the generic finite-number guard).
- The state gains an int32 `anneal_clock` field, initialised to -2**30 like `since_shift`.
- **Key absent (the off path):** `anneal_clock` is set to the new `since_shift` every step and is never read. The anneal reads `state.since_shift` as before. The numerics are the parent's verbatim.
- **Key present:** the anneal reads `state.anneal_clock`, decoded with the same -2**30 offset, as `s` in `max(floor, step*(1+s/tau)^-1/2)`. When the detector fires (`mean(shifted) >= ident_frac`), the new clock is `1 + floor(f * s_prev)`, where s_prev is the decoded pre-step clock. On other steps it is `anneal_clock + 1`. With **f = 0** the reset gives 1, the same as `since_shift`'s reset, so the result is bit-exactly the parent (pinned in a test).
- The identifier (`since_shift`, match times, remap, reference/post statistics) is untouched in every case.

## Arms
`..._anneal_f2t1000_pr25` (f=0.25) and `..._pr50` (f=0.5). All other hyperparameters are identical to the parent (pinned in a test). The two values bracket the constant-step optimum, as described above. They are reported separately.

## Seed plan and gates (60 tasks × 5000 steps)
1. **Env check:** re-run `upgd_w_control` seed 0 and the parent seed 0 at 60 tasks on this branch. Both must equal the stored 3c38b025 shards bit-for-bit (`wave_identmap50_r_idet003_s005625_anneal/shards`). This also checks that the state-field addition leaves the parent unchanged. On failure: VOID, stop, no automatic re-run.
2. **Parent:** reuse the stored #3580 shards (612491db `anneal_f2t1000_confirm_r1/shards`, seeds 0-13) verbatim, with SHA256SUMS. Analysis uses `per_task_accuracy[:60]`; a 60-task run is a bit-exact prefix of the 200-task run, verified on seeds 0-9.
3. **Tune, seed 0 only:** pr25 and pr50 are reported separately. CHOSEN = the larger paired gain, which must be > 0. If neither beats the parent: STOP (negative).
4. **Screen, seeds 1-3:** the paired mean must be > 0, else STOP (negative).
5. **Confirm, seeds 4-13.** Bars, evaluated on seeds 4-13 only: paired mean ≥ +0.002, at least 7/10 seeds improved, and 95% CI lower bound > 0 (t = 2.262, df = 9). All three must pass.

## Prior
About 8%. The stored shards and logs record no detector-trigger counts (see "Detector frequency" in the plan), so the false-trigger rationale stays unmeasured. The anneal-peak block pattern is weak evidence in favour, and the small f4t1000 effects are weak evidence against.

## Disclosures
- This experiment was chosen after #3586 (f4t1000 negative) and the anneal-peak tune-gate negative, and after the parent's results on seeds 0-13 had already been seen.
- Seeds 10-13 were seen in #3580's parent shards.
- This is a development screen only; it is nonpromoting.
