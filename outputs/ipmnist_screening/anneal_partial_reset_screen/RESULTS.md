# RESULTS: partial anneal reset on anneal_f2t1000: FAIL (mean bar)

Preregistration: b1c5e283. Branch cut from 3c38b025 (not rebased), so it stays comparable with the stored #3580 parent. 60 tasks × 5000 steps. Paired against the #3580 parent shards (612491db), using `per_task_accuracy[:60]` (see `parent_reused/SHA256SUMS`).

## Env check: PASS
On this branch, `upgd_w_control` seed 0 and the parent seed 0 both reproduce the 3c38b025 shards bit-for-bit over 60/60 tasks. This includes the new `anneal_clock` state field.

## Tune (seed 0 only), each f reported separately
| f | paired gain | tasks improved | blocks 1-15 / 16-30 / 31-45 / 46-60 |
|---|---|---|---|
| 0.25 (pr25) | +0.000183 | 31/60 | −0.000267 / +0.000120 / +0.000533 / +0.000347 |
| 0.5 (pr50) | +0.000807 | 47/60 | −0.000587 / +0.000520 / +0.001587 / +0.001707 |

CHOSEN: pr50.

## Screen (seeds 1-3): GO
seed1 +0.000387, seed2 +0.000877, seed3 +0.000370; mean +0.000544 > 0.
Mean blocks: −0.000511 / +0.000422 / +0.000796 / +0.001471.

## Confirm (seeds 4-13); the bars are evaluated here only
| seed | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|
| gain | +0.000640 | +0.000597 | +0.000727 | +0.000653 | +0.000707 | +0.000623 | +0.000410 | +0.000640 | +0.000510 | +0.000627 |

- Mean is +0.000613 (SE 0.000029) and the 95% t CI (df=9, t=2.262) is [+0.000547, +0.000680].
- 10/10 seeds improved.
- Mean blocks: −0.000609 / +0.000427 / +0.001111 / +0.001525.

| Bar | Result |
|---|---|
| mean ≥ +0.002 | **FAIL** (+0.000613) |
| ≥ 7/10 improved | PASS (10/10) |
| CI lower bound > 0 | PASS (+0.000547) |

**VERDICT: FAIL.** The effect is real and consistent, but it is about a third of the preregistered effect-size bar.

## Observations (not claims)
- The gain is negative in tasks 1-15 and grows in every later block (about +0.0015 in tasks 46-60). A longer horizon might show a larger mean, but that was not preregistered here and would need its own preregistration (for example 200 tasks on unseen seeds).
- The trigger frequency is still unmeasured: shards and logs carry no detector counts.

## Run notes and disclosures
- Times, UK time (GMT), 2026-10-31: prereg commit b1c5e283 00:33:45; launch 1 00:48:49; re-launch after fix 27bf898a (01:04:13) at 01:04:17; finished 01:13:13.
- Coupled weight decay (not separated): with a runtime step, the body update uses `decay = jnp.where(lr == jnp.float32(step_size), jnp.float32(param_decay), 1.0 - lr * weight_decay)` (ipmnist_screening.py lines 5049-5053 at 6bbfdccc; weight_decay = 0.01, step_size = 0.005625). So the per-step shrink 1 - lr·0.01 moves with the step: after a true boundary, the parent restarts at 1 - 5.625e-5, while pr50 restarts at about 1 - 3.0e-5 (lr ≈ 0.00301 at s ≈ 2500). The partial reset changes both the early post-shift step and the effective weight decay, so the +0.000613 cannot be attributed to the step alone. No arm separated the two.
- On the first launch (00:48:49 UK time (GMT)), the launcher's `stage()` overwrote the shell variable that held the chosen config, so the screen analysis was called without its argument. It crashed and the launcher logged "STOP after screen". That log is kept at /tmp/pr_partial_reset_launch_run1.log, which is not committed.
- This was a harness bug, not a gate decision: the three screen shards had completed. I fixed `stage()` to use local variables (27bf898a) while no runs were active and re-ran the idempotent launcher (01:04:17 UK time (GMT)). It skipped every existing shard, re-checked the env check from the stored shards (PASS), re-derived the tune and screen identically, and ran only the confirm seeds.
- This experiment was chosen after #3586 and the anneal-peak negative. The parent's seed 0-13 results were already seen, including seeds 10-13.
- Development screen only; nonpromoting.
