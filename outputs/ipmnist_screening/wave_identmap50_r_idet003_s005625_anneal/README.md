# wave_identmap50_r_idet003_s005625_anneal: within-task body step anneal

Preregistration: `PREREGISTRATION.md` (committed first, at 94f41e69). Arm and pin: f5dae3cb.
Lane status: development_screening_diagnostic, nonpromoting.

**Arm:** `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`
- Body step: `lr = max(0.0028125, 0.005625 * (1 + since_shift/1000) ** -0.5)`.
- Every other setting is the parent's (`rls_head_resid_identmap50_r_idet003_s005625`, merged on
  main at fd25c476).
- Switched off (`anneal_floor == step_size`), the live path reproduces the PARENT bit-exactly. This
  is pinned on the offset stream, where the identifier fires.

## Tune (seed 0, 3 tasks; mean accuracy)
| cell | mean acc |
|---|---|
| parent | 0.82733 |
| f2t1000 | 0.81406 (advanced) |
| f4t1000 | 0.81313 |
| f2t250 | 0.80920 |
| f4t250 | 0.79640 |

Every cell scored below the parent over 3 tasks. As preregistered, the best cell still advanced.
The 3-task result points the opposite way from the 60-task one (see the per-task columns below):
the anneal costs early in the run and pays late, which matches ledger #21's warning about short
diagnostics.

## Screen and extension (60 tasks × 5000; local parent re-run on this branch)
| seed | parent | anneal f2t1000 | diff | first-15 diff | last-15 diff |
|---|---|---|---|---|---|
| 0 | 0.93429 | 0.93632 | +0.00203 | -0.00300 | +0.00535 |
| 1 | 0.93457 | 0.93707 | +0.00250 | -0.00244 | +0.00615 |
| 2 | 0.93467 | 0.93711 | +0.00243 | -0.00255 | +0.00508 |
| 3 | 0.93457 | 0.93664 | +0.00207 | -0.00263 | +0.00541 |
| 4 | 0.93541 | 0.93780 | +0.00239 | -0.00293 | +0.00600 |
| 5 | 0.93558 | 0.93755 | +0.00197 | -0.00299 | +0.00580 |
| 6 | 0.93546 | 0.93738 | +0.00192 | -0.00307 | +0.00547 |
| 7 | 0.93470 | 0.93733 | +0.00263 | -0.00252 | +0.00589 |
| 8 | 0.93501 | 0.93738 | +0.00237 | -0.00328 | +0.00653 |
| 9 | 0.93494 | 0.93717 | +0.00223 | -0.00292 | +0.00531 |

- **Screen (seeds 0-2):** mean +0.00232, 95% t-CI [+0.00169, +0.00295], 3/3 seeds improved. This
  cleared the bar (≥ +0.002 and ≥ 2/3), so seeds 3-9 were run.
- **10-seed decision (seeds 0-9):** mean **+0.00226**, 95% t-CI [**+0.00208, +0.00243**], **10/10**
  seeds improved. It passes the bar (≥ +0.002, CI lower bound > 0).
- **Held out:** seeds 10-19 were not run.
- **Horizon:** the gain is negative over the first 15 tasks and about +0.0057 over the last 15. It
  grows with horizon, so a 200-task confirmation is the natural next step.
- **Pairing base:** the local parent seeds 0-2 match the S&P wave's parent bit for bit (0.93429 /
  0.93457 / 0.93467) under an identical venv. The parent was still re-run here, as preregistered,
  because the code commit differs.
- **Control `upgd_w_control` re-measure (seeds 0-2):** 0.77748 / 0.77856 / 0.77724, mean 0.77776.
  This is bit-exact with the stored shards.

## Re-pair check
#3118 and #3123 were rechecked at handoff (2026-10-30, about 18:30 GMT). Both are still open and
unmerged, and main is still fd25c476, so no re-pairing was needed.

## Interruption note
The agent session was interrupted once, while it was waiting on wave 3. The detached runs were not
affected: all 23 shards completed and are intact (each has 60 tasks), and no shard was partial or
re-run. Two leftover wait loops belonging to the agent were killed by exact PID. No result was
edited.

## Files
- Job lists: `jobs_wave1.txt` (tune, parent seeds 0-2 and control), `jobs_wave2.txt` (anneal seeds
  0-2) and `jobs_wave3.txt` (anneal and parent, seeds 3-9).
- Results: `tune/`, `shards/`, `logs/`.
