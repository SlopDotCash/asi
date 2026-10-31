# Preregistration: wd0 constant-step grid extension 2 (steps 0.03 / 0.04 / 0.05 vs 0.02)

Written and committed before any run (2026-10-31, GMT). Base commit ef85fff8 (head of the
wd0_const_step_extend screen, where step 0.02 beat 0.01 at the grid edge). Line citations are to
`alberta_framework/benchmarks/ipmnist_screening.py` at ef85fff8.

## Question
At `weight_decay 0` with a constant body step (`anneal_floor == step_size`, so
`lr = max(floor, step*rsqrt(1+s/tau))` is constant, L5517-5521), does a body step above 0.02 do better?
At wd0 `param_decay = 1 - step_size*weight_decay` is exactly 1 (L4885, used at L5054-5058), so the
step only scales the gradient step.

## Arms
- Incumbent: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` (step 0.02, floor 0.02), registered at ef85fff8.
- New: `..._s03_anneal_f1t1000_wd0` (0.03), `..._s04_...` (0.04), `..._s05_...` (0.05), identical except `step_size == anneal_floor`.
- Naming: the registry tag is the decimal digits of the step (`s02` = 0.02, `s004` = 0.004 already
  exists from wd0_const_step_retune), so 0.03/0.04/0.05 are tagged `s03`/`s04`/`s05`. The incumbent
  referred to as "0.02" is the `s02` cell.

## Seeds (unseen; 0 hits for seeds 123-133 in shard names across all refs, 12:37:52 GMT)
- Tune (dev) seed: 123. Screen/confirm seeds: 124-133. All at 60 tasks.

## Procedure
0. **Cache pre-flight:** the OpenML cache must match `openml_cache.sha256` (5 files, same count) or the launcher stops before any job (STATUS=PREFLIGHT_FAIL).
1. **Env check (VOID-stop):** `upgd_w_control` seed 0 (60 tasks) must match the committed shard at 3c38b025, and `..._s02_anneal_f1t1000_wd0` seed 113 (60 tasks) must match the committed `wd0_const_step_extend/screen/` shard at ef85fff8, bit for bit on all 60 per-task accuracies. Any mismatch or failed job: VOID, stop.
2. **Tune gate (seed 123):** paired mean per-task accuracy gain of each new cell over s02. Diverged cells (below) are excluded. STOP unless the best eligible cell beats s02 by more than +0.0005. Ties (within 1e-9) go to the smaller step. If every new cell diverges: STOP (all diverged). If the incumbent's tune shard diverges or is missing after one retry: VOID.
3. **Screen (seeds 124-133):** winner vs s02, paired. PASS needs all of: mean gain >= +0.002, at least 7/10 seeds improved, CI95 lower bound > 0 (t = 2.262, df 9), **and no diverged winner seed**. 10-task block gains are reported.
4. **Edge rule:** if 0.05 wins, the result carries an EDGE FLAG (the optimum may lie higher); the verdict is unchanged.

## Divergence rules (fixed in advance; enforced mechanically by `analyze.py`)
A shard is **DIVERGED** if any of:
- it is missing after one retry (crash);
- any per-task accuracy or per-task loss is non-finite (NaN or Inf);
- any per-task accuracy is outside [0, 1];
- any single-task accuracy is below the floor **0.5**.

Justification for 0.5: chance on 10-class MNIST is 0.1. Across all 61 earlier wd0 constant-step
(`f1t1000_wd0`) 60-task shards committed at ef85fff8 (25 in `wd0_const_step_retune/`, 25 in
`wd0_const_step_extend/`, 11 in `wd0_floor_retune/`), the lowest single-task accuracy is 0.7104
(`s005625` seed 75), there are no non-finite accuracies or losses, and the largest per-task loss
is 0.2585. A healthy run therefore sits well above 0.5, while a diverged learner falls towards chance.
The identmap learner has no finite-state guard (L4849-5530), and accuracy is
`argmax(logits) == y` (L5264); `jnp.argmax` ranks NaN as the maximum, so divergence may show as an
accuracy collapse with finite accuracies; hence the floor and the loss check.

Consequences:
- **Tune:** diverged cells are excluded from winning (never a win). All new cells diverged: STOP.
- **Screen:** a diverged winner seed is a FAIL for that seed (counted as not improved, gain set to -1.0) and is flagged. **If any screen seed of the winner diverges, the verdict is FAIL regardless of the mean.**
- A diverged incumbent shard in the tune or the screen: VOID.

## Resources
At most 4 one-thread workers; 3 GB MemAvailable floor at job start and an in-run memguard that pauses the newest worker below 3 GB. No commits while it runs.
