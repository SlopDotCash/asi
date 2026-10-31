# Preregistration: within-task body step anneal keyed on since_shift

This file is committed before any implementation or run on this branch.

## Parent
`rls_head_resid_identmap50_r_idet003_s005625` is merged on main (fd25c476). Its 200-task, 20-seed
confirmation gave +0.026599 paired, with 20/20 seeds improved.

## Single new variable
The parent's constant body step (`step_size = 0.005625`) becomes a runtime step that anneals within
each detected task:

    s  = steps since the identifier's last detected shift
    lr = max(anneal_floor, step_size * (1 + s / anneal_tau) ** -0.5)     (float32)

- `s` is the identmap state's `since_shift` as it stands *before* the current step. The counter
  restarts at 1 on the step after the raw-stream trigger fires (the parent's own detector; the
  `ident_reset_frac = 0.03` trigger). Before the first detected shift, the counter starts at
  -2**30, so `s = since_shift + 2**30`, which is the number of steps since the stream began.
- Identifier remaps at 50/200/2000 do NOT restart `s`.
- `lr` replaces `step_size` in the gated, decayed SGD body update (`_gated_sgd`), both in the step
  term and in the decay factor `1 - lr * weight_decay`.
- The utility gate, the RLS head (λ = 1, ridge 1, P-reset 0.05), the identifier and every other
  hyperparameter are exactly the parent's.

The new variable is the pair `(anneal_floor, anneal_tau)`, which together form one schedule
mechanism.

Rationale: the open step chain (#3118 → #3123) keeps gaining as the constant step shrinks, which
points to a constant-step noise floor. The theory doc names within-task convergence as the
remaining headroom. An anneal keeps the parent's step at the boundary and lowers it late in the
task. Prior: ledger #21 warns that 60-task gains can overstate 200-task gains for early-life
mechanisms, but this one acts late in each task.

### Switched-off check, exactly as the code implements it
- **The `anneal_floor` key is absent** (the parent's registration). The factory takes the parent
  code path unchanged, with no runtime step traced.
- **Live path with `anneal_floor = step_size`** (keys present, any `anneal_tau`). The schedule is
  computed every step and passed to the body update as a float32 array. Because
  `step_size * (1 + s/τ) ** -0.5 <= step_size`, the max returns exactly `float32(step_size)`. The
  decay factor is taken as `float32(1 - step_size * weight_decay)` (the parent's constant) whenever
  `lr == float32(step_size)`, and computed as `1 - lr * weight_decay` otherwise.

On this path the arm must reproduce the **parent** (not the control) bit-exactly. A pin test runs
this live path on a stream where the detector fires and the identifier remaps (the per-pixel
offset stream with `ident_match_at = 10`, the same one the S&P pin used), and asserts bitwise
equality of per-task accuracy and loss with the parent. I will confirm that a deliberate break
(the floor set 1 ulp-scale below the start step, i.e. `anneal_floor = 0.99 * step_size`, with a
short τ) makes the bitwise assertion fail.

## Protocol
1. **Tuning:** seed 0, 3 tasks × 5000 samples. The preregistered grid has 4 cells, each paired
   against a local parent re-run on seed 0 at 3 tasks:
   - `f2t250`: floor = step/2 = 0.0028125, τ = 250
   - `f2t1000`: floor = 0.0028125, τ = 1000
   - `f4t250`: floor = step/4 = 0.00140625, τ = 250
   - `f4t1000`: floor = 0.00140625, τ = 1000

   The cell with the highest seed-0 mean accuracy advances, even if it is below the parent.
   Ledger #21 says 3-task diagnostics rank poorly, so the screen is the decision.
2. **Screen:** the advanced cell on seeds 0-2 at 60 tasks × 5000, paired per seed against a
   **local parent** run. Parent shards from earlier waves on this box (the S&P wave) are reused
   only if the commit-of-learner-code and environment match exactly. Otherwise, the parent is
   re-run here. The control `upgd_w_control` is re-measured on seeds 0-2.
3. **Bar:** the mean paired difference (candidate − parent) must be ≥ +0.002, with at least 2 of 3
   seeds improved. Below the bar, the result is negative: it gets a ledger entry and no further
   seeds are run.
4. **Extension:** seeds 3-9 at 60 tasks are run only if the bar is cleared. The 10-seed decision
   uses the same bar and requires a 95% t-CI lower bound > 0.
5. **Held out:** seeds 10-19 are not run on this branch.
6. **Before handoff:** recheck #3118 and #3123. If either has merged, re-pair against the new
   best merged parent.

The report gives the per-seed paired table, the mean, the 95% t-CI (df = n−1), the count of
seeds improved and the control re-measure. Lane status: development_screening_diagnostic,
nonpromoting.
