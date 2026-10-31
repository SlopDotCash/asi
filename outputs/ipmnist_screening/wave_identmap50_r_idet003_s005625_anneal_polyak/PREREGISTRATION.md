# Preregistration: Polyak-style iterate averaging of the body, on top of the step anneal

This file is committed before any implementation or run on this branch.

## Parent and base
- **Parent:** `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`. Its code is at
  3c38b025, which this branch starts from; it is not on main.
  - Its 60-task screen gave +0.00226 over `..._idet003_s005625` (10/10 seeds).
  - Its 200-task, 20-seed confirmation gave +0.006576 (20/20 seeds), on branch
    `confirm/identmap50-r-idet003-s005625-anneal-200task`.
- **Not the parent:** the #3123 settings.
- **Main:** fd25c476, unmoved at the 19:37 GMT check.
- **Duplicate check, 19:38 GMT:**
  - The repo text has no hits for polyak, iterate averaging, SWA or averaged weights; the only
    "EMA weight" hits are unrelated overflow comments.
  - A GitHub issue/PR search found 0 results each for polyak, iterate averaging and SWA. Its 34
    "EMA weights" hits are all overflow and other fixes, and its one "weight averaging" hit is an
    RLS covariance fix.

## Single new variable
The parent learner is unchanged. In addition, the arm keeps an averaged copy `A` of the four
residual-trained body tensors (`w1, b1, w2, b2`).
- **Prediction only:** the deployed prediction becomes `argmax(wout.T @ phi_A(x_norm))`, where
  `phi_A` is the parent's feature map evaluated with `A` in place of the raw body.
- **Body update unchanged:** the body's gradient, gate, anneal step and update are exactly the
  parent's, computed on the raw body.
- **Average update, after each step:**

      n = 1 if reset else n + 1
      w = 1 / min(n, polyak_window)                 (float32)
      A = (1 - w) * A + w * body_new

  This is a uniform (Polyak) average since the last reset, which turns into an EMA with
  window `polyak_window` once `n` exceeds it.
- **Resets:** the average restarts (`n = 1`, so `A = body_new` exactly) at:
  - every detected shift, meaning the identifier's raw-stream trigger (`since_shift` restarts at 1);
  - **every identifier remap**, meaning any step on which the input permutation `remap` changes
    (matches at 50/200/2000). The reason is that the body adapts to each new remap, so weights
    from before the remap are tuned to a different input layout.
- **Initial state:** `A = params`, `n = 1`.
- **Head features (the second grid axis, `polyak_head_avg`):**
  - **0:** the RLS head trains on the raw body's features, exactly as the parent does.
  - **1:** the RLS head's update (gain, error, P) uses the averaged features `phi_A` and the
    deployed logits.
- **Reported loss:** the per-step loss metric is the deployed prediction's squared error.
  Plasticity keeps the parent's after-update formula, measured against that loss.

### Switched-off checks, exactly as the code implements them
- **`polyak_window` key absent** (the parent's registration): the factory returns the parent
  learner unwrapped. No average state exists and nothing is traced. Proof: the parent's jaxpr is
  byte-identical before and after this branch's code change.
- **Live path with `polyak_window = 1`** (key present, either head setting): the average is
  computed every step through the wrapper and the prediction-body plumbing. With `w = 1`,
  `A = 0*A + 1*body_new` equals `body_new` exactly, so prediction (and, with `head_avg = 1`, the
  head update) sees the raw body.

  On this path the arm must reproduce the **parent** (`..._anneal_f2t1000`) bit-exactly, for both
  head settings. A pin test asserts bitwise equality of per-task accuracy and loss, on a stream
  where the detector fires and the identifier remaps (the per-pixel offset stream with
  `ident_match_at = 10`, as in the anneal and S&P pins). I will show that a deliberate break
  (`A = 0.999*A + 0.001*body_new` replacing exact reset arithmetic, or a window of 2) makes
  it fail.

## Protocol
1. **Tuning:** seed 0, **30 tasks** × 5000 samples. The grid has 4 cells, each paired against a
   local parent run on seed 0 at 30 tasks:
   - `w100h0`: window 100, head on raw features
   - `w100h1`: window 100, head on averaged features
   - `w1000h0`: window 1000, head on raw features
   - `w1000h1`: window 1000, head on averaged features

   The cell with the highest seed-0 mean accuracy over the 30 tasks advances, even if it is below
   the parent.
   - **Why 30 tasks instead of 3:** the anneal's 3-task tune was misleading, because its gain is
     negative early and positive late (about −0.0028 over the first 15 tasks and +0.0057 over the
     last 15 of 60).
   - An iterate average also acts late within and across tasks.
   - Compute allows it: one 30-task run takes about 1-2 minutes on this box.
   - 30 tasks is half the screen horizon while still being a single seed.
   - The tasks 1-3 and 1-15 numbers are also reported, for ledger #21.
2. **Screen:** the advanced cell on seeds 0-2 at 60 tasks × 5000, paired per seed against the
   parent at 60 tasks.
   - **Parent shards:** the anneal wave's 60-task parent shards (tracked at 3c38b025) are reused
     only if all three checks pass:
     - the parent's jaxpr is identical before and after the code change;
     - the venv freeze is identical;
     - a seed-0 60-task parent re-run at this branch's code commit matches the stored shard's
       per-task accuracy and loss exactly.

     Otherwise, the parent is re-run here.
   - **Control:** `upgd_w_control` is re-measured on seeds 0-2.
3. **Bar:** the mean paired difference (candidate − parent) must be ≥ +0.002, with at least 2 of 3
   seeds improved. Below the bar, the result is negative: it gets a ledger entry and no further
   seeds are run.
4. **Extension:** seeds 3-9 at 60 tasks are run only if the bar is cleared. The 10-seed decision
   uses the same bar and requires a 95% t-CI lower bound > 0.
5. **Held out:** seeds 10-19 at 60 tasks are not run on this branch.
6. **Before handoff:** recheck #3118 and #3123. If either has merged, re-pair against the new
   best merged parent.

The report gives:
- the per-seed paired table, the mean, the 95% t-CI (df = n−1) and the count of seeds improved;
- the horizon breakdown (tasks 1-15, 16-30, 31-45 and 46-60);
- the control re-measure.

Lane status: development_screening_diagnostic, nonpromoting.

## Stop rules
- A crashed or partial shard (my own) is re-run unchanged and listed.
- A NaN or collapsed shard is recorded as is.
- No new shard starts during cargo-build-sbf or below 3 GB of available RAM.
- No outcome changes the grid, the bar or the seeds.
