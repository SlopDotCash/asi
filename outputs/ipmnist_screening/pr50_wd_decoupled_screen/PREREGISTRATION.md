# Preregistration: weight-decay-decoupled control for partial reset f=0.5 (development screen)

Written before any run of the new arms. Branch cut from 3c38b025. It carries cherry-picks of the partial-reset code (c0efe4c8, here 2fcf424d) and its tests (054d1504, here 57d96c05). The partial-reset branch itself is untouched.

## Why
In the partial-reset screen (RESULTS e0612611), pr50 gained +0.000613 paired on seeds 4-13, against the #3580 parent prefix. But the runtime anneal step also drives the body weight decay. `ipmnist_screening.py` lines 5049-5053 at 3c38b025/6bbfdccc read:
`decay = jnp.where(lr == jnp.float32(step_size), jnp.float32(param_decay), 1.0 - lr * weight_decay)`
with weight_decay = 0.01 and step_size = 0.005625. So pr50 changed both the step and the decay after each shift. This screen separates the two, to decide whether f=0.5 at 200 tasks is worth running.

## Design (one numeric key, because hyperparameters must be finite numbers, so a string 'arm'/'parent' key is impossible)
- `anneal_decay_reset_frac` = f_d is valid only with `anneal_reset_frac` and requires 0 ≤ f_d < 1.
- A second int32 clock, `decay_clock`, restarts at `1 + floor(f_d·s)` on a detected shift and otherwise counts +1.
- The decay reads the same anneal schedule `max(floor, step·(1+s/tau)^-1/2)` evaluated on `decay_clock`. It is passed into the gated SGD as `decay_lr`, which replaces `lr` only inside the decay expression above. The step still uses the `anneal_reset_frac` clock.
- With the key absent, the decay reads `lr` exactly as before, so pr50 and the parent are unchanged.
- f_d = 0 restarts at 1, identical to the parent's `since_shift` clock, so the decay is the parent's own full-reset decay. f_d = f_s reproduces the coupled arm.
- Arms (all else equals pr50):
  - `…_pr50_steponly`: f_s = 0.5, f_d = 0, i.e. the pr50 step with the parent's decay.
  - `…_pr50_decayonly`: f_s = 0, f_d = 0.5, i.e. the parent's step with the pr50 decay. This costs 10 runs and makes the decomposition two-sided with a visible interaction.
- Pins:
  - f_d = f_s = 0.5 equals pr50 bit-exactly.
  - f_s = f_d = 0 equals the parent bit-exactly.
  - f_d = 0.9 with f_s = 0.5 differs (the deliberate break).
  - Invalid values are rejected, and f_d without f_s is rejected.
  - Registry test updated.

## Plan (60 tasks × 5000 steps, no tuning, f = 0.5 fixed)
1. **Env check:** re-run `upgd_w_control` seed 0 and the parent seed 0 (vs 3c38b025 shards) and the stored **pr50 seed 4** (vs e0612611, the parent of the arm under test, per the #3585 lesson). All must match bit-for-bit. On any mismatch: VOID, stop, no automatic re-run.
2. **Reuse, verbatim with SHA256SUMS:** parent seeds 4-13 from 612491db (#3580, analysed on [:60]) and pr50 seeds 4-13 from e0612611.
3. **Run** steponly and decayonly on seeds 4-13, the same seeds as the partial-reset confirm, so all results pair.
4. **Report:**
   - each component vs the parent and each split arm minus pr50, with every seed;
   - mean, SE, 95% t CI (df=9, t=2.262), and the count of seeds improved;
   - gains by 15-task block;
   - the interaction, pr50 − steponly − decayonly + parent.

## Decision rule (preregistered)
Let S = steponly mean gain vs the parent and C = pr50 mean gain vs the parent (both recomputed here, C ≈ +0.000613).
- **STEP-REAL** (recommend preregistering f=0.5 at 200 tasks): S ≥ 0.5·C **and** CI lower bound of S > 0.
- **DECAY-DRIVEN** (do not recommend the 200-task run): CI lower bound of S ≤ 0 **or** S < 0.25·C.
- Otherwise **MIXED**: no 200-task recommendation without a new preregistration.

The decayonly arm is descriptive; it does not enter the rule.

## Prior
P(STEP-REAL) ≈ 45%, P(MIXED) ≈ 30%, P(DECAY-DRIVEN) ≈ 25%. The step change after a shift (0.005625 → ~0.0030) is the larger direct lever. But the decay difference compounds, roughly a few percent of extra weight shrink per task.

## Runtime
About 25 min with 4 one-thread workers:
- env check about 9 min, limited by the control run;
- 20 arm runs at about 2.4 min each, about 13 min.

## Disclosures
- The coupling at lines 5049-5053, as quoted above.
- This was chosen after the partial-reset result (FAIL on the mean bar, 10/10 seeds improved).
- The parent and pr50 shards are reused, and seeds 4-13 were already seen for both.
- Development screen only; nonpromoting.
