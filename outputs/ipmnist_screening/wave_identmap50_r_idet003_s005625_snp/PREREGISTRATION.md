# Preregistration — shrink-and-perturb at the identifier's first remap

Committed before any implementation or run on this branch.

## Parent
`rls_head_resid_identmap50_r_idet003_s005625`. It is merged on main (fd25c476), and its
200-task/20-seed confirmation came out at +0.026599 paired, 20/20 seeds.

## Single new variable
This is Shrink & Perturb (Ash & Adams 2020, "On Warm-Starting Neural Network Training"),
applied to the residual body tensors `w1, b1, w2, b2` once per detected task. The update fires
at the identifier's **first** remap of that task, which is the step where
`ref_frozen and since_shift == ident_match_at` (50 post-shift samples). Rule:

    theta <- snp_shrink * theta + snp_sigma * fresh

`fresh` is a new draw from the body's init family, `U(-1/sqrt(fan_in), +1/sqrt(fan_in))`, where
fan_in is the input dimension of the layer the tensor belongs to. The draw is keyed from
`jr.fold_in(step_key, SALT)`, so the parent's own key stream is never consumed or altered. The
refinement matches at 200 and 2000 do NOT trigger S&P. The RLS head, the identifier, the gate and
every other hyperparameter stay exactly as in the parent.

The new variable is the pair `(snp_shrink, snp_sigma)`. They form one mechanism ("how hard to
soften the body at the remap"), and both are tuned together as one knob.

### The off setting, exactly as the code implements it
- **The `snp_shrink` key is absent** from the hyperparameters (the parent's registration). The
  factory compiles no S&P code at all, and the step function is the parent's verbatim.
- **`snp_shrink = 1.0, snp_sigma = 0.0` with the key present.** The live S&P path RUNS: at every
  first-remap event it computes `jnp.where(event, 1.0*theta + 0.0*fresh, theta)`, drawing `fresh`
  from the folded key. In float32 that is bit-exactly `theta`. The pin test checks this live path,
  not just the key-absent path, against the parent bitwise, on a stream where the identifier
  actually fires (per-pixel offset data with `ident_match_at = 10`). The suite already used that
  stream to prove the identifier is live.

## Priors (stated up front; both point negative)
- **Negative-results ledger #3:** "Perturbation noise is not additive to good input
  conditioning" (UPGD perturbation, a different learner, but the same idea of noise in the body).
- **#3125 (L2-Init on the r02 chain):** rejected at −0.01117 paired, with the loss uniform across
  life. The diagnosis was that pulling the body back toward its initial draw erodes the matured
  feature map, and this family shows no late-life plasticity loss (last-15 slopes ~0, plasticity
  ~0.70). S&P is a discrete, event-gated cousin: it shrinks toward zero and injects init-scale
  noise once per task, rather than decaying continuously toward the init. The specific question
  here is whether a one-shot softening right when the input layout is remapped helps the body
  re-fit the remapped input.

## Protocol
1. **Tuning**: seed 0 only, 3 tasks x 5000 samples, `--noise-mode` default (the same as all
   lane screens). There are two cells, and each is paired against a local re-run of the parent
   on seed 0 at 3 tasks:
   - A: `snp_shrink = 0.8, snp_sigma = 0.01` (mild)
   - B: `snp_shrink = 0.5, snp_sigma = 0.1` (Ash & Adams-scale)
   The cell with the higher seed-0 mean accuracy advances. If both are below the parent, the
   higher one still advances to the screen, so the screen is the decision, not the tune.
2. **Screen**: the advanced cell on seeds 0, 1, 2, at 60 tasks x 5000. It is paired per seed
   against a **local re-run of the parent** on the same box (this box reproduces the stored
   parent about 0.0002 low; stored #3117 shards are not used as the pairing base). The control
   `upgd_w_control` is re-run on seeds 0-2 to re-measure the environment against its stored
   shards.
3. **Bar**: mean paired difference (candidate − local parent) ≥ **+0.002**, with at least 2 of 3
   seeds improved. Below the bar means a negative result: an entry in
   docs/evidence/negative-results.md and no further seeds.
4. **Extension**: seeds 3-9 (60 tasks) are run ONLY if the screen clears the bar. The decision
   then uses all 10 seeds with the same bar plus a 95% t-CI lower bound > 0.
5. **Held out**: seeds 10-19 are not run on this branch under any outcome.

The report gives the per-seed paired table, the mean, the 95% t-CI (t, df = n−1), the
seeds-improved count, and the control re-measure. Lane status:
development_screening_diagnostic, nonpromoting.
