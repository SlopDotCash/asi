# IPMNIST mechanism-combination screening — runbook

Development screening diagnostic (never promotable evidence) hunting for
configurations that beat the reproduced UPGD-W SOTA on the ICLR-2024 online
Input-permuted MNIST protocol. Code:
`alberta_framework/benchmarks/ipmnist_screening.py`
(tests: `tests/test_ipmnist_screening.py`).

## Proxy

Same protocol as `alberta_framework/benchmarks/upgd_ipmnist.py` at a reduced
horizon: **60 tasks x 5,000 steps** (vs 200 tasks). Because `build_schedule`
folds the task index into per-seed keys, a 60-task run is an **exact prefix**
of the 200-task run for the same seed. Validation therefore checks
bit-level prefix agreement of the control shards against the completed
full-horizon shards in `outputs/upgd_ipmnist/partials/` (read-only), plus
ordering preservation (UPGD-W > AdamW). **Audit note (2026-08-02,
`AUDIT.md`):** the bit-level prefix holds exactly WITHIN the screening
runner (every 60-task shard is a bitwise prefix of its 200-task
`confirm_full/` shard, max diff 0.0) and across runners for AdamW
(max diff 5e-9 = shard rounding), but NOT across runners for UPGD-W:
`run_ipmnist` executes vmap-batched, the screening runner unbatched, and
the two XLA compilations of the same UPGD-W step diverge by 1-2 ulp
within 10 steps (bitwise-equal inputs/keys/noise verified), which chaos
amplifies to ≤0.0096 per-task accuracy jitter (~+0.0004 60-task mean).
`proxy_validation.json` therefore reports `all_prefixes_match=false` for
`upgd_w_control` — a harness float-reassociation artifact, not a
protocol or schedule difference (the AdamW bitwise match proves
data/schedule/init/forward/backward identity across the two runners). From the completed 10-seed full runs,
at 60 tasks: UPGD-W 0.7777 vs AdamW 0.7553 (paired +0.0224, all 10 seeds
ordered; per-seed stderr ~0.0004). Do not shrink task_length below 5000.

## Arms (registry names)

- `upgd_w_control` / `adamw_control` — published configs (control + ordering).
- `upgd_idbd`, `upgd_idbd_meta1e2` — UPGD-W + IDBD per-weight step-sizes
  (meta signal = gated loss gradient; log-alpha clipped to [-10, 0]).
- `upgd_autostep` — UPGD-W + Autostep per-weight step-sizes (global M).
- `upgd_l2init` — UPGD-W decay pulls toward initial weights.
- `upgd_ema_norm` — EMA input normalizer (decay 0.999) in front of the MLP.
- `upgd_cbp`, `adamw_cbp` — CBP-style dormant-unit recycling
  (rho=1e-4, utility decay 0.99, maturity 100; protocol-uniform re-init,
  per-unit optimizer-state reset).
- `upgd_w_{sigma005,sigma02,udecay0999,udecay099999,wd0005,wd002}` —
  hyperparameter-neighborhood star around the published UPGD-W config.
- `upgd_w_wclip_{k1,k2,k1_wd0,k2_wd0}` — UPGD-W + per-layer weight clipping
  to `[-kappa*s_l, +kappa*s_l]`, `s_l = 1/sqrt(fan_in)` init bound (Elsayed,
  Lan, Lyle & Mahmood, RLC 2024); kappa in {1,2} x weight decay in {0.01, 0}.
  Jobs in `jobs2.txt`; kappa=inf reduces bit-exactly to the control (pinned).
- `upgd_w_localgate` — utility gate normalized by the per-tensor max instead
  of the network-global max (zero-max guarded; single-tensor parity pinned).

Seeds 0-2 per arm; identical seeds across arms => paired comparison. All
screening shards (including the wclip/localgate wave) use the exact per-step
noise path. (Stale as of the confirmation waves: the screening runner
gained `--noise-mode pool` and `confirm_full/worker_confirm.sh` uses it —
but only for arms that declare a pool `noise_update`. In practice only
`upgd_w_control` and `upgd_w_wd0005` confirm shards are pool64; every
other `confirm_full/` shard fell back to exact `step` mode — check the
`noise_mode` field in each shard.)

## Execution

One process per (config, seed), `OMP_NUM_THREADS=1`, 15-way parallel
(3+ cores left for the detached OPMNIST processes):

```bash
cd outputs/ipmnist_screening
xargs -P 15 -n 2 bash worker.sh < jobs.txt   # idempotent; skips done shards
```

Measured costs at 1 thread, per 5,000-step task: control UPGD ~30s,
IDBD/Autostep ~34s, CBP ~25-29s, ema_norm ~18s, AdamW ~4-9s. A 60-task seed
run is ~20-35 min.

## Resume

Workers are idempotent per shard (`shards/<config>_seed<seed>.json`); rerun
the xargs line to fill in whatever is missing. Logs per worker under `logs/`.

## Validation + merge (after controls / after all shards)

```bash
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening validate-proxy \
  --shards outputs/ipmnist_screening/shards/upgd_w_control_seed*.json \
           outputs/ipmnist_screening/shards/adamw_control_seed*.json \
  --partials-dir outputs/upgd_ipmnist/partials \
  --output outputs/ipmnist_screening/proxy_validation.json

.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/ipmnist_screening/shards/*.json \
  --control-name upgd_w_control \
  --output outputs/ipmnist_screening/summary.json
```

A config with paired mean improvement > 0.005 over `upgd_w_control` is a
full-protocol confirmation candidate (200 tasks, 10+ seeds via the
`upgd_ipmnist` v3 lane) — screening results alone claim nothing.

`smoke/` holds tiny CLI calibration runs only; ignore for analysis.

## Wave 3 (`jobs3.txt`)

Two additional arms, seeds 0-2 each (6 jobs in `jobs3.txt`; `jobs.txt` and
`jobs2.txt` untouched). `upgd_w_fade_head` puts FADE-style meta-learned
per-parameter weight decay (Ramesh, Lewandowski & Schmidhuber, arXiv
2604.27063) on the output layer only: `lambda_i = exp(gamma_i)` replaces the
head's fixed decoupled decay, adapted by `gamma_i += theta_lambda * delta_t *
x_i * g_i` with a forward-mode sensitivity trace `g_i <- g_i * max(0, 1 -
lambda_i - alpha*x_i^2) - lambda_i*w_i` (published alpha=0.005, gamma_0=-6.9,
theta_lambda=0.1; gamma capped at 0; lambda=0 reduces bit-exactly to the
zero-head-decay control — pinned). Hidden layers stay published UPGD-W.
`upgd_w_idbd_swift` is the `upgd_idbd` arm plus SwiftTD's two supervised-mode
stabilizers (`alberta_framework/core/swift_td.py` forms, phi_i = gated
gradient z_i): a network-global overshoot bound scaling each update by
`eta/tau` when `tau = sum_i alpha_i z_i^2 > eta` (eta=0.1), and persistent
step-size decay `log_alpha_i += ln(eps) * z_i^2` plus meta-trace reset on
trigger (eps=0.99); `eta=inf, eps=1` reduces bit-exactly to `upgd_idbd`
(pinned). Both are exact-noise (`noise_mode="step"`) arms with no pool-mode
update. Run once the main sweep frees cores, with the same worker pattern:
`cd outputs/ipmnist_screening && xargs -P 6 -n 2 bash worker.sh < jobs3.txt`
(idempotent; shards land in `shards/` and merge with the existing control
shards through the standard `merge` command).

## Wave 4 (`jobs4.txt`) — theory-driven novel arms (QUEUED, not launched)

Eight arms x seeds 0-2 (24 jobs in `jobs4.txt`; earlier jobs files untouched).
Derivations, falsifiable predictions, and refutation criteria for every arm
live in `CONTINUAL_LEARNING_THEORY.md` (repo root). All arms are exact-noise
(`noise_mode="step"`) with no pool-mode update; all reductions below are
pinned by unit tests in `tests/test_ipmnist_screening.py`.

- `guarded_cbp_adam` — the `adamw_cbp` leader plus UPGD-style utility
  *protection only*: Adam's applied per-weight delta is scaled by
  `1 - guard_scale * gate`, gate = UPGD's sigmoid/global-max squashing of the
  bias-corrected `-w*g` utility EMA (utility_decay 0.9999). No perturbation
  (CBP supplies regeneration); moments see raw gradients; recycled units also
  reset their guard utility (fresh units restart at the neutral 0.5 gate).
  `guard_scale=0` reduces bit-exactly to `adamw_cbp` (pinned).
- `adamw_cbp_noreset` — `adamw_cbp` WITHOUT the per-unit Adam m/v/count reset
  at CBP replacement. NOTE: the reset ("vreset") is already the `adamw_cbp`
  default, so the moment-freshness mechanism is dissected by ablation.
  `cbp_replacement_rate=0` reduces to `adamw_control` (pinned for both arms).
- `upgd_w_sigma0` — lean UPGD-W at sigma=0 (pure gated SGD + decoupled
  decay); the per-step noise draw (~85-90% of UPGD step cost) is skipped
  entirely, bit-exact vs the control factory at noise_std=0 (pinned). Runs at
  near-AdamW cost (~5-9s/task vs ~30s single-thread).
- `upgd_alpha_utility` — UPGD-W whose protection gate reads passive IDBD
  step-size drift (log_alpha kept as a statistic on the raw gradient, never
  applied as a rate; gate = sigmoid(drift / global max |drift|), 0.5 at zero
  drift). meta 1e-2, initial_step_size 0.01. `meta=0` reduces bit-exactly to
  the closed-form half-gated step (pinned). Cost ~ `upgd_idbd` (~34s/task).
- `adamw_cbp_{r3e5,r3e4,m50,m200}` — axis-aligned mini-star on the untuned
  leader: replacement_rate {3e-5, 3e-4} and maturity {50, 200}, one axis at a
  time (defaults 1e-4 / 100 held elsewhere). Cost ~ `adamw_cbp`.

Run (only when the confirmation runs + sweep free cores — the box is
contended; do NOT start these alongside the 200-task lanes):

```bash
cd outputs/ipmnist_screening
xargs -P 8 -n 2 bash worker.sh < jobs4.txt   # idempotent; skips done shards
```

Then re-merge with the standard `merge` command (all shards, control
`upgd_w_control`); for the adamw-family arms the interesting paired contrast
is additionally vs `adamw_cbp` (`merge --control-name adamw_cbp` on the
adamw-family shards into a SEPARATE summary file, e.g.
`summary_wave4_adamcbp.json` — never overwrite `summary.json` with a
different control).

## Wave-4b (2026-08-01, appended late): `adamw_cbp_ema_norm`

- `adamw_cbp_ema_norm` — composition arm: the exact `adamw_cbp` update
  behind the exact `upgd_ema_norm` EMA input normalizer (same
  norm_decay=0.999 / norm_epsilon=1e-8, same per-step state threading;
  normalizer state carried in the arm state). `norm_enabled=0` reduces
  bit-exactly to `adamw_cbp` (pinned); the normalizer path is pinned
  bitwise against `upgd_ema_norm`'s on a shared stream. Cost ~ `adamw_cbp`.
- Ops: the three seeds (0-2) were APPENDED to `jobs4.txt` so endgame3's
  per-shard wait loop (which re-reads jobs4.txt each cycle) counts them and
  the final merge includes them. Because endgame3's first blocking
  `xargs < jobs4.txt` had already buffered the original 24 lines and its 6h
  deadline (started 17:18) will lapse before that xargs drains, the three
  workers were ALSO launched directly, detached, at `-P 3` via `worker.sh`
  (idempotent; endgame3's restart branch cannot double-run them — it only
  fires when no `ipmnist_screening run` processes are alive).
- Interesting paired contrasts once merged: vs `upgd_w_control` (standard),
  and vs `adamw_cbp` / `upgd_ema_norm` on the adamw-family/normalized side
  (separate summary files as above; never overwrite `summary.json`).

## Wave 6 (`jobs6.txt`, 2026-08-02): `sgd_ema_norm` — the gate ablation

- Motivation: `upgd_ema_norm` leads the screen (0.8529; 0.85357 confirmed at
  200 tasks with the n=3 then available — 0.85362 at the final n=10) and `upgd_ema_norm_sigma0` TIES it (0.8520), so under input
  conditioning the method is normalize + utility-GATED SGD + decay. This arm
  closes the dissection: does the gate itself matter?
- `sgd_ema_norm` — plain SGD with decoupled weight decay
  (`w <- w*(1 - lr*wd) - lr*grad`) behind the EXACT `upgd_ema_norm` EMA input
  normalizer (norm_decay 0.999, norm_epsilon 1e-8, same state threading;
  normalizer states pinned bitwise against `upgd_ema_norm`'s on a shared
  stream). lr 0.01, wd 0.01 (the leader's non-noise values). NO utility, NO
  gate, NO noise — the per-step RNG key is unused; hand-computed-trajectory
  reduction pinned in `tests/test_ipmnist_screening.py`.
- Ops: seeds 0-2 in `jobs6.txt`, launched detached 2026-08-02 via
  `setsid xargs -P 3 -n 2 bash worker.sh < jobs6.txt` (idempotent; shards
  land in `shards/`, launch log `wave6_launch.log`). Cost ~ `upgd_w_sigma0`
  (no noise draw) minus the utility bookkeeping — the cheapest UPGD-family
  arm yet.
- Interpretation once merged (paired vs `upgd_ema_norm` /
  `upgd_ema_norm_sigma0` on shared seeds; separate summary file, never
  overwrite `summary.json`): a tie at ~0.85 means the gate is NOT
  load-bearing on this no-recurrence protocol and the SOTA here reduces to
  normalize + SGD + decay; a drop toward the raw-input UPGD/AdamW band
  (~0.75-0.78) means the utility gate IS the mechanism and
  normalized-gated-SGD stands as the method.

## Comparison wave (`jobs_comparison.txt`, 2026-08-02): published mechanisms under our conditioning

The reviewer-demanded comparison rows — "our conditioning + THEIR mechanism
vs our conditioning + our gate". Five arms, seeds 0-2 (15 jobs in
`jobs_comparison.txt`), all behind the champion's EMA input normalizer
(decay 0.99, eps 1e-8) on a plain-SGD base (lr 0.01): no utility gate, no
perturbation. Mechanism rules verified against the source papers
(2026-08-02); every factory reduces bit-exactly to the shared
normalized-SGD base when its mechanism constant is inert (pinned in
`tests/test_ipmnist_screening.py::TestComparisonArms`).

- `sgd_ema_norm_d099` — the mechanism-free floor: plain SGD + decoupled
  wd 0.01 (`sgd_ema_norm` retimed to the champion's decay-0.99 conditioning).
- `wclip_ema_norm` — Weight Clipping (Elsayed, Lan, Lyle & Mahmood, RLC
  2024, Algorithm 1): SGD then per-layer clip of weights AND biases to
  ±2/sqrt(fan_in) (kappa=2, the paper's example); wd 0 (their standalone
  configuration — clipping is their alternative to decay-family
  regularizers). kappa=inf reduces to the base (pinned).
- `fade_head_ema_norm` — FADE meta-learned per-parameter weight decay on
  the output layer only (arXiv 2604.27063; published alpha=0.005,
  gamma0=-6.9, theta=0.1), hidden layers undecayed (the paper adapts the
  final layer only). theta=0 + gamma0=-inf reduces to the base (pinned).
- `snr_ema_norm` — Self-Normalized Resets (Farias & Jozefiak, arXiv
  2410.20098, Algorithm 1): per-unit geometric-tail test P(A >= a) <= eta
  on the unit's estimated firing rate; rejected units re-init incoming
  weights+bias from the protocol uniform and zero outgoing. eta=0.005
  (inside the paper's PM sweep grid 0.08..0.00125); firing rate = bias-
  corrected EMA (decay 0.999) of the per-step firing indicator — the
  streaming stand-in for their fixed trailing window (their own geometric
  reduction), documented deviation. NOTE their experiments batch 16
  examples/step (healthy units ~never batch-silent); at one example/step
  the same eta is necessarily more trigger-happy — that protocol
  difference is part of what the row measures. eta=0 reduces to the base
  (pinned). Cost note: the faithful full re-init draw (~282k uniforms/step)
  makes this the slowest comparison arm (~40-70s/task vs ~1.5s/task for
  the other rows on the same box).
- `l2init_ema_norm` — L2-Init (Kumar et al.): decoupled decay pulls toward
  the initial weights, lambda = wd = 0.01 (the raw-input `upgd_l2init`
  value). wd=0 reduces to the base (pinned).

Ops: launched detached 2026-08-02 via
`xargs -P 5 -n 2 bash worker.sh < jobs_comparison.txt` (idempotent; launch
log `comparison_launch.log`). 200-task step-mode confirmations for the two
rows at/above the champion (`l2init_ema_norm` cleared the +0.002 auto-
confirm bar; `sgd_ema_norm_d099` tied the champion and is the ablation
needed to attribute the l2init win): `confirm_full/jobs_comparison_confirm.txt`
via `worker_confirm.sh` (pool64 attempt falls back to exact step — all five
arms are noise-free). Results + interpretation: `FINAL_REPORT.md`
comparison section; merged rankings in `summary_comparison.json` (control
`upgd_w_control`) and `summary_comparison_vs_champion.json` (control
`sigma0_ndecay099`) — separate files, `summary.json` untouched.

## rls_head wave (2026-08-03) — champion body + streaming-RLS readout

Convergence-shortfall attack (section (w) of `ipmnist_screening.py`):
the `sigma0_shiftnorm_d099` champion body with a streaming recursive-
least-squares one-hot readout on the bias-augmented 150-dim penultimate
features. Design axes: forgetting factor `rls_lambda` {0.995, 0.999, 1.0},
detector-driven P reset (`rls_reset_frac`, reusing the champion's shift
detector; threshold 0.05 calibrated at 2.8x margin over the within-task
detector maximum), initial/reset ridge, P trace cap, and the body error
signal (`head_resid`: parallel champion SGD head vs the RLS residual).

Verdicts (60-task screen, paired seeds 0-2; summary_rls_head.json):

- **`rls_head_resid_l1_preset005` 0.86938 — new screen leader** (+0.0054
  vs the champion 0.86396, +0.0037 vs `disc_r1_pscale_norms` 0.8657, all
  seeds improve on both): residual-driven body on the wind-up-immune
  head (lambda 1.0 + P reset). **200-task confirmation, n=20 (seeds
  0-19, held-out 3-19 untouched until the screen passed): 0.87114 +/-
  0.00010 vs the champion's 0.86449 +/- 0.00009 on the same seeds —
  paired +0.00665 +/- 0.00013, all 20 seeds improve (min +0.0055),
  late-window (t181-200) paired diff +0.0064 all-positive (drift-free),
  no run collapses (min per-task 0.7322 = its own task 1).** Shards:
  `confirm_rls_head/`; merged paired summary:
  `summary_rls_head_confirm.json`. The 60-task screen is a bitwise
  prefix of the 200-task runs (max abs diff 0.0, seeds 0-2). All
  numbers remain development_screening_diagnostic, nonpromoting.
- Every lambda<1 arm without resets collapsed (covariance wind-up on
  sparse learned features; ledger #26); resid at ridge 1.0 without
  resets collapsed by task 5-9 (ledger #27); small ridge won 2-task
  diagnostics and lost the horizon (ledger #28).
- The plateau question splits by mechanism (ledger #29): the readout
  ALONE does not move the 0.904 within-task plateau (parallel arms
  0.90443/0.90195), but the residual-trained body does — the winner
  measures plateau 0.91490 (+0.0107, 37% of the 0.029 convergence
  shortfall), a uniform ~+0.011 lift across steps 1000-5000 at a small
  0-100-step boundary cost.

Ops: shards via `worker.sh` (waves rls_jobs{,2,3}.txt, scratchpad);
collapsed shards keep non-finite per-task losses and are excluded from
merges by the validator. Merged ranking: `summary_rls_head.json`
(control `upgd_w_control`; `summary.json` untouched).

## EMA warmup contract clarification (2026-08-16)

The checked-in EMA-normalized shards and the live estimator both use
`min(decay, 1 - 1 / (accepted_count + 1))`. Issue #469 made the previously
implicit initialization contract explicit: the zero mean and unit variance
are one prior pseudo-sample, so the first effective decay is
`min(decay, 1/2)` and `decay=1` is a prior-regularized cumulative estimator,
not an ordinary sample mean or Welford estimator. The recurrence and
historical shard values did not change. New configuration receipts use
distinct EMA-prior semantics labels; the loader continues to accept the older
generic labels for compatibility and canonicalizes them when re-serialized.
As always, do not rewrite these development artifacts, and bind any later run
to its exact source revision.

## Gate-ablation r2 (2026-08-19)

Pre-registered n=10 paired screen resolving the issue #52 ambiguous band:
`rls_head_resid_l1_preset005` vs `rls_head_resid_l1_preset005_nogate`, both
arms remeasured on one CPU runner (HEAD `cc877f785666`), seeds 0-9, 60 tasks
x 5000, step noise. Pre-registration: issue #1937 (evaluation seeds 10-19
held out, frozen decision rule, win bar +0.002).

Result: paired +0.001712 ± 0.000174 (stderr_diff), all ten per-seed diffs
positive (9.84x stderr), control 0.869634 ± 0.000250, candidate
0.871345 ± 0.000250. Below the +0.002 win bar, so the ambiguous-band rule
applied: reported plainly, no 200-task confirmation, no evaluation-seed
touch, no threshold move. Recorded fact: gate removal is consistently
not-worse and 2.9x cheaper (about 129 s vs about 375 s per shard on CPU).
Remeasured control seeds 0-2 track the archived n=3 within the run-to-run
jitter class. Artifacts: `gate_ablation_r2/shards/` (20 v2 shards),
`gate_ablation_r2/summary.json` (control `rls_head_resid_l1_preset005`),
per-shard logs. Ledger entry 16.
