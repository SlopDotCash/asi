# RESULTS: decay-only on fresh seeds 30-39 vs plain lower weight_decay (development screen, nonpromoting)

Prereg ccb788d6 (02:16 GMT, 31 Oct 2026); launcher 028202f3; run 02:21-02:59 GMT, 31 Oct 2026.
Env check PASS: all three references matched bit for bit (control and parent seed 0 vs 3c38b025, decay-only seed 4 vs e9f6125c).
60 tasks, seeds 30-39, paired per seed; 95% t CI, df=9.

| Arm | mean acc |
|---|---|
| parent `…_anneal_f2t1000` | 0.936786 |
| A `…_pr50_decayonly` | 0.938492 |
| B `…_wd00084` | 0.939219 |
| B2 `…_wd0005` (descriptive) | 0.944808 |

| Comparison | mean | CI95 | improved | blocks (15 tasks) |
|---|---|---|---|---|
| A − parent | +0.001706 | [+0.001580, +0.001831] | 10/10 | +0.001303 / +0.001876 / +0.001840 / +0.001804 |
| B − parent | +0.002432 | [+0.002367, +0.002498] | 10/10 | +0.002020 / +0.002596 / +0.002627 / +0.002487 |
| B2 − parent | +0.008022 | [+0.007840, +0.008204] | 10/10 | +0.006456 / +0.008387 / +0.008576 / +0.008669 |
| D = A − B | −0.000727 | [−0.000828, −0.000625] | 0/10 | −0.000717 / −0.000720 / −0.000787 / −0.000683 |
| A − B2 | −0.006316 | [−0.006481, −0.006152] | 0/10 | |

Per-seed values are in ANALYSIS.txt.

**A bars: FAIL.** The mean, +0.001706, is below +0.002. The other two bars hold: 10/10 seeds improved and the CI lower bound is above 0.

**Timing read: AMOUNT-ONLY.** The CI of D lies entirely below 0. A plain constant weight_decay of 0.0084, matched to A's average decay, beats shift-timed decay on every seed. The decay clock adds nothing beyond lowering the amount of decay.

**B2 (0.005) is descriptive only.** It is +0.008022 on 10/10 seeds, about 4.7× A's gain. It is post hoc and outside the decision rule; any use of it needs its own preregistered weight-decay sweep on fresh seeds. It is nonpromoting.

Disclosures:
- Post hoc after #3589 and #3593.
- The coupling: `decay = jnp.where(lr == step_size, param_decay, 1 - lr*weight_decay)`.
- B's value was derived analytically.
- A non-negative weight_decay guard was added; it does not change any valid run.
- Branch base 3c38b025; main has moved since.
- Development screen only; nonpromoting.
