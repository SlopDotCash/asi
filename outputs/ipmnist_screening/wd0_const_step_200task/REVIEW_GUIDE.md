**Review guide: wd0 constant-step 200-task confirmation**

This guide points to the files behind the headline result and lists the commands that check it. It also covers the review points raised on this PR.

**Headline comparison**

- Candidate: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0`. This is the constant body step 0.02 (`anneal_floor == step_size == 0.02`, `weight_decay 0`).
- Incumbent: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0`.
- Design: 200 tasks, 20 paired seeds (134 to 153), fixed before the run in `DEVELOPMENT_PLAN.md`.
- Shards: `confirm/` holds 40 JSON files, one per arm and seed, listed in `confirm_jobs.txt`.
- Environment check: `envcheck/`, `envcheck.py` and `ENVCHECK_STATUS` (PASS).
- Analysis: `analyze.py` reads only `confirm/` and prints the verdict. Its committed output is `ANALYSIS.txt`.
- Note: `wd0_confirm200/analyze.py` belongs to an earlier stacked step (wd0 vs wd 0.01). It is not the headline comparison.

**Reproduce the verdict from the committed shards (seconds, no benchmark run)**

From the repository root:

```
python outputs/ipmnist_screening/wd0_const_step_200task/analyze.py
```

Re-run at head 484546adb4d7c46cca9a66684499b7c6c475f390 with this follow-up applied. The output is byte-identical to `ANALYSIS.txt`. Last lines, verbatim:

```
rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0 vs rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0
blocks (50 tasks, mean over 20 valid seeds): +0.014213 / +0.006742 / +0.005351 / +0.004353
mean +0.007665 SE 0.000037 CI95 [+0.007588, +0.007741] improved 20/20 diverged 0/20
BARS: mean>=+0.002 True, >=14/20 True, CI_lo>0 True; no diverged seed True -> PASS
```

The verdict rules (classify and screen_verdict) are tested in `tests/test_ipmnist_screening.py::TestWd0ConstStep200TaskConfirm`.

**Schedule logic and its tests**

The anneal, partial-reset and decoupled-decay arithmetic now lives in four small module-level helpers in `alberta_framework/benchmarks/ipmnist_screening.py`:

- `_decode_anneal_clock`
- `_anneal_lr`
- `_advance_anneal_clock`
- `_runtime_decay`

The learner calls these helpers in place of the same expressions written inline. The operations are unchanged. On the pin stream, 16 runs gave bit-identical per-task accuracy and loss before and after the move. Those runs covered both headline arms, pr25, pr50, pr50_steponly, pr50_decayonly, wd00084 and the anneal_f2t1000 parent, each at registered tau and at tau 2.

`tests/test_ipmnist_screening.py::TestAnnealScheduleUnits` tests the helpers directly on tiny synthetic inputs, on CPU, in about 2 s. It covers:

- clock decoding before and after the first shift;
- +1 advance;
- the partial-reset restart, including small prev;
- floor rather than round;
- the anneal schedule and its floor;
- decay branch selection;
- that both headline arms run with no partial or decoupled reset at weight_decay 0.

Mutation check: each mutation below was applied to the helpers and the class was re-run. Every one made at least one test fail:

- restart always 1;
- no +1 advance;
- round instead of floor;
- no clock offset decoding;
- decay always general;
- decay always constant;
- min instead of max for the floor.

Earlier stacked steps already added learner-level pins in `TestRLSHeadIdentMapStepAnneal`, `TestRLSHeadIdentMapAnnealPartialReset` and `TestRLSHeadIdentMapAnnealDecayDecoupled`:

- f = 0 equals the parent bit-exactly;
- f_d == f_s equals the coupled arm bit-exactly;
- a deliberate break differs;
- invalid values are rejected.

**Code risks raised in review**

- `step = lr` (Array next to a float `step_size`): not a bug. `step` is typed `float | Array`. Under jit, `lr` is a float32 scalar. The float default is used only when no runtime step is given.
- `dl == jnp.float32(step_size)`: this equality is deliberate. It reproduces the parent's precomputed `param_decay` bit-exactly when the step equals the base step. Any other `dl` takes `1 - dl*weight_decay`, which is the same quantity up to float32 rounding, so a miss gives no wrong decay. Behaviour is unchanged. It does not affect the headline: both arms run at `weight_decay 0`, where both branches give exactly 1. The constant-step arm keeps `lr == float32(0.02)` at every clock, so it always takes the equality branch. A test checks this.
- `restart = 1 + floor(reset_frac * prev)` gives 1 when `reset_frac * prev < 1`, for example `0.5 * 1`. That is the defined rule, and it is the same as a full reset. With `shift_refractory 0`, back-to-back triggers can reach it in the pr25 and pr50 screening arms. Changing it would change those committed screen results, so it is left as is and documented here as a known property. The headline arms do not use `anneal_reset_frac` or `anneal_decay_reset_frac`, so this path never runs for them.
- `anneal_clock` advancing when `partial_on=False`: it mirrors `since_shift`, and `decay_clock` mirrors `anneal_clock` when `decay_on=False`. Neither is read in those modes. This is state bookkeeping kept so the state shape is the same for every arm. It has no effect on results.

**Known limitations, unchanged**

- The arm selection is post hoc, and the result is a development screening diagnostic, not a promoted result.
- This PR is stacked on our own #3578, #3610, #3626, #3632, #3636 and #3638, so its diff includes their source and test changes to `ipmnist_screening.py`.
- No committed result file was changed by this follow-up.
