# Preregistration: constant weight_decay sweep on anneal_f2t1000 (development screen, nonpromoting)

Written before any run on this branch. The branch is cut from 3c38b025 and carries six `-x` cherry-picks:
- partial-reset code and tests: 7d7cbf8d, fd2b1b18;
- decay-clock code and tests: 6c4c6029, 9e77f760;
- the constant-wd arms with the non-negative guard, and their tests: 3907238d, 202c8329.

The constant-wd arm commit sits inside the same registry blocks as the partial-reset and decay-clock arms, so those commits are needed for a clean pick. With their knobs off they leave the parent unchanged; this is pinned bit-exact.

## Origin (post hoc, disclosed)
The decay follow-up (#3596) found the decay-only gain to be AMOUNT-ONLY. Its descriptive extra arm, a plain weight_decay of 0.005, gained +0.008022 on seeds 30-39, CI [+0.007840, +0.008204], 10/10 seeds. That arm was outside the decision rule.

The parent's weight_decay of 0.01 is the **untuned UPGD-W published default** (`_upgd_hp`, inherited by `_rls_head_hp`). No arm in this lane ever changed it.

The coupling: `decay = jnp.where(lr == step_size, param_decay, 1 - lr*weight_decay)` in `_gated_sgd`. So weight_decay scales the decay along the whole anneal schedule.

Ancestor evidence, on other families:
- `upgd_w_wd0005` was better: 0.78431 vs 0.779147 (10 seeds, 200 tasks).
- `upgd_ema_norm_wd0005` was worse: 0.84745 vs 0.85362 (3 seeds).

## Duplicate check
Done before branching; full output is in wd-sweep-plan.md. No PR or issue lowers or tunes weight_decay on identmap, rls_head or anneal arms. No ref adds such a registry arm except our own #3596 branch.

## Seeds
- **Development and tuning seed: 30.** It was used in #3596, so it is already seen. Using it for both development and tuning is disclosed.
- **Tie-break seeds: 31-39.** Also seen in #3596; used only if needed.
- **Evaluation seeds: 40-49.** Never run for this family. Method:
  - `git log --all --name-only` over all 3143 refs, which include 3042 PR heads: 381 family files, seeds 0-39 only.
  - `find` over every pumpdev worktree and job list.
  - Zero hits for seeds 40-49 in both.

## Arms (60 tasks × 5000 steps)
- Parent: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`, weight_decay 0.01.
- Sweep arms `…_wd{0005,00025,0001,0}`, weight_decay {0.005, 0.0025, 0.001, 0}. Each is the parent with only weight_decay changed.
- weight_decay = 0 is safe on this path. Decay becomes 1 - lr·0 = 1, and weight_decay is never divided by and never passed through a log or square root. It is pinned finite.

## Stages and gates (fixed; no further tuning)
1. **Env check.** Re-run three stored shards and compare bit for bit:
   - control seed 0 and parent seed 0, against 3c38b025;
   - `…_wd0005` seed 30, against 11a5a7ff.

   Any mismatch or failed job: VOID, stop, no automatic re-run.
2. **Tune (seed 30).** Run the parent and wd {0.0025, 0.001, 0}. The wd0005 seed-30 result is the env-check shard, already verified identical to the stored one.
   - If no value beats the parent (paired gain > 0), STOP.
   - Freeze the value with the best seed-30 gain.
   - **Tie rule:** any value within 0.0005 of the best gain is tied. Tied values are scored by their mean gain over seeds 30-39, using the stored 11a5a7ff parent and wd0005 shards and running missing tied arms on seeds 31-39. Highest mean wins; exact equality goes to the larger weight_decay.
3. **Screen (seeds 40-42).** Run the parent and the frozen value. The mean paired gain must be > 0, else STOP.
4. **Confirm (seeds 43-49).** Run the parent and the frozen value.
5. **Bars on all of seeds 40-49 (10 seeds).** All three must hold for PASS:
   - mean paired gain ≥ +0.002;
   - at least 7/10 seeds improved;
   - 95% t CI lower bound > 0 (t = 2.262, df = 9).

   Otherwise FAIL. Report every seed and gains by 15-task block.

## After a PASS (not part of this run)
A PASS is a 60-task development screen. It stays nonpromoting until a separate preregistered **200-task, 20-seed confirmation**, which is needed because lower decay can trade long-horizon plasticity for short-horizon accuracy.

Free seeds for that: **50-69**, unseen at any task count. The 200-task confirmation of the parent family in #3580 used seeds 0-19, and seeds 40-49 are spent here.

## Prior
- P(some value beats the parent on seed 30) ≈ 97%.
- Frozen value: 0.0025 about 40%, 0.001 about 25%, 0.005 about 20%, 0 about 15%.
- P(screen passes) ≈ 92%.
- P(all bars pass on 40-49) ≈ 85%.
- P(a later 200-task confirmation holds) ≈ 55%.

## Runtime
4 one-thread workers, 3 GB RAM floor:

| Stage | Runs | Time |
|---|---|---|
| Env check | 3 | about 9 min (control-bound) |
| Tune | 4 | about 3 min |
| Screen | 6 | about 5 min |
| Confirm | 14 | about 9 min |
| **Total** | | **about 26 min** |
| Tie-break, if triggered | up to 27 | up to 17 min more |

## Disclosures
- Post hoc after #3596 (and #3593, #3589).
- The step/decay coupling above.
- 0.01 is the untuned UPGD-W default.
- Seed 30 is used for both development and tuning.
- Ancestor evidence is mixed: wd0005 helped on UPGD-W and hurt with the EMA normaliser.
- The non-negative weight_decay guard (from #3596) is carried.
- The branch base 3c38b025 is behind main (ee9df14c).
- Development screen only; nonpromoting.
