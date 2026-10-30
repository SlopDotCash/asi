# PREREGISTRATION: anneal peak on rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000

Frozen before any arm run. Development screening diagnostic: nonpromoting.
Branch `screen/identmap50-r-idet003-s005625-anneal-peak`, cut from 3c38b025.

## Arm
`rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_p{75,10,15}` is the parent spec plus `anneal_peak` = P, with P ∈ {0.0075, 0.01, 0.015}.

The body step at s steps since the last detected shift is
`max(0.0028125, P * (1 + s/1000)^-1/2)`.
Every other hyperparameter is the parent's, including `step_size` 0.005625. If `anneal_peak` is absent, P = step_size, which is the parent bit-exactly (pinned).

## Parent
`rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000` (confirmed in #3580, holdout #3585).

## Seed plan
All stages run 60 tasks × 5000 steps.

1. **Tune (seed 0 only).** Run all three peaks on seed 0. Choose the P with the highest seed-0 paired mean gain over the parent. If no peak beats the parent on seed 0, STOP (negative). Seed 0 is reported separately and is never part of the screen or the bars.
2. **Screen (seeds 1-3, chosen P).** Continue only if the mean paired gain is > 0.
3. **Confirmation (seeds 4-13, chosen P).** The bars below apply to these 10 seeds only.

## Bars (seeds 4-13, paired vs parent, all required)
- Mean paired gain (mean per-task accuracy, arm − parent) ≥ +0.002.
- At least 7/10 seeds improved.
- 95% t CI lower bound > 0, using t = 2.262 (df = 9).

The result is a PASS only if all three bars hold. Every seed is listed, along with the 15-task block gains. Nothing is dropped.

## Parent results (no parent re-runs except the environment check)
A 60-task run is a bit-exact prefix of the 200-task run on the same seed. This was verified for seeds 0-9: the 3c38b025 wave shards equal `per_task_accuracy[:60]` of the 612491db 200-task shards.

The parent at 60 tasks is therefore taken as the first 60 entries of the stored #3580 200-task shards (612491db) for seeds 0-13. These are copied with their SHA256 recorded.

## Environment check (before any arm run; any mismatch means VOID, stop and report, no automatic re-run)
1. Control `upgd_w_control`, seed 0, 60 tasks, compared against the stored 3c38b025 shard.
2. Parent `..._anneal_f2t1000`, seed 0, 60 tasks, compared against the stored 3c38b025 wave shard.

`per_task_accuracy` must be bitwise equal in both cases.

## Pin tests (tests/test_ipmnist_screening.py)
- `anneal_peak == step_size` gives a result bitwise equal to the parent.
- A deliberate break, `anneal_peak = step_size * 1.0001`, must differ from the parent.
- Invalid values are rejected: `anneal_peak` < floor, `anneal_peak` ≤ 0, and `anneal_peak` = inf or nan.

## Runtime (one heavy slot, at most 4 one-thread workers, 3 GB available-RAM floor)
A 60-task seed takes about 2.9 min single-thread.

| Stage | Runs | Estimated time |
|---|---|---|
| Environment check | 2 parallel | about 3 min |
| Tune | 3 parallel | about 3 min |
| Screen | 3 parallel | about 3 min |
| Confirmation | 10 runs, 4 workers | about 9 min |

The total is about 20 min of wall time.

## Prior
P(PASS) ≈ 10%. The expected gain is +0.0005 to +0.0015.

The headwind is that on the gated chain the constant step kept improving below 0.005625, down to 0.0031640625 (#3118, #3121 and #3122: the step axis closed at 0.0031640625). Larger constant steps were worse.

The case for the arm is that the step was calibrated without decay. The anneal now controls late-task noise, and the early-task start has never been varied. On the other side, the f4t1000 negative (#3586) showed that the late-task side is used up.

## Disclosures
- The arm was chosen after the f4t1000 negative, using post-hoc reasoning from its per-block gains.
- Seed 0 is used for tuning and was also the environment-check seed in earlier waves.
- Parent seeds 0-9 at 60 tasks were in the 3c38b025 screen.
- Parent seeds 10-13 already appeared at the 60-task prefix inside the 200-task, 20-seed confirmation (#3580). The arm was never run on seeds 1-19.
- The parent results are reused, not re-run, apart from the environment-check seed.
- The confirmation is 60 tasks only. A 200-task confirmation is needed before any promotion claim.
