# RiverSwim differential SARSA with an optimistic initial reward rate: development plan

Committed before any life of this calibration was run. Times are GMT.

## Question

Does the scorecard's differential-SARSA arm pass the RiverSwim control gate when the only change is
an optimistic initial reward-rate estimate R̄0? This answers the question raised in #3184 with a
single-change arm: same manifest settings, same seeds protocol, same gate, one opt-in field.

Credit: the RiverSwim reference-life pipeline and the SARSA Q0 reference life come from #3308. Our
own Q0 calibration of SARSA is in our own #3652; it is not reused here except as background.

## Why R̄0 and not Q0

A uniform Q0 offset is a no-op for differential SARSA: the TD error
δ = r − R̄ + Q(s′,a′) − Q(s,a) is unchanged by adding a constant to every Q, so greedy choices are
unchanged (tested at core level in `tests/test_reference_life_controls.py`). Optimism has to enter
through R̄. With R̄0 above every achievable gain, every early δ is negative, so tried actions are
pushed below untried ones and the agent keeps trying actions it has not yet taken.

## Rewards, optimal rate and R̄0 (base ea6eee9d0da3fce024028810b60edfb7fa4389b4)

- `alberta_framework/streams/closed_loop.py:639-640@ea6eee9d`: `reward_left = 0.005`,
  `reward_right = 1.0`, canonicalized to float32 at 667-668 (0.004999999888241291 and 1.0). All
  other transitions pay 0 (per-step reward set {0, 0.004999999888241291, 1.0}).
- `closed_loop.py:925@ea6eee9d` `optimal_average_reward()`: 0.8571501418872549 (always RIGHT).
  Uniform random gain: 0.003042582531504523.
- **R̄0 = 1.0**, the largest per-step reward. No policy's gain can exceed the largest per-step
  reward, so R̄0 ≥ the optimal gain holds without relying on the computed optimum, and 1.0 is exact
  in float32. Choosing exactly the optimum would make optimism vanish as soon as R̄ decays slightly;
  values above 1 only add push-down with no extra guarantee.

## R̄ step size, update and decay

- `alberta_framework/core/average_reward.py:2528,2574@ea6eee9d`:
  R̄ ← R̄ + β·δ with β = `average_reward_step_size` = 0.01 (scorecard value, unchanged).
- `average_reward.py:2171@ea6eee9d`: `DifferentialSARSAAgent.init(..., average_reward=0.0)`; the
  adapter passes R̄0 only when the new field is set.
- While rewards are near 0, R̄ ≈ R̄0·(1 − β)^t: one e-fold per 100 steps, half-life ≈ 69 steps.
  Smoke check on consumed seed 70000: R̄ = 0.6116 after 50 events (0.99^50 = 0.605).
  The cumulative push-down on Q is roughly α·R̄0/β ≈ 10 with α = 0.1 (a hypothesis, not a gate).

## Adapter change

`initial_average_reward: float | None = None` on the differential-SARSA control config in
`alberta_framework/reference_life_controls.py`. Unset: config, manifest id and
transcripts are unchanged (golden manifest and 300-event transcript tests). Set: must be finite and
≥ 0 (float32-canonical, −0.0 stored as +0.0), and is written into the config and manifest.

## Protocol

- Seeds: 91510-91521 (12). Fresh: no hit in main's text or history other than one unrelated float
  value, and no hit in any open pull request diff at staging time.
- Horizon 20,000 events per life, the scorecard RiverSwim environment.
- Arms per seed: `differential_sarsa` (scorecard), `differential_sarsa_rbar0` (R̄0 = 1.0),
  `random`, `privileged_oracle`. 48 lives.
- Gate (scorecard constants): normalized score (reward − random) / (oracle − random) on
  `differential_sarsa_rbar0`, paired across seeds; PASS iff the one-sided 95% LCB
  (t = 2.201, 11 df) > 0.10. Otherwise FAIL, and FAIL is reported as is.
  The default arm and the paired rbar0 − default difference are descriptive only.

## Environment check (before any seed life)

1. Seed 70000, `differential_sarsa`, 20,000 events: bit-exact against the stored scorecard shard
   (decision fields, manifest id, transcript hash, persistent bytes).
2. Seed 70000 `random` and `privileged_oracle`: decision fields exact; the oracle sum may differ only
   as sequential sum (stored) versus N·g (runner).
3. #3308 SARSA Q0 reference life, seed 81010: bit-exact on every field.

Any mismatch: VOID, stop, no seed life is run.

## VOID rules

VOID if any life is missing, duplicated, not completed, short, non-finite, has an unexpected
parameter-change count (learning arms must change, controls must not), changed persistent bytes,
mixed runtime identity, or the oracle − random scale is not positive. VOID means stop and report:
no re-run, no seed replacement, every seed and life is kept and reported.

## Execution

`launch.sh` (one job, at most 4 one-thread workers) under `memguard.sh`; `analyze.py` writes the
verdict. Tests: `tests/test_riverswim_diffsarsa_rbar0_calibration.py`.
