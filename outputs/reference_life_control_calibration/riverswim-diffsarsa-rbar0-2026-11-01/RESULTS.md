# RiverSwim differential SARSA, R̄0 = 1.0: results

Run 2026-11-01 03:42-04:16 GMT at fc5d025b4ee9a0938b17d1568f947adfba448b02, per DEVELOPMENT_PLAN.md (committed before any life). Development-only evidence.

**Verdict: PASS.** Gate on `differential_sarsa_rbar0`: mean 0.8888, sd 0.0071, one-sided 95% LCB 0.8843 (t = 2.201, threshold > 0.10). All 12 seeds kept; no VOID reasons.

## Environment check: PASS

- Seed 70000 `differential_sarsa`, 20,000 events: bit-exact with the stored scorecard shard (transcript 7fb78d72…, reward 96.25999784842134, manifest, bytes).
- Seed 70000 `random` (reward 66.21999925747514) and `privileged_oracle` (16976.0): decision fields exact; oracle sum checked under the sequential-sum versus N·g rule.
- #3308 SARSA Q0 reference life, seed 81010: bit-exact on every field (transcript a34f3cf9…, reward 2923.519999988377).

## Per seed (normalized: (reward − random) / (oracle − random))

| seed | R̄0 arm | default arm | R̄0 − default | R̄0 high-end visits | R̄0 final R̄ |
|---|---|---|---|---|---|
| 91510 | 0.8865 | 0.0030 | 0.8835 | 15403 | 0.734 |
| 91511 | 0.8915 | 0.0028 | 0.8887 | 15492 | 0.803 |
| 91512 | 0.8878 | 0.0021 | 0.8856 | 15414 | 0.853 |
| 91513 | 0.8864 | 0.0017 | 0.8847 | 15402 | 0.848 |
| 91514 | 0.8818 | 0.0024 | 0.8794 | 15317 | 0.795 |
| 91515 | 0.9014 | 0.0004 | 0.9010 | 15689 | 0.841 |
| 91516 | 0.8811 | 0.0013 | 0.8798 | 15337 | 0.881 |
| 91517 | 0.9035 | 0.0013 | 0.9022 | 15711 | 0.803 |
| 91518 | 0.8911 | 0.0023 | 0.8887 | 15485 | 0.806 |
| 91519 | 0.8842 | 0.0018 | 0.8824 | 15405 | 0.728 |
| 91520 | 0.8835 | 0.0023 | 0.8812 | 15346 | 0.691 |
| 91521 | 0.8871 | 0.0033 | 0.8838 | 15409 | 0.860 |

Descriptive: default arm mean 0.0021 (sd 0.0008, never reached the high end); paired R̄0 − default mean 0.8868 (sd 0.0075, LCB 0.8820).

## Limits

- One environment (RiverSwim, scorecard settings), one horizon (20,000), 12 seeds, one R̄0 value (1.0, the largest per-step reward). No sweep over R̄0 or step sizes.
- R̄0 = 1.0 uses knowledge of the reward scale (an upper bound on the gain), not of the optimal policy.
- Final R̄ (0.69-0.88) is below the optimal gain 0.857 on most seeds: ε stays at 0.02 and β = 0.01 makes R̄ noisy; only the end-of-life value was recorded, not a trajectory.
- Development-only evidence; it does not change scorecard defaults or stored results.
- During the run the memory guard paused and resumed workers twice; each life is a deterministic single process, so pausing does not change results.

