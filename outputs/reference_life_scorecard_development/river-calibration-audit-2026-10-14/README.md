# RiverSwim control diagnostic

This is a read-only secondary analysis of the 72 retained RiverSwim shards from
`run2-2026-10-09`; it is permanently nonpromoting and consumes no new seeds.
The analysis checks complete shard status, the frozen plan digest, the 12-seed
roster, and one shared source identity before reporting descriptive summaries.

| arm | reward sum mean ± sample SD | reward range | high-end visits mean (range) | zero-visit seeds |
|---|---:|---:|---:|---:|
| `random` | 60.483 ± 11.079 | 42.815–80.580 | 54.2 (28–85) | 0 / 12 |
| `sarsa` | 6304.942 ± 7672.066 | 93.425–15088.125 | 6340.2 (0–15310) | 3 / 12 |
| `differential_sarsa` | 96.606 ± 0.145 | 96.260–96.820 | 0.0 (0–0) | 12 / 12 |
| `prototype` | 5463.069 ± 5917.434 | 78.865–12741.715 | 5713.6 (0–13454) | 6 / 12 |
| `prototype_frozen` | 1080.111 ± 3537.927 | 1.025–12313.005 | 1467.1 (0–13000) | 6 / 12 |
| `privileged_oracle` | 17094.667 ± 102.282 | 16976.000–17336.000 | 17095.4 (16977–17337) | 0 / 12 |

## Seed-level pattern

Discounted SARSA reward totals range from 93.425 to 15088.125; its high-end visit counts range from 0 to 15310. Prototype has the same qualitative spread. Differential SARSA records zero high-end visits on every seed and remains near random reward, while the privileged oracle is included only as a scale/control reference.

Across these same 12 seeds, Pearson correlations between reward total and high-end visits are:

- `sarsa`: 0.999999.
- `differential_sarsa`: undefined (zero visit variance).
- `prototype`: 0.999995.

## Interpretation and next bounded question

The retained records are consistent with an exploration/reachability bottleneck:
successful discounted-SARSA and Prototype lives visit the rewarding end many
times, while low-return lives often do not. This is a descriptive association,
not a causal finding; high-end visits are an outcome telemetry field, and the
same consumed seeds cannot be used to select or confirm a new policy.

A fresh, separately frozen development pilot should compare the existing
discounted-SARSA control with a slower epsilon-decay schedule while retaining
matched random and finite-horizon-oracle scale controls. It should use new seed
IDs, report the complete per-seed returns and high-end visits, and stop after a
small screening roster unless the preregistered continuation rule is met. No
threshold or result from the 2026-10-09 scorecard should be changed.

Reproduce the secondary analysis with:

```bash
python3 outputs/reference_life_scorecard_development/river-calibration-audit-2026-10-14/analyze.py
```

The JSON output beside this report contains all per-seed values and the exact
plan digest. This analysis does not validate execution beyond the retained shard
consistency fields and does not establish benchmark or scientific performance.
