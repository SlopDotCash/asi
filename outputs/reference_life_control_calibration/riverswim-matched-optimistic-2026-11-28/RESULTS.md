# Matched full-horizon RiverSwim controls

Six fresh development seeds completed all four predeclared lives at 20,000 accepted events. The 24 raw records cover 480,000 events.

Retain both optimistic controls. The sign of the difference varied across pairs or included a tie.

The mean normalized return was 0.889834 for discounted SARSA and 0.887927 for differential SARSA. The paired difference (discounted minus differential) was +0.001907, with standard error 0.002275. Discounted led on 3 seeds, differential on 3, with 0 ties. The common mean oracle-minus-random denominator was 16938.571667.

| Seed | Random reward | Oracle reward | Discounted reward | Differential reward | Normalized difference | High-end visits, discounted / differential |
|---:|---:|---:|---:|---:|---:|---:|
| 612670 | 55.375 | 17056.000 | 15117.500 | 15211.495 | -0.005549 | 15349 / 15457 |
| 612671 | 62.565 | 17092.000 | 15296.385 | 15124.530 | +0.010146 | 15566 / 15391 |
| 612672 | 64.470 | 17113.000 | 15162.485 | 15180.385 | -0.001057 | 15426 / 15466 |
| 612673 | 43.265 | 16867.000 | 14944.505 | 14839.590 | +0.006194 | 15163 / 15052 |
| 612674 | 48.535 | 16809.000 | 15034.420 | 15001.480 | +0.001945 | 15264 / 15250 |
| 612675 | 72.360 | 17041.000 | 15226.415 | 15230.420 | -0.000236 | 15474 / 15486 |

[Plan](PLAN.md) · [Raw lives](lives/) · [Unrounded statistics, hashes, and runtime](summary.json) · [Resource measurements](resources.tsv)

Each life ran in a fresh CPU process. Wall time and resident memory describe this Mac during concurrent use. The two optimistic controls share the same seed, lifecycle and normalizers within each pair. Different retained adapter runners bind their own source dependencies. The recorded run source is the published plan commit `06702c614899f6ae5417b77dc05b922e3dc766ab`. This six-seed development comparison has no scorecard or scientific promotion authority.

Recalculate from the repository root with the Python standard library.

```sh
d=outputs/reference_life_control_calibration/riverswim-matched-optimistic-2026-11-28
python3 "$d/summarize.py" "$d/lives" | diff "$d/summary.json" -
```
