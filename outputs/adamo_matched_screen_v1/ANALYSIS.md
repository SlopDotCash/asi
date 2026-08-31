# AdamO matched bounded-development screen — analysis (v1)

Permanently nonpromoting development measurement. Fulfills the empirical
acceptance criteria of [#1560](https://github.com/SlopDotCash/asi/issues/1560)
(comparator merged in [#1620](https://github.com/SlopDotCash/asi/pull/1620)).

- Pre-registration (frozen before seeds 15601-15603 ran):
  <https://github.com/SlopDotCash/asi/issues/1560#issuecomment-5473394736>
- Runner: `asi-adamo-diagnostic`, schema `asi.adamo_dynamical_isometry_diagnostic.v1`,
  profile `bounded-development` (784-300-150-10 ReLU MLP, 8 tasks x 64 steps,
  one example per step), four frozen development seeds x four arms.
- Dataset: caller-materialized NPZ from `load_mnist_train()` (OpenML `mnist_784`
  v1, first 60,000 rows, [-1,1]) validated through `validated_ipmnist_data`;
  sha256 `eba9c88497f36270265fa7bea0e3f69661a8679bfe2ac474d94bcbcd47702b3a`;
  bound in every receipt.
- Retention: verbatim receipts in `receipts/` (sha256 manifest inside
  `analysis_summary.json`); analysis implementation in `analyze_adamo.py`
  implements the frozen plan and nothing else.
- Measured on a clean tree at main head `c9aba7b54dedd647f8bd5f5c7bf6780b1413b676`
  (receipts bind the exact adapter/runner bytes by SHA-256). Runtime: CPU,
  JAX 0.11.0, NumPy 2.5.1, Python 3.14.4.

## Outcome: inconclusive-negative (retained)

At the matched 512-step qualification budget the AdamO decoupled Gram-isometry
step (lambda=1e-3, paper strength) produces **no measurable isometry
stabilization and no consistent accuracy, loss, or plasticity effect** against
its own exact mechanism-off reduction. The result is retained as a negative /
inconclusive outcome per the issue's acceptance criteria; it excludes nothing
at the paper's horizons and makes no claim beyond this budget.

## Inert reduction (bit-exact, all seeds)

`adamo_inert` equals `adamw_control` bit-for-bit on all four seeds (identical
per-task accuracy trajectories and final `parameter_sha256`):
`[True, True, True, True]`.

## Primary paired comparisons (mean per-task over 8 tasks, per seed)

`adamo_l1e3 - adamo_inert` (mechanism effect at matched budget):

| field | per-seed deltas (15600..15603) | mean | all-positive |
|---|---|---|---|
| accuracy | +0.005859, 0.0, +0.009766, 0.0 | +0.003906 | no (two exact ties) |
| loss | +0.000020, -0.000030, -0.000128, -0.000004 | -0.000035 | no |
| plasticity | -0.000313, +0.000248, +0.000896, +0.000012 | +0.000211 | no |

`adamo_l1e3 - adam_iso_joint_l1e3` (decoupled vs joint gradient mixing):

| field | per-seed deltas | mean | all-negative |
|---|---|---|---|
| accuracy | -0.007813, -0.013672, +0.003906, -0.003906 | -0.005371 | no |
| loss | +0.008267, -0.001560, -0.001859, +0.001669 | +0.001629 | no |
| plasticity | -0.016416, -0.003281, -0.000555, -0.005968 | -0.006555 | **yes** |

No arm shows an all-seeds-positive paired improvement; the only frozen rule
that fires is the decoupled-vs-joint plasticity contrast (all seeds negative
for the decoupled form: joint moment mixing moved faster and lost accuracy
meanwhile — direction-consistent with, but far weaker than, a plasticity/
accuracy trade-off claim, so it is recorded, not claimed).

## Jacobian / isometry readout (final task, sentinel row zero)

| arm | cond (clipped 1e12) | RMS dist from 1 | min/max singular value | weight-Gram penalty |
|---|---|---|---|---|
| adamw_control | 2.28 | 0.9057 | 0.0549 / 0.1181 | 224.69 |
| adamo_inert | 2.28 | 0.9057 | 0.0549 / 0.1181 | 224.69 |
| adamo_l1e3 | 2.45 | 0.9109 | 0.0526 / 0.1290 | 224.67 |
| adam_iso_joint_l1e3 | 2.12 | 0.8765 | 0.0877 / 0.1860 | 178.27 |

(per-seed values in `analysis_summary.json`; the table shows seed 15600.)
The penalty did not pull the sentinel Jacobian toward isometry at this budget:
RMS distance from one and the clipped condition number are unchanged-to-worse
for `adamo_l1e3` versus the reduction control, and the weight-Gram penalty is
essentially unmoved. The joint-mixing ablation perturbs the Jacobian more
without an accuracy benefit.

## Resources (per arm-seed run; receipts bound)

data steps 512, updates 512, observations 512, model queries 1032, Jacobian
reverse rows 80, parameters 282,160, persistent numeric bytes 3,386,040, peak
Gram working bytes 360,000, logical compute 600,436,480 parameter-touch units;
timing telemetry-only (about 4.4 s/arm on this CPU runner, about 22 s per
four-arm seed run).

## Declared identity

Provider `zai`, model `glm-5.3-flash`, client `Claude Code`; self-reported.
A signed measured-run receipt and its private trace cover the session that
produced these artifacts.
