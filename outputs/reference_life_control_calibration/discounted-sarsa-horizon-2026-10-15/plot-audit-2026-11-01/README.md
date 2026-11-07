# Post-collection checked plotting entry point

The frozen `plot.py` is retained byte for byte. This independent supplemental
entry point addresses review observations without changing the collected data,
original figure or preregistered experiment.

`plot_checked.py` checks each expected arm and seed, the corresponding gamma,
lambda and step-size configuration, accepted event count, shard schema, all
three trace lengths and finite numeric rewards in the plotted 0 to 1 range
before rendering. It uses the fixed 20,000-event panel and rejects other horizons instead of silently drawing
a misleading axis. Endpoint subsampling includes the last point only once.
Nested output directories can be created. Outputs cannot be placed inside the
frozen experiment or input directory and existing figures are never overwritten.

This is a descriptive plotting guard, not an independent science validator,
replay, resource attestation or evidence-promotion decision. The original
experiment validator and the separate resource-policy audit remain responsible for reward-lattice, provenance, resource and decision
checks. The original manifest does not cover this new supplement, which has
its own manifest.

Run from the repository root with the project environment, specifying a new
output directory outside the experiment and record directories:

```text
python outputs/reference_life_control_calibration/discounted-sarsa-horizon-2026-10-15/plot-audit-2026-11-01/plot_checked.py --records outputs/reference_life_control_calibration/discounted-sarsa-horizon-2026-10-15 --out /new/outside/path
python -m pytest tests/test_riverswim_discount_plot_checked.py -q -o addopts=""
```
