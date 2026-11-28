# RiverSwim optimistic controls on a shared full-length protocol

The [reference-life status](../../../docs/status.md#reference-life-hillclimb) retains two optimistic controls. Their 20,000-event comparisons used different seeds. A three-seed head-to-head at 5,000 events omitted random and privileged-oracle normalization and split directions. This development comparison asks whether either optimistic control consistently leads when both run for 20,000 events on the same streams with both normalizers. The project scorecard and its original gates remain frozen.

Use six development seeds 612670 through 612675, chosen before outcomes were observed. On each seed run discounted SARSA with initial Q = 10, differential SARSA with initial average reward = 1, uniform random and the privileged finite-horizon oracle. Use the two archived calibration runners and their current-source dependency paths, the same lifecycle identifier across arms on a seed, the CPU backend, one fresh Python process per life and exactly 20,000 accepted events. Record all 24 raw outcomes including negative and failed lives, source and runtime bindings, state bytes, transcript digest, accepted events and wall time as telemetry. Use the project environment and no external service. The launcher uses at most two simultaneous lives and rotates arm order across seeds.

The primary statistic is the paired normalized reward difference, discounted minus differential. Divide each seed's difference by the common mean oracle-minus-random reward across all six seeds, matching the retained normalization. Report all paired outcomes, mean, standard error and sign count. Also report normalized returns of both controls and high-end visits. A nonfinite or nonpositive denominator, a failed or incomplete life, a missing arm, or inconsistent source/runtime identity within a runner voids the comparison. Use the shared source fingerprint for the common dynamics and separately record the two adapter-runner fingerprints. Only a strict common direction on all six seeds identifies a directional lead for this screen; otherwise retain both. There is no tuning or alternative statistic for selection. The sample is a six-seed development comparison. It has no scientific-promotion or full-scorecard authority.

From the repository root, choose an empty output directory. Run one arm/seed per process with the committed driver. The driver writes each record once. A valid four-arm set uses the arm names listed by its help. Run each seed and arm exactly once.

```sh
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 JAX_PLATFORMS=cpu
dir=outputs/reference_life_control_calibration/riverswim-matched-optimistic-2026-11-28
.venv/bin/python "$dir/run_life.py" --help
# Example for the first life. Repeat in fresh processes for all four arms and six seeds.
.venv/bin/python "$dir/run_life.py" sarsa_q0 612670 "$output_directory"
```
