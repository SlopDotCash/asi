# RiverSwim discounted SARSA eligibility-trace diagnostic

Status: plan and execution script frozen before the six comparison lives.
This directory currently contains no comparison result.

The single changed variable is `lamda`: baseline `0.0` versus candidate `0.8`.
All environment, initialization, exploration and step-size defaults stay equal.
The causal hypothesis, complete seed roster, bounded budget, fixed diagnostics
and decision rule are in [plan.json](plan.json). This uses the existing
discounted SARSA implementation, not a port or replication of an external paper.
The adapter forwards `lamda` to the learner; its inactive action-head traces
receive the shared TD error in `alberta_framework/core/sarsa.py`.

This is separate from #3214's exploration-decay comparison, #3184's differential
SARSA calibration and #3193's descriptive audit of retained shards. The six
comparison lives use new development seeds 100401, 100402 and 100403. A one-event
API preflight in each arm used seed 100499, excluded from the comparison.

Run from the repository root using the project venv. Each arm/seed is a fresh
process; the output path must be new and outside the checkout during collection
so that every process begins from the same clean input commit.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python outputs/reference_life_control_calibration/discounted-sarsa-lambda-2026-10-15/run.py --seed 100401 --arm baseline --out /new/path/baseline-seed100401.json
python outputs/reference_life_control_calibration/discounted-sarsa-lambda-2026-10-15/run.py --seed 100401 --arm trace --out /new/path/trace-seed100401.json
```

Repeat exactly for seeds 100402 and 100403, with at most two processes at once.
No retries, replacement seeds, faster substitute execution or post-result
parameter changes are authorized by this plan.

Development only, permanently nonpromoting: no frozen-scorecard qualification,
`reference-dev` population, transfer/retention result or scientific claim.
Numeric agent payload bytes are measured separately from total process/native
memory. Timing is telemetry, and hashes bind consistency, not execution
attestation. The scoped `.gitattributes` preserves the captured input bytes.
