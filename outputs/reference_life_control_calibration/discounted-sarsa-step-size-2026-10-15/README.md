# RiverSwim eligibility-trace step-size probe

Preregistered plan; no comparison result exists at this commit.

This follows the negative lambda result in #3218 with a different causal
question: is the original step too large when eligibility traces accumulate?
The four arms form a 2-by-2 factorial: `lamda` 0 or 0.8, and `step_size` 0.05
or 0.014. The no-trace reduced-step arm distinguishes generic step-size effects
from a trace-specific interaction. The reduced step is fixed by the heuristic
`0.05 * (1 - 0.9 * 0.8)`, not a sweep or a claim of theoretical optimality.

Read [plan.json](plan.json) for the fixed contrasts, thresholds, seed roster,
schedule, event budget and limits. Twelve fresh-process full reference lives
use seeds 100601-100603, 20,000 accepted/executed events per life. None of the
previous experiment's shards or seeds is reused.

The primary contrast is reduced minus original step at fixed `lamda=0.8`.
Every paired difference must be positive and their mean exceed 0.01. The
prespecified factorial interaction is evaluated separately; neither a reference
comparison nor a secondary diagnostic can rescue primary failure. All results
will be retained, including negative or inconclusive outcomes.

Use the repository project venv from the root, one fresh process per arm/seed,
with a clean committed input tree. Write to new paths outside the checkout.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python outputs/reference_life_control_calibration/discounted-sarsa-step-size-2026-10-15/run.py --seed 100601 --arm trace_original --out /new/path/trace_original-seed100601.json
python outputs/reference_life_control_calibration/discounted-sarsa-step-size-2026-10-15/run.py --seed 100601 --arm trace_scaled --out /new/path/trace_scaled-seed100601.json
```

Execute the complete fixed schedule in the plan with at most two processes
concurrently. No retries or substitution on failure. The plan and script bytes
will remain fixed after public preregistration. A separate seed100699 API
preflight is excluded from all comparisons.

This is permanently nonpromoting development work. No framework change,
frozen-scorecard qualification, reference configuration, scientific claim,
native peak-memory or calibrated speed claim is made. The scoped attributes
preserve exact input and output bytes. Digests bind consistency, not execution
attestation.
