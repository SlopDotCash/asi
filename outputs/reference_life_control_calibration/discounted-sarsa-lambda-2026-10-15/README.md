# RiverSwim discounted SARSA eligibility-trace diagnostic

Completed permanently nonpromoting development comparison. Decision:
`bounded_hypothesis_negative`. The predeclared primary rule is not rescued by any
secondary diagnostic. No candidate or reference configuration is selected.

## Hypothesis and fixed comparison

Only discounted SARSA `lamda` changes: `0.0` versus `0.8`. Eligibility traces
may propagate a delayed upstream reward faster, but may also reinforce the small
left reward. This tests the existing implementation, not an external paper port.
The adapter forwards `lamda` to `SARSAAgent`; inactive action-head traces receive
the shared TD error. The per-step decay is approximately `gamma * lambda = 0.72`.

The complete [plan](plan.json) and [execution script](run.py) were committed at
`023726c8b643cc332b73e220d67c26b4a886f334`, then opened as draft
[PR #3218](https://github.com/SlopDotCash/asi/pull/3218) before comparison execution.
The original plan/script bytes are unchanged. This differs from #3214's epsilon
decay, #3184's differential-SARSA calibration and #3193's retained-record audit.

All six new lives ran through the existing `ReferenceLifeRunner`: two arms,
three fresh development seeds (100401, 100402, 100403), 20,000 accepted/executed
events each. The original scorecard controls, gates and artifacts are unchanged.
The control was remeasured for every paired seed; no result from #3214 is reused.
Initial action-value tables match exactly within each pair. Both arms keep the
default six-state RiverSwim, gamma 0.9, step size 0.05, empty hidden network and
epsilon 0.5 to 0.02 over 2,500 steps. Only `lamda` differs in the recorded configs.

## Primary result

The rule was fixed before execution: every paired lifetime mean-reward delta
must be positive, and their mean must exceed 0.01. Values below are reward per
transition, not total lifetime return.

| Development seed | Lambda 0 | Lambda 0.8 | Paired delta |
| --- | ---: | ---: | ---: |
| 100401 | 0.735496750 | 0.007047500 | -0.728449250 |
| 100402 | 0.745071500 | 0.004780500 | -0.740291000 |
| 100403 | 0.004782500 | 0.004771250 | -0.000011250 |

Baseline: **0.495116917 +/- 0.424669047**. Trace arm:
**0.005533083 +/- 0.001311531**. Paired difference:
**-0.489583833 +/- 0.424023634**. These are means +/- sample standard
deviations over three paired development seeds, not confidence bounds.

Two pairs have large losses; seed 100403 is effectively a near-tie at the small
left reward, with neither arm discovering the upstream reward. The three
negative signs therefore do not imply three equally large effects.

## Fixed secondary diagnostics

Each cell below lists baseline / trace. `None` means no upstream reward was
observed before the fixed horizon; it is censored, not a zero event index.
Rightmost visits count post-transition states; reward uses the pre-transition
state and selected action, so these are different counts.

| Seed | First upstream reward event | Rightmost visits | Last-2000 mean reward | Parameter-change events |
| --- | ---: | ---: | ---: | ---: |
| 100401 | 661 / 86 | 14931 / 70 | 0.800500000 / 0.004930000 | 20000 / 18985 |
| 100402 | 943 / None | 15107 / 0 | 0.826500000 / 0.004925000 | 20000 / 19268 |
| 100403 | None / None | 0 / 0 | 0.004947500 / 0.004947500 | 20000 / 19680 |

Each record also retains all six action values (two per state) at accepted-event
counts 0, 100, 1000, 2500, 5000, 10000 and 20000. These diagnostic snapshots are
read-only linear-head weights plus biases and consume no RNG. They describe
policy evolution; they do not isolate the causal roles of shared bias, trace
credit and reward discovery. No unplanned ablation is substituted for this test.

Seed 100401 separates discovery from sustained exploitation: the trace arm first
earns the upstream reward at event 86, earlier than the baseline's 661, yet its
last-window mean falls to 0.00493 while baseline reaches 0.8005. At the fixed
event-100 snapshot its state-0 values favor right (0.75802 versus 0.30561 for
left); by event 1000 they favor left (0.04021 versus 0.03224 for right). These
observations contradict the predicted lifetime benefit at this setting. They
do not establish which component causes the reversal; that would require a
separately planned intervention.

## Resources, identity and execution

All 120,000 planned events completed. Initial and final numeric control payloads
are equal in every shard: 220 bytes, 55 elements, 23 leaves (17 floating leaves).
This is numeric agent state only; total process/native peak memory, environment
and runner allocations are not included. Lambda adds trace arithmetic, so equal
payload does not establish equal compute cost. Parameter-change counts above
may differ despite equal event and update-attempt budgets.

At most two fresh CPU processes ran concurrently. `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were 1; JAX preallocation was false.
Per-shard wall seconds remain telemetry, not a qualified latency comparison.
Runtime: Windows AMD64, CPython 3.12.4, JAX/jaxlib 0.11.0, NumPy 2.5.1,
Chex 0.1.92. Every shard binds the same clean input commit and all 228 framework,
project-config and lockfile byte digests. Each recorded source digest was also
checked against that commit's Git blob. Framework bytes match base
`415e7d225a4338492275be6cafa69d87c158c30a`; no implementation was edited.

Two one-event API preflights at seed 100499 completed before comparison, one
per arm, and are excluded. There were no comparison retries, seed substitutions,
partial shards, source-drift attempts or parameter changes after observing data.
Git transport failed before execution; the identical preregistration tree and
commit were published through GitHub's Git API and their SHAs matched locally.

## Reproduction and independent verification

From the repository root, use the project venv and the thread settings above:

```sh
python outputs/reference_life_control_calibration/discounted-sarsa-lambda-2026-10-15/run.py --seed 100401 --arm baseline --out /new/path/baseline-seed100401.json
python outputs/reference_life_control_calibration/discounted-sarsa-lambda-2026-10-15/run.py --seed 100401 --arm trace --out /new/path/trace-seed100401.json
```

Repeat exactly for seeds 100402 and 100403, in fresh processes, at the frozen
input commit, with new output paths outside the checkout and at most two workers.
The original roster is already consumed development data; do not reclassify a
replay as fresh evidence or use these seeds for scientific promotion.

The independent [summary tool](summarize.py) validates the complete roster,
source/runtime/plan/script equality, one-variable config difference, counters,
resource endpoints, every reward against its prior-state/action lattice, every
transition's support, all aggregate metrics and the fixed snapshot inventory.
It re-derives [summary.json](summary.json) from all raw records:

```sh
python outputs/reference_life_control_calibration/discounted-sarsa-lambda-2026-10-15/summarize.py --records outputs/reference_life_control_calibration/discounted-sarsa-lambda-2026-10-15 --inputs outputs/reference_life_control_calibration/discounted-sarsa-lambda-2026-10-15 --source-root . --out /new/path/recomputed-summary.json
```

The runner itself validates each keyed stochastic transition. The independent
summary checks support and reward arithmetic; it does not independently attest
execution. Digests bind consistency, not authenticity. MANIFEST.sha256 includes
the retained plan, run script, summary tool, report and six complete raw traces;
the scoped `.gitattributes` prevents newline conversion of captured bytes.

Validation: existing reference-control tests **34 passed**; repository Ruff and
explicit retained-script Ruff passed; `mypy --platform linux` passed for 226
source files; `git diff --check` passed. The five evidence-registry statuses and
error lists remain identical to the saved baseline (exit 2). Full fast/scientific
suites were not rerun for this evidence-only change on Windows.

## Limits

This rejects or bounds only the stated lambda setting under this initialization,
reward structure, constant step size, shared linear-head bias, exploration
schedule and horizon. Three development seeds do not establish a confidence
claim, general SARSA(lambda) inferiority, transfer or retention. This never
qualifies the frozen scorecard, populates `reference-dev` or creates scientific
evidence. Any follow-up needs a new causal hypothesis and separately declared
comparison; these results cannot justify post-hoc tuning of the frozen gates.
