# Independent RiverSwim SARSA diagnostic checks

The linear discounted-SARSA calibration records proposed in PRs
[#3220](https://github.com/SlopDotCash/asi/pull/3220) and
[#3225](https://github.com/SlopDotCash/asi/pull/3225) retain actions, rewards,
positions, and diagnostic snapshots. Their existing validators reconstruct
rewards and aggregates but explicitly do not independently replay the learner.
A finite, correctly shaped Q table or norm can therefore disagree with the
recorded updates without failing that diagnostic check.

From the repository root, run this supplemental check on an existing record:

```bash
.venv/bin/python -B tools/check_riverswim_sarsa.py /absolute/path/to/shard.json
```

It reads one file and prints JSON to stdout. Exit 0 means the recorded
diagnostics are consistent with the supported recurrence, conditional on the
assumptions below; exit 2 means malformed, unsupported, or inconsistent input.
It creates no files, imports no ASI learner, runs no environment, and consumes
no new benchmark seeds. No network request or untrusted artifact script is
executed. Both source schemas are supported; other schemas fail closed.

## Scope and limits

- The supported model has two linear action heads, no hidden layers or layer
  normalization, LMS updates, zero initial biases and traces, accumulating
  eligibility, one-hot observations, and continuing unit environment discount.
  Discounted SARSA still applies its configured gamma in its own TD target.
- Initial Q values and the complete action sequence are inputs, not verified
  outcomes of initialization or the policy. Source/runtime provenance,
  epsilon-greedy choices, RNG keys, and environment stochastic samples are
  **not verified**. Use the original artifact validator and provenance checks
  as well; this tool does not replace them.
- The independent NumPy equations check Q values, eligibility L2/max norms,
  and parameter L2 norms. At lambda zero the production implementation freezes
  inactive head trace storage; these inactive traces receive no update credit
  but remain present in its diagnostic norms. This detail is checked too.
- The input schemas omit the final next action, although SARSA uses it in the
  last bootstrap. The checker evaluates both actions against the *complete*
  final snapshot and reports `consistent_final_next_actions`. Multiple matches
  remain ambiguous. A match does not certify which action was sampled.
- NumPy and XLA may round fused operations differently. All scalar diagnostic
  and Q comparisons use absolute tolerance `1e-4`, with zero relative
  tolerance, reported in the result. Smaller discrepancies are not ruled out.
  This diagnostic tolerance is not a qualified scientific acceptance gate or
  a portable exact-resume guarantee, and must not be relaxed to rescue a run.
- Input is bounded to 32 MiB, 20,000 events, 2–12 states, and 32 snapshots.
  Unsupported configurations and malformed, duplicate-key or nonfinite JSON
  are rejected. Resource counts, summaries, experiment plan compliance, and
  source manifests remain the original validators' responsibility.

Successful replay provides additional consistency evidence, not authenticated
execution or proof of learning benefit. Every result remains permanently
nonpromoting and cannot populate `reference-dev`. Original records, failed
gates, source manifests, and consumed seed roles must remain unchanged.

## Local verification

```bash
.venv/bin/python -m pytest tests/test_riverswim_sarsa_replay.py -q -p no:cacheprovider
.venv/bin/python -m ruff check tools/check_riverswim_sarsa.py tests/test_riverswim_sarsa_replay.py
.venv/bin/python -m mypy --strict tools/check_riverswim_sarsa.py
```

The tests compare the independent recurrence to short deterministic sequences
through the public SARSA update, including lambda zero and nonzero, then
reject finite snapshot corruption. These are correctness tests, not a
benchmark campaign or scientific evaluation.

Small 100-event prefixes derived from both PRs are kept under
`tests/fixtures/riverswim_sarsa/`, with their original source commits and PR
links. They are explicitly derived test inputs, not original shards or new
evidence. Tests never fetch GitHub or execute those PRs' scripts.
