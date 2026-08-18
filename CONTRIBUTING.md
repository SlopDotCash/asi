# Contributing to ASI

ASI is a continual-learning hillclimb. Contributions are judged by the uncertainty they resolve
and the end-to-end progress they enable, not by diff size or the number of new mechanisms.

Contributions are accepted through GitHub pull requests against `main` and reviewed under the
repository's normal maintainer process. By submitting a contribution, you agree that it is
licensed under the repository's Apache-2.0 license and that you have the right to submit it on
those terms. Maintainers decide whether to accept, reject, hold, or request changes.

Before starting, read the [research roadmap](docs/research/asi-roadmap.md),
[current status](docs/status.md), [SOTA landscape](docs/research/sota-landscape.md),
[evidence methodology](docs/evidence/methodology.md), and
[negative-results ledger](docs/evidence/negative-results.md). For Forager work, also read
[FORAGER_BENCHMARK.md](FORAGER_BENCHMARK.md).

## Choose an uphill contribution

The strongest contributions do at least one of the following:

- complete or challenge the current reference-life scorecard;
- reproduce a competitive published method under an ASI protocol;
- connect an existing mechanism to a real observation-to-action consumer;
- disprove a plausible mechanism hypothesis with a cheap, decisive ablation;
- demonstrate recurrence, transfer, or control benefit for a subsystem winner;
- close an ownership, checkpoint, resource, safety, or application gate; or
- improve strict reproducibility, provenance, or claim validation.

A new API without a named consumer and metric is usually not uphill. A favorable benchmark result
without a matched control, resource accounting, and protocol identity is not a result ASI can
use.

## Propose the experiment before the expensive run

An experiment proposal should answer:

1. **Bottleneck:** What observed failure or uncertainty limits the current baseline?
2. **Hypothesis:** What mechanism should change which metric, and why?
3. **Control:** What current-source baseline and contemporary competitor will be rerun?
4. **Ablation:** What smallest comparison distinguishes the proposed mechanism from capacity,
   compute, replay, pretraining, task cues, or tuning?
5. **Protocol:** What stream/environment version, horizon, schedule, seeds, primary statistic,
   uncertainty calculation, and multiple-comparison rule are fixed?
6. **Resources:** What trainable and persistent state, examples, updates, replay, model queries,
   wall time, peak memory, and latency are charged?
7. **Failure condition:** What result ends or redirects the idea?
8. **Integration:** Which reference-agent path could consume the winner, and what regression panel
   must it pass?

Use development data to choose and reject ideas. Do not describe a development protocol as
preregistered, held out, or promoting. A scientific evaluation is a separate, explicitly
authorized protocol frozen before its untouched seeds are observed.

## Reproducing a paper

The [research library](docs/research/sota-landscape.md) is the comparison backlog. When adding or
updating an entry, record:

- paper title, stable primary-source link, version/date, and official code when available;
- the exact claimed setting and metric;
- architecture, optimizer, task information, replay/pretraining, and update budget;
- why the result is or is not comparable to an ASI lane;
- local status: unreviewed, paper-audited, implementation-planned, implemented, smoke-checked,
  development-screened, development-confirmed, or scientifically evaluated;
- the smallest faithful implementation and an inert/reduction test where possible; and
- licenses or dependency constraints that affect reuse.

Do not copy a reported number into an ASI leaderboard unless the protocol is demonstrably the
same. Permuted-MNIST variants often differ in task count, examples per task, batch size, boundary
signals, replay, pretraining, evaluation timing, and metric. World-model comparisons often differ
in observation modality, replay ratio, pretraining data, planning budget, and model size. Report
those differences rather than normalizing them away in prose.

For a faithful local comparator:

- pin the source revision and dependency/runtime identities;
- preserve the authors' method before adding ASI adaptations;
- test the published update rule on a small deterministic case;
- add a reduction test showing that disabling the mechanism recovers its declared base when the
  mathematics says it should;
- rerun the base in the same runner, process model, and seed schedule; and
- keep paper reproduction, protocol adaptation, and novel composition as separate arms.

## Development workflow

Use Python 3.12 and the project virtual environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Library changes are failing-test-first. State is normally an immutable Chex dataclass/JAX PyTree,
and all stochastic roots use explicit `jax.random` keys. Keep `requires-python >= 3.12`,
`numpy >= 1.26`, and the existing `alberta_framework` import surface intact because the sibling
robot track imports it in-process.

Run the narrowest useful checks first:

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -q
.venv/bin/python -m pytest -m "not slow and not package" tests -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
```

Before handoff, broaden in proportion to risk:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest --collect-only -q
.venv/bin/alberta-evidence-status
```

The test markers are:

- `unit`: fast isolated behavior and contracts;
- `integration`: component, persistence, process, or CLI boundaries;
- `scientific`: a frozen promoted-evidence protocol;
- `slow`: more than roughly 30 seconds serial and excluded from fast CI; and
- `package`: built-distribution and installed-entry-point checks.

Benchmarks and campaigns run through their scripts or CLIs, never inside ordinary pytest. Keep
tests cheap unless a lane is deliberately registered as scientific.

## Evidence and output safety

Never auto-promote. Passing tests, successful replay, more seeds, or a larger effect does not
change a run's promotion class.

Pinned evidence and sealed campaign artifacts are immutable. Do not edit, overwrite, regenerate
in place, or delete them. In particular, preserve the paths named in `CLAUDE.md`/`AGENTS.md`,
including the five-claim registry artifacts and sealed Forager roots. Active IPMNIST and UPGD
campaign directories are append-only. New work uses a new path and, when its contract requires
it, a new schema version.

Registered source hashes are load-bearing. Before editing evaluation or benchmark code:

```bash
.venv/bin/alberta-evidence-status
```

Source drift making an artifact `invalid` is correct fail-closed behavior. Do not silence it by
changing hashes, broadening a compatibility exception, weakening validation, or editing the
artifact. Explain the invalidation and follow the protocol's re-evaluation rules.

Development, calibration, and consumed evidence seeds never become untouched promotion seeds.
Thresholds are calibrated on development data with margin, then frozen before a claim-bearing
run. A failed frozen gate remains a valid rejection.

## Pull-request checklist

- [ ] The change names a bottleneck, hypothesis, consumer, and success/failure condition.
- [ ] The live control is rerun or the reason it is unnecessary is explicit.
- [ ] Comparators are strong, current, and protocol-matched; mismatches are disclosed.
- [ ] RNG, state ownership, checkpointing, and resource costs are explicit.
- [ ] Tests cover the failure first and include reduction/inert behavior where relevant.
- [ ] Development, release, and scientific evidence classes are not conflated.
- [ ] No pinned artifact was mutated and no consumed seed was relabeled.
- [ ] Negative or bounded outcomes are recorded durably.
- [ ] `README.md`, `docs/status.md`, the research library, and runbooks are updated only where the
      implementation or evidence actually changed.
- [ ] `CLAUDE.md` and `AGENTS.md` remain byte-identical.
- [ ] Targeted tests, fast tests, Ruff, mypy, and evidence status were run or any environment
      blocker is reported.

## Documentation style

Use **ASI** for the current project. Preserve Alberta names for upstream history, Plan-specific
mechanisms, compatibility APIs/CLIs, and historical evidence IDs. Avoid unscoped words such as
“solved,” “SOTA,” “production,” “safe,” or “complete.” A SOTA statement must name its benchmark
and version, protocol, resource envelope, metric, comparison roster, statistical rule, evidence
class, and as-of date.

Overview documents point to mutable result indexes instead of copying rankings that will drift.
If adding a root document, link it from `README.md`. `CLAUDE.md` and `AGENTS.md` are identical by
policy: author the former and copy the exact content to the latter.

## Review standard

Reviewers should ask whether the change makes the next decision more reliable. Correctness and
reproducibility are necessary, but mechanism count is not progress by itself. A clean negative
result that closes an attractive dead end is welcome; record it so the project does not pay for
the same lesson twice.
