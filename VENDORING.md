# Fork status: ASI and Alberta Framework

This directory began as a vendored copy of the **Alberta Framework** — a JAX
implementation of
[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) (Sutton,
Bowling, Pilarski 2022). It is now **ASI**, an independent continual-learning
research and hillclimbing line, not a lightly-patched vendor drop. The Alberta
Plan remains a major inspiration and historical foundation, but it is not
ASI's binding roadmap. The divergence from the imported snapshot is
substantial and intentional.

- **Fork point:** `lalalune/alberta` @
  `2ac35333efae45cf969ce02ec1f2703476fed6c2`
- **Canonical upstream repository URL:** https://github.com/lalalune/alberta
  (this is the single upstream identity; the `j-klawson/alberta-framework`
  URLs that older `pyproject.toml`/`CITATION.cff` revisions pointed at are
  stale and are no longer referenced)
- **License:** Apache-2.0 (see `LICENSE`)

## Why it lives here

ASI targets an end-to-end continual-learning agent that can scale to real work,
especially robotics. Today, `eliza-robot` uses the inherited Alberta
continual-RL control subset to train across sequences of robot tasks and
provides standard-RL comparison paths including PPO. Its ASIMOV-1 plumbing is
an integration target, not ASI performance or robotics-readiness evidence.
The framework is imported in-process from the robot's Python 3.12 environment,
which is why `requires-python` is `>=3.12` and the NumPy floor is `>=1.26`.

## Naming and compatibility

**ASI** is the current project, repository, and documentation identity. The
following Alberta names remain intentionally stable:

- the `alberta-framework` Python distribution;
- the `alberta_framework` import namespace;
- the `alberta-*` console commands and Alberta-specific Step APIs; and
- historical `alberta.*` artifact schemas, benchmark candidate IDs, paths,
  and immutable records.

These names are compatibility and provenance interfaces. A future namespace
migration would be a separate versioned compatibility project.

## Divergence from the fork point

The fork-point commit is not present in this repository's history (the tree
arrived already diverged), so the divergence is described by capabilities
rather than brittle file and test counts:

- **`alberta_framework/evaluation/`** — fork-local subpackage containing
  strict evidence artifacts and validators, the evidence-registry manifest
  (`evidence_manifest.py` / `alberta-evidence-status`), and the evidence
  CLIs.
- **`alberta_framework/benchmarks/`** — fork-local subpackage containing
  the Forager family (matched-current campaign machinery, RNG parity,
  `official_foragax`/OCI, open screen, historical reconstruction), the
  published-protocol replication lanes (`upgd_ipmnist`,
  `upgd_label_emnist`, `ipmnist_screening`).
- **`alberta_framework/core/`** — additions since the fork point include
  `swift_td`, `stacked_horde`, learned-state and memory components, UPGD,
  option/value-duration support, world models, feature lifecycles,
  and the `PrototypeAgent` composition surface.
- **`alberta_framework/streams/`** — fork-local additions include
  `gauntlet`, `closed_loop`, and `recurring_multiagent`.
- **`tests/`** — tests for upstream-only `benchmarks/`, `examples/`, and
  narrative documents are not carried when their implementation is absent.
- **Top level**: `docs/status.md`, `docs/evidence/methodology.md`,
  `FORAGER_BENCHMARK.md`, the execution runbooks and campaign audits, the
  `outputs/` evidence artifacts, and this file are fork-local.
  `CHANGELOG.md` continues upstream numbering (0.27.0 was cut here), and
  `pyproject.toml` registers the current console scripts.

Because of this, "re-sync from upstream" is no longer a patch-reapplication
exercise; treat any future sync as a merge between diverged development lines.

Not carried from upstream: its repository metadata and non-runtime trees such
as the root-level `benchmarks/` tree, historical `docs/`, `examples/`, and
scripts. This fork has its own `.github/` and `docs/` contents.

## The benchmarks-shim hazard (fixed in 0.27.0)

Upstream kept its benchmark drivers in a repository-root `benchmarks/` tree,
and `alberta_framework/__init__.py` ended with a compatibility shim that
registered that root package under the `alberta_framework.benchmarks` name.
Once this fork added a real `alberta_framework.benchmarks` subpackage, the
shim became a hazard: with any unrelated top-level `benchmarks/` directory
importable (for example an upstream checkout on `sys.path`), the shim could
bind the foreign package into `sys.modules` under the subpackage's name and
shadow the packaged integrations.

As of 0.27.0 the shim is removed. Normal Python submodule resolution loads the
packaged `alberta_framework.benchmarks` tree without importing the benchmark
stack during every base-package import.

## Continual-RL subset used by the robot package

The robot package's direct imports are `alberta_framework.core.actor_critic`,
`core.continual_backprop`, `core.initializers`, `core.normalizers`,
`core.optimizers`, and the top-level re-exports `SARSAAgent`, `SARSAConfig`,
and `ObGDBounding`. The top-level package must remain importable from the
robot environment, but benchmark campaigns and the 12-step and prototype
machinery are not robot dependencies.
