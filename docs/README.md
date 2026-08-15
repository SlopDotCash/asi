# Documentation

This directory holds research documentation that previously crowded the repository root.
The package overview and normal development entry points remain in
[`README.md`](../README.md).

## Current references

- [`research/asi-roadmap.md`](research/asi-roadmap.md) — ASI's mission, hillclimb loop,
  application ladder, whole-life scorecard, and current program priorities.
- [`design/asi-reference-agent-protocol.md`](design/asi-reference-agent-protocol.md) — Proposed
  shared reference-life protocol, state ownership, dispatch invariants, adapter boundaries, and
  exact-resume acceptance gate. Its [L0 transaction ledger](../alberta_framework/reference_agent.py)
  and [17 retained tests](../tests/test_reference_agent_protocol.py) complete only the first
  acceptance step; no adapter, whole-life runner/checkpoint, exact-resume result, or
  `reference-dev` agent exists.
- [`status.md`](status.md) — requirement-to-evidence status and completion gates.
- [`evidence/methodology.md`](evidence/methodology.md) — evidence levels, artifact rules,
  validators, and the property map.
- [`evidence/negative-results.md`](evidence/negative-results.md) — concluded negative and
  bounded results. Check this before opening a new experimental lane.
- [`audits/repository-larp-audit.md`](audits/repository-larp-audit.md) — repository-wide audit
  of unsupported surface, evidence state, metadata drift, and the retained credible core.
- [`research/ipmnist-theory.md`](research/ipmnist-theory.md) — mechanistic interpretation of
  the development-only IPMNIST campaign.
- [`research/ipmnist-campaign-index.md`](research/ipmnist-campaign-index.md) — mutable pointer
  to the current IPMNIST summary and the supersession status of append-only records.
- [`design/rtu-taylor-correction.md`](design/rtu-taylor-correction.md) — derivation and limits
  of the optional RTU approximation.

## Runbooks

- [`runbooks/foragax-open-screen.md`](runbooks/foragax-open-screen.md)

Runbooks retain their stated issuance and promotion boundaries. An unissued or
development-only runbook does not authorize a scientific result.

## Archive

- [`archive/forager-comparator-audit.md`](archive/forager-comparator-audit.md)
- [`archive/historical-forager-reconstruction.md`](archive/historical-forager-reconstruction.md)

Archived documents are dated records, not current status pages. Current campaign artifacts
live under `outputs/`; immutable or append-only rules in the agent guide still apply.
