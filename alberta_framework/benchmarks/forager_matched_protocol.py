"""Strict, reward-agnostic protocol for matched-current Forager comparisons.

This module validates scientific intent and execution identity only.  It does
not execute agents, read result artifacts, or authorize evidence promotion.
The schema deliberately lives outside :mod:`forager_matrix` so the canonical
behaviour of matrix schemas 2.2--2.4 remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, cast
from urllib.parse import urlsplit

FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION: Final = "alberta.forager_matched_protocol.v1"
MATCHED_PROTOCOL_SCHEMA_VERSION: Final = FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION
FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_selection_result.v1"
)

_MAX_MANIFEST_BYTES: Final = 2 * 1024 * 1024
_MAX_JSON_NODES: Final = 50_000
_MAX_JSON_NESTING: Final = 64
_MAX_SEEDS: Final = 4_096
_MAX_CANDIDATES: Final = 256
_MAX_TRANSFORMS: Final = 32
_MAX_HYPOTHESES: Final = 64
_MAX_SEED: Final = 2**31 - 1
_MAX_UINT64: Final = 2**64 - 1
_MAX_HORIZON: Final = 2**31 - 1
_MAX_RESOURCE_COUNT: Final = 2**63 - 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_GIT_TREE_SHA1 = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DOTTED_TARGET = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9._+-]*)?")
_CONTAINER_USER = re.compile(r"[1-9][0-9]*:[1-9][0-9]*")

Stage = Literal["open_tuning", "sealed_evaluation"]
Stratum = Literal[
    "alberta_learning",
    "external_learning",
    "privileged_context",
    "historical_orientation",
]
AnalysisRole = Literal["inferential", "descriptive_only"]
SelectionStatistic = Literal["mean", "conservative_ci_endpoint"]
SelectionOutcomeStatus = Literal["pending", "resolved"]
AnalysisMethod = Literal[
    "paired_percentile_bootstrap_lower_bound",
    "paired_sign_flip",
]
TransformValue = str | int | float | bool | None

_EXACT_SHARED_RNG_IMPLEMENTATIONS: Final = frozenset(
    {
        "upstream_ppo",
        "upstream_rtu_ppo",
        "official_ppo",
        "official_rtu_ppo",
    }
)
_KNOWN_PRIVILEGED_IMPLEMENTATIONS: Final = frozenset(
    {
        "upstream_search_oracle",
        "official_search_oracle",
    }
)
_KNOWN_HISTORICAL_IMPLEMENTATIONS: Final = frozenset(
    {
        "historical_dqn",
        "historical_ppo",
        "historical_rtu_ppo",
    }
)
_CAPABILITY_DESCRIPTOR_SCHEMA: Final = "alberta.forager_candidate_capability_descriptor.v1"


class ForagerMatchedProtocolError(ValueError):
    """The matched-current protocol is malformed or internally inconsistent."""


@dataclass(frozen=True)
class MatchedTask:
    """One exact current Forager task identity."""

    task_id: str
    preset: str
    environment_id: str
    foragax_distribution: str
    foragax_version: str
    observation_type: str
    aperture_size: int
    task_identity_sha256: str
    environment_rng_schedule_sha256: str

    def __post_init__(self) -> None:
        aperture_size = _require_int(
            self.aperture_size,
            "task.aperture_size",
            minimum=1,
            maximum=65_535,
        )
        if aperture_size % 2 != 1:
            raise ForagerMatchedProtocolError("task.aperture_size must be odd")
        object.__setattr__(
            self,
            "aperture_size",
            aperture_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "preset": self.preset,
            "environment_id": self.environment_id,
            "foragax_distribution": self.foragax_distribution,
            "foragax_version": self.foragax_version,
            "observation_type": self.observation_type,
            "aperture_size": self.aperture_size,
            "task_identity_sha256": self.task_identity_sha256,
            "environment_rng_schedule_sha256": self.environment_rng_schedule_sha256,
        }


@dataclass(frozen=True)
class AllowedTransform:
    """One typed, reviewable transformation of an upstream configuration."""

    transform_type: str
    target: str
    value_type: Literal["string", "integer", "number", "boolean", "null"]
    value: TransformValue

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_type": self.transform_type,
            "target": self.target,
            "value_type": self.value_type,
            "value": self.value,
        }


@dataclass(frozen=True)
class SourceBinding:
    """Content-addressed source provenance for one candidate.

    ``git_tree`` means the exact candidate is present in ``base_commit`` and
    the repository's native SHA-1 ``tree_git_sha1``; its snapshot descriptor is
    null.  ``reviewed_snapshot`` means ``base_commit`` is ancestry context only:
    the exact dirty/untracked source is instead the archive, inventory, and
    SHA-256 snapshot descriptor named here.  A candidate capability receipt
    must bind this complete descriptor.
    """

    provenance_kind: Literal["git_tree", "reviewed_snapshot"]
    repository: str
    base_commit: str
    tree_git_sha1: str | None
    archive_sha256: str
    inventory_sha256: str
    snapshot_descriptor_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_kind": self.provenance_kind,
            "repository": self.repository,
            "base_commit": self.base_commit,
            "tree_git_sha1": self.tree_git_sha1,
            "archive_sha256": self.archive_sha256,
            "inventory_sha256": self.inventory_sha256,
            "snapshot_descriptor_sha256": self.snapshot_descriptor_sha256,
        }


@dataclass(frozen=True)
class ConfigurationBinding:
    """Original and derived configuration identities."""

    original_path: str
    original_sha256: str
    derived_sha256: str
    allowed_transforms: tuple[AllowedTransform, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_path": self.original_path,
            "original_sha256": self.original_sha256,
            "derived_sha256": self.derived_sha256,
            "allowed_transforms": [item.to_dict() for item in self.allowed_transforms],
        }


@dataclass(frozen=True)
class SeedContract:
    """Exact transport and effective-seed proof."""

    transport: Literal[
        "direct",
        "top_level_seed",
        "nested_experiment_seed_offset",
        "adapter_injected",
    ]
    offset: int
    effective_seed_expression: Literal["active_seed", "active_seed_plus_offset"]
    effective_seed_proof_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "offset": self.offset,
            "effective_seed_expression": self.effective_seed_expression,
            "effective_seed_proof_sha256": self.effective_seed_proof_sha256,
        }


@dataclass(frozen=True)
class ExecutionSemantics:
    """Interaction-count and rollout/update semantics."""

    rollout_steps: int | None
    num_rollouts: int | None
    update_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_steps": self.rollout_steps,
            "num_rollouts": self.num_rollouts,
            "update_semantics": self.update_semantics,
        }


@dataclass(frozen=True)
class ObservationAccess:
    """Observation surface and any privileged fields visible to a candidate."""

    access_mode: Literal[
        "partial_observation",
        "privileged_global_objects",
        "privileged_reward_grid",
        "historical_legacy",
    ]
    observation_type: str
    aperture_size: int
    privileged_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_mode": self.access_mode,
            "observation_type": self.observation_type,
            "aperture_size": self.aperture_size,
            "privileged_fields": list(self.privileged_fields),
        }


@dataclass(frozen=True)
class EnvironmentRNGContract:
    """Environment key schedule used to establish paired blocks."""

    identity: str
    schedule_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.identity, "environment_rng.identity")
        _require_sha256(self.schedule_sha256, "environment_rng.schedule_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "schedule_sha256": self.schedule_sha256}


@dataclass(frozen=True)
class AgentRNGContract:
    """Agent RNG identity and environment-key isolation declaration."""

    identity: str
    environment_key_shared: bool

    def __post_init__(self) -> None:
        _require_identifier(self.identity, "agent_rng.identity")
        object.__setattr__(
            self,
            "environment_key_shared",
            _require_bool(self.environment_key_shared, "agent_rng.environment_key_shared"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "environment_key_shared": self.environment_key_shared,
        }


@dataclass(frozen=True)
class CandidateRuntimeBinding:
    """Candidate-side task/runtime identities and externally verifiable capability receipt.

    The receipt digest is not a self-attested trust boolean.  A verifier outside
    this reward-agnostic parser must resolve ``qualification_trust_anchor_identity``
    and verify that the receipt binds the candidate source, derived configuration,
    entrypoint, observation access, and RNG topology.  The qualified source
    descriptor is recomputed locally from source, configuration, entrypoint,
    observation, RNG, runtime, and role declarations.  It includes the exact
    :class:`SourceBinding` snapshot descriptor and is the subject the external
    receipt must contain.
    """

    image_sha256: str
    runtime_profile_sha256: str
    task_identity_sha256: str
    qualified_capability_descriptor_sha256: str
    capability_qualification_receipt_sha256: str
    qualification_trust_anchor_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_sha256": self.image_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "task_identity_sha256": self.task_identity_sha256,
            "qualified_capability_descriptor_sha256": (self.qualified_capability_descriptor_sha256),
            "capability_qualification_receipt_sha256": (
                self.capability_qualification_receipt_sha256
            ),
            "qualification_trust_anchor_identity": self.qualification_trust_anchor_identity,
        }


@dataclass(frozen=True)
class ResourceAccounting:
    """Static resource accounting; zero is a meaningful declared value."""

    parameter_count: int
    optimizer_update_count: int
    replay_capacity_transitions: int
    recurrent_state_elements: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_count",
            _require_int(
                self.parameter_count,
                "resources.parameter_count",
                minimum=0,
                maximum=_MAX_RESOURCE_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "optimizer_update_count",
            _require_int(
                self.optimizer_update_count,
                "resources.optimizer_update_count",
                minimum=0,
                maximum=_MAX_RESOURCE_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "replay_capacity_transitions",
            _require_int(
                self.replay_capacity_transitions,
                "resources.replay_capacity_transitions",
                minimum=0,
                maximum=_MAX_RESOURCE_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "recurrent_state_elements",
            _require_int(
                self.recurrent_state_elements,
                "resources.recurrent_state_elements",
                minimum=0,
                maximum=_MAX_RESOURCE_COUNT,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_count": self.parameter_count,
            "optimizer_update_count": self.optimizer_update_count,
            "replay_capacity_transitions": self.replay_capacity_transitions,
            "recurrent_state_elements": self.recurrent_state_elements,
        }


@dataclass(frozen=True)
class PairingEligibility:
    """Fail-closed pairing classification."""

    analysis_role: AnalysisRole
    eligible: bool
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        eligible = _require_bool(self.eligible, "pairing.eligible")
        if eligible and (self.analysis_role != "inferential" or self.exclusion_reasons):
            raise ForagerMatchedProtocolError(
                "pairing eligible candidates must be inferential with no exclusion reasons"
            )
        if not eligible and (
            self.analysis_role != "descriptive_only" or not self.exclusion_reasons
        ):
            raise ForagerMatchedProtocolError(
                "pairing ineligible candidates must be descriptive with exclusion reasons"
            )
        object.__setattr__(
            self,
            "eligible",
            eligible,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_role": self.analysis_role,
            "eligible": self.eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


@dataclass(frozen=True)
class MatchedCandidate:
    """One fully provenance-bound candidate."""

    candidate_id: str
    selection_group: str
    stratum: Stratum
    implementation_kind: str
    entrypoint_family: str
    source: SourceBinding
    configuration: ConfigurationBinding
    seed_contract: SeedContract
    execution_semantics: ExecutionSemantics
    observation_access: ObservationAccess
    environment_rng: EnvironmentRNGContract
    agent_rng: AgentRNGContract
    runtime_binding: CandidateRuntimeBinding
    resources: ResourceAccounting
    pairing: PairingEligibility

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "selection_group": self.selection_group,
            "stratum": self.stratum,
            "implementation_kind": self.implementation_kind,
            "entrypoint_family": self.entrypoint_family,
            "source": self.source.to_dict(),
            "configuration": self.configuration.to_dict(),
            "seed_contract": self.seed_contract.to_dict(),
            "execution_semantics": self.execution_semantics.to_dict(),
            "observation_access": self.observation_access.to_dict(),
            "environment_rng": self.environment_rng.to_dict(),
            "agent_rng": self.agent_rng.to_dict(),
            "runtime_binding": self.runtime_binding.to_dict(),
            "resources": self.resources.to_dict(),
            "pairing": self.pairing.to_dict(),
        }


def candidate_capability_descriptor_sha256(candidate: MatchedCandidate) -> str:
    """Return the locally reproducible subject of a candidate qualification receipt."""
    if type(candidate) is not MatchedCandidate:
        raise ForagerMatchedProtocolError("candidate must be a MatchedCandidate")
    binding = candidate.runtime_binding
    payload: dict[str, Any] = {
        "agent_rng": candidate.agent_rng.to_dict(),
        "candidate_id": candidate.candidate_id,
        "configuration": candidate.configuration.to_dict(),
        "entrypoint_family": candidate.entrypoint_family,
        "environment_rng": candidate.environment_rng.to_dict(),
        "execution_semantics": candidate.execution_semantics.to_dict(),
        "implementation_kind": candidate.implementation_kind,
        "observation_access": candidate.observation_access.to_dict(),
        "pairing": candidate.pairing.to_dict(),
        "resources": candidate.resources.to_dict(),
        "runtime_identity": {
            "image_sha256": binding.image_sha256,
            "runtime_profile_sha256": binding.runtime_profile_sha256,
            "task_identity_sha256": binding.task_identity_sha256,
        },
        "schema_version": _CAPABILITY_DESCRIPTOR_SCHEMA,
        "seed_contract": candidate.seed_contract.to_dict(),
        "selection_group": candidate.selection_group,
        "source": candidate.source.to_dict(),
        "stratum": candidate.stratum,
    }
    return hashlib.sha256(_canonical_plain_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class SelectionSlot:
    """One deterministic winner slot, identified only by group and one-based rank."""

    selection_group: str
    rank: int

    def __post_init__(self) -> None:
        _require_identifier(self.selection_group, "selection_slot.selection_group")
        object.__setattr__(
            self,
            "rank",
            _require_int(self.rank, "selection_slot.rank", minimum=1, maximum=_MAX_CANDIDATES),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"selection_group": self.selection_group, "rank": self.rank}


@dataclass(frozen=True)
class SelectionGroup:
    """Ordered tuning population and deterministic advancement count for one family."""

    selection_group: str
    candidate_ids: tuple[str, ...]
    advance_count: int

    def __post_init__(self) -> None:
        advance_count = _require_int(
            self.advance_count,
            "selection_group.advance_count",
            minimum=1,
            maximum=_MAX_CANDIDATES,
        )
        if advance_count > len(self.candidate_ids):
            raise ForagerMatchedProtocolError(
                "selection_group.advance_count may not exceed the group candidate count"
            )
        object.__setattr__(
            self,
            "advance_count",
            advance_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_group": self.selection_group,
            "candidate_ids": list(self.candidate_ids),
            "advance_count": self.advance_count,
        }


@dataclass(frozen=True)
class SelectionPlan:
    """Open-tuning selection contract retained in both stages.

    ``conservative_ci_endpoint`` is the lower endpoint of a two-sided,
    equal-tail percentile interval.  Its selected quantile is therefore
    exactly ``(1 - confidence) / 2``; the schema records all three identities
    explicitly so a consumer cannot reinterpret ``confidence`` as a one-sided
    lower-bound confidence level.
    """

    metric: str
    metric_implementation_sha256: str
    candidate_universe_sha256: str
    direction: Literal["maximize"]
    statistic: SelectionStatistic
    statistic_implementation_sha256: str
    confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_rng_identity: Literal["numpy_generator_pcg64"]
    bootstrap_rng_implementation_sha256: str
    resampling_unit: Literal["candidate_seed_block"]
    quantile_method: Literal["linear"]
    bootstrap_interval: Literal["two_sided_equal_tail"]
    conservative_endpoint: Literal["lower"]
    endpoint_quantile: Literal["(1-confidence)/2"]
    tie_break: Literal["candidate_id_ascending"]
    groups: tuple[SelectionGroup, ...]

    @property
    def slots(self) -> tuple[SelectionSlot, ...]:
        """Return the stage-invariant ordered winner slots."""
        return tuple(
            SelectionSlot(group.selection_group, rank)
            for group in self.groups
            for rank in range(1, group.advance_count + 1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "metric_implementation_sha256": self.metric_implementation_sha256,
            "candidate_universe_sha256": self.candidate_universe_sha256,
            "direction": self.direction,
            "statistic": self.statistic,
            "statistic_implementation_sha256": self.statistic_implementation_sha256,
            "confidence": self.confidence,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_rng_identity": self.bootstrap_rng_identity,
            "bootstrap_rng_implementation_sha256": (self.bootstrap_rng_implementation_sha256),
            "resampling_unit": self.resampling_unit,
            "quantile_method": self.quantile_method,
            "bootstrap_interval": self.bootstrap_interval,
            "conservative_endpoint": self.conservative_endpoint,
            "endpoint_quantile": self.endpoint_quantile,
            "tie_break": self.tie_break,
            "groups": [group.to_dict() for group in self.groups],
        }

    @property
    def plan_sha256(self) -> str:
        """SHA-256 of the complete normalized selection plan."""
        return hashlib.sha256(_canonical_plain_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class EvaluationPanel:
    """Stage-invariant panel expressed through winner slots plus fixed diagnostics."""

    selection_slots: tuple[SelectionSlot, ...]
    fixed_descriptive_candidate_ids: tuple[str, ...]
    alberta_primary_slot: SelectionSlot
    primary_nonprivileged_external_baseline_slot: SelectionSlot
    require_complete_blocks: Literal[True]
    pairing_failure_policy: Literal["fail_closed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_slots": [slot.to_dict() for slot in self.selection_slots],
            "fixed_descriptive_candidate_ids": list(self.fixed_descriptive_candidate_ids),
            "alberta_primary_slot": self.alberta_primary_slot.to_dict(),
            "primary_nonprivileged_external_baseline_slot": (
                self.primary_nonprivileged_external_baseline_slot.to_dict()
            ),
            "require_complete_blocks": self.require_complete_blocks,
            "pairing_failure_policy": self.pairing_failure_policy,
        }


@dataclass(frozen=True)
class MatchedHypothesis:
    """One ordered, necessarily paired directional hypothesis."""

    hypothesis_id: str
    intervention_slot: SelectionSlot
    comparator_slot: SelectionSlot
    estimand: Literal["paired_mean_difference"]
    method: AnalysisMethod
    alternative: Literal["greater"]
    difference_order: Literal["intervention_minus_comparator"]
    paired: Literal[True]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "intervention_slot": self.intervention_slot.to_dict(),
            "comparator_slot": self.comparator_slot.to_dict(),
            "estimand": self.estimand,
            "method": self.method,
            "alternative": self.alternative,
            "difference_order": self.difference_order,
            "paired": self.paired,
        }


@dataclass(frozen=True)
class PrimaryBootstrapAnalysis:
    """Exact one-sided primary method matching the paired statistics contract."""

    method: Literal["paired_percentile_bootstrap_lower_bound"]
    resamples: int
    seed: int
    confidence: float
    primary_margin: float
    rng_algorithm: Literal["PCG64"]
    quantile_method: Literal["linear"]
    implementation_sha256: str
    gate: Literal["lower_bound_strictly_greater_than_margin"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "resamples": self.resamples,
            "seed": self.seed,
            "confidence": self.confidence,
            "primary_margin": self.primary_margin,
            "rng_algorithm": self.rng_algorithm,
            "quantile_method": self.quantile_method,
            "implementation_sha256": self.implementation_sha256,
            "gate": self.gate,
        }


@dataclass(frozen=True)
class SecondarySignFlipAnalysis:
    """Exact one-sided secondary method matching the paired statistics contract."""

    method: Literal["paired_sign_flip"]
    monte_carlo_resamples: int
    seed: int
    exact_max_pairs: Literal[20]
    rng_algorithm: Literal["PCG64"]
    implementation_sha256: str
    alternative: Literal["greater"]
    multiplicity_method: Literal["holm"]
    familywise_alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "monte_carlo_resamples": self.monte_carlo_resamples,
            "seed": self.seed,
            "exact_max_pairs": self.exact_max_pairs,
            "rng_algorithm": self.rng_algorithm,
            "implementation_sha256": self.implementation_sha256,
            "alternative": self.alternative,
            "multiplicity_method": self.multiplicity_method,
            "familywise_alpha": self.familywise_alpha,
        }


@dataclass(frozen=True)
class MatchedAnalysisPlan:
    """Stage-invariant exact analysis template for resolved candidate comparisons."""

    metric: str
    metric_implementation_sha256: str
    metric_direction: Literal["maximize"]
    primary: PrimaryBootstrapAnalysis
    secondary: SecondarySignFlipAnalysis

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "metric_implementation_sha256": self.metric_implementation_sha256,
            "metric_direction": self.metric_direction,
            "primary": self.primary.to_dict(),
            "secondary": self.secondary.to_dict(),
        }


@dataclass(frozen=True)
class ResolvedSelectionSlot:
    """Candidate selected for one stage-invariant winner slot."""

    selection_group: str
    rank: int
    candidate_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.selection_group, "resolved_slot.selection_group")
        object.__setattr__(
            self,
            "rank",
            _require_int(self.rank, "resolved_slot.rank", minimum=1, maximum=_MAX_CANDIDATES),
        )
        _require_identifier(self.candidate_id, "resolved_slot.candidate_id")

    @property
    def slot(self) -> SelectionSlot:
        return SelectionSlot(self.selection_group, self.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_group": self.selection_group,
            "rank": self.rank,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True)
class SelectionOutcome:
    """Pending in open tuning; fully digest-bound and resolved in sealed evaluation."""

    status: SelectionOutcomeStatus
    open_protocol_sha256: str | None
    selection_result_sha256: str | None
    resolved_slots: tuple[ResolvedSelectionSlot, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "open_protocol_sha256": self.open_protocol_sha256,
            "selection_result_sha256": self.selection_result_sha256,
            "resolved_slots": [item.to_dict() for item in self.resolved_slots],
        }


@dataclass(frozen=True)
class RankedSelectionGroup:
    """Complete deterministic candidate ranking and opaque external evidence receipt."""

    selection_group: str
    ranked_candidate_ids: tuple[str, ...]
    ranking_evidence_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.selection_group, "ranked_selection_group.selection_group")
        if type(self.ranked_candidate_ids) is not tuple or not all(
            type(cid) is str and cid for cid in self.ranked_candidate_ids
        ):
            raise ForagerMatchedProtocolError(
                "ranked_candidate_ids must be a tuple of candidate IDs"
            )
        _require_sha256(
            self.ranking_evidence_sha256, "ranked_selection_group.ranking_evidence_sha256"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_group": self.selection_group,
            "ranked_candidate_ids": list(self.ranked_candidate_ids),
            "ranking_evidence_sha256": self.ranking_evidence_sha256,
        }


@dataclass(frozen=True)
class ForagerMatchedSelectionResult:
    """Canonical, reward-opaque selection result bound to one open protocol."""

    schema_version: Literal["alberta.forager_matched_selection_result.v1"]
    open_protocol_sha256: str
    selection_plan_sha256: str
    tuning_seeds: tuple[int, ...]
    ranked_groups: tuple[RankedSelectionGroup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "open_protocol_sha256": self.open_protocol_sha256,
            "selection_plan_sha256": self.selection_plan_sha256,
            "tuning_seeds": list(self.tuning_seeds),
            "ranked_groups": [group.to_dict() for group in self.ranked_groups],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_selection_result_bytes(self)

    @property
    def selection_result_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True)
class ResolvedHypothesis:
    """Candidate-level resolution of one slot-based hypothesis."""

    hypothesis_id: str
    intervention_candidate_id: str
    comparator_candidate_id: str
    method: AnalysisMethod
    alternative: Literal["greater"]
    difference_order: Literal["intervention_minus_comparator"]

    def comparison_spec_payload(self) -> dict[str, str]:
        """Return the exact explicit-ID payload consumed by statistics ComparisonSpec."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "intervention_id": self.intervention_candidate_id,
            "comparator_id": self.comparator_candidate_id,
            "alternative": self.alternative,
            "difference_order": self.difference_order,
        }


@dataclass(frozen=True)
class SealedProtocolValidation:
    """Immutable, reward-free proof that a sealed transition resolves exactly."""

    open_protocol_sha256: str
    selection_result_sha256: str
    resolved_slots: tuple[ResolvedSelectionSlot, ...]
    evaluation_candidate_ids: tuple[str, ...]
    primary_intervention_candidate_id: str
    primary_comparator_candidate_id: str
    resolved_hypotheses: tuple[ResolvedHypothesis, ...]


@dataclass(frozen=True)
class MultiplicityPolicy:
    """Ordered Holm family for the secondary hypotheses only."""

    method: Literal["holm"]
    alpha: float
    hypothesis_ids: tuple[str, ...]
    primary_excluded: Literal[True]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "hypothesis_ids": list(self.hypothesis_ids),
            "primary_excluded": self.primary_excluded,
        }


@dataclass(frozen=True)
class DescriptiveContext:
    """Exact ordered membership of a non-inferential context stratum."""

    candidate_ids: tuple[str, ...]
    analysis_role: Literal["descriptive_only"]
    selection_eligible: Literal[False]
    pairing_eligible: Literal[False]

    def __post_init__(self) -> None:
        if type(self.candidate_ids) is not tuple or not all(
            type(cid) is str and cid for cid in self.candidate_ids
        ):
            raise ForagerMatchedProtocolError("candidate_ids must be a tuple of candidate IDs")
        if self.analysis_role != "descriptive_only":
            raise ForagerMatchedProtocolError("analysis_role must be 'descriptive_only'")
        if self.selection_eligible is not False:
            raise ForagerMatchedProtocolError("selection_eligible must be False")
        if self.pairing_eligible is not False:
            raise ForagerMatchedProtocolError("pairing_eligible must be False")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "analysis_role": self.analysis_role,
            "selection_eligible": self.selection_eligible,
            "pairing_eligible": self.pairing_eligible,
        }


@dataclass(frozen=True)
class CPUSandboxContract:
    """Minimum immutable CPU OCI sandbox declaration."""

    network: Literal["none"]
    root_filesystem: Literal["read_only"]
    capabilities: Literal["all_dropped"]
    no_new_privileges: Literal[True]
    container_user: str
    host_devices: tuple[str, ...]
    writable_tmpfs_only: Literal[True]

    def to_dict(self) -> dict[str, Any]:
        return {
            "network": self.network,
            "root_filesystem": self.root_filesystem,
            "capabilities": self.capabilities,
            "no_new_privileges": self.no_new_privileges,
            "container_user": self.container_user,
            "host_devices": [],
            "writable_tmpfs_only": self.writable_tmpfs_only,
        }


@dataclass(frozen=True)
class MatchedRuntime:
    """One externally qualified immutable CPU executor shared by current candidates.

    The parser validates the receipt and trust-anchor identities as mandatory
    content-addressed references.  Authenticating that external evidence remains
    the responsibility of the sealed-run verifier identified by the trust anchor.
    """

    executor_kind: Literal["oci"]
    image_sha256: str
    runtime_profile_sha256: str
    executor_qualification_receipt_sha256: str
    qualification_trust_anchor_identity: str
    source_mount_mode: Literal["read_only_content_addressed_mount"]
    default_prng: Literal["threefry2x32"]
    threefry_partitionable: Literal[True]
    platform: Literal["cpu"]
    sandbox: CPUSandboxContract

    def to_dict(self) -> dict[str, Any]:
        return {
            "executor_kind": self.executor_kind,
            "image_sha256": self.image_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "executor_qualification_receipt_sha256": (self.executor_qualification_receipt_sha256),
            "qualification_trust_anchor_identity": self.qualification_trust_anchor_identity,
            "source_mount_mode": self.source_mount_mode,
            "default_prng": self.default_prng,
            "threefry_partitionable": self.threefry_partitionable,
            "platform": self.platform,
            "sandbox": self.sandbox.to_dict(),
        }


@dataclass(frozen=True)
class ForagerMatchedProtocol:
    """Validated and normalized matched-current comparison protocol."""

    schema_version: Literal["alberta.forager_matched_protocol.v1"]
    stage: Stage
    task: MatchedTask
    horizon: int
    tuning_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    active_seeds: tuple[int, ...]
    candidates: tuple[MatchedCandidate, ...]
    selection_plan: SelectionPlan
    selection_outcome: SelectionOutcome
    analysis_plan: MatchedAnalysisPlan
    evaluation_panel: EvaluationPanel
    primary_hypothesis: MatchedHypothesis
    secondary_hypotheses: tuple[MatchedHypothesis, ...]
    multiplicity_policy: MultiplicityPolicy
    privileged_context: DescriptiveContext
    historical_orientation: DescriptiveContext
    runtime: MatchedRuntime
    candidate_index: Mapping[str, MatchedCandidate] = field(
        compare=False,
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete canonical JSON value."""
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "task": self.task.to_dict(),
            "horizon": self.horizon,
            "tuning_seeds": list(self.tuning_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "active_seeds": list(self.active_seeds),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection_plan": self.selection_plan.to_dict(),
            "selection_outcome": self.selection_outcome.to_dict(),
            "analysis_plan": self.analysis_plan.to_dict(),
            "evaluation_panel": self.evaluation_panel.to_dict(),
            "primary_hypothesis": self.primary_hypothesis.to_dict(),
            "secondary_hypotheses": [
                hypothesis.to_dict() for hypothesis in self.secondary_hypotheses
            ],
            "multiplicity_policy": self.multiplicity_policy.to_dict(),
            "privileged_context": self.privileged_context.to_dict(),
            "historical_orientation": self.historical_orientation.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Canonical JSON bytes with no timestamps or host-derived data."""
        return canonical_json_bytes(self)

    @property
    def protocol_sha256(self) -> str:
        """SHA-256 of :attr:`canonical_bytes`."""
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _duplicate_free_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedProtocolError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ForagerMatchedProtocolError(f"non-finite JSON number {token!r} is forbidden")


def _parse_json_float(token: str) -> float:
    try:
        value = float(token)
    except (OverflowError, ValueError) as exc:
        raise ForagerMatchedProtocolError(f"invalid JSON number {token!r}") from exc
    if not math.isfinite(value):
        raise ForagerMatchedProtocolError(f"non-finite JSON number {token!r} is forbidden")
    return value


def _validate_json_complexity(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedProtocolError("protocol exceeds the JSON node limit")
        if depth > _MAX_JSON_NESTING:
            raise ForagerMatchedProtocolError("protocol exceeds the JSON nesting limit")
        if type(item) is dict:
            for key in item:
                if type(key) is not str:
                    raise ForagerMatchedProtocolError("JSON object keys must be strings")
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ForagerMatchedProtocolError(
                        "JSON object keys must contain valid Unicode"
                    ) from exc
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedProtocolError(
                    "JSON strings must contain valid Unicode"
                ) from exc
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ForagerMatchedProtocolError("non-finite JSON numbers are forbidden")
        elif item is not None and not isinstance(item, (bool, int)):
            raise ForagerMatchedProtocolError(
                f"protocol contains a non-JSON value of type {type(item).__name__}"
            )


def decode_strict_json(data: bytes | str) -> Any:
    """Decode duplicate-free finite UTF-8 JSON with bounded complexity."""
    data_type = type(data)
    if data_type is not bytes and data_type is not str:
        raise ForagerMatchedProtocolError("protocol must be exact bytes or string JSON")
    try:
        if type(data) is bytes:
            if len(data) > _MAX_MANIFEST_BYTES:
                raise ForagerMatchedProtocolError("protocol exceeds the file-size limit")
            text = data.decode("utf-8")
        else:
            raw_text = cast(str, data)
            if len(raw_text) > _MAX_MANIFEST_BYTES:
                raise ForagerMatchedProtocolError("protocol exceeds the file-size limit")
            if len(raw_text.encode("utf-8")) > _MAX_MANIFEST_BYTES:
                raise ForagerMatchedProtocolError("protocol exceeds the file-size limit")
            text = raw_text
        decoded = json.loads(
            text,
            object_pairs_hook=_duplicate_free_object,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_json_float,
        )
    except ForagerMatchedProtocolError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise ForagerMatchedProtocolError(f"protocol is not strict UTF-8 JSON: {exc}") from exc
    _validate_json_complexity(decoded)
    return decoded


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedProtocolError(f"{path} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ForagerMatchedProtocolError(f"{path} keys must be strings")
    return cast(Mapping[str, Any], value)


def _require_array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        raise ForagerMatchedProtocolError(f"{path} must be a JSON array")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    path: str,
    required: set[str],
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise ForagerMatchedProtocolError(f"{path} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ForagerMatchedProtocolError(f"{path} contains unknown keys: {', '.join(unknown)}")


def _require_string(value: Any, path: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ForagerMatchedProtocolError(
            f"{path} must be a non-empty string of at most {maximum} characters"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedProtocolError(f"{path} must contain valid Unicode") from exc
    return value


def _require_identifier(value: Any, path: str) -> str:
    text = _require_string(value, path, maximum=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ForagerMatchedProtocolError(f"{path} must be a safe identifier")
    return text


def _require_literal(value: Any, path: str, choices: tuple[str, ...]) -> str:
    text = _require_string(value, path, maximum=128)
    if text not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise ForagerMatchedProtocolError(f"{path} must be one of {allowed}")
    return text


def _require_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ForagerMatchedProtocolError(f"{path} must be a boolean")
    return value


def _require_int(
    value: Any,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise ForagerMatchedProtocolError(f"{path} must be an integer")
    if value < minimum or value > maximum:
        raise ForagerMatchedProtocolError(f"{path} must lie in [{minimum}, {maximum}]")
    return value


def _require_optional_positive_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path, minimum=1, maximum=_MAX_HORIZON)


def _require_probability(value: Any, path: str) -> float:
    if type(value) not in (int, float):
        raise ForagerMatchedProtocolError(f"{path} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ForagerMatchedProtocolError(f"{path} must be a finite number") from exc
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ForagerMatchedProtocolError(f"{path} must lie strictly between zero and one")
    return number


def _require_nonnegative_float(value: Any, path: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ForagerMatchedProtocolError(f"{path} must be a finite nonnegative JSON float")
    return value


def _require_probability_float(value: Any, path: str) -> float:
    if type(value) is not float:
        raise ForagerMatchedProtocolError(f"{path} must be a JSON float")
    return _require_probability(value, path)


def _require_sha256(value: Any, path: str) -> str:
    text = _require_string(value, path, maximum=64)
    if _SHA256.fullmatch(text) is None:
        raise ForagerMatchedProtocolError(f"{path} must be a lowercase 64-character SHA-256 digest")
    return text


def _require_safe_relative_path(value: Any, path: str) -> str:
    text = _require_string(value, path, maximum=512)
    candidate = PurePosixPath(text)
    if (
        "\\" in text
        or "\x00" in text
        or text.startswith("~")
        or candidate.is_absolute()
        or candidate.as_posix() != text
        or any(
            part in ("", ".", "..") or _PATH_COMPONENT.fullmatch(part) is None
            for part in candidate.parts
        )
    ):
        raise ForagerMatchedProtocolError(f"{path} must be a safe relative POSIX path")
    return candidate.as_posix()


def _require_repository(value: Any, path: str) -> str:
    text = _require_string(value, path, maximum=512)
    try:
        parsed = urlsplit(text)
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
        repository_path = parsed.path.rstrip("/")
    except ValueError as exc:
        raise ForagerMatchedProtocolError(
            f"{path} must be a credential-free HTTPS repository URL"
        ) from exc
    if (
        parsed.scheme != "https"
        or not hostname
        or username is not None
        or password is not None
        or port is not None
        or not repository_path
        or parsed.query
        or parsed.fragment
        or "\\" in text
        or any(character.isspace() or ord(character) < 0x20 for character in text)
    ):
        raise ForagerMatchedProtocolError(f"{path} must be a credential-free HTTPS repository URL")
    return text.rstrip("/")


def _require_unique_identifiers(value: Any, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    items = _require_array(value, path)
    if not allow_empty and not items:
        raise ForagerMatchedProtocolError(f"{path} must not be empty")
    normalized = tuple(
        _require_identifier(item, f"{path}[{index}]") for index, item in enumerate(items)
    )
    if len(set(normalized)) != len(normalized):
        raise ForagerMatchedProtocolError(f"{path} contains duplicate identifiers")
    return normalized


def _require_seeds(value: Any, path: str) -> tuple[int, ...]:
    items = _require_array(value, path)
    if not items:
        raise ForagerMatchedProtocolError(f"{path} must not be empty")
    if len(items) > _MAX_SEEDS:
        raise ForagerMatchedProtocolError(f"{path} contains too many seeds")
    seeds = tuple(
        _require_int(item, f"{path}[{index}]", minimum=0, maximum=_MAX_SEED)
        for index, item in enumerate(items)
    )
    if len(set(seeds)) != len(seeds):
        raise ForagerMatchedProtocolError(f"{path} contains duplicate seeds")
    return seeds


def _parse_task(value: Any) -> MatchedTask:
    path = "protocol.task"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "task_id",
            "preset",
            "environment_id",
            "foragax_distribution",
            "foragax_version",
            "observation_type",
            "aperture_size",
            "task_identity_sha256",
            "environment_rng_schedule_sha256",
        },
    )
    preset = _require_literal(payload["preset"], f"{path}.preset", ("field_of_view",))
    version = _require_string(payload["foragax_version"], f"{path}.foragax_version", maximum=64)
    if _VERSION.fullmatch(version) is None:
        raise ForagerMatchedProtocolError(f"{path}.foragax_version is invalid")
    aperture = _require_int(
        payload["aperture_size"],
        f"{path}.aperture_size",
        minimum=1,
        maximum=65_535,
    )
    return MatchedTask(
        task_id=_require_identifier(payload["task_id"], f"{path}.task_id"),
        preset=preset,
        environment_id=_require_identifier(payload["environment_id"], f"{path}.environment_id"),
        foragax_distribution=_require_identifier(
            payload["foragax_distribution"], f"{path}.foragax_distribution"
        ),
        foragax_version=version,
        observation_type=_require_identifier(
            payload["observation_type"], f"{path}.observation_type"
        ),
        aperture_size=aperture,
        task_identity_sha256=_require_sha256(
            payload["task_identity_sha256"], f"{path}.task_identity_sha256"
        ),
        environment_rng_schedule_sha256=_require_sha256(
            payload["environment_rng_schedule_sha256"],
            f"{path}.environment_rng_schedule_sha256",
        ),
    )


def _parse_transform(value: Any, path: str) -> AllowedTransform:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"transform_type", "target", "value_type", "value"},
    )
    transform_type = _require_identifier(payload["transform_type"], f"{path}.transform_type")
    target = _require_string(payload["target"], f"{path}.target", maximum=256)
    if _DOTTED_TARGET.fullmatch(target) is None:
        raise ForagerMatchedProtocolError(f"{path}.target must be a dotted configuration key")
    value_type = _require_literal(
        payload["value_type"],
        f"{path}.value_type",
        ("string", "integer", "number", "boolean", "null"),
    )
    raw_value = payload["value"]
    normalized: TransformValue
    if value_type == "string":
        normalized = _require_string(raw_value, f"{path}.value", maximum=256)
    elif value_type == "integer":
        normalized = _require_int(
            raw_value,
            f"{path}.value",
            minimum=-(2**63),
            maximum=2**63 - 1,
        )
    elif value_type == "number":
        if type(raw_value) not in (int, float):
            raise ForagerMatchedProtocolError(f"{path}.value must be a finite number")
        try:
            normalized = float(raw_value)
        except (OverflowError, ValueError) as exc:
            raise ForagerMatchedProtocolError(f"{path}.value must be a finite number") from exc
        if not math.isfinite(normalized):
            raise ForagerMatchedProtocolError(f"{path}.value must be a finite number")
    elif value_type == "boolean":
        normalized = _require_bool(raw_value, f"{path}.value")
    else:
        if raw_value is not None:
            raise ForagerMatchedProtocolError(f"{path}.value must be null")
        normalized = None
    return AllowedTransform(
        transform_type=transform_type,
        target=target,
        value_type=cast(Literal["string", "integer", "number", "boolean", "null"], value_type),
        value=normalized,
    )


def _parse_source(value: Any, path: str) -> SourceBinding:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "provenance_kind",
            "repository",
            "base_commit",
            "tree_git_sha1",
            "archive_sha256",
            "inventory_sha256",
            "snapshot_descriptor_sha256",
        },
    )
    provenance_kind = _require_literal(
        payload["provenance_kind"],
        f"{path}.provenance_kind",
        ("git_tree", "reviewed_snapshot"),
    )
    commit = _require_string(payload["base_commit"], f"{path}.base_commit", maximum=64)
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ForagerMatchedProtocolError(f"{path}.base_commit must be a full lowercase commit ID")
    raw_tree = payload["tree_git_sha1"]
    tree: str | None
    raw_snapshot = payload["snapshot_descriptor_sha256"]
    snapshot: str | None
    if provenance_kind == "git_tree":
        tree = _require_string(raw_tree, f"{path}.tree_git_sha1", maximum=40)
        if _GIT_TREE_SHA1.fullmatch(tree) is None:
            raise ForagerMatchedProtocolError(
                f"{path}.tree_git_sha1 must be a full lowercase Git SHA-1"
            )
        if raw_snapshot is not None:
            raise ForagerMatchedProtocolError(
                f"{path}.snapshot_descriptor_sha256 must be null for git_tree provenance"
            )
        snapshot = None
    else:
        if raw_tree is not None:
            raise ForagerMatchedProtocolError(
                f"{path}.tree_git_sha1 must be null for reviewed_snapshot provenance"
            )
        tree = None
        snapshot = _require_sha256(raw_snapshot, f"{path}.snapshot_descriptor_sha256")
    return SourceBinding(
        provenance_kind=cast(Literal["git_tree", "reviewed_snapshot"], provenance_kind),
        repository=_require_repository(payload["repository"], f"{path}.repository"),
        base_commit=commit,
        tree_git_sha1=tree,
        archive_sha256=_require_sha256(payload["archive_sha256"], f"{path}.archive_sha256"),
        inventory_sha256=_require_sha256(payload["inventory_sha256"], f"{path}.inventory_sha256"),
        snapshot_descriptor_sha256=snapshot,
    )


def _parse_configuration(value: Any, path: str) -> ConfigurationBinding:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"original_path", "original_sha256", "derived_sha256", "allowed_transforms"},
    )
    transform_payload = _require_array(payload["allowed_transforms"], f"{path}.allowed_transforms")
    if len(transform_payload) > _MAX_TRANSFORMS:
        raise ForagerMatchedProtocolError(f"{path}.allowed_transforms contains too many items")
    transforms = tuple(
        _parse_transform(item, f"{path}.allowed_transforms[{index}]")
        for index, item in enumerate(transform_payload)
    )
    targets = [item.target for item in transforms]
    if len(set(targets)) != len(targets):
        raise ForagerMatchedProtocolError(f"{path}.allowed_transforms repeats a target")
    original = _require_sha256(payload["original_sha256"], f"{path}.original_sha256")
    derived = _require_sha256(payload["derived_sha256"], f"{path}.derived_sha256")
    if original == derived and transforms:
        raise ForagerMatchedProtocolError(
            f"{path}.allowed_transforms must be empty when configuration hashes match"
        )
    if original != derived and not transforms:
        raise ForagerMatchedProtocolError(
            f"{path}.allowed_transforms must explain a changed derived configuration"
        )
    return ConfigurationBinding(
        original_path=_require_safe_relative_path(
            payload["original_path"], f"{path}.original_path"
        ),
        original_sha256=original,
        derived_sha256=derived,
        allowed_transforms=transforms,
    )


def _parse_seed_contract(value: Any, path: str) -> SeedContract:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"transport", "offset", "effective_seed_expression", "effective_seed_proof_sha256"},
    )
    transport = _require_literal(
        payload["transport"],
        f"{path}.transport",
        ("direct", "top_level_seed", "nested_experiment_seed_offset", "adapter_injected"),
    )
    expression = _require_literal(
        payload["effective_seed_expression"],
        f"{path}.effective_seed_expression",
        ("active_seed", "active_seed_plus_offset"),
    )
    offset = _require_int(
        payload["offset"], f"{path}.offset", minimum=-_MAX_SEED, maximum=_MAX_SEED
    )
    if expression == "active_seed" and offset != 0:
        raise ForagerMatchedProtocolError(
            f"{path}.offset must be zero for the active_seed expression"
        )
    return SeedContract(
        transport=cast(
            Literal[
                "direct",
                "top_level_seed",
                "nested_experiment_seed_offset",
                "adapter_injected",
            ],
            transport,
        ),
        offset=offset,
        effective_seed_expression=cast(
            Literal["active_seed", "active_seed_plus_offset"], expression
        ),
        effective_seed_proof_sha256=_require_sha256(
            payload["effective_seed_proof_sha256"],
            f"{path}.effective_seed_proof_sha256",
        ),
    )


def _parse_execution_semantics(value: Any, path: str, *, horizon: int) -> ExecutionSemantics:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"rollout_steps", "num_rollouts", "update_semantics"},
    )
    rollout_steps = _require_optional_positive_int(
        payload["rollout_steps"], f"{path}.rollout_steps"
    )
    num_rollouts = _require_optional_positive_int(payload["num_rollouts"], f"{path}.num_rollouts")
    if (rollout_steps is None) != (num_rollouts is None):
        raise ForagerMatchedProtocolError(
            f"{path}.rollout_steps and num_rollouts must either both be null or both be integers"
        )
    if rollout_steps is not None:
        if horizon % rollout_steps != 0:
            raise ForagerMatchedProtocolError(
                f"{path}.rollout_steps must divide protocol.horizon exactly"
            )
        if num_rollouts != horizon // rollout_steps:
            raise ForagerMatchedProtocolError(
                f"{path}.num_rollouts must equal horizon divided by rollout_steps"
            )
    return ExecutionSemantics(
        rollout_steps=rollout_steps,
        num_rollouts=num_rollouts,
        update_semantics=_require_identifier(
            payload["update_semantics"], f"{path}.update_semantics"
        ),
    )


def _parse_observation_access(value: Any, path: str) -> ObservationAccess:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"access_mode", "observation_type", "aperture_size", "privileged_fields"},
    )
    mode = _require_literal(
        payload["access_mode"],
        f"{path}.access_mode",
        (
            "partial_observation",
            "privileged_global_objects",
            "privileged_reward_grid",
            "historical_legacy",
        ),
    )
    aperture = _require_int(
        payload["aperture_size"],
        f"{path}.aperture_size",
        minimum=-1,
        maximum=65_535,
    )
    if aperture == 0:
        raise ForagerMatchedProtocolError(f"{path}.aperture_size may not be zero")
    fields = _require_unique_identifiers(
        payload["privileged_fields"], f"{path}.privileged_fields", allow_empty=True
    )
    if mode == "partial_observation" and fields:
        raise ForagerMatchedProtocolError(
            f"{path}.privileged_fields must be empty for partial observations"
        )
    if mode in ("privileged_global_objects", "privileged_reward_grid") and not fields:
        raise ForagerMatchedProtocolError(
            f"{path}.privileged_fields must describe privileged access"
        )
    return ObservationAccess(
        access_mode=cast(
            Literal[
                "partial_observation",
                "privileged_global_objects",
                "privileged_reward_grid",
                "historical_legacy",
            ],
            mode,
        ),
        observation_type=_require_identifier(
            payload["observation_type"], f"{path}.observation_type"
        ),
        aperture_size=aperture,
        privileged_fields=fields,
    )


def _parse_environment_rng(value: Any, path: str) -> EnvironmentRNGContract:
    payload = _require_object(value, path)
    _require_exact_keys(payload, path, {"identity", "schedule_sha256"})
    return EnvironmentRNGContract(
        identity=_require_identifier(payload["identity"], f"{path}.identity"),
        schedule_sha256=_require_sha256(payload["schedule_sha256"], f"{path}.schedule_sha256"),
    )


def _parse_agent_rng(value: Any, path: str) -> AgentRNGContract:
    payload = _require_object(value, path)
    _require_exact_keys(payload, path, {"identity", "environment_key_shared"})
    return AgentRNGContract(
        identity=_require_identifier(payload["identity"], f"{path}.identity"),
        environment_key_shared=_require_bool(
            payload["environment_key_shared"], f"{path}.environment_key_shared"
        ),
    )


def _parse_runtime_binding(value: Any, path: str) -> CandidateRuntimeBinding:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "image_sha256",
            "runtime_profile_sha256",
            "task_identity_sha256",
            "qualified_capability_descriptor_sha256",
            "capability_qualification_receipt_sha256",
            "qualification_trust_anchor_identity",
        },
    )
    return CandidateRuntimeBinding(
        image_sha256=_require_sha256(payload["image_sha256"], f"{path}.image_sha256"),
        runtime_profile_sha256=_require_sha256(
            payload["runtime_profile_sha256"], f"{path}.runtime_profile_sha256"
        ),
        task_identity_sha256=_require_sha256(
            payload["task_identity_sha256"], f"{path}.task_identity_sha256"
        ),
        qualified_capability_descriptor_sha256=_require_sha256(
            payload["qualified_capability_descriptor_sha256"],
            f"{path}.qualified_capability_descriptor_sha256",
        ),
        capability_qualification_receipt_sha256=_require_sha256(
            payload["capability_qualification_receipt_sha256"],
            f"{path}.capability_qualification_receipt_sha256",
        ),
        qualification_trust_anchor_identity=_require_identifier(
            payload["qualification_trust_anchor_identity"],
            f"{path}.qualification_trust_anchor_identity",
        ),
    )


def _parse_resources(value: Any, path: str) -> ResourceAccounting:
    payload = _require_object(value, path)
    keys = {
        "parameter_count",
        "optimizer_update_count",
        "replay_capacity_transitions",
        "recurrent_state_elements",
    }
    _require_exact_keys(payload, path, keys)
    values = {
        key: _require_int(payload[key], f"{path}.{key}", minimum=0, maximum=_MAX_RESOURCE_COUNT)
        for key in keys
    }
    return ResourceAccounting(
        parameter_count=values["parameter_count"],
        optimizer_update_count=values["optimizer_update_count"],
        replay_capacity_transitions=values["replay_capacity_transitions"],
        recurrent_state_elements=values["recurrent_state_elements"],
    )


def _parse_pairing(value: Any, path: str) -> PairingEligibility:
    payload = _require_object(value, path)
    _require_exact_keys(payload, path, {"analysis_role", "eligible", "exclusion_reasons"})
    role = _require_literal(
        payload["analysis_role"], f"{path}.analysis_role", ("inferential", "descriptive_only")
    )
    eligible = _require_bool(payload["eligible"], f"{path}.eligible")
    reasons = _require_unique_identifiers(
        payload["exclusion_reasons"], f"{path}.exclusion_reasons", allow_empty=True
    )
    return PairingEligibility(
        analysis_role=cast(AnalysisRole, role),
        eligible=eligible,
        exclusion_reasons=reasons,
    )


def _parse_candidate(value: Any, path: str, *, horizon: int) -> MatchedCandidate:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "candidate_id",
            "selection_group",
            "stratum",
            "implementation_kind",
            "entrypoint_family",
            "source",
            "configuration",
            "seed_contract",
            "execution_semantics",
            "observation_access",
            "environment_rng",
            "agent_rng",
            "runtime_binding",
            "resources",
            "pairing",
        },
    )
    stratum = _require_literal(
        payload["stratum"],
        f"{path}.stratum",
        (
            "alberta_learning",
            "external_learning",
            "privileged_context",
            "historical_orientation",
        ),
    )
    return MatchedCandidate(
        candidate_id=_require_identifier(payload["candidate_id"], f"{path}.candidate_id"),
        selection_group=_require_identifier(payload["selection_group"], f"{path}.selection_group"),
        stratum=cast(Stratum, stratum),
        implementation_kind=_require_identifier(
            payload["implementation_kind"], f"{path}.implementation_kind"
        ),
        entrypoint_family=_require_identifier(
            payload["entrypoint_family"], f"{path}.entrypoint_family"
        ),
        source=_parse_source(payload["source"], f"{path}.source"),
        configuration=_parse_configuration(payload["configuration"], f"{path}.configuration"),
        seed_contract=_parse_seed_contract(payload["seed_contract"], f"{path}.seed_contract"),
        execution_semantics=_parse_execution_semantics(
            payload["execution_semantics"], f"{path}.execution_semantics", horizon=horizon
        ),
        observation_access=_parse_observation_access(
            payload["observation_access"], f"{path}.observation_access"
        ),
        environment_rng=_parse_environment_rng(
            payload["environment_rng"], f"{path}.environment_rng"
        ),
        agent_rng=_parse_agent_rng(payload["agent_rng"], f"{path}.agent_rng"),
        runtime_binding=_parse_runtime_binding(
            payload["runtime_binding"], f"{path}.runtime_binding"
        ),
        resources=_parse_resources(payload["resources"], f"{path}.resources"),
        pairing=_parse_pairing(payload["pairing"], f"{path}.pairing"),
    )


def _parse_selection_plan(value: Any) -> SelectionPlan:
    path = "protocol.selection_plan"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "metric",
            "metric_implementation_sha256",
            "candidate_universe_sha256",
            "direction",
            "statistic",
            "statistic_implementation_sha256",
            "confidence",
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_rng_identity",
            "bootstrap_rng_implementation_sha256",
            "resampling_unit",
            "quantile_method",
            "bootstrap_interval",
            "conservative_endpoint",
            "endpoint_quantile",
            "tie_break",
            "groups",
        },
    )
    groups_payload = _require_array(payload["groups"], f"{path}.groups")
    if not groups_payload:
        raise ForagerMatchedProtocolError(f"{path}.groups must not be empty")
    if len(groups_payload) > _MAX_CANDIDATES:
        raise ForagerMatchedProtocolError(f"{path}.groups contains too many items")
    groups: list[SelectionGroup] = []
    for index, item in enumerate(groups_payload):
        item_path = f"{path}.groups[{index}]"
        group_payload = _require_object(item, item_path)
        _require_exact_keys(
            group_payload,
            item_path,
            {"selection_group", "candidate_ids", "advance_count"},
        )
        candidate_ids = _require_unique_identifiers(
            group_payload["candidate_ids"], f"{item_path}.candidate_ids", allow_empty=False
        )
        advance_count = _require_int(
            group_payload["advance_count"],
            f"{item_path}.advance_count",
            minimum=1,
            maximum=_MAX_CANDIDATES,
        )
        groups.append(
            SelectionGroup(
                selection_group=_require_identifier(
                    group_payload["selection_group"], f"{item_path}.selection_group"
                ),
                candidate_ids=candidate_ids,
                advance_count=advance_count,
            )
        )
    if len({group.selection_group for group in groups}) != len(groups):
        raise ForagerMatchedProtocolError(f"{path}.groups repeats a selection group")
    direction = _require_literal(payload["direction"], f"{path}.direction", ("maximize",))
    statistic_value = _require_literal(
        payload["statistic"],
        f"{path}.statistic",
        ("mean", "conservative_ci_endpoint"),
    )
    tie_break = _require_literal(
        payload["tie_break"], f"{path}.tie_break", ("candidate_id_ascending",)
    )
    rng_identity = _require_literal(
        payload["bootstrap_rng_identity"],
        f"{path}.bootstrap_rng_identity",
        ("numpy_generator_pcg64",),
    )
    resampling_unit = _require_literal(
        payload["resampling_unit"],
        f"{path}.resampling_unit",
        ("candidate_seed_block",),
    )
    quantile_method = _require_literal(
        payload["quantile_method"], f"{path}.quantile_method", ("linear",)
    )
    bootstrap_interval = _require_literal(
        payload["bootstrap_interval"],
        f"{path}.bootstrap_interval",
        ("two_sided_equal_tail",),
    )
    conservative_endpoint = _require_literal(
        payload["conservative_endpoint"],
        f"{path}.conservative_endpoint",
        ("lower",),
    )
    endpoint_quantile = _require_literal(
        payload["endpoint_quantile"],
        f"{path}.endpoint_quantile",
        ("(1-confidence)/2",),
    )
    return SelectionPlan(
        metric=_require_identifier(payload["metric"], f"{path}.metric"),
        metric_implementation_sha256=_require_sha256(
            payload["metric_implementation_sha256"], f"{path}.metric_implementation_sha256"
        ),
        candidate_universe_sha256=_require_sha256(
            payload["candidate_universe_sha256"], f"{path}.candidate_universe_sha256"
        ),
        direction=cast(Literal["maximize"], direction),
        statistic=cast(SelectionStatistic, statistic_value),
        statistic_implementation_sha256=_require_sha256(
            payload["statistic_implementation_sha256"],
            f"{path}.statistic_implementation_sha256",
        ),
        confidence=_require_probability(payload["confidence"], f"{path}.confidence"),
        bootstrap_resamples=_require_int(
            payload["bootstrap_resamples"],
            f"{path}.bootstrap_resamples",
            minimum=1,
            maximum=10_000_000,
        ),
        bootstrap_seed=_require_int(
            payload["bootstrap_seed"],
            f"{path}.bootstrap_seed",
            minimum=0,
            maximum=_MAX_SEED,
        ),
        bootstrap_rng_identity=cast(Literal["numpy_generator_pcg64"], rng_identity),
        bootstrap_rng_implementation_sha256=_require_sha256(
            payload["bootstrap_rng_implementation_sha256"],
            f"{path}.bootstrap_rng_implementation_sha256",
        ),
        resampling_unit=cast(Literal["candidate_seed_block"], resampling_unit),
        quantile_method=cast(Literal["linear"], quantile_method),
        bootstrap_interval=cast(Literal["two_sided_equal_tail"], bootstrap_interval),
        conservative_endpoint=cast(Literal["lower"], conservative_endpoint),
        endpoint_quantile=cast(Literal["(1-confidence)/2"], endpoint_quantile),
        tie_break=cast(Literal["candidate_id_ascending"], tie_break),
        groups=tuple(groups),
    )


def _parse_selection_slot(value: Any, path: str) -> SelectionSlot:
    payload = _require_object(value, path)
    _require_exact_keys(payload, path, {"selection_group", "rank"})
    return SelectionSlot(
        selection_group=_require_identifier(payload["selection_group"], f"{path}.selection_group"),
        rank=_require_int(payload["rank"], f"{path}.rank", minimum=1, maximum=_MAX_CANDIDATES),
    )


def _parse_selection_slots(
    value: Any, path: str, *, allow_empty: bool
) -> tuple[SelectionSlot, ...]:
    items = _require_array(value, path)
    if not allow_empty and not items:
        raise ForagerMatchedProtocolError(f"{path} must not be empty")
    slots = tuple(
        _parse_selection_slot(item, f"{path}[{index}]") for index, item in enumerate(items)
    )
    if len(set(slots)) != len(slots):
        raise ForagerMatchedProtocolError(f"{path} contains duplicate selection slots")
    return slots


def _parse_evaluation_panel(value: Any) -> EvaluationPanel:
    path = "protocol.evaluation_panel"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "selection_slots",
            "fixed_descriptive_candidate_ids",
            "alberta_primary_slot",
            "primary_nonprivileged_external_baseline_slot",
            "require_complete_blocks",
            "pairing_failure_policy",
        },
    )
    complete = _require_bool(payload["require_complete_blocks"], f"{path}.require_complete_blocks")
    if complete is not True:
        raise ForagerMatchedProtocolError(f"{path}.require_complete_blocks must be true")
    failure_policy = _require_literal(
        payload["pairing_failure_policy"],
        f"{path}.pairing_failure_policy",
        ("fail_closed",),
    )
    return EvaluationPanel(
        selection_slots=_parse_selection_slots(
            payload["selection_slots"], f"{path}.selection_slots", allow_empty=False
        ),
        fixed_descriptive_candidate_ids=_require_unique_identifiers(
            payload["fixed_descriptive_candidate_ids"],
            f"{path}.fixed_descriptive_candidate_ids",
            allow_empty=True,
        ),
        alberta_primary_slot=_parse_selection_slot(
            payload["alberta_primary_slot"], f"{path}.alberta_primary_slot"
        ),
        primary_nonprivileged_external_baseline_slot=_parse_selection_slot(
            payload["primary_nonprivileged_external_baseline_slot"],
            f"{path}.primary_nonprivileged_external_baseline_slot",
        ),
        require_complete_blocks=True,
        pairing_failure_policy=cast(Literal["fail_closed"], failure_policy),
    )


def _parse_analysis_plan(value: Any) -> MatchedAnalysisPlan:
    path = "protocol.analysis_plan"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"metric", "metric_implementation_sha256", "metric_direction", "primary", "secondary"},
    )
    primary_path = f"{path}.primary"
    primary = _require_object(payload["primary"], primary_path)
    _require_exact_keys(
        primary,
        primary_path,
        {
            "method",
            "resamples",
            "seed",
            "confidence",
            "primary_margin",
            "rng_algorithm",
            "quantile_method",
            "implementation_sha256",
            "gate",
        },
    )
    primary_method = _require_literal(
        primary["method"], primary_path + ".method", ("paired_percentile_bootstrap_lower_bound",)
    )
    primary_rng = _require_literal(
        primary["rng_algorithm"], primary_path + ".rng_algorithm", ("PCG64",)
    )
    primary_quantile = _require_literal(
        primary["quantile_method"], primary_path + ".quantile_method", ("linear",)
    )
    primary_gate = _require_literal(
        primary["gate"],
        primary_path + ".gate",
        ("lower_bound_strictly_greater_than_margin",),
    )

    secondary_path = f"{path}.secondary"
    secondary = _require_object(payload["secondary"], secondary_path)
    _require_exact_keys(
        secondary,
        secondary_path,
        {
            "method",
            "monte_carlo_resamples",
            "seed",
            "exact_max_pairs",
            "rng_algorithm",
            "implementation_sha256",
            "alternative",
            "multiplicity_method",
            "familywise_alpha",
        },
    )
    secondary_method = _require_literal(
        secondary["method"], secondary_path + ".method", ("paired_sign_flip",)
    )
    secondary_rng = _require_literal(
        secondary["rng_algorithm"], secondary_path + ".rng_algorithm", ("PCG64",)
    )
    alternative = _require_literal(
        secondary["alternative"], secondary_path + ".alternative", ("greater",)
    )
    multiplicity_method = _require_literal(
        secondary["multiplicity_method"],
        secondary_path + ".multiplicity_method",
        ("holm",),
    )
    exact_max = _require_int(
        secondary["exact_max_pairs"],
        secondary_path + ".exact_max_pairs",
        minimum=20,
        maximum=20,
    )
    metric_direction = _require_literal(
        payload["metric_direction"], f"{path}.metric_direction", ("maximize",)
    )
    return MatchedAnalysisPlan(
        metric=_require_identifier(payload["metric"], f"{path}.metric"),
        metric_implementation_sha256=_require_sha256(
            payload["metric_implementation_sha256"], f"{path}.metric_implementation_sha256"
        ),
        metric_direction=cast(Literal["maximize"], metric_direction),
        primary=PrimaryBootstrapAnalysis(
            method=cast(Literal["paired_percentile_bootstrap_lower_bound"], primary_method),
            resamples=_require_int(
                primary["resamples"], primary_path + ".resamples", minimum=1, maximum=10_000_000
            ),
            seed=_require_int(
                primary["seed"], primary_path + ".seed", minimum=0, maximum=_MAX_UINT64
            ),
            confidence=_require_probability_float(
                primary["confidence"], primary_path + ".confidence"
            ),
            primary_margin=_require_nonnegative_float(
                primary["primary_margin"], primary_path + ".primary_margin"
            ),
            rng_algorithm=cast(Literal["PCG64"], primary_rng),
            quantile_method=cast(Literal["linear"], primary_quantile),
            implementation_sha256=_require_sha256(
                primary["implementation_sha256"], primary_path + ".implementation_sha256"
            ),
            gate=cast(Literal["lower_bound_strictly_greater_than_margin"], primary_gate),
        ),
        secondary=SecondarySignFlipAnalysis(
            method=cast(Literal["paired_sign_flip"], secondary_method),
            monte_carlo_resamples=_require_int(
                secondary["monte_carlo_resamples"],
                secondary_path + ".monte_carlo_resamples",
                minimum=1,
                maximum=10_000_000,
            ),
            seed=_require_int(
                secondary["seed"], secondary_path + ".seed", minimum=0, maximum=_MAX_UINT64
            ),
            exact_max_pairs=cast(Literal[20], exact_max),
            rng_algorithm=cast(Literal["PCG64"], secondary_rng),
            implementation_sha256=_require_sha256(
                secondary["implementation_sha256"], secondary_path + ".implementation_sha256"
            ),
            alternative=cast(Literal["greater"], alternative),
            multiplicity_method=cast(Literal["holm"], multiplicity_method),
            familywise_alpha=_require_probability_float(
                secondary["familywise_alpha"], secondary_path + ".familywise_alpha"
            ),
        ),
    )


def _parse_hypothesis(
    value: Any,
    path: str,
    *,
    expected_method: AnalysisMethod,
) -> MatchedHypothesis:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "hypothesis_id",
            "intervention_slot",
            "comparator_slot",
            "estimand",
            "method",
            "alternative",
            "difference_order",
            "paired",
        },
    )
    estimand = _require_literal(
        payload["estimand"], f"{path}.estimand", ("paired_mean_difference",)
    )
    method = _require_literal(payload["method"], f"{path}.method", (expected_method,))
    alternative = _require_literal(payload["alternative"], f"{path}.alternative", ("greater",))
    difference_order = _require_literal(
        payload["difference_order"],
        f"{path}.difference_order",
        ("intervention_minus_comparator",),
    )
    paired = _require_bool(payload["paired"], f"{path}.paired")
    if paired is not True:
        raise ForagerMatchedProtocolError(f"{path}.paired must be true")
    return MatchedHypothesis(
        hypothesis_id=_require_identifier(payload["hypothesis_id"], f"{path}.hypothesis_id"),
        intervention_slot=_parse_selection_slot(
            payload["intervention_slot"], f"{path}.intervention_slot"
        ),
        comparator_slot=_parse_selection_slot(
            payload["comparator_slot"], f"{path}.comparator_slot"
        ),
        estimand=cast(Literal["paired_mean_difference"], estimand),
        method=cast(AnalysisMethod, method),
        alternative=cast(Literal["greater"], alternative),
        difference_order=cast(Literal["intervention_minus_comparator"], difference_order),
        paired=True,
    )


def _parse_multiplicity_policy(value: Any) -> MultiplicityPolicy:
    path = "protocol.multiplicity_policy"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"method", "alpha", "hypothesis_ids", "primary_excluded"},
    )
    method = _require_literal(payload["method"], f"{path}.method", ("holm",))
    primary_excluded = _require_bool(payload["primary_excluded"], f"{path}.primary_excluded")
    if primary_excluded is not True:
        raise ForagerMatchedProtocolError(f"{path}.primary_excluded must be true")
    return MultiplicityPolicy(
        method=cast(Literal["holm"], method),
        alpha=_require_probability(payload["alpha"], f"{path}.alpha"),
        hypothesis_ids=_require_unique_identifiers(
            payload["hypothesis_ids"], f"{path}.hypothesis_ids", allow_empty=True
        ),
        primary_excluded=True,
    )


def _parse_resolved_selection_slot(value: Any, path: str) -> ResolvedSelectionSlot:
    payload = _require_object(value, path)
    _require_exact_keys(payload, path, {"selection_group", "rank", "candidate_id"})
    return ResolvedSelectionSlot(
        selection_group=_require_identifier(payload["selection_group"], f"{path}.selection_group"),
        rank=_require_int(payload["rank"], f"{path}.rank", minimum=1, maximum=_MAX_CANDIDATES),
        candidate_id=_require_identifier(payload["candidate_id"], f"{path}.candidate_id"),
    )


def _parse_selection_outcome(
    value: Any,
    *,
    stage: Stage,
    expected_slots: tuple[SelectionSlot, ...],
) -> SelectionOutcome:
    path = "protocol.selection_outcome"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"status", "open_protocol_sha256", "selection_result_sha256", "resolved_slots"},
    )
    status = _require_literal(payload["status"], f"{path}.status", ("pending", "resolved"))
    items = _require_array(payload["resolved_slots"], f"{path}.resolved_slots")
    if len(items) > _MAX_CANDIDATES:
        raise ForagerMatchedProtocolError(f"{path}.resolved_slots contains too many items")
    resolved = tuple(
        _parse_resolved_selection_slot(item, f"{path}.resolved_slots[{index}]")
        for index, item in enumerate(items)
    )
    if stage == "open_tuning":
        if (
            status != "pending"
            or payload["open_protocol_sha256"] is not None
            or payload["selection_result_sha256"] is not None
            or resolved
        ):
            raise ForagerMatchedProtocolError(
                f"{path} must be pending with null digests and no resolutions during open tuning"
            )
        return SelectionOutcome(
            status="pending",
            open_protocol_sha256=None,
            selection_result_sha256=None,
            resolved_slots=(),
        )
    if status != "resolved":
        raise ForagerMatchedProtocolError(f"{path} must be resolved during sealed evaluation")
    open_digest = _require_sha256(payload["open_protocol_sha256"], f"{path}.open_protocol_sha256")
    result_digest = _require_sha256(
        payload["selection_result_sha256"], f"{path}.selection_result_sha256"
    )
    if tuple(item.slot for item in resolved) != expected_slots:
        raise ForagerMatchedProtocolError(
            f"{path}.resolved_slots must exactly match the selection-plan slots and order"
        )
    if len({item.candidate_id for item in resolved}) != len(resolved):
        raise ForagerMatchedProtocolError(f"{path}.resolved_slots repeats a selected candidate")
    return SelectionOutcome(
        status="resolved",
        open_protocol_sha256=open_digest,
        selection_result_sha256=result_digest,
        resolved_slots=resolved,
    )


def _parse_context(value: Any, path: str) -> DescriptiveContext:
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {"candidate_ids", "analysis_role", "selection_eligible", "pairing_eligible"},
    )
    role = _require_literal(
        payload["analysis_role"], f"{path}.analysis_role", ("descriptive_only",)
    )
    selection = _require_bool(payload["selection_eligible"], f"{path}.selection_eligible")
    pairing = _require_bool(payload["pairing_eligible"], f"{path}.pairing_eligible")
    if selection or pairing:
        raise ForagerMatchedProtocolError(f"{path} must be ineligible for selection and pairing")
    return DescriptiveContext(
        candidate_ids=_require_unique_identifiers(
            payload["candidate_ids"], f"{path}.candidate_ids", allow_empty=True
        ),
        analysis_role=cast(Literal["descriptive_only"], role),
        selection_eligible=False,
        pairing_eligible=False,
    )


def _parse_runtime(value: Any) -> MatchedRuntime:
    path = "protocol.runtime"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "executor_kind",
            "image_sha256",
            "runtime_profile_sha256",
            "executor_qualification_receipt_sha256",
            "qualification_trust_anchor_identity",
            "source_mount_mode",
            "default_prng",
            "threefry_partitionable",
            "platform",
            "sandbox",
        },
    )
    sandbox_path = f"{path}.sandbox"
    sandbox_payload = _require_object(payload["sandbox"], sandbox_path)
    _require_exact_keys(
        sandbox_payload,
        sandbox_path,
        {
            "network",
            "root_filesystem",
            "capabilities",
            "no_new_privileges",
            "container_user",
            "host_devices",
            "writable_tmpfs_only",
        },
    )
    network = _require_literal(sandbox_payload["network"], f"{sandbox_path}.network", ("none",))
    root = _require_literal(
        sandbox_payload["root_filesystem"],
        f"{sandbox_path}.root_filesystem",
        ("read_only",),
    )
    capabilities = _require_literal(
        sandbox_payload["capabilities"],
        f"{sandbox_path}.capabilities",
        ("all_dropped",),
    )
    no_new = _require_bool(
        sandbox_payload["no_new_privileges"], f"{sandbox_path}.no_new_privileges"
    )
    writable_tmpfs = _require_bool(
        sandbox_payload["writable_tmpfs_only"], f"{sandbox_path}.writable_tmpfs_only"
    )
    if not no_new or not writable_tmpfs:
        raise ForagerMatchedProtocolError(
            f"{sandbox_path} must enable no_new_privileges and writable_tmpfs_only"
        )
    user = _require_string(
        sandbox_payload["container_user"], f"{sandbox_path}.container_user", maximum=64
    )
    if _CONTAINER_USER.fullmatch(user) is None:
        raise ForagerMatchedProtocolError(
            f"{sandbox_path}.container_user must be a non-root numeric uid:gid"
        )
    devices = _require_array(sandbox_payload["host_devices"], f"{sandbox_path}.host_devices")
    if devices:
        raise ForagerMatchedProtocolError(f"{sandbox_path}.host_devices must be empty")
    partitionable = _require_bool(
        payload["threefry_partitionable"], f"{path}.threefry_partitionable"
    )
    if partitionable is not True:
        raise ForagerMatchedProtocolError(f"{path}.threefry_partitionable must be true")
    executor = _require_literal(payload["executor_kind"], f"{path}.executor_kind", ("oci",))
    mount = _require_literal(
        payload["source_mount_mode"],
        f"{path}.source_mount_mode",
        ("read_only_content_addressed_mount",),
    )
    prng = _require_literal(payload["default_prng"], f"{path}.default_prng", ("threefry2x32",))
    platform = _require_literal(payload["platform"], f"{path}.platform", ("cpu",))
    return MatchedRuntime(
        executor_kind=cast(Literal["oci"], executor),
        image_sha256=_require_sha256(payload["image_sha256"], f"{path}.image_sha256"),
        runtime_profile_sha256=_require_sha256(
            payload["runtime_profile_sha256"], f"{path}.runtime_profile_sha256"
        ),
        executor_qualification_receipt_sha256=_require_sha256(
            payload["executor_qualification_receipt_sha256"],
            f"{path}.executor_qualification_receipt_sha256",
        ),
        qualification_trust_anchor_identity=_require_identifier(
            payload["qualification_trust_anchor_identity"],
            f"{path}.qualification_trust_anchor_identity",
        ),
        source_mount_mode=cast(Literal["read_only_content_addressed_mount"], mount),
        default_prng=cast(Literal["threefry2x32"], prng),
        threefry_partitionable=True,
        platform=cast(Literal["cpu"], platform),
        sandbox=CPUSandboxContract(
            network=cast(Literal["none"], network),
            root_filesystem=cast(Literal["read_only"], root),
            capabilities=cast(Literal["all_dropped"], capabilities),
            no_new_privileges=True,
            container_user=user,
            host_devices=(),
            writable_tmpfs_only=True,
        ),
    )


def _validate_candidate_contracts(
    *,
    candidates: tuple[MatchedCandidate, ...],
    task: MatchedTask,
    runtime: MatchedRuntime,
    all_seeds: tuple[int, ...],
) -> None:
    for candidate in candidates:
        path = f"protocol.candidates[{candidate.candidate_id}]"
        binding = candidate.runtime_binding
        if candidate.stratum != "historical_orientation":
            if binding.image_sha256 != runtime.image_sha256:
                raise ForagerMatchedProtocolError(
                    f"{path} runtime image does not match protocol.runtime"
                )
            if binding.runtime_profile_sha256 != runtime.runtime_profile_sha256:
                raise ForagerMatchedProtocolError(
                    f"{path} runtime profile does not match protocol.runtime"
                )
            if binding.task_identity_sha256 != task.task_identity_sha256:
                raise ForagerMatchedProtocolError(
                    f"{path} task identity does not match protocol.task"
                )
            if (
                binding.qualification_trust_anchor_identity
                != runtime.qualification_trust_anchor_identity
            ):
                raise ForagerMatchedProtocolError(
                    f"{path} capability receipt trust anchor does not match protocol.runtime"
                )

        effective = tuple(seed + candidate.seed_contract.offset for seed in all_seeds)
        if any(seed < 0 or seed > _MAX_SEED for seed in effective):
            raise ForagerMatchedProtocolError(f"{path} effective seed lies outside int31 range")
        if candidate.pairing.eligible and candidate.seed_contract.offset != 0:
            raise ForagerMatchedProtocolError(
                f"{path} pairing-eligible candidates must preserve the active seed exactly"
            )

        shared = candidate.agent_rng.environment_key_shared
        shared_identity = candidate.agent_rng.identity == "shared_agent_environment_rng_v1"
        if shared != shared_identity:
            raise ForagerMatchedProtocolError(
                f"{path} agent RNG identity and environment-key sharing declaration disagree"
            )

        if candidate.implementation_kind in _KNOWN_PRIVILEGED_IMPLEMENTATIONS and (
            candidate.stratum != "privileged_context"
            or candidate.observation_access.access_mode
            not in ("privileged_global_objects", "privileged_reward_grid")
        ):
            raise ForagerMatchedProtocolError(
                f"{path} known privileged implementation must remain in the privileged stratum"
            )
        if (
            candidate.implementation_kind in _KNOWN_HISTORICAL_IMPLEMENTATIONS
            or candidate.entrypoint_family == "historical_legacy"
        ) and candidate.stratum != "historical_orientation":
            raise ForagerMatchedProtocolError(
                f"{path} known historical implementation must remain historical orientation"
            )

        access_mode = candidate.observation_access.access_mode
        if access_mode in ("privileged_global_objects", "privileged_reward_grid"):
            if candidate.stratum != "privileged_context":
                raise ForagerMatchedProtocolError(
                    f"{path} privileged observation access must use the privileged stratum"
                )
        elif access_mode == "historical_legacy":
            if candidate.stratum != "historical_orientation":
                raise ForagerMatchedProtocolError(
                    f"{path} legacy observation access must use the historical stratum"
                )
        elif candidate.stratum in ("privileged_context", "historical_orientation"):
            raise ForagerMatchedProtocolError(
                f"{path} context stratum does not match its observation access mode"
            )

        if candidate.implementation_kind in _EXACT_SHARED_RNG_IMPLEMENTATIONS:
            if (
                candidate.stratum != "external_learning"
                or candidate.agent_rng.identity != "shared_agent_environment_rng_v1"
                or not shared
                or candidate.environment_rng.identity != "shared_agent_environment_rng_v1"
                or candidate.pairing.eligible
                or candidate.pairing.analysis_role != "descriptive_only"
                or "shared_agent_environment_rng" not in candidate.pairing.exclusion_reasons
            ):
                raise ForagerMatchedProtocolError(
                    f"{path} exact upstream PPO/RTU-PPO must be shared-RNG, descriptive, "
                    "and pairing-ineligible"
                )

        if candidate.stratum == "privileged_context":
            if (
                candidate.pairing.eligible
                or candidate.pairing.analysis_role != "descriptive_only"
                or not candidate.observation_access.privileged_fields
                or "privileged_observation_access" not in candidate.pairing.exclusion_reasons
            ):
                raise ForagerMatchedProtocolError(
                    f"{path} privileged candidates must be descriptive and pairing-ineligible"
                )
        elif candidate.stratum == "historical_orientation":
            if (
                candidate.pairing.eligible
                or candidate.pairing.analysis_role != "descriptive_only"
                or candidate.observation_access.access_mode != "historical_legacy"
                or "historical_runtime_mismatch" not in candidate.pairing.exclusion_reasons
            ):
                raise ForagerMatchedProtocolError(
                    f"{path} historical candidates must be descriptive legacy orientation"
                )
        elif candidate.pairing.eligible:
            if (
                candidate.environment_rng.identity != "dedicated_environment_split_chain_v1"
                or candidate.environment_rng.schedule_sha256 != task.environment_rng_schedule_sha256
                or shared
                or candidate.agent_rng.identity != "isolated_agent_rng_v1"
            ):
                raise ForagerMatchedProtocolError(
                    f"{path} pairing-eligible candidates require the common dedicated "
                    "environment RNG schedule and isolated agent RNG"
                )
            access = candidate.observation_access
            if (
                access.access_mode != "partial_observation"
                or access.privileged_fields
                or access.observation_type != task.observation_type
                or access.aperture_size != task.aperture_size
            ):
                raise ForagerMatchedProtocolError(
                    f"{path} pairing-eligible candidates must use the common partial observation"
                )

        if binding.qualified_capability_descriptor_sha256 != candidate_capability_descriptor_sha256(
            candidate
        ):
            raise ForagerMatchedProtocolError(
                f"{path} capability receipt subject does not match candidate semantics"
            )


def _validate_cross_references(
    *,
    candidates: tuple[MatchedCandidate, ...],
    selection_plan: SelectionPlan,
    selection_outcome: SelectionOutcome,
    analysis_plan: MatchedAnalysisPlan,
    evaluation_panel: EvaluationPanel,
    primary: MatchedHypothesis,
    secondary: tuple[MatchedHypothesis, ...],
    multiplicity: MultiplicityPolicy,
    privileged: DescriptiveContext,
    historical: DescriptiveContext,
) -> None:
    index = {candidate.candidate_id: candidate for candidate in candidates}
    if (
        analysis_plan.metric != selection_plan.metric
        or analysis_plan.metric_implementation_sha256 != selection_plan.metric_implementation_sha256
        or analysis_plan.metric_direction != selection_plan.direction
    ):
        raise ForagerMatchedProtocolError(
            "protocol.analysis_plan metric identity must exactly match selection_plan"
        )
    grouped_candidate_ids: dict[str, list[str]] = {}
    eligible_group_order: list[str] = []
    for candidate in candidates:
        if candidate.pairing.eligible:
            if candidate.selection_group not in grouped_candidate_ids:
                eligible_group_order.append(candidate.selection_group)
                grouped_candidate_ids[candidate.selection_group] = []
            grouped_candidate_ids[candidate.selection_group].append(candidate.candidate_id)
    planned_group_order = [group.selection_group for group in selection_plan.groups]
    if planned_group_order != eligible_group_order:
        raise ForagerMatchedProtocolError(
            "protocol.selection_plan group IDs/order must exactly cover pairing-eligible "
            "candidate groups"
        )
    for group in selection_plan.groups:
        expected_ids = tuple(grouped_candidate_ids.get(group.selection_group, []))
        if group.candidate_ids != expected_ids:
            raise ForagerMatchedProtocolError(
                f"protocol.selection_plan group {group.selection_group!r} candidate IDs/order "
                "must exactly match pairing-eligible candidates"
            )
        strata = {index[candidate_id].stratum for candidate_id in group.candidate_ids}
        if len(strata) != 1:
            raise ForagerMatchedProtocolError(
                f"protocol.selection_plan group {group.selection_group!r} may not mix strata"
            )

    expected_slots = selection_plan.slots
    if evaluation_panel.selection_slots != expected_slots:
        raise ForagerMatchedProtocolError(
            "protocol.evaluation_panel selection slots/order must exactly match the "
            "selection-plan winner slots"
        )
    expected_descriptive = tuple(
        candidate.candidate_id
        for candidate in candidates
        if not candidate.pairing.eligible and candidate.stratum != "historical_orientation"
    )
    if evaluation_panel.fixed_descriptive_candidate_ids != expected_descriptive:
        raise ForagerMatchedProtocolError(
            "protocol.evaluation_panel fixed descriptive IDs/order must exactly match all "
            "nonhistorical pairing-ineligible candidates"
        )
    primary_slots = (
        evaluation_panel.alberta_primary_slot,
        evaluation_panel.primary_nonprivileged_external_baseline_slot,
    )
    if primary_slots[0] == primary_slots[1] or any(
        slot not in evaluation_panel.selection_slots for slot in primary_slots
    ):
        raise ForagerMatchedProtocolError(
            "protocol.evaluation_panel primary slots must be distinct panel selection slots"
        )
    groups_by_id = {group.selection_group: group for group in selection_plan.groups}
    alberta_group = groups_by_id[primary_slots[0].selection_group]
    baseline_group = groups_by_id[primary_slots[1].selection_group]
    if any(
        index[candidate_id].stratum != "alberta_learning"
        for candidate_id in alberta_group.candidate_ids
    ):
        raise ForagerMatchedProtocolError(
            "protocol.evaluation_panel Alberta primary slot must select an Alberta group"
        )
    if any(
        index[candidate_id].stratum != "external_learning"
        for candidate_id in baseline_group.candidate_ids
    ):
        raise ForagerMatchedProtocolError(
            "protocol.evaluation_panel external primary slot must select an external group"
        )

    if selection_outcome.status == "resolved":
        for resolution in selection_outcome.resolved_slots:
            group = groups_by_id[resolution.selection_group]
            if resolution.candidate_id not in group.candidate_ids:
                raise ForagerMatchedProtocolError(
                    "protocol.selection_outcome resolves a slot to a candidate outside its group"
                )

    expected_privileged = tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.stratum == "privileged_context"
    )
    expected_historical = tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.stratum == "historical_orientation"
    )
    if privileged.candidate_ids != expected_privileged:
        raise ForagerMatchedProtocolError(
            "protocol.privileged_context candidate IDs/order must exactly match the "
            "privileged candidate stratum"
        )
    if historical.candidate_ids != expected_historical:
        raise ForagerMatchedProtocolError(
            "protocol.historical_orientation candidate IDs/order must exactly match the "
            "historical candidate stratum"
        )

    if primary.intervention_slot != primary_slots[0] or primary.comparator_slot != primary_slots[1]:
        raise ForagerMatchedProtocolError(
            "protocol.primary_hypothesis slots must exactly match the evaluation primary roles"
        )
    hypothesis_ids = [primary.hypothesis_id]
    seen_pairs: set[frozenset[SelectionSlot]] = {
        frozenset((primary.intervention_slot, primary.comparator_slot))
    }
    for index_number, hypothesis in enumerate(secondary):
        path = f"protocol.secondary_hypotheses[{index_number}]"
        for slot in (
            hypothesis.intervention_slot,
            hypothesis.comparator_slot,
        ):
            if slot not in evaluation_panel.selection_slots:
                raise ForagerMatchedProtocolError(
                    f"{path} references a selection slot outside the panel"
                )
        if hypothesis.intervention_slot == hypothesis.comparator_slot:
            raise ForagerMatchedProtocolError(f"{path} compares a selection slot with itself")
        pair = frozenset((hypothesis.intervention_slot, hypothesis.comparator_slot))
        if pair in seen_pairs:
            raise ForagerMatchedProtocolError(
                f"{path} duplicates or reverses an earlier slot contrast"
            )
        seen_pairs.add(pair)
        hypothesis_ids.append(hypothesis.hypothesis_id)
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ForagerMatchedProtocolError("protocol hypothesis IDs must be unique")
    expected_secondary_ids = tuple(item.hypothesis_id for item in secondary)
    if multiplicity.hypothesis_ids != expected_secondary_ids:
        raise ForagerMatchedProtocolError(
            "protocol.multiplicity_policy hypothesis IDs/order must exactly match "
            "secondary_hypotheses"
        )
    if (
        analysis_plan.secondary.multiplicity_method != multiplicity.method
        or analysis_plan.secondary.familywise_alpha != multiplicity.alpha
    ):
        raise ForagerMatchedProtocolError(
            "protocol.analysis_plan secondary Holm method/alpha must exactly match "
            "multiplicity_policy"
        )


def parse_forager_matched_protocol(value: Any) -> ForagerMatchedProtocol:
    """Decode if needed, then validate and normalize a matched protocol."""
    if type(value) in (bytes, str):
        value = decode_strict_json(value)
    _validate_json_complexity(value)
    payload = _require_object(value, "protocol")
    top_level_keys = {
        "schema_version",
        "stage",
        "task",
        "horizon",
        "tuning_seeds",
        "evaluation_seeds",
        "active_seeds",
        "candidates",
        "selection_plan",
        "selection_outcome",
        "analysis_plan",
        "evaluation_panel",
        "primary_hypothesis",
        "secondary_hypotheses",
        "multiplicity_policy",
        "privileged_context",
        "historical_orientation",
        "runtime",
    }
    _require_exact_keys(payload, "protocol", top_level_keys)
    schema = _require_string(payload["schema_version"], "protocol.schema_version", maximum=64)
    if schema != FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION:
        raise ForagerMatchedProtocolError(
            f"protocol.schema_version must be {FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION!r}"
        )
    stage_value = _require_literal(
        payload["stage"], "protocol.stage", ("open_tuning", "sealed_evaluation")
    )
    stage_value = cast(Stage, stage_value)
    task = _parse_task(payload["task"])
    horizon = _require_int(payload["horizon"], "protocol.horizon", minimum=1, maximum=_MAX_HORIZON)
    tuning_seeds = _require_seeds(payload["tuning_seeds"], "protocol.tuning_seeds")
    evaluation_seeds = _require_seeds(payload["evaluation_seeds"], "protocol.evaluation_seeds")
    overlap = sorted(set(tuning_seeds) & set(evaluation_seeds))
    if overlap:
        raise ForagerMatchedProtocolError(
            "protocol tuning and evaluation seeds overlap: "
            + ", ".join(str(seed) for seed in overlap)
        )
    active_seeds = _require_seeds(payload["active_seeds"], "protocol.active_seeds")
    expected_active = tuning_seeds if stage_value == "open_tuning" else evaluation_seeds
    if active_seeds != expected_active:
        raise ForagerMatchedProtocolError(
            "protocol.active_seeds must exactly match the seed set for protocol.stage"
        )

    candidate_payload = _require_array(payload["candidates"], "protocol.candidates")
    if not candidate_payload or len(candidate_payload) > _MAX_CANDIDATES:
        raise ForagerMatchedProtocolError(
            f"protocol.candidates must contain between 1 and {_MAX_CANDIDATES} items"
        )
    candidates = tuple(
        _parse_candidate(item, f"protocol.candidates[{index}]", horizon=horizon)
        for index, item in enumerate(candidate_payload)
    )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ForagerMatchedProtocolError("protocol.candidates contains duplicate candidate IDs")

    selection_plan = _parse_selection_plan(payload["selection_plan"])
    selection_outcome = _parse_selection_outcome(
        payload["selection_outcome"],
        stage=stage_value,
        expected_slots=selection_plan.slots,
    )
    analysis_plan = _parse_analysis_plan(payload["analysis_plan"])
    evaluation_panel = _parse_evaluation_panel(payload["evaluation_panel"])
    primary = _parse_hypothesis(
        payload["primary_hypothesis"],
        "protocol.primary_hypothesis",
        expected_method="paired_percentile_bootstrap_lower_bound",
    )
    secondary_payload = _require_array(
        payload["secondary_hypotheses"], "protocol.secondary_hypotheses"
    )
    if len(secondary_payload) > _MAX_HYPOTHESES:
        raise ForagerMatchedProtocolError(
            "protocol.secondary_hypotheses contains too many hypotheses"
        )
    secondary = tuple(
        _parse_hypothesis(
            item,
            f"protocol.secondary_hypotheses[{index}]",
            expected_method="paired_sign_flip",
        )
        for index, item in enumerate(secondary_payload)
    )
    multiplicity = _parse_multiplicity_policy(payload["multiplicity_policy"])
    privileged = _parse_context(payload["privileged_context"], "protocol.privileged_context")
    historical = _parse_context(
        payload["historical_orientation"], "protocol.historical_orientation"
    )
    runtime = _parse_runtime(payload["runtime"])

    all_seeds = tuning_seeds + evaluation_seeds
    _validate_candidate_contracts(
        candidates=candidates,
        task=task,
        runtime=runtime,
        all_seeds=all_seeds,
    )
    _validate_cross_references(
        candidates=candidates,
        selection_plan=selection_plan,
        selection_outcome=selection_outcome,
        analysis_plan=analysis_plan,
        evaluation_panel=evaluation_panel,
        primary=primary,
        secondary=secondary,
        multiplicity=multiplicity,
        privileged=privileged,
        historical=historical,
    )
    candidate_index = MappingProxyType(
        {candidate.candidate_id: candidate for candidate in candidates}
    )
    return ForagerMatchedProtocol(
        schema_version=FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION,
        stage=stage_value,
        task=task,
        horizon=horizon,
        tuning_seeds=tuning_seeds,
        evaluation_seeds=evaluation_seeds,
        active_seeds=active_seeds,
        candidates=candidates,
        selection_plan=selection_plan,
        selection_outcome=selection_outcome,
        analysis_plan=analysis_plan,
        evaluation_panel=evaluation_panel,
        primary_hypothesis=primary,
        secondary_hypotheses=secondary,
        multiplicity_policy=multiplicity,
        privileged_context=privileged,
        historical_orientation=historical,
        runtime=runtime,
        candidate_index=candidate_index,
    )


def parse_forager_matched_selection_result(value: Any) -> ForagerMatchedSelectionResult:
    """Validate a canonicalizable, reward-opaque open-tuning selection result."""
    if type(value) is ForagerMatchedSelectionResult:
        value = value.to_dict()
    if type(value) in (bytes, str):
        value = decode_strict_json(value)
    _validate_json_complexity(value)
    path = "selection_result"
    payload = _require_object(value, path)
    _require_exact_keys(
        payload,
        path,
        {
            "schema_version",
            "open_protocol_sha256",
            "selection_plan_sha256",
            "tuning_seeds",
            "ranked_groups",
        },
    )
    schema = _require_string(payload["schema_version"], f"{path}.schema_version", maximum=64)
    if schema != FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION:
        raise ForagerMatchedProtocolError(
            f"{path}.schema_version must be {FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION!r}"
        )
    group_values = _require_array(payload["ranked_groups"], f"{path}.ranked_groups")
    if not group_values or len(group_values) > _MAX_CANDIDATES:
        raise ForagerMatchedProtocolError(
            f"{path}.ranked_groups must contain between 1 and {_MAX_CANDIDATES} items"
        )
    ranked_groups: list[RankedSelectionGroup] = []
    for index, item in enumerate(group_values):
        item_path = f"{path}.ranked_groups[{index}]"
        group = _require_object(item, item_path)
        _require_exact_keys(
            group,
            item_path,
            {"selection_group", "ranked_candidate_ids", "ranking_evidence_sha256"},
        )
        ranked_groups.append(
            RankedSelectionGroup(
                selection_group=_require_identifier(
                    group["selection_group"], f"{item_path}.selection_group"
                ),
                ranked_candidate_ids=_require_unique_identifiers(
                    group["ranked_candidate_ids"],
                    f"{item_path}.ranked_candidate_ids",
                    allow_empty=False,
                ),
                ranking_evidence_sha256=_require_sha256(
                    group["ranking_evidence_sha256"],
                    f"{item_path}.ranking_evidence_sha256",
                ),
            )
        )
    if len({group.selection_group for group in ranked_groups}) != len(ranked_groups):
        raise ForagerMatchedProtocolError(f"{path}.ranked_groups repeats a selection group")
    return ForagerMatchedSelectionResult(
        schema_version=FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION,
        open_protocol_sha256=_require_sha256(
            payload["open_protocol_sha256"], f"{path}.open_protocol_sha256"
        ),
        selection_plan_sha256=_require_sha256(
            payload["selection_plan_sha256"], f"{path}.selection_plan_sha256"
        ),
        tuning_seeds=_require_seeds(payload["tuning_seeds"], f"{path}.tuning_seeds"),
        ranked_groups=tuple(ranked_groups),
    )


def _canonical_plain_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ForagerMatchedProtocolError(f"value is not canonical JSON data: {exc}") from exc


def canonical_selection_result_bytes(
    value: ForagerMatchedSelectionResult | Mapping[str, Any] | bytes | str,
) -> bytes:
    """Return canonical bytes after replaying all selection-result validation."""
    return _canonical_plain_json_bytes(parse_forager_matched_selection_result(value).to_dict())


def canonical_selection_result_sha256(
    value: ForagerMatchedSelectionResult | Mapping[str, Any] | bytes | str,
) -> str:
    """Return the canonical SHA-256 for a validated selection result."""
    return hashlib.sha256(canonical_selection_result_bytes(value)).hexdigest()


def _parse_protocol_instance(
    value: ForagerMatchedProtocol | Mapping[str, Any] | bytes | str,
) -> ForagerMatchedProtocol:
    if type(value) is ForagerMatchedProtocol:
        return parse_forager_matched_protocol(value.to_dict())
    return parse_forager_matched_protocol(value)


def _stage_invariant_protocol_bytes(protocol: ForagerMatchedProtocol) -> bytes:
    payload = protocol.to_dict()
    for key in ("stage", "active_seeds", "selection_outcome"):
        del payload[key]
    return _canonical_plain_json_bytes(payload)


def seal_forager_matched_protocol(
    open_protocol: ForagerMatchedProtocol | Mapping[str, Any] | bytes | str,
    selection_result: ForagerMatchedSelectionResult | Mapping[str, Any] | bytes | str,
) -> ForagerMatchedProtocol:
    """Mechanically resolve an open protocol into its sealed-evaluation form.

    This function is reward-blind and authority-blind.  It does not authenticate
    the ranking evidence named by ``selection_result``; callers must obtain that
    result through the separately authenticated selection workflow.  Its only
    job is to make the stage transition reproducible without hand-editing JSON,
    while proving that every stage-invariant field remains byte-identical.
    """
    open_value = _parse_protocol_instance(open_protocol)
    if open_value.stage != "open_tuning":
        raise ForagerMatchedProtocolError("only an open_tuning protocol can be sealed")
    result = parse_forager_matched_selection_result(selection_result)
    if result.open_protocol_sha256 != open_value.protocol_sha256:
        raise ForagerMatchedProtocolError(
            "selection result is not bound to the canonical open protocol"
        )
    if result.selection_plan_sha256 != open_value.selection_plan.plan_sha256:
        raise ForagerMatchedProtocolError(
            "selection result is not bound to the canonical selection plan"
        )
    if result.tuning_seeds != open_value.tuning_seeds:
        raise ForagerMatchedProtocolError(
            "selection result tuning seeds/order do not match the open protocol"
        )
    plan_groups = open_value.selection_plan.groups
    if len(result.ranked_groups) != len(plan_groups):
        raise ForagerMatchedProtocolError(
            "selection result groups/order do not match the selection plan"
        )
    resolutions: list[dict[str, Any]] = []
    for plan_group, ranked_group in zip(
        plan_groups, result.ranked_groups, strict=True
    ):
        if ranked_group.selection_group != plan_group.selection_group:
            raise ForagerMatchedProtocolError(
                "selection result groups/order do not match the selection plan"
            )
        if (
            len(ranked_group.ranked_candidate_ids) != len(plan_group.candidate_ids)
            or set(ranked_group.ranked_candidate_ids) != set(plan_group.candidate_ids)
        ):
            raise ForagerMatchedProtocolError(
                f"selection result group {plan_group.selection_group!r} must rank every "
                "preregistered candidate exactly once"
            )
        resolutions.extend(
            {
                "selection_group": plan_group.selection_group,
                "rank": rank,
                "candidate_id": ranked_group.ranked_candidate_ids[rank - 1],
            }
            for rank in range(1, plan_group.advance_count + 1)
        )

    sealed_payload = open_value.to_dict()
    sealed_payload["stage"] = "sealed_evaluation"
    sealed_payload["active_seeds"] = list(open_value.evaluation_seeds)
    sealed_payload["selection_outcome"] = {
        "status": "resolved",
        "open_protocol_sha256": open_value.protocol_sha256,
        "selection_result_sha256": result.selection_result_sha256,
        "resolved_slots": resolutions,
    }
    sealed = parse_forager_matched_protocol(sealed_payload)
    validate_sealed_protocol_transition(
        open_value,
        sealed,
        result,
        result.selection_result_sha256,
    )
    return sealed


def validate_sealed_protocol_transition(
    open_protocol: ForagerMatchedProtocol | Mapping[str, Any] | bytes | str,
    sealed_protocol: ForagerMatchedProtocol | Mapping[str, Any] | bytes | str,
    selection_result: ForagerMatchedSelectionResult | Mapping[str, Any] | bytes | str,
    selection_result_sha256: str | None = None,
) -> SealedProtocolValidation:
    """Prove a sealed protocol is the immutable resolution of one open protocol.

    This function never consumes rewards.  It verifies the canonical predecessor
    and result digests, the full candidate permutations reported for every tuning
    group, winner ranks, and the exact slot resolution of the panel and hypotheses.
    Ranking-evidence and qualification receipts remain mandatory external evidence
    that must be authenticated through their declared trust anchor.
    """
    open_value = _parse_protocol_instance(open_protocol)
    sealed_value = _parse_protocol_instance(sealed_protocol)
    if open_value.stage != "open_tuning":
        raise ForagerMatchedProtocolError("open protocol must have stage open_tuning")
    if sealed_value.stage != "sealed_evaluation":
        raise ForagerMatchedProtocolError("sealed protocol must have stage sealed_evaluation")
    if _stage_invariant_protocol_bytes(open_value) != _stage_invariant_protocol_bytes(sealed_value):
        raise ForagerMatchedProtocolError(
            "sealed protocol changed a field outside stage, active_seeds, or selection_outcome"
        )

    open_digest = open_value.protocol_sha256
    outcome = sealed_value.selection_outcome
    if outcome.open_protocol_sha256 != open_digest:
        raise ForagerMatchedProtocolError(
            "sealed protocol open_protocol_sha256 does not match the canonical open protocol"
        )
    result = parse_forager_matched_selection_result(selection_result)
    result_digest = result.selection_result_sha256
    if selection_result_sha256 is not None:
        expected_digest = _require_sha256(
            selection_result_sha256, "selection_result_sha256 argument"
        )
        if result_digest != expected_digest:
            raise ForagerMatchedProtocolError(
                "selection result payload does not match the supplied canonical digest"
            )
    if outcome.selection_result_sha256 != result_digest:
        raise ForagerMatchedProtocolError(
            "sealed protocol selection_result_sha256 does not match the canonical result"
        )
    if result.open_protocol_sha256 != open_digest:
        raise ForagerMatchedProtocolError(
            "selection result is not bound to the canonical open protocol"
        )
    if result.selection_plan_sha256 != open_value.selection_plan.plan_sha256:
        raise ForagerMatchedProtocolError(
            "selection result is not bound to the canonical selection plan"
        )
    if result.tuning_seeds != open_value.tuning_seeds:
        raise ForagerMatchedProtocolError(
            "selection result tuning seeds/order do not match the open protocol"
        )

    plan_groups = open_value.selection_plan.groups
    if tuple(group.selection_group for group in result.ranked_groups) != tuple(
        group.selection_group for group in plan_groups
    ):
        raise ForagerMatchedProtocolError(
            "selection result groups/order do not match the selection plan"
        )
    expected_resolutions: list[ResolvedSelectionSlot] = []
    for plan_group, ranked_group in zip(plan_groups, result.ranked_groups, strict=True):
        if len(ranked_group.ranked_candidate_ids) != len(plan_group.candidate_ids) or set(
            ranked_group.ranked_candidate_ids
        ) != set(plan_group.candidate_ids):
            raise ForagerMatchedProtocolError(
                f"selection result group {plan_group.selection_group!r} must rank every "
                "preregistered candidate exactly once"
            )
        expected_resolutions.extend(
            ResolvedSelectionSlot(
                selection_group=plan_group.selection_group,
                rank=rank,
                candidate_id=ranked_group.ranked_candidate_ids[rank - 1],
            )
            for rank in range(1, plan_group.advance_count + 1)
        )
    if outcome.resolved_slots != tuple(expected_resolutions):
        raise ForagerMatchedProtocolError(
            "sealed protocol slot resolutions do not match the canonical selection result"
        )

    slot_index = {item.slot: item.candidate_id for item in expected_resolutions}
    selected_panel_ids = tuple(
        slot_index[slot] for slot in sealed_value.evaluation_panel.selection_slots
    )
    evaluation_ids = (
        selected_panel_ids + sealed_value.evaluation_panel.fixed_descriptive_candidate_ids
    )
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ForagerMatchedProtocolError(
            "resolved evaluation panel contains a candidate more than once"
        )
    hypotheses = (sealed_value.primary_hypothesis,) + sealed_value.secondary_hypotheses
    resolved_hypotheses = tuple(
        ResolvedHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            intervention_candidate_id=slot_index[hypothesis.intervention_slot],
            comparator_candidate_id=slot_index[hypothesis.comparator_slot],
            method=hypothesis.method,
            alternative=hypothesis.alternative,
            difference_order=hypothesis.difference_order,
        )
        for hypothesis in hypotheses
    )
    primary = resolved_hypotheses[0]
    candidate_index = sealed_value.candidate_index
    if candidate_index[primary.intervention_candidate_id].stratum != "alberta_learning":
        raise ForagerMatchedProtocolError(
            "resolved primary intervention is not an Alberta candidate"
        )
    if candidate_index[primary.comparator_candidate_id].stratum != "external_learning":
        raise ForagerMatchedProtocolError("resolved primary comparator is not external learning")
    return SealedProtocolValidation(
        open_protocol_sha256=open_digest,
        selection_result_sha256=result_digest,
        resolved_slots=tuple(expected_resolutions),
        evaluation_candidate_ids=evaluation_ids,
        primary_intervention_candidate_id=primary.intervention_candidate_id,
        primary_comparator_candidate_id=primary.comparator_candidate_id,
        resolved_hypotheses=resolved_hypotheses,
    )


def normalize_forager_matched_protocol(value: Any) -> dict[str, Any]:
    """Validate a value and return its complete normalized JSON object."""
    return parse_forager_matched_protocol(value).to_dict()


def canonical_json_bytes(value: ForagerMatchedProtocol | Mapping[str, Any]) -> bytes:
    """Return deterministic bytes after replaying protocol validation."""
    if type(value) is ForagerMatchedProtocol:
        payload = parse_forager_matched_protocol(value.to_dict()).to_dict()
    else:
        payload = parse_forager_matched_protocol(value).to_dict()
    return _canonical_plain_json_bytes(payload)


def canonical_json_sha256(value: ForagerMatchedProtocol | Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_forager_matched_protocol(path: str | Path) -> ForagerMatchedProtocol:
    """Open without following symlinks and require exact canonical protocol bytes."""
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ForagerMatchedProtocolError(
            f"could not open protocol as a regular non-symlink file: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ForagerMatchedProtocolError(
                "protocol path must resolve to a regular non-symlink file"
            )
        if before.st_size > _MAX_MANIFEST_BYTES:
            raise ForagerMatchedProtocolError("protocol exceeds the file-size limit")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedProtocolError(f"could not read protocol: {exc}") from exc
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(raw) != before.st_size:
        raise ForagerMatchedProtocolError("protocol changed while it was being read")
    protocol = parse_forager_matched_protocol(decode_strict_json(raw))
    if raw != protocol.canonical_bytes:
        raise ForagerMatchedProtocolError("protocol file must contain exact canonical JSON bytes")
    return protocol


# Short aliases are useful to callers without weakening the schema-specific API.
parse_matched_protocol = parse_forager_matched_protocol
load_matched_protocol = load_forager_matched_protocol
normalize_matched_protocol = normalize_forager_matched_protocol


__all__ = [
    "FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION",
    "FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION",
    "MATCHED_PROTOCOL_SCHEMA_VERSION",
    "AllowedTransform",
    "AgentRNGContract",
    "CPUSandboxContract",
    "CandidateRuntimeBinding",
    "ConfigurationBinding",
    "DescriptiveContext",
    "EnvironmentRNGContract",
    "EvaluationPanel",
    "ExecutionSemantics",
    "ForagerMatchedProtocol",
    "ForagerMatchedProtocolError",
    "ForagerMatchedSelectionResult",
    "MatchedAnalysisPlan",
    "MatchedCandidate",
    "MatchedHypothesis",
    "MatchedRuntime",
    "MatchedTask",
    "MultiplicityPolicy",
    "ObservationAccess",
    "PairingEligibility",
    "PrimaryBootstrapAnalysis",
    "RankedSelectionGroup",
    "ResourceAccounting",
    "SeedContract",
    "SelectionGroup",
    "SelectionOutcome",
    "SelectionPlan",
    "SelectionSlot",
    "SecondarySignFlipAnalysis",
    "SealedProtocolValidation",
    "ResolvedHypothesis",
    "ResolvedSelectionSlot",
    "SourceBinding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_selection_result_bytes",
    "canonical_selection_result_sha256",
    "candidate_capability_descriptor_sha256",
    "decode_strict_json",
    "load_forager_matched_protocol",
    "load_matched_protocol",
    "normalize_forager_matched_protocol",
    "normalize_matched_protocol",
    "parse_forager_matched_protocol",
    "parse_forager_matched_selection_result",
    "parse_matched_protocol",
    "seal_forager_matched_protocol",
    "validate_sealed_protocol_transition",
]
