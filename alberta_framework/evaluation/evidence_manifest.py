"""Versioned, machine-readable claim manifest for continual-learning evidence.

This module does not reinterpret scientific results and does not trust file
presence.  It invokes each versioned artifact's strict loader and validator,
checks the artifact against the repository's frozen claim contract, and records
one of five statuses:

``accepted``
    Scientific evidence is valid, contract-bound, and passed its frozen gate.
``valid-rejection``
    The artifact is internally valid but failed at least one frozen gate.
``not-run``
    The expected artifact is absent.
``invalid``
    Parsing, integrity, schema, provenance, or contract validation failed.
``verified-nonpromoting``
    A valid unit/smoke result; its type makes scientific promotion impossible.

The manifest is operational metadata, not a new scientific artifact or
signature.  Its SHA-256 fields identify the exact artifact and source bytes
that were inspected.  Unit and smoke records are structurally barred from
scientific promotion, even if their local validators pass.
An overall accepted status means only that every *listed narrow gate* passed;
it is explicitly not an Alberta Plan completion certificate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FunctionType
from typing import Literal, NoReturn, Protocol, cast

from alberta_framework._bounded_containers import require_json_text_nesting
from alberta_framework.evaluation.continual_ia import (
    ContinualIAConfig,
    IAAcceptanceThresholds,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    SCHEMA_VERSION as IA_SCHEMA_VERSION,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    _config_payload as _ia_configuration_payload,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    _protocol_payload as _ia_protocol_payload,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    _threshold_payload as _ia_threshold_payload,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    _validate_ia_historical_replay_projection,
    load_ia_evidence_artifact,
    validate_ia_evidence_artifact,
)
from alberta_framework.evaluation.continual_multiagent import (
    AcceptanceThresholds,
    ContinualMultiAgentConfig,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    SCHEMA_VERSION as MULTIAGENT_SCHEMA_VERSION,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    _config_payload as _multiagent_configuration_payload,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    _protocol_payload as _multiagent_protocol_payload,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    _threshold_payload as _multiagent_threshold_payload,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    load_evidence_artifact as load_multiagent_artifact,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    validate_evidence_artifact as validate_multiagent_artifact,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    SCHEMA_VERSION as FTL_SCHEMA_VERSION,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    _configuration_payload as _ftl_configuration_payload,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    _protocol_payload as _ftl_protocol_payload,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    _threshold_payload as _ftl_threshold_payload,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    load_ftl_decision_artifact,
    validate_ftl_decision_artifact,
)
from alberta_framework.evaluation.ftl_decision_fidelity import (
    DecisionFidelityConfig,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    SCHEMA_VERSION as RECURRING_FEATURE_SCHEMA_VERSION,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    _configuration_payload as _recurring_configuration_payload,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    _criteria_payload as _recurring_criteria_payload,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    _protocol_payload as _recurring_protocol_payload,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    load_recurring_feature_artifact,
    validate_recurring_feature_artifact,
)
from alberta_framework.evaluation.scale_robust_feature import (
    frozen_configuration_payload,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    SCHEMA_VERSION as SCALE_ROBUST_SCHEMA_VERSION,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _protocol_payload as _scale_robust_protocol_payload,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _threshold_payload as _scale_robust_threshold_payload,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    load_evidence_artifact as load_scale_robust_artifact,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    validate_evidence_artifact as validate_scale_robust_artifact,
)
from alberta_framework.recurring_feature_gate import (
    RecurringFeatureGateCriteria,
    RecurringFeatureProtocol,
)

_STRICT_IA_ARTIFACT_VALIDATOR = validate_ia_evidence_artifact

MANIFEST_SCHEMA_VERSION = "alberta.evidence_claim_manifest.v1"
_MAX_JSON_NESTING_DEPTH = 64
CLAIM_CONTRACT_VERSION = "alberta.evidence_claim_contract.v1"
DIRTY_STATE_POLICY_VERSION = "alberta.registered_source_dirty_policy.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

_FTL_CLAIM_ID = "ftl_world_model_decision_fidelity"
_FTL_ARTIFACT_PATH = Path("outputs/ftl_decision/evidence.v1.json")
_FTL_SOURCE_PATHS = (
    Path("alberta_framework/core/ftl_world_model.py"),
    Path("alberta_framework/evaluation/ftl_decision_fidelity.py"),
    Path("alberta_framework/evaluation/ftl_decision_artifact.py"),
    Path("alberta_framework/evaluation/ftl_decision_cli.py"),
)
_FTL_ARTIFACT_BUILDER_PATH = Path(
    "alberta_framework/evaluation/ftl_decision_artifact.py"
)
_FTL_INVARIANT_SOURCE_PATHS = tuple(
    path for path in _FTL_SOURCE_PATHS if path != _FTL_ARTIFACT_BUILDER_PATH
)
_FTL_REPLAY_PATH = Path(
    "outputs/ftl_decision/reproductions/nonpromoting_consumed_seed_replay.json"
)
_FTL_ATTESTATION_PATH = Path("outputs/ftl_decision/reproductions/attestation.json")
_FTL_ATTESTATION_SCHEMA = "alberta.ftl_consumed_seed_source_compatibility.v1"
_FTL_HISTORICAL_CHAIN_SCHEMA = "alberta.historical_ftl_acceptance_chain.v1"
_FTL_SOURCE_DRIFT_VALIDATOR_ERROR = (
    "scientific_payload.source_provenance does not match the current pinned "
    "source hashes"
)
_FTL_SOURCE_DRIFT_CONTRACT_ERROR = (
    "artifact source hashes do not match the current registered sources"
)
# These anchors identify the immutable original v1 artifact.  Unlike the IA
# chain, no archive containing the exact historical artifact-builder source is
# available.  The compatibility chain below therefore does not claim archived
# source recoverability: it requires the original bytes, reconstructs every
# scientific statistic and acceptance decision after projecting only the
# builder hash, and requires an exact consumed-seed replay under current
# sources.
_FTL_HISTORICAL_ARTIFACT_BYTES_SHA256 = (
    "9f45a307e23c250fff9870c081daa22857894fcd0d7bed03efd2b270be577e4d"
)
_FTL_HISTORICAL_SCIENTIFIC_SHA256 = (
    "4c0c50273358c2c409a63d19ae52831ef1f6bad4645cfdf252c23b62425e0148"
)
_FTL_REPLAY_COMMAND = (
    "python",
    "-m",
    "alberta_framework.evaluation.ftl_decision_cli",
    "--output",
    _FTL_REPLAY_PATH.as_posix(),
)
_FTL_COMPARISON_EXCLUDED_PATHS = (
    "$.operational_metadata",
    (
        "$.scientific_payload.source_provenance.source_sha256."
        "alberta_framework/evaluation/ftl_decision_artifact.py"
    ),
    "$.scientific_digest.sha256",
)
_FTL_COMPARISON_SCOPE = (
    "the complete artifact after removing operational metadata, the changed "
    "artifact-builder source hash, and only its scientific-digest derivative; "
    "each original scientific digest and strict validator result is checked "
    "independently"
)
_FTL_CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan"
_FTL_ATTESTATION_INTERPRETATION = {
    "historical_result": (
        "the immutable original v1 artifact remains the promoting held-out "
        "result; its acceptance is reconstructed from primitive rows rather "
        "than trusted from the stored acceptance flag"
    ),
    "compatibility_result": (
        "current pinned sources reproduce every deterministic v1 scientific "
        "field exactly on the already-consumed v1 seed schedule after "
        "normalizing only the changed artifact-builder source hash"
    ),
    "promotion_result": (
        "none from the replay; it is a source-compatibility check and cannot "
        "become fresh held-out evidence"
    ),
}
_FTL_ATTESTATION_LIMITATIONS = (
    "Seeds 30-59 were consumed by the original promoted v1 experiment.",
    "The replay cannot support retuning, a stronger claim, or promotion of a "
    "changed protocol.",
    "No archive containing the exact historical artifact-builder source is "
    "available; the chain establishes deterministic scientific compatibility, "
    "not full historical source recoverability.",
    "The world-model, evaluator, and CLI source hashes must remain exactly "
    "equal to the original artifact; drift in any of them invalidates the chain.",
    "SHA-256 identifiers establish integrity and reproducibility context, not "
    "authorship or authenticity.",
    "A changed FTL protocol or new current-code promotion requires "
    "preregistration and untouched seeds.",
)

_IA_CLAIM_ID = "continual_intelligence_amplification"
_IA_ARTIFACT_PATH = Path("outputs/continual_ia/evidence.json")
_IA_SOURCE_PATHS = (
    Path("alberta_framework/evaluation/continual_ia.py"),
    Path("alberta_framework/evaluation/continual_ia_artifact.py"),
    Path("alberta_framework/evaluation/continual_ia_cli.py"),
    Path("alberta_framework/core/intelligence_amplification.py"),
    Path("alberta_framework/core/average_reward.py"),
    Path("alberta_framework/core/oak.py"),
    Path("alberta_framework/core/options.py"),
    Path("alberta_framework/streams/closed_loop.py"),
)
_IA_SNAPSHOT_MANIFEST_PATH = Path("outputs/continual_ia/source_snapshot_v1/manifest.json")
_IA_ATTESTATION_PATH = Path("outputs/continual_ia/reproductions/attestation.json")
_IA_REPLAY_PATH = Path("outputs/continual_ia/reproductions/nonpromoting_consumed_seed_replay.json")
_IA_SNAPSHOT_SCHEMA = "alberta.historical_source_snapshot.v1"
_IA_ATTESTATION_SCHEMA = "alberta.ia_consumed_seed_source_compatibility.v1"
_IA_HISTORICAL_CHAIN_SCHEMA = "alberta.historical_ia_rejection_chain.v1"
_IA_SOURCE_DRIFT_VALIDATOR_ERROR = (
    "content.source_provenance does not match the current pinned sources"
)
_IA_SOURCE_DRIFT_CONTRACT_ERROR = (
    "artifact source hashes do not match the current registered sources"
)
_IA_REDUCED_EXCLUDED_PATHS = (
    "$.content.source_provenance",
    "$.content_digest",
    "$.operational_diagnostics",
)
_IA_REDUCED_INCLUDED_SCOPE = (
    "the complete remaining artifact, including schema, frozen protocol, "
    "configuration, thresholds, all primitive per-seed traces, aggregates, "
    "and acceptance decisions"
)
_IA_CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan"
_IA_REPLAY_COMMAND = (
    "python",
    "-m",
    "alberta_framework.evaluation.continual_ia_cli",
    "--output",
    _IA_REPLAY_PATH.as_posix(),
)
_IA_ARCHIVE_ROLES = {
    "alberta_framework-0.26.0-py3-none-any.whl": (
        "exact recoverable package snapshot containing every artifact-pinned source path"
    ),
    "alberta_framework-0.26.0.tar.gz": "matching source distribution",
}
_IA_SNAPSHOT_LIMITATIONS = (
    "The v1 artifact pinned eight selected files, not every byte imported by Python or JAX.",
    "SHA-256 identifiers establish integrity and reproducibility context, not "
    "authorship or authenticity.",
    "The live worktree has since changed; this snapshot preserves the "
    "historical rejected result and does not validate a claim about current code.",
    "A replay on consumed v1 seeds can test compatibility but cannot become "
    "fresh held-out evidence.",
    "A changed IA protocol or promoted current-code claim requires a "
    "preregistered v2 and untouched seeds.",
)
_IA_ATTESTATION_INTERPRETATION = {
    "historical_result": (
        "the original v1 artifact remains a frozen valid rejection under its "
        "archived source snapshot"
    ),
    "compatibility_result": (
        "current pinned sources reproduce every v1 scientific field exactly "
        "on the consumed v1 schedule"
    ),
    "promotion_result": (
        "none; this replay is a source-compatibility check, not new held-out evidence"
    ),
}
_IA_ATTESTATION_LIMITATIONS = (
    "Seeds 30-59 were already consumed by the original v1 experiment.",
    "The replay cannot support retuning, a stronger claim, or promotion of a changed protocol.",
    "The frozen 0.10 action-changing intervention-rate threshold remains "
    "failed and was not lowered.",
    "A changed IA protocol or new current-code promotion requires "
    "preregistration and untouched seeds.",
)

EvidenceClass = Literal["unit", "smoke", "scientific"]
EvidenceLevel = Literal["L0", "L1", "L2", "L3"]
_EVIDENCE_CLASSES: frozenset[str] = frozenset({"unit", "smoke", "scientific"})
_EVIDENCE_LEVELS: frozenset[str] = frozenset({"L0", "L1", "L2", "L3"})
_PATH_TYPE = type(Path())
_MAX_SPEC_TEXT_BYTES = 64 * 1024
_MAX_SPEC_SEQUENCE_ITEMS = 256
_MAX_SPEC_MAPPING_ITEMS = 256
_MAX_SPEC_NESTING_DEPTH = 64
_MAX_SPEC_TOTAL_VALUES = 4096

DIRTY_STATE_POLICY: Mapping[str, object] = {
    "policy_version": DIRTY_STATE_POLICY_VERSION,
    "clean_worktree_required": False,
    "registered_source_hash_match_required": True,
    "unregistered_changes": (
        "recorded as operational provenance; neither sufficient for promotion "
        "nor independently disqualifying"
    ),
    "registered_source_changes": (
        "invalidate persisted evidence until the frozen protocol is rerun and "
        "a new artifact passes its strict validator"
    ),
}


class ValidationResult(Protocol):
    """Common result surface implemented by all promoted validators."""

    @property
    def valid(self) -> bool:
        """Whether parsing, schema, provenance, and reconstruction passed."""

    @property
    def accepted(self) -> bool:
        """Whether the valid artifact also passed its frozen scientific gate."""

    @property
    def errors(self) -> tuple[str, ...]:
        """Fail-closed validation errors."""


def _require_exact_text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > _MAX_SPEC_TEXT_BYTES or len(value.encode("utf-8")) > _MAX_SPEC_TEXT_BYTES:
        raise ValueError(f"{name} exceeds {_MAX_SPEC_TEXT_BYTES} UTF-8 bytes")
    return value


def _require_exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_exact_path(name: str, value: object) -> Path:
    if type(value) is not _PATH_TYPE:
        raise ValueError(f"{name} must be an exact platform Path")
    path = value
    encoded = path.as_posix().encode("utf-8")
    if (
        not path.parts
        or path.is_absolute()
        or any(part == ".." for part in path.parts)
        or len(encoded) > _MAX_SPEC_TEXT_BYTES
    ):
        raise ValueError(f"{name} must be a non-empty repository-relative Path")
    return path


def _require_exact_str_tuple(name: str, value: object) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or not value
        or len(value) > _MAX_SPEC_SEQUENCE_ITEMS
    ):
        raise ValueError(f"{name} must be a non-empty tuple of non-empty strings")
    for item in value:
        _require_exact_text(name, item)
    return value


def _require_path_tuple(name: str, value: object) -> tuple[Path, ...]:
    if (
        type(value) is not tuple
        or not value
        or len(value) > _MAX_SPEC_SEQUENCE_ITEMS
    ):
        raise ValueError(f"{name} must be a non-empty tuple of Path values")
    return tuple(
        _require_exact_path(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )


def _require_exact_json_value(
    name: str,
    value: object,
    *,
    ancestors: frozenset[int] = frozenset(),
    depth: int = 0,
    remaining: int = _MAX_SPEC_TOTAL_VALUES,
) -> int:
    if depth > _MAX_SPEC_NESTING_DEPTH:
        raise ValueError(f"{name} exceeds the maximum nesting depth")
    if remaining < 1:
        raise ValueError(f"{name} exceeds the aggregate value limit")
    remaining -= 1
    value_type = type(value)
    if value_type is dict:
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{name} must not contain a cycle")
        mapping = cast(dict[object, object], value)
        if len(mapping) > _MAX_SPEC_MAPPING_ITEMS:
            raise ValueError(f"{name} exceeds the mapping item limit")
        next_ancestors = ancestors | {identity}
        for key, item in mapping.items():
            _require_exact_text(f"{name} key", key)
            remaining = _require_exact_json_value(
                f"{name}.{key}",
                item,
                ancestors=next_ancestors,
                depth=depth + 1,
                remaining=remaining,
            )
        return remaining
    if value_type is list:
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{name} must not contain a cycle")
        items = cast(list[object], value)
        if len(items) > _MAX_SPEC_SEQUENCE_ITEMS:
            raise ValueError(f"{name} exceeds the sequence item limit")
        next_ancestors = ancestors | {identity}
        for index, item in enumerate(items):
            remaining = _require_exact_json_value(
                f"{name}[{index}]",
                item,
                ancestors=next_ancestors,
                depth=depth + 1,
                remaining=remaining,
            )
        return remaining
    if value_type is str:
        if len(cast(str, value).encode("utf-8")) > _MAX_SPEC_TEXT_BYTES:
            raise ValueError(f"{name} exceeds {_MAX_SPEC_TEXT_BYTES} UTF-8 bytes")
        return remaining
    if value_type is int:
        if cast(int, value).bit_length() > _MAX_SPEC_TEXT_BYTES * 4:
            raise ValueError(f"{name} exceeds the integer size limit")
        return remaining
    if value_type is bool or value is None:
        return remaining
    if value_type is float and math.isfinite(cast(float, value)):
        return remaining
    raise ValueError(f"{name} must contain only finite exact JSON values")


def _require_identity_mapping(name: str, value: object) -> dict[str, object]:
    if type(value) is not dict or not value:
        raise ValueError(f"{name} must be a non-empty dict with exact string keys")
    _require_exact_json_value(name, value)
    return value


def _require_identity_callable(name: str, value: object) -> Callable[..., object]:
    if type(value) is not FunctionType:
        raise ValueError(f"{name} must be an exact function")
    return value


@dataclass(frozen=True)
class EvidenceSpec:
    """One immutable evidence-claim contract."""

    name: str
    claim_scope: str
    evidence_class: EvidenceClass
    evidence_level: EvidenceLevel
    promotes_scientific_claim: bool
    relative_path: Path
    expected_schema: str
    command_argv: tuple[str, ...]
    protocol: Mapping[str, object]
    configuration: Mapping[str, object]
    seeds: Mapping[str, object]
    thresholds: Mapping[str, object]
    limitations: tuple[str, ...]
    source_paths: tuple[Path, ...]
    required_environment_fields: tuple[str, ...]
    loader: Callable[[Path], Mapping[str, object]]
    validator: Callable[[Mapping[str, object]], ValidationResult]

    def __post_init__(self) -> None:
        _require_exact_text("name", self.name)
        _require_exact_text("claim_scope", self.claim_scope)
        if type(self.evidence_class) is not str or self.evidence_class not in _EVIDENCE_CLASSES:
            raise ValueError("evidence_class must be a known evidence class")
        if type(self.evidence_level) is not str or self.evidence_level not in _EVIDENCE_LEVELS:
            raise ValueError("evidence_level must be a known evidence level")
        _require_exact_bool("promotes_scientific_claim", self.promotes_scientific_claim)
        relative_path = _require_exact_path("relative_path", self.relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("relative_path must remain under the repository root")
        _require_exact_text("expected_schema", self.expected_schema)
        _require_exact_str_tuple("command_argv", self.command_argv)
        _require_identity_mapping("protocol", self.protocol)
        _require_identity_mapping("configuration", self.configuration)
        _require_identity_mapping("seeds", self.seeds)
        _require_identity_mapping("thresholds", self.thresholds)
        _require_exact_str_tuple("limitations", self.limitations)
        source_paths = _require_path_tuple("source_paths", self.source_paths)
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("source_paths must be unique")
        if any(path.is_absolute() or ".." in path.parts for path in source_paths):
            raise ValueError("source_paths must remain under the repository root")
        _require_exact_str_tuple("required_environment_fields", self.required_environment_fields)
        _require_identity_callable("loader", self.loader)
        _require_identity_callable("validator", self.validator)


@dataclass(frozen=True)
class _HistoricalIAChainValidation:
    """Result of the nonpromoting historical-rejection evidence chain."""

    valid: bool
    errors: tuple[str, ...]
    artifacts: tuple[dict[str, object], ...]
    record: dict[str, object]

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("valid must be a boolean")
        if type(self.errors) is not tuple or not all(isinstance(e, str) for e in self.errors):
            raise ValueError("errors must be a tuple of strings")
        if type(self.artifacts) is not tuple:
            raise ValueError("artifacts must be a tuple")
        if type(self.record) is not dict:
            raise ValueError("record must be a dict")


@dataclass(frozen=True)
class _HistoricalFTLChainValidation:
    """Result of the historical-acceptance/current-compatibility chain."""

    valid: bool
    accepted: bool
    errors: tuple[str, ...]
    artifacts: tuple[dict[str, object], ...]
    record: dict[str, object]

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("valid must be a boolean")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        if type(self.errors) is not tuple or not all(isinstance(e, str) for e in self.errors):
            raise ValueError("errors must be a tuple of strings")
        if type(self.artifacts) is not tuple:
            raise ValueError("artifacts must be a tuple")
        if type(self.record) is not dict:
            raise ValueError("record must be a dict")


def _required_mapping(
    value: Mapping[str, object],
    key: str,
    *,
    owner: str,
) -> Mapping[str, object]:
    candidate = value.get(key)
    if not isinstance(candidate, Mapping) or not candidate:
        raise RuntimeError(f"{owner}.{key} must be a non-empty mapping")
    return candidate


def _required_string_tuple(
    value: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    owner: str,
) -> tuple[str, ...]:
    for key in keys:
        candidate = value.get(key)
        if (
            isinstance(candidate, list)
            and candidate
            and all(type(item) is str and item for item in candidate)
        ):
            return tuple(candidate)
    joined = " or ".join(f"{owner}.{key}" for key in keys)
    raise RuntimeError(f"{joined} must be a non-empty string array")


def _required_string(
    value: Mapping[str, object],
    key: str,
    *,
    owner: str,
) -> str:
    candidate = value.get(key)
    if type(candidate) is not str or not candidate:
        raise RuntimeError(f"{owner}.{key} must be a non-empty string")
    return candidate


_RECURRING_PROTOCOL = _recurring_protocol_payload()
_SCALE_ROBUST_PROTOCOL = _scale_robust_protocol_payload()
_FTL_PROTOCOL = _ftl_protocol_payload()
_MULTIAGENT_PROTOCOL = _multiagent_protocol_payload()
_IA_PROTOCOL = _ia_protocol_payload()

EVIDENCE_SPECS: tuple[EvidenceSpec, ...] = (
    EvidenceSpec(
        name="recurring_pair_features",
        claim_scope=_required_string(
            _RECURRING_PROTOCOL,
            "supported_claim",
            owner="recurring_pair_features.protocol",
        ),
        evidence_class="scientific",
        evidence_level="L2",
        promotes_scientific_claim=True,
        relative_path=Path("outputs/recurring_feature/evidence.v1.json"),
        expected_schema=RECURRING_FEATURE_SCHEMA_VERSION,
        command_argv=(
            "python",
            "-m",
            "alberta_framework.evaluation.recurring_feature_cli",
            "--output",
            "outputs/recurring_feature/evidence.v1.json",
        ),
        protocol=_RECURRING_PROTOCOL,
        configuration=_recurring_configuration_payload(RecurringFeatureProtocol()),
        seeds=_required_mapping(
            _RECURRING_PROTOCOL,
            "seed_roles",
            owner="recurring_pair_features.protocol",
        ),
        thresholds=_recurring_criteria_payload(RecurringFeatureGateCriteria()),
        limitations=_required_string_tuple(
            _RECURRING_PROTOCOL,
            ("excluded_claims", "limitations"),
            owner="recurring_pair_features.protocol",
        ),
        source_paths=(
            Path("alberta_framework/recurring_feature_gate.py"),
            Path("alberta_framework/core/interaction_features.py"),
            Path("alberta_framework/evaluation/recurring_feature_artifact.py"),
            Path("alberta_framework/evaluation/recurring_feature_cli.py"),
        ),
        required_environment_fields=("python", "platform", "packages"),
        loader=load_recurring_feature_artifact,
        validator=validate_recurring_feature_artifact,
    ),
    EvidenceSpec(
        name="scale_robust_pair_features",
        claim_scope=_required_string(
            _SCALE_ROBUST_PROTOCOL,
            "supported_claim",
            owner="scale_robust_pair_features.protocol",
        ),
        evidence_class="scientific",
        evidence_level="L2",
        promotes_scientific_claim=True,
        relative_path=Path("outputs/scale_robust_feature/evidence.v2.json"),
        expected_schema=SCALE_ROBUST_SCHEMA_VERSION,
        command_argv=(
            "python",
            "-m",
            "alberta_framework.evaluation.scale_robust_feature_cli",
            "--output",
            "outputs/scale_robust_feature/evidence.v2.json",
        ),
        protocol=_SCALE_ROBUST_PROTOCOL,
        configuration=frozen_configuration_payload(),
        seeds=_required_mapping(
            _SCALE_ROBUST_PROTOCOL,
            "seed_roles",
            owner="scale_robust_pair_features.protocol",
        ),
        thresholds=_scale_robust_threshold_payload(),
        limitations=_required_string_tuple(
            _SCALE_ROBUST_PROTOCOL,
            ("excluded_claims", "limitations"),
            owner="scale_robust_pair_features.protocol",
        ),
        source_paths=(
            Path("alberta_framework/core/interaction_features.py"),
            Path("alberta_framework/streams/gauntlet.py"),
            Path("alberta_framework/evaluation/scale_robust_feature.py"),
            Path("alberta_framework/evaluation/scale_robust_feature_artifact.py"),
            Path("alberta_framework/evaluation/scale_robust_feature_cli.py"),
        ),
        required_environment_fields=("python", "platform", "packages"),
        loader=load_scale_robust_artifact,
        validator=validate_scale_robust_artifact,
    ),
    EvidenceSpec(
        name="ftl_world_model_decision_fidelity",
        claim_scope=_required_string(
            _FTL_PROTOCOL,
            "supported_claim",
            owner="ftl_world_model_decision_fidelity.protocol",
        ),
        evidence_class="scientific",
        evidence_level="L2",
        promotes_scientific_claim=True,
        relative_path=Path("outputs/ftl_decision/evidence.v1.json"),
        expected_schema=FTL_SCHEMA_VERSION,
        command_argv=(
            "python",
            "-m",
            "alberta_framework.evaluation.ftl_decision_cli",
            "--output",
            "outputs/ftl_decision/evidence.v1.json",
        ),
        protocol=_FTL_PROTOCOL,
        configuration=_ftl_configuration_payload(DecisionFidelityConfig()),
        seeds=_required_mapping(
            _FTL_PROTOCOL,
            "seed_roles",
            owner="ftl_world_model_decision_fidelity.protocol",
        ),
        thresholds=_ftl_threshold_payload(),
        limitations=_required_string_tuple(
            _FTL_PROTOCOL,
            ("excluded_claims", "limitations"),
            owner="ftl_world_model_decision_fidelity.protocol",
        ),
        source_paths=_FTL_SOURCE_PATHS,
        required_environment_fields=("python", "platform", "packages"),
        loader=load_ftl_decision_artifact,
        validator=validate_ftl_decision_artifact,
    ),
    EvidenceSpec(
        name="recurring_multiagent_coadaptation",
        claim_scope=_required_string(
            _MULTIAGENT_PROTOCOL,
            "supported_claim",
            owner="recurring_multiagent_coadaptation.protocol",
        ),
        evidence_class="scientific",
        evidence_level="L2",
        promotes_scientific_claim=True,
        relative_path=Path("outputs/continual_multiagent/evidence.json"),
        expected_schema=MULTIAGENT_SCHEMA_VERSION,
        command_argv=(
            "python",
            "-m",
            "alberta_framework.evaluation.continual_multiagent_cli",
            "--output",
            "outputs/continual_multiagent/evidence.json",
        ),
        protocol=_MULTIAGENT_PROTOCOL,
        configuration=_multiagent_configuration_payload(ContinualMultiAgentConfig()),
        seeds=_required_mapping(
            _MULTIAGENT_PROTOCOL,
            "seed_roles",
            owner="recurring_multiagent_coadaptation.protocol",
        ),
        thresholds=_multiagent_threshold_payload(AcceptanceThresholds()),
        limitations=_required_string_tuple(
            _MULTIAGENT_PROTOCOL,
            ("excluded_claims", "limitations"),
            owner="recurring_multiagent_coadaptation.protocol",
        ),
        source_paths=(
            Path("alberta_framework/evaluation/continual_multiagent.py"),
            Path("alberta_framework/evaluation/continual_multiagent_artifact.py"),
            Path("alberta_framework/evaluation/continual_multiagent_cli.py"),
            Path("alberta_framework/streams/recurring_multiagent.py"),
            Path("alberta_framework/utils/metrics.py"),
        ),
        required_environment_fields=("python", "platform", "packages"),
        loader=load_multiagent_artifact,
        validator=validate_multiagent_artifact,
    ),
    EvidenceSpec(
        name="continual_intelligence_amplification",
        claim_scope=_required_string(
            _IA_PROTOCOL,
            "supported_claim",
            owner="continual_intelligence_amplification.protocol",
        ),
        evidence_class="scientific",
        evidence_level="L2",
        promotes_scientific_claim=True,
        relative_path=Path("outputs/continual_ia/evidence.json"),
        expected_schema=IA_SCHEMA_VERSION,
        command_argv=(
            "python",
            "-m",
            "alberta_framework.evaluation.continual_ia_cli",
            "--output",
            "outputs/continual_ia/evidence.json",
        ),
        protocol=_IA_PROTOCOL,
        configuration=_ia_configuration_payload(ContinualIAConfig()),
        seeds=_required_mapping(
            _IA_PROTOCOL,
            "seed_roles",
            owner="continual_intelligence_amplification.protocol",
        ),
        thresholds=_ia_threshold_payload(IAAcceptanceThresholds()),
        limitations=_required_string_tuple(
            _IA_PROTOCOL,
            ("excluded_claims", "limitations"),
            owner="continual_intelligence_amplification.protocol",
        ),
        source_paths=_IA_SOURCE_PATHS,
        required_environment_fields=("python", "platform", "packages"),
        loader=load_ia_evidence_artifact,
        validator=validate_ia_evidence_artifact,
    ),
)


def _validated_evidence_specs(value: object) -> tuple[EvidenceSpec, ...]:
    """Validate the exact registry roster before any claim is inspected."""

    if type(value) is not tuple or not value:
        raise ValueError("EVIDENCE_SPECS must be a non-empty exact tuple")
    if not all(type(spec) is EvidenceSpec for spec in value):
        raise ValueError("EVIDENCE_SPECS must contain only exact EvidenceSpec values")
    specs = cast(tuple[EvidenceSpec, ...], value)
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("EVIDENCE_SPECS claim names must be unique")
    paths = tuple(spec.relative_path for spec in specs)
    if len(set(paths)) != len(paths):
        raise ValueError("EVIDENCE_SPECS artifact paths must be unique")
    return specs


EVIDENCE_SPECS = _validated_evidence_specs(EVIDENCE_SPECS)


def _artifact_digest(artifact: Mapping[str, object]) -> str | None:
    """Extract a validator-bound scientific/content digest when present."""

    for key in ("scientific_digest", "content_digest"):
        record = artifact.get(key)
        if not isinstance(record, Mapping):
            continue
        value = record.get("sha256")
        if (
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            return value
    return None


def _strict_json_clone(value: object) -> object:
    """Return a detached JSON-compatible value or fail on non-finite data."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _canonical_sha256(value: object) -> str:
    """Hash one exact finite JSON value using the IA canonicalization."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _load_strict_json_object(path: Path) -> dict[str, object]:
    """Load a finite UTF-8 JSON object while rejecting duplicate keys."""

    text = path.read_text(encoding="utf-8")
    require_json_text_nesting(
        text,
        max_depth=_MAX_JSON_NESTING_DEPTH,
        name="evidence manifest",
    )
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except RecursionError as exc:
        raise ValueError("evidence manifest exceeds the JSON nesting limit") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return parsed


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validation_environment() -> dict[str, object]:
    """Capture the runtime that performed validation, never protocol results."""

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "alberta-framework": _package_version("alberta-framework"),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "numpy": _package_version("numpy"),
        },
    }


def _git_output(root: Path, arguments: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_snapshot(
    root: Path,
    spec: EvidenceSpec,
) -> tuple[dict[str, object], list[str]]:
    """Hash the exact registered sources and record current VCS provenance."""

    hashes: dict[str, str | None] = {}
    errors: list[str] = []
    for relative_path in spec.source_paths:
        path = root / relative_path
        try:
            hashes[relative_path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            hashes[relative_path.as_posix()] = None
            errors.append(f"unable to hash source {relative_path}: {error}")

    head = _git_output(root, ("rev-parse", "--verify", "HEAD"))
    dirty_output = _git_output(
        root,
        ("status", "--porcelain", "--untracked-files=normal"),
    )
    return (
        {
            "repository_root": str(root),
            "git_head": head,
            "worktree_dirty": None if dirty_output is None else bool(dirty_output),
            "source_sha256": hashes,
            "interpretation": (
                "reproducibility and integrity identifiers only; hashes are "
                "not signatures and do not establish artifact authenticity"
            ),
        },
        errors,
    )


def _scientific_content(
    artifact: Mapping[str, object],
) -> Mapping[str, object] | None:
    for key in ("scientific_payload", "content"):
        candidate = artifact.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _artifact_environment(
    artifact: Mapping[str, object],
) -> Mapping[str, object] | None:
    for operational_key in ("operational_metadata", "operational_diagnostics"):
        operational = artifact.get(operational_key)
        if not isinstance(operational, Mapping):
            continue
        for environment_key in ("runtime", "runtime_provenance", "environment"):
            candidate = operational.get(environment_key)
            if isinstance(candidate, Mapping):
                return candidate
    return None


def _artifact_limitations(
    protocol: Mapping[str, object],
) -> tuple[str, ...] | None:
    for key in ("excluded_claims", "limitations"):
        candidate = protocol.get(key)
        if (
            isinstance(candidate, list)
            and candidate
            and all(type(item) is str and item for item in candidate)
        ):
            return tuple(candidate)
    return None


def _validate_spec(spec: EvidenceSpec) -> None:
    """Reject malformed registries before any artifact can be inspected."""

    if not spec.name or not spec.claim_scope:
        raise ValueError("claim name and scope must be non-empty")
    if spec.evidence_class not in {"unit", "smoke", "scientific"}:
        raise ValueError(f"unsupported evidence class for {spec.name}")
    if spec.evidence_level not in {"L0", "L1", "L2", "L3"}:
        raise ValueError(f"unsupported evidence level for {spec.name}")
    allowed_levels = {
        "unit": {"L0"},
        "smoke": {"L0", "L1"},
        "scientific": {"L2", "L3"},
    }
    if spec.evidence_level not in allowed_levels[spec.evidence_class]:
        raise ValueError(
            f"{spec.name}: {spec.evidence_class} evidence cannot use {spec.evidence_level}"
        )
    if spec.promotes_scientific_claim != (spec.evidence_class == "scientific"):
        raise ValueError(
            f"{spec.name}: only scientific L2/L3 evidence may promote a scientific claim"
        )
    if (
        spec.relative_path.is_absolute()
        or ".." in spec.relative_path.parts
        or not spec.relative_path.parts
    ):
        raise ValueError("evidence specification paths must remain under root")
    if not spec.expected_schema or ".v" not in spec.expected_schema:
        raise ValueError(f"{spec.name}: expected schema must be versioned")
    if not spec.command_argv or any(not argument for argument in spec.command_argv):
        raise ValueError(f"{spec.name}: exact command argv must be non-empty")
    if not spec.protocol or type(spec.protocol.get("protocol_version")) is not str:
        raise ValueError(f"{spec.name}: protocol must carry a protocol_version")
    if not spec.configuration:
        raise ValueError(f"{spec.name}: configuration must be non-empty")
    if not spec.seeds:
        raise ValueError(f"{spec.name}: seeds must be non-empty")
    if not spec.thresholds:
        raise ValueError(f"{spec.name}: thresholds must be non-empty")
    if not spec.limitations or any(not limitation for limitation in spec.limitations):
        raise ValueError(f"{spec.name}: limitations must be non-empty")
    if not spec.source_paths or len(set(spec.source_paths)) != len(spec.source_paths):
        raise ValueError(f"{spec.name}: source paths must be non-empty and unique")
    for source_path in spec.source_paths:
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("source provenance paths must remain under root")
    if (
        not spec.required_environment_fields
        or len(set(spec.required_environment_fields)) != len(spec.required_environment_fields)
        or any(not field for field in spec.required_environment_fields)
    ):
        raise ValueError(f"{spec.name}: required environment fields must be non-empty and unique")


def _contract_errors(
    artifact: Mapping[str, object],
    spec: EvidenceSpec,
    source_snapshot: Mapping[str, object],
) -> tuple[
    list[str],
    Mapping[str, object] | None,
    Mapping[str, object] | None,
    Mapping[str, object] | None,
]:
    """Bind a parsed artifact to every frozen field in its claim contract."""

    errors: list[str] = []
    scientific = _scientific_content(artifact)
    environment = _artifact_environment(artifact)
    artifact_source: Mapping[str, object] | None = None
    if scientific is None:
        errors.append("artifact has no scientific_payload or content object")
        return errors, scientific, environment, artifact_source

    observed_protocol = scientific.get("protocol")
    if observed_protocol != spec.protocol:
        errors.append("artifact protocol does not match the registered frozen protocol")
    if scientific.get("configuration") != spec.configuration:
        errors.append("artifact configuration does not match the registered frozen configuration")
    observed_thresholds = scientific.get("thresholds")
    if observed_thresholds is None:
        observed_thresholds = scientific.get("criteria")
    if observed_thresholds != spec.thresholds:
        errors.append("artifact thresholds do not match the registered frozen thresholds")

    if isinstance(observed_protocol, Mapping):
        if observed_protocol.get("seed_roles") != spec.seeds:
            errors.append("artifact seeds do not match the registered frozen seed roles")
        if _artifact_limitations(observed_protocol) != spec.limitations:
            errors.append("artifact limitations do not match the registered claim limitations")

    candidate_source = scientific.get("source_provenance")
    if isinstance(candidate_source, Mapping):
        artifact_source = candidate_source
        if candidate_source.get("source_sha256") != source_snapshot.get("source_sha256"):
            errors.append("artifact source hashes do not match the current registered sources")
    else:
        errors.append("artifact source_provenance must be an object")

    if spec.promotes_scientific_claim:
        if environment is None:
            errors.append("scientific artifact must carry runtime environment provenance")
        else:
            missing_environment = [
                field for field in spec.required_environment_fields if field not in environment
            ]
            if missing_environment:
                errors.append(
                    "artifact runtime environment is missing required fields: "
                    + ", ".join(missing_environment)
                )
    return errors, scientific, environment, artifact_source


def _chain_mapping(
    value: object,
    *,
    location: str,
    expected_keys: set[str],
    errors: list[str],
) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return None
    if set(value) != expected_keys:
        errors.append(f"{location} keys do not match the frozen chain schema")
    return value


def _ftl_acceptance_passed(
    artifact: Mapping[str, object],
    *,
    location: str,
    errors: list[str],
) -> bool | None:
    scientific = artifact.get("scientific_payload")
    if not isinstance(scientific, Mapping):
        errors.append(f"{location}.scientific_payload must be an object")
        return None
    acceptance = scientific.get("acceptance")
    if not isinstance(acceptance, Mapping):
        errors.append(f"{location}.scientific_payload.acceptance must be an object")
        return None
    passed = acceptance.get("passed")
    if not isinstance(passed, bool):
        errors.append(
            f"{location}.scientific_payload.acceptance.passed must be boolean"
        )
        return None
    return passed


def _ftl_scientific_digest(
    artifact: Mapping[str, object],
    *,
    location: str,
    errors: list[str],
) -> str | None:
    scientific = artifact.get("scientific_payload")
    digest = artifact.get("scientific_digest")
    if not isinstance(scientific, Mapping):
        errors.append(f"{location}.scientific_payload must be an object")
        return None
    if not isinstance(digest, Mapping):
        errors.append(f"{location}.scientific_digest must be an object")
        return None
    recorded = digest.get("sha256")
    if not _is_sha256(recorded):
        errors.append(
            f"{location}.scientific_digest.sha256 must be lowercase SHA-256"
        )
        return None
    computed = _canonical_sha256(scientific)
    if recorded != computed:
        errors.append(
            f"{location} scientific digest mismatch: recorded {recorded}, "
            f"computed {computed}"
        )
    return computed


def _reduced_ftl_artifact(
    artifact: Mapping[str, object],
    *,
    location: str,
    errors: list[str],
) -> dict[str, object] | None:
    """Remove exactly the attested operational and builder-hash-derived paths."""

    reduced = _strict_json_clone(artifact)
    if not isinstance(reduced, dict):
        errors.append(f"{location} must clone to an object")
        return None
    if "operational_metadata" not in reduced:
        errors.append(f"{location}.operational_metadata is required before reduction")
        return None
    del reduced["operational_metadata"]

    scientific = reduced.get("scientific_payload")
    if not isinstance(scientific, dict):
        errors.append(f"{location}.scientific_payload must be an object before reduction")
        return None
    source = scientific.get("source_provenance")
    if not isinstance(source, dict):
        errors.append(
            f"{location}.scientific_payload.source_provenance must be an "
            "object before reduction"
        )
        return None
    hashes = source.get("source_sha256")
    if not isinstance(hashes, dict):
        errors.append(
            f"{location}.scientific_payload.source_provenance.source_sha256 "
            "must be an object before reduction"
        )
        return None
    builder_key = _FTL_ARTIFACT_BUILDER_PATH.as_posix()
    if builder_key not in hashes:
        errors.append(
            f"{location}.scientific_payload.source_provenance.source_sha256."
            f"{builder_key} is required before reduction"
        )
        return None
    del hashes[builder_key]

    digest = reduced.get("scientific_digest")
    if not isinstance(digest, dict):
        errors.append(f"{location}.scientific_digest must be an object before reduction")
        return None
    if "sha256" not in digest:
        errors.append(
            f"{location}.scientific_digest.sha256 is required before reduction"
        )
        return None
    del digest["sha256"]
    return reduced


def _ftl_supporting_artifact_record(
    root: Path,
    relative_path: Path,
    *,
    role: str,
    parsed: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path = root / relative_path
    try:
        raw = path.read_bytes()
    except OSError:
        raw = None
    return {
        "role": role,
        "path": relative_path.as_posix(),
        "present": raw is not None,
        "bytes_sha256": (
            hashlib.sha256(raw).hexdigest() if raw is not None else None
        ),
        "scientific_content_sha256": (
            _artifact_digest(parsed) if parsed is not None else None
        ),
        "scientific_promotion_allowed": False,
    }


def _ftl_source_mismatch_errors(
    artifact_source: object,
    current_source: object,
    *,
    label: str,
) -> list[str]:
    if not isinstance(artifact_source, Mapping) or not isinstance(
        current_source, Mapping
    ):
        return [f"{label} and validation source hashes must both be objects"]
    errors: list[str] = []
    for relative_path in sorted(set(artifact_source) | set(current_source)):
        artifact_hash = artifact_source.get(relative_path)
        current_hash = current_source.get(relative_path)
        if artifact_hash != current_hash:
            errors.append(
                f"{label} source hash mismatch for {relative_path}: artifact "
                f"{artifact_hash!r}, current {current_hash!r}"
            )
    return errors


def _ia_acceptance_passed(
    artifact: Mapping[str, object],
    *,
    location: str,
    errors: list[str],
) -> bool | None:
    content = artifact.get("content")
    if not isinstance(content, Mapping):
        errors.append(f"{location}.content must be an object")
        return None
    acceptance = content.get("acceptance")
    if not isinstance(acceptance, Mapping):
        errors.append(f"{location}.content.acceptance must be an object")
        return None
    passed = acceptance.get("passed")
    if not isinstance(passed, bool):
        errors.append(f"{location}.content.acceptance.passed must be boolean")
        return None
    return passed


def _ia_content_digest(
    artifact: Mapping[str, object],
    *,
    location: str,
    errors: list[str],
) -> str | None:
    content = artifact.get("content")
    digest = artifact.get("content_digest")
    if not isinstance(content, Mapping):
        errors.append(f"{location}.content must be an object")
        return None
    if not isinstance(digest, Mapping):
        errors.append(f"{location}.content_digest must be an object")
        return None
    recorded = digest.get("sha256")
    if not _is_sha256(recorded):
        errors.append(f"{location}.content_digest.sha256 must be lowercase SHA-256")
        return None
    computed = _canonical_sha256(content)
    if recorded != computed:
        errors.append(
            f"{location} content digest mismatch: recorded {recorded}, computed {computed}"
        )
    return computed


def _reduced_ia_artifact(
    artifact: Mapping[str, object],
    *,
    location: str,
    errors: list[str],
) -> dict[str, object] | None:
    """Remove exactly the three attested non-scientific/provenance paths."""

    reduced = _strict_json_clone(artifact)
    if not isinstance(reduced, dict):
        errors.append(f"{location} must clone to an object")
        return None
    for key in ("content_digest", "operational_diagnostics"):
        if key not in reduced:
            errors.append(f"{location}.{key} is required before reduction")
            return None
        del reduced[key]
    content = reduced.get("content")
    if not isinstance(content, dict):
        errors.append(f"{location}.content must be an object before reduction")
        return None
    if "source_provenance" not in content:
        errors.append(f"{location}.content.source_provenance is required before reduction")
        return None
    del content["source_provenance"]
    return reduced


def _ia_supporting_artifact_record(
    root: Path,
    relative_path: Path,
    *,
    role: str,
    parsed: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path = root / relative_path
    try:
        raw = path.read_bytes()
    except OSError:
        raw = None
    return {
        "role": role,
        "path": relative_path.as_posix(),
        "present": raw is not None,
        "bytes_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
        "scientific_content_sha256": (_artifact_digest(parsed) if parsed is not None else None),
        "scientific_promotion_allowed": False,
    }


def _validate_ia_archived_sources(
    root: Path,
    pinned: Mapping[str, object],
    *,
    errors: list[str],
) -> None:
    snapshot_dir = root / _IA_SNAPSHOT_MANIFEST_PATH.parent
    wheel_name = "alberta_framework-0.26.0-py3-none-any.whl"
    sdist_name = "alberta_framework-0.26.0.tar.gz"
    wheel_path = snapshot_dir / wheel_name
    sdist_path = snapshot_dir / sdist_name
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            names = wheel.namelist()
            for relative_path, expected_hash in pinned.items():
                if names.count(relative_path) != 1:
                    errors.append(f"historical wheel must contain {relative_path!r} exactly once")
                    continue
                observed = hashlib.sha256(wheel.read(relative_path)).hexdigest()
                if observed != expected_hash:
                    errors.append(
                        "historical wheel source hash mismatch for "
                        f"{relative_path}: expected {expected_hash}, observed {observed}"
                    )
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        errors.append(f"cannot validate historical wheel: {type(error).__name__}: {error}")

    try:
        with tarfile.open(sdist_path, mode="r:gz") as source_distribution:
            member_names = [member.name for member in source_distribution.getmembers()]
            for relative_path, expected_hash in pinned.items():
                archive_path = f"alberta_framework-0.26.0/{relative_path}"
                if member_names.count(archive_path) != 1:
                    errors.append(
                        f"historical source distribution must contain {archive_path!r} exactly once"
                    )
                    continue
                member = source_distribution.getmember(archive_path)
                if not member.isfile():
                    errors.append(
                        f"historical source distribution member {archive_path!r} "
                        "must be a regular file"
                    )
                    continue
                extracted = source_distribution.extractfile(member)
                if extracted is None:
                    errors.append(
                        f"cannot read historical source distribution member {archive_path!r}"
                    )
                    continue
                observed = hashlib.sha256(extracted.read()).hexdigest()
                if observed != expected_hash:
                    errors.append(
                        "historical source distribution hash mismatch for "
                        f"{relative_path}: expected {expected_hash}, observed {observed}"
                    )
    except (OSError, KeyError, tarfile.TarError) as error:
        errors.append(
            f"cannot validate historical source distribution: {type(error).__name__}: {error}"
        )


def _ia_source_mismatch_errors(
    artifact_source: object,
    current_source: object,
) -> list[str]:
    if not isinstance(artifact_source, Mapping) or not isinstance(current_source, Mapping):
        return ["current replay and validation source hashes must both be objects"]
    errors: list[str] = []
    for relative_path in sorted(set(artifact_source) | set(current_source)):
        replay_hash = artifact_source.get(relative_path)
        current_hash = current_source.get(relative_path)
        if replay_hash != current_hash:
            errors.append(
                "current replay source hash mismatch for "
                f"{relative_path}: replay {replay_hash!r}, current {current_hash!r}"
            )
    return errors


def _validate_historical_ftl_chain(
    root: Path,
    spec: EvidenceSpec,
    *,
    primary_artifact: Mapping[str, object],
    primary_raw_bytes: bytes,
    primary_validation: ValidationResult,
    primary_contract_errors: Sequence[str],
    source_snapshot: Mapping[str, object],
) -> _HistoricalFTLChainValidation:
    """Validate the sole FTL v1 historical-acceptance compatibility path.

    The original held-out run remains the only promoting evidence.  The
    current-source rerun uses those already-consumed seeds and is therefore
    compatibility evidence only.  No threshold, seed role, configuration, or
    scientific value may change.  The sole allowed source difference is the
    artifact-builder file; the world model, evaluator, and CLI must remain
    byte-identical to the original artifact's pins.
    """

    errors: list[str] = []
    loaded: dict[str, dict[str, object]] = {}
    json_paths = {
        "historical artifact": spec.relative_path,
        "compatibility attestation": _FTL_ATTESTATION_PATH,
        "current consumed-seed replay": _FTL_REPLAY_PATH,
    }
    for label, relative_path in json_paths.items():
        try:
            loaded[label] = _load_strict_json_object(root / relative_path)
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"cannot load {label}: {type(error).__name__}: {error}")

    historical = loaded.get("historical artifact")
    attestation = loaded.get("compatibility attestation")
    replay = loaded.get("current consumed-seed replay")
    artifacts = (
        _ftl_supporting_artifact_record(
            root,
            _FTL_ATTESTATION_PATH,
            role="historical_compatibility_attestation",
            parsed=attestation,
        ),
        _ftl_supporting_artifact_record(
            root,
            _FTL_REPLAY_PATH,
            role="nonpromoting_consumed_seed_replay",
            parsed=replay,
        ),
    )

    historical_acceptance: bool | None = None
    historical_scientific_digest: str | None = None
    historical_reduced_digest: str | None = None
    replay_acceptance: bool | None = None
    replay_scientific_digest: str | None = None
    replay_reduced_digest: str | None = None
    exact_reduced_match = False
    projected_validation_valid = False
    projected_validation_accepted = False
    projected_validation_errors: tuple[str, ...] = ()
    replay_validation_valid = False
    replay_validation_accepted = False
    replay_validation_errors: tuple[str, ...] = ()

    if primary_validation.valid:
        errors.append("historical FTL primary unexpectedly validates current sources")
    if primary_validation.accepted:
        errors.append(
            "historical FTL primary validator must report acceptance=false "
            "before compatibility reconstruction"
        )
    if primary_validation.errors != (_FTL_SOURCE_DRIFT_VALIDATOR_ERROR,):
        errors.append(
            "historical FTL primary validator errors are not exactly the "
            "allowlisted artifact-builder source-drift diagnostic"
        )
    if tuple(primary_contract_errors) != (_FTL_SOURCE_DRIFT_CONTRACT_ERROR,):
        errors.append(
            "historical FTL primary contract errors are not exactly the "
            "allowlisted artifact-builder source-drift diagnostic"
        )

    primary_bytes_digest = hashlib.sha256(primary_raw_bytes).hexdigest()
    if primary_bytes_digest != _FTL_HISTORICAL_ARTIFACT_BYTES_SHA256:
        errors.append(
            "historical FTL artifact byte digest does not match the immutable "
            "v1 anchor"
        )

    historical_source_hashes: Mapping[str, object] | None = None
    if historical is not None:
        if historical != primary_artifact:
            errors.append("strictly reloaded historical FTL artifact differs from primary loader")
        try:
            current_primary_bytes = (root / spec.relative_path).read_bytes()
        except OSError as error:
            errors.append(f"cannot reread historical FTL primary artifact: {error}")
        else:
            if current_primary_bytes != primary_raw_bytes:
                errors.append("historical FTL primary artifact changed during validation")

        historical_acceptance = _ftl_acceptance_passed(
            historical,
            location="historical_artifact",
            errors=errors,
        )
        if historical_acceptance is not True:
            errors.append("historical FTL artifact scientific acceptance must be true")
        historical_scientific_digest = _ftl_scientific_digest(
            historical,
            location="historical_artifact",
            errors=errors,
        )
        if historical_scientific_digest != _FTL_HISTORICAL_SCIENTIFIC_SHA256:
            errors.append(
                "historical FTL scientific digest does not match the immutable "
                "v1 anchor"
            )
        scientific = historical.get("scientific_payload")
        source = (
            scientific.get("source_provenance")
            if isinstance(scientific, Mapping)
            else None
        )
        historical_source_hashes = (
            source.get("source_sha256") if isinstance(source, Mapping) else None
        )

    source_keys = tuple(path.as_posix() for path in spec.source_paths)
    current_source_hashes = source_snapshot.get("source_sha256")
    if not isinstance(historical_source_hashes, Mapping):
        errors.append("historical FTL source hashes must be an object")
    else:
        if set(historical_source_hashes) != set(source_keys):
            errors.append(
                "historical FTL source hashes do not contain exactly the four "
                "registered paths"
            )
        if any(not _is_sha256(value) for value in historical_source_hashes.values()):
            errors.append(
                "every historical FTL source hash must be lowercase SHA-256"
            )
    if not isinstance(current_source_hashes, Mapping):
        errors.append("current FTL validation source hashes must be an object")
    else:
        if set(current_source_hashes) != set(source_keys):
            errors.append(
                "current FTL source snapshot does not contain exactly the four "
                "registered paths"
            )
        if any(not _is_sha256(value) for value in current_source_hashes.values()):
            errors.append("every current FTL source hash must be lowercase SHA-256")

    if isinstance(historical_source_hashes, Mapping) and isinstance(
        current_source_hashes, Mapping
    ):
        drifted_paths = tuple(
            path
            for path in source_keys
            if historical_source_hashes.get(path) != current_source_hashes.get(path)
        )
        expected_drift = (_FTL_ARTIFACT_BUILDER_PATH.as_posix(),)
        if drifted_paths != expected_drift:
            errors.append(
                "historical/current FTL source drift must be exactly the "
                f"artifact-builder path; observed {list(drifted_paths)!r}"
            )
        for path in _FTL_INVARIANT_SOURCE_PATHS:
            key = path.as_posix()
            if historical_source_hashes.get(key) != current_source_hashes.get(key):
                errors.append(
                    f"historical/current invariant FTL source drift at {key}"
                )

    # Recompute the original acceptance with the current strict validator after
    # projecting only the one changed builder hash and its deterministic digest.
    # This does not mutate or rewrite the historical artifact.
    if historical is not None and isinstance(current_source_hashes, Mapping):
        projected = _strict_json_clone(historical)
        if not isinstance(projected, dict):
            errors.append("historical FTL projection must clone to an object")
        else:
            projected_scientific = projected.get("scientific_payload")
            projected_source = (
                projected_scientific.get("source_provenance")
                if isinstance(projected_scientific, dict)
                else None
            )
            projected_hashes = (
                projected_source.get("source_sha256")
                if isinstance(projected_source, dict)
                else None
            )
            projected_digest = projected.get("scientific_digest")
            builder_key = _FTL_ARTIFACT_BUILDER_PATH.as_posix()
            current_builder_hash = current_source_hashes.get(builder_key)
            if (
                not isinstance(projected_scientific, dict)
                or not isinstance(projected_hashes, dict)
                or not isinstance(projected_digest, dict)
                or not _is_sha256(current_builder_hash)
            ):
                errors.append(
                    "historical FTL projection lacks a valid scientific "
                    "payload, digest, or current builder hash"
                )
            else:
                projected_hashes[builder_key] = current_builder_hash
                projected_digest["sha256"] = _canonical_sha256(
                    projected_scientific
                )
                try:
                    projected_validation = spec.validator(projected)
                    projected_validation_valid = projected_validation.valid
                    projected_validation_accepted = projected_validation.accepted
                    projected_validation_errors = projected_validation.errors
                    if not isinstance(projected_validation_valid, bool):
                        raise TypeError("validator valid field must be boolean")
                    if not isinstance(projected_validation_accepted, bool):
                        raise TypeError("validator accepted field must be boolean")
                    if not isinstance(projected_validation_errors, tuple) or not all(
                        isinstance(error, str)
                        for error in projected_validation_errors
                    ):
                        raise TypeError(
                            "validator errors field must be a tuple of strings"
                        )
                except Exception as error:
                    errors.append(
                        "projected historical FTL validator raised "
                        f"{type(error).__name__}: {error}"
                    )
                else:
                    if not projected_validation_valid:
                        errors.extend(
                            f"projected historical FTL validator: {error}"
                            for error in projected_validation_errors
                        )
                        if not projected_validation_errors:
                            errors.append(
                                "projected historical FTL validator reported "
                                "invalid without diagnostics"
                            )
                    elif projected_validation_errors:
                        errors.append(
                            "projected historical FTL validator reported valid "
                            "with non-empty errors"
                        )
                    if not projected_validation_accepted:
                        errors.append(
                            "projected historical FTL acceptance reconstruction "
                            "must pass"
                        )
                projected_contract_errors, _, _, _ = _contract_errors(
                    projected,
                    spec,
                    source_snapshot,
                )
                errors.extend(
                    f"projected historical FTL contract: {error}"
                    for error in projected_contract_errors
                )

    replay_source_hashes: object = None
    if replay is not None:
        replay_acceptance = _ftl_acceptance_passed(
            replay,
            location="current_source_replay",
            errors=errors,
        )
        if replay_acceptance is not True:
            errors.append(
                "current consumed-seed FTL replay scientific acceptance must "
                "be true, while remaining nonpromoting"
            )
        replay_scientific_digest = _ftl_scientific_digest(
            replay,
            location="current_source_replay",
            errors=errors,
        )
        try:
            replay_validation = spec.validator(replay)
            replay_validation_valid = replay_validation.valid
            replay_validation_accepted = replay_validation.accepted
            replay_validation_errors = replay_validation.errors
            if not isinstance(replay_validation_valid, bool):
                raise TypeError("validator valid field must be boolean")
            if not isinstance(replay_validation_accepted, bool):
                raise TypeError("validator accepted field must be boolean")
            if not isinstance(replay_validation_errors, tuple) or not all(
                isinstance(error, str) for error in replay_validation_errors
            ):
                raise TypeError("validator errors field must be a tuple of strings")
        except Exception as error:
            errors.append(
                f"strict current FTL replay validator raised "
                f"{type(error).__name__}: {error}"
            )
        else:
            if not replay_validation_valid:
                errors.extend(
                    f"strict current FTL replay validator: {error}"
                    for error in replay_validation_errors
                )
                if not replay_validation_errors:
                    errors.append(
                        "strict current FTL replay validator reported invalid "
                        "without diagnostics"
                    )
            elif replay_validation_errors:
                errors.append(
                    "strict current FTL replay validator reported valid with "
                    "non-empty errors"
                )
            if not replay_validation_accepted:
                errors.append(
                    "strict current FTL replay validator must reconstruct "
                    "acceptance=true"
                )

        replay_contract_errors, _, _, _ = _contract_errors(
            replay,
            spec,
            source_snapshot,
        )
        errors.extend(
            f"strict current FTL replay contract: {error}"
            for error in replay_contract_errors
        )
        replay_scientific = replay.get("scientific_payload")
        replay_source = (
            replay_scientific.get("source_provenance")
            if isinstance(replay_scientific, Mapping)
            else None
        )
        replay_source_hashes = (
            replay_source.get("source_sha256")
            if isinstance(replay_source, Mapping)
            else None
        )
        errors.extend(
            _ftl_source_mismatch_errors(
                replay_source_hashes,
                current_source_hashes,
                label="current FTL replay",
            )
        )

    if historical is not None and replay is not None:
        historical_reduced = _reduced_ftl_artifact(
            historical,
            location="historical_artifact",
            errors=errors,
        )
        replay_reduced = _reduced_ftl_artifact(
            replay,
            location="current_source_replay",
            errors=errors,
        )
        if historical_reduced is not None:
            historical_reduced_digest = _canonical_sha256(historical_reduced)
        if replay_reduced is not None:
            replay_reduced_digest = _canonical_sha256(replay_reduced)
        exact_reduced_match = (
            historical_reduced is not None
            and replay_reduced is not None
            and historical_reduced == replay_reduced
        )
        if not exact_reduced_match:
            errors.append(
                "historical and current FTL replay artifacts are not exactly "
                "equal after only the three attested exclusions"
            )

    historical_attestation: Mapping[str, object] | None = None
    replay_attestation: Mapping[str, object] | None = None
    comparison: Mapping[str, object] | None = None
    if attestation is not None:
        _chain_mapping(
            attestation,
            location="ftl_compatibility_attestation",
            expected_keys={
                "schema_version",
                "historical_artifact",
                "current_source_replay",
                "comparison",
                "interpretation",
                "limitations",
            },
            errors=errors,
        )
        if attestation.get("schema_version") != _FTL_ATTESTATION_SCHEMA:
            errors.append(
                "FTL compatibility attestation schema must be "
                f"{_FTL_ATTESTATION_SCHEMA!r}"
            )
        historical_attestation = _chain_mapping(
            attestation.get("historical_artifact"),
            location="ftl_compatibility_attestation.historical_artifact",
            expected_keys={
                "path",
                "bytes_sha256",
                "scientific_payload_sha256",
                "result",
            },
            errors=errors,
        )
        replay_attestation = _chain_mapping(
            attestation.get("current_source_replay"),
            location="ftl_compatibility_attestation.current_source_replay",
            expected_keys={
                "path",
                "bytes_sha256",
                "scientific_payload_sha256",
                "command",
                "role",
                "seed_role",
            },
            errors=errors,
        )
        comparison = _chain_mapping(
            attestation.get("comparison"),
            location="ftl_compatibility_attestation.comparison",
            expected_keys={
                "excluded_paths",
                "included_scope",
                "canonicalization",
                "historical_reduced_sha256",
                "current_replay_reduced_sha256",
                "exact_match",
                "original_digests_independently_validated",
            },
            errors=errors,
        )
        if attestation.get("interpretation") != _FTL_ATTESTATION_INTERPRETATION:
            errors.append("FTL compatibility interpretation wording is invalid")
        if attestation.get("limitations") != list(_FTL_ATTESTATION_LIMITATIONS):
            errors.append("FTL compatibility limitations do not match frozen wording")

    if historical_attestation is not None:
        if historical_attestation.get("path") != spec.relative_path.as_posix():
            errors.append("attested historical FTL artifact path is invalid")
        if (
            historical_attestation.get("bytes_sha256")
            != _FTL_HISTORICAL_ARTIFACT_BYTES_SHA256
        ):
            errors.append("attested historical FTL artifact byte digest is invalid")
        if (
            historical_attestation.get("scientific_payload_sha256")
            != _FTL_HISTORICAL_SCIENTIFIC_SHA256
        ):
            errors.append(
                "attested historical FTL scientific payload digest is invalid"
            )
        if historical_attestation.get("result") != "accepted-at-generation":
            errors.append(
                "attested historical FTL result must be accepted-at-generation"
            )

    if replay_attestation is not None:
        if replay_attestation.get("path") != _FTL_REPLAY_PATH.as_posix():
            errors.append("attested current FTL replay path is invalid")
        if replay_attestation.get("command") != list(_FTL_REPLAY_COMMAND):
            errors.append("attested current FTL replay command is not the frozen argv")
        if (
            replay_attestation.get("role")
            != "nonpromoting_consumed_seed_replay"
        ):
            errors.append("attested current FTL replay role is invalid")
        if (
            replay_attestation.get("seed_role")
            != "consumed_original_promoted_held_out_evidence_30_59"
        ):
            errors.append("attested current FTL replay seed role is invalid")
        try:
            replay_bytes_digest = _path_sha256(root / _FTL_REPLAY_PATH)
        except OSError as error:
            errors.append(f"cannot hash current consumed-seed FTL replay: {error}")
        else:
            if replay_attestation.get("bytes_sha256") != replay_bytes_digest:
                errors.append(
                    "current FTL replay byte digest mismatch: attested "
                    f"{replay_attestation.get('bytes_sha256')!r}, observed "
                    f"{replay_bytes_digest!r}"
                )
        if (
            replay_attestation.get("scientific_payload_sha256")
            != replay_scientific_digest
        ):
            errors.append("attested current FTL replay scientific digest is invalid")

    if comparison is not None:
        if comparison.get("excluded_paths") != list(
            _FTL_COMPARISON_EXCLUDED_PATHS
        ):
            errors.append(
                "FTL comparison exclusions are not exactly the frozen three paths"
            )
        if comparison.get("included_scope") != _FTL_COMPARISON_SCOPE:
            errors.append("FTL comparison included-scope wording is invalid")
        if comparison.get("canonicalization") != _FTL_CANONICALIZATION:
            errors.append("FTL comparison canonicalization is invalid")
        if (
            comparison.get("historical_reduced_sha256")
            != historical_reduced_digest
        ):
            errors.append("attested historical FTL reduced digest is invalid")
        if (
            comparison.get("current_replay_reduced_sha256")
            != replay_reduced_digest
        ):
            errors.append("attested current FTL replay reduced digest is invalid")
        if comparison.get("exact_match") is not True:
            errors.append("FTL comparison must explicitly attest exact_match=true")
        if comparison.get("original_digests_independently_validated") is not True:
            errors.append(
                "FTL comparison must attest independent validation of both "
                "original scientific digests"
            )

    accepted = (
        not errors
        and historical_acceptance is True
        and projected_validation_valid
        and projected_validation_accepted
        and replay_acceptance is True
        and replay_validation_valid
        and replay_validation_accepted
        and exact_reduced_match
    )
    valid = accepted
    record = {
        "schema_version": _FTL_HISTORICAL_CHAIN_SCHEMA,
        "valid": valid,
        "classification": (
            "historical-accepted-compatible-current-sources"
            if valid
            else "invalid"
        ),
        "historical_scientific_claim_supported": accepted,
        "current_replay_scientific_promotion_allowed": False,
        "primary_live_source_drift_resolved": valid,
        "historical_artifact": {
            "stored_acceptance": historical_acceptance,
            "bytes_sha256": primary_bytes_digest,
            "scientific_payload_sha256": historical_scientific_digest,
            "reconstructed_validator_valid": projected_validation_valid,
            "reconstructed_validator_accepted": projected_validation_accepted,
            "reconstructed_validator_errors": list(
                projected_validation_errors
            ),
        },
        "current_source_replay": {
            "role": "nonpromoting_consumed_seed_replay",
            "consumed_seed_schedule": True,
            "strict_validator_valid": replay_validation_valid,
            "validator_accepted": replay_validation_accepted,
            "validator_errors": list(replay_validation_errors),
            "scientific_acceptance": replay_acceptance,
            "scientific_payload_sha256": replay_scientific_digest,
        },
        "comparison": {
            "excluded_paths": list(_FTL_COMPARISON_EXCLUDED_PATHS),
            "historical_reduced_sha256": historical_reduced_digest,
            "current_replay_reduced_sha256": replay_reduced_digest,
            "exact_match": exact_reduced_match,
        },
        "historical_source_recoverability": {
            "exact_artifact_builder_source_archived": False,
            "claim": (
                "deterministic scientific compatibility only; not complete "
                "historical source recovery"
            ),
        },
        "interpretation": {
            **_FTL_ATTESTATION_INTERPRETATION,
            "manifest_effect": (
                "supports only the original narrow v1 held-out claim; the "
                "consumed-seed replay itself never promotes"
            ),
        },
        "limitations": list(_FTL_ATTESTATION_LIMITATIONS),
        "errors": errors,
    }
    return _HistoricalFTLChainValidation(
        valid=valid,
        accepted=accepted,
        errors=tuple(errors),
        artifacts=artifacts,
        record=record,
    )


def _validate_historical_ia_chain(
    root: Path,
    spec: EvidenceSpec,
    *,
    primary_artifact: Mapping[str, object],
    primary_raw_bytes: bytes,
    primary_validation: ValidationResult,
    primary_contract_errors: Sequence[str],
    source_snapshot: Mapping[str, object],
) -> _HistoricalIAChainValidation:
    """Verify the sole historical IA v1 rejection-retention path.

    The consumed-seed replay is compatibility evidence only.  Even a perfect
    replay can retain the original result only as a historical rejection; it
    can never promote a current-source scientific claim.
    """

    errors: list[str] = []
    loaded: dict[str, dict[str, object]] = {}
    json_paths = {
        "historical artifact": spec.relative_path,
        "source snapshot manifest": _IA_SNAPSHOT_MANIFEST_PATH,
        "nonpromotion attestation": _IA_ATTESTATION_PATH,
        "current consumed-seed replay": _IA_REPLAY_PATH,
    }
    for label, relative_path in json_paths.items():
        try:
            loaded[label] = _load_strict_json_object(root / relative_path)
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"cannot load {label}: {type(error).__name__}: {error}")

    historical = loaded.get("historical artifact")
    snapshot = loaded.get("source snapshot manifest")
    attestation = loaded.get("nonpromotion attestation")
    replay = loaded.get("current consumed-seed replay")
    artifacts = (
        _ia_supporting_artifact_record(
            root,
            _IA_SNAPSHOT_MANIFEST_PATH,
            role="historical_source_snapshot_manifest",
            parsed=snapshot,
        ),
        *(
            _ia_supporting_artifact_record(
                root,
                _IA_SNAPSHOT_MANIFEST_PATH.parent / archive_name,
                role="historical_source_archive",
            )
            for archive_name in _IA_ARCHIVE_ROLES
        ),
        _ia_supporting_artifact_record(
            root,
            _IA_ATTESTATION_PATH,
            role="historical_nonpromotion_attestation",
            parsed=attestation,
        ),
        _ia_supporting_artifact_record(
            root,
            _IA_REPLAY_PATH,
            role="nonpromoting_consumed_seed_replay",
            parsed=replay,
        ),
    )

    replay_validation_valid = False
    replay_validation_accepted = False
    replay_validation_errors: tuple[str, ...] = ()
    historical_acceptance: bool | None = None
    replay_acceptance: bool | None = None
    historical_content_digest: str | None = None
    replay_content_digest: str | None = None
    historical_reduced_digest: str | None = None
    replay_reduced_digest: str | None = None
    exact_reduced_match = False

    if primary_validation.valid:
        errors.append("historical primary validator unexpectedly reported valid current sources")
    if primary_validation.accepted:
        errors.append("historical primary validator must report acceptance=false")
    if primary_validation.errors != (_IA_SOURCE_DRIFT_VALIDATOR_ERROR,):
        errors.append(
            "historical primary validator errors are not exactly the allowlisted "
            "live-source-drift diagnostic"
        )
    if tuple(primary_contract_errors) != (_IA_SOURCE_DRIFT_CONTRACT_ERROR,):
        errors.append(
            "historical primary contract errors are not exactly the allowlisted "
            "live-source-drift diagnostic"
        )

    if historical is not None:
        if historical != primary_artifact:
            errors.append("strictly reloaded historical artifact differs from primary loader")
        try:
            current_primary_bytes = (root / spec.relative_path).read_bytes()
        except OSError as error:
            errors.append(f"cannot reread historical primary artifact: {error}")
        else:
            if current_primary_bytes != primary_raw_bytes:
                errors.append("historical primary artifact changed during validation")
        historical_acceptance = _ia_acceptance_passed(
            historical,
            location="historical_artifact",
            errors=errors,
        )
        if historical_acceptance is not False:
            errors.append("historical artifact scientific acceptance must be false")
        historical_content_digest = _ia_content_digest(
            historical,
            location="historical_artifact",
            errors=errors,
        )

    source_keys = tuple(path.as_posix() for path in spec.source_paths)
    pinned: Mapping[str, object] | None = None
    snapshot_artifact: Mapping[str, object] | None = None
    snapshot_archives: Mapping[str, object] | None = None
    if snapshot is not None:
        _chain_mapping(
            snapshot,
            location="source_snapshot_manifest",
            expected_keys={
                "schema_version",
                "artifact",
                "archives",
                "artifact_pinned_source_sha256",
                "verification",
                "limitations",
            },
            errors=errors,
        )
        if snapshot.get("schema_version") != _IA_SNAPSHOT_SCHEMA:
            errors.append(f"source snapshot schema must be {_IA_SNAPSHOT_SCHEMA!r}")
        snapshot_artifact = _chain_mapping(
            snapshot.get("artifact"),
            location="source_snapshot_manifest.artifact",
            expected_keys={
                "path",
                "schema_version",
                "bytes_sha256",
                "scientific_content_sha256",
                "result",
            },
            errors=errors,
        )
        snapshot_archives = _chain_mapping(
            snapshot.get("archives"),
            location="source_snapshot_manifest.archives",
            expected_keys=set(_IA_ARCHIVE_ROLES),
            errors=errors,
        )
        pinned = _chain_mapping(
            snapshot.get("artifact_pinned_source_sha256"),
            location="source_snapshot_manifest.artifact_pinned_source_sha256",
            expected_keys=set(source_keys),
            errors=errors,
        )
        if len(source_keys) != 8:
            errors.append("historical IA chain requires exactly eight registered source paths")
        if pinned is not None and any(not _is_sha256(value) for value in pinned.values()):
            errors.append("every historical IA pinned source hash must be lowercase SHA-256")
        verification = snapshot.get("verification")
        if verification != {
            "all_artifact_pinned_paths_present_in_wheel": True,
            "all_artifact_pinned_path_hashes_match_wheel": True,
            "artifact_preserved_byte_for_byte": True,
        }:
            errors.append("source snapshot verification attestation wording is invalid")
        if snapshot.get("limitations") != list(_IA_SNAPSHOT_LIMITATIONS):
            errors.append("source snapshot limitations do not match the frozen wording")

    if snapshot_artifact is not None:
        if snapshot_artifact.get("path") != spec.relative_path.as_posix():
            errors.append("source snapshot artifact path does not identify the primary artifact")
        if snapshot_artifact.get("schema_version") != spec.expected_schema:
            errors.append("source snapshot artifact schema does not match the claim contract")
        if snapshot_artifact.get("result") != "valid-rejection-at-generation":
            errors.append("source snapshot must attest valid-rejection-at-generation")
        observed_primary_bytes = hashlib.sha256(primary_raw_bytes).hexdigest()
        if snapshot_artifact.get("bytes_sha256") != observed_primary_bytes:
            errors.append(
                "historical artifact byte digest mismatch: snapshot "
                f"{snapshot_artifact.get('bytes_sha256')!r}, observed "
                f"{observed_primary_bytes!r}"
            )
        if snapshot_artifact.get("scientific_content_sha256") != historical_content_digest:
            errors.append("historical artifact content digest does not match source snapshot")

    if snapshot_archives is not None:
        for archive_name, expected_role in _IA_ARCHIVE_ROLES.items():
            archive_record = _chain_mapping(
                snapshot_archives.get(archive_name),
                location=f"source_snapshot_manifest.archives.{archive_name}",
                expected_keys={"sha256", "role"},
                errors=errors,
            )
            if archive_record is None:
                continue
            if archive_record.get("role") != expected_role:
                errors.append(f"historical archive role is invalid for {archive_name}")
            archive_path = root / _IA_SNAPSHOT_MANIFEST_PATH.parent / archive_name
            try:
                observed_archive_digest = _path_sha256(archive_path)
            except OSError as error:
                errors.append(f"cannot hash historical archive {archive_name}: {error}")
                continue
            if archive_record.get("sha256") != observed_archive_digest:
                errors.append(
                    f"historical archive digest mismatch for {archive_name}: "
                    f"recorded {archive_record.get('sha256')!r}, "
                    f"observed {observed_archive_digest!r}"
                )

    if historical is not None and pinned is not None:
        historical_content = historical.get("content")
        historical_source = (
            historical_content.get("source_provenance")
            if isinstance(historical_content, Mapping)
            else None
        )
        historical_source_hashes = (
            historical_source.get("source_sha256")
            if isinstance(historical_source, Mapping)
            else None
        )
        if historical_source_hashes != pinned:
            errors.append("historical artifact source hashes do not equal the archived pinned map")
        _validate_ia_archived_sources(root, pinned, errors=errors)

    historical_attestation: Mapping[str, object] | None = None
    replay_attestation: Mapping[str, object] | None = None
    comparison: Mapping[str, object] | None = None
    if attestation is not None:
        _chain_mapping(
            attestation,
            location="nonpromotion_attestation",
            expected_keys={
                "schema_version",
                "historical_artifact",
                "current_source_replay",
                "comparison",
                "interpretation",
                "limitations",
            },
            errors=errors,
        )
        if attestation.get("schema_version") != _IA_ATTESTATION_SCHEMA:
            errors.append(f"nonpromotion attestation schema must be {_IA_ATTESTATION_SCHEMA!r}")
        historical_attestation = _chain_mapping(
            attestation.get("historical_artifact"),
            location="nonpromotion_attestation.historical_artifact",
            expected_keys={
                "path",
                "bytes_sha256",
                "scientific_content_sha256",
                "source_snapshot_manifest",
            },
            errors=errors,
        )
        replay_attestation = _chain_mapping(
            attestation.get("current_source_replay"),
            location="nonpromotion_attestation.current_source_replay",
            expected_keys={
                "path",
                "bytes_sha256",
                "scientific_content_sha256",
                "command",
            },
            errors=errors,
        )
        comparison = _chain_mapping(
            attestation.get("comparison"),
            location="nonpromotion_attestation.comparison",
            expected_keys={
                "excluded_paths",
                "included_scope",
                "canonicalization",
                "historical_reduced_sha256",
                "current_replay_reduced_sha256",
                "exact_match",
            },
            errors=errors,
        )
        if attestation.get("interpretation") != _IA_ATTESTATION_INTERPRETATION:
            errors.append("nonpromotion attestation interpretation wording is invalid")
        if attestation.get("limitations") != list(_IA_ATTESTATION_LIMITATIONS):
            errors.append("nonpromotion attestation limitations do not match frozen wording")

    primary_bytes_digest = hashlib.sha256(primary_raw_bytes).hexdigest()
    if historical_attestation is not None:
        if historical_attestation.get("path") != spec.relative_path.as_posix():
            errors.append("attested historical artifact path is invalid")
        if historical_attestation.get("bytes_sha256") != primary_bytes_digest:
            errors.append("attested historical artifact byte digest is invalid")
        if historical_attestation.get("scientific_content_sha256") != historical_content_digest:
            errors.append("attested historical artifact content digest is invalid")
        if (
            historical_attestation.get("source_snapshot_manifest")
            != _IA_SNAPSHOT_MANIFEST_PATH.as_posix()
        ):
            errors.append("attested historical source snapshot path is invalid")
        if snapshot_artifact is not None and (
            historical_attestation.get("bytes_sha256") != snapshot_artifact.get("bytes_sha256")
            or historical_attestation.get("scientific_content_sha256")
            != snapshot_artifact.get("scientific_content_sha256")
        ):
            errors.append("snapshot and attestation disagree about historical artifact digests")

    if replay is not None:
        replay_acceptance = _ia_acceptance_passed(
            replay,
            location="current_source_replay",
            errors=errors,
        )
        if replay_acceptance is not False:
            errors.append("current consumed-seed replay scientific acceptance must be false")
        replay_content_digest = _ia_content_digest(
            replay,
            location="current_source_replay",
            errors=errors,
        )
        try:
            if validate_ia_evidence_artifact is _STRICT_IA_ARTIFACT_VALIDATOR:
                replay_validation = _validate_ia_historical_replay_projection(replay)
            else:
                replay_validation = validate_ia_evidence_artifact(replay)
            replay_validation_valid = replay_validation.valid
            replay_validation_accepted = replay_validation.accepted
            replay_validation_errors = replay_validation.errors
            if not isinstance(replay_validation_valid, bool):
                raise TypeError("validator valid field must be boolean")
            if not isinstance(replay_validation_accepted, bool):
                raise TypeError("validator accepted field must be boolean")
            if not isinstance(replay_validation_errors, tuple) or not all(
                isinstance(error, str) for error in replay_validation_errors
            ):
                raise TypeError("validator errors field must be a tuple of strings")
        except Exception as error:
            errors.append(f"strict current replay validator raised {type(error).__name__}: {error}")
        else:
            if not replay_validation_valid:
                errors.extend(
                    f"strict current replay validator: {error}"
                    for error in replay_validation_errors
                )
                if not replay_validation_errors:
                    errors.append(
                        "strict current replay validator reported invalid without diagnostics"
                    )
            elif replay_validation_errors:
                errors.append(
                    "strict current replay validator reported valid with non-empty errors"
                )
            if replay_validation_accepted:
                errors.append("strict current replay validator must report acceptance=false")

        replay_contract_errors, _, _, _ = _contract_errors(
            replay,
            spec,
            source_snapshot,
        )
        errors.extend(
            f"strict current replay contract: {error}" for error in replay_contract_errors
        )
        replay_content = replay.get("content")
        replay_source = (
            replay_content.get("source_provenance") if isinstance(replay_content, Mapping) else None
        )
        replay_source_hashes = (
            replay_source.get("source_sha256") if isinstance(replay_source, Mapping) else None
        )
        errors.extend(
            _ia_source_mismatch_errors(
                replay_source_hashes,
                source_snapshot.get("source_sha256"),
            )
        )

    if replay_attestation is not None:
        if replay_attestation.get("path") != _IA_REPLAY_PATH.as_posix():
            errors.append("attested current replay path is invalid")
        if replay_attestation.get("command") != list(_IA_REPLAY_COMMAND):
            errors.append("attested current replay command is not the frozen argv")
        try:
            replay_bytes_digest = _path_sha256(root / _IA_REPLAY_PATH)
        except OSError as error:
            errors.append(f"cannot hash current consumed-seed replay: {error}")
        else:
            if replay_attestation.get("bytes_sha256") != replay_bytes_digest:
                errors.append(
                    "current replay byte digest mismatch: attested "
                    f"{replay_attestation.get('bytes_sha256')!r}, observed "
                    f"{replay_bytes_digest!r}"
                )
        if replay_attestation.get("scientific_content_sha256") != replay_content_digest:
            errors.append("attested current replay content digest is invalid")

    if historical is not None and replay is not None:
        historical_reduced = _reduced_ia_artifact(
            historical,
            location="historical_artifact",
            errors=errors,
        )
        replay_reduced = _reduced_ia_artifact(
            replay,
            location="current_source_replay",
            errors=errors,
        )
        if historical_reduced is not None:
            historical_reduced_digest = _canonical_sha256(historical_reduced)
        if replay_reduced is not None:
            replay_reduced_digest = _canonical_sha256(replay_reduced)
        exact_reduced_match = (
            historical_reduced is not None
            and replay_reduced is not None
            and historical_reduced == replay_reduced
        )
        if not exact_reduced_match:
            errors.append(
                "historical and current replay artifacts are not exactly equal "
                "after only the three attested exclusions"
            )

    if comparison is not None:
        if comparison.get("excluded_paths") != list(_IA_REDUCED_EXCLUDED_PATHS):
            errors.append("comparison exclusions are not exactly the frozen three paths")
        if comparison.get("included_scope") != _IA_REDUCED_INCLUDED_SCOPE:
            errors.append("comparison included-scope wording is invalid")
        if comparison.get("canonicalization") != _IA_CANONICALIZATION:
            errors.append("comparison canonicalization is invalid")
        if comparison.get("historical_reduced_sha256") != historical_reduced_digest:
            errors.append("attested historical reduced digest is invalid")
        if comparison.get("current_replay_reduced_sha256") != replay_reduced_digest:
            errors.append("attested current replay reduced digest is invalid")
        if comparison.get("exact_match") is not True:
            errors.append("comparison must explicitly attest exact_match=true")

    valid = not errors
    record = {
        "schema_version": _IA_HISTORICAL_CHAIN_SCHEMA,
        "valid": valid,
        "classification": "historical-valid-rejection" if valid else "invalid",
        "scientific_promotion_allowed": False,
        "primary_live_source_drift_resolved": valid,
        "historical_artifact": {
            "accepted": historical_acceptance,
            "bytes_sha256": primary_bytes_digest,
            "scientific_content_sha256": historical_content_digest,
            "archived_source_count": len(pinned) if pinned is not None else None,
        },
        "current_source_replay": {
            "role": "nonpromoting_consumed_seed_replay",
            "consumed_seed_schedule": True,
            "strict_validator_valid": replay_validation_valid,
            "validator_accepted": replay_validation_accepted,
            "validator_errors": list(replay_validation_errors),
            "scientific_acceptance": replay_acceptance,
            "scientific_content_sha256": replay_content_digest,
        },
        "comparison": {
            "excluded_paths": list(_IA_REDUCED_EXCLUDED_PATHS),
            "historical_reduced_sha256": historical_reduced_digest,
            "current_replay_reduced_sha256": replay_reduced_digest,
            "exact_match": exact_reduced_match,
        },
        "interpretation": {
            **_IA_ATTESTATION_INTERPRETATION,
            "manifest_effect": (
                "may retain only the original historical valid-rejection "
                "classification; never a current-source acceptance"
            ),
        },
        "limitations": list(_IA_ATTESTATION_LIMITATIONS),
        "errors": errors,
    }
    return _HistoricalIAChainValidation(
        valid=valid,
        errors=tuple(errors),
        artifacts=artifacts,
        record=record,
    )


def _artifact_record(
    spec: EvidenceSpec,
    *,
    present: bool,
    observed_schema: object,
    raw_bytes: bytes | None,
    scientific_digest: str | None,
) -> dict[str, object]:
    return {
        "role": "primary_evidence",
        "path": spec.relative_path.as_posix(),
        "expected_schema_version": spec.expected_schema,
        "observed_schema_version": observed_schema,
        "present": present,
        "bytes_sha256": (hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None),
        "scientific_content_sha256": scientific_digest,
    }


def _common_record(
    spec: EvidenceSpec,
    *,
    validated_at_utc: str,
    validation_environment: Mapping[str, object],
    source_snapshot: Mapping[str, object],
    artifact_environment: Mapping[str, object] | None,
    artifact_source: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "contract_version": CLAIM_CONTRACT_VERSION,
        "claim_id": spec.name,
        "scope": spec.claim_scope,
        "level": spec.evidence_level,
        "evidence_class": spec.evidence_class,
        "scientific_promotion_allowed": spec.promotes_scientific_claim,
        "command": {
            "argv": list(spec.command_argv),
            "display": shlex.join(spec.command_argv),
            "shell": False,
        },
        "protocol": _strict_json_clone(spec.protocol),
        "configuration": _strict_json_clone(spec.configuration),
        "seeds": _strict_json_clone(spec.seeds),
        "thresholds": _strict_json_clone(spec.thresholds),
        "environment_provenance": {
            "required_artifact_fields": list(spec.required_environment_fields),
            "validation_runtime": _strict_json_clone(validation_environment),
            "artifact_runtime": (
                _strict_json_clone(artifact_environment)
                if artifact_environment is not None
                else None
            ),
        },
        "source_provenance": {
            "validation_snapshot": _strict_json_clone(source_snapshot),
            "artifact_snapshot": (
                _strict_json_clone(artifact_source) if artifact_source is not None else None
            ),
        },
        "dirty_state_policy": _strict_json_clone(DIRTY_STATE_POLICY),
        "limitations": list(spec.limitations),
        "validation_timestamp_utc": validated_at_utc,
    }


def _historical_ftl_chain_is_eligible(
    spec: EvidenceSpec,
    *,
    schema_matches: bool,
    source_errors: Sequence[str],
    validation: ValidationResult,
    contract_errors: Sequence[str],
) -> bool:
    """Allow fallback only for the exact frozen FTL v1 builder-hash drift."""

    return (
        spec.name == _FTL_CLAIM_ID
        and spec.relative_path == _FTL_ARTIFACT_PATH
        and spec.expected_schema == FTL_SCHEMA_VERSION
        and spec.evidence_class == "scientific"
        and spec.evidence_level == "L2"
        and spec.promotes_scientific_claim
        and spec.protocol == _FTL_PROTOCOL
        and spec.configuration
        == _ftl_configuration_payload(DecisionFidelityConfig())
        and spec.thresholds == _ftl_threshold_payload()
        and spec.source_paths == _FTL_SOURCE_PATHS
        and schema_matches
        and not source_errors
        and not validation.valid
        and not validation.accepted
        and validation.errors == (_FTL_SOURCE_DRIFT_VALIDATOR_ERROR,)
        and tuple(contract_errors) == (_FTL_SOURCE_DRIFT_CONTRACT_ERROR,)
    )


def _historical_ia_chain_is_eligible(
    spec: EvidenceSpec,
    *,
    schema_matches: bool,
    source_errors: Sequence[str],
    validation: ValidationResult,
    contract_errors: Sequence[str],
) -> bool:
    """Allow a fallback attempt only for the exact registered IA v1 drift."""

    return (
        spec.name == _IA_CLAIM_ID
        and spec.relative_path == _IA_ARTIFACT_PATH
        and spec.expected_schema == IA_SCHEMA_VERSION
        and spec.evidence_class == "scientific"
        and spec.evidence_level == "L2"
        and spec.promotes_scientific_claim
        and spec.protocol == _IA_PROTOCOL
        and spec.configuration == _ia_configuration_payload(ContinualIAConfig())
        and spec.thresholds == _ia_threshold_payload(IAAcceptanceThresholds())
        and spec.source_paths == _IA_SOURCE_PATHS
        and schema_matches
        and not source_errors
        and not validation.valid
        and not validation.accepted
        and validation.errors == (_IA_SOURCE_DRIFT_VALIDATOR_ERROR,)
        and tuple(contract_errors) == (_IA_SOURCE_DRIFT_CONTRACT_ERROR,)
    )


def _inspect_spec(
    root: Path,
    spec: EvidenceSpec,
    *,
    validated_at_utc: str,
    validation_environment: Mapping[str, object],
) -> dict[str, object]:
    path = root / spec.relative_path
    source_snapshot, source_errors = _source_snapshot(root, spec)
    absent_artifact = _artifact_record(
        spec,
        present=False,
        observed_schema=None,
        raw_bytes=None,
        scientific_digest=None,
    )
    if not path.is_file():
        status = "invalid" if source_errors else "not-run"
        common = _common_record(
            spec,
            validated_at_utc=validated_at_utc,
            validation_environment=validation_environment,
            source_snapshot=source_snapshot,
            artifact_environment=None,
            artifact_source=None,
        )
        return {
            **common,
            "status": status,
            "present": False,
            "valid": False,
            "validator_accepted": False,
            "accepted": False,
            "scientific_claim_supported": False,
            "errors": [*source_errors, "artifact is absent"],
            "artifacts": [absent_artifact],
        }

    raw_bytes: bytes | None = None
    try:
        raw_bytes = path.read_bytes()
        artifact = spec.loader(path)
        if not isinstance(artifact, Mapping):
            raise TypeError("artifact loader must return a mapping")
        validation = spec.validator(artifact)
        validation_valid = validation.valid
        validation_accepted = validation.accepted
        validation_errors = validation.errors
        if not isinstance(validation_valid, bool):
            raise TypeError("validator valid field must be boolean")
        if not isinstance(validation_accepted, bool):
            raise TypeError("validator accepted field must be boolean")
        if not isinstance(validation_errors, tuple) or not all(
            isinstance(error, str) for error in validation_errors
        ):
            raise TypeError("validator errors field must be a tuple of strings")
    except Exception as error:
        common = _common_record(
            spec,
            validated_at_utc=validated_at_utc,
            validation_environment=validation_environment,
            source_snapshot=source_snapshot,
            artifact_environment=None,
            artifact_source=None,
        )
        return {
            **common,
            "status": "invalid",
            "present": True,
            "valid": False,
            "validator_accepted": False,
            "accepted": False,
            "scientific_claim_supported": False,
            "errors": [
                *source_errors,
                f"{type(error).__name__}: {error}",
            ],
            "artifacts": [
                _artifact_record(
                    spec,
                    present=True,
                    observed_schema=None,
                    raw_bytes=raw_bytes,
                    scientific_digest=None,
                )
            ],
        }

    schema = artifact.get("schema_version")
    schema_matches = schema == spec.expected_schema
    contract_errors, _, artifact_environment, artifact_source = _contract_errors(
        artifact,
        spec,
        source_snapshot,
    )
    base_errors = [*source_errors, *validation_errors, *contract_errors]
    if not schema_matches:
        base_errors.append(
            f"manifest registry expected schema {spec.expected_schema!r}, observed {schema!r}"
        )
    historical_ftl_chain: _HistoricalFTLChainValidation | None = None
    historical_ia_chain: _HistoricalIAChainValidation | None = None
    if _historical_ftl_chain_is_eligible(
        spec,
        schema_matches=schema_matches,
        source_errors=source_errors,
        validation=validation,
        contract_errors=contract_errors,
    ):
        assert raw_bytes is not None
        historical_ftl_chain = _validate_historical_ftl_chain(
            root,
            spec,
            primary_artifact=artifact,
            primary_raw_bytes=raw_bytes,
            primary_validation=validation,
            primary_contract_errors=contract_errors,
            source_snapshot=source_snapshot,
        )
    if _historical_ia_chain_is_eligible(
        spec,
        schema_matches=schema_matches,
        source_errors=source_errors,
        validation=validation,
        contract_errors=contract_errors,
    ):
        assert raw_bytes is not None
        historical_ia_chain = _validate_historical_ia_chain(
            root,
            spec,
            primary_artifact=artifact,
            primary_raw_bytes=raw_bytes,
            primary_validation=validation,
            primary_contract_errors=contract_errors,
            source_snapshot=source_snapshot,
        )
    if historical_ftl_chain is not None and historical_ftl_chain.valid:
        errors: list[str] = []
        valid = True
        validation_basis = (
            "historical-accepted-consumed-seed-compatibility-chain"
        )
        effective_validator_accepted = historical_ftl_chain.accepted
    elif historical_ia_chain is not None and historical_ia_chain.valid:
        errors = []
        valid = True
        validation_basis = "archived-source-historical-rejection-chain"
        effective_validator_accepted = validation_accepted
    else:
        errors = base_errors
        if historical_ftl_chain is not None:
            errors.extend(
                f"historical FTL acceptance chain: {error}"
                for error in historical_ftl_chain.errors
            )
            validation_basis = "historical-acceptance-compatibility-chain-failed"
        elif historical_ia_chain is not None:
            errors.extend(
                f"historical IA rejection chain: {error}"
                for error in historical_ia_chain.errors
            )
            validation_basis = "historical-rejection-chain-failed"
        else:
            validation_basis = "current-registered-sources"
        valid = validation_valid and schema_matches and not errors
        effective_validator_accepted = validation_accepted
    validator_accepted = effective_validator_accepted
    accepted = (
        valid
        and validator_accepted
        and spec.promotes_scientific_claim
        and spec.evidence_class == "scientific"
        and spec.evidence_level in {"L2", "L3"}
    )
    if not valid:
        status = "invalid"
    elif not spec.promotes_scientific_claim:
        status = "verified-nonpromoting"
    elif accepted:
        status = "accepted"
    else:
        status = "valid-rejection"
    common = _common_record(
        spec,
        validated_at_utc=validated_at_utc,
        validation_environment=validation_environment,
        source_snapshot=source_snapshot,
        artifact_environment=artifact_environment,
        artifact_source=artifact_source,
    )
    artifacts = [
        _artifact_record(
            spec,
            present=True,
            observed_schema=schema,
            raw_bytes=raw_bytes,
            scientific_digest=_artifact_digest(artifact),
        )
    ]
    if historical_ftl_chain is not None:
        artifacts.extend(historical_ftl_chain.artifacts)
    if historical_ia_chain is not None:
        artifacts.extend(historical_ia_chain.artifacts)
    record: dict[str, object] = {
        **common,
        "status": status,
        "present": True,
        "valid": valid,
        "validation_basis": validation_basis,
        "primary_validator_valid": validation_valid,
        "primary_validator_accepted": validation_accepted,
        "validator_accepted": validator_accepted,
        "accepted": accepted,
        "scientific_claim_supported": accepted,
        "errors": errors,
        "artifacts": artifacts,
    }
    if historical_ftl_chain is not None:
        record["historical_validation"] = historical_ftl_chain.record
    if historical_ia_chain is not None:
        record["historical_validation"] = historical_ia_chain.record
    return record


def build_evidence_manifest(
    root: Path = REPO_ROOT,
    *,
    specs: Sequence[EvidenceSpec] = EVIDENCE_SPECS,
) -> dict[str, object]:
    """Validate every registered claim and return the operational manifest."""

    if not specs:
        raise ValueError("at least one evidence specification is required")
    for spec in specs:
        _validate_spec(spec)
    names = tuple(spec.name for spec in specs)
    paths = tuple(spec.relative_path for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("evidence specification names must be unique")
    if len(set(paths)) != len(paths):
        raise ValueError("evidence specification paths must be unique")

    resolved_root = root.resolve()
    validated_at_utc = datetime.now(UTC).isoformat()
    validation_environment = _validation_environment()
    records = [
        _inspect_spec(
            resolved_root,
            spec,
            validated_at_utc=validated_at_utc,
            validation_environment=validation_environment,
        )
        for spec in specs
    ]
    statuses = [str(record["status"]) for record in records]
    promoting_records = [
        record for record in records if bool(record["scientific_promotion_allowed"])
    ]
    all_present = bool(promoting_records) and all(
        bool(record["present"]) for record in promoting_records
    )
    all_valid = bool(promoting_records) and all(
        bool(record["valid"]) for record in promoting_records
    )
    all_accepted = bool(promoting_records) and all(
        bool(record["accepted"]) for record in promoting_records
    )
    if "invalid" in statuses:
        overall_status = "invalid"
    elif any(record["status"] == "not-run" for record in promoting_records):
        overall_status = "not-run"
    elif not promoting_records:
        overall_status = "nonpromoting"
    elif all_accepted:
        overall_status = "accepted"
    else:
        overall_status = "valid-rejection"

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "claim_contract_version": CLAIM_CONTRACT_VERSION,
        "root": str(resolved_root),
        "validation_timestamp_utc": validated_at_utc,
        "interpretation": (
            "operational manifest of strict narrow-artifact validators; not a "
            "scientific artifact, signature, or Alberta Plan completion "
            "certificate. Unit and smoke evidence cannot promote claims."
        ),
        "overall_status": overall_status,
        "all_required_present": all_present,
        "all_required_valid": all_valid,
        "all_required_accepted": all_accepted,
        "claim_count": len(records),
        "required_scientific_claim_count": len(promoting_records),
        "supported_scientific_claim_count": sum(
            bool(record["scientific_claim_supported"]) for record in records
        ),
        "claims": records,
    }


def evidence_manifest_exit_code(manifest: Mapping[str, object]) -> int:
    """Return 0 accepted, 1 not-run/valid rejection, or 2 invalid."""

    status = manifest.get("overall_status")
    if status == "accepted":
        return 0
    if status in {"nonpromoting", "not-run", "valid-rejection"}:
        return 1
    return 2


def evidence_manifest_json(manifest: Mapping[str, object]) -> str:
    """Serialize the claim manifest as strict, stable JSON."""

    return (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


__all__ = [
    "CLAIM_CONTRACT_VERSION",
    "DIRTY_STATE_POLICY",
    "DIRTY_STATE_POLICY_VERSION",
    "EVIDENCE_SPECS",
    "MANIFEST_SCHEMA_VERSION",
    "REPO_ROOT",
    "EvidenceClass",
    "EvidenceLevel",
    "EvidenceSpec",
    "ValidationResult",
    "build_evidence_manifest",
    "evidence_manifest_exit_code",
    "evidence_manifest_json",
]
