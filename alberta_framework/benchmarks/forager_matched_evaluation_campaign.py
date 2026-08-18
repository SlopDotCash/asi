"""Pure sealed-evaluation scheduling for the matched-current Forager campaign.

The sealed panel has six arms: four slots resolved by open-tuning selection
(the Alberta winner plus the top three external candidates) and the two fixed
descriptive candidates (``exact_ppo``, ``search_oracle``), which are never
selection- or pairing-eligible.  Every arm runs the frozen 30-seed evaluation
block, giving the 180-cell seed-major schedule built here (see
:mod:`forager_matched_open_protocol` for the frozen panel and seed blocks).

This module owns only reward-free content construction and replay.  It does
not read or write campaign state, execute candidates, authenticate a seal, or
authorize scientific promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

from alberta_framework.benchmarks import forager_matched_protocol as protocol

MATCHED_SEALED_TRANSITION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_sealed_transition.v1"
)
MATCHED_SEALED_SCHEDULE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_sealed_evaluation_schedule.v1"
)
MATCHED_SEALED_EVALUATION_SCHEDULE_SCHEMA_VERSION: Final = (
    MATCHED_SEALED_SCHEDULE_SCHEMA_VERSION
)

# Frozen matched-current panel shape: 4 selection slots (1 alberta rank +
# 3 external ranks) + 2 fixed descriptive candidates = 6 arms, each over the
# 30-seed evaluation block disjoint from tuning => 6 x 30 = 180 cells.
_EXPECTED_SELECTED_CANDIDATES: Final = 4
_EXPECTED_FIXED_DESCRIPTIVE_CANDIDATES: Final = 2
_EXPECTED_EVALUATION_CANDIDATES: Final = 6
_EXPECTED_EVALUATION_SEEDS: Final = 30
_EXPECTED_EVALUATION_CELLS: Final = 180
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class ForagerMatchedEvaluationCampaignError(ValueError):
    """A sealed transition or pure evaluation schedule failed closed."""


@dataclass(frozen=True, slots=True)
class _ValidatedTransition:
    sealed_protocol: protocol.ForagerMatchedProtocol
    transition: protocol.SealedProtocolValidation


def _plain_json(value: Any) -> Any:
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ForagerMatchedEvaluationCampaignError(
                    "canonical mappings require string keys"
                )
            result[key] = _plain_json(item)
        return result
    if type(value) is list or type(value) is tuple:
        return [_plain_json(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ForagerMatchedEvaluationCampaignError(
        f"canonical content contains unsupported {type(value).__name__}"
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedEvaluationCampaignError(
            "value is not canonical JSON"
        ) from exc


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedEvaluationCampaignError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _normalize_sealed_protocol(
    value: protocol.ForagerMatchedProtocol,
) -> protocol.ForagerMatchedProtocol:
    if type(value) is not protocol.ForagerMatchedProtocol:
        raise TypeError("sealed_protocol must be a ForagerMatchedProtocol")
    try:
        sealed = protocol.parse_forager_matched_protocol(value.to_dict())
    except protocol.ForagerMatchedProtocolError as exc:
        raise ForagerMatchedEvaluationCampaignError(
            f"sealed protocol is invalid: {exc}"
        ) from exc
    if sealed.stage != "sealed_evaluation":
        raise ForagerMatchedEvaluationCampaignError(
            "sealed protocol must have stage sealed_evaluation"
        )
    if sealed.selection_outcome.status != "resolved":
        raise ForagerMatchedEvaluationCampaignError(
            "sealed protocol selection outcome must be resolved"
        )
    if sealed.active_seeds != sealed.evaluation_seeds:
        raise ForagerMatchedEvaluationCampaignError(
            "sealed protocol active seeds must be exactly the evaluation seeds"
        )
    return sealed


def _expected_transition(
    sealed: protocol.ForagerMatchedProtocol,
) -> protocol.SealedProtocolValidation:
    outcome = sealed.selection_outcome
    if outcome.open_protocol_sha256 is None or outcome.selection_result_sha256 is None:
        raise ForagerMatchedEvaluationCampaignError(
            "sealed protocol is missing its open-protocol or selection-result binding"
        )
    resolved_slots = outcome.resolved_slots
    expected_slots = sealed.selection_plan.slots
    if tuple(item.slot for item in resolved_slots) != expected_slots:
        raise ForagerMatchedEvaluationCampaignError(
            "sealed protocol resolved-slot order differs from the selection plan"
        )
    slot_index = {item.slot: item.candidate_id for item in resolved_slots}
    if len(slot_index) != len(resolved_slots):
        raise ForagerMatchedEvaluationCampaignError(
            "sealed protocol resolves a selection slot more than once"
        )

    selected_ids = tuple(
        slot_index[slot] for slot in sealed.evaluation_panel.selection_slots
    )
    fixed_ids = sealed.evaluation_panel.fixed_descriptive_candidate_ids
    evaluation_ids = selected_ids + fixed_ids
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ForagerMatchedEvaluationCampaignError(
            "sealed evaluation panel contains a candidate more than once"
        )
    for candidate_id in selected_ids:
        candidate = sealed.candidate_index[candidate_id]
        if (
            candidate.pairing.analysis_role != "inferential"
            or candidate.pairing.eligible is not True
        ):
            raise ForagerMatchedEvaluationCampaignError(
                "selected evaluation candidates must be pairing-eligible and inferential"
            )
    for candidate_id in fixed_ids:
        candidate = sealed.candidate_index[candidate_id]
        if (
            candidate.pairing.analysis_role != "descriptive_only"
            or candidate.pairing.eligible is not False
        ):
            raise ForagerMatchedEvaluationCampaignError(
                "fixed evaluation candidates must remain descriptive and pairing-ineligible"
            )

    hypotheses = (sealed.primary_hypothesis,) + sealed.secondary_hypotheses
    resolved_hypotheses = tuple(
        protocol.ResolvedHypothesis(
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
    return protocol.SealedProtocolValidation(
        open_protocol_sha256=outcome.open_protocol_sha256,
        selection_result_sha256=outcome.selection_result_sha256,
        resolved_slots=resolved_slots,
        evaluation_candidate_ids=evaluation_ids,
        primary_intervention_candidate_id=primary.intervention_candidate_id,
        primary_comparator_candidate_id=primary.comparator_candidate_id,
        resolved_hypotheses=resolved_hypotheses,
    )


def _validate_transition(
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: protocol.SealedProtocolValidation,
) -> _ValidatedTransition:
    sealed = _normalize_sealed_protocol(sealed_protocol)
    if type(transition) is not protocol.SealedProtocolValidation:
        raise TypeError("transition must be a SealedProtocolValidation")
    expected = _expected_transition(sealed)
    if transition != expected:
        field_names = (
            "open_protocol_sha256",
            "selection_result_sha256",
            "resolved_slots",
            "evaluation_candidate_ids",
            "primary_intervention_candidate_id",
            "primary_comparator_candidate_id",
            "resolved_hypotheses",
        )
        drifted = [
            name
            for name in field_names
            if getattr(transition, name) != getattr(expected, name)
        ]
        raise ForagerMatchedEvaluationCampaignError(
            "typed sealed transition differs from the sealed protocol: "
            + ", ".join(drifted)
        )
    return _ValidatedTransition(sealed_protocol=sealed, transition=transition)


def _validate_matched_current_schedule_shape(
    validated: _ValidatedTransition,
) -> None:
    sealed = validated.sealed_protocol
    selected_count = len(sealed.evaluation_panel.selection_slots)
    fixed_count = len(sealed.evaluation_panel.fixed_descriptive_candidate_ids)
    if selected_count != _EXPECTED_SELECTED_CANDIDATES:
        raise ForagerMatchedEvaluationCampaignError(
            "sealed evaluation schedule requires exactly four selected candidates"
        )
    if fixed_count != _EXPECTED_FIXED_DESCRIPTIVE_CANDIDATES:
        raise ForagerMatchedEvaluationCampaignError(
            "sealed evaluation schedule requires exactly two fixed descriptive candidates"
        )
    if len(validated.transition.evaluation_candidate_ids) != (
        _EXPECTED_EVALUATION_CANDIDATES
    ):
        raise ForagerMatchedEvaluationCampaignError(
            "sealed evaluation schedule requires exactly six candidates"
        )
    if (
        len(sealed.evaluation_seeds) != _EXPECTED_EVALUATION_SEEDS
        or len(set(sealed.evaluation_seeds)) != _EXPECTED_EVALUATION_SEEDS
    ):
        raise ForagerMatchedEvaluationCampaignError(
            "sealed evaluation schedule requires the exact 30-seed evaluation block"
        )
    if set(sealed.evaluation_seeds) & set(sealed.tuning_seeds):
        raise ForagerMatchedEvaluationCampaignError(
            "sealed evaluation seeds overlap the open tuning seeds"
        )


def _transition_descriptor(
    validated: _ValidatedTransition,
) -> dict[str, Any]:
    transition = validated.transition
    return {
        "schema_version": MATCHED_SEALED_TRANSITION_SCHEMA_VERSION,
        "open_protocol_sha256": transition.open_protocol_sha256,
        "selection_result_sha256": transition.selection_result_sha256,
        "sealed_protocol_sha256": validated.sealed_protocol.protocol_sha256,
        "resolved_slots": [item.to_dict() for item in transition.resolved_slots],
        "evaluation_candidate_ids": list(transition.evaluation_candidate_ids),
        "primary_intervention_candidate_id": (
            transition.primary_intervention_candidate_id
        ),
        "primary_comparator_candidate_id": transition.primary_comparator_candidate_id,
        "resolved_hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "intervention_candidate_id": item.intervention_candidate_id,
                "comparator_candidate_id": item.comparator_candidate_id,
                "method": item.method,
                "alternative": item.alternative,
                "difference_order": item.difference_order,
            }
            for item in transition.resolved_hypotheses
        ],
    }


def build_sealed_transition_descriptor(
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: protocol.SealedProtocolValidation,
) -> dict[str, Any]:
    """Return the canonical v1 descriptor for one typed sealed transition."""
    return _transition_descriptor(_validate_transition(sealed_protocol, transition))


def canonical_sealed_transition_descriptor_bytes(
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: protocol.SealedProtocolValidation,
) -> bytes:
    """Return canonical bytes for the fully replayed sealed transition descriptor."""
    return _canonical_json_bytes(
        build_sealed_transition_descriptor(sealed_protocol, transition)
    )


def canonical_sealed_transition_descriptor_sha256(
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: protocol.SealedProtocolValidation,
) -> str:
    """Return the canonical digest of the fully replayed transition descriptor."""
    return hashlib.sha256(
        canonical_sealed_transition_descriptor_bytes(sealed_protocol, transition)
    ).hexdigest()


def build_sealed_evaluation_schedule(
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: protocol.SealedProtocolValidation,
) -> dict[str, Any]:
    """Build the canonical, reward-free 6-by-30 sealed evaluation schedule."""
    validated = _validate_transition(sealed_protocol, transition)
    _validate_matched_current_schedule_shape(validated)
    sealed = validated.sealed_protocol
    candidate_ids = validated.transition.evaluation_candidate_ids
    transition_sha256 = _canonical_sha256(_transition_descriptor(validated))
    unsigned: dict[str, Any] = {
        "schema_version": MATCHED_SEALED_SCHEDULE_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "stage": "sealed_evaluation",
        "protocol_sha256": sealed.protocol_sha256,
        "sealed_transition_sha256": transition_sha256,
        "ordering": "seed_major_then_sealed_evaluation_candidate_order",
        "candidate_order": list(candidate_ids),
        "active_seeds": list(sealed.evaluation_seeds),
        "cells": [
            {"ordinal": ordinal, "candidate_id": candidate_id, "seed": seed}
            for ordinal, (seed, candidate_id) in enumerate(
                (seed, candidate_id)
                for seed in sealed.evaluation_seeds
                for candidate_id in candidate_ids
            )
        ],
        "promotion_authorized": False,
        "external_verification_required": True,
    }
    if len(unsigned["cells"]) != _EXPECTED_EVALUATION_CELLS:
        raise ForagerMatchedEvaluationCampaignError(
            "sealed evaluation schedule must contain exactly 180 cells"
        )
    return {**unsigned, "schedule_sha256": _canonical_sha256(unsigned)}


def _decode_schedule(
    value: Mapping[str, Any] | bytes | str,
) -> tuple[dict[str, Any], bytes | None]:
    raw: bytes | None = None
    if type(value) is bytes:
        raw = value
        encoded: bytes | str = value
    elif type(value) is str:
        try:
            raw = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ForagerMatchedEvaluationCampaignError(
                "schedule is not valid UTF-8 JSON"
            ) from exc
        encoded = value
    elif type(value) is dict:
        plain = _plain_json(value)
        if type(plain) is not dict:
            raise ForagerMatchedEvaluationCampaignError(
                "schedule must be a JSON object"
            )
        return cast(dict[str, Any], plain), None
    else:
        raise TypeError("schedule must be a mapping, bytes, or string")
    try:
        decoded = protocol.decode_strict_json(encoded)
    except protocol.ForagerMatchedProtocolError as exc:
        raise ForagerMatchedEvaluationCampaignError(
            f"schedule is not strict JSON: {exc}"
        ) from exc
    if type(decoded) is not dict:
        raise ForagerMatchedEvaluationCampaignError("schedule must be a JSON object")
    return cast(dict[str, Any], decoded), raw


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {
                key: _freeze_json(item)
                for key, item in cast(dict[str, Any], value).items()
            }
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def parse_sealed_evaluation_schedule(
    value: Mapping[str, Any] | bytes | str,
    *,
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: protocol.SealedProtocolValidation,
    expected_schedule_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Strictly parse and replay a schedule against its protocol and typed transition."""
    payload, raw = _decode_schedule(value)
    expected_keys = {
        "schema_version",
        "classification",
        "stage",
        "protocol_sha256",
        "sealed_transition_sha256",
        "ordering",
        "candidate_order",
        "active_seeds",
        "cells",
        "promotion_authorized",
        "external_verification_required",
        "schedule_sha256",
    }
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        extra = sorted(set(payload) - expected_keys)
        raise ForagerMatchedEvaluationCampaignError(
            f"schedule fields differ (missing={missing}, extra={extra})"
        )
    declared = _require_sha256(payload["schedule_sha256"], "schedule_sha256")
    unsigned = dict(payload)
    del unsigned["schedule_sha256"]
    if _canonical_sha256(unsigned) != declared:
        raise ForagerMatchedEvaluationCampaignError(
            "schedule self-hash does not match its unsigned content"
        )
    expected_digest = (
        None
        if expected_schedule_sha256 is None
        else _require_sha256(expected_schedule_sha256, "expected_schedule_sha256")
    )
    if expected_digest is not None and declared != expected_digest:
        raise ForagerMatchedEvaluationCampaignError(
            "schedule does not match the expected schedule digest"
        )
    expected = build_sealed_evaluation_schedule(sealed_protocol, transition)
    if payload != expected:
        raise ForagerMatchedEvaluationCampaignError(
            "schedule differs from the replayed sealed evaluation schedule"
        )
    canonical = _canonical_json_bytes(payload)
    if raw is not None and raw != canonical:
        raise ForagerMatchedEvaluationCampaignError(
            "schedule bytes are not the exact canonical encoding"
        )
    return cast(Mapping[str, Any], _freeze_json(payload))


def canonical_sealed_evaluation_schedule_bytes(
    value: Mapping[str, Any] | bytes | str,
    *,
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: protocol.SealedProtocolValidation,
    expected_schedule_sha256: str | None = None,
) -> bytes:
    """Return canonical bytes after strict schedule self-hash and replay validation."""
    parsed = parse_sealed_evaluation_schedule(
        value,
        sealed_protocol=sealed_protocol,
        transition=transition,
        expected_schedule_sha256=expected_schedule_sha256,
    )
    return _canonical_json_bytes(parsed)


__all__ = [
    "ForagerMatchedEvaluationCampaignError",
    "MATCHED_SEALED_EVALUATION_SCHEDULE_SCHEMA_VERSION",
    "MATCHED_SEALED_SCHEDULE_SCHEMA_VERSION",
    "MATCHED_SEALED_TRANSITION_SCHEMA_VERSION",
    "build_sealed_evaluation_schedule",
    "build_sealed_transition_descriptor",
    "canonical_sealed_evaluation_schedule_bytes",
    "canonical_sealed_transition_descriptor_bytes",
    "canonical_sealed_transition_descriptor_sha256",
    "parse_sealed_evaluation_schedule",
]
