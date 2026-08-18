"""Strict aggregation contract for matched, permanently nonpromoting runs."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_INT32_MAX = 2**31 - 1
_MAX_BYTES = 256 * 1024 * 1024
_MAX_ARMS = 16
_MAX_SEEDS = 128
_MAX_TEXT = 512
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAMESPACE = re.compile(r"outputs/[a-z0-9][a-z0-9._/-]{0,255}")


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TEXT:
        raise ValueError(f"{name} must be a bounded non-empty string")
    return value


def _count(name: str, value: object, *, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


@dataclass(frozen=True)
class MatchedDevelopmentPlan:
    protocol_id: str
    paper_revision: str
    reference_code_revision: str
    source_sha256: str
    report_namespace: str
    metric_name: str
    control: str
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    steps: int

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "paper_revision",
            "reference_code_revision",
            "report_namespace",
            "metric_name",
            "control",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        if type(self.source_sha256) is not str or _SHA256.fullmatch(self.source_sha256) is None:
            raise ValueError("source_sha256 must be lowercase hexadecimal SHA-256")
        if (
            _NAMESPACE.fullmatch(self.report_namespace) is None
            or ".." in self.report_namespace.split("/")
        ):
            raise ValueError("report_namespace must be a bounded relative outputs path")
        if type(self.arms) is not tuple or not 2 <= len(self.arms) <= _MAX_ARMS:
            raise ValueError("arms must be an exact tuple with 2 to 16 entries")
        resolved_arms = tuple(_text("arm", arm) for arm in self.arms)
        if len(set(resolved_arms)) != len(resolved_arms) or self.control not in resolved_arms:
            raise ValueError("arms must be unique and contain control")
        if type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= _MAX_SEEDS:
            raise ValueError("seeds must be an exact tuple with 1 to 128 entries")
        resolved_seeds = tuple(_count("seed", seed) for seed in self.seeds)
        if len(set(resolved_seeds)) != len(resolved_seeds):
            raise ValueError("seeds must be unique")
        object.__setattr__(self, "steps", _count("steps", self.steps, maximum=1_000_000))
        if self.steps < 1:
            raise ValueError("steps must be positive")


@dataclass(frozen=True)
class ArmRecord:
    arm: str
    seed: int
    observation_sha256: str
    updates: int
    data_steps: int
    model_queries: int
    persistent_bytes: int
    peak_working_set_bytes: int
    timing_seconds: float
    metric: float
    transaction_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _text("arm", self.arm))
        object.__setattr__(self, "seed", _count("seed", self.seed))
        if (
            type(self.observation_sha256) is not str
            or _SHA256.fullmatch(self.observation_sha256) is None
        ):
            raise ValueError("observation_sha256 must be lowercase hexadecimal SHA-256")
        for name in ("updates", "data_steps", "model_queries"):
            object.__setattr__(self, name, _count(name, getattr(self, name)))
        for name in ("persistent_bytes", "peak_working_set_bytes"):
            object.__setattr__(
                self, name, _count(name, getattr(self, name), maximum=_MAX_BYTES)
            )
        if (
            type(self.timing_seconds) is not float
            or not math.isfinite(self.timing_seconds)
            or self.timing_seconds < 0.0
            or self.timing_seconds > 604_800.0
        ):
            raise ValueError("timing_seconds must be a finite non-negative float")
        if type(self.metric) is not float or not math.isfinite(self.metric):
            raise ValueError("metric must be a finite float")
        if type(self.transaction_valid) is not bool:
            raise ValueError("transaction_valid must be a bool")


def build_matched_report(
    plan: MatchedDevelopmentPlan, records: tuple[ArmRecord, ...]
) -> dict[str, Any]:
    """Validate and aggregate one exact arm-by-seed cross product."""
    if type(plan) is not MatchedDevelopmentPlan or type(records) is not tuple:
        raise ValueError("plan and records must use exact protocol types")
    expected_count = len(plan.arms) * len(plan.seeds)
    if len(records) != expected_count:
        raise ValueError("records must contain the complete arm-seed cross product")
    indexed: dict[tuple[str, int], ArmRecord] = {}
    observation_by_seed: dict[int, str] = {}
    for record in records:
        if type(record) is not ArmRecord:
            raise ValueError("records must contain exact ArmRecord values")
        key = (record.arm, record.seed)
        if record.arm not in plan.arms or record.seed not in plan.seeds or key in indexed:
            raise ValueError("records contain an unknown or duplicate arm-seed identity")
        if not record.transaction_valid:
            raise ValueError("every record transaction must be valid")
        if (record.updates, record.data_steps, record.model_queries) != (
            plan.steps,
            plan.steps,
            plan.steps,
        ):
            raise ValueError("updates, data steps, and queries must match matched steps")
        prior = observation_by_seed.setdefault(record.seed, record.observation_sha256)
        if prior != record.observation_sha256:
            raise ValueError("arms for one seed must share one observation identity")
        indexed[key] = record
    expected = {(arm, seed) for arm in plan.arms for seed in plan.seeds}
    if set(indexed) != expected:
        raise ValueError("records must contain the complete arm-seed cross product")
    candidates = tuple(arm for arm in plan.arms if arm != plan.control)
    if len(candidates) != 1:
        raise ValueError("v1 reports require exactly one candidate and one control")
    candidate = candidates[0]
    differences = tuple(
        indexed[(candidate, seed)].metric - indexed[(plan.control, seed)].metric
        for seed in plan.seeds
    )
    if not all(math.isfinite(difference) for difference in differences):
        raise ValueError("paired differences must fit in finite float64")
    mean_difference = float(np.mean(np.asarray(differences, dtype=np.float64)))
    if not math.isfinite(mean_difference):
        raise ValueError("mean paired difference must fit in finite float64")
    return {
        "schema": "asi.matched-development.report.v1",
        "plan": asdict(plan),
        "records": tuple(asdict(record) for record in records),
        "candidate": candidate,
        "paired_differences": differences,
        "mean_paired_difference": mean_difference,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
