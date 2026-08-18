"""Strict aggregation and retention for matched, permanently nonpromoting runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Any, cast

_INT32_MAX = 2**31 - 1
_MAX_BYTES = 256 * 1024 * 1024
_MAX_ARMS = 16
_MAX_SEEDS = 128
_MAX_TEXT_BYTES = 512
_MAX_TIMING_SECONDS = 604_800.0
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PATH_SEGMENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a bounded non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if len(encoded) > _MAX_TEXT_BYTES:
        raise ValueError(f"{name} must be a bounded non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def _count(name: str, value: object, *, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


def _namespace(value: object) -> str:
    resolved = _text("report_namespace", value)
    segments = resolved.split("/")
    if (
        len(segments) < 2
        or segments[0] != "outputs"
        or any(_PATH_SEGMENT.fullmatch(segment) is None for segment in segments)
    ):
        raise ValueError("report_namespace must be a canonical bounded relative outputs path")
    return resolved


def _finite_float(name: str, value: object, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (nonnegative and value < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be a {qualifier} float")
    return value


@dataclass(frozen=True)
class MatchedDevelopmentPlan:
    """Frozen two-arm schedule and resource contract.

    ``source_sha256`` is supplied by the issue-specific protocol.  This generic contract
    validates and binds that identifier; it does not derive or authenticate current source.
    """

    protocol_id: str
    paper_revision: str
    reference_code_revision: str
    source_sha256: str
    report_namespace: str
    metric_name: str
    higher_is_better: bool
    tie_tolerance: float
    control: str
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    expected_updates: int
    expected_observations: int
    expected_data_steps: int
    expected_model_queries: int
    persistent_byte_budget: int
    working_set_byte_budget: int

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "paper_revision",
            "reference_code_revision",
            "metric_name",
            "control",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "source_sha256", _sha256("source_sha256", self.source_sha256))
        object.__setattr__(self, "report_namespace", _namespace(self.report_namespace))
        if type(self.higher_is_better) is not bool:
            raise ValueError("higher_is_better must be a bool")
        object.__setattr__(
            self,
            "tie_tolerance",
            _finite_float("tie_tolerance", self.tie_tolerance, nonnegative=True),
        )
        if type(self.arms) is not tuple or not 2 <= len(self.arms) <= _MAX_ARMS:
            raise ValueError("arms must be an exact tuple with 2 to 16 entries")
        resolved_arms = tuple(_text("arm", arm) for arm in self.arms)
        if len(set(resolved_arms)) != len(resolved_arms) or self.control not in resolved_arms:
            raise ValueError("arms must be unique and contain control")
        if len(resolved_arms) != 2:
            raise ValueError("matched report v1 requires exactly one candidate and one control")
        if type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= _MAX_SEEDS:
            raise ValueError("seeds must be an exact tuple with 1 to 128 entries")
        resolved_seeds = tuple(_count("seed", seed) for seed in self.seeds)
        if len(set(resolved_seeds)) != len(resolved_seeds):
            raise ValueError("seeds must be unique")
        for name in (
            "expected_updates",
            "expected_observations",
            "expected_data_steps",
            "expected_model_queries",
        ):
            object.__setattr__(self, name, _count(name, getattr(self, name), maximum=1_000_000))
        for name in ("persistent_byte_budget", "working_set_byte_budget"):
            object.__setattr__(self, name, _count(name, getattr(self, name), maximum=_MAX_BYTES))
        record_count = len(resolved_arms) * len(resolved_seeds)
        for name in (
            "expected_updates",
            "expected_observations",
            "expected_data_steps",
            "expected_model_queries",
        ):
            if getattr(self, name) > _INT32_MAX // record_count:
                raise ValueError(f"aggregate {name} exceeds the signed-int32 bound")


@dataclass(frozen=True)
class ArmRecord:
    plan_sha256: str
    arm: str
    seed: int
    observation_sha256: str
    updates: int
    observations: int
    data_steps: int
    model_queries: int
    persistent_bytes: int
    peak_working_set_bytes: int
    timing_seconds: float
    metric: float
    transaction_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_sha256", _sha256("plan_sha256", self.plan_sha256))
        object.__setattr__(self, "arm", _text("arm", self.arm))
        object.__setattr__(self, "seed", _count("seed", self.seed))
        object.__setattr__(
            self,
            "observation_sha256",
            _sha256("observation_sha256", self.observation_sha256),
        )
        for name in ("updates", "observations", "data_steps", "model_queries"):
            object.__setattr__(self, name, _count(name, getattr(self, name)))
        for name in ("persistent_bytes", "peak_working_set_bytes"):
            object.__setattr__(self, name, _count(name, getattr(self, name), maximum=_MAX_BYTES))
        object.__setattr__(
            self,
            "timing_seconds",
            _finite_float("timing_seconds", self.timing_seconds, nonnegative=True),
        )
        if self.timing_seconds > _MAX_TIMING_SECONDS:
            raise ValueError("timing_seconds exceeds the protocol ceiling")
        object.__setattr__(self, "metric", _finite_float("metric", self.metric))
        if type(self.transaction_valid) is not bool:
            raise ValueError("transaction_valid must be a bool")


def _plan_dict(plan: MatchedDevelopmentPlan) -> dict[str, Any]:
    return {
        "protocol_id": plan.protocol_id,
        "paper_revision": plan.paper_revision,
        "reference_code_revision": plan.reference_code_revision,
        "source_sha256": plan.source_sha256,
        "report_namespace": plan.report_namespace,
        "metric_name": plan.metric_name,
        "higher_is_better": plan.higher_is_better,
        "tie_tolerance": plan.tie_tolerance,
        "control": plan.control,
        "arms": plan.arms,
        "seeds": plan.seeds,
        "expected_updates": plan.expected_updates,
        "expected_observations": plan.expected_observations,
        "expected_data_steps": plan.expected_data_steps,
        "expected_model_queries": plan.expected_model_queries,
        "persistent_byte_budget": plan.persistent_byte_budget,
        "working_set_byte_budget": plan.working_set_byte_budget,
    }


def _record_dict(record: ArmRecord) -> dict[str, Any]:
    return {
        "plan_sha256": record.plan_sha256,
        "arm": record.arm,
        "seed": record.seed,
        "observation_sha256": record.observation_sha256,
        "updates": record.updates,
        "observations": record.observations,
        "data_steps": record.data_steps,
        "model_queries": record.model_queries,
        "persistent_bytes": record.persistent_bytes,
        "peak_working_set_bytes": record.peak_working_set_bytes,
        "timing_seconds": record.timing_seconds,
        "metric": record.metric,
        "transaction_valid": record.transaction_valid,
    }


def _revalidate_plan(plan: object) -> MatchedDevelopmentPlan:
    if type(plan) is not MatchedDevelopmentPlan:
        raise ValueError("plan must use the exact MatchedDevelopmentPlan type")
    return MatchedDevelopmentPlan(**_plan_dict(plan))


def _revalidate_record(record: object) -> ArmRecord:
    if type(record) is not ArmRecord:
        raise ValueError("records must contain exact ArmRecord values")
    return ArmRecord(**_record_dict(record))


def _canonical_json_bytes_raw(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("report must have one finite canonical JSON representation") from error
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("canonical report exceeds the byte ceiling")
    return encoded


def plan_sha256(plan: MatchedDevelopmentPlan) -> str:
    """Return the canonical identity of a fully revalidated plan."""
    resolved = _revalidate_plan(plan)
    return hashlib.sha256(_canonical_json_bytes_raw(_plan_dict(resolved))).hexdigest()


def _finite_mean(name: str, values: tuple[float, ...]) -> float:
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ValueError(f"{name} must fit in finite float64") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must fit in finite float64")
    return result


def _stderr(values: tuple[float, ...], mean: float) -> float:
    if len(values) == 1:
        return 0.0
    centered = tuple(value - mean for value in values)
    if not all(math.isfinite(value) for value in centered):
        raise ValueError("paired stderr must fit in finite float64")
    scale = max(abs(value) for value in centered)
    if scale == 0.0:
        return 0.0
    normalized = math.fsum((value / scale) ** 2 for value in centered)
    result = scale * math.sqrt(normalized / (len(values) * (len(values) - 1)))
    if not math.isfinite(result):
        raise ValueError("paired stderr must fit in finite float64")
    return result


def _arm_summary(records: tuple[ArmRecord, ...]) -> dict[str, int | float]:
    metrics = tuple(record.metric for record in records)
    metric_mean = _finite_mean("arm mean metric", metrics)
    return {
        "record_count": len(records),
        "mean_metric": metric_mean,
        "metric_stderr": _stderr(metrics, metric_mean),
        "total_updates": sum(record.updates for record in records),
        "total_observations": sum(record.observations for record in records),
        "total_data_steps": sum(record.data_steps for record in records),
        "total_model_queries": sum(record.model_queries for record in records),
        "mean_persistent_bytes": _finite_mean(
            "mean persistent bytes", tuple(float(record.persistent_bytes) for record in records)
        ),
        "max_persistent_bytes": max(record.persistent_bytes for record in records),
        "mean_peak_working_set_bytes": _finite_mean(
            "mean peak working bytes",
            tuple(float(record.peak_working_set_bytes) for record in records),
        ),
        "max_peak_working_set_bytes": max(
            record.peak_working_set_bytes for record in records
        ),
        "total_timing_seconds": _finite_mean(
            "mean timing", tuple(record.timing_seconds for record in records)
        )
        * len(records),
    }


def build_matched_report(
    plan: MatchedDevelopmentPlan, records: tuple[ArmRecord, ...]
) -> dict[str, Any]:
    """Validate and aggregate one exact arm-by-seed cross product."""
    resolved_plan = _revalidate_plan(plan)
    if type(records) is not tuple:
        raise ValueError("records must be an exact tuple")
    expected_count = len(resolved_plan.arms) * len(resolved_plan.seeds)
    if len(records) != expected_count:
        raise ValueError("records must contain the complete arm-seed cross product")
    identity = plan_sha256(resolved_plan)
    indexed: dict[tuple[str, int], ArmRecord] = {}
    observation_by_seed: dict[int, str] = {}
    total_persistent = 0
    total_working = 0
    total_timing = 0.0
    for untrusted_record in records:
        record = _revalidate_record(untrusted_record)
        if record.plan_sha256 != identity:
            raise ValueError("every record must bind the exact canonical plan identity")
        key = (record.arm, record.seed)
        if (
            record.arm not in resolved_plan.arms
            or record.seed not in resolved_plan.seeds
            or key in indexed
        ):
            raise ValueError("records contain an unknown or duplicate arm-seed identity")
        if not record.transaction_valid:
            raise ValueError("every record transaction must be valid")
        if (record.updates, record.observations, record.data_steps, record.model_queries) != (
            resolved_plan.expected_updates,
            resolved_plan.expected_observations,
            resolved_plan.expected_data_steps,
            resolved_plan.expected_model_queries,
        ):
            raise ValueError("record counters must equal the plan's expected counters")
        if (
            record.persistent_bytes > resolved_plan.persistent_byte_budget
            or record.peak_working_set_bytes > resolved_plan.working_set_byte_budget
        ):
            raise ValueError("record resources exceed the plan budget")
        if record.persistent_bytes > _MAX_BYTES - total_persistent:
            raise ValueError("aggregate resource totals exceed the byte ceiling")
        if record.peak_working_set_bytes > _MAX_BYTES - total_working:
            raise ValueError("aggregate resource totals exceed the byte ceiling")
        total_persistent += record.persistent_bytes
        total_working += record.peak_working_set_bytes
        total_timing += record.timing_seconds
        if not math.isfinite(total_timing) or total_timing > _MAX_TIMING_SECONDS:
            raise ValueError("aggregate timing exceeds the protocol ceiling")
        prior = observation_by_seed.setdefault(record.seed, record.observation_sha256)
        if prior != record.observation_sha256:
            raise ValueError("arms for one seed must share one observation identity")
        indexed[key] = record
    expected = {
        (arm, seed) for seed in resolved_plan.seeds for arm in resolved_plan.arms
    }
    if set(indexed) != expected:
        raise ValueError("records must contain the complete arm-seed cross product")
    canonical_records = tuple(
        indexed[(arm, seed)] for seed in resolved_plan.seeds for arm in resolved_plan.arms
    )
    candidate = next(arm for arm in resolved_plan.arms if arm != resolved_plan.control)
    raw_differences = tuple(
        indexed[(candidate, seed)].metric - indexed[(resolved_plan.control, seed)].metric
        for seed in resolved_plan.seeds
    )
    if not all(math.isfinite(difference) for difference in raw_differences):
        raise ValueError("paired differences must fit in finite float64")
    oriented = tuple(
        difference if resolved_plan.higher_is_better else -difference
        for difference in raw_differences
    )
    mean_improvement = _finite_mean("mean paired difference", oriented)
    if mean_improvement > resolved_plan.tie_tolerance:
        outcome = "improved"
    elif mean_improvement < -resolved_plan.tie_tolerance:
        outcome = "worse"
    else:
        outcome = "tied"
    summaries = {
        arm: _arm_summary(tuple(indexed[(arm, seed)] for seed in resolved_plan.seeds))
        for arm in resolved_plan.arms
    }
    payload: dict[str, Any] = {
        "schema": "asi.matched-development.report.v2",
        "plan_sha256": identity,
        "plan": _plan_dict(resolved_plan),
        "records": tuple(_record_dict(record) for record in canonical_records),
        "candidate": candidate,
        "paired_differences": raw_differences,
        "mean_paired_difference": _finite_mean(
            "mean raw paired difference", raw_differences
        ),
        "mean_oriented_improvement": mean_improvement,
        "paired_stderr": _stderr(oriented, mean_improvement),
        "outcome": outcome,
        "arm_summaries": summaries,
        "aggregate_persistent_bytes": total_persistent,
        "aggregate_peak_working_set_bytes": total_working,
        "aggregate_timing_seconds": total_timing,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    payload["report_sha256"] = hashlib.sha256(_canonical_json_bytes_raw(payload)).hexdigest()
    return payload


def _exact_dict(name: str, value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact dict")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be exact strings")
    return value


def _from_report_plan(value: object) -> MatchedDevelopmentPlan:
    raw = _exact_dict("report plan", value)
    expected = tuple(MatchedDevelopmentPlan.__dataclass_fields__)
    if len(raw) != len(expected) or not all(name in raw for name in expected):
        raise ValueError("report plan fields do not match the exact schema")
    converted = dict(raw)
    if type(converted["arms"]) is list:
        converted["arms"] = tuple(converted["arms"])
    if type(converted["seeds"]) is list:
        converted["seeds"] = tuple(converted["seeds"])
    return MatchedDevelopmentPlan(**converted)


def _from_report_record(value: object) -> ArmRecord:
    raw = _exact_dict("report record", value)
    expected = tuple(ArmRecord.__dataclass_fields__)
    if len(raw) != len(expected) or not all(name in raw for name in expected):
        raise ValueError("report record fields do not match the exact schema")
    return ArmRecord(**raw)


def validate_matched_report(report: object) -> dict[str, Any]:
    """Strictly reload and reconstruct a report; caller fields are never trusted."""
    raw = _exact_dict("report", report)
    if len(raw) > 32:
        raise ValueError("report has too many top-level fields")
    plan = _from_report_plan(raw.get("plan"))
    raw_records = raw.get("records")
    if type(raw_records) not in (tuple, list):
        raise ValueError("report records must be an exact tuple or JSON list")
    record_values = cast(tuple[object, ...] | list[object], raw_records)
    expected_count = len(plan.arms) * len(plan.seeds)
    if len(record_values) != expected_count:
        raise ValueError("report records do not match the bounded plan cross product")
    records = tuple(_from_report_record(record) for record in record_values)
    rebuilt = build_matched_report(plan, records)
    if _canonical_json_bytes_raw(raw) != _canonical_json_bytes_raw(rebuilt):
        raise ValueError("report does not match the canonical report derived from its records")
    return rebuilt


def canonical_report_bytes(report: object) -> bytes:
    """Return the one allowed finite, sorted UTF-8 JSON encoding."""
    return _canonical_json_bytes_raw(validate_matched_report(report))


def retain_matched_report(report: object, *, repository_root: Path) -> Path:
    """Publish one report with exclusive create, then strictly reload it."""
    if type(repository_root) is not PosixPath or not repository_root.is_absolute():
        raise ValueError("repository_root must be an exact absolute POSIX Path")
    validated = validate_matched_report(report)
    namespace = _from_report_plan(validated["plan"]).report_namespace
    root_resolved = repository_root.resolve(strict=True)
    if not root_resolved.is_dir():
        raise ValueError("repository_root must resolve to an existing directory")
    directory = root_resolved
    for segment in namespace.split("/"):
        directory /= segment
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("report namespace contains a non-directory or symbolic link")
        else:
            directory.mkdir(mode=0o755)
    destination = directory / "report.v2.json"
    encoded = _canonical_json_bytes_raw(validated)
    temporary = directory / f".report.v2.{validated['report_sha256']}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
    loaded = destination.read_bytes()
    if loaded != encoded or canonical_report_bytes(json.loads(loaded)) != encoded:
        raise RuntimeError("retained matched report failed strict reload validation")
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination
