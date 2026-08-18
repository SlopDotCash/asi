"""Host-side, nonexecuting qualification plan for fixed-action RNG parity.

This module deliberately never launches Docker.  It freezes two distinct OCI
collector commands over one public/open seed and provides a strict receipt
constructor that combines their hash-only outputs.  The receipt remains
content identity: its external executor-receipt digest must be authenticated by
the caller's trust resolver before it can qualify an evidence binding.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from alberta_framework.benchmarks import forager_rng_parity as parity

HOST_QUALIFICATION_PLAN_SCHEMA: Final = "alberta.forager_fixed_action_rng_parity.host_plan.v1"
HOST_QUALIFICATION_RECEIPT_SCHEMA: Final = "alberta.forager_fixed_action_rng_parity.host_receipt.v1"
HOST_QUALIFICATION_CLASSIFICATION: Final = "open_runtime_qualification_nonpromoting"
CONTAINER_PROBE_PATH: Final = "/tmp/src/forager_rng_parity.py"
PROBE_SOURCE_PLACEHOLDER: Final = "{verified_probe_source}"
DOCKER_PATH: Final = "/usr/bin/docker"
LIVE_DOCKER_EXECUTION_DEFERRED: Final = True

# This seed is intentionally public and outside every confidential evaluation
# lease.  Changing it or the actions changes the plan digest and schema artifact.
OPEN_QUALIFICATION_CONFIG: Final = parity.FixedActionProbeConfig(
    seed=17,
    actions=(0, 1, 2, 3, 3, 2, 1, 0),
)


class ForagerRngParityQualificationError(ValueError):
    """A host plan, collector pair, or qualification receipt is invalid."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(parity.canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ForagerRngParityQualificationError(f"{label} must be a lowercase SHA-256")
    return value


def _collector_command_template(
    config: parity.FixedActionProbeConfig,
    collector: Literal["wrapper", "direct"],
) -> tuple[str, ...]:
    actions = ",".join(str(action) for action in config.actions)
    return (
        DOCKER_PATH,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--pids-limit=512",
        "--mount=type=tmpfs,destination=/tmp/src,tmpfs-mode=0555,tmpfs-size=1048576",
        (
            "--mount=type=bind,source="
            + PROBE_SOURCE_PLACEHOLDER
            + ",destination="
            + CONTAINER_PROBE_PATH
            + ",readonly"
        ),
        "--tmpfs=/run/alberta:rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700",
        f"--workdir={parity.REQUIRED_SOURCE_ROOT.as_posix()}",
        "--env=HOME=/run/alberta",
        "--env=JAX_ENABLE_COMPILATION_CACHE=false",
        "--env=JAX_PLATFORM_NAME=cpu",
        "--env=JAX_PLATFORMS=cpu",
        "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        "--env=LD_LIBRARY_PATH=",
        "--env=LD_PRELOAD=",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=PYTHONBREAKPOINT=",
        "--env=PYTHONHASHSEED=0",
        "--env=PYTHONHOME=",
        "--env=PYTHONINSPECT=",
        "--env=PYTHONNOUSERSITE=1",
        "--env=PYTHONPATH=",
        "--env=PYTHONSTARTUP=",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONUSERBASE=",
        "--env=PYTHONUTF8=1",
        "--env=TMPDIR=/run/alberta",
        "--env=TZ=UTC",
        "--env=XDG_CACHE_HOME=/run/alberta",
        parity.REQUIRED_OCI_IMAGE_ID,
        parity.REQUIRED_PYTHON_EXECUTABLE.as_posix(),
        "-I",
        "-B",
        CONTAINER_PROBE_PATH,
        f"--seed={config.seed}",
        f"--actions={actions}",
        f"--collector={collector}",
    )


@dataclass(frozen=True)
class HostQualificationPlan:
    """Frozen public probe and two nonexecuted OCI command templates."""

    probe_source_sha256: str
    config: parity.FixedActionProbeConfig = OPEN_QUALIFICATION_CONFIG

    def __post_init__(self) -> None:
        _require_sha256(self.probe_source_sha256, "probe_source_sha256")
        if type(self.config) is not parity.FixedActionProbeConfig:
            raise ForagerRngParityQualificationError("config must be a FixedActionProbeConfig")
        if self.config != OPEN_QUALIFICATION_CONFIG:
            raise ForagerRngParityQualificationError(
                "host qualification must use the frozen public probe"
            )

    @property
    def wrapper_command_template(self) -> tuple[str, ...]:
        return _collector_command_template(self.config, "wrapper")

    @property
    def direct_command_template(self) -> tuple[str, ...]:
        return _collector_command_template(self.config, "direct")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_QUALIFICATION_PLAN_SCHEMA,
            "classification": HOST_QUALIFICATION_CLASSIFICATION,
            "seed_class": "public_open_qualification",
            "promotion_authorized": False,
            "live_docker_execution_deferred": LIVE_DOCKER_EXECUTION_DEFERRED,
            "required_oci_image_id": parity.REQUIRED_OCI_IMAGE_ID,
            "probe_source_sha256": self.probe_source_sha256,
            "probe": self.config.to_dict(),
            "process_model": "two_distinct_ephemeral_oci_processes",
            "wrapper_command_template": list(self.wrapper_command_template),
            "direct_command_template": list(self.direct_command_template),
        }

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["plan_sha256"] = self.plan_sha256
        return payload

    @property
    def canonical_bytes(self) -> bytes:
        return parity.canonical_json_bytes(self.to_dict())


def build_host_qualification_plan(probe_source: Path) -> HostQualificationPlan:
    """Bind a stable non-symlink local probe file without launching Docker."""
    try:
        digest, _contents = parity._read_verified_file(
            probe_source,
            label="host parity probe source",
            require_read_only=False,
        )
    except parity.ForagerRngParityError as exc:
        raise ForagerRngParityQualificationError(str(exc)) from exc
    return HostQualificationPlan(probe_source_sha256=digest)


def materialize_collector_commands(
    plan: HostQualificationPlan,
    probe_source: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Substitute one reverified host source path into both command templates."""
    if type(plan) is not HostQualificationPlan:
        raise ForagerRngParityQualificationError("plan must be a HostQualificationPlan")
    rebuilt = build_host_qualification_plan(probe_source)
    if rebuilt != plan:
        raise ForagerRngParityQualificationError("probe source differs from the frozen plan")
    source = probe_source.as_posix()
    if any(character in source for character in (",", "\n", "\r", "\x00")):
        raise ForagerRngParityQualificationError(
            "probe source path cannot be represented by the OCI bind contract"
        )

    def materialize(template: tuple[str, ...]) -> tuple[str, ...]:
        command = tuple(argument.replace(PROBE_SOURCE_PLACEHOLDER, source) for argument in template)
        if any(PROBE_SOURCE_PLACEHOLDER in argument for argument in command):
            raise ForagerRngParityQualificationError("collector command was not materialized")
        return command

    return materialize(plan.wrapper_command_template), materialize(plan.direct_command_template)


@dataclass(frozen=True)
class HostQualificationReceipt:
    """Hash-bound collector pair awaiting external executor-receipt authentication."""

    plan_sha256: str
    executor_qualification_receipt_sha256: str
    wrapper_collector_sha256: str
    direct_collector_sha256: str
    result: parity.ParityProbeResult

    def __post_init__(self) -> None:
        _require_sha256(self.plan_sha256, "plan_sha256")
        _require_sha256(
            self.executor_qualification_receipt_sha256,
            "executor_qualification_receipt_sha256",
        )
        _require_sha256(self.wrapper_collector_sha256, "wrapper_collector_sha256")
        _require_sha256(self.direct_collector_sha256, "direct_collector_sha256")
        if self.wrapper_collector_sha256 == self.direct_collector_sha256:
            raise ForagerRngParityQualificationError(
                "distinct collector payloads must have distinct identities"
            )
        if type(self.result) is not parity.ParityProbeResult:
            raise ForagerRngParityQualificationError("result must be a ParityProbeResult")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_QUALIFICATION_RECEIPT_SCHEMA,
            "classification": HOST_QUALIFICATION_CLASSIFICATION,
            "status": "content_complete_external_executor_receipt_unverified",
            "promotion_authorized": False,
            "external_executor_receipt_requires_trust_resolver": True,
            "required_oci_image_id": parity.REQUIRED_OCI_IMAGE_ID,
            "plan_sha256": self.plan_sha256,
            "executor_qualification_receipt_sha256": (self.executor_qualification_receipt_sha256),
            "wrapper_collector_sha256": self.wrapper_collector_sha256,
            "direct_collector_sha256": self.direct_collector_sha256,
            "result": self.result.to_dict(),
        }

    @property
    def payload_sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["payload_sha256"] = self.payload_sha256
        return payload

    @property
    def canonical_bytes(self) -> bytes:
        return parity.canonical_json_bytes(self.to_dict())


def qualify_collector_outputs(
    plan: HostQualificationPlan,
    wrapper_output: Mapping[str, Any] | bytes | str,
    direct_output: Mapping[str, Any] | bytes | str,
    *,
    executor_qualification_receipt_sha256: str,
) -> HostQualificationReceipt:
    """Validate and combine outputs expected from the two frozen OCI commands."""
    if type(plan) is not HostQualificationPlan:
        raise ForagerRngParityQualificationError("plan must be a HostQualificationPlan")
    executor_digest = _require_sha256(
        executor_qualification_receipt_sha256,
        "executor_qualification_receipt_sha256",
    )
    try:
        wrapper = parity.validate_collector_result(wrapper_output)
        direct = parity.validate_collector_result(direct_output)
    except parity.ForagerRngParityError as exc:
        raise ForagerRngParityQualificationError(str(exc)) from exc
    if wrapper.collector != "wrapper" or direct.collector != "direct":
        raise ForagerRngParityQualificationError("collector roles do not match the frozen commands")
    if wrapper.config != plan.config or direct.config != plan.config:
        raise ForagerRngParityQualificationError("collector probe differs from the frozen plan")
    if (
        wrapper.runtime.probe_module_sha256 != plan.probe_source_sha256
        or direct.runtime.probe_module_sha256 != plan.probe_source_sha256
    ):
        raise ForagerRngParityQualificationError("collector probe source differs from the plan")
    try:
        result = parity.compare_collector_results(wrapper, direct)
    except parity.ForagerRngParityError as exc:
        raise ForagerRngParityQualificationError(str(exc)) from exc
    return HostQualificationReceipt(
        plan_sha256=plan.plan_sha256,
        executor_qualification_receipt_sha256=executor_digest,
        wrapper_collector_sha256=wrapper.payload_sha256,
        direct_collector_sha256=direct.payload_sha256,
        result=result,
    )


def validate_host_qualification_receipt(
    value: Mapping[str, Any] | bytes | str,
    plan: HostQualificationPlan,
    wrapper_output: Mapping[str, Any] | bytes | str,
    direct_output: Mapping[str, Any] | bytes | str,
    *,
    expected_executor_qualification_receipt_sha256: str,
) -> HostQualificationReceipt:
    """Replay a receipt from its caller-held plan, collectors, and executor identity."""
    try:
        decoded = (
            parity.decode_strict_json(value)
            if type(value) in (bytes, str)
            else parity.decode_strict_json(parity.canonical_json_bytes(value))
        )
    except parity.ForagerRngParityError as exc:
        raise ForagerRngParityQualificationError(str(exc)) from exc
    if type(decoded) is not dict:
        raise ForagerRngParityQualificationError("qualification receipt must be an object")
    expected = qualify_collector_outputs(
        plan,
        wrapper_output,
        direct_output,
        executor_qualification_receipt_sha256=(expected_executor_qualification_receipt_sha256),
    )
    if parity.canonical_json_bytes(cast(dict[str, Any], decoded)) != expected.canonical_bytes:
        raise ForagerRngParityQualificationError(
            "qualification receipt does not replay from caller-held inputs"
        )
    return expected


__all__ = [
    "CONTAINER_PROBE_PATH",
    "DOCKER_PATH",
    "ForagerRngParityQualificationError",
    "HOST_QUALIFICATION_CLASSIFICATION",
    "HOST_QUALIFICATION_PLAN_SCHEMA",
    "HOST_QUALIFICATION_RECEIPT_SCHEMA",
    "HostQualificationPlan",
    "HostQualificationReceipt",
    "LIVE_DOCKER_EXECUTION_DEFERRED",
    "OPEN_QUALIFICATION_CONFIG",
    "PROBE_SOURCE_PLACEHOLDER",
    "build_host_qualification_plan",
    "materialize_collector_commands",
    "qualify_collector_outputs",
    "validate_host_qualification_receipt",
]
