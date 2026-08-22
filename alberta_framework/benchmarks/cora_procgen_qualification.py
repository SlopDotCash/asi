"""Fail-closed qualification plan for CORA's external Procgen family.

This module imports and executes no CORA or Procgen code.  It binds audited
source and wheel identities and defines the create-only receipt accepted after
a separately authorized isolated-runtime check.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Literal

SCHEMA = "asi.cora_procgen_create_qualification.development.v1"
CORA_PROCGEN_TASKS = (
    "climber-v0",
    "dodgeball-v0",
    "ninja-v0",
    "starpilot-v0",
    "bigfish-v0",
    "fruitbot-v0",
)


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ProcgenQualificationPlan:
    issue: int = 1581
    paper_revision: str = "arXiv:2110.10067v2"
    cora_repository: str = "https://github.com/AGI-Labs/continual_rl.git"
    cora_commit: str = "f2754bb282757829765beb4703f24b87efa13ff9"
    cora_tree: str = "3c296057e717401053ce0acfe362adeef395aede"
    cora_archive_sha256: str = (
        "d634325bd7cc450e68ee55fd5b83118fa4b8d11c0e5e6284daa6bff0a60436db"
    )
    cora_license_sha256: str = (
        "37df918c349040efba06271ed929ffd623506ef2d4a0a7e4ce46e7749ba0cad7"
    )
    procgen_repository: str = "https://github.com/openai/procgen.git"
    procgen_version: str = "0.10.7"
    procgen_commit: str = "5e1dbf341d291eff40d1f9e0c0a0d5003643aebf"
    procgen_tree: str = "0cb587203bb4d55e001283ad6550f6bc1ef95ad4"
    procgen_archive_sha256: str = (
        "22940ad0f1fdb4ad1eab3303ce23d3a0ea536700bb1d7c299bee64dbc7c57e9b"
    )
    procgen_wheel_filename: str = (
        "procgen-0.10.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    procgen_wheel_sha256: str = (
        "4b594d14e42f0f2166e59a9e294477b906eb99c90c3343b570b7124cbc865f53"
    )
    procgen_license_sha256: str = (
        "62fead824b483f3dbba9ffc2074d8ffb5420839481dfa883b91de6f5ae68bc45"
    )
    procgen_asset_licenses_sha256: str = (
        "2d8857c3fe4252eb19eaf1187274a0aa4ea8c4ca0bea389268f7b118c0c72141"
    )
    platform_tag: str = "manylinux2014_x86_64"
    python_version: str = "3.10"
    compatibility_matrix: tuple[str, ...] = (
        "CPython==3.10.x",
        "torch==1.13.1+cpu (proposal; artifact unbound)",
        "torchvision==0.14.1+cpu (proposal; artifact unbound)",
        "gym==0.25.2 (official CORA upper bound; artifact unbound)",
        "procgen==0.10.7 (Linux wheel bound above)",
    )
    tasks: tuple[str, ...] = CORA_PROCGEN_TASKS
    cycles: int = 5
    train_steps_per_task: int = 5_000_000
    train_levels: int = 200
    eval_levels: int = 0
    start_level: int = 0
    distribution_mode: str = "easy"
    official_seed_semantics: str = "os.urandom-derived seed_to_set"
    compatibility_seed_semantics: str = "caller-injected exact uint32 seed_to_set"
    compatibility_smoke_has_official_seed_parity: bool = False
    dependency_lock_complete: bool = False
    runtime_qualified: bool = False
    execution_authorized: bool = False
    promotion_policy: str = "permanently_nonpromoting_qualification"

    def validate(self) -> None:
        if type(self.issue) is not int or self.issue != 1581:
            raise ValueError("issue identity drift")
        expected = ProcgenQualificationPlan()
        if self != expected:
            raise ValueError("Procgen qualification plan differs from the audited registry")
        for name in (
            "cora_archive_sha256",
            "cora_license_sha256",
            "procgen_archive_sha256",
            "procgen_wheel_sha256",
            "procgen_license_sha256",
            "procgen_asset_licenses_sha256",
        ):
            _digest(getattr(self, name), name)
        if (
            type(self.compatibility_matrix) is not tuple
            or type(self.tasks) is not tuple
            or self.tasks != CORA_PROCGEN_TASKS
            or self.dependency_lock_complete is not False
            or self.runtime_qualified is not False
            or self.execution_authorized is not False
            or self.compatibility_smoke_has_official_seed_parity is not False
        ):
            raise ValueError("external runtime gates must remain closed")


def plan_sha256() -> str:
    plan = ProcgenQualificationPlan()
    plan.validate()
    payload = json.dumps(
        dataclasses.asdict(plan), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


Outcome = Literal["success", "failure"]
Phase = Literal["train", "evaluation"]
SeedMode = Literal["official_os_random", "deterministic_compatibility"]


@dataclasses.dataclass(frozen=True, slots=True)
class ProcgenCreateReceipt:
    plan_sha256: str
    outcome: Outcome
    task_id: str
    phase: Phase
    seed_mode: SeedMode
    injected_seed: int | None = None
    observation_sha256: str | None = None
    reward_payload: None = None
    termination_payload: None = None
    failure_kind: str | None = None
    environment_instances_created: int = 0
    environment_steps: int = 0
    policy_queries: int = 0
    model_queries: int = 0
    elapsed_ns: int = 0
    timing_telemetry_only: bool = True
    schema: str = SCHEMA
    development_only: bool = True
    scientific_promotion_allowed: bool = False


def validate_create_receipt(value: object) -> ProcgenCreateReceipt:
    if type(value) is not ProcgenCreateReceipt:
        raise ValueError("receipt must be an exact ProcgenCreateReceipt")
    receipt = value
    if receipt.schema != SCHEMA or receipt.plan_sha256 != plan_sha256():
        raise ValueError("receipt plan identity mismatch")
    if (
        type(receipt.task_id) is not str
        or receipt.task_id not in CORA_PROCGEN_TASKS
        or type(receipt.phase) is not str
        or receipt.phase not in ("train", "evaluation")
        or type(receipt.outcome) is not str
        or receipt.outcome not in ("success", "failure")
        or type(receipt.seed_mode) is not str
        or receipt.seed_mode not in ("official_os_random", "deterministic_compatibility")
    ):
        raise ValueError("receipt categorical identity mismatch")
    if receipt.seed_mode == "official_os_random":
        if receipt.injected_seed is not None:
            raise ValueError("official seed mode cannot carry an injected seed")
    elif (
        type(receipt.injected_seed) is not int
        or not 0 <= receipt.injected_seed <= 2**32 - 1
    ):
        raise ValueError("deterministic seed mode requires an exact uint32 seed")
    counters = (
        receipt.environment_instances_created,
        receipt.environment_steps,
        receipt.policy_queries,
        receipt.model_queries,
        receipt.elapsed_ns,
    )
    if any(type(counter) is not int or not 0 <= counter <= 2**63 - 1 for counter in counters):
        raise ValueError("receipt counters must be nonnegative signed-int64 integers")
    if (
        receipt.environment_steps != 0
        or receipt.policy_queries != 0
        or receipt.model_queries != 0
        or receipt.reward_payload is not None
        or receipt.termination_payload is not None
    ):
        raise ValueError("qualification receipt must remain create-only")
    if receipt.outcome == "success":
        if (
            receipt.environment_instances_created != 1
            or receipt.observation_sha256 is None
            or receipt.failure_kind is not None
        ):
            raise ValueError("success receipt must contain one observation and no failure")
        _digest(receipt.observation_sha256, "observation_sha256")
    else:
        failure_kind = receipt.failure_kind
        if type(failure_kind) is not str or not failure_kind:
            raise ValueError("failure receipt must retain one bounded failure kind")
        try:
            failure_bytes = failure_kind.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("failure kind must be valid bounded UTF-8") from error
        if (
            receipt.environment_instances_created != 0
            or receipt.observation_sha256 is not None
            or len(failure_bytes) > 128
        ):
            raise ValueError("failure receipt must retain one bounded failure kind")
    if (
        receipt.timing_telemetry_only is not True
        or receipt.development_only is not True
        or receipt.scientific_promotion_allowed is not False
    ):
        raise ValueError("receipt must remain telemetry-only and nonpromoting")
    return receipt


__all__ = [
    "CORA_PROCGEN_TASKS",
    "SCHEMA",
    "ProcgenCreateReceipt",
    "ProcgenQualificationPlan",
    "plan_sha256",
    "validate_create_receipt",
]
