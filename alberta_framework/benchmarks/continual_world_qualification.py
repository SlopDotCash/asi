"""Strict setup receipts for an isolated Continual World CW20 smoke lane.

This module deliberately does not import the legacy TensorFlow/Gym/MuJoCo
stack.  A separately built immutable runtime executes the official benchmark;
the host validates its fixed-action trace before any learning comparison is
allowed.  The smoke lane is permanently nonpromoting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import operator
from typing import SupportsIndex, cast

import numpy as np

SCHEMA = "asi.continual-world.fixed-action-smoke.v1"
PAPER_REVISION = "arXiv:2105.10919v3"
OFFICIAL_COMMIT = "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"
METAWORLD_COMMIT = "0875192baaa91c43523708f55866d98eaf3facaf"
OFFICIAL_SETUP_BLOB = "125705947e810f8a41e7f9560d429d444c06694f"
OFFICIAL_DOCKERFILE_BLOB = "6282da996f68e58549f3fcf15ba5d326b1f5ef50"
OFFICIAL_TREE_SHA256 = "514f88bbf29ad64a4fbbc1bc403b6a4eca540e9f"
OFFICIAL_ARCHIVE_SHA256 = "21ed7d404975be7ca12fbb315eeece14f25cc0580dc6b074a81eac791a2d03d9"
OFFICIAL_LICENSE_STATUS = "blocked-no-license-file"
METAWORLD_TREE_SHA256 = "3024851b45599fe678718b92156d6e004c17039c"
METAWORLD_ARCHIVE_SHA256 = "fa1f0336719ef8110c7c22c411ace52c7936deacddce0a1f011ebde3989ec5a5"
METAWORLD_LICENSE_SHA256 = "9d4c6640ecd8cb9e3fe55eb923517fb75a241b74949817121399260c8f549243"
OBSERVED_MUJOCO_ARCHIVE_SHA256 = (
    "ba8560040f6ca47dbd89e4731bc9e06080a99eba4583cda95cdedca802389153"
)
OBSERVED_MUJOCO_KEY_SHA256 = "bffe403bce6978d329239c83e874e0fd412740d149834b8c051689ba4a9adecc"
LICENSE_DISPOSITION_APPROVED = False
EXTERNAL_EXECUTION_AUTHORIZED = False
FROZEN_DEVELOPMENT_SEED = 1_580_000
FROZEN_STEPS_PER_TASK = 2
CW10_TASKS = (
    "hammer-v1",
    "push-wall-v1",
    "faucet-close-v1",
    "push-back-v1",
    "stick-pull-v1",
    "handle-press-side-v1",
    "push-v1",
    "shelf-place-v1",
    "window-close-v1",
    "peg-unplug-side-v1",
)
CW20_TASKS = CW10_TASKS + CW10_TASKS
FIXED_ACTION = (0.0, 0.0, 0.0, 0.0)
_MAX_BYTES = 1 << 30
_INT32_MAX = 2**31 - 1


def _int(value: object, *, name: str, minimum: int = 0, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} lies outside the bounded protocol")
    return result


def _digest(value: object, *, name: str, prefixed: bool = True) -> str:
    prefix = "sha256:" if prefixed else ""
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
        or any(character not in "0123456789abcdef" for character in value[len(prefix) :])
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 identity")
    return value


def _canonical(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("payload must be finite canonical JSON") from error
    if len(encoded) > 1 << 20:
        raise ValueError("payload exceeds the one-MiB receipt ceiling")
    return encoded


@dataclasses.dataclass(frozen=True, slots=True)
class IsolatedRuntimeIdentity:
    """Content identity supplied only after the legacy runtime is built."""

    image_digest: str
    mujoco_archive_sha256: str
    python_version: str
    tensorflow_version: str
    mujoco_py_version: str
    gym_version: str
    numpy_version: str
    platform: str

    def __post_init__(self) -> None:
        _digest(self.image_digest, name="image_digest")
        _digest(self.mujoco_archive_sha256, name="mujoco_archive_sha256", prefixed=False)
        for name in (
            "python_version",
            "tensorflow_version",
            "mujoco_py_version",
            "gym_version",
            "numpy_version",
            "platform",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode("utf-8")) > 128:
                raise ValueError(f"{name} must be bounded exact text")


@dataclasses.dataclass(frozen=True, slots=True)
class ContinualWorldSmokePlan:
    """Frozen workload plus one immutable, externally built runtime identity."""

    runtime: IsolatedRuntimeIdentity
    seed: int = FROZEN_DEVELOPMENT_SEED
    steps_per_task: int = FROZEN_STEPS_PER_TASK

    def __post_init__(self) -> None:
        if type(self.runtime) is not IsolatedRuntimeIdentity:
            raise ValueError("runtime must be an exact IsolatedRuntimeIdentity")
        if _int(self.seed, name="seed") != FROZEN_DEVELOPMENT_SEED:
            raise ValueError("development seed is frozen")
        if _int(self.steps_per_task, name="steps_per_task", minimum=1, maximum=200) != 2:
            raise ValueError("smoke steps_per_task is frozen")

    def payload(self) -> dict[str, object]:
        return {
            "paper_revision": PAPER_REVISION,
            "official_commit": OFFICIAL_COMMIT,
            "metaworld_commit": METAWORLD_COMMIT,
            "official_setup_blob": OFFICIAL_SETUP_BLOB,
            "official_dockerfile_blob": OFFICIAL_DOCKERFILE_BLOB,
            "source_audit": {
                "official_tree_sha256": OFFICIAL_TREE_SHA256,
                "official_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
                "official_license_status": OFFICIAL_LICENSE_STATUS,
                "metaworld_tree_sha256": METAWORLD_TREE_SHA256,
                "metaworld_archive_sha256": METAWORLD_ARCHIVE_SHA256,
                "metaworld_license_sha256": METAWORLD_LICENSE_SHA256,
                "mujoco_archive_sha256": OBSERVED_MUJOCO_ARCHIVE_SHA256,
                "mujoco_key_sha256": OBSERVED_MUJOCO_KEY_SHA256,
                "official_runtime_reconstructible": False,
            },
            "license_disposition_approved": LICENSE_DISPOSITION_APPROVED,
            "external_execution_authorized": EXTERNAL_EXECUTION_AUTHORIZED,
            "runtime": dataclasses.asdict(self.runtime),
            "seed": self.seed,
            "tasks": list(CW20_TASKS),
            "steps_per_task": self.steps_per_task,
            "fixed_action": list(FIXED_ACTION),
            "allowed_boundary_information": ["evaluator_task_index", "evaluator_task_boundary"],
            "allowed_task_information": ["evaluator_task_name"],
            "learner_boundary_information": [],
            "learner_task_information": [],
            "development_only": True,
            "scientific_promotion_allowed": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.payload())).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ContinualWorldSmokeReceipt:
    """Bounded result of one official-runtime, fixed-action CW20 smoke trace."""

    schema: str
    plan: ContinualWorldSmokePlan
    plan_sha256: str
    action_sha256: str
    observation_sha256: str
    reward_sha256: str
    success_sha256: str
    task_index_sha256: str
    environment_steps: int
    data_steps: int
    learner_updates: int
    model_queries: int
    persistent_mechanism_bytes: int
    persistent_environment_numeric_bytes: int
    timing_ns: int
    timing_is_telemetry_only: bool
    mechanism_off: bool
    outcome: str
    negative_outcome_retained: bool
    development_only: bool
    scientific_promotion_allowed: bool

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("unsupported Continual World receipt schema")
        if type(self.plan) is not ContinualWorldSmokePlan:
            raise ValueError("plan must be an exact ContinualWorldSmokePlan")
        if _digest(self.plan_sha256, name="plan_sha256", prefixed=False) != self.plan.sha256:
            raise ValueError("plan_sha256 does not bind the exact plan")
        for name in (
            "action_sha256",
            "observation_sha256",
            "reward_sha256",
            "success_sha256",
            "task_index_sha256",
        ):
            _digest(getattr(self, name), name=name, prefixed=False)
        horizon = len(CW20_TASKS) * self.plan.steps_per_task
        if _int(self.environment_steps, name="environment_steps", minimum=1) != horizon:
            raise ValueError("environment_steps do not match the frozen CW20 smoke horizon")
        exact_zero = ("data_steps", "learner_updates", "model_queries")
        if any(_int(getattr(self, name), name=name) != 0 for name in exact_zero):
            raise ValueError("fixed-action mechanism-off cannot train or query a model")
        expected_mechanism_bytes = len(FIXED_ACTION) * np.dtype(np.float32).itemsize
        if (
            _int(self.persistent_mechanism_bytes, name="persistent_mechanism_bytes")
            != expected_mechanism_bytes
        ):
            raise ValueError("persistent_mechanism_bytes must equal the fixed float32 action")
        _int(
            self.persistent_environment_numeric_bytes,
            name="persistent_environment_numeric_bytes",
            minimum=1,
            maximum=_MAX_BYTES,
        )
        _int(self.timing_ns, name="timing_ns", maximum=2**63 - 1)
        if self.timing_is_telemetry_only is not True or self.mechanism_off is not True:
            raise ValueError("smoke timing/mechanism policy drifted")
        if type(self.outcome) is not str or self.outcome not in {
            "supported",
            "rejected",
            "inconclusive",
        }:
            raise ValueError("outcome must use the frozen retention vocabulary")
        if (
            self.negative_outcome_retained is not True
            or self.development_only is not True
            or self.scientific_promotion_allowed is not False
        ):
            raise ValueError("Continual World smoke receipts are permanently nonpromoting")

    def payload(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(json.dumps(dataclasses.asdict(self))))


def _array_hash(domain: bytes, value: np.ndarray) -> str:
    digest = hashlib.sha256(domain)
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_smoke_receipt(
    plan: ContinualWorldSmokePlan,
    *,
    actions: np.ndarray,
    observations: np.ndarray,
    rewards: np.ndarray,
    successes: np.ndarray,
    task_indices: np.ndarray,
    persistent_environment_numeric_bytes: int,
    timing_ns: int,
    outcome: str,
) -> ContinualWorldSmokeReceipt:
    """Snapshot one external official-runtime trace into a strict host receipt."""
    if LICENSE_DISPOSITION_APPROVED is not True or EXTERNAL_EXECUTION_AUTHORIZED is not True:
        raise PermissionError(
            "Continual World license disposition and external execution authorization "
            "must both be approved before trace admission"
        )
    if type(plan) is not ContinualWorldSmokePlan:
        raise ValueError("plan must be exact")
    horizon = len(CW20_TASKS) * plan.steps_per_task
    arrays = (
        ("actions", actions, np.dtype(np.float32), (horizon, 4)),
        ("observations", observations, np.dtype(np.float32), (horizon, 32)),
        ("rewards", rewards, np.dtype(np.float32), (horizon,)),
        ("successes", successes, np.dtype(np.bool_), (horizon,)),
        ("task_indices", task_indices, np.dtype(np.int32), (horizon,)),
    )
    snapshots: dict[str, np.ndarray] = {}
    for name, value, dtype, shape in arrays:
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name} must be an exact {dtype} array of shape {shape}")
        snapshot = value.copy()
        if dtype.kind == "f" and not np.isfinite(snapshot).all():
            raise ValueError(f"{name} must be finite")
        snapshots[name] = snapshot
    expected_actions = np.zeros((horizon, 4), dtype=np.float32)
    if not np.array_equal(snapshots["actions"], expected_actions):
        raise ValueError("actions do not match the exact fixed-action mechanism-off schedule")
    expected_indices = np.repeat(np.arange(20, dtype=np.int32), plan.steps_per_task)
    if not np.array_equal(snapshots["task_indices"], expected_indices):
        raise ValueError("task_indices do not follow the exact CW20 boundary schedule")
    return ContinualWorldSmokeReceipt(
        schema=SCHEMA,
        plan=plan,
        plan_sha256=plan.sha256,
        action_sha256=_array_hash(b"asi.cw.actions.v1\0", snapshots["actions"]),
        observation_sha256=_array_hash(b"asi.cw.observations.v1\0", snapshots["observations"]),
        reward_sha256=_array_hash(b"asi.cw.rewards.v1\0", snapshots["rewards"]),
        success_sha256=_array_hash(b"asi.cw.successes.v1\0", snapshots["successes"]),
        task_index_sha256=_array_hash(b"asi.cw.tasks.v1\0", snapshots["task_indices"]),
        environment_steps=horizon,
        data_steps=0,
        learner_updates=0,
        model_queries=0,
        persistent_mechanism_bytes=16,
        persistent_environment_numeric_bytes=persistent_environment_numeric_bytes,
        timing_ns=timing_ns,
        timing_is_telemetry_only=True,
        mechanism_off=True,
        outcome=outcome,
        negative_outcome_retained=True,
        development_only=True,
        scientific_promotion_allowed=False,
    )


def validate_smoke_payload(payload: object) -> ContinualWorldSmokeReceipt:
    """Reject expanded or hostile serialized smoke receipts."""
    if type(payload) is not dict:
        raise ValueError("payload must be an exact object")
    root = cast(dict[object, object], payload)
    fields = {field.name for field in dataclasses.fields(ContinualWorldSmokeReceipt)}
    if any(type(key) is not str for key in root) or set(root) != fields:
        raise ValueError("payload fields differ from the exact schema")
    plan_raw = root["plan"]
    if type(plan_raw) is not dict:
        raise ValueError("serialized plan must be an exact object")
    plan_dict = cast(dict[object, object], plan_raw)
    expected_plan_fields = {field.name for field in dataclasses.fields(ContinualWorldSmokePlan)}
    if (
        any(type(key) is not str for key in plan_dict)
        or set(plan_dict) != expected_plan_fields
    ):
        raise ValueError("serialized plan fields differ from the exact schema")
    runtime_raw = plan_dict["runtime"]
    if type(runtime_raw) is not dict:
        raise ValueError("serialized runtime must be an exact object")
    runtime_dict = cast(dict[object, object], runtime_raw)
    runtime_fields = {field.name for field in dataclasses.fields(IsolatedRuntimeIdentity)}
    if any(type(key) is not str for key in runtime_dict) or set(runtime_dict) != runtime_fields:
        raise ValueError("serialized runtime fields differ from the exact schema")
    runtime = IsolatedRuntimeIdentity(**cast(dict[str, str], runtime_dict))
    plan = ContinualWorldSmokePlan(
        runtime=runtime,
        seed=plan_dict["seed"],  # type: ignore[arg-type]
        steps_per_task=plan_dict["steps_per_task"],  # type: ignore[arg-type]
    )
    if plan_dict != dataclasses.asdict(plan):
        raise ValueError("serialized plan differs from the frozen protocol")
    kwargs = {cast(str, key): value for key, value in root.items() if key != "plan"}
    return ContinualWorldSmokeReceipt(plan=plan, **kwargs)  # type: ignore[arg-type]


def protocol_gap_record() -> tuple[str, ...]:
    """Material gaps that keep the smoke receipt from becoming a paper comparison."""
    return (
        "the official runtime recipe leaves most transitive package versions and "
        "archive bytes unpinned",
        "the official stack uses legacy mujoco-py, Gym, TensorFlow, and Meta-World v1",
        "the smoke uses two fixed-action steps per task, not one million training steps per task",
        "no SAC learner, replay buffer, multihead network, or continual-learning method "
        "is executed",
        "paper metrics require success evaluations, single-task references, forgetting, "
        "and forward transfer",
        "paper results use twenty seeds and 90% bootstrap confidence intervals",
        "environment numeric bytes exclude interpreter, simulator native heap, and renderer "
        "allocations",
        "timing is telemetry only and the consistency hashes are not execution attestation",
    )
