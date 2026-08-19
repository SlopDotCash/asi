"""Bounded Normalize-and-Project comparator on the current hidden MNIST lane.

This permanently nonpromoting development comparator reuses the data schedule
and hidden-network state introduced by issue #1583.  It is not the paper's
random-label CIFAR-10 or sequential-ALE protocol and makes no parity claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import operator
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks.plasticity_comparators import nap_project
from alberta_framework.benchmarks.plasticity_diagnostics import (
    INPUT_DIM,
    N_CLASSES,
    PROFILES,
    DiagnosticProfile,
    MLPState,
    _arrays,
    _dataset_sha,
    _effective_rank,
    _forward,
    _init_state,
    _schedule,
    _state_sha256,
    _step,
)

SCHEMA = "asi.nap_ipmnist_comparator.development.v1"
PAPER_REVISION = "arXiv:2407.01800v1"
PAPER_IDENTITY = "NeurIPS-2024:c04d37be05ba74419d2d5705972a9d64"
DEPENDENCY_COMMIT = "8383d6438b81c7620189c6fedba30c345994cb12"
PLASTICINE_REPOSITORY = "https://github.com/RLE-Foundation/Plasticine.git"
PLASTICINE_COMMIT = "aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8"
FROZEN_SEEDS = (15_640, 15_641, 15_642, 15_643)
ARM_IDS = (
    "sgd_current_control",
    "nap_mechanism_off",
    "normalization_only",
    "projection_only",
    "nap",
)
ArmID = Literal[
    "sgd_current_control",
    "nap_mechanism_off",
    "normalization_only",
    "projection_only",
    "nap",
]
_NORMALIZATION_EPSILON = 1e-5


def _runtime_identity() -> tuple[str, str, str, str]:
    return (platform.python_version(), jax.__version__, np.__version__, jax.default_backend())


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


def _digest(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclasses.dataclass(frozen=True, slots=True)
class NaPCatalogEntry:
    issue: int = 1564
    paper_revision: str = PAPER_REVISION
    final_paper_identity: str = PAPER_IDENTITY
    official_nap_code_available: bool = False
    official_nap_code_status: str = (
        "paper cites DQN Zoo as its Rainbow baseline; no method-specific official NaP "
        "repository is disclosed in the final paper or was located in the primary-source audit"
    )
    disclosed_baseline_repository: str = "https://github.com/deepmind/dqn_zoo"
    disclosed_baseline_revision: None = None
    secondary_implementation_repository: str = PLASTICINE_REPOSITORY
    secondary_implementation_commit: str = PLASTICINE_COMMIT
    secondary_implementation_official: bool = False
    asi_dependency_commit: str = DEPENDENCY_COMMIT
    protocol_differences: tuple[str, ...] = (
        "cumulative input-permuted MNIST rather than 20M-step CIFAR-10 random labels",
        "two width-bounded MLP hidden layers rather than the paper CNN/VGG or Rainbow",
        "per-example SGD rather than the paper Adam supervised and Rainbow optimizers",
        "fixed unlearned LayerNorm without paper scale/offset parameters",
        "biases retained and unprojected although the paper removes redundant biases",
        "hidden weights projected every update; final output is not projected",
        "no paper global/per-layer effective-learning-rate schedule matching",
    )
    development_only: bool = True
    scientific_promotion_allowed: bool = False

    def validate(self) -> None:
        if (
            type(self.issue) is not int
            or self.issue != 1564
            or type(self.paper_revision) is not str
            or self.paper_revision != PAPER_REVISION
            or type(self.final_paper_identity) is not str
            or self.final_paper_identity != PAPER_IDENTITY
        ):
            raise ValueError("NaP paper/catalog identity drift")
        if type(self.official_nap_code_available) is not bool or self.official_nap_code_available:
            raise ValueError("official NaP code must remain explicitly absent")
        for name in (
            "official_nap_code_status",
            "disclosed_baseline_repository",
            "secondary_implementation_repository",
            "secondary_implementation_commit",
            "asi_dependency_commit",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode()) > 1024:
                raise ValueError(f"{name} must be bounded exact text")
        if self.disclosed_baseline_revision is not None:
            raise ValueError("paper-disclosed baseline must not acquire an invented revision")
        if (
            self.secondary_implementation_repository != PLASTICINE_REPOSITORY
            or self.secondary_implementation_commit != PLASTICINE_COMMIT
            or type(self.secondary_implementation_official) is not bool
            or self.secondary_implementation_official
            or self.asi_dependency_commit != DEPENDENCY_COMMIT
        ):
            raise ValueError("secondary/dependency provenance drift")
        if (
            type(self.protocol_differences) is not tuple
            or len(self.protocol_differences) != 7
            or any(type(value) is not str or not value for value in self.protocol_differences)
        ):
            raise ValueError("paper protocol differences must remain explicit")
        if self.development_only is not True or self.scientific_promotion_allowed is not False:
            raise ValueError("NaP comparator must remain permanently nonpromoting")


@dataclasses.dataclass(frozen=True, slots=True)
class NaPReceipt:
    data_steps: int
    observations: int
    data_bytes_read: int
    training_model_queries: int
    diagnostic_model_queries: int
    model_queries: int
    parameter_updates: int
    normalization_queries: int
    normalization_elements: int
    projection_events: int
    projected_tensor_queries: int
    projected_elements: int
    logical_forward_macs: int
    logical_gradient_macs: int
    logical_auxiliary_scalar_ops: int
    state_persistent_bytes: int
    projection_target_persistent_bytes: int
    elapsed_ns: int
    timing_telemetry_only: bool = True

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name != "timing_telemetry_only":
                _exact_int(getattr(self, field.name), field.name, 0, 2**63 - 1)
        if self.data_steps == 0 or self.model_queries == 0 or self.state_persistent_bytes == 0:
            raise ValueError("receipt must describe a non-empty execution")
        if self.timing_telemetry_only is not True:
            raise ValueError("timing must remain telemetry-only")


@dataclasses.dataclass(frozen=True, slots=True)
class NaPArmResult:
    arm_id: str
    normalization_enabled: bool
    projection_enabled: bool
    task_accuracy: tuple[float, ...]
    task_loss: tuple[float, ...]
    dead_unit_fraction: tuple[float, ...]
    effective_rank: tuple[float, ...]
    initial_hidden_norms: tuple[float, float]
    final_hidden_norms: tuple[float, float]
    final_state_sha256: str
    receipt: NaPReceipt

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError("unknown NaP arm")
        if (
            type(self.normalization_enabled) is not bool
            or type(self.projection_enabled) is not bool
        ):
            raise ValueError("mechanism flags must be exact bools")
        expected_flags = {
            "sgd_current_control": (False, False),
            "nap_mechanism_off": (False, False),
            "normalization_only": (True, False),
            "projection_only": (False, True),
            "nap": (True, True),
        }
        if (self.normalization_enabled, self.projection_enabled) != expected_flags[self.arm_id]:
            raise ValueError("NaP arm flag identity drift")
        curves = (
            self.task_accuracy,
            self.task_loss,
            self.dead_unit_fraction,
            self.effective_rank,
        )
        if any(type(curve) is not tuple or not curve for curve in curves):
            raise ValueError("NaP curves must be non-empty exact tuples")
        if len({len(curve) for curve in curves}) != 1:
            raise ValueError("NaP curves must have matching lengths")
        for value in self.task_accuracy + self.dead_unit_fraction:
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("probability curves must contain bounded finite exact floats")
        for value in self.task_loss + self.effective_rank:
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError("loss/rank curves must contain nonnegative finite exact floats")
        for norms in (self.initial_hidden_norms, self.final_hidden_norms):
            if (
                type(norms) is not tuple
                or len(norms) != 2
                or any(
                    type(value) is not float or not math.isfinite(value) or value <= 0
                    for value in norms
                )
            ):
                raise ValueError("hidden norm pairs must be positive finite exact floats")
        _digest(self.final_state_sha256, "final state identity")
        if type(self.receipt) is not NaPReceipt:
            raise ValueError("receipt must be an exact NaPReceipt")


@dataclasses.dataclass(frozen=True, slots=True)
class NaPResult:
    schema: str
    catalog: NaPCatalogEntry
    profile_id: str
    profile: DiagnosticProfile
    seed: int
    dataset_sha256: str
    schedule_sha256: str
    source_sha256: str
    dependency_source_sha256: str
    nap_project_dependency_source_sha256: str
    runtime_identity: tuple[str, str, str, str]
    task_protocol: str
    labels_permuted: bool
    task_boundaries_visible_to_learner: bool
    task_ids_visible_to_learner: bool
    observations_matched_before_causal_divergence: bool
    arms: tuple[NaPArmResult, ...]
    negative_results_must_be_retained: bool = True
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    paper_parity_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("NaP result schema drift")
        if type(self.catalog) is not NaPCatalogEntry:
            raise ValueError("catalog must be exact")
        self.catalog.validate()
        if type(self.profile_id) is not str or self.profile_id not in PROFILES:
            raise ValueError("unknown profile")
        if type(self.profile) is not DiagnosticProfile or self.profile != PROFILES[self.profile_id]:
            raise ValueError("profile payload differs from the immutable registry")
        if type(self.seed) is not int or self.seed not in FROZEN_SEEDS:
            raise ValueError("seed is outside the frozen NaP development schedule")
        for name in (
            "dataset_sha256",
            "schedule_sha256",
            "source_sha256",
            "dependency_source_sha256",
            "nap_project_dependency_source_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.source_sha256 != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("NaP source identity drift")
        dependency_path = Path(__file__).with_name("plasticity_diagnostics.py")
        dependency_sha256 = hashlib.sha256(dependency_path.read_bytes()).hexdigest()
        if self.dependency_source_sha256 != dependency_sha256:
            raise ValueError("#1583 dependency source identity drift")
        nap_project_path = Path(__file__).with_name("plasticity_comparators.py")
        nap_project_sha256 = hashlib.sha256(nap_project_path.read_bytes()).hexdigest()
        if self.nap_project_dependency_source_sha256 != nap_project_sha256:
            raise ValueError("nap_project dependency source identity drift")
        if self.runtime_identity != _runtime_identity():
            raise ValueError("current runtime identity drift")
        if (
            type(self.task_protocol) is not str
            or self.task_protocol != "cumulative-input-permuted-mnist"
            or self.labels_permuted is not False
            or self.task_boundaries_visible_to_learner is not False
            or self.task_ids_visible_to_learner is not False
            or self.observations_matched_before_causal_divergence is not True
        ):
            raise ValueError("task or allowed-information declaration drift")
        if type(self.arms) is not tuple or tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise ValueError("NaP arm roster drift")
        if (
            self.negative_results_must_be_retained is not True
            or self.development_only is not True
            or self.scientific_promotion_allowed is not False
            or self.paper_parity_claimed is not False
        ):
            raise ValueError("NaP result promotion/retention flags drift")


def _normalize(preactivation: Array) -> Array:
    centered = preactivation - jnp.mean(preactivation, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + _NORMALIZATION_EPSILON)


def _forward_arm(state: MLPState, inputs: Array, normalize: Array) -> tuple[Array, Array, Array]:
    first = inputs @ state.w1 + state.b1
    first = jax.lax.cond(normalize, _normalize, lambda value: value, first)
    hidden1 = jax.nn.relu(first)
    second = hidden1 @ state.w2 + state.b2
    second = jax.lax.cond(normalize, _normalize, lambda value: value, second)
    hidden2 = jax.nn.relu(second)
    return hidden2 @ state.w3 + state.b3, hidden1, hidden2


def _loss_arm(
    state: MLPState, inputs: Array, label: Array, normalize: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
    logits, hidden1, hidden2 = _forward_arm(state, inputs, normalize)
    return -jax.nn.log_softmax(logits)[label], (logits, hidden1, hidden2)


@jax.jit
def _nap_step(
    state: MLPState,
    inputs: Array,
    label: Array,
    key: Array,
    learning_rate: Array,
    normalize: Array,
) -> tuple[MLPState, Array, Array]:
    del key  # The matched current lane receives a key, but replacement is disabled.
    (loss, (logits, hidden1, hidden2)), gradients = jax.value_and_grad(
        _loss_arm, has_aux=True, allow_int=True
    )(state, inputs, label, normalize)
    trainable = tuple(
        value - learning_rate * gradient if index < 6 else value
        for index, (value, gradient) in enumerate(zip(state, gradients, strict=True))
    )
    updated = MLPState(*trainable)
    utility1 = 0.99 * state.utility1 + 0.01 * hidden1 * jnp.mean(jnp.abs(state.w2), axis=1)
    utility2 = 0.99 * state.utility2 + 0.01 * hidden2 * jnp.mean(jnp.abs(state.w3), axis=1)
    updated = updated._replace(
        utility1=utility1,
        utility2=utility2,
        age1=state.age1 + 1,
        age2=state.age2 + 1,
    )
    return updated, loss, jnp.argmax(logits)


@functools.partial(jax.jit, static_argnames=("initial_norms", "enabled"))
def _project_hidden(
    state: MLPState, *, initial_norms: tuple[float, float], enabled: bool
) -> MLPState:
    return state._replace(
        w1=nap_project(state.w1, initial_norm=initial_norms[0], enabled=enabled),
        w2=nap_project(state.w2, initial_norm=initial_norms[1], enabled=enabled),
    )


def _schedule_sha(tasks: tuple[tuple[np.ndarray, np.ndarray], ...]) -> str:
    digest = hashlib.sha256(b"asi-nap-cumulative-ipmnist-schedule-v1\0")
    for images, labels in tasks:
        digest.update(np.asarray(images.shape, dtype="<i8").tobytes())
        digest.update(images.astype("<f4", copy=False).tobytes(order="C"))
        digest.update(labels.astype("<i4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _arm_flags(arm_id: ArmID) -> tuple[bool, bool]:
    return {
        "sgd_current_control": (False, False),
        "nap_mechanism_off": (False, False),
        "normalization_only": (True, False),
        "projection_only": (False, True),
        "nap": (True, True),
    }[arm_id]


def _run_arm(
    arm_id: ArmID,
    tasks: tuple[tuple[np.ndarray, np.ndarray], ...],
    profile: DiagnosticProfile,
    seed: int,
) -> NaPArmResult:
    normalization_enabled, projection_enabled = _arm_flags(arm_id)
    key = jr.key(seed)
    key, init_key = jr.split(key)
    state = _init_state(init_key, profile.hidden_width)
    initial_norms = (float(jnp.linalg.norm(state.w1)), float(jnp.linalg.norm(state.w2)))
    accuracies: list[float] = []
    losses: list[float] = []
    dead: list[float] = []
    ranks: list[float] = []
    started = time.perf_counter_ns()
    normalize_array = jnp.asarray(normalization_enabled)
    for inputs, labels in tasks:
        correct = 0
        task_losses: list[float] = []
        for inputs_row, label in zip(inputs, labels, strict=True):
            key, step_key = jr.split(key)
            if arm_id == "sgd_current_control":
                state, loss, prediction, _ = _step(
                    state,
                    jnp.asarray(inputs_row),
                    jnp.asarray(label),
                    step_key,
                    jnp.asarray(profile.learning_rate, dtype=jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.asarray(profile.maturity_threshold, dtype=jnp.int32),
                )
            else:
                state, loss, prediction = _nap_step(
                    state,
                    jnp.asarray(inputs_row),
                    jnp.asarray(label),
                    step_key,
                    jnp.asarray(profile.learning_rate, dtype=jnp.float32),
                    normalize_array,
                )
            state = _project_hidden(
                state, initial_norms=initial_norms, enabled=projection_enabled
            )
            correct += int(prediction == label)
            task_losses.append(float(loss))
        if normalization_enabled:
            _, hidden1, hidden2 = _forward_arm(state, jnp.asarray(inputs), normalize_array)
        else:
            _, hidden1, hidden2 = _forward(state, jnp.asarray(inputs))
        host1, host2 = np.asarray(hidden1), np.asarray(hidden2)
        dead_count = np.sum(np.all(host1 == 0.0, axis=0)) + np.sum(
            np.all(host2 == 0.0, axis=0)
        )
        accuracies.append(float(correct / len(labels)))
        losses.append(float(np.mean(task_losses)))
        dead.append(float(dead_count / (2 * profile.hidden_width)))
        ranks.append(_effective_rank(host2))
    elapsed_ns = time.perf_counter_ns() - started
    steps = profile.n_tasks * profile.examples_per_task
    training_model_queries = steps * 2
    diagnostic_model_queries = steps
    model_queries = training_model_queries + diagnostic_model_queries
    forward_macs = (
        INPUT_DIM * profile.hidden_width
        + profile.hidden_width * profile.hidden_width
        + profile.hidden_width * N_CLASSES
    )
    normalization_queries = model_queries * 2 if normalization_enabled else 0
    normalization_elements = normalization_queries * profile.hidden_width
    projection_events = steps if projection_enabled else 0
    projected_tensor_queries = projection_events * 2
    projected_elements = projection_events * (
        INPUT_DIM * profile.hidden_width + profile.hidden_width * profile.hidden_width
    )
    state_bytes = sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(state))
    receipt = NaPReceipt(
        data_steps=steps,
        observations=steps,
        data_bytes_read=steps * (INPUT_DIM * 4 + 4),
        training_model_queries=training_model_queries,
        diagnostic_model_queries=diagnostic_model_queries,
        model_queries=model_queries,
        parameter_updates=steps,
        normalization_queries=normalization_queries,
        normalization_elements=normalization_elements,
        projection_events=projection_events,
        projected_tensor_queries=projected_tensor_queries,
        projected_elements=projected_elements,
        logical_forward_macs=model_queries * forward_macs,
        logical_gradient_macs=steps * 2 * forward_macs,
        logical_auxiliary_scalar_ops=normalization_elements * 5 + projected_elements * 3,
        state_persistent_bytes=state_bytes,
        projection_target_persistent_bytes=8 if projection_enabled else 0,
        elapsed_ns=elapsed_ns,
    )
    return NaPArmResult(
        arm_id=arm_id,
        normalization_enabled=normalization_enabled,
        projection_enabled=projection_enabled,
        task_accuracy=tuple(accuracies),
        task_loss=tuple(losses),
        dead_unit_fraction=tuple(dead),
        effective_rank=tuple(ranks),
        initial_hidden_norms=initial_norms,
        final_hidden_norms=(
            float(jnp.linalg.norm(state.w1)),
            float(jnp.linalg.norm(state.w2)),
        ),
        final_state_sha256=_state_sha256(state),
        receipt=receipt,
    )


def run_comparator(
    images: object,
    labels: object,
    *,
    seed: object,
    profile_id: object = "contract-smoke",
) -> NaPResult:
    data, targets = _arrays(images, labels)
    if type(seed) is not int or seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen NaP development schedule")
    if type(profile_id) is not str or profile_id not in PROFILES:
        raise ValueError("unknown NaP profile")
    profile = PROFILES[profile_id]
    if data.shape[0] < profile.examples_per_task:
        raise ValueError("dataset has too few examples for the selected profile")
    tasks = _schedule(data, targets, profile, seed)
    result = NaPResult(
        schema=SCHEMA,
        catalog=NaPCatalogEntry(),
        profile_id=profile.profile_id,
        profile=profile,
        seed=seed,
        dataset_sha256=_dataset_sha(data, targets),
        schedule_sha256=_schedule_sha(tasks),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_source_sha256=hashlib.sha256(
            Path(__file__).with_name("plasticity_diagnostics.py").read_bytes()
        ).hexdigest(),
        nap_project_dependency_source_sha256=hashlib.sha256(
            Path(__file__).with_name("plasticity_comparators.py").read_bytes()
        ).hexdigest(),
        runtime_identity=_runtime_identity(),
        task_protocol="cumulative-input-permuted-mnist",
        labels_permuted=False,
        task_boundaries_visible_to_learner=False,
        task_ids_visible_to_learner=False,
        observations_matched_before_causal_divergence=True,
        arms=tuple(_run_arm(cast(ArmID, arm), tasks, profile, seed) for arm in ARM_IDS),
    )
    return validate_result(result)


def validate_result(value: object) -> NaPResult:
    if type(value) is not NaPResult:
        raise ValueError("result must be an exact NaPResult")
    NaPResult.__post_init__(value)
    profile = PROFILES[value.profile_id]
    steps = profile.n_tasks * profile.examples_per_task
    forward_macs = (
        INPUT_DIM * profile.hidden_width
        + profile.hidden_width * profile.hidden_width
        + profile.hidden_width * N_CLASSES
    )
    initial_state = _init_state(jr.key(0), profile.hidden_width)
    expected_state_bytes = sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(initial_state))
    for arm in value.arms:
        if type(arm) is not NaPArmResult or type(arm.receipt) is not NaPReceipt:
            raise ValueError("arms and receipts must have exact types")
        NaPArmResult.__post_init__(arm)
        NaPReceipt.__post_init__(arm.receipt)
        normalize, project = _arm_flags(cast(ArmID, arm.arm_id))
        training_model_queries = steps * 2
        diagnostic_model_queries = steps
        model_queries = training_model_queries + diagnostic_model_queries
        norm_queries = model_queries * 2 if normalize else 0
        norm_elements = norm_queries * profile.hidden_width
        projection_events = steps if project else 0
        projected_elements = projection_events * (
            INPUT_DIM * profile.hidden_width + profile.hidden_width * profile.hidden_width
        )
        expected = {
            "data_steps": steps,
            "observations": steps,
            "data_bytes_read": steps * (INPUT_DIM * 4 + 4),
            "training_model_queries": training_model_queries,
            "diagnostic_model_queries": diagnostic_model_queries,
            "model_queries": model_queries,
            "parameter_updates": steps,
            "normalization_queries": norm_queries,
            "normalization_elements": norm_elements,
            "projection_events": projection_events,
            "projected_tensor_queries": projection_events * 2,
            "projected_elements": projected_elements,
            "logical_forward_macs": model_queries * forward_macs,
            "logical_gradient_macs": steps * 2 * forward_macs,
            "logical_auxiliary_scalar_ops": norm_elements * 5 + projected_elements * 3,
            "state_persistent_bytes": expected_state_bytes,
            "projection_target_persistent_bytes": 8 if project else 0,
        }
        if len(arm.task_accuracy) != profile.n_tasks or any(
            getattr(arm.receipt, name) != expected_value
            for name, expected_value in expected.items()
        ):
            raise ValueError("NaP curve or exact resource receipt mismatch")
        if project and any(
            not math.isclose(final, initial, rel_tol=2e-6, abs_tol=2e-6)
            for initial, final in zip(
                arm.initial_hidden_norms, arm.final_hidden_norms, strict=True
            )
        ):
            raise ValueError("projected hidden norms drifted from their initial radii")
    current, mechanism_off, *_ = value.arms
    if (
        current.task_accuracy != mechanism_off.task_accuracy
        or current.task_loss != mechanism_off.task_loss
        or current.dead_unit_fraction != mechanism_off.dead_unit_fraction
        or current.effective_rank != mechanism_off.effective_rank
        or current.initial_hidden_norms != mechanism_off.initial_hidden_norms
        or current.final_hidden_norms != mechanism_off.final_hidden_norms
        or current.final_state_sha256 != mechanism_off.final_state_sha256
        or dataclasses.replace(current.receipt, elapsed_ns=0)
        != dataclasses.replace(mechanism_off.receipt, elapsed_ns=0)
    ):
        raise ValueError("NaP mechanism-off does not reduce exactly to the current SGD lane")
    return value


def qualification_gates() -> Mapping[str, object]:
    return {
        "execution_authorized": False,
        "paper_parity_allowed": False,
        "official_nap_code_available": False,
        "random_label_cifar": {
            "qualified": False,
            "paper_protocol": "20M steps; 200 random target resets; Adam 1e-4",
        },
        "sequential_ale": {
            "qualified": False,
            "paper_protocol": "Rainbow/DQN Zoo; 20M frames/game; restarted cosine schedule",
        },
        "blockers": (
            "method-specific official source absent",
            "paper datasets and task streams not qualified",
            "paper scale-offset and learning-rate schedules not implemented",
            "matched paper baselines and fresh scientific seeds absent",
        ),
    }


def _json_result(result: NaPResult) -> str:
    return json.dumps(
        dataclasses.asdict(result), allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--seed", type=int, default=FROZEN_SEEDS[0])
    parser.add_argument("--profile", choices=tuple(PROFILES), default="contract-smoke")
    parser.add_argument("--catalog", action="store_true")
    args = parser.parse_args(argv)
    if args.catalog:
        catalog = NaPCatalogEntry()
        catalog.validate()
        print(
            json.dumps(
                {
                    "catalog": dataclasses.asdict(catalog),
                    "qualification_gates": qualification_gates(),
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    if args.dataset is None:
        parser.error("--dataset is required unless --catalog is used")
    if (
        args.dataset.is_symlink()
        or not args.dataset.is_file()
        or args.dataset.stat().st_size > 256 * 1024 * 1024
    ):
        raise ValueError("dataset NPZ must be a bounded regular file")
    with np.load(args.dataset, allow_pickle=False) as payload:
        if set(payload.files) != {"images", "labels"}:
            raise ValueError("dataset NPZ must contain exactly images and labels")
        result = run_comparator(
            payload["images"], payload["labels"], seed=args.seed, profile_id=args.profile
        )
    print(_json_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
