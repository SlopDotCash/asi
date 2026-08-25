"""Bounded low-cost activation and feature comparator on the current hidden MNIST lane.

Issue #1566. This permanently nonpromoting development comparator reuses the data
schedule and bounded profiles introduced by issue #1583 and swaps only the
nonlinearity between arms, so the three mechanisms are compared against a shared
control rather than against each other's protocols.

It is not the papers' class-incremental CIFAR/TinyImageNet, ALE, or MuJoCo
protocols and makes no parity or ranking claim against their reported curves.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping
from typing import Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks.plasticity_comparators import (
    deep_fourier_features,
    interval_dropout,
    smooth_leaky,
)
from alberta_framework.benchmarks.plasticity_diagnostics import (
    INPUT_DIM,
    N_CLASSES,
    PROFILES,
    DiagnosticProfile,
    _arrays,
    _dataset_sha,
    _effective_rank,
    _schedule,
)

SCHEMA = "asi.low_cost_controls_ipmnist_comparator.development.v1"

# Pinned authoritative versions. Each preprint is recorded beside the final
# published record; the published record is authoritative where one exists.
SMOOTH_LEAKY_PAPER_REVISION = "arXiv:2509.22562v4"
SMOOTH_LEAKY_PAPER_IDENTITY = "ICLR-2026:XZf6wObHX4"
AID_PAPER_REVISION = "arXiv:2502.01342v2"
AID_PAPER_IDENTITY = "ICML-2025:park25b"
DEEP_FOURIER_PAPER_REVISION = "arXiv:2410.20634v1"
DEEP_FOURIER_PAPER_IDENTITY = "ICLR-2025:NIkfix2eDQ"

# Official code located for exactly one of the three mechanisms. It is disclosed
# only in the camera-ready PDF's link annotations and carries no license file, so
# it is recorded as a read-only disambiguation reference and nothing is copied.
SMOOTH_LEAKY_OFFICIAL_REPOSITORY = "https://github.com/lute47lillo/activations_plasticity"
SMOOTH_LEAKY_OFFICIAL_LICENSE = "none present"

DEPENDENCY_COMMIT = "8383d6438b81c7620189c6fedba30c345994cb12"
FROZEN_SEEDS = (15_660, 15_661, 15_662, 15_663)

# Figure 5 defaults from the Smooth-Leaky paper.
SMOOTH_LEAKY_ALPHA = 0.1
SMOOTH_LEAKY_POWER = 3.0
SMOOTH_LEAKY_CURVATURE = 5.0
# ReLU is AID_p at p = 1; a linear network is p = 0.5. The development value sits
# between them so the arm is neither reduction.
AID_RELU_PROBABILITY = 0.75

# Each mechanism carries its own declared off-reduction from the comparator
# registry: smooth_leaky alpha=1, aid relu_probability=1, deep_fourier disabled.
# `aid_off` is ReLU by construction and must therefore reproduce the control
# exactly, which is what pins the harness itself as mechanism-neutral.
ArmID = Literal[
    "sgd_current_control",
    "smooth_leaky",
    "smooth_leaky_off",
    "aid",
    "aid_off",
    "deep_fourier",
    "deep_fourier_off",
]
ARM_IDS: tuple[ArmID, ...] = (
    "sgd_current_control",
    "smooth_leaky",
    "smooth_leaky_off",
    "aid",
    "aid_off",
    "deep_fourier",
    "deep_fourier_off",
)

# Deep Fourier features emit two post-activation values per preactivation. The
# lane matches post-activation width across arms, so this arm carries half the
# preactivations and every downstream weight keeps the control's shape. That is
# the paper's own "reducing the effective width of the layer"; the resulting
# parameter delta is reported in the receipt rather than hidden.
_DEEP_FOURIER_WIDTH_DIVISOR = 2


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{name} must be an exact int within [{low}, {high}]")
    return value


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _mechanism_enabled(arm_id: str) -> bool:
    return not arm_id.endswith("_off") and arm_id != "sgd_current_control"


def _init_params(key: Array, width: int, preactivation_width: int) -> dict[str, Array]:
    k1, k2, k3 = jr.split(key, 3)
    return {
        "w1": jr.normal(k1, (INPUT_DIM, preactivation_width), dtype=jnp.float32)
        * math.sqrt(2.0 / INPUT_DIM),
        "b1": jnp.zeros((preactivation_width,), dtype=jnp.float32),
        "w2": jr.normal(k2, (width, preactivation_width), dtype=jnp.float32)
        * math.sqrt(2.0 / width),
        "b2": jnp.zeros((preactivation_width,), dtype=jnp.float32),
        "w3": jr.normal(k3, (width, N_CLASSES), dtype=jnp.float32) * math.sqrt(2.0 / width),
        "b3": jnp.zeros((N_CLASSES,), dtype=jnp.float32),
    }


def _activate(arm_id: ArmID, preactivation: Array, key: Array, training: bool) -> Array:
    if arm_id == "sgd_current_control":
        return jax.nn.relu(preactivation)
    if arm_id in ("smooth_leaky", "smooth_leaky_off"):
        return smooth_leaky(
            preactivation,
            alpha=1.0 if arm_id.endswith("_off") else SMOOTH_LEAKY_ALPHA,
            power=SMOOTH_LEAKY_POWER,
            curvature=SMOOTH_LEAKY_CURVATURE,
        )
    if arm_id in ("aid", "aid_off"):
        return interval_dropout(
            preactivation,
            key,
            relu_probability=1.0 if arm_id.endswith("_off") else AID_RELU_PROBABILITY,
            training=training,
        )
    if arm_id in ("deep_fourier", "deep_fourier_off"):
        return deep_fourier_features(preactivation, enabled=arm_id == "deep_fourier")
    raise ValueError(f"unknown arm: {arm_id}")


def _preactivation_width(arm_id: ArmID, width: int) -> int:
    """Preactivation count for an arm, so post-activation width matches the control."""
    if arm_id == "deep_fourier":
        if width % _DEEP_FOURIER_WIDTH_DIVISOR != 0:
            raise ValueError("deep Fourier arm requires an even hidden width")
        return width // _DEEP_FOURIER_WIDTH_DIVISOR
    return width


def _forward(
    params: Mapping[str, Array], inputs: Array, arm_id: ArmID, key: Array, training: bool
) -> tuple[Array, Array, Array]:
    key1, key2 = jr.split(key)
    hidden1 = _activate(arm_id, inputs @ params["w1"] + params["b1"], key1, training)
    hidden2 = _activate(arm_id, hidden1 @ params["w2"] + params["b2"], key2, training)
    return hidden2 @ params["w3"] + params["b3"], hidden1, hidden2


def _loss(
    params: Mapping[str, Array], inputs: Array, label: Array, arm_id: ArmID, key: Array
) -> Array:
    logits, _, _ = _forward(params, inputs, arm_id, key, True)
    return -jax.nn.log_softmax(logits)[label]


def _step(
    params: dict[str, Array],
    inputs: Array,
    label: Array,
    key: Array,
    learning_rate: float,
    arm_id: ArmID,
) -> tuple[dict[str, Array], Array, Array]:
    loss, grads = jax.value_and_grad(_loss)(params, inputs, label, arm_id, key)
    updated = {name: value - learning_rate * grads[name] for name, value in params.items()}
    logits, _, _ = _forward(params, inputs, arm_id, key, False)
    return updated, loss, jnp.argmax(logits)


def _persistent_bytes(params: Mapping[str, Array]) -> int:
    return int(sum(int(value.size) * int(value.dtype.itemsize) for value in params.values()))


@dataclasses.dataclass(frozen=True, slots=True)
class LowCostCatalogEntry:
    """Primary-source provenance for the three mechanisms, as audited."""

    issue: int = 1566
    smooth_leaky_paper_revision: str = SMOOTH_LEAKY_PAPER_REVISION
    smooth_leaky_paper_identity: str = SMOOTH_LEAKY_PAPER_IDENTITY
    aid_paper_revision: str = AID_PAPER_REVISION
    aid_paper_identity: str = AID_PAPER_IDENTITY
    deep_fourier_paper_revision: str = DEEP_FOURIER_PAPER_REVISION
    deep_fourier_paper_identity: str = DEEP_FOURIER_PAPER_IDENTITY
    # Official code was located for Smooth-Leaky only, and only inside the
    # camera-ready PDF's link annotations. It has no license file, so it grants
    # no redistribution or derivative rights and nothing is copied from it.
    smooth_leaky_official_code_available: bool = True
    smooth_leaky_official_repository: str = SMOOTH_LEAKY_OFFICIAL_REPOSITORY
    smooth_leaky_official_license: str = SMOOTH_LEAKY_OFFICIAL_LICENSE
    smooth_leaky_official_code_copied: bool = False
    aid_official_code_available: bool = False
    deep_fourier_official_code_available: bool = False
    asi_dependency_commit: str = DEPENDENCY_COMMIT
    protocol_differences: tuple[str, ...] = (
        "cumulative input-permuted MNIST rather than the papers' class-incremental "
        "CIFAR/TinyImageNet, ALE, or non-stationary MuJoCo protocols",
        "two width-bounded MLP hidden layers rather than the papers' CNN/ResNet or "
        "Rainbow architectures",
        "per-example SGD rather than the papers' minibatch Adam or RL optimizers",
        "the deep Fourier arm halves its preactivation count to hold post-activation "
        "width equal to the control, so downstream weights keep the control's shape",
        "AID uses the simplified two-interval form at a single fixed probability, not "
        "a tuned interval schedule",
        "Randomized Smooth-Leaky is out of scope; only the fixed-leak variant runs",
    )
    development_only: bool = True
    scientific_promotion_allowed: bool = False

    def validate(self) -> None:
        if type(self.issue) is not int or self.issue != 1566:
            raise ValueError("low-cost controls catalog issue drift")
        identities = (
            (self.smooth_leaky_paper_revision, SMOOTH_LEAKY_PAPER_REVISION),
            (self.smooth_leaky_paper_identity, SMOOTH_LEAKY_PAPER_IDENTITY),
            (self.aid_paper_revision, AID_PAPER_REVISION),
            (self.aid_paper_identity, AID_PAPER_IDENTITY),
            (self.deep_fourier_paper_revision, DEEP_FOURIER_PAPER_REVISION),
            (self.deep_fourier_paper_identity, DEEP_FOURIER_PAPER_IDENTITY),
            (self.asi_dependency_commit, DEPENDENCY_COMMIT),
        )
        for value, expected in identities:
            if type(value) is not str or value != expected:
                raise ValueError("paper/dependency identity drift")
        if self.smooth_leaky_official_code_available is not True:
            raise ValueError("located Smooth-Leaky official code must stay recorded as present")
        if self.smooth_leaky_official_license != SMOOTH_LEAKY_OFFICIAL_LICENSE:
            raise ValueError("unlicensed status of the official repository must stay recorded")
        if self.smooth_leaky_official_code_copied is not False:
            raise ValueError("no bytes may be copied from an unlicensed repository")
        for name in ("aid_official_code_available", "deep_fourier_official_code_available"):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must remain explicitly absent")
        if (
            type(self.protocol_differences) is not tuple
            or len(self.protocol_differences) != 6
            or any(type(value) is not str or not value for value in self.protocol_differences)
        ):
            raise ValueError("paper protocol differences must remain explicit")
        if self.development_only is not True or self.scientific_promotion_allowed is not False:
            raise ValueError("low-cost comparator must remain permanently nonpromoting")


@dataclasses.dataclass(frozen=True, slots=True)
class LowCostArmResult:
    arm_id: str
    mechanism_enabled: bool
    preactivation_width: int
    task_accuracy: tuple[float, ...]
    task_loss: tuple[float, ...]
    dead_unit_fraction: tuple[float, ...]
    effective_rank: tuple[float, ...]
    persistent_bytes: int
    parameter_updates: int
    model_queries: int
    elapsed_ns: int

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError(f"unknown arm: {self.arm_id}")
        if (
            type(self.mechanism_enabled) is not bool
            or self.mechanism_enabled != _mechanism_enabled(self.arm_id)
        ):
            raise ValueError("mechanism flag differs from the frozen arm roster")
        _exact_int(self.preactivation_width, "preactivation_width", 1, 4096)
        _exact_int(self.persistent_bytes, "persistent_bytes", 1, 2**63 - 1)
        _exact_int(self.parameter_updates, "parameter_updates", 1, 2**63 - 1)
        _exact_int(self.model_queries, "model_queries", 1, 2**63 - 1)
        _exact_int(self.elapsed_ns, "elapsed_ns", 0, 2**63 - 1)
        curves = (
            self.task_accuracy,
            self.task_loss,
            self.dead_unit_fraction,
            self.effective_rank,
        )
        if any(type(curve) is not tuple or not curve for curve in curves):
            raise ValueError("diagnostic curves must be non-empty exact tuples")
        if len({len(curve) for curve in curves}) != 1:
            raise ValueError("diagnostic curves must have equal lengths")
        if any(
            type(value) is not float or not math.isfinite(value)
            for curve in curves
            for value in curve
        ):
            raise ValueError("diagnostic curves must hold finite exact floats")
        if any(
            not 0.0 <= value <= 1.0
            for value in self.task_accuracy + self.dead_unit_fraction
        ):
            raise ValueError("probability curves must lie in [0, 1]")
        if any(value < 0.0 for value in self.task_loss + self.effective_rank):
            raise ValueError("loss and rank curves must be nonnegative")


def _run_arm(
    arm_id: ArmID,
    tasks: tuple[tuple[np.ndarray, np.ndarray], ...],
    profile: DiagnosticProfile,
    seed: int,
) -> LowCostArmResult:
    preactivation_width = _preactivation_width(arm_id, profile.hidden_width)
    key = jr.key(seed)
    key, init_key = jr.split(key)
    params = _init_params(init_key, profile.hidden_width, preactivation_width)

    accuracies: list[float] = []
    losses: list[float] = []
    dead: list[float] = []
    ranks: list[float] = []
    updates = 0
    queries = 0
    started = time.perf_counter_ns()

    for inputs, labels in tasks:
        correct = 0
        task_losses: list[float] = []
        for row, label in zip(inputs, labels, strict=True):
            key, step_key = jr.split(key)
            params, loss, prediction = _step(
                params,
                jnp.asarray(row),
                jnp.asarray(label),
                step_key,
                profile.learning_rate,
                arm_id,
            )
            updates += 1
            # One training forward for the gradient and one evaluation forward
            # for the reported prediction.
            queries += 2
            correct += int(prediction == label)
            task_losses.append(float(loss))
        key, eval_key = jr.split(key)
        # Evaluation forwards are deterministic: the stochastic arm uses its
        # declared test-time scaling rather than a fresh draw.
        _, hidden1, hidden2 = _forward(params, jnp.asarray(inputs), arm_id, eval_key, False)
        host1, host2 = np.asarray(hidden1), np.asarray(hidden2)
        dead_units = int(np.sum(np.all(host1 == 0.0, axis=0))) + int(
            np.sum(np.all(host2 == 0.0, axis=0))
        )
        accuracies.append(float(correct / len(labels)))
        losses.append(float(np.mean(task_losses)))
        dead.append(float(dead_units / (host1.shape[1] + host2.shape[1])))
        ranks.append(float(_effective_rank(host2)))

    return LowCostArmResult(
        arm_id=arm_id,
        mechanism_enabled=_mechanism_enabled(arm_id),
        preactivation_width=preactivation_width,
        task_accuracy=tuple(accuracies),
        task_loss=tuple(losses),
        dead_unit_fraction=tuple(dead),
        effective_rank=tuple(ranks),
        persistent_bytes=_persistent_bytes(params),
        parameter_updates=updates,
        model_queries=queries,
        elapsed_ns=time.perf_counter_ns() - started,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class LowCostResult:
    schema: str
    profile_id: str
    seed: int
    dataset_sha256: str
    catalog: LowCostCatalogEntry
    arms: tuple[LowCostArmResult, ...]
    development_only: bool = True

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("schema drift")
        if type(self.profile_id) is not str or self.profile_id not in PROFILES:
            raise ValueError("unknown profile")
        if type(self.seed) is not int or self.seed not in FROZEN_SEEDS:
            raise ValueError("seed must be an exact frozen development seed")
        _digest(self.dataset_sha256, "dataset identity")
        if type(self.catalog) is not LowCostCatalogEntry:
            raise ValueError("catalog must be an exact LowCostCatalogEntry")
        if (
            type(self.arms) is not tuple
            or len(self.arms) != len(ARM_IDS)
            or any(type(arm) is not LowCostArmResult for arm in self.arms)
        ):
            raise ValueError("every declared arm must be reported")
        if tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise ValueError("arm order must match the declared arm list")
        if type(self.development_only) is not bool or self.development_only is not True:
            raise ValueError("result must remain permanently nonpromoting")
        self.catalog.validate()
        profile = PROFILES[self.profile_id]
        expected_updates = profile.n_tasks * profile.examples_per_task
        for arm in self.arms:
            LowCostArmResult.__post_init__(arm)
            if len(arm.task_accuracy) != profile.n_tasks:
                raise ValueError("diagnostic curves must bind every frozen task")
            if arm.preactivation_width != _preactivation_width(
                cast(ArmID, arm.arm_id), profile.hidden_width
            ):
                raise ValueError("preactivation width differs from the frozen arm geometry")
            if arm.parameter_updates != expected_updates:
                raise ValueError("parameter updates differ from the frozen schedule")


def run_comparator(
    images: object,
    labels: object,
    *,
    profile_id: str = "contract-smoke",
    seed: int = FROZEN_SEEDS[0],
) -> LowCostResult:
    """Run every arm against one frozen schedule and return the matched result."""
    if profile_id not in PROFILES:
        raise ValueError("unknown profile")
    profile = PROFILES[profile_id]
    image_array, label_array = _arrays(images, labels)
    tasks = _schedule(image_array, label_array, profile, seed)
    arms = tuple(_run_arm(arm_id, tasks, profile, seed) for arm_id in ARM_IDS)
    return LowCostResult(
        schema=SCHEMA,
        profile_id=profile_id,
        seed=seed,
        dataset_sha256=_dataset_sha(image_array, label_array),
        catalog=LowCostCatalogEntry(),
        arms=arms,
    )


def qualification_gates() -> Mapping[str, object]:
    """Declared, machine-checkable gates for this development lane."""
    return {
        "issue": 1566,
        "schema": SCHEMA,
        "arms": ARM_IDS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "mechanism_off_reductions": {
            "smooth_leaky_off": "alpha=1",
            "aid_off": "relu_probability=1",
            "deep_fourier_off": "feature_enabled=False",
        },
    }
