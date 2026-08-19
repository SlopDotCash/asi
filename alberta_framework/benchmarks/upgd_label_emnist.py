"""UPGD Label-permuted EMNIST replication runner (Elsayed & Mahmood, ICLR 2024).

This lane implements a development replication diagnostic based on the *online
Label-permuted EMNIST* protocol of

    Elsayed, M. & Mahmood, A. R. (2024). Addressing Loss of Plasticity and
    Catastrophic Forgetting in Continual Learning. ICLR 2024.
    https://openreview.net/forum?id=sKPzAXoylB

using the task, network, horizon, and best statistics-run hyperparameters from
the authors' MIT-licensed repository (https://github.com/mohmdelsayed/upgd,
commit ``b75e90ad4b09c28971ac9dbb902a8fd86709b28c`` -- the same commit audited
by :mod:`alberta_framework.core.canonical_upgd` and the Input-permuted MNIST
lane in :mod:`alberta_framework.benchmarks.upgd_ipmnist`):

- 1,000,000 training examples presented one per step (batch size 1) from the
  EMNIST *balanced* split train set (112,800 examples, 47 classes,
  ``core/task/label_permuted_emnist.py``).
- Labels are permuted every 2,500 steps => 400 tasks. Upstream's
  ``change_all_lables`` runs at step 0, so the first task is itself permuted,
  and it applies ``randperm(47)[targets]`` *to the already-permuted targets*:
  the label mapping of task ``t`` is the composition of ``t+1`` fresh uniform
  permutations (statistically equivalent to one fresh uniform permutation per
  task). This lane reproduces the composition exactly.
- ``change_all_lables`` also recreates the shuffled ``DataLoader`` at every
  boundary, so each task presents 2,500 examples drawn without replacement
  from a fresh shuffle of the train split -- the same per-task sampling shape
  as the Input-permuted MNIST lane.
- Inputs scaled to ``[-1, 1]`` (``ToTensor`` then ``Normalize((0.5,), (0.5,))``).
- Network: 784 -> 300 -> ReLU -> 150 -> ReLU -> 47 MLP
  (``core/network/fcn_relu.py``), PyTorch-default Linear init.
- Loss: 47-class softmax cross-entropy on the single example. Metric: online
  accuracy of the pre-update prediction, averaged per task.
- Learners (best statistics-run hyperparameters from
  ``experiments/statistics_output_permuted_emnist.py`` at the audited commit;
  the ``zip(learners, grids)`` pairing there places
  ``FirstOrderGlobalUPGDLearner`` (= UPGD-W, protecting) on the second grid --
  the same pairing convention audited for the Input-permuted MNIST lane):

  * **UPGD-W**: ``lr=0.01, beta_utility=0.9, sigma=0.001, weight_decay=0.0``.
  * **AdamW** (released decoupled-decay Adam): ``lr=1e-4, beta1=0.0,
    beta2=0.9999, eps=1e-8, weight_decay=0.1``.

Documented deviations from the released code (this is therefore not a claim of
complete published-protocol exactness):

- **Dataset plumbing**: upstream loads EMNIST through torchvision; this lane
  uses OpenML dataset ``EMNIST_Balanced`` (data_id 41039, 131,600 rows =
  balanced train+test). The loader keeps the first 112,800 rows as the train
  split only after verifying their class counts match the torchvision train
  split exactly (2,400/class train, 400/class test remainder); otherwise it
  fails closed. Row-identity with torchvision is *not* verified, and OpenML's
  pixel order may be transposed relative to torchvision -- both are fixed
  input permutations, irrelevant to a fully-connected network's dynamics.
- Upstream permutation/shuffle streams are unseeded; here every stream
  derives from the run seed (same policy as the IPMNIST lane).
- Task metric blocks are exact ``[2500t, 2500(t+1))`` (upstream logging is
  shifted one step past the boundary).
- Bias corrections are float32; the UPGD-W inner loop reuses the IPMNIST
  lane's :func:`~alberta_framework.benchmarks.upgd_ipmnist.lean_upgd_w_update`
  scan-optimized step, which is pinned by a supplied-noise parity test to the
  audited ``CanonicalUPGD`` ``official_experiment_global`` protecting profile.

The CLI exposes a small ``plan`` / ``shard`` / ``merge`` lifecycle: a plan
pins config, learner arms, seeds, and dataset digests; each shard runs one
learner x seed and binds the plan hash; merge validates the exact planned
Cartesian product (or records incomplete coverage explicitly). Benchmark
executions happen through this CLI, never inside pytest.

**EMA input-conditioning transfer arms** (mechanism-transfer diagnostic from
the IPMNIST screening campaign; factories imported from
:mod:`alberta_framework.benchmarks.ipmnist_screening`, so the normalizer and
update equations are exactly the pinned screening-lane ones):

- ``upgd_ema_norm`` — published EMNIST UPGD-W behind the screening lane's EMA
  input normalizer (``norm_decay=0.999``, ``norm_epsilon=1e-8``).
- ``upgd_ema_norm_sigma0`` — same, without the perturbation
  (``noise_std=0.0``).
- ``sgd_ema_norm`` — bare-conditioning control: plain SGD + decoupled decay
  behind the same normalizer; weight decay matched to the published EMNIST
  UPGD-W value (0.0 here — the IPMNIST screening arm used 0.01 to match that
  protocol's UPGD-W decay).

Unlike Input-permuted MNIST, the Label-permuted EMNIST input marginal is
STATIONARY (features never permute; labels do), so these arms test whether
input conditioning helps when there is no input non-stationarity to fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed, require_unique_jax_seeds
from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningStepFn,
    _make_sgd_ema_norm_learner,
    _make_upgd_ema_norm_learner,
    _wrap_grad_learner,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    _LEARNER_FACTORIES,
    IPMNISTConfig,
    LearnerInitFn,
    atomic_write_new_json,
    init_mlp_params,
    task_index_for_step,
    validated_ipmnist_data,
)
from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2**31 - 1

_ACTUAL_INT_TYPES: frozenset[type] = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


logger = logging.getLogger(__name__)

LabelEMNISTLearner = Literal[
    "upgd_w", "adamw", "upgd_ema_norm", "upgd_ema_norm_sigma0", "sgd_ema_norm"
]

BENCHMARK = "upgd_label_permuted_emnist"
PLAN_SCHEMA = "alberta.upgd_label_emnist.plan.v1"
PARTIAL_SCHEMA = "alberta.upgd_label_emnist.partial.v1"
ARTIFACT_SCHEMA = "alberta.upgd_label_emnist.artifact.v1"

EMNIST_OPENML_DATA_ID = 41039
EMNIST_OPENML_NAME = "EMNIST_Balanced"
EMNIST_TOTAL_ROWS = 131_600
EMNIST_TRAIN_ROWS = 112_800
EMNIST_TRAIN_PER_CLASS = 2_400
EMNIST_TEST_PER_CLASS = 400

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_replication_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
    "execution_attestation": False,
}

PROTOCOL_DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "code": "openml_dataset_plumbing",
        "scope": "dataset",
        "description": (
            "OpenML EMNIST_Balanced (41039) first 112800 rows, accepted only when class "
            "counts match the torchvision balanced train split; row identity and pixel "
            "orientation with torchvision are unverified (both irrelevant to an MLP)"
        ),
    },
    {
        "code": "seeded_streams",
        "scope": "rng_schedule",
        "description": (
            "label-permutation and example streams are seed-derived; upstream streams "
            "are unseeded"
        ),
    },
    {
        "code": "task_aligned_logging",
        "scope": "metric_blocks",
        "description": "task blocks are [t*L, (t+1)*L); upstream logging is shifted one step",
    },
    {
        "code": "float32_bias_corrections",
        "scope": "numeric_precision",
        "description": "bias corrections remain float32; upstream mixes float64 scalars",
    },
)

#: Best statistics-run hyperparameters for the protecting UPGD-W arm, from
#: ``experiments/statistics_output_permuted_emnist.py`` (``upgd2_grid`` paired
#: with ``FirstOrderGlobalUPGDLearner`` under ``zip``) at the audited commit.
UPGD_W_PROTOCOL_HYPERPARAMETERS: dict[str, float] = {
    "step_size": 0.01,
    "utility_decay": 0.9,
    "noise_std": 0.001,
    "weight_decay": 0.0,
}
#: Best statistics-run hyperparameters for the released decoupled-decay Adam.
ADAMW_PROTOCOL_HYPERPARAMETERS: dict[str, float] = {
    "step_size": 1e-4,
    "beta1": 0.0,
    "beta2": 0.9999,
    "eps": 1e-8,
    "weight_decay": 0.1,
}

#: Screening-lane EMA input-normalizer settings (the exact values screened and
#: confirmed on IPMNIST: ``upgd_ema_norm`` in
#: :mod:`alberta_framework.benchmarks.ipmnist_screening`).
_EMA_NORM_SETTINGS: dict[str, float] = {"norm_decay": 0.999, "norm_epsilon": 1e-8}

#: Published EMNIST UPGD-W behind the screening lane's EMA input normalizer.
UPGD_EMA_NORM_HYPERPARAMETERS: dict[str, float] = {
    **UPGD_W_PROTOCOL_HYPERPARAMETERS,
    **_EMA_NORM_SETTINGS,
}
#: The same arm without the perturbation (is noise load-bearing here?).
UPGD_EMA_NORM_SIGMA0_HYPERPARAMETERS: dict[str, float] = {
    **UPGD_EMA_NORM_HYPERPARAMETERS,
    "noise_std": 0.0,
}
#: Bare-conditioning control: plain SGD + decoupled decay behind the same
#: normalizer. Weight decay matches the published EMNIST UPGD-W decay (0.0;
#: the IPMNIST screening arm used 0.01 to match that protocol's UPGD-W).
SGD_EMA_NORM_HYPERPARAMETERS: dict[str, float] = {
    "step_size": 0.01,
    "weight_decay": 0.0,
    **_EMA_NORM_SETTINGS,
}

_LEARNER_DEFAULT_HYPERPARAMETERS: dict[str, dict[str, float]] = {
    "upgd_w": UPGD_W_PROTOCOL_HYPERPARAMETERS,
    "adamw": ADAMW_PROTOCOL_HYPERPARAMETERS,
    "upgd_ema_norm": UPGD_EMA_NORM_HYPERPARAMETERS,
    "upgd_ema_norm_sigma0": UPGD_EMA_NORM_SIGMA0_HYPERPARAMETERS,
    "sgd_ema_norm": SGD_EMA_NORM_HYPERPARAMETERS,
}


def _validated_hyperparameter(name: object, value: object) -> float:
    """Validate one JSON override in its exact host and float32 execution domains."""
    host_name = _require_exact_str("name", name)
    if type(value) is not int and type(value) is not float:
        raise ValueError(f"hyperparameter '{host_name}' must be a finite JSON number")
    label = f"hyperparameter '{host_name}'"
    if host_name in {"step_size", "eps", "norm_epsilon"}:
        return validated_float32_scalar(label, value, positive=True)
    if name in {"utility_decay", "beta1", "beta2", "norm_decay"}:
        return validated_float32_scalar(
            label, value, lower=0.0, upper=1.0, upper_inclusive=False
        )
    return validated_float32_scalar(label, value, lower=0.0)


def _wrapped_v1_factory(
    name: str,
) -> Callable[[dict[str, float]], tuple[LearnerInitFn, ScreeningStepFn]]:
    """Adapt a grads-interface ``upgd_ipmnist`` learner to the full-step API.

    :func:`~alberta_framework.benchmarks.ipmnist_screening._wrap_grad_learner`
    mirrors this lane's original inner-step ordering exactly (pre-update
    accuracy, post-update plasticity), so the v1 arms are behavior-preserving
    under the registry refactor (pinned by tiny-trajectory regression tests).
    """

    def factory(hp: dict[str, float]) -> tuple[LearnerInitFn, ScreeningStepFn]:
        return _wrap_grad_learner(*_LEARNER_FACTORIES[name](hp))

    return factory


#: Full-step learner factories (``(params, state, x, y, key) ->
#: (params, state, (accuracy, loss, plasticity))``). Normalized arms reuse the
#: pinned screening-lane factories so the EMA normalizer equations and state
#: threading are exactly the IPMNIST-screened ones.
_FULL_STEP_FACTORIES: dict[
    str, Callable[[dict[str, float]], tuple[LearnerInitFn, ScreeningStepFn]]
] = {
    "upgd_w": _wrapped_v1_factory("upgd_w"),
    "adamw": _wrapped_v1_factory("adamw"),
    "upgd_ema_norm": _make_upgd_ema_norm_learner,
    "upgd_ema_norm_sigma0": _make_upgd_ema_norm_learner,
    "sgd_ema_norm": _make_sgd_ema_norm_learner,
}

#: Published reference numbers. The paper reports curves (its Figure for
#: label-permuted EMNIST), not a table; these are approximate read-offs.
#: UPGD-W *rises* over the 400 tasks (feature consolidation helps), while
#: plasticity-only baselines sit far lower. The AdamW read-off is
#: low-confidence (curves for the non-consolidating baselines cluster).
PAPER_REFERENCE: dict[str, Any] = {
    "citation": (
        "Elsayed & Mahmood (2024). Addressing Loss of Plasticity and "
        "Catastrophic Forgetting in Continual Learning. ICLR 2024."
    ),
    "openreview": "https://openreview.net/forum?id=sKPzAXoylB",
    "official_repository": "https://github.com/mohmdelsayed/upgd",
    "official_commit": "b75e90ad4b09c28971ac9dbb902a8fd86709b28c",
    "protocol_files": [
        "core/task/label_permuted_emnist.py",
        "core/network/fcn_relu.py",
        "core/run/run_stats.py",
        "experiments/statistics_output_permuted_emnist.py",
        "core/optim/weight_upgd/first_order.py",
        "core/optim/adam.py",
    ],
    "n_seeds": 20,
    "approximate_average_online_accuracy": {
        "upgd_w": 0.74,
        "adamw": 0.35,
    },
    "qualitative": (
        "UPGD-W rises toward ~0.73-0.75 across 400 tasks; plasticity-only and "
        "plain-optimizer baselines sit near ~0.35."
    ),
    "reference_kind": "figure_readoff_low_confidence",
}

#: Reproduction gaps above this are flagged for investigation, never absorbed.
#: Wider than the IPMNIST threshold because the figure read-off itself is
#: less certain on this benchmark.
REPRODUCTION_GAP_THRESHOLD = 0.03


@dataclass(frozen=True)
class LabelEMNISTConfig:
    """Label-permuted EMNIST protocol configuration.

    Defaults match the selected ICLR-2024 configuration; tests shrink every
    field. Configuration equality alone is not protocol exactness.

    Args:
        n_tasks: Number of label-permutation tasks (published: 400).
        task_length: Online steps per task (published: 2,500).
        input_dim: Flattened input dimensionality (published: 784).
        hidden1: First hidden layer width (published: 300).
        hidden2: Second hidden layer width (published: 150).
        n_classes: Output classes (published: 47).
    """

    n_tasks: int = 400
    task_length: int = 2500
    input_dim: int = 784
    hidden1: int = 300
    hidden2: int = 150
    n_classes: int = 47

    def __post_init__(self) -> None:
        for name in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes"):
            host_name = _require_exact_str("name", name)
            value = getattr(self, host_name)
            actual_type = type(value)
            if (
                not any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES)
                or value <= 0
            ):
                raise ValueError(f"{host_name} must be a positive integer")

    @property
    def n_steps(self) -> int:
        """Total online steps (examples) in a run."""
        return self.n_tasks * self.task_length

    @property
    def matches_selected_publication_configuration(self) -> bool:
        """Whether shape and horizon match the selected publication configuration."""
        return self == LabelEMNISTConfig()

    def to_config(self) -> dict[str, int]:
        """Return a JSON-serializable configuration."""
        return {
            "n_tasks": self.n_tasks,
            "task_length": self.task_length,
            "input_dim": self.input_dim,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "n_classes": self.n_classes,
        }


@chex.dataclass(frozen=True)
class LabelEMNISTSchedule:
    """Per-seed label-permutation and example schedule.

    Attributes:
        label_permutations: ``(n_tasks, n_classes)`` int32; row ``t`` maps an
            original dataset label to the label presented during task ``t``.
            Row ``t`` is the composition ``r_t o r_{t-1} o ... o r_0`` of
            fresh uniform permutations, matching upstream's cumulative
            ``randperm(47)[targets]`` mutation (task 0 is already permuted).
        example_indices: ``(n_tasks, task_length)`` int32; row ``t`` holds the
            dataset rows presented during task ``t``, drawn without
            replacement from a fresh per-task shuffle of the train split.
    """

    label_permutations: Array
    example_indices: Array


def build_schedule(key: Array, config: LabelEMNISTConfig, n_train: int) -> LabelEMNISTSchedule:
    """Build the deterministic label/example schedule for one seed.

    Args:
        key: Per-seed schedule key.
        config: Protocol configuration.
        n_train: Number of rows in the train split; must be at least
            ``config.task_length`` so tasks sample without replacement.

    Returns:
        The schedule; task ``t`` is exactly steps
        ``[t * task_length, (t + 1) * task_length)``.
    """
    if type(config) is not LabelEMNISTConfig:
        raise TypeError("config must be an exact LabelEMNISTConfig")
    if type(n_train) is not int:
        raise TypeError("n_train must be an exact built-in int")
    if not 1 <= n_train <= _INT32_MAX:
        raise ValueError("n_train must be positive and at most signed-int32")
    if n_train < config.task_length:
        raise ValueError(
            f"n_train={n_train} is smaller than task_length={config.task_length}; "
            "per-task sampling is without replacement"
        )
    key_perm, key_sample = jr.split(key)
    tasks = jnp.arange(config.n_tasks)

    def compose(previous: Array, task: Array) -> tuple[Array, Array]:
        fresh = jr.permutation(jr.fold_in(key_perm, task), config.n_classes)
        current = fresh[previous]
        return current, current

    identity = jnp.arange(config.n_classes)
    _, label_permutations = jax.lax.scan(compose, identity, tasks)
    example_indices = jax.vmap(
        lambda task: jr.permutation(jr.fold_in(key_sample, task), n_train)[: config.task_length]
    )(tasks)
    return LabelEMNISTSchedule(  # type: ignore[call-arg]
        label_permutations=label_permutations.astype(jnp.int32),
        example_indices=example_indices.astype(jnp.int32),
    )


def _require_finite_real(name: str, value: object) -> float:
    if type(value) is not int and type(value) is not float:
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _require_nonempty_string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_seed_identities(values: object, *, name: str) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{name} must be an exact tuple of unique integer seeds")
    return require_unique_jax_seeds(values, name=name)


@dataclass(frozen=True)
class LabelEMNISTRunResult:
    """Host-side result of one learner's multi-seed run.

    Per-task arrays have shape ``(n_seeds, n_tasks)``. ``per_step_accuracy``
    (``(n_seeds, n_tasks, task_length)``), ``initial_params``,
    ``label_permutations``, and ``example_indices`` are populated only when
    the run was invoked with ``return_per_step=True`` (debug/testing scale).
    """

    learner: str
    hyperparameters: dict[str, float]
    seeds: tuple[int, ...]
    config: LabelEMNISTConfig
    per_task_accuracy: np.ndarray
    per_task_loss: np.ndarray
    per_task_plasticity: np.ndarray
    average_online_accuracy: np.ndarray
    wall_clock_seconds: float
    per_step_accuracy: np.ndarray | None = None
    initial_params: dict[str, np.ndarray] | None = None
    label_permutations: np.ndarray | None = None
    example_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "learner", _require_nonempty_string("learner", self.learner))
        if type(self.hyperparameters) is not dict:
            raise TypeError("hyperparameters must be a dict")
        normalized_hyperparameters: dict[str, float] = {}
        for name, value in self.hyperparameters.items():
            if type(name) is not str or not name:
                raise TypeError("hyperparameters keys must be exact non-empty strings")
            if (type(value) is not int and type(value) is not float) or not math.isfinite(value):
                raise ValueError("hyperparameters values must be finite built-in numbers")
            normalized_hyperparameters[name] = float(value)
        object.__setattr__(self, "hyperparameters", normalized_hyperparameters)
        object.__setattr__(self, "seeds", _require_seed_identities(self.seeds, name="seeds"))
        if type(self.config) is not LabelEMNISTConfig:
            raise TypeError("config must be a LabelEMNISTConfig")
        shape = (len(self.seeds), self.config.n_tasks)
        for name in (
            "per_task_accuracy",
            "per_task_loss",
            "per_task_plasticity",
        ):
            value = getattr(self, name)
            if (
                type(value) is not np.ndarray
                or value.shape != shape
                or value.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            ):
                raise ValueError(f"{name} must have the exact floating run-result shape")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if type(self.average_online_accuracy) is not np.ndarray or (
            self.average_online_accuracy.shape != (len(self.seeds),)
            or self.average_online_accuracy.dtype
            not in (np.dtype(np.float32), np.dtype(np.float64))
        ):
            raise ValueError("average_online_accuracy must have the exact float32 seed shape")
        if not np.isfinite(self.average_online_accuracy).all():
            raise ValueError("average_online_accuracy must contain only finite values")
        if np.any((self.per_task_accuracy < 0.0) | (self.per_task_accuracy > 1.0)):
            raise ValueError("per_task_accuracy must lie in [0, 1]")
        if np.any(self.per_task_loss < 0.0):
            raise ValueError("per_task_loss must be non-negative")
        if np.any((self.per_task_plasticity < 0.0) | (self.per_task_plasticity > 1.0)):
            raise ValueError("per_task_plasticity must lie in [0, 1]")
        if np.any((self.average_online_accuracy < 0.0) | (self.average_online_accuracy > 1.0)):
            raise ValueError("average_online_accuracy must lie in [0, 1]")
        debug_values = (
            self.per_step_accuracy,
            self.initial_params,
            self.label_permutations,
            self.example_indices,
        )
        if any(value is None for value in debug_values) != all(
            value is None for value in debug_values
        ):
            raise ValueError("debug result fields must be all present or all absent")
        if self.per_step_accuracy is not None:
            step_shape = (*shape, self.config.task_length)
            if (
                type(self.per_step_accuracy) is not np.ndarray
                or self.per_step_accuracy.shape != step_shape
                or self.per_step_accuracy.dtype
                not in (np.dtype(np.float32), np.dtype(np.float64))
                or not np.isfinite(self.per_step_accuracy).all()
            ):
                raise ValueError("per_step_accuracy must match the exact float32 step shape")
            if self.label_permutations is None or self.example_indices is None:
                raise ValueError("debug schedules must be present with per_step_accuracy")
            if type(self.label_permutations) is not np.ndarray or (
                self.label_permutations.shape
                != (len(self.seeds), self.config.n_tasks, self.config.n_classes)
                or self.label_permutations.dtype != np.int32
            ):
                raise ValueError("label_permutations must match the exact int32 schedule shape")
            if type(self.example_indices) is not np.ndarray or (
                self.example_indices.shape
                != (len(self.seeds), self.config.n_tasks, self.config.task_length)
                or self.example_indices.dtype != np.int32
            ):
                raise ValueError("example_indices must match the exact int32 schedule shape")
            if type(self.initial_params) is not dict or not self.initial_params:
                raise ValueError("initial_params must be one non-empty exact dictionary")
            for name, array_value in self.initial_params.items():
                if type(name) is not str or not name or type(array_value) is not np.ndarray:
                    raise TypeError("initial_params must map exact strings to exact ndarrays")
                if not np.issubdtype(array_value.dtype, np.floating) or not np.isfinite(
                    array_value
                ).all():
                    raise ValueError("initial_params arrays must contain finite floating values")
        object.__setattr__(
            self,
            "wall_clock_seconds",
            _require_finite_real("wall_clock_seconds", self.wall_clock_seconds),
        )
        if self.wall_clock_seconds < 0.0:
            raise ValueError("wall_clock_seconds must be non-negative")


def resolve_hyperparameters(
    learner: object, overrides: dict[str, float] | None = None
) -> dict[str, float]:
    """Merge overrides into the learner's published defaults, rejecting unknown keys."""
    host_learner = _require_exact_str("learner", learner)
    if host_learner not in _LEARNER_DEFAULT_HYPERPARAMETERS:
        raise ValueError(
            f"unknown learner '{host_learner}'; expected one of "
            f"{sorted(_LEARNER_DEFAULT_HYPERPARAMETERS)}"
        )
    merged = dict(_LEARNER_DEFAULT_HYPERPARAMETERS[host_learner])
    if overrides:
        unknown = set(overrides) - set(merged)
        if unknown:
            raise ValueError(f"unknown hyperparameters for '{host_learner}': {sorted(unknown)}")
        validated: dict[str, float] = {}
        for name, value in overrides.items():
            validated[name] = _validated_hyperparameter(name, value)
        merged.update(validated)
    return merged


def run_label_emnist(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    learner: LabelEMNISTLearner,
    seeds: Sequence[int],
    config: LabelEMNISTConfig | None = None,
    hyperparameters: dict[str, float] | None = None,
    return_per_step: bool = False,
    progress_every: int | None = None,
) -> LabelEMNISTRunResult:
    """Run the online Label-permuted EMNIST protocol for one learner.

    Args:
        data_x: ``(n_train, input_dim)`` float32 inputs, already normalized
            to ``[-1, 1]`` (see :func:`load_emnist_balanced_train`).
        data_y: ``(n_train,)`` integer class labels in ``[0, n_classes)``.
        learner: ``"upgd_w"`` or ``"adamw"``.
        seeds: Run seeds; all seeds execute in parallel under ``vmap``.
        config: Task/network configuration (defaults to the selected
            publication shape).
        hyperparameters: Optional overrides of the published defaults.
        return_per_step: Also return per-step accuracies plus the initial
            parameters and schedules (debug/testing scale only).
        progress_every: Log progress every N tasks (None = silent).

    Returns:
        Host-side result arrays; see :class:`LabelEMNISTRunResult`.
    """
    if config is not None and type(config) is not LabelEMNISTConfig:
        raise TypeError("config must be an exact LabelEMNISTConfig or None")
    if type(return_per_step) is not bool:
        raise TypeError("return_per_step must be a boolean")
    if progress_every is not None and (
        type(progress_every) is not int or progress_every <= 0
    ):
        raise ValueError("progress_every must be a positive integer or None")
    seed_tuple = require_unique_jax_seeds(seeds, name="seeds")
    if config is None:
        config = LabelEMNISTConfig()
    resolved_x, resolved_y = validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        min_length=config.task_length,
    )
    hp = resolve_hyperparameters(learner, hyperparameters)
    init_fn, step_fn = _FULL_STEP_FACTORIES[learner](hp)

    data_x = jnp.asarray(resolved_x, dtype=jnp.float32)
    data_y = jnp.asarray(resolved_y, dtype=jnp.int32)
    n_train = int(data_x.shape[0])

    seeds_array = jnp.asarray(seed_tuple, dtype=jnp.uint32)
    initialization_config = IPMNISTConfig(**config.to_config())

    def init_seed(seed: Array) -> tuple[dict[str, Array], Any, LabelEMNISTSchedule, Array]:
        root = jr.key(seed)
        key_init, key_schedule, key_noise = jr.split(root, 3)
        params = init_mlp_params(key_init, initialization_config)
        return params, init_fn(params), build_schedule(key_schedule, config, n_train), key_noise

    params, opt_state, schedules, noise_keys = jax.jit(jax.vmap(init_seed))(seeds_array)

    initial_params_host: dict[str, np.ndarray] | None = None
    if return_per_step:
        initial_params_host = {name: np.asarray(value) for name, value in params.items()}

    def run_task(
        params: dict[str, Array],
        opt_state: Any,
        noise_key: Array,
        label_permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array, Array]:
        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], tuple[Array, Array, Array]]:
            step_params, step_state, key = carry
            x = data_x[example]
            y = label_permutation[data_y[example]]
            key, step_key = jr.split(key)
            new_params, new_state, (accuracy, loss, plasticity) = step_fn(
                step_params, step_state, x, y, step_key
            )
            return (new_params, new_state, key), (accuracy, loss, plasticity)

        (params, opt_state, noise_key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, opt_state, noise_key), examples
        )
        return params, opt_state, noise_key, accuracies, losses, plasticities

    run_task_batched = jax.jit(jax.vmap(run_task))

    task_accuracy: list[np.ndarray] = []
    task_loss: list[np.ndarray] = []
    task_plasticity: list[np.ndarray] = []
    step_accuracy: list[np.ndarray] = []
    started = time.monotonic()
    for task in range(config.n_tasks):
        params, opt_state, noise_keys, accuracies, losses, plasticities = run_task_batched(
            params,
            opt_state,
            noise_keys,
            schedules.label_permutations[:, task],
            schedules.example_indices[:, task],
        )
        task_accuracy.append(np.asarray(accuracies.mean(axis=1)))
        task_loss.append(np.asarray(losses.mean(axis=1)))
        task_plasticity.append(np.asarray(plasticities.mean(axis=1)))
        if return_per_step:
            step_accuracy.append(np.asarray(accuracies))
        if progress_every is not None and (task + 1) % progress_every == 0:
            elapsed = time.monotonic() - started
            logger.info(
                "%s task %d/%d online_acc=%.4f elapsed=%.1fs",
                learner,
                task + 1,
                config.n_tasks,
                float(task_accuracy[-1].mean()),
                elapsed,
            )

    per_task_accuracy = np.stack(task_accuracy, axis=1)
    return LabelEMNISTRunResult(
        learner=learner,
        hyperparameters=hp,
        seeds=seed_tuple,
        config=config,
        per_task_accuracy=per_task_accuracy,
        per_task_loss=np.stack(task_loss, axis=1),
        per_task_plasticity=np.stack(task_plasticity, axis=1),
        average_online_accuracy=per_task_accuracy.mean(axis=1),
        wall_clock_seconds=time.monotonic() - started,
        per_step_accuracy=np.stack(step_accuracy, axis=1) if return_per_step else None,
        initial_params=initial_params_host,
        label_permutations=np.asarray(schedules.label_permutations) if return_per_step else None,
        example_indices=np.asarray(schedules.example_indices) if return_per_step else None,
    )


# =============================================================================
# EMNIST loading (OpenML EMNIST_Balanced, data_id 41039)
# =============================================================================


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_openml_data_home() -> Path:
    """Return the ``fetch_openml`` ``data_home`` for the EMNIST cache."""
    return _repo_root() / "outputs" / "upgd_label_emnist" / "openml_cache"


def materialized_array_sha256(value: np.ndarray) -> str:
    """Digest an array's dtype, shape, and C-order bytes."""
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _npy_cache_paths(home: Path) -> tuple[Path, Path, Path]:
    return (
        home / "emnist_balanced_train_x.npy",
        home / "emnist_balanced_train_y.npy",
        home / "emnist_balanced_train_meta.json",
    )


def load_emnist_balanced_train(
    data_home: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the EMNIST balanced train split, scaled to ``[-1, 1]``.

    Uses ``fetch_openml(data_id=41039)`` (``EMNIST_Balanced``: 131,600 rows =
    train+test of the balanced split). The first 112,800 rows are kept as the
    train split only after verifying their per-class counts equal the
    torchvision balanced train split (2,400/class; remaining 18,800 rows
    400/class); a mismatch fails closed rather than silently changing the
    protocol. The transform matches ``ToTensor`` + ``Normalize((0.5,), (0.5,))``.

    The parsed arrays are cached as ``.npy`` beside the OpenML cache (the
    ARFF parse costs minutes; the cache loads in seconds) with digests pinned
    in a meta file; cache hits re-verify the digests.

    Returns:
        ``(x, y, info)`` where ``x`` is ``(112800, 784)`` float32 in
        ``[-1, 1]``, ``y`` is ``(112800,)`` int32 in ``[0, 47)``, and ``info``
        records provenance facts bound into plans.
    """
    home = data_home if data_home is not None else default_openml_data_home()
    x_path, y_path, meta_path = _npy_cache_paths(home)
    if x_path.is_file() and y_path.is_file() and meta_path.is_file():
        x = np.load(x_path)
        y = np.load(y_path)
        cached_meta = _strict_json_object(meta_path)
        if (
            materialized_array_sha256(x) != cached_meta["x_sha256"]
            or materialized_array_sha256(y) != cached_meta["y_sha256"]
        ):
            raise RuntimeError(f"EMNIST npy cache under {home} does not match its pinned digests")
        return x, y, dict(cached_meta)

    try:
        from sklearn.datasets import fetch_openml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scikit-learn is required to load OpenML EMNIST") from exc

    logger.info("fetching OpenML EMNIST_Balanced (data_id=%d) into %s", EMNIST_OPENML_DATA_ID, home)
    raw = fetch_openml(
        data_id=EMNIST_OPENML_DATA_ID,
        as_frame=False,
        data_home=str(home),
        n_retries=3,
        delay=2.0,
        parser="liac-arff",
    )
    x_all = np.asarray(raw.data, dtype=np.float32)
    y_all = np.asarray(raw.target, dtype=np.int32)
    if x_all.shape != (EMNIST_TOTAL_ROWS, 784) or y_all.shape != (EMNIST_TOTAL_ROWS,):
        raise RuntimeError(
            f"unexpected EMNIST_Balanced shape: x={x_all.shape} y={y_all.shape}"
        )
    counts_first = np.bincount(y_all[:EMNIST_TRAIN_ROWS], minlength=47)
    counts_tail = np.bincount(y_all[EMNIST_TRAIN_ROWS:], minlength=47)
    ordering_consistent = bool(
        (counts_first == EMNIST_TRAIN_PER_CLASS).all()
        and (counts_tail == EMNIST_TEST_PER_CLASS).all()
    )
    if not ordering_consistent:
        raise RuntimeError(
            "OpenML EMNIST_Balanced row order is not consistent with a "
            "train-then-test layout (per-class counts differ); refusing to "
            "guess the train split"
        )
    x = np.ascontiguousarray((x_all[:EMNIST_TRAIN_ROWS] / 255.0 - 0.5) / 0.5, dtype=np.float32)
    y = np.ascontiguousarray(y_all[:EMNIST_TRAIN_ROWS])
    meta: dict[str, Any] = {
        "source": f"openml:{EMNIST_OPENML_NAME}:data_id={EMNIST_OPENML_DATA_ID}",
        "total_rows": EMNIST_TOTAL_ROWS,
        "train_rows_used": EMNIST_TRAIN_ROWS,
        "train_order_check": "first_112800_rows_have_2400_per_class_and_tail_400_per_class",
        "input_scaling": "(x/255 - 0.5) / 0.5",
        "parser": "liac-arff",
        "x_sha256": materialized_array_sha256(x),
        "y_sha256": materialized_array_sha256(y),
    }
    home.mkdir(parents=True, exist_ok=True)
    np.save(x_path, x)
    np.save(y_path, y)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return x, y, meta


# =============================================================================
# Summaries, comparison
# =============================================================================


def _stderr(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        raise ValueError(
            "stderr is undefined for fewer than two observations; "
            "refusing to report a look-alike zero"
        )
    return float(values.std(ddof=1) / math.sqrt(values.shape[0]))


def summarize_result(result: LabelEMNISTRunResult) -> dict[str, Any]:
    """Reduce one learner's run to JSON-serializable summary statistics."""
    accuracy = result.average_online_accuracy
    if accuracy.shape[0] < 2:
        raise ValueError(
            "stderr is undefined for fewer than two observations; "
            "refusing to report a look-alike zero"
        )
    quarter = max(1, result.config.n_tasks // 4)
    per_seed_first = result.per_task_accuracy[:, :quarter].mean(axis=1)
    per_seed_last = result.per_task_accuracy[:, -quarter:].mean(axis=1)
    return {
        "learner": result.learner,
        "hyperparameters": result.hyperparameters,
        "seeds": list(result.seeds),
        "n_seeds": int(accuracy.shape[0]),
        "average_online_accuracy_mean": float(accuracy.mean()),
        "average_online_accuracy_stderr": _stderr(accuracy),
        "per_seed_average_online_accuracy": [round(float(v), 6) for v in accuracy],
        "first_quarter_accuracy_mean": float(per_seed_first.mean()),
        "last_quarter_accuracy_mean": float(per_seed_last.mean()),
        "accuracy_drift_last_minus_first": float(per_seed_last.mean() - per_seed_first.mean()),
        "average_plasticity_mean": float(result.per_task_plasticity.mean()),
        "per_task_accuracy_mean": [
            round(float(v), 6) for v in result.per_task_accuracy.mean(axis=0)
        ],
        "per_task_accuracy_stderr": [
            round(_stderr(result.per_task_accuracy[:, t]), 6)
            for t in range(result.config.n_tasks)
        ],
        "per_seed_per_task_accuracy": [
            [round(float(v), 6) for v in row] for row in result.per_task_accuracy
        ],
        "wall_clock_seconds": round(result.wall_clock_seconds, 2),
    }


def build_comparison(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare run summaries against the published reference read-offs."""
    reference = PAPER_REFERENCE["approximate_average_online_accuracy"]
    comparison: dict[str, Any] = {
        "reference": PAPER_REFERENCE,
        "gap_threshold": REPRODUCTION_GAP_THRESHOLD,
        "learners": {},
    }
    for learner, summary in summaries.items():
        if learner not in reference:
            continue
        ours = summary["average_online_accuracy_mean"]
        published = float(reference[learner])
        gap = ours - published
        comparison["learners"][learner] = {
            "ours": round(ours, 6),
            "published_approximate": published,
            "gap": round(gap, 6),
            "reproduction_gap_flagged": bool(abs(gap) > REPRODUCTION_GAP_THRESHOLD),
        }
    if "upgd_w" in summaries and "adamw" in summaries:
        comparison["upgd_w_beats_adamw"] = bool(
            summaries["upgd_w"]["average_online_accuracy_mean"]
            > summaries["adamw"]["average_online_accuracy_mean"]
        )
        drift = summaries["upgd_w"]["accuracy_drift_last_minus_first"]
        comparison["upgd_w_rises"] = bool(drift > 0.0)
    return comparison


def _environment_payload() -> dict[str, str]:
    return {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
    }


# =============================================================================
# plan / shard / merge lifecycle
# =============================================================================


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value independently of its locator or pretty encoding."""
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_plan_payload(
    config: LabelEMNISTConfig,
    seed_ids: Sequence[int],
    dataset_meta: Mapping[str, Any],
    learners: Sequence[str] = ("upgd_w", "adamw"),
    hyperparameter_overrides: Mapping[str, Mapping[str, float]] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the pre-run plan binding config, arms, seeds, and dataset digests."""
    raw_seeds = tuple(seed_ids)
    if not raw_seeds:
        raise ValueError("a plan requires at least one seed")
    seeds = require_unique_jax_seeds(raw_seeds, name="seed IDs")
    if seeds != tuple(sorted(seeds)):
        raise ValueError("seed IDs must be sorted")
    learner_ids = tuple(learners)
    if not learner_ids or len(set(learner_ids)) != len(learner_ids):
        raise ValueError("learner IDs must be a non-empty unique sequence")
    overrides = hyperparameter_overrides or {}
    unknown_override_learners = set(overrides) - set(learner_ids)
    if unknown_override_learners:
        raise ValueError(f"overrides name unplanned learners: {sorted(unknown_override_learners)}")
    hyperparameters = {
        learner: resolve_hyperparameters(learner, dict(overrides.get(learner, {})))
        for learner in learner_ids
    }
    required_meta = {"source", "train_rows_used", "x_sha256", "y_sha256"}
    missing = required_meta - set(dataset_meta)
    if missing:
        raise ValueError(f"dataset_meta is missing required fields: {sorted(missing)}")
    body = {
        "benchmark": BENCHMARK,
        "config": {**config.to_config(), "n_steps": config.n_steps},
        "matches_selected_publication_configuration": (
            config.matches_selected_publication_configuration
        ),
        "learner_ids": list(learner_ids),
        "hyperparameters": hyperparameters,
        "seed_ids": list(seeds),
        "planned_shard_count": len(learner_ids) * len(seeds),
        "dataset": dict(dataset_meta),
        "deviations": [dict(deviation) for deviation in PROTOCOL_DEVIATIONS],
        "paper_reference": PAPER_REFERENCE,
        "notes": list(notes),
    }
    return {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "created_unix": int(time.time()),
        "plan": body,
        "plan_sha256": canonical_json_sha256(body),
    }


def _strict_json_object(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            host_key = _require_exact_str("key", key)
            if host_key in parsed:
                raise ValueError(f"duplicate JSON key: '{host_key}'")
            parsed[host_key] = value
        return parsed

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number is forbidden: {value}")
        return parsed

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
        parse_float=parse_float,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: payload must be one JSON object")
    return payload


def _validated_float_hyperparameters(
    value: object, learner: str, *, context: str
) -> dict[str, float]:
    """Require every hyperparameter to be an exact finite ``float``.

    Python's ``0 == 0.0`` and ``True == 1.0`` make ``float(v)`` coercion and
    dict ``==`` alias-tolerant, so an int/bool-aliased arm could slip through
    intake and be measured under a different hyperparameter vector than the
    registered one. Mirror ``ipmnist_screening``'s strict gate instead:
    ``type(...) is float`` (rejecting bool, the int subclass, and every other
    alias) plus finiteness, before any comparison.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{context}: {learner} hyperparameters must be an object")
    invalid = sorted(
        str(name)
        for name, entry in value.items()
        if type(entry) is not float or not math.isfinite(entry)
    )
    if invalid:
        raise ValueError(
            f"{context}: {learner} hyperparameters must be finite floats "
            f"(int/bool aliases are forbidden); invalid field(s): {invalid}"
        )
    return {str(name): cast(float, entry) for name, entry in value.items()}


def load_plan(path: Path) -> dict[str, Any]:
    """Load and structurally validate a v1 plan file."""
    payload = _strict_json_object(path)
    if payload.get("schema") != PLAN_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError(f"{path} is not a {PLAN_SCHEMA} payload")
    if payload.get("evidence_policy") != NONPROMOTING_POLICY:
        raise ValueError(f"{path}: evidence policy must remain permanently nonpromoting")
    body = payload.get("plan")
    if not isinstance(body, dict):
        raise ValueError(f"{path}: plan body must be an object")
    if payload.get("plan_sha256") != canonical_json_sha256(body):
        raise ValueError(f"{path}: plan_sha256 does not match the plan body")
    if body.get("benchmark") != BENCHMARK:
        raise ValueError(f"{path}: plan benchmark is not {BENCHMARK}")
    config_payload = dict(body["config"])
    n_steps = config_payload.pop("n_steps", None)
    config = LabelEMNISTConfig(**config_payload)
    if n_steps != config.n_steps:
        raise ValueError(f"{path}: plan n_steps is inconsistent with config")
    learner_ids = body["learner_ids"]
    if sorted(body["hyperparameters"]) != sorted(learner_ids):
        raise ValueError(f"{path}: hyperparameter learner keys differ from learner_ids")
    for learner in learner_ids:
        provided = _validated_float_hyperparameters(
            body["hyperparameters"][learner], learner, context=str(path)
        )
        resolved = resolve_hyperparameters(learner, provided)
        if sorted(resolved) != sorted(provided) or any(
            resolved[name].hex() != provided[name].hex() for name in provided
        ):
            raise ValueError(
                f"{path}: {learner} hyperparameters must list the complete arm"
            )
    raw_seeds = body.get("seed_ids")
    if not isinstance(raw_seeds, list):
        raise ValueError(f"{path}: plan seed_ids must be a list")
    seeds = require_unique_jax_seeds(raw_seeds, name=f"{path}: plan seed_ids")
    if seeds != tuple(sorted(seeds)):
        raise ValueError(f"{path}: plan seed_ids must be sorted")
    if body["planned_shard_count"] != len(learner_ids) * len(seeds):
        raise ValueError(f"{path}: planned_shard_count is inconsistent")
    return payload


def partial_payload(
    result: LabelEMNISTRunResult, plan_sha256: str
) -> dict[str, Any]:
    """Serialize one single-seed run shard bound to its plan hash."""
    seeds = require_unique_jax_seeds(result.seeds, name="result seeds")
    if len(seeds) != 1:
        raise ValueError("a v1 partial must contain exactly one seed")
    return {
        "schema": PARTIAL_SCHEMA,
        "schema_version": 1,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "plan_sha256": plan_sha256,
        "learner": result.learner,
        "hyperparameters": result.hyperparameters,
        "seed_id": seeds[0],
        "config": result.config.to_config(),
        "per_task_accuracy": [round(float(v), 6) for v in result.per_task_accuracy[0]],
        "per_task_loss": [round(float(v), 6) for v in result.per_task_loss[0]],
        "per_task_plasticity": [round(float(v), 6) for v in result.per_task_plasticity[0]],
        "average_online_accuracy": round(float(result.average_online_accuracy[0]), 6),
        "wall_clock_seconds": round(result.wall_clock_seconds, 2),
    }


def _validated_partial(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    payload = _strict_json_object(path)
    if payload.get("schema") != PARTIAL_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError(f"{path} is not a {PARTIAL_SCHEMA} payload")
    if payload.get("evidence_policy") != NONPROMOTING_POLICY:
        raise ValueError(f"{path}: evidence policy must remain permanently nonpromoting")
    body = plan["plan"]
    if payload.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError(f"{path}: shard is bound to a different plan")
    learner = _require_exact_str("learner", payload.get("learner"))
    if learner not in body["learner_ids"]:
        raise ValueError(f"{path}: learner '{learner}' is not planned")
    shard_hp = _validated_float_hyperparameters(
        payload.get("hyperparameters"), str(learner), context=str(path)
    )
    planned_hp = _validated_float_hyperparameters(
        body["hyperparameters"][learner], str(learner), context=f"{path}: plan"
    )
    if sorted(shard_hp) != sorted(planned_hp) or any(
        shard_hp[name].hex() != planned_hp[name].hex() for name in shard_hp
    ):
        raise ValueError(f"{path}: hyperparameters differ from the plan")
    seed = require_jax_seed(payload.get("seed_id"), name=f"{path}: seed_id")
    if seed not in body["seed_ids"]:
        host_seed = payload.get("seed_id")
        if type(host_seed) is int and not isinstance(host_seed, bool):
            raise ValueError(f"{path}: seed_id '{host_seed}' is not planned")
        if type(host_seed) is str:
            host_s = _require_exact_str("seed_id", host_seed)
            raise ValueError(f"{path}: seed_id '{host_s}' is not planned")
        raise ValueError(f"{path}: seed_id is not planned")
    plan_config = {k: v for k, v in body["config"].items() if k != "n_steps"}
    if payload.get("config") != plan_config:
        raise ValueError(f"{path}: config differs from the plan")
    n_tasks = plan_config["n_tasks"]
    for field, (lower, upper) in {
        "per_task_accuracy": (0.0, 1.0),
        "per_task_loss": (0.0, None),
        "per_task_plasticity": (0.0, 1.0),
    }.items():
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != n_tasks:
            raise ValueError(f"{path}: {field} must be a list of {n_tasks} numbers")
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)) or not np.all(array >= lower):
            raise ValueError(f"{path}: {field} values are outside the allowed range")
        if upper is not None and not np.all(array <= upper):
            raise ValueError(f"{path}: {field} values are outside the allowed range")
    mean_accuracy = float(np.asarray(payload["per_task_accuracy"], dtype=np.float64).mean())
    if abs(mean_accuracy - float(payload["average_online_accuracy"])) > 1e-5:
        raise ValueError(f"{path}: average_online_accuracy does not match per-task mean")
    return payload


def merge_partials(
    plan: dict[str, Any],
    paths: Sequence[Path],
    allow_incomplete: bool = False,
) -> tuple[dict[str, LabelEMNISTRunResult], dict[str, Any]]:
    """Merge shard files against a plan, enforcing exact planned coverage.

    Args:
        plan: A validated plan payload (see :func:`load_plan`).
        paths: Shard JSON paths.
        allow_incomplete: Permit missing planned pairs; the coverage payload
            then records them explicitly and marks the merge incomplete.

    Returns:
        ``(results_by_learner, coverage)``.
    """
    if not paths:
        raise ValueError("no partial result files given")
    body = plan["plan"]
    config = LabelEMNISTConfig(
        **{k: v for k, v in body["config"].items() if k != "n_steps"}
    )
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        payload = _validated_partial(Path(path), plan)
        identity = (payload["learner"], payload["seed_id"])
        if identity in seen:
            raise ValueError(f"duplicate shard for learner={identity[0]} seed={identity[1]}")
        seen[identity] = payload
    planned = {
        (learner, seed) for learner in body["learner_ids"] for seed in body["seed_ids"]
    }
    missing = sorted(planned - set(seen))
    if missing and not allow_incomplete:
        raise ValueError(f"missing planned shards: {missing}")
    coverage = {
        "planned_shard_count": len(planned),
        "merged_shard_count": len(seen),
        "complete": not missing,
        "missing_pairs": [[learner, seed] for learner, seed in missing],
    }
    results: dict[str, LabelEMNISTRunResult] = {}
    for learner in body["learner_ids"]:
        shards = sorted(
            (payload for (name, _), payload in seen.items() if name == learner),
            key=lambda payload: payload["seed_id"],
        )
        if not shards:
            continue
        per_task_accuracy = np.asarray(
            [shard["per_task_accuracy"] for shard in shards], dtype=np.float64
        )
        results[learner] = LabelEMNISTRunResult(
            learner=learner,
            hyperparameters=dict(shards[0]["hyperparameters"]),
            seeds=tuple(shard["seed_id"] for shard in shards),
            config=config,
            per_task_accuracy=per_task_accuracy,
            per_task_loss=np.asarray(
                [shard["per_task_loss"] for shard in shards], dtype=np.float64
            ),
            per_task_plasticity=np.asarray(
                [shard["per_task_plasticity"] for shard in shards], dtype=np.float64
            ),
            average_online_accuracy=per_task_accuracy.mean(axis=1),
            wall_clock_seconds=float(sum(shard["wall_clock_seconds"] for shard in shards)),
        )
    return results, coverage


def build_artifact(
    plan: dict[str, Any],
    results: dict[str, LabelEMNISTRunResult],
    coverage: dict[str, Any],
    partial_paths: Sequence[Path] = (),
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the permanently nonpromoting merge artifact."""
    if not results:
        raise ValueError("at least one learner result is required")
    summaries = {learner: summarize_result(result) for learner, result in results.items()}
    manifest = []
    for path_value in sorted(Path(p) for p in partial_paths):
        raw = path_value.read_bytes()
        manifest.append(
            {
                "path": path_value.as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "created_unix": int(time.time()),
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "plan_sha256": plan["plan_sha256"],
        "plan": plan["plan"],
        "coverage": coverage,
        "partial_manifest": manifest,
        "learners": summaries,
        "comparison": build_comparison(summaries),
        "environment": _environment_payload(),
        "notes": list(notes),
    }


# =============================================================================
# CLI
# =============================================================================


def _cmd_plan(args: argparse.Namespace) -> None:
    config = LabelEMNISTConfig(n_tasks=args.n_tasks, task_length=args.task_length)
    raw_seeds = [int(part) for part in args.seed_list.split(",") if part.strip()]
    seeds = require_unique_jax_seeds(raw_seeds, name="seed IDs")
    data_home = args.data_home if args.data_home is not None else default_openml_data_home()
    _, _, meta = load_emnist_balanced_train(data_home)
    learners = [name.strip() for name in args.learners.split(",") if name.strip()]
    payload = build_plan_payload(config, seeds, meta, learners, notes=args.note)
    atomic_write_new_json(args.plan_out, payload)
    logger.info("wrote plan %s (plan_sha256=%s)", args.plan_out, payload["plan_sha256"])


def _cmd_shard(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    body = plan["plan"]
    host_lid = _require_exact_str("learner_id", args.learner_id)
    if host_lid not in body["learner_ids"]:
        raise SystemExit(f"learner '{host_lid}' is not planned")
    if args.seed_id not in body["seed_ids"]:
        raise SystemExit(f"seed {args.seed_id} is not planned")
    data_home = args.data_home if args.data_home is not None else default_openml_data_home()
    data_x, data_y, meta = load_emnist_balanced_train(data_home)
    for field in ("x_sha256", "y_sha256"):
        if meta[field] != body["dataset"][field]:
            raise SystemExit(f"dataset {field} differs from the plan; refusing to run")
    config = LabelEMNISTConfig(
        **{k: v for k, v in body["config"].items() if k != "n_steps"}
    )
    hp = {k: float(v) for k, v in body["hyperparameters"][args.learner_id].items()}
    logger.info(
        "shard learner=%s seed=%d tasks=%d task_length=%d",
        args.learner_id,
        args.seed_id,
        config.n_tasks,
        config.task_length,
    )
    result = run_label_emnist(
        data_x,
        data_y,
        args.learner_id,
        [args.seed_id],
        config=config,
        hyperparameters=hp,
        progress_every=args.progress_every,
    )
    atomic_write_new_json(args.partial_out, partial_payload(result, plan["plan_sha256"]))
    logger.info(
        "wrote shard %s (avg online acc %.4f, wall %.1fs)",
        args.partial_out,
        float(result.average_online_accuracy[0]),
        result.wall_clock_seconds,
    )


def _cmd_merge(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    results, coverage = merge_partials(
        plan, args.partials, allow_incomplete=args.allow_incomplete
    )
    artifact = build_artifact(
        plan, results, coverage, partial_paths=args.partials, notes=args.note
    )
    atomic_write_new_json(args.output, artifact)
    logger.info(
        "merged %d shards -> %s (coverage complete=%s)",
        coverage["merged_shard_count"],
        args.output,
        coverage["complete"],
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: ``plan`` / ``shard`` / ``merge``."""
    parser = argparse.ArgumentParser(
        description="UPGD Label-permuted EMNIST replication lane (nonpromoting)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="issue an immutable pre-run plan")
    plan_parser.add_argument("--plan-out", type=Path, required=True)
    plan_parser.add_argument("--seed-list", required=True,
                             help="explicit comma-separated sorted unique seeds")
    plan_parser.add_argument("--learners", default="upgd_w,adamw")
    plan_parser.add_argument("--n-tasks", type=int, default=400)
    plan_parser.add_argument("--task-length", type=int, default=2500)
    plan_parser.add_argument("--data-home", type=Path, default=None)
    plan_parser.add_argument("--note", action="append", default=[])
    plan_parser.set_defaults(func=_cmd_plan)

    shard_parser = sub.add_parser("shard", help="run exactly one planned learner x seed")
    shard_parser.add_argument("--plan", type=Path, required=True)
    shard_parser.add_argument("--learner-id", required=True)
    shard_parser.add_argument("--seed-id", type=int, required=True)
    shard_parser.add_argument("--partial-out", type=Path, required=True)
    shard_parser.add_argument("--data-home", type=Path, default=None)
    shard_parser.add_argument("--progress-every", type=int, default=10)
    shard_parser.set_defaults(func=_cmd_shard)

    merge_parser = sub.add_parser("merge", help="merge planned shards into an artifact")
    merge_parser.add_argument("--plan", type=Path, required=True)
    merge_parser.add_argument("--partials", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--allow-incomplete", action="store_true",
                              help="permit missing planned shards; coverage records them")
    merge_parser.add_argument("--note", action="append", default=[])
    merge_parser.set_defaults(func=_cmd_merge)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    args.func(args)


__all__ = [
    "ADAMW_PROTOCOL_HYPERPARAMETERS",
    "ARTIFACT_SCHEMA",
    "BENCHMARK",
    "EMNIST_OPENML_DATA_ID",
    "EMNIST_TRAIN_ROWS",
    "LabelEMNISTConfig",
    "LabelEMNISTRunResult",
    "LabelEMNISTSchedule",
    "PAPER_REFERENCE",
    "PARTIAL_SCHEMA",
    "PLAN_SCHEMA",
    "PROTOCOL_DEVIATIONS",
    "REPRODUCTION_GAP_THRESHOLD",
    "SGD_EMA_NORM_HYPERPARAMETERS",
    "UPGD_EMA_NORM_HYPERPARAMETERS",
    "UPGD_EMA_NORM_SIGMA0_HYPERPARAMETERS",
    "UPGD_W_PROTOCOL_HYPERPARAMETERS",
    "build_artifact",
    "build_comparison",
    "build_plan_payload",
    "build_schedule",
    "canonical_json_sha256",
    "default_openml_data_home",
    "load_emnist_balanced_train",
    "load_plan",
    "main",
    "materialized_array_sha256",
    "merge_partials",
    "partial_payload",
    "resolve_hyperparameters",
    "run_label_emnist",
    "summarize_result",
    "task_index_for_step",
]


if __name__ == "__main__":
    main()
