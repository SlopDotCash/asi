"""UPGD Input-permuted MNIST replication runner (Elsayed & Mahmood, ICLR 2024).

This lane implements a development replication diagnostic based on the
*online Input-permuted MNIST* protocol of

    Elsayed, M. & Mahmood, A. R. (2024). Addressing Loss of Plasticity and
    Catastrophic Forgetting in Continual Learning. ICLR 2024.
    https://openreview.net/forum?id=sKPzAXoylB

using the selected network, task shape, horizon, and hyperparameters from the
authors' MIT-licensed repository
(https://github.com/mohmdelsayed/upgd, audited at commit
``b75e90ad4b09c28971ac9dbb902a8fd86709b28c`` -- the same commit profiled by
:mod:`alberta_framework.core.canonical_upgd`):

- 1,000,000 MNIST training examples presented one per step (batch size 1),
  drawn from the canonical 60,000-example train split; each 5,000-step task
  reshuffles the split and takes the first 5,000 examples without replacement
  (``core/task/input_permuted_mnist.py`` recreates its ``DataLoader`` with
  ``shuffle=True`` at every permutation).
- A fresh input-pixel permutation every 5,000 steps => 200 tasks. The first
  task is itself permuted (upstream calls ``permute()`` at step 0).
- Inputs scaled to ``[-1, 1]`` (torchvision ``ToTensor`` then
  ``Normalize((0.5,), (0.5,))``).
- Network: 784 -> 300 -> ReLU -> 150 -> ReLU -> 10 MLP
  (``core/network/fcn_relu.py`` ``FullyConnectedReLU``), PyTorch default
  ``nn.Linear`` init: weights and biases ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
- Loss: 10-class softmax cross-entropy on the single example.
- Metric: online accuracy of the prediction made *before* each update,
  averaged per task; headline number = mean over tasks (= mean over steps
  because tasks have equal length here). Plasticity per step is
  ``clip(1 - loss_after_update / max(loss, 1e-8), 0, 1)`` on the same example
  (``core/run/run_stats.py``).
- Learners (best published hyperparameters, from
  ``experiments/statistics_input_permuted_mnist.py`` at the audited commit):

  * **UPGD-W** = ``FirstOrderGlobalUPGDLearner`` -- utility-gated perturbed
    SGD with decoupled weight decay. Implemented here by
    :class:`~alberta_framework.core.canonical_upgd.CanonicalUPGD` with
    ``profile="official_experiment_global"``, ``mode="protecting"``, which is
    equation-exact against the released ``FirstOrderGlobalUPGD`` optimizer.
    Best config: ``lr=0.01, beta_utility=0.9999, sigma=0.1,
    weight_decay=0.01``.
  * **AdamW** = the released ``core/optim/adam.py`` Adam, which applies
    decoupled weight decay (``p.data.add_(p.data, alpha=-wd*lr)`` before the
    Adam step). Implemented by
    :class:`~alberta_framework.core.baseline_optimizers.Adam` with
    ``weight_decay``. Best config: ``lr=1e-4, beta1=0.0, beta2=0.99,
    eps=1e-8, weight_decay=0.0`` (the paper's grid selected zero decay, so
    the published "Adam" baseline *is* AdamW at wd=0).

Documented deviations from the released code (therefore this is not a claim of
complete published-protocol exactness):

- Upstream draws each task's pixel permutation from an *unseeded*
  ``numpy.random.default_rng()``, so its permutation sequence is not
  reproducible. Here every stream (init, permutations, per-task example
  draws, perturbation noise) derives from the run seed via ``jr.key`` /
  ``jr.fold_in``.
- Upstream's per-task logging buffer flushes at ``i % 5000 == 0`` *after*
  step ``i`` runs, so each logged block covers steps ``[5000t+1, 5000(t+1)]``
  -- shifted one step past the permutation boundary -- and the final 4,999
  steps are dropped. This lane uses exact blocks ``[5000t, 5000(t+1))``
  aligned with the permutation schedule; the difference is 1 step in 5,000.
- Bias corrections are computed in float32 (upstream mixes float64 Python
  scalars into float32 tensors).
- Exact ``upgd_w`` runs use :func:`lean_upgd_w_update`, a scan-optimized
  restatement of the ``CanonicalUPGD`` ``official_experiment_global``
  protecting profile (the canonical transform's finiteness/mask bookkeeping
  triples CPU step time on this all-finite protocol). A supplied-noise parity
  unit test pins it to the canonical implementation exactly. Perturbations
  are drawn as one flat ``N(0, sigma^2)`` vector per step and sliced per
  parameter -- same distribution as upstream's per-tensor ``randn_like``.
- ``run_ipmnist(noise_mode="pool")`` is a screening-only fast mode that
  replaces the dominant per-step perturbation generation with random slices
  of a per-task noise pool (see the noise-pool section below); pool results
  are refused by :func:`partial_payload` and never enter stored artifacts.

MNIST arrives through the same OpenML plumbing the step2 runners use
(``sklearn.datasets.fetch_openml("mnist_784", version=1)``); the loader
reuses the existing step2 cache when present instead of re-downloading. The
first 60,000 OpenML rows are the canonical torchvision train split.

The module retains the historical v2 one-seed shard and merge interface for
reproducing existing development records. It has no direct aggregate mode.

Benchmark executions happen through this CLI, never inside pytest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import (
    require_jax_seed,
    require_unique_jax_seeds,
)
from alberta_framework.core.baseline_optimizers import Adam
from alberta_framework.core.canonical_upgd import CanonicalUPGD, CanonicalUPGDConfig
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    select_transaction,
)

logger = logging.getLogger(__name__)


def _preflight_new_output(path: Path) -> Path:
    """Reject an occupied output before an expensive benchmark starts."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
    return destination


def atomic_write_new(path: Path, data: bytes) -> Path:
    """Publish complete bytes at a new path without replacing existing data."""

    destination = _preflight_new_output(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o444)
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {destination}"
            ) from exc
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_new_json(path: Path, value: object) -> Path:
    """Publish a deterministic JSON document at a new path."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return atomic_write_new(path, (encoded + "\n").encode("utf-8"))

IPMNISTLearner = Literal["upgd_w", "adamw"]

PARTIAL_SCHEMA_V1 = "upgd_ipmnist.partial.v1"
PARTIAL_SCHEMA = "alberta.upgd_ipmnist.partial.v2"
ARTIFACT_SCHEMA = "alberta.upgd_ipmnist.artifact.v2"

# Structured, factual departures from the selected released implementation.
# These are emitted by v2 shards and artifacts; they are not a self-attested
# protocol-conformance conclusion.
PROTOCOL_DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "code": "seeded_streams",
        "scope": "rng_schedule",
        "description": (
            "permutation and example streams are seed-derived; upstream permutations are unseeded"
        ),
    },
    {
        "code": "task_aligned_logging",
        "scope": "metric_blocks",
        "description": (
            "task blocks are [t*L, (t+1)*L); upstream logging is shifted by one step"
        ),
    },
    {
        "code": "float32_bias_corrections",
        "scope": "numeric_precision",
        "description": "bias corrections remain float32; upstream mixes float64 scalars",
    },
)

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_replication_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
    "execution_attestation": False,
}

#: Best published hyperparameters for the statistics (headline-figure) runs,
#: from ``experiments/statistics_input_permuted_mnist.py`` at commit
#: ``b75e90ad4b09c28971ac9dbb902a8fd86709b28c``.
UPGD_W_PROTOCOL_HYPERPARAMETERS: dict[str, float] = {
    "step_size": 0.01,
    "utility_decay": 0.9999,
    "noise_std": 0.1,
    "weight_decay": 0.01,
}
ADAMW_PROTOCOL_HYPERPARAMETERS: dict[str, float] = {
    "step_size": 1e-4,
    "beta1": 0.0,
    "beta2": 0.99,
    "eps": 1e-8,
    "weight_decay": 0.0,
}

#: Published reference numbers this lane compares against. The accuracies are
#: approximate read-offs from the paper's Input-permuted MNIST online-accuracy
#: figure (the paper reports curves, not a table); the hyperparameters and
#: selected settings come from the released code. The paper averages 20 seeds.
PAPER_REFERENCE: dict[str, Any] = {
    "citation": (
        "Elsayed & Mahmood (2024). Addressing Loss of Plasticity and "
        "Catastrophic Forgetting in Continual Learning. ICLR 2024."
    ),
    "openreview": "https://openreview.net/forum?id=sKPzAXoylB",
    "official_repository": "https://github.com/mohmdelsayed/upgd",
    "official_commit": "b75e90ad4b09c28971ac9dbb902a8fd86709b28c",
    "protocol_files": [
        "core/task/input_permuted_mnist.py",
        "core/network/fcn_relu.py",
        "core/run/run_stats.py",
        "experiments/statistics_input_permuted_mnist.py",
        "core/optim/weight_upgd/first_order.py",
        "core/optim/adam.py",
    ],
    "n_seeds": 20,
    "approximate_average_online_accuracy": {
        "upgd_w": 0.78,
        "adamw": 0.68,
        "s_ewc_s_si_s_mas": [0.70, 0.72],
    },
    "qualitative": (
        "UPGD-W holds ~0.78 flat across all 200 tasks; Adam(W) decays toward "
        "~0.68; streaming EWC/SI/MAS land between them."
    ),
    "reference_kind": "figure_readoff",
}

#: Absolute reproduction-gap threshold beyond which a run must be flagged for
#: investigation rather than reported as a clean reproduction.
REPRODUCTION_GAP_THRESHOLD = 0.02

_PLASTICITY_LOSS_FLOOR = 1e-8


@dataclass(frozen=True)
class IPMNISTConfig:
    """Input-permuted MNIST protocol configuration.

    Defaults match the selected ICLR-2024 network and task configuration; tests
    shrink every field. Configuration equality alone is not protocol exactness.

    Args:
        n_tasks: Number of permutation tasks (published: 200).
        task_length: Online steps per task (published: 5,000).
        input_dim: Flattened input dimensionality (published: 784).
        hidden1: First hidden layer width (published: 300).
        hidden2: Second hidden layer width (published: 150).
        n_classes: Output classes (published: 10).
    """

    n_tasks: int = 200
    task_length: int = 5000
    input_dim: int = 784
    hidden1: int = 300
    hidden2: int = 150
    n_classes: int = 10

    def __post_init__(self) -> None:
        for name in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def n_steps(self) -> int:
        """Total online steps (examples) in a run."""
        return self.n_tasks * self.task_length

    @property
    def matches_selected_publication_configuration(self) -> bool:
        """Whether shape and horizon match the selected publication configuration."""
        return self == IPMNISTConfig()

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
class IPMNISTSchedule:
    """Per-seed permutation and example schedule.

    Attributes:
        permutations: ``(n_tasks, input_dim)`` int32; row ``t`` is the pixel
            permutation active for every step of task ``t``.
        example_indices: ``(n_tasks, task_length)`` int32; row ``t`` holds the
            dataset rows presented during task ``t``, drawn without
            replacement from a fresh per-task shuffle of the train split.
    """

    permutations: Array
    example_indices: Array


def task_index_for_step(step: Array | int, task_length: int) -> Array | int:
    """Return the task index active at ``step`` (permutes every ``task_length``)."""
    return step // task_length


def build_schedule(key: Array, config: IPMNISTConfig, n_train: int) -> IPMNISTSchedule:
    """Build the deterministic permutation/example schedule for one seed.

    Args:
        key: Per-seed schedule key.
        config: Protocol configuration.
        n_train: Number of rows in the train split; must be at least
            ``config.task_length`` so tasks sample without replacement.

    Returns:
        The schedule; task ``t`` is exactly steps
        ``[t * task_length, (t + 1) * task_length)``.
    """
    if n_train < config.task_length:
        raise ValueError(
            f"n_train={n_train} is smaller than task_length={config.task_length}; "
            "per-task sampling is without replacement"
        )
    key_perm, key_sample = jr.split(key)
    tasks = jnp.arange(config.n_tasks)
    permutations = jax.vmap(
        lambda task: jr.permutation(jr.fold_in(key_perm, task), config.input_dim)
    )(tasks)
    example_indices = jax.vmap(
        lambda task: jr.permutation(jr.fold_in(key_sample, task), n_train)[: config.task_length]
    )(tasks)
    return IPMNISTSchedule(  # type: ignore[call-arg]
        permutations=permutations.astype(jnp.int32),
        example_indices=example_indices.astype(jnp.int32),
    )


def init_mlp_params(key: Array, config: IPMNISTConfig) -> dict[str, Array]:
    """Initialize the 300x150 ReLU MLP with PyTorch-default Linear init.

    ``nn.Linear.reset_parameters`` draws weights from
    ``kaiming_uniform_(a=sqrt(5))`` = U(-1/sqrt(fan_in), +1/sqrt(fan_in)) and
    biases from the same interval.
    """
    sizes = [
        (config.input_dim, config.hidden1),
        (config.hidden1, config.hidden2),
        (config.hidden2, config.n_classes),
    ]
    keys = jr.split(key, 2 * len(sizes))
    params: dict[str, Array] = {}
    for index, (fan_in, fan_out) in enumerate(sizes):
        bound = 1.0 / math.sqrt(fan_in)
        params[f"w{index + 1}"] = jr.uniform(
            keys[2 * index], (fan_in, fan_out), jnp.float32, -bound, bound
        )
        params[f"b{index + 1}"] = jr.uniform(
            keys[2 * index + 1], (fan_out,), jnp.float32, -bound, bound
        )
    return params


_REAL_NUMERIC_DTYPE_KINDS = frozenset({"i", "u", "f"})
"""Concrete dtype kinds admitted as ``data_x``: signed/unsigned integers and floats.

An allowlist is load-bearing here: ``np.issubdtype(np.timedelta64, np.number)``
is true, and a ``NaT`` timedelta casts to a *finite* float32 sentinel, so a
``number``-based check would let non-finite-in-spirit data through the finite
gate.  Kind codes exclude bool (``b``), complex (``c``), timedelta (``m``),
datetime (``M``), and every non-numeric kind.
"""


def validated_ipmnist_data(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    *,
    input_dim: int | None,
    n_classes: int | None,
    min_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data_x, data_y)`` as finite float32 / int32 arrays inside the protocol domain.

    JAX gathers clamp out-of-range indices instead of raising, so a label at
    or above ``n_classes`` would silently be scored and trained as the last
    class, and a non-finite input would still yield an in-range accuracy.
    Every runner that indexes the softmax by label must go through here
    before touching any hyperparameter resolver or learner factory.

    Args:
        data_x: Candidate ``(n_train, input_dim)`` example matrix.
        data_y: Candidate ``(n_train,)`` integer label vector.
        input_dim: Required example width (``None`` = any).
        n_classes: Exclusive label upper bound (``None`` = any).
        min_length: Required minimum row count -- pass ``config.task_length``
            so per-task sampling without replacement is feasible
            (``None`` = any).

    Raises:
        ValueError: If ``data_x`` is not a finite ``(n_train, input_dim)``
            matrix of a real numeric dtype kind with at least ``min_length``
            rows, or ``data_y`` is not an aligned vector of integer labels in
            ``[0, n_classes)``.
    """
    raw_x = np.asarray(jax.device_get(data_x))
    raw_y = np.asarray(jax.device_get(data_y))
    if raw_x.ndim != 2:
        raise ValueError("data_x must be a two-dimensional example matrix")
    if input_dim is not None and raw_x.shape[1] != input_dim:
        raise ValueError(f"data_x must have shape (n_train, {input_dim})")
    if raw_y.shape != (raw_x.shape[0],):
        raise ValueError("data_y must be (n_train,) aligned with data_x")
    if min_length is not None and raw_x.shape[0] < min_length:
        raise ValueError(
            "dataset smaller than task_length; cannot sample without replacement"
        )
    if raw_x.dtype.kind not in _REAL_NUMERIC_DTYPE_KINDS:
        raise ValueError("data_x must use a real numeric, non-boolean dtype")
    if raw_y.dtype.kind not in {"i", "u"}:
        raise ValueError("data_y must contain integer class labels")
    if np.any(raw_y < 0) or np.any(raw_y > np.iinfo(np.int32).max):
        raise ValueError("data_y class labels must fit non-negative int32")
    resolved_x = np.asarray(raw_x, dtype=np.float32)
    resolved_y = np.asarray(raw_y, dtype=np.int32)
    if not np.all(np.isfinite(resolved_x)):
        raise ValueError("data_x must contain only finite values")
    if n_classes is not None and np.any(resolved_y >= n_classes):
        raise ValueError(f"data_y class labels must be smaller than {n_classes}")
    return resolved_x, resolved_y


def mlp_logits(params: dict[str, Array], x: Array) -> Array:
    """Forward pass of the protocol MLP for a single flattened example."""
    hidden = jax.nn.relu(x @ params["w1"] + params["b1"])
    hidden = jax.nn.relu(hidden @ params["w2"] + params["b2"])
    return hidden @ params["w3"] + params["b3"]


def cross_entropy_loss(
    params: dict[str, Array], x: Array, y: Array
) -> tuple[Array, Array]:
    """Softmax cross-entropy of one example; returns ``(loss, logits)``."""
    logits = mlp_logits(params, x)
    return -jax.nn.log_softmax(logits)[y], logits


LearnerInitFn = Callable[[dict[str, Array]], Any]


@chex.dataclass(frozen=True, mappable_dataclass=False)
class LearnerUpdateResult:
    """Checked learner step with backward-compatible two-value unpacking.

    Existing benchmark and screening callers unpack learner steps as
    ``params, state = step_fn(...)``.  Iteration deliberately preserves that
    surface while ``update_applied`` makes an atomic rejection observable to
    callers that need the checked contract.
    """

    params: dict[str, Array]
    state: Any
    update_applied: Array

    def __iter__(self) -> Iterator[Any]:
        yield self.params
        yield self.state


LearnerStepFn = Callable[
    [dict[str, Array], Any, dict[str, Array], Array],
    tuple[dict[str, Array], Any] | LearnerUpdateResult,
]


@chex.dataclass(frozen=True)
class LeanUPGDState:
    """Carry state of the lean UPGD-W step: raw utility EMA + global clock."""

    utility: dict[str, Array]
    step: Array


def canonical_upgd_w(hp: dict[str, float]) -> CanonicalUPGD:
    """The audited reference optimizer the lean step must match exactly."""
    return CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=hp["step_size"],
            utility_decay=hp["utility_decay"],
            noise_std=hp["noise_std"],
            weight_decay=hp["weight_decay"],
            mode="protecting",
            profile="official_experiment_global",
            normalization=None,
        )
    )


def lean_upgd_w_update(
    params: dict[str, Array],
    state: LeanUPGDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: dict[str, float],
) -> tuple[dict[str, Array], LeanUPGDState]:
    """One UPGD-W step, equation-identical to ``FirstOrderGlobalUPGD``.

    This is a scan-optimized restatement of
    ``CanonicalUPGD(profile="official_experiment_global", mode="protecting")``
    for the all-finite, no-mask benchmark inner loop (the canonical transform
    spends most of its time on finiteness/mask bookkeeping this protocol
    never exercises). ``tests/test_upgd_ipmnist.py`` holds a supplied-noise
    parity test asserting exact agreement with the canonical implementation;
    any equation change must keep that test passing.

    ``noise`` is the already-scaled perturbation ``xi ~ N(0, sigma^2)``.
    """
    beta = hp["utility_decay"]
    step_size = hp["step_size"]
    decay = 1.0 - step_size * hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility = {
        name: beta * state.utility[name] + (1.0 - beta) * (-grads[name] * params[name])
        for name in params
    }
    global_max = jnp.max(
        jnp.stack([jnp.max(utility[name]) for name in sorted(params)])
    )
    bias_correction = 1.0 - jnp.power(
        jnp.asarray(beta, dtype=jnp.float32), count.astype(jnp.float32)
    )
    new_params = {}
    for name in params:
        gate = jax.nn.sigmoid((utility[name] / bias_correction) / global_max)
        new_params[name] = params[name] * decay - step_size * (
            (grads[name] + noise[name]) * (1.0 - gate)
        )
    return new_params, LeanUPGDState(utility=utility, step=count)  # type: ignore[call-arg]


# =============================================================================
# Screening-only noise pool (OpenAI-ES-style shared noise table)
# =============================================================================
#
# ``noise_mode="pool"`` replaces the dominant per-step cost of the UPGD-W
# lane -- generating a fresh 282,160-element ``N(0, sigma^2)`` perturbation
# vector every step (~80%+ of single-core step time; threefry bits + erfinv
# have no cheaper bit-exact implementation on CPU) -- with one pool of
# ``noise_pool_steps`` steps' worth of normals regenerated per task and a
# random contiguous slice per step. Per-step marginals stay exactly
# ``N(0, sigma^2)`` and elements within a step stay independent, but values
# are REUSED across steps at random alignments, so pool trajectories are a
# screening-only approximation: :func:`partial_payload` refuses them and they
# must never feed stored artifacts. The update equations are the
# unchanged parity-tested :func:`lean_upgd_w_update` consuming supplied noise.


#: Learners whose update consumes a perturbation (pool mode applies to these;
#: for the rest, pool mode falls back to the exact deterministic step).
_STOCHASTIC_LEARNERS: frozenset[str] = frozenset({"upgd_w"})


def _sorted_param_shapes(config: IPMNISTConfig) -> dict[str, tuple[int, ...]]:
    """Parameter shapes of the protocol MLP keyed by name."""
    return {
        "b1": (config.hidden1,),
        "b2": (config.hidden2,),
        "b3": (config.n_classes,),
        "w1": (config.input_dim, config.hidden1),
        "w2": (config.hidden1, config.hidden2),
        "w3": (config.hidden2, config.n_classes),
    }


def _split_flat_noise(
    flat: Array, shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, Array]:
    """Slice one flat noise vector per sorted parameter name (the lane's order)."""
    out: dict[str, Array] = {}
    offset = 0
    for name in sorted(shapes):
        count = int(np.prod(shapes[name]))
        out[name] = flat[offset:offset + count].reshape(shapes[name])
        offset += count
    return out


def _make_upgd_w_learner(hp: dict[str, float]) -> tuple[LearnerInitFn, LearnerStepFn]:
    """UPGD-W (``FirstOrderGlobalUPGD``) via the lean parity-tested step."""
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def step_fn(
        params: dict[str, Array], state: LeanUPGDState, grads: dict[str, Array], key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState]:
        names = sorted(params)
        shapes = [params[name].shape for name in names]
        counts = [int(np.prod(shape)) for shape in shapes]
        flat = jr.normal(key, (sum(counts),), jnp.float32) * noise_std
        chunks = jnp.split(flat, np.cumsum(counts)[:-1])
        noise = {
            name: chunk.reshape(shape)
            for name, chunk, shape in zip(names, chunks, shapes, strict=True)
        }
        return lean_upgd_w_update(params, state, grads, noise, hp)

    return init_fn, step_fn


def _make_adamw_learner(hp: dict[str, float]) -> tuple[LearnerInitFn, LearnerStepFn]:
    """AdamW via the baseline Adam with decoupled weight decay."""
    optimizer = Adam(
        step_size=hp["step_size"],
        beta1=hp["beta1"],
        beta2=hp["beta2"],
        eps=hp["eps"],
        weight_decay=hp["weight_decay"],
    )

    def init_fn(params: dict[str, Array]) -> dict[str, Any]:
        return {name: optimizer.init_for_shape(value.shape) for name, value in params.items()}

    def step_fn(
        params: dict[str, Array], state: dict[str, Any], grads: dict[str, Array], key: Array
    ) -> LearnerUpdateResult:
        del key  # AdamW is deterministic
        candidate_params: dict[str, Array] = {}
        candidate_state: dict[str, Any] = {}
        update_applied = jnp.asarray(True, dtype=jnp.bool_)
        for name, value in params.items():
            leaf_update = optimizer.update_from_gradient_checked(
                state[name], grads[name], error=None, param=value
            )
            candidate_params[name] = value - leaf_update.step
            candidate_state[name] = leaf_update.new_state
            update_applied = update_applied & leaf_update.update_applied

        # A finite optimizer step can still overflow when it is applied to a
        # finite parameter.  The learner owns the enclosing transaction, so
        # validate and select the complete parameter and optimizer-state
        # trees only after every leaf candidate has been assembled.
        update_applied = (
            update_applied
            & floating_tree_is_finite(params)
            & floating_tree_is_finite(state)
            & floating_tree_is_finite(grads)
            & floating_tree_is_finite(candidate_params)
            & floating_tree_is_finite(candidate_state)
        )
        return LearnerUpdateResult(
            params=select_transaction(update_applied, candidate_params, params),
            state=select_transaction(update_applied, candidate_state, state),
            update_applied=update_applied,
        )  # type: ignore[call-arg]

    return init_fn, step_fn


_LEARNER_FACTORIES: dict[str, Callable[[dict[str, float]], tuple[LearnerInitFn, LearnerStepFn]]] = {
    "upgd_w": _make_upgd_w_learner,
    "adamw": _make_adamw_learner,
}
_LEARNER_DEFAULT_HYPERPARAMETERS: dict[str, dict[str, float]] = {
    "upgd_w": UPGD_W_PROTOCOL_HYPERPARAMETERS,
    "adamw": ADAMW_PROTOCOL_HYPERPARAMETERS,
}


@dataclass(frozen=True)
class IPMNISTRunResult:
    """Host-side result of one learner's multi-seed run.

    Per-task arrays have shape ``(n_seeds, n_tasks)``. ``per_step_accuracy``
    (``(n_seeds, n_tasks, task_length)``), ``initial_params``,
    ``permutations``, and ``example_indices`` are populated only when the run
    was invoked with ``return_per_step=True`` (debug/testing scale).
    """

    learner: str
    hyperparameters: dict[str, float]
    seeds: tuple[int, ...]
    config: IPMNISTConfig
    per_task_accuracy: np.ndarray
    per_task_loss: np.ndarray
    per_task_plasticity: np.ndarray
    average_online_accuracy: np.ndarray
    wall_clock_seconds: float
    per_step_accuracy: np.ndarray | None = None
    initial_params: dict[str, np.ndarray] | None = None
    permutations: np.ndarray | None = None
    example_indices: np.ndarray | None = None
    noise_mode: str = "step"


def resolve_hyperparameters(
    learner: str, overrides: dict[str, float] | None = None
) -> dict[str, float]:
    """Merge overrides into the learner's published defaults, rejecting unknown keys."""
    if learner not in _LEARNER_DEFAULT_HYPERPARAMETERS:
        raise ValueError(f"unknown learner {learner!r}; expected one of "
                         f"{sorted(_LEARNER_DEFAULT_HYPERPARAMETERS)}")
    merged = dict(_LEARNER_DEFAULT_HYPERPARAMETERS[learner])
    if overrides:
        unknown = set(overrides) - set(merged)
        if unknown:
            raise ValueError(f"unknown hyperparameters for {learner}: {sorted(unknown)}")
        for name, value in overrides.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"hyperparameter {name!r} must be a finite int or float")
        merged.update({name: float(value) for name, value in overrides.items()})
    return merged


def run_ipmnist(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    learner: IPMNISTLearner,
    seeds: Sequence[int],
    config: IPMNISTConfig | None = None,
    hyperparameters: dict[str, float] | None = None,
    return_per_step: bool = False,
    progress_every: int | None = None,
    noise_mode: Literal["step", "pool"] = "step",
    noise_pool_steps: int = 64,
) -> IPMNISTRunResult:
    """Run the online Input-permuted MNIST protocol for one learner.

    Under the default ``noise_mode="step"`` the inner loop is the historical
    dict-based loop (bit-compatible with the completed lane results, modulo
    XLA version); ``noise_mode="pool"`` swaps only the noise source.

    Args:
        data_x: ``(n_train, input_dim)`` float32 inputs, already normalized
            to ``[-1, 1]`` (see :func:`load_mnist_train`).
        data_y: ``(n_train,)`` integer class labels.
        learner: ``"upgd_w"`` or ``"adamw"``.
        seeds: Run seeds; all seeds execute in parallel under ``vmap``.
        config: Task/network configuration (defaults to the selected publication shape).
        hyperparameters: Optional overrides of the published defaults.
        return_per_step: Also return per-step accuracies plus the initial
            parameters and schedules (debug/testing scale only -- the full
            protocol would materialize ``n_seeds x 1e6`` floats).
        progress_every: Log progress every N tasks (None = silent).
        noise_mode: ``"step"`` (default, protocol-exact: one fresh
            ``N(0, sigma^2)`` vector per step) or ``"pool"`` -- a
            SCREENING-ONLY approximation that regenerates a noise pool of
            ``noise_pool_steps`` steps' worth of normals once per task and
            slices each step's perturbation at a random offset (OpenAI-ES
            style shared noise table). Per-step marginals stay exactly
            ``N(0, sigma^2)`` but values are reused across steps, so pool
            results are refused by :func:`partial_payload` and must never
            feed stored artifacts.
        noise_pool_steps: Pool size in steps (>= 2) for ``noise_mode="pool"``;
            memory is ``noise_pool_steps * n_params * 4`` bytes per seed.

    Returns:
        Host-side result arrays; see :class:`IPMNISTRunResult`.
    """
    seed_tuple = require_unique_jax_seeds(seeds, name="seeds")
    if config is None:
        config = IPMNISTConfig()
    if noise_mode not in ("step", "pool"):
        raise ValueError(f"noise_mode must be 'step' or 'pool', got {noise_mode!r}")
    if noise_mode == "pool" and noise_pool_steps < 2:
        raise ValueError(f"noise_pool_steps must be >= 2, got {noise_pool_steps}")
    resolved_x, resolved_y = validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        min_length=config.task_length,
    )
    hp = resolve_hyperparameters(learner, hyperparameters)
    init_fn, step_fn = _LEARNER_FACTORIES[learner](hp)
    shapes = _sorted_param_shapes(config)
    n_flat = int(sum(np.prod(shape) for shape in shapes.values()))

    data_x = jnp.asarray(resolved_x, dtype=jnp.float32)
    data_y = jnp.asarray(resolved_y, dtype=jnp.int32)
    n_train = int(data_x.shape[0])

    seeds_array = jnp.asarray(seed_tuple, dtype=jnp.uint32)

    use_pool = noise_mode == "pool" and learner in _STOCHASTIC_LEARNERS
    pool_len = int(noise_pool_steps) * n_flat if use_pool else 0
    pool_noise_std = float(hp["noise_std"]) if use_pool else 0.0

    def init_seed(seed: Array) -> tuple[dict[str, Array], Any, IPMNISTSchedule, Array]:
        root = jr.key(seed)
        key_init, key_schedule, key_noise = jr.split(root, 3)
        params = init_mlp_params(key_init, config)
        return params, init_fn(params), build_schedule(key_schedule, config, n_train), key_noise

    params, opt_state, schedules, noise_keys = jax.jit(jax.vmap(init_seed))(seeds_array)

    initial_params_host: dict[str, np.ndarray] | None = None
    if return_per_step:
        initial_params_host = {name: np.asarray(value) for name, value in params.items()}

    def run_task(
        params: dict[str, Array],
        opt_state: Any,
        noise_key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array, Array]:
        if use_pool:
            noise_key, pool_key = jr.split(noise_key)
            pool = jr.normal(pool_key, (pool_len,), jnp.float32) * pool_noise_std

        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], tuple[Array, Array, Array]]:
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                step_params, x, y
            )
            accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
            key, step_key = jr.split(key)
            if use_pool:
                offset = jr.randint(step_key, (), 0, pool_len - n_flat + 1)
                flat_noise = jax.lax.dynamic_slice(pool, (offset,), (n_flat,))
                noise = _split_flat_noise(flat_noise, shapes)
                new_params, new_state = lean_upgd_w_update(
                    step_params, step_state, grads, noise, hp
                )
            else:
                new_params, new_state = step_fn(step_params, step_state, grads, step_key)
            loss_after, _ = cross_entropy_loss(new_params, x, y)
            plasticity = jnp.clip(
                1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
            )
            return (new_params, new_state, key), (accuracy, loss, plasticity)

        (params, opt_state, noise_key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, opt_state, noise_key), examples
        )
        return params, opt_state, noise_key, accuracies, losses, plasticities

    run_task_batched = jax.jit(jax.vmap(run_task), donate_argnums=(0, 1, 2))

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
            schedules.permutations[:, task],
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
    per_task_loss = np.stack(task_loss, axis=1)
    per_task_plasticity = np.stack(task_plasticity, axis=1)
    return IPMNISTRunResult(
        learner=learner,
        hyperparameters=hp,
        seeds=seed_tuple,
        config=config,
        per_task_accuracy=per_task_accuracy,
        per_task_loss=per_task_loss,
        per_task_plasticity=per_task_plasticity,
        average_online_accuracy=per_task_accuracy.mean(axis=1),
        wall_clock_seconds=time.monotonic() - started,
        per_step_accuracy=np.stack(step_accuracy, axis=1) if return_per_step else None,
        initial_params=initial_params_host,
        permutations=np.asarray(schedules.permutations) if return_per_step else None,
        example_indices=np.asarray(schedules.example_indices) if return_per_step else None,
        noise_mode=noise_mode,
    )


# =============================================================================
# MNIST loading (same OpenML plumbing as the step2 runners)
# =============================================================================

_STEP2_CACHE_NAME = "step2_published_mnist_openml_cache"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_openml_data_home() -> Path:
    """Return the ``fetch_openml`` ``data_home`` to use.

    Preference order: an existing step2 MNIST cache in this repository's
    ``outputs/``; an existing step2 cache in a sibling full upstream clone
    (``../../alberta/outputs/``); otherwise a fresh cache under
    ``outputs/upgd_ipmnist/openml_cache`` (triggers one download).
    """
    root = _repo_root()
    candidates = (
        root / "outputs" / _STEP2_CACHE_NAME,
        root.parent.parent / "alberta" / "outputs" / _STEP2_CACHE_NAME,
    )
    for candidate in candidates:
        if (candidate / "openml").is_dir():
            return candidate
    return root / "outputs" / "upgd_ipmnist" / "openml_cache"


def load_mnist_train(data_home: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the canonical 60,000-example MNIST train split, scaled to ``[-1, 1]``.

    Uses ``sklearn.datasets.fetch_openml("mnist_784", version=1)`` -- the same
    plumbing as the step2 runners -- whose first 60,000 rows are the
    torchvision train split the published protocol uses. The transform
    matches ``ToTensor`` + ``Normalize((0.5,), (0.5,))``.
    """
    try:
        from sklearn.datasets import fetch_openml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scikit-learn is required to load OpenML MNIST") from exc

    home = data_home if data_home is not None else default_openml_data_home()
    raw = fetch_openml(
        "mnist_784", version=1, as_frame=False, data_home=str(home), n_retries=3, delay=2.0
    )
    x = np.asarray(raw.data, dtype=np.float32)[:60_000]
    y = np.asarray(raw.target, dtype=np.int32)[:60_000]
    x = (x / 255.0 - 0.5) / 0.5
    return x, y


# =============================================================================
# Summaries, comparison, artifact
# =============================================================================


def summarize_result(result: IPMNISTRunResult) -> dict[str, Any]:
    """Reduce one learner's run to JSON-serializable summary statistics."""
    accuracy = result.average_online_accuracy
    n_seeds = accuracy.shape[0]
    quarter = max(1, result.config.n_tasks // 4)
    per_seed_first = result.per_task_accuracy[:, :quarter].mean(axis=1)
    per_seed_last = result.per_task_accuracy[:, -quarter:].mean(axis=1)

    def _stderr(values: np.ndarray) -> float:
        if values.shape[0] < 2:
            return 0.0
        return float(values.std(ddof=1) / math.sqrt(values.shape[0]))

    return {
        "learner": result.learner,
        "hyperparameters": result.hyperparameters,
        "seeds": list(result.seeds),
        "n_seeds": n_seeds,
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
        "per_task_plasticity_mean": [
            round(float(v), 6) for v in result.per_task_plasticity.mean(axis=0)
        ],
        "per_seed_per_task_accuracy": [
            [round(float(v), 6) for v in row] for row in result.per_task_accuracy
        ],
        "wall_clock_seconds": round(result.wall_clock_seconds, 2),
    }


def build_comparison(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare run summaries against the published reference numbers.

    Any absolute gap above :data:`REPRODUCTION_GAP_THRESHOLD` is flagged as a
    reproduction gap to investigate -- never silently absorbed.
    """
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
    return comparison


def _protocol_payload(config: IPMNISTConfig) -> dict[str, object]:
    """Return factual protocol/configuration fields without a conformance claim."""
    return {
        **config.to_config(),
        "n_steps": config.n_steps,
        "matches_selected_publication_configuration": (
            config.matches_selected_publication_configuration
        ),
        "selected_publication_configuration_match_scope": "network_task_shape_and_horizon_only",
        "dataset": "OpenML mnist_784 v1, first 60000 rows (torchvision train split)",
        "input_scaling": "(x/255 - 0.5) / 0.5",
        "loss": "softmax cross-entropy, one example per step",
        "metric": "online accuracy of the pre-update prediction, averaged per task",
        "plasticity": "clip(1 - loss_after/max(loss, 1e-8), 0, 1) per step",
    }


def _environment_payload() -> dict[str, str]:
    return {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
    }


def _validate_result_set(
    results: Mapping[str, IPMNISTRunResult], config: IPMNISTConfig
) -> None:
    if not results:
        raise ValueError("at least one learner result is required")
    for learner, result in results.items():
        if learner != result.learner:
            raise ValueError(f"result key {learner!r} does not match payload learner")
        if result.config != config:
            raise ValueError(f"{learner}: result config does not match artifact config")
        require_unique_jax_seeds(result.seeds, name=f"{learner} seeds")


def _study_design_payload(results: Mapping[str, IPMNISTRunResult]) -> dict[str, object]:
    seed_ids = {learner: list(result.seeds) for learner, result in sorted(results.items())}
    seed_counts = {learner: len(seeds) for learner, seeds in seed_ids.items()}
    schedules = list(seed_ids.values())
    all_share = all(seeds == schedules[0] for seeds in schedules[1:])
    return {
        "published_seed_count": int(PAPER_REFERENCE["n_seeds"]),
        "exact_seed_ids_by_learner": seed_ids,
        "exact_seed_count_by_learner": seed_counts,
        "all_learners_share_seed_ids": all_share,
        "matches_published_seed_count": all(
            count == int(PAPER_REFERENCE["n_seeds"]) for count in seed_counts.values()
        ),
    }


def build_legacy_v1_artifact(
    results: dict[str, IPMNISTRunResult],
    config: IPMNISTConfig,
    data_home: Path,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Rebuild the immutable v1 payload shape for compatibility tests only.

    The legacy ``protocol.is_protocol_exact`` field meant equality with the
    selected default configuration.  It is preserved here only so the completed
    v1 diagnostic can be independently recomputed; active CLI output uses v2.
    """
    _validate_result_set(results, config)
    summaries = {learner: summarize_result(result) for learner, result in results.items()}
    return {
        "benchmark": "upgd_input_permuted_mnist",
        "schema_version": 1,
        "created_unix": time.time(),
        "protocol": {
            **config.to_config(),
            "n_steps": config.n_steps,
            "is_protocol_exact": config.matches_selected_publication_configuration,
            "dataset": "OpenML mnist_784 v1, first 60000 rows (torchvision train split)",
            "input_scaling": "(x/255 - 0.5) / 0.5",
            "loss": "softmax cross-entropy, one example per step",
            "metric": "online accuracy of the pre-update prediction, averaged per task",
            "plasticity": "clip(1 - loss_after/max(loss, 1e-8), 0, 1) per step",
        },
        "provenance": {
            "openml_data_home": str(data_home),
            "deviations": [
                "seeded permutation/example streams (upstream permutations are unseeded)",
                "exact task blocks [t*L, (t+1)*L) (upstream logging is shifted one step)",
                "float32 bias corrections (upstream mixes float64 scalars)",
            ],
        },
        "learners": summaries,
        "comparison": build_comparison(summaries),
        "environment": _environment_payload(),
        "notes": list(notes),
    }


def build_artifact(
    results: dict[str, IPMNISTRunResult],
    config: IPMNISTConfig,
    data_home: Path,
    notes: Sequence[str] = (),
    *,
    partial_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Assemble a strict, permanently nonpromoting v2 result artifact."""
    _validate_result_set(results, config)
    summaries = {learner: summarize_result(result) for learner, result in results.items()}
    return {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": 2,
        "benchmark": "upgd_input_permuted_mnist",
        "created_unix": time.time(),
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "protocol": _protocol_payload(config),
        "study_design": _study_design_payload(results),
        "deviations": [dict(deviation) for deviation in PROTOCOL_DEVIATIONS],
        "partial_manifest": _v2_partial_manifest(partial_paths),
        "provenance": {
            "openml_data_home": str(data_home),
            "worker_source_bound": False,
            "full_import_closure_bound": False,
            "worker_command_bound": False,
            "worker_environment_bound": False,
            "dataset_bytes_bound": False,
        },
        "learners": summaries,
        "comparison": build_comparison(summaries),
        "environment": _environment_payload(),
        "notes": list(notes),
    }


# =============================================================================
# Partial (per-seed shard) results -- one process per seed parallelism
# =============================================================================


_V1_PARTIAL_FIELDS = {
    "schema",
    "learner",
    "hyperparameters",
    "seeds",
    "config",
    "per_task_accuracy",
    "per_task_loss",
    "per_task_plasticity",
    "wall_clock_seconds",
}
_V2_PARTIAL_FIELDS = {
    "schema",
    "schema_version",
    "evidence_policy",
    "learner",
    "hyperparameters",
    "seed_id",
    "seed_count",
    "config",
    "matches_selected_publication_configuration",
    "selected_publication_configuration_match_scope",
    "deviations",
    "per_task_accuracy",
    "per_task_loss",
    "per_task_plasticity",
    "wall_clock_seconds",
}


def partial_payload(result: IPMNISTRunResult) -> dict[str, Any]:
    """Serialize one run shard with the strict, nonpromoting v2 schema."""
    seeds = require_unique_jax_seeds(result.seeds, name="result seeds")
    if len(seeds) != 1:
        raise ValueError("a v2 partial must contain exactly one seed")
    if result.noise_mode != "step":
        raise ValueError(
            "only exact per-step noise results may become v2 shards; got "
            f"noise_mode={result.noise_mode!r} (screening-only approximation)"
        )
    return {
        "schema": PARTIAL_SCHEMA,
        "schema_version": 2,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "learner": result.learner,
        "hyperparameters": result.hyperparameters,
        "seed_id": seeds[0],
        "seed_count": 1,
        "config": result.config.to_config(),
        "matches_selected_publication_configuration": (
            result.config.matches_selected_publication_configuration
        ),
        "selected_publication_configuration_match_scope": (
            "network_task_shape_and_horizon_only"
        ),
        "deviations": [dict(deviation) for deviation in PROTOCOL_DEVIATIONS],
        "per_task_accuracy": result.per_task_accuracy.tolist(),
        "per_task_loss": result.per_task_loss.tolist(),
        "per_task_plasticity": result.per_task_plasticity.tolist(),
        "wall_clock_seconds": result.wall_clock_seconds,
    }


def _decode_strict_json_object(raw: bytes, *, path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate JSON key: {key!r}")
            parsed[key] = value
        return parsed

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number is forbidden: {value}")
        return parsed

    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
        parse_float=parse_float,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: payload must be one JSON object")
    return payload


def _strict_json_object(path: Path) -> dict[str, Any]:
    resolved = Path(path)
    return _decode_strict_json_object(resolved.read_bytes(), path=resolved)


def _v2_partial_manifest(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Bind supplied v2 shard bytes to exact learner/seed identities."""
    def identity(entry: Mapping[str, object]) -> tuple[str, int]:
        learner = entry["learner"]
        if not isinstance(learner, str):
            raise ValueError("v2 partial manifest contains an invalid identity")
        seed = require_jax_seed(
            entry["seed_id"], name="v2 partial manifest seed_id"
        )
        return learner, seed

    entries: list[dict[str, object]] = []
    for path_value in paths:
        path = Path(path_value)
        raw = path.read_bytes()
        payload = _decode_strict_json_object(raw, path=path)
        if payload.get("schema") != PARTIAL_SCHEMA:
            raise ValueError(f"{path}: partial manifest accepts only strict v2 shards")
        learner = payload.get("learner")
        if not isinstance(learner, str):
            raise ValueError(f"{path}: partial manifest identity is invalid")
        seed = require_jax_seed(
            payload.get("seed_id"), name=f"{path}: partial manifest seed_id"
        )
        entries.append(
            {
                "learner": learner,
                "seed_id": seed,
                "path": path.as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return sorted(entries, key=identity)


def _validated_partial_payload(
    path: Path, *, schema: str, seed_field: str
) -> dict[str, Any]:
    payload = _strict_json_object(path)
    expected_fields = _V2_PARTIAL_FIELDS if schema == PARTIAL_SCHEMA else _V1_PARTIAL_FIELDS
    if set(payload) != expected_fields:
        raise ValueError(f"{path}: fields do not match {schema}")
    if payload.get("schema") != schema:
        raise ValueError(f"{path} is not a {schema} payload")
    if schema == PARTIAL_SCHEMA:
        if payload.get("schema_version") != 2:
            raise ValueError(f"{path}: schema_version must be 2")
        if payload.get("evidence_policy") != NONPROMOTING_POLICY:
            raise ValueError(f"{path}: evidence policy must remain permanently nonpromoting")
        if payload.get("deviations") != list(PROTOCOL_DEVIATIONS):
            raise ValueError(f"{path}: structured deviations do not match the v2 contract")

    learner = payload.get("learner")
    if not isinstance(learner, str) or learner not in _LEARNER_FACTORIES:
        raise ValueError(f"{path}: unsupported learner")
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping) or not hyperparameters:
        raise ValueError(f"{path}: hyperparameters must be a non-empty object")
    for name, value in hyperparameters.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{path}: hyperparameters must be finite named numbers")
        if not math.isfinite(float(value)):
            raise ValueError(f"{path}: hyperparameters must be finite named numbers")

    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping) or set(config_payload) != set(
        IPMNISTConfig().to_config()
    ):
        raise ValueError(f"{path}: config fields do not match the v2 contract")
    try:
        config = IPMNISTConfig(**dict(config_payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid config: {exc}") from exc
    if schema == PARTIAL_SCHEMA and payload.get(
        "matches_selected_publication_configuration"
    ) is not config.matches_selected_publication_configuration:
        raise ValueError(f"{path}: selected-publication configuration marker is inconsistent")
    if schema == PARTIAL_SCHEMA and payload.get(
        "selected_publication_configuration_match_scope"
    ) != "network_task_shape_and_horizon_only":
        raise ValueError(f"{path}: selected-publication match scope is unsupported")

    raw_seeds = payload.get(seed_field)
    if schema == PARTIAL_SCHEMA:
        seed_ids = list(
            require_unique_jax_seeds((raw_seeds,), name=f"{path}: seed_id")
        )
        seed_count = payload.get("seed_count")
        if type(seed_count) is not int or seed_count != 1:
            raise ValueError(f"{path}: seed_count must equal one")
    else:
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ValueError(f"{path}: {seed_field} must be a non-empty list")
        seed_ids = list(
            require_unique_jax_seeds(raw_seeds, name=f"{path}: {seed_field}")
        )

    expected_shape = (len(seed_ids), config.n_tasks)
    matrix_bounds = {
        "per_task_accuracy": (0.0, 1.0, True),
        "per_task_loss": (0.0, None, False),
        "per_task_plasticity": (0.0, 1.0, True),
    }
    for field, (lower, upper, inclusive) in matrix_bounds.items():
        try:
            matrix = np.asarray(payload.get(field), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {field} must be a numeric matrix") from exc
        if matrix.shape != expected_shape or not bool(np.all(np.isfinite(matrix))):
            raise ValueError(f"{path}: {field} must have finite shape {expected_shape}")
        lower_ok = matrix >= lower if inclusive else matrix > lower
        if not bool(np.all(lower_ok)) or (upper is not None and not bool(np.all(matrix <= upper))):
            raise ValueError(f"{path}: {field} values are outside the allowed range")
    wall_clock = payload.get("wall_clock_seconds")
    if (
        isinstance(wall_clock, bool)
        or not isinstance(wall_clock, (int, float))
        or not math.isfinite(float(wall_clock))
        or float(wall_clock) <= 0.0
    ):
        raise ValueError(f"{path}: wall_clock_seconds must be finite and positive")
    return payload


def _merge_partial_results(
    paths: Sequence[Path], *, schema: str, seed_field: str
) -> dict[str, IPMNISTRunResult]:
    """Merge strict shard payloads into one result per learner.

    All shards of a learner must share config and hyperparameters; seeds must
    not repeat. Rows are ordered by seed.
    """
    if not paths:
        raise ValueError("no partial result files given")
    by_learner: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = _validated_partial_payload(Path(path), schema=schema, seed_field=seed_field)
        by_learner.setdefault(payload["learner"], []).append(payload)

    merged: dict[str, IPMNISTRunResult] = {}
    for learner, shards in by_learner.items():
        reference = shards[0]
        for shard in shards[1:]:
            if shard["config"] != reference["config"]:
                raise ValueError(f"{learner}: shards disagree on config")
            if shard["hyperparameters"] != reference["hyperparameters"]:
                raise ValueError(f"{learner}: shards disagree on hyperparameters")
        rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for shard in shards:
            accuracy = np.asarray(shard["per_task_accuracy"], dtype=np.float64)
            loss = np.asarray(shard["per_task_loss"], dtype=np.float64)
            plasticity = np.asarray(shard["per_task_plasticity"], dtype=np.float64)
            shard_seeds = [shard[seed_field]] if schema == PARTIAL_SCHEMA else shard[seed_field]
            for row_index, seed in enumerate(shard_seeds):
                rows.append(
                    (int(seed), accuracy[row_index], loss[row_index], plasticity[row_index])
                )
        seeds_seen = [row[0] for row in rows]
        if len(set(seeds_seen)) != len(seeds_seen):
            raise ValueError(f"{learner}: duplicate seeds across shards: {sorted(seeds_seen)}")
        rows.sort(key=lambda row: row[0])
        per_task_accuracy = np.stack([row[1] for row in rows])
        merged[learner] = IPMNISTRunResult(
            learner=learner,
            hyperparameters=dict(reference["hyperparameters"]),
            seeds=tuple(row[0] for row in rows),
            config=IPMNISTConfig(**reference["config"]),
            per_task_accuracy=per_task_accuracy,
            per_task_loss=np.stack([row[2] for row in rows]),
            per_task_plasticity=np.stack([row[3] for row in rows]),
            average_online_accuracy=per_task_accuracy.mean(axis=1),
            wall_clock_seconds=float(sum(shard["wall_clock_seconds"] for shard in shards)),
        )
    return merged


def merge_partial_results(paths: Sequence[Path]) -> dict[str, IPMNISTRunResult]:
    """Merge only strict v2 shards; legacy v1 shards fail closed."""
    return _merge_partial_results(paths, schema=PARTIAL_SCHEMA, seed_field="seed_id")


def merge_legacy_v1_partial_results(paths: Sequence[Path]) -> dict[str, IPMNISTRunResult]:
    """Merge immutable v1 shards for historical recomputation only."""
    return _merge_partial_results(paths, schema=PARTIAL_SCHEMA_V1, seed_field="seeds")


def main_v2_compat(argv: Sequence[str] | None = None) -> None:
    """Historical development-v2 shard and merge CLI.

    One-seed shard writing and historical v2 shard merging remain available.
    The unverifiable direct aggregate mode is intentionally disabled.
    """
    parser = argparse.ArgumentParser(
        description="LEGACY DEVELOPMENT ONLY: UPGD IPMNIST v2 compatibility"
    )
    parser.add_argument("--learners", default="upgd_w,adamw",
                        help="comma-separated subset of: upgd_w, adamw")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed value")
    parser.add_argument("--seed-list", default=None,
                        help="explicit comma-separated seeds (overrides --seeds/--seed-start)")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--task-length", type=int, default=5000)
    parser.add_argument("--data-home", type=Path, default=None,
                        help="fetch_openml data_home (defaults to the step2 cache when present)")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/upgd_ipmnist/results.schema-v2.json"))
    parser.add_argument("--partial-out", type=Path, default=None,
                        help="write a mergeable per-seed shard JSON here instead of an artifact")
    parser.add_argument("--merge-partials", type=Path, nargs="+", default=None,
                        help="skip running; merge strict v2 shard JSONs into --output")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--note", action="append", default=[],
                        help="free-form provenance note recorded in the artifact")
    args = parser.parse_args(argv)

    if args.partial_out is None and args.merge_partials is None:
        parser.error(
            "direct aggregate output is disabled; use --partial-out or --merge-partials"
        )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

    config = IPMNISTConfig(n_tasks=args.n_tasks, task_length=args.task_length)
    data_home = args.data_home if args.data_home is not None else default_openml_data_home()

    if args.merge_partials is not None:
        results = merge_partial_results(args.merge_partials)
        merged_configs = {result.config for result in results.values()}
        if len(merged_configs) != 1:
            raise SystemExit("shards span multiple protocol configs; merge them separately")
        artifact = build_artifact(
            results,
            merged_configs.pop(),
            data_home,
            notes=args.note,
            partial_paths=args.merge_partials,
        )
        atomic_write_new_json(args.output, artifact)
        logger.info("merged %d shards -> %s", len(args.merge_partials), args.output)
        return

    learners = [name.strip() for name in args.learners.split(",") if name.strip()]
    for name in learners:
        if name not in _LEARNER_FACTORIES:
            raise SystemExit(f"unknown learner {name!r}")

    if args.seed_list is not None:
        raw_seeds = [int(part) for part in args.seed_list.split(",") if part.strip()]
    else:
        raw_seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    seeds = require_unique_jax_seeds(raw_seeds, name="seeds")

    logger.info("loading MNIST from data_home=%s", data_home)
    data_x, data_y = load_mnist_train(data_home)
    logger.info("train split: x=%s y=%s", data_x.shape, data_y.shape)
    results = {}
    for name in learners:
        logger.info("running %s for %d seeds x %d steps", name, len(seeds), config.n_steps)
        results[name] = run_ipmnist(
            data_x,
            data_y,
            name,
            seeds,
            config=config,
            progress_every=args.progress_every,
        )
        logger.info(
            "%s done: average online accuracy %.4f (wall %.1fs)",
            name,
            float(results[name].average_online_accuracy.mean()),
            results[name].wall_clock_seconds,
        )

    if args.partial_out is not None:
        if len(learners) != 1:
            raise SystemExit("--partial-out expects exactly one learner per shard")
        if len(seeds) != 1:
            raise SystemExit("--partial-out expects exactly one seed per v2 shard")
        atomic_write_new_json(args.partial_out, partial_payload(results[learners[0]]))
        logger.info("wrote shard %s", args.partial_out)
        return

    raise AssertionError("legacy v2 compatibility must produce a shard or merge existing shards")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the retained development-v2 shard and merge interface."""

    main_v2_compat(argv)


if __name__ == "__main__":
    main()
