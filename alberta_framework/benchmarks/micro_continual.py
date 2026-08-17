"""First-principles micro-continual benchmark suite (numbers instead of MNIST).

Seconds-scale synthetic non-stationary online classification streams that
distill each difficulty axis the IPMNIST campaign measured
(``docs/research/ipmnist-theory.md``) into its minimal form. The base
distribution is ``n_classes`` Gaussian mixture classes in ``dim`` dimensions,
with three MNIST-mimicking structural ingredients that calibration proved
load-bearing for proxy validity (see ``outputs/micro_continual/SUITE.md``):

- a *heterogeneous per-dimension scale spectrum* plus per-dimension offsets
  (MNIST pixel marginals vary wildly; this gives the conditioning axis its
  teeth);
- *within-class multimodality* (``n_components`` Gaussian components per
  class — MNIST classes are mixtures of writing styles; this is what makes
  features expensive to learn and worth protecting);
- *sparse, localized component structure* (each component displaces only
  ``component_sparsity`` dimensions and each class signals on a
  ``class_sparsity`` fraction — the stroke-locality mimic; dense random
  geometry makes features too diffuse for protection to pay).

Every ``regime_length`` steps the stream applies one regime transform:

- **M1** ``input_permutation`` — fresh coordinate permutation per regime
  (the input-shift axis; micro analogue of Input-permuted MNIST).
- **M2** ``label_permutation`` — inputs untouched, bijective label remap per
  regime (the label-shift axis; micro analogue of label-permuted EMNIST).
- **M3** ``scale_shift`` — global multiplicative rescaling ``x -> c_r * x``
  per regime (the conditioning axis in isolation).
- **M4** ``recurrence`` — coordinate permutations drawn from a small pool
  that revisits (the memory axis).

All transforms are bijections of the base distribution, so the **Bayes
accuracy is known by construction and regime-invariant**: the true model is
class-conditional diagonal Gaussians, its Bayes rule is closed-form
(whitened nearest mean), and :func:`bayes_reference` evaluates it to
arbitrary Monte-Carlo precision (with an exact closed form for two classes,
:func:`two_class_bayes_accuracy`). Streams are pure functions of
``(config, seed)`` and fully scan-able; a full method run takes seconds.

The method ladder (:data:`LADDER_ARMS`) reuses the campaign's registered
update equations (``ipmnist_screening`` factories) so the micro suite is
validated as a *proxy*: :func:`transfer_validation` checks that the ladder
reproduces the full-protocol ordering (conditioning dominates, gate
small-positive, Adam decays, streaming naive Bayes between raw UPGD-W and
conditioned SGD, champion on top). The suite is the fitness function for the
update-rule discovery lane; coordination doc:
``outputs/micro_continual/SUITE.md``.

Everything here is a development diagnostic — never promotable scientific
evidence.

----

A **provisional digits-based suite** (``MICRO_SUITE`` /
:func:`build_micro_stream`, authored by the rule-discovery track before this
canonical suite landed) is retained at the bottom of the module so the
discovery harness keeps importing; see ``outputs/micro_continual/SUITE.md``
for the reconciliation plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import operator
import os
import platform
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._fixed_count_selection import stable_smallest_mask
from alberta_framework._seed_validation import require_jax_seed, require_unique_jax_seeds
from alberta_framework._strict_json import load_strict_json_object
from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningStepFn,
    _finite_wall_clock_total,
    _make_naive_bayes_learner,
    _make_sgd_ema_norm_learner,
    _make_upgd_shiftnorm_learner,
    _validated_wall_clock_seconds,
    _wrap_grad_learner,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LearnerInitFn,
    _make_adamw_learner,
    _make_upgd_w_learner,
    atomic_write_new,
    init_mlp_params,
)
from alberta_framework.core._float32_scalars import validated_float32_scalar


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


logger = logging.getLogger(__name__)

MICRO_GAUSS_SUITE_VERSION = "gauss-v1"

MICRO_SHARD_SCHEMA = "alberta.micro_continual.shard.v1"
MICRO_SUMMARY_SCHEMA = "alberta.micro_continual.summary.v1"
MICRO_VALIDATION_SCHEMA = "alberta.micro_continual.transfer_validation.v1"

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}

#: Stream families, in M1..M4 order.
FAMILIES: tuple[str, ...] = (
    "input_permutation",
    "label_permutation",
    "scale_shift",
    "recurrence",
)

#: The campaign method ladder run for proxy validation, worst-to-best on the
#: full protocol (raw arms, then conditioned arms; naive Bayes is the
#: gradient-free row).
LADDER_ARMS: tuple[str, ...] = (
    "sgd_raw",
    "adamw",
    "upgd_raw",
    "sgd_norm",
    "gated_norm",
    "naive_bayes",
)

# RNG domain separators (fold_in constants) for the per-seed key chains.
_STREAM_DOMAIN = 101
_INIT_DOMAIN = 202
_STEP_DOMAIN = 303
_BAYES_DOMAIN = 404


def _is_registered_subclass(cls: type, abc: type) -> bool:
    """Run one ABC subclass check, normalizing metaclass hook failures to False."""
    try:
        return issubclass(cls, abc)
    except Exception:
        # ``issubclass`` against an ABC hashes ``cls`` for its caches, so a
        # metaclass whose ``__hash__`` raises would otherwise leak the hook's
        # exception through the documented ValueError boundary.
        return False


def _require_finite_real(value: object, name: str) -> float:
    """Return one canonical float after rejecting non-real and non-finite values.

    The message never formats ``value``: repr hooks on untrusted objects must
    not run, and ordinary conversion failures normalize to ``ValueError``
    while ``BaseException`` (e.g. ``KeyboardInterrupt``) still propagates.
    """
    message = f"{name} must be a finite real number"
    if type(value) is bool or not _is_registered_subclass(type(value), Real):
        raise ValueError(message)
    try:
        number = float(cast(Any, value))
    except Exception as exc:
        raise ValueError(message) from exc
    if not math.isfinite(number):
        raise ValueError(message)
    return number


def _freeze_micro_hyperparameters(value: object, *, context: str) -> Mapping[str, float]:
    """Copy one primitive-only hyperparameter mapping behind an immutable view."""
    if not _is_registered_subclass(type(value), Mapping):
        raise ValueError(f"{context} must be an object with non-empty string keys")
    try:
        items = list(cast(Mapping[object, object], value).items())
    except Exception as exc:
        raise ValueError(
            f"{context} must be an object with non-empty string keys"
        ) from exc
    frozen: dict[str, float] = {}
    for key, raw_value in items:
        if type(key) is not str or not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        host_key = _require_exact_str("key", key)
        frozen[host_key] = _require_finite_real(raw_value, f"{context}['{host_key}']")
    return MappingProxyType(frozen)


def _require_nonempty_string(value: object, *, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_builtin_finite_real(value: object, *, context: str) -> float:
    """Canonicalize a result scalar without invoking numeric subclass hooks."""
    if type(value) not in (int, float):
        raise ValueError(f"{context} must be a finite built-in real number")
    number = float(cast("int | float", value))
    if not math.isfinite(number):
        raise ValueError(f"{context} must be a finite built-in real number")
    return number


# =============================================================================
# Configuration
# =============================================================================

_INT32_MAX: int = 2**31 - 1
_ACTUAL_INT_TYPES: tuple[type, ...] = (
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
)


def _require_positive_int32(value: object, *, name: str, minimum: int = 1) -> int:
    """Canonicalize one trusted positive integer in JAX's shape/index domain."""
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    number = operator.index(cast(SupportsIndex, value))
    if not minimum <= number <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return number


def _require_float32_resource(name: str, scalars: int) -> None:
    """Bound one aggregate of four-byte JAX leaves before any allocation."""
    if scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _materialized_stream_scalars(
    *, n_regimes: int, regime_length: int, dim: int, n_classes: int, n_components: int
) -> int:
    """Return the exact scalar count of all retained GaussianMicroStream leaves."""
    n_steps = n_regimes * regime_length
    component_dim = n_classes * n_components * dim
    regime_dim = n_regimes * dim
    regime_class = n_regimes * n_classes
    return (
        2 * n_steps * dim
        + 3 * n_steps
        + component_dim
        + dim
        + regime_dim
        + regime_class
        + 2 * n_regimes
    )


@dataclass(frozen=True)
class MicroStreamConfig:
    """One micro stream: family + geometry + regime schedule.

    The defaults are the **transfer-validated M1 operating point** frozen by
    the calibration campaign (``outputs/micro_continual/SUITE.md``): they
    reproduce the full-protocol method ordering on the ladder.

    Args:
        family: One of :data:`FAMILIES` (M1..M4).
        n_regimes: Number of regime blocks (the micro analogue of tasks).
        regime_length: Online steps per regime.
        dim: Input dimensionality.
        n_classes: Number of Gaussian mixture classes.
        n_components: Gaussian components per class (within-class
            multimodality; ``1`` restores unimodal clusters and the exact
            two-class closed form).
        spectrum_decades: Per-dimension scale spectrum spans
            ``10**0 .. 10**-spectrum_decades`` (log-spaced); ``0`` is a
            homogeneous (well-conditioned) stream.
        mean_separation: Class-mean spread per signalling dimension, in units
            of that dimension's scale.
        component_scale: Component displacement magnitude per active
            dimension, in units of that dimension's scale.
        component_sparsity: Number of dimensions each component displaces
            (localized features; capped at ``dim``).
        class_sparsity: Fraction of dimensions carrying class-mean signal.
        noise_scale: Within-class noise per dimension, in units of that
            dimension's scale.
        offset_scale: Per-dimension mean offset, in units of that dimension's
            scale (MNIST pixels have heterogeneous nonzero means).
        scale_shift_min: M3 only — lower bound of the log-uniform per-regime
            global scale factor.
        scale_shift_max: M3 only — upper bound of that factor.
        recurrence_pool: M4 only — size of the recurring permutation pool
            (the first ``recurrence_pool`` regimes introduce each element
            once; later regimes revisit uniformly at random).
    """

    family: str = "input_permutation"
    n_regimes: int = 100
    regime_length: int = 5000
    dim: int = 256
    n_classes: int = 10
    n_components: int = 6
    spectrum_decades: float = 2.0
    mean_separation: float = 0.4
    component_scale: float = 1.2
    component_sparsity: int = 10
    class_sparsity: float = 0.2
    noise_scale: float = 1.0
    offset_scale: float = 1.0
    scale_shift_min: float = 0.25
    scale_shift_max: float = 4.0
    recurrence_pool: int = 5

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        for name in ("n_regimes", "regime_length", "dim", "n_classes", "n_components"):
            object.__setattr__(
                self,
                name,
                _require_positive_int32(getattr(self, name), name=name),
            )
        sparsity = _require_positive_int32(
            self.component_sparsity, name="component_sparsity"
        )
        pool = _require_positive_int32(
            self.recurrence_pool, name="recurrence_pool", minimum=2
        )
        object.__setattr__(self, "component_sparsity", sparsity)
        object.__setattr__(self, "recurrence_pool", pool)
        if sparsity > self.dim:
            raise ValueError(
                f"component_sparsity ({sparsity}) must not exceed dim ({self.dim})"
            )
        float_domains: dict[str, dict[str, object]] = {
            "spectrum_decades": {"lower": 0.0},
            "mean_separation": {"positive": True},
            "component_scale": {"lower": 0.0},
            "class_sparsity": {"positive": True, "upper": 1.0},
            "noise_scale": {"positive": True},
            "offset_scale": {"lower": 0.0},
            "scale_shift_min": {"positive": True},
            "scale_shift_max": {"positive": True},
        }
        for name, domain in float_domains.items():
            canonical = validated_float32_scalar(
                name,
                getattr(self, name),
                **cast(dict[str, Any], domain),
            )
            object.__setattr__(self, name, canonical)

        scale_min_sink = float(np.float32(self.scale_shift_min))
        scale_max_sink = float(np.float32(self.scale_shift_max))
        if not scale_min_sink < scale_max_sink:
            raise ValueError(
                "scale_shift bounds must satisfy 0 < scale_shift_min < "
                "scale_shift_max after float32 narrowing"
            )
        if self.family == "recurrence":
            if pool > self.n_regimes:
                raise ValueError(
                    f"recurrence_pool ({pool}) must not exceed n_regimes "
                    f"({self.n_regimes})"
                )
        _require_float32_resource(
            "GaussianMicroStream outputs", self.materialized_stream_scalars
        )

    @property
    def n_steps(self) -> int:
        """Total online steps in one run."""
        return self.n_regimes * self.regime_length

    @property
    def materialized_stream_scalars(self) -> int:
        """Exact retained scalar count of the materialized stream.

        Every named generation-work array is bounded by one of these returned
        shapes. Compiler-internal and transient allocator buffers are not part
        of this persistent-output contract.
        """
        return _materialized_stream_scalars(
            n_regimes=self.n_regimes,
            regime_length=self.regime_length,
            dim=self.dim,
            n_classes=self.n_classes,
            n_components=self.n_components,
        )

    @property
    def materialized_stream_bytes(self) -> int:
        """Exact retained bytes; every returned leaf is float32 or int32."""
        return 4 * self.materialized_stream_scalars

    def to_config(self) -> dict[str, Any]:
        """JSON-serializable configuration (roundtrips through the constructor)."""
        return {
            "family": self.family,
            "n_regimes": self.n_regimes,
            "regime_length": self.regime_length,
            "dim": self.dim,
            "n_classes": self.n_classes,
            "n_components": self.n_components,
            "spectrum_decades": self.spectrum_decades,
            "mean_separation": self.mean_separation,
            "component_scale": self.component_scale,
            "component_sparsity": self.component_sparsity,
            "class_sparsity": self.class_sparsity,
            "noise_scale": self.noise_scale,
            "offset_scale": self.offset_scale,
            "scale_shift_min": self.scale_shift_min,
            "scale_shift_max": self.scale_shift_max,
            "recurrence_pool": self.recurrence_pool,
        }

    @classmethod
    def from_mapping(
        cls, mapping: object, *, source: str = "stream_config"
    ) -> MicroStreamConfig:
        """Reconstruct a stream config from a serialized object.

        The mapping must contain exactly the :meth:`to_config` key set.
        Omitted fields must not be filled from dataclass defaults: a truncated
        shard would otherwise reconstruct a different stream identity.
        """
        if not _is_registered_subclass(type(mapping), Mapping):
            raise ValueError(f"{source} must be an object")
        try:
            items = list(cast(Mapping[object, object], mapping).items())
        except Exception as exc:
            raise ValueError(f"{source} must be a readable object") from exc
        try:
            if any(type(key) is not str for key, _ in items):
                raise ValueError(f"{source} keys must be exact strings")
            payload = dict(cast(list[tuple[str, Any]], items))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{source} must be a readable object") from exc
        expected = {field.name for field in fields(cls)}
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"unexpected {extra}")
            raise ValueError(
                f"{source} must contain exactly the serialized key set; "
                + "; ".join(details)
            )
        return cls(**payload)


# =============================================================================
# Stream generation
# =============================================================================


def dim_scale_spectrum(config: MicroStreamConfig) -> Array:
    """Per-dimension scales, log-spaced ``10**0 .. 10**-spectrum_decades``.

    The heterogeneous-marginals mimic: MNIST pixel statistics span from
    near-constant corner pixels to high-variance center pixels, which is what
    makes raw-input optimization ill-conditioned and input-statistics
    tracking valuable. ``spectrum_decades=0`` gives a homogeneous stream.
    """
    exponents = jnp.linspace(0.0, -config.spectrum_decades, config.dim)
    return jnp.power(10.0, exponents).astype(jnp.float32)


def _stream_keys(
    config: MicroStreamConfig, seed: int
) -> tuple[Array, Array, Array, Array, Array]:
    if type(config) is not MicroStreamConfig:
        raise TypeError("config must be a MicroStreamConfig")
    seed = require_jax_seed(seed, name="seed")
    root = jr.fold_in(jr.key(seed), _STREAM_DOMAIN)
    key_geometry, key_labels, key_components, key_noise, key_regime = jr.split(root, 5)
    return key_geometry, key_labels, key_components, key_noise, key_regime


def class_geometry(config: MicroStreamConfig, seed: int) -> tuple[Array, Array]:
    """True generative parameters for one seed: ``(component_means, dim_sigma)``.

    ``component_means`` is ``f32[n_classes, n_components, dim]`` — per-class
    Gaussian mixture components; ``dim_sigma`` is ``f32[dim]`` (shared
    diagonal noise). Class centers signal on a ``class_sparsity`` fraction of
    dimensions; each component displaces exactly ``component_sparsity``
    dimensions (localized features). Everything carries the per-dimension
    scale spectrum, so per-dimension SNR structure is scale-free while raw
    scales span ``spectrum_decades`` decades.
    """
    key_geometry, _, _, _, _ = _stream_keys(config, seed)
    (
        key_offset,
        key_means,
        key_displacement,
        key_class_mask,
        key_component_mask,
    ) = jr.split(key_geometry, 5)
    scales = dim_scale_spectrum(config)
    c, k, d = config.n_classes, config.n_components, config.dim
    offsets = config.offset_scale * scales * jr.normal(key_offset, (d,), jnp.float32)
    class_mask = (
        jr.uniform(key_class_mask, (c, d)) < config.class_sparsity
    ).astype(jnp.float32)
    class_means = offsets + config.mean_separation * scales * jr.normal(
        key_means, (c, d), jnp.float32
    ) * class_mask
    if config.component_sparsity >= d:
        component_mask = jnp.ones((c, k, d), dtype=jnp.float32)
    else:
        scores = jr.uniform(key_component_mask, (c, k, d))
        component_mask = stable_smallest_mask(scores, config.component_sparsity).astype(
            jnp.float32
        )
    displacements = config.component_scale * scales * jr.normal(
        key_displacement, (c, k, d), jnp.float32
    ) * component_mask
    component_means = class_means[:, None, :] + displacements
    dim_sigma = (config.noise_scale * scales).astype(jnp.float32)
    return component_means.astype(jnp.float32), dim_sigma


def _require_finite_geometry(component_means: Array, dim_sigma: Array) -> None:
    if not bool(
        jnp.all(jnp.isfinite(component_means))
        & jnp.all(jnp.isfinite(dim_sigma))
    ):
        raise ValueError("micro stream geometry must remain finite in float32")


@dataclass(frozen=True)
class GaussianMicroStream:
    """A materialized micro stream plus its generating ground truth.

    ``x``/``y`` are what the learner sees; ``base_x``/``base_y`` are the
    pre-transform draws; the regime arrays record the exact transform of each
    regime so tests (and identification-style methods) can invert them.
    """

    x: Array
    y: Array
    base_x: Array
    base_y: Array
    regime_ids: Array
    component_means: Array
    dim_sigma: Array
    permutations: Array
    label_maps: Array
    scale_factors: Array
    regime_pool_ids: Array


def assemble_observed(
    base_x: Array,
    base_y: Array,
    regime_ids: Array,
    permutations: Array,
    label_maps: Array,
    scale_factors: Array,
) -> tuple[Array, Array]:
    """Apply the per-regime transforms to the base stream.

    ``x[t] = scale[r_t] * base_x[t][permutations[r_t]]`` (the protocol's
    gather convention) and ``y[t] = label_maps[r_t][base_y[t]]``.
    """
    step_perms = permutations[regime_ids]
    x = jnp.take_along_axis(base_x, step_perms, axis=1)
    x = x * scale_factors[regime_ids][:, None]
    y = label_maps[regime_ids, base_y]
    return x.astype(jnp.float32), y.astype(jnp.int32)


def _identity_permutations(config: MicroStreamConfig) -> Array:
    return jnp.tile(
        jnp.arange(config.dim, dtype=jnp.int32), (config.n_regimes, 1)
    )


def _identity_label_maps(config: MicroStreamConfig) -> Array:
    return jnp.tile(
        jnp.arange(config.n_classes, dtype=jnp.int32), (config.n_regimes, 1)
    )


def generate_stream(config: MicroStreamConfig, seed: int) -> GaussianMicroStream:
    """Materialize the deterministic stream for one ``(config, seed)``."""
    _, key_labels, key_components, key_noise, key_regime = _stream_keys(config, seed)
    key_perm, key_label_map, key_scale, key_pool, key_pool_schedule = jr.split(
        key_regime, 5
    )
    component_means, dim_sigma = class_geometry(config, seed)
    _require_finite_geometry(component_means, dim_sigma)

    n_steps = config.n_steps
    base_y = jr.randint(key_labels, (n_steps,), 0, config.n_classes).astype(jnp.int32)
    base_z = jr.randint(key_components, (n_steps,), 0, config.n_components)
    noise = jr.normal(key_noise, (n_steps, config.dim), jnp.float32)
    base_x = (
        component_means[base_y, base_z] + dim_sigma[None, :] * noise
    ).astype(jnp.float32)
    regime_ids = (
        jnp.arange(n_steps, dtype=jnp.int32) // config.regime_length
    ).astype(jnp.int32)

    permutations = _identity_permutations(config)
    label_maps = _identity_label_maps(config)
    scale_factors = jnp.ones(config.n_regimes, dtype=jnp.float32)
    regime_pool_ids = jnp.arange(config.n_regimes, dtype=jnp.int32)

    regimes = jnp.arange(config.n_regimes)
    if config.family == "input_permutation":
        permutations = jax.vmap(
            lambda r: jr.permutation(jr.fold_in(key_perm, r), config.dim)
        )(regimes).astype(jnp.int32)
    elif config.family == "label_permutation":
        label_maps = jax.vmap(
            lambda r: jr.permutation(jr.fold_in(key_label_map, r), config.n_classes)
        )(regimes).astype(jnp.int32)
    elif config.family == "scale_shift":
        scale_factors = jnp.exp(
            jr.uniform(
                key_scale,
                (config.n_regimes,),
                jnp.float32,
                math.log(config.scale_shift_min),
                math.log(config.scale_shift_max),
            )
        ).astype(jnp.float32)
    elif config.family == "recurrence":
        pool = config.recurrence_pool
        pool_permutations = jax.vmap(
            lambda p: jr.permutation(jr.fold_in(key_pool, p), config.dim)
        )(jnp.arange(pool)).astype(jnp.int32)
        introduction = jnp.arange(pool, dtype=jnp.int32)
        n_tail = config.n_regimes - pool
        tail = jr.randint(key_pool_schedule, (n_tail,), 0, pool).astype(jnp.int32)
        regime_pool_ids = jnp.concatenate([introduction, tail])
        permutations = pool_permutations[regime_pool_ids]

    x, y = assemble_observed(
        base_x, base_y, regime_ids, permutations, label_maps, scale_factors
    )
    return GaussianMicroStream(
        x=x,
        y=y,
        base_x=base_x,
        base_y=base_y,
        regime_ids=regime_ids,
        component_means=component_means,
        dim_sigma=dim_sigma,
        permutations=permutations,
        label_maps=label_maps,
        scale_factors=scale_factors,
        regime_pool_ids=regime_pool_ids,
    )


# =============================================================================
# Analytic Bayes reference
# =============================================================================


@dataclass(frozen=True)
class BayesReference:
    """Bayes-optimal accuracy of one ``(config, seed)`` stream geometry.

    Regime-invariant by construction: every family transform is a bijection
    acting covariantly on the generative parameters, so the induced Bayes
    rule makes identical predictions in every regime (pinned by test).
    """

    bayes_accuracy: float
    mc_sem: float
    n_samples: int
    chance: float
    seed: int


def bayes_predict(component_means: Array, dim_sigma: Array, x: Array) -> Array:
    """Bayes-optimal predictions for ``x`` (``f32[n, dim]``) under the true model.

    Equal class priors, equal component weights, shared diagonal covariance:
    the Bayes rule is ``argmax_c logsumexp_k(-0.5 * mahalanobis²(x, mu_ck))``.
    Components are visited sequentially so memory stays
    ``O(n * n_classes * n_components + n * dim)`` instead of materializing
    the ``(n, C, K, dim)`` difference tensor.  Computing each coordinate
    difference before squaring also avoids the catastrophic cancellation in
    the expanded ``||x||² - 2 x·mu + ||mu||²`` identity.
    """
    c, k, d = component_means.shape
    whitened_means = (component_means / dim_sigma[None, None, :]).reshape(c * k, d)
    whitened_x = x / dim_sigma[None, :]

    def squared_distances(mean: Array) -> Array:
        delta = whitened_x - mean[None, :]
        return jnp.sum(delta * delta, axis=1)

    # lax.map evaluates one component at a time.  Unlike vmap/broadcasting,
    # this makes the bounded-memory contract explicit while retaining a
    # vectorized reduction across samples and dimensions for each component.
    d2 = jax.lax.map(squared_distances, whitened_means).T.reshape(-1, c, k)
    scores = jax.scipy.special.logsumexp(-0.5 * d2, axis=2)
    return jnp.argmax(scores, axis=1).astype(jnp.int32)


def two_class_bayes_accuracy(mu0: Array, mu1: Array, dim_sigma: Array) -> float:
    """Exact Bayes accuracy for two equal-prior diagonal-Gaussian classes.

    ``Phi(delta / 2)`` with ``delta`` the Mahalanobis distance between the
    class means — the closed-form cross-check for the Monte-Carlo reference
    (applies to ``n_components=1`` geometries).
    """
    whitened = (jnp.asarray(mu0) - jnp.asarray(mu1)) / jnp.asarray(dim_sigma)
    delta = float(jnp.sqrt(jnp.sum(whitened * whitened)))
    return 0.5 * (1.0 + math.erf(delta / (2.0 * math.sqrt(2.0))))


def bayes_reference(
    config: MicroStreamConfig, seed: int, n_samples: int = 200_000
) -> BayesReference:
    """Monte-Carlo Bayes accuracy of the seed's geometry (chunked, exact rule).

    Fresh draws from the base distribution (independent of the stream), the
    closed-form Bayes rule, and a binomial standard error. Applies to every
    regime of all four families (transform invariance).
    """
    n_samples = _require_positive_int32(n_samples, name="n_samples")
    seed = require_jax_seed(seed, name="seed")
    chunk_size = min(20_000, n_samples)
    class_components = config.n_classes * config.n_components
    # Exact named host-visible arrays retained at the peak of bayes_predict:
    # geometry + whitened geometry, y/z/eps/x, whitened x, d2, scores, and
    # predictions. The one-component delta and expression/compiler temporaries
    # are intentionally outside this explicitly named work contract.
    bayes_work_scalars = (
        2 * class_components * config.dim
        + config.dim
        + 3 * chunk_size * config.dim
        + chunk_size * class_components
        + chunk_size * config.n_classes
        + class_components
        + 4 * chunk_size
    )
    _require_float32_resource(
        "Bayes-reference named array work",
        bayes_work_scalars,
    )
    component_means, dim_sigma = class_geometry(config, seed)
    _require_finite_geometry(component_means, dim_sigma)
    key = jr.fold_in(jr.key(seed), _BAYES_DOMAIN)
    n_correct = 0
    drawn = 0
    chunk_index = 0
    while drawn < n_samples:
        chunk = min(chunk_size, n_samples - drawn)
        chunk_key = jr.fold_in(key, chunk_index)
        key_y, key_z, key_eps = jr.split(chunk_key, 3)
        y = jr.randint(key_y, (chunk,), 0, config.n_classes)
        z = jr.randint(key_z, (chunk,), 0, config.n_components)
        eps = jr.normal(key_eps, (chunk, config.dim), jnp.float32)
        x = component_means[y, z] + dim_sigma[None, :] * eps
        predictions = bayes_predict(component_means, dim_sigma, x)
        n_correct += int(jnp.sum(predictions == y))
        drawn += chunk
        chunk_index += 1
    accuracy = n_correct / n_samples
    return BayesReference(
        bayes_accuracy=accuracy,
        mc_sem=math.sqrt(accuracy * (1.0 - accuracy) / n_samples),
        n_samples=n_samples,
        chance=1.0 / config.n_classes,
        seed=seed,
    )


# =============================================================================
# Method-ladder arms (campaign equations, reused)
# =============================================================================


MicroArmFactory = Callable[[Mapping[str, float]], tuple[LearnerInitFn, ScreeningStepFn]]


@dataclass(frozen=True)
class MicroArmSpec:
    """One ladder arm: a named learner configuration (campaign equations)."""

    name: str
    mechanism: str
    hyperparameters: Mapping[str, float]
    factory: MicroArmFactory
    description: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, context="MicroArmSpec.name")
        _require_nonempty_string(self.mechanism, context="MicroArmSpec.mechanism")
        object.__setattr__(
            self,
            "hyperparameters",
            _freeze_micro_hyperparameters(
                self.hyperparameters, context="MicroArmSpec.hyperparameters"
            ),
        )


def _make_sgd_raw_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    """Plain SGD (optionally with decoupled decay) on raw inputs."""
    step_size = hp["step_size"]
    decay = 1.0 - step_size * hp["weight_decay"]

    def init_fn(params: dict[str, Array]) -> Array:
        del params
        return jnp.zeros((), dtype=jnp.int32)

    def step_fn(
        params: dict[str, Array], state: Array, grads: dict[str, Array], key: Array
    ) -> tuple[dict[str, Array], Array]:
        del key  # deterministic
        new_params = {
            name: params[name] * decay - step_size * grads[name] for name in params
        }
        return new_params, state

    return _wrap_grad_learner(init_fn, step_fn)


def _upgd_raw_factory(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _wrap_grad_learner(*_make_upgd_w_learner(dict(hp)))


def _adamw_factory(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, ScreeningStepFn]:
    return _wrap_grad_learner(*_make_adamw_learner(dict(hp)))


def _build_arm_registry() -> dict[str, MicroArmSpec]:
    specs = [
        MicroArmSpec(
            name="sgd_raw",
            mechanism="baseline",
            hyperparameters={"step_size": 0.01, "weight_decay": 0.0},
            factory=_make_sgd_raw_learner,
            description="Plain SGD on raw inputs (mechanism-free floor).",
        ),
        MicroArmSpec(
            name="adamw",
            mechanism="adaptive_rate",
            hyperparameters=dict(ADAMW_PROTOCOL_HYPERPARAMETERS),
            factory=_adamw_factory,
            description="Published protocol AdamW (fast early, decays late).",
        ),
        MicroArmSpec(
            name="upgd_raw",
            mechanism="utility_gated_perturbation",
            hyperparameters=dict(UPGD_W_PROTOCOL_HYPERPARAMETERS),
            factory=_upgd_raw_factory,
            description="Published UPGD-W on raw inputs (the ICLR-2024 SOTA form).",
        ),
        MicroArmSpec(
            name="sgd_norm",
            mechanism="input_conditioning",
            hyperparameters={
                "step_size": 0.01,
                "weight_decay": 0.01,
                "norm_decay": 0.99,
                "norm_epsilon": 1e-8,
            },
            factory=_make_sgd_ema_norm_learner,
            description=(
                "EMA input normalization (decay 0.99) + SGD + decoupled decay "
                "(the campaign's sgd_ema_norm_d099 mechanism-free conditioned floor)."
            ),
        ),
        MicroArmSpec(
            name="gated_norm",
            mechanism="conditioned_utility_gate",
            hyperparameters={
                "step_size": 0.01,
                "utility_decay": 0.9999,
                "weight_decay": 0.01,
                "norm_decay": 0.99,
                "norm_epsilon": 1e-8,
                "fast_decay": 0.9,
                "shift_k": 1.0,
                "shift_delta": 0.02,
                "shift_refractory": 0.0,
            },
            factory=_make_upgd_shiftnorm_learner,
            description=(
                "The campaign champion form (sigma0_shiftnorm_d099): "
                "shift-triggered per-feature EMA normalization (decay 0.99) -> "
                "utility-gated SGD -> decoupled decay, no perturbation."
            ),
        ),
        MicroArmSpec(
            name="naive_bayes",
            mechanism="streaming_generative_classifier",
            hyperparameters={"nb_decay": 0.98, "nb_var_epsilon": 1e-4},
            factory=_make_naive_bayes_learner,
            description=(
                "Streaming class-conditional diagonal Gaussians (no gradients, "
                "no MLP); nb_decay matches the campaign arm, the variance floor "
                "is rescaled to the micro spectrum (smallest dimension variance "
                "1e-4 at the 2-decade default)."
            ),
        ),
    ]
    return {spec.name: spec for spec in specs}


MICRO_ARM_REGISTRY: Mapping[str, MicroArmSpec] = MappingProxyType(_build_arm_registry())


def micro_arm_spec(name: object) -> MicroArmSpec:
    """Look up one ladder arm; raises ``KeyError`` for unknown names."""
    host_name = _require_exact_str("name", name)
    spec = MICRO_ARM_REGISTRY.get(host_name)
    if spec is None:
        raise KeyError(f"unknown micro arm '{host_name}'; known: {sorted(MICRO_ARM_REGISTRY)}")
    return spec


# =============================================================================
# Runner
# =============================================================================


@dataclass(frozen=True)
class MicroRunResult:
    """Host-side per-regime results of one ``(family, arm, seed)`` run."""

    family: str
    arm_name: str
    mechanism: str
    hyperparameters: Mapping[str, float]
    seed: int
    hidden1: int
    hidden2: int
    stream_config: MicroStreamConfig
    per_regime_accuracy: np.ndarray
    per_regime_loss: np.ndarray
    per_regime_plasticity: np.ndarray
    overall_accuracy: float
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _require_nonempty_string(self.family, context="MicroRunResult.family"),
        )
        _require_nonempty_string(self.arm_name, context="MicroRunResult.arm_name")
        _require_nonempty_string(self.mechanism, context="MicroRunResult.mechanism")
        object.__setattr__(
            self,
            "hyperparameters",
            _freeze_micro_hyperparameters(
                self.hyperparameters, context="MicroRunResult.hyperparameters"
            ),
        )
        object.__setattr__(
            self, "seed", require_jax_seed(self.seed, name="MicroRunResult.seed")
        )
        object.__setattr__(
            self,
            "hidden1",
            _require_positive_int32(self.hidden1, name="MicroRunResult.hidden1"),
        )
        object.__setattr__(
            self,
            "hidden2",
            _require_positive_int32(self.hidden2, name="MicroRunResult.hidden2"),
        )
        object.__setattr__(
            self,
            "overall_accuracy",
            _require_builtin_finite_real(
                self.overall_accuracy,
                context="MicroRunResult.overall_accuracy",
            ),
        )
        object.__setattr__(
            self,
            "wall_clock_seconds",
            _require_builtin_finite_real(
                self.wall_clock_seconds,
                context="MicroRunResult.wall_clock_seconds",
            ),
        )


def run_micro_arm(
    config: MicroStreamConfig,
    arm: str | MicroArmSpec,
    seed: int,
    hidden1: int = 75,
    hidden2: int = 38,
) -> MicroRunResult:
    """Run one ladder arm on one stream seed (paired: the stream and the MLP
    init depend only on the seed, not on the arm).

    The metric is the protocol's online-accuracy-while-learning: each
    prediction is scored before the update that consumes the example, and the
    per-regime value is the mean over the regime's steps.
    """
    spec = arm if isinstance(arm, MicroArmSpec) else micro_arm_spec(arm)
    seed = require_jax_seed(seed, name="seed")
    stream = generate_stream(config, seed)
    net = IPMNISTConfig(
        n_tasks=config.n_regimes,
        task_length=config.regime_length,
        input_dim=config.dim,
        hidden1=hidden1,
        hidden2=hidden2,
        n_classes=config.n_classes,
    )
    key_init = jr.fold_in(jr.key(jnp.uint32(seed)), _INIT_DOMAIN)
    key_steps = jr.fold_in(jr.key(jnp.uint32(seed)), _STEP_DOMAIN)
    params = init_mlp_params(key_init, net)
    init_fn, step_fn = spec.factory(dict(spec.hyperparameters))
    state = init_fn(params)

    def run_stream(
        params: dict[str, Array], state: Any, key: Array, xs: Array, ys: Array
    ) -> tuple[Array, Array, Array]:
        def one_step(
            carry: tuple[dict[str, Array], Any, Array], step_xy: tuple[Array, Array]
        ) -> tuple[tuple[dict[str, Array], Any, Array], tuple[Array, Array, Array]]:
            step_params, step_state, key = carry
            x, y = step_xy
            key, step_key = jr.split(key)
            new_params, new_state, metrics = step_fn(
                step_params, step_state, x, y, step_key
            )
            return (new_params, new_state, key), metrics

        _, (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, state, key), (xs, ys)
        )
        return accuracies, losses, plasticities

    run_jit = jax.jit(run_stream)
    started = time.monotonic()
    accuracies, losses, plasticities = run_jit(
        params, state, key_steps, stream.x, stream.y
    )
    shape = (config.n_regimes, config.regime_length)
    per_regime_accuracy = np.asarray(accuracies, dtype=np.float64).reshape(shape).mean(axis=1)
    per_regime_loss = np.asarray(losses, dtype=np.float64).reshape(shape).mean(axis=1)
    per_regime_plasticity = (
        np.asarray(plasticities, dtype=np.float64).reshape(shape).mean(axis=1)
    )
    wall_clock = time.monotonic() - started
    return MicroRunResult(
        family=config.family,
        arm_name=spec.name,
        mechanism=spec.mechanism,
        hyperparameters=MappingProxyType(dict(spec.hyperparameters)),
        seed=int(seed),
        hidden1=hidden1,
        hidden2=hidden2,
        stream_config=config,
        per_regime_accuracy=per_regime_accuracy,
        per_regime_loss=per_regime_loss,
        per_regime_plasticity=per_regime_plasticity,
        overall_accuracy=float(per_regime_accuracy.mean()),
        wall_clock_seconds=wall_clock,
    )


# =============================================================================
# Shards / merge
# =============================================================================


def micro_shard_path(out_dir: Path | str, family: str, arm_name: str, seed: int) -> Path:
    """Canonical shard location: ``{out}/{family}_{arm}_seed{seed}.json``."""
    return Path(out_dir) / f"{family}_{arm_name}_seed{seed}.json"


def micro_shard_payload(result: MicroRunResult) -> dict[str, Any]:
    """Serialize one run to a mergeable shard, recording the spec that actually ran."""
    host_arm = _require_exact_str("arm_name", result.arm_name)
    if host_arm not in MICRO_ARM_REGISTRY:
        raise ValueError(
            f"arm_name '{host_arm}' is not a registered micro arm; "
            "unregistered results cannot be serialized into the registered-arm shard schema"
        )
    return {
        "schema": MICRO_SHARD_SCHEMA,
        "suite_version": MICRO_GAUSS_SUITE_VERSION,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "family": result.family,
        "arm_name": result.arm_name,
        "mechanism": result.mechanism,
        "hyperparameters": dict(result.hyperparameters),
        "seed": result.seed,
        "hidden1": result.hidden1,
        "hidden2": result.hidden2,
        "stream_config": result.stream_config.to_config(),
        "per_regime_accuracy": [round(float(v), 8) for v in result.per_regime_accuracy],
        "per_regime_loss": [round(float(v), 8) for v in result.per_regime_loss],
        "per_regime_plasticity": [
            round(float(v), 8) for v in result.per_regime_plasticity
        ],
        "overall_average_online_accuracy": float(result.overall_accuracy),
        "wall_clock_seconds": round(result.wall_clock_seconds, 3),
        "created_unix": time.time(),
        "environment": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _strict_json_text(payload: dict[str, Any]) -> str:
    """Serialize one object as RFC-valid JSON (no NaN / Infinity tokens)."""
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_micro_shard(path: Path | str, payload: dict[str, Any]) -> None:
    """Atomically publish one immutable shard (refuses an occupied path)."""
    encoded = _strict_json_text(payload).encode("utf-8")
    atomic_write_new(Path(path), encoded)


_MICRO_CURVE_DOMAINS: dict[str, tuple[float, float | None]] = {
    "per_regime_accuracy": (0.0, 1.0),
    "per_regime_loss": (0.0, None),
    "per_regime_plasticity": (0.0, 1.0),
}


def _validated_curve(
    value: object,
    *,
    n_regimes: int,
    lower: float,
    upper: float | None,
    context: str,
) -> list[float]:
    """Return one per-regime curve as exact finite floats inside its metric domain."""
    if not isinstance(value, list) or len(value) != n_regimes:
        raise ValueError(f"{context} must be a list of {n_regimes} finite real numbers")
    curve: list[float] = []
    for index, entry in enumerate(value):
        number = _require_finite_real(entry, f"{context}[{index}]")
        if number < lower or (upper is not None and number > upper):
            domain = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
            raise ValueError(f"{context}[{index}] must lie in {domain}, got '{number}'")
        curve.append(number)
    return curve


def load_micro_shard(path: Path | str) -> dict[str, Any]:
    """Load and structurally validate one micro shard."""
    path = Path(path)
    payload = load_strict_json_object(path)
    if payload.get("schema") != MICRO_SHARD_SCHEMA:
        raise ValueError(f"{path}: schema mismatch (expected {MICRO_SHARD_SCHEMA})")
    if payload.get("suite_version") != MICRO_GAUSS_SUITE_VERSION:
        host_suite = payload.get("suite_version")
        if type(host_suite) is str:
            host_suite_str = _require_exact_str("suite_version", host_suite)
            raise ValueError(
                f"{path}: suite_version mismatch (expected '{MICRO_GAUSS_SUITE_VERSION}', "
                f"got '{host_suite_str}')"
            )
        raise ValueError(
            f"{path}: suite_version mismatch "
            f"(expected '{MICRO_GAUSS_SUITE_VERSION}', got non-string)"
        )
    host_arm_name = payload.get("arm_name")
    if host_arm_name not in MICRO_ARM_REGISTRY:
        if type(host_arm_name) is str:
            host_arm_str = _require_exact_str("arm_name", host_arm_name)
            raise ValueError(f"{path}: unknown arm '{host_arm_str}'")
        raise ValueError(f"{path}: unknown arm is not an exact string")
    arm_spec = MICRO_ARM_REGISTRY[payload["arm_name"]]
    if not isinstance(payload.get("mechanism"), str) or not payload["mechanism"]:
        raise ValueError(f"{path}: mechanism must be a non-empty string")
    if not isinstance(payload.get("hyperparameters"), dict):
        raise ValueError(f"{path}: hyperparameters must be an object")
    payload["hyperparameters"] = dict(
        _freeze_micro_hyperparameters(
            payload["hyperparameters"], context=f"{path}: hyperparameters"
        )
    )
    if payload["mechanism"] != arm_spec.mechanism:
        host_spec_name = _require_exact_str("arm_spec.name", arm_spec.name)
        raise ValueError(
            f"{path}: mechanism does not match registered arm '{host_spec_name}'"
        )
    if payload["hyperparameters"] != dict(arm_spec.hyperparameters):
        host_spec_name2 = _require_exact_str("arm_spec.name", arm_spec.name)
        raise ValueError(
            f"{path}: hyperparameters do not match registered arm '{host_spec_name2}'"
        )
    environment = payload.get("environment")
    required_environment_fields = ("jax", "numpy", "python", "platform")
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(field), str) or not environment[field]
        for field in required_environment_fields
    ):
        raise ValueError(
            f"{path}: environment must record non-empty jax, numpy, python, and platform strings"
        )
    config = MicroStreamConfig.from_mapping(
        payload.get("stream_config"), source=f"{path}: stream_config"
    )
    if payload.get("family") != config.family:
        host_family = payload.get("family")
        host_cfg_family = _require_exact_str("config.family", config.family)
        if type(host_family) is str:
            host_fam_str = _require_exact_str("family", host_family)
            raise ValueError(
                f"{path}: family '{host_fam_str}' does not match "
                f"stream_config family '{host_cfg_family}'"
            )
        raise ValueError(
            f"{path}: family is not an exact string, does not match "
            f"stream_config family '{host_cfg_family}'"
        )
    for fieldname, (lower, upper) in _MICRO_CURVE_DOMAINS.items():
        payload[fieldname] = _validated_curve(
            payload.get(fieldname),
            n_regimes=config.n_regimes,
            lower=lower,
            upper=upper,
            context=f"{path}: {fieldname}",
        )
    payload["wall_clock_seconds"] = _validated_wall_clock_seconds(
        payload.get("wall_clock_seconds"), path
    )
    payload["seed"] = require_jax_seed(payload.get("seed"), name=f"{path}: seed")
    for fieldname in ("hidden1", "hidden2"):
        value = payload.get(fieldname)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{path}: {fieldname} must be a positive integer")
    return payload


def _late_window_slope(per_regime: np.ndarray, window: int) -> float:
    """OLS slope (accuracy per regime) over the final ``window`` regimes."""
    tail = per_regime[-window:]
    x = np.arange(tail.shape[0], dtype=np.float64)
    x = x - x.mean()
    denominator = float(np.sum(x * x))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x * (tail - tail.mean())) / denominator)


def _micro_shard_batch_contract(
    shards: Sequence[dict[str, Any]],
) -> tuple[MicroStreamConfig, dict[str, str]]:
    """Validate fields that must be identical across one derived artifact."""
    if not shards:
        raise ValueError("no shards given")
    configs = {tuple(sorted(shard["stream_config"].items())) for shard in shards}
    if len(configs) != 1:
        raise ValueError("shards span multiple stream configs; merge them separately")
    nets = {(shard["hidden1"], shard["hidden2"]) for shard in shards}
    if len(nets) != 1:
        raise ValueError("shards span multiple network sizes; merge them separately")
    reference_environment = shards[0]["environment"]
    environment_mismatches = [
        f"{shard['arm_name']}/seed={shard['seed']}"
        for shard in shards
        if shard["environment"] != reference_environment
    ]
    if environment_mismatches:
        raise ValueError(
            "shards span multiple runtime environments; process same-environment runs "
            f"separately (mismatched: {environment_mismatches})"
        )
    return (
        MicroStreamConfig.from_mapping(
            shards[0]["stream_config"], source="stream_config"
        ),
        dict(reference_environment),
    )


def _validate_micro_arm_contract(
    arm_name: str,
    per_seed: Mapping[int, dict[str, Any]],
) -> None:
    """Reject cross-seed mechanism or hyperparameter drift within one arm."""
    seeds = sorted(per_seed)
    for fieldname in ("hyperparameters", "mechanism"):
        reference = per_seed[seeds[0]][fieldname]
        mismatched = [
            seed for seed in seeds if per_seed[seed][fieldname] != reference
        ]
        if mismatched:
            host_arm = _require_exact_str("arm_name", arm_name)
            host_field = _require_exact_str("fieldname", fieldname)
            if type(reference) is str:
                host_ref = _require_exact_str("reference", reference)
                raise ValueError(
                    f"arm '{host_arm}' has inconsistent '{host_field}' across seeds: "
                    f"seed {seeds[0]} used '{host_ref}', seed(s) {mismatched} used "
                    "different values"
                )
            raise ValueError(
                f"arm '{host_arm}' has inconsistent '{host_field}' across seeds: "
                f"seed {seeds[0]} used non-string, seed(s) {mismatched} used "
                "different values"
            )


def merge_micro_shards(
    paths: Sequence[Path | str], bayes_samples: int = 200_000
) -> dict[str, Any]:
    """Merge shards of one (family, config) into a ranked summary with the
    analytic Bayes reference attached.

    Every arm must carry the same seed set: the ranking and the Bayes
    reference are only meaningful as paired comparisons on shared streams.

    Raises:
        ValueError: If shards duplicate an ``(arm, seed)`` pair, span more
            than one stream config or environment, drift within an arm, or
            cover different seed sets across arms.
    """
    shards = [load_micro_shard(path) for path in paths]
    config, reference_environment = _micro_shard_batch_contract(shards)
    quarter = max(1, config.n_regimes // 4)

    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for shard in shards:
        per_seed = by_arm.setdefault(shard["arm_name"], {})
        if shard["seed"] in per_seed:
            raise ValueError(
                f"duplicate shard for arm={shard['arm_name']} seed={shard['seed']}"
            )
        per_seed[shard["seed"]] = shard
    seed_sets = {arm: tuple(sorted(per_seed)) for arm, per_seed in sorted(by_arm.items())}
    if len(set(seed_sets.values())) != 1:
        raise ValueError(
            f"seed sets differ across arms: {seed_sets}; "
            "merge_micro_shards ranks arms on paired seeds only"
        )

    entries: list[dict[str, Any]] = []
    all_seeds: set[int] = set()
    for arm_name, per_seed in sorted(by_arm.items()):
        seeds = sorted(per_seed)
        all_seeds.update(seeds)
        _validate_micro_arm_contract(arm_name, per_seed)
        wall_clock_values = [per_seed[s]["wall_clock_seconds"] for s in seeds]
        host_arm2 = _require_exact_str("arm_name", arm_name)
        wall_clock_total = _finite_wall_clock_total(
            wall_clock_values,
            context=f"arm '{host_arm2}'",
        )
        curves = np.stack(
            [
                np.asarray(per_seed[s]["per_regime_accuracy"], dtype=np.float64)
                for s in seeds
            ]
        )
        per_seed_avg = curves.mean(axis=1)
        slopes = np.asarray(
            [_late_window_slope(curves[i], quarter) for i in range(len(seeds))]
        )
        entries.append(
            {
                "arm_name": arm_name,
                "mechanism": per_seed[seeds[0]]["mechanism"],
                "hyperparameters": per_seed[seeds[0]]["hyperparameters"],
                "seeds": seeds,
                "n_seeds": len(seeds),
                "average_online_accuracy_mean": float(per_seed_avg.mean()),
                "average_online_accuracy_stderr": (
                    float(per_seed_avg.std(ddof=1) / math.sqrt(len(seeds)))
                    if len(seeds) > 1
                    else 0.0
                ),
                "per_seed_average_online_accuracy": [
                    round(float(v), 6) for v in per_seed_avg
                ],
                "first_regime_accuracy_mean": float(curves[:, 0].mean()),
                "first_window_accuracy_mean": float(curves[:, :quarter].mean()),
                "late_window_accuracy_mean": float(curves[:, -quarter:].mean()),
                "late_window_slope_mean": float(slopes.mean()),
                "wall_clock_seconds_total": round(wall_clock_total, 3),
                "wall_clock_seconds_mean": round(
                    float(np.mean(wall_clock_values)),
                    3,
                ),
            }
        )
    entries.sort(key=lambda e: float(e["average_online_accuracy_mean"]), reverse=True)

    reference_seeds = sorted(all_seeds)
    references = [
        bayes_reference(config, seed, n_samples=bayes_samples)
        for seed in reference_seeds
    ]
    bayes_payload = {
        "seeds": reference_seeds,
        "n_samples": bayes_samples,
        "per_seed_bayes_accuracy": [
            round(reference.bayes_accuracy, 6) for reference in references
        ],
        "per_seed_mc_sem": [round(reference.mc_sem, 8) for reference in references],
        "bayes_accuracy_mean": float(
            np.mean([reference.bayes_accuracy for reference in references])
        ),
        "chance": 1.0 / config.n_classes,
    }
    return {
        "schema": MICRO_SUMMARY_SCHEMA,
        "suite_version": MICRO_GAUSS_SUITE_VERSION,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "created_unix": time.time(),
        "family": config.family,
        "stream_config": config.to_config(),
        "environment": dict(reference_environment),
        "hidden1": shards[0]["hidden1"],
        "hidden2": shards[0]["hidden2"],
        "n_shards": len(shards),
        "quarter_window": quarter,
        "results": entries,
        "bayes_reference": bayes_payload,
    }


# =============================================================================
# Transfer validation (pre-registered full-protocol ordering checks)
# =============================================================================


def transfer_validation(
    per_arm: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, Any]:
    """Check that the micro ladder reproduces the full-protocol ordering.

    ``per_arm`` maps arm name -> seed -> per-regime online accuracy. All
    :data:`LADDER_ARMS` must be present with identical seed sets (paired
    comparison). Primary checks (all must pass for ``transfer_valid``):

    - ``conditioning_dominates`` — sgd_norm beats upgd_raw on every seed and
      the conditioning delta is at least twice the gate delta (full protocol:
      +0.061 vs +0.011).
    - ``gate_small_positive`` — gated_norm beats sgd_norm by a positive but
      small margin (at most half the conditioning delta).
    - ``adam_decays`` — adamw's late-window accuracy is below its first
      window and its late-window slope is negative (full protocol: slope
      -0.00184/task, decays 0.78 -> 0.68).
    - ``adam_below_upgd_raw`` — adamw's overall metric is below upgd_raw's
      (full protocol at 200 tasks: 0.68 < 0.779).
    - ``naive_bayes_placement`` — upgd_raw < naive_bayes < sgd_norm (V3:
      0.7778 < 0.7851 < 0.8399).
    - ``champion_top`` — gated_norm is the best arm (champion 0.86449).

    Secondary (reported, not gating): ``adam_fast_early`` — adamw wins the
    first regime over upgd_raw (full protocol t1: 0.7694 vs 0.6928).
    """
    missing = [arm for arm in LADDER_ARMS if arm not in per_arm]
    if missing:
        raise ValueError(f"missing ladder arms: {missing}")
    seed_sets = {arm: tuple(sorted(per_arm[arm])) for arm in LADDER_ARMS}
    if len(set(seed_sets.values())) != 1:
        raise ValueError(f"seed sets differ across arms: {seed_sets}")
    seeds = list(seed_sets[LADDER_ARMS[0]])
    if not seeds:
        raise ValueError("no seeds given")
    lengths = {
        np.asarray(per_arm[arm][seed]).shape[0] for arm in LADDER_ARMS for seed in seeds
    }
    if len(lengths) != 1:
        raise ValueError(f"per-regime curves differ in length: {sorted(lengths)}")
    n_regimes = lengths.pop()
    quarter = max(1, n_regimes // 4)

    curves = {
        arm: np.stack([np.asarray(per_arm[arm][seed], dtype=np.float64) for seed in seeds])
        for arm in LADDER_ARMS
    }
    overall = {arm: curves[arm].mean(axis=1) for arm in LADDER_ARMS}
    first_regime = {arm: curves[arm][:, 0] for arm in LADDER_ARMS}
    first_window = {arm: curves[arm][:, :quarter].mean(axis=1) for arm in LADDER_ARMS}
    late_window = {arm: curves[arm][:, -quarter:].mean(axis=1) for arm in LADDER_ARMS}
    slopes = {
        arm: np.asarray(
            [_late_window_slope(curves[arm][i], quarter) for i in range(len(seeds))]
        )
        for arm in LADDER_ARMS
    }

    conditioning_delta = overall["sgd_norm"] - overall["upgd_raw"]
    gate_delta = overall["gated_norm"] - overall["sgd_norm"]

    checks: list[dict[str, Any]] = []

    def add_check(
        name: str,
        passed: bool,
        campaign_reference: str,
        values: dict[str, Any],
        primary: bool = True,
    ) -> None:
        checks.append(
            {
                "name": name,
                "primary": primary,
                "passed": bool(passed),
                "campaign_reference": campaign_reference,
                "values": values,
            }
        )

    add_check(
        "conditioning_dominates",
        bool(
            conditioning_delta.mean() > 0.0
            and np.all(conditioning_delta > 0.0)
            and conditioning_delta.mean() >= 2.0 * max(float(gate_delta.mean()), 0.0)
        ),
        "full protocol: conditioning +0.061 vs gate +0.011 "
        "(docs/research/ipmnist-theory.md decomposition)",
        {
            "conditioning_delta_mean": float(conditioning_delta.mean()),
            "per_seed_conditioning_delta": [round(float(v), 6) for v in conditioning_delta],
            "gate_delta_mean": float(gate_delta.mean()),
        },
    )
    add_check(
        "gate_small_positive",
        bool(
            gate_delta.mean() > 0.0
            and gate_delta.mean() <= conditioning_delta.mean() / 2.0
        ),
        "full protocol: gate +0.011, an order below conditioning +0.061",
        {
            "gate_delta_mean": float(gate_delta.mean()),
            "per_seed_gate_delta": [round(float(v), 6) for v in gate_delta],
            "all_seeds_positive": bool(np.all(gate_delta > 0.0)),
        },
    )
    add_check(
        "adam_decays",
        bool(
            late_window["adamw"].mean() < first_window["adamw"].mean()
            and slopes["adamw"].mean() < 0.0
        ),
        "full protocol: AdamW window means 0.7803 -> 0.7375, late slope "
        "-0.00184/task (accumulating Mode-2 damage)",
        {
            "first_window_mean": float(first_window["adamw"].mean()),
            "late_window_mean": float(late_window["adamw"].mean()),
            "late_window_slope_mean": float(slopes["adamw"].mean()),
        },
    )
    add_check(
        "adam_below_upgd_raw",
        bool(overall["adamw"].mean() < overall["upgd_raw"].mean()),
        "full protocol at 200 tasks: AdamW ~0.68 < UPGD-W 0.779",
        {
            "adamw_overall_mean": float(overall["adamw"].mean()),
            "upgd_raw_overall_mean": float(overall["upgd_raw"].mean()),
        },
    )
    add_check(
        "adam_fast_early",
        bool(first_regime["adamw"].mean() > first_regime["upgd_raw"].mean()),
        "full protocol task 1: AdamW 0.7694 > UPGD-W 0.6928 (+0.077)",
        {
            "adamw_first_regime_mean": float(first_regime["adamw"].mean()),
            "upgd_raw_first_regime_mean": float(first_regime["upgd_raw"].mean()),
        },
        primary=False,
    )
    add_check(
        "naive_bayes_placement",
        bool(
            overall["upgd_raw"].mean()
            < overall["naive_bayes"].mean()
            < overall["sgd_norm"].mean()
        ),
        "V3 development validation: naive Bayes 0.7851 beats published UPGD-W "
        "0.7778 but stays below conditioned SGD 0.8399",
        {
            "upgd_raw_overall_mean": float(overall["upgd_raw"].mean()),
            "naive_bayes_overall_mean": float(overall["naive_bayes"].mean()),
            "sgd_norm_overall_mean": float(overall["sgd_norm"].mean()),
        },
    )
    best_arm = max(LADDER_ARMS, key=lambda arm: float(overall[arm].mean()))
    add_check(
        "champion_top",
        best_arm == "gated_norm",
        "campaign champion sigma0_shiftnorm_d099 0.86449 (n=20) tops the ladder",
        {
            "best_arm": best_arm,
            "gated_norm_overall_mean": float(overall["gated_norm"].mean()),
        },
    )

    primary_checks = [check for check in checks if check["primary"]]
    return {
        "schema": MICRO_VALIDATION_SCHEMA,
        "suite_version": MICRO_GAUSS_SUITE_VERSION,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "created_unix": time.time(),
        "seeds": seeds,
        "n_regimes": int(n_regimes),
        "quarter_window": quarter,
        "arms": {
            arm: {
                "overall_mean": float(overall[arm].mean()),
                "per_seed_overall": [round(float(v), 6) for v in overall[arm]],
                "first_regime_mean": float(first_regime[arm].mean()),
                "first_window_mean": float(first_window[arm].mean()),
                "late_window_mean": float(late_window[arm].mean()),
                "late_window_slope_mean": float(slopes[arm].mean()),
            }
            for arm in LADDER_ARMS
        },
        "checks": checks,
        "transfer_valid": bool(all(check["passed"] for check in primary_checks)),
        "secondary_all_passed": bool(
            all(check["passed"] for check in checks if not check["primary"])
        ),
    }


def transfer_validation_from_shards(paths: Sequence[Path | str]) -> dict[str, Any]:
    """Build and run :func:`transfer_validation` from ladder shards (M1 only)."""
    shards = [load_micro_shard(path) for path in paths]
    config, environment = _micro_shard_batch_contract(shards)
    families = {shard["family"] for shard in shards}
    if families != {"input_permutation"}:
        raise ValueError(
            "transfer validation is defined on the input_permutation family "
            f"(M1); got {sorted(families)}"
        )
    raw_per_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for shard in shards:
        per_seed = raw_per_arm.setdefault(shard["arm_name"], {})
        if shard["seed"] in per_seed:
            raise ValueError(
                f"duplicate shard for arm={shard['arm_name']} seed={shard['seed']}"
            )
        per_seed[shard["seed"]] = shard
    per_arm: dict[str, dict[int, np.ndarray]] = {}
    for arm_name, raw_per_seed in raw_per_arm.items():
        _validate_micro_arm_contract(arm_name, raw_per_seed)
        per_arm[arm_name] = {
            seed: np.asarray(shard["per_regime_accuracy"], dtype=np.float64)
            for seed, shard in raw_per_seed.items()
        }
    report = transfer_validation(per_arm)
    report["family"] = "input_permutation"
    report["stream_config"] = config.to_config()
    report["environment"] = environment
    report["hidden1"] = shards[0]["hidden1"]
    report["hidden2"] = shards[0]["hidden2"]
    return report


# =============================================================================
# CLI
# =============================================================================


def _atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically (re)write one derived JSON artifact (summaries are
    regenerable from immutable shards, so replacement is allowed here)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _strict_json_text(payload)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _add_stream_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = MicroStreamConfig()
    parser.add_argument("--n-regimes", type=int, default=defaults.n_regimes)
    parser.add_argument("--regime-length", type=int, default=defaults.regime_length)
    parser.add_argument("--dim", type=int, default=defaults.dim)
    parser.add_argument("--n-classes", type=int, default=defaults.n_classes)
    parser.add_argument("--n-components", type=int, default=defaults.n_components)
    parser.add_argument(
        "--spectrum-decades", type=float, default=defaults.spectrum_decades
    )
    parser.add_argument(
        "--mean-separation", type=float, default=defaults.mean_separation
    )
    parser.add_argument(
        "--component-scale", type=float, default=defaults.component_scale
    )
    parser.add_argument(
        "--component-sparsity", type=int, default=defaults.component_sparsity
    )
    parser.add_argument(
        "--class-sparsity", type=float, default=defaults.class_sparsity
    )
    parser.add_argument("--noise-scale", type=float, default=defaults.noise_scale)
    parser.add_argument("--offset-scale", type=float, default=defaults.offset_scale)
    parser.add_argument("--scale-min", type=float, default=defaults.scale_shift_min)
    parser.add_argument("--scale-max", type=float, default=defaults.scale_shift_max)
    parser.add_argument(
        "--recurrence-pool", type=int, default=defaults.recurrence_pool
    )
    parser.add_argument("--hidden1", type=int, default=75)
    parser.add_argument("--hidden2", type=int, default=38)


def _config_from_args(args: argparse.Namespace) -> MicroStreamConfig:
    return MicroStreamConfig(
        family=args.family,
        n_regimes=args.n_regimes,
        regime_length=args.regime_length,
        dim=args.dim,
        n_classes=args.n_classes,
        n_components=args.n_components,
        spectrum_decades=args.spectrum_decades,
        mean_separation=args.mean_separation,
        component_scale=args.component_scale,
        component_sparsity=args.component_sparsity,
        class_sparsity=args.class_sparsity,
        noise_scale=args.noise_scale,
        offset_scale=args.offset_scale,
        scale_shift_min=args.scale_min,
        scale_shift_max=args.scale_max,
        recurrence_pool=args.recurrence_pool,
    )


def _run_or_skip_shard(
    config: MicroStreamConfig,
    arm_name: str,
    seed: int,
    out_dir: Path,
    hidden1: int,
    hidden2: int,
) -> Path:
    """Idempotent shard execution: existing shards are validated and kept.

    A shard is only reused when it was produced by the same stream config
    and the same network size; anything else must go to a fresh directory.
    """
    path = micro_shard_path(out_dir, config.family, arm_name, seed)
    if path.exists():
        payload = load_micro_shard(path)
        if payload["stream_config"] != config.to_config():
            raise ValueError(
                f"{path}: existing shard was produced by a different stream "
                "config; use a fresh --out directory"
            )
        if (payload["hidden1"], payload["hidden2"]) != (hidden1, hidden2):
            raise ValueError(
                f"{path}: existing shard was produced by a different network size "
                f"(hidden1={payload['hidden1']}, hidden2={payload['hidden2']}); "
                f"requested hidden1={hidden1}, hidden2={hidden2}; "
                "use a fresh --out directory"
            )
        logger.info("shard exists, skipping: %s", path)
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_micro_arm(config, arm_name, seed, hidden1=hidden1, hidden2=hidden2)
    write_micro_shard(path, micro_shard_payload(result))
    logger.info(
        "%s/%s seed=%d overall=%.4f wall=%.2fs -> %s",
        config.family,
        arm_name,
        seed,
        result.overall_accuracy,
        result.wall_clock_seconds,
        path,
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Micro-suite CLI: ``run`` one shard; ``ladder`` runs arms x seeds, merges
    a summary, and (on M1 with the full ladder) writes the transfer-validation
    receipt (exit 0 = ordering reproduced, 2 = not)."""
    parser = argparse.ArgumentParser(
        description="First-principles micro-continual benchmark suite (M1-M4)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one (family, arm, seed) shard")
    run_p.add_argument("--family", required=True, choices=FAMILIES)
    run_p.add_argument("--arm", required=True, choices=sorted(MICRO_ARM_REGISTRY))
    run_p.add_argument("--seed", type=int, required=True)
    run_p.add_argument("--out", type=Path, required=True)
    _add_stream_arguments(run_p)

    ladder_p = sub.add_parser(
        "ladder", help="run the method ladder, merge, and validate the proxy"
    )
    ladder_p.add_argument("--family", required=True, choices=FAMILIES)
    ladder_p.add_argument("--seeds", type=int, nargs="+", required=True)
    ladder_p.add_argument(
        "--arms", nargs="+", choices=sorted(MICRO_ARM_REGISTRY), default=list(LADDER_ARMS)
    )
    ladder_p.add_argument("--out", type=Path, required=True)
    ladder_p.add_argument("--bayes-samples", type=int, default=200_000)
    _add_stream_arguments(ladder_p)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    config = _config_from_args(args)

    if args.command == "run":
        seed = require_jax_seed(args.seed, name="seed")
        _run_or_skip_shard(
            config, args.arm, seed, args.out, args.hidden1, args.hidden2
        )
        return 0

    # ladder
    seeds = require_unique_jax_seeds(args.seeds, name="seeds")
    paths: list[Path] = []
    for arm_name in args.arms:
        for seed in seeds:
            paths.append(
                _run_or_skip_shard(
                    config, arm_name, seed, args.out, args.hidden1, args.hidden2
                )
            )
    summary = merge_micro_shards(paths, bayes_samples=args.bayes_samples)
    summary_path = args.out / f"summary_{config.family}.json"
    _atomic_replace_json(summary_path, summary)
    logger.info("merged %d shards -> %s", summary["n_shards"], summary_path)

    if config.family == "input_permutation" and set(args.arms) >= set(LADDER_ARMS):
        report = transfer_validation_from_shards(paths)
        transfer_path = args.out / f"transfer_{config.family}.json"
        _atomic_replace_json(transfer_path, report)
        logger.info(
            "transfer_valid=%s (primary checks: %s) -> %s",
            report["transfer_valid"],
            {c["name"]: c["passed"] for c in report["checks"] if c["primary"]},
            transfer_path,
        )
        if not report["transfer_valid"]:
            logger.error(
                "micro proxy did NOT reproduce the full-protocol ordering; "
                "receipt preserved at %s", transfer_path,
            )
            return 2
    return 0


# =============================================================================
# Provisional digits-based suite (rule-discovery track compatibility)
# =============================================================================
#
# Authored by the update-rule discovery track before the canonical
# first-principles suite above landed; its harness (and
# ``tests/test_rule_discovery.py``) imports this surface. Kept intact pending
# reconciliation — see ``outputs/micro_continual/SUITE.md``. New work should
# target the Gaussian suite above (analytic Bayes references, campaign-parity
# arms, transfer-validated M1).

MICRO_SUITE_VERSION = "provisional-v1"

#: Search tasks — fitness may read these.
SEARCH_TASKS: tuple[str, ...] = ("M1", "M2", "M3")
#: Holdout tasks — selection validation ONLY; never search fitness.
HOLDOUT_TASKS: tuple[str, ...] = ("M4", "M1p")


@dataclass(frozen=True)
class MicroTaskConfig:
    """One micro-suite stream family.

    Attributes:
        name: Registry key (``M1``/``M2``/``M3``/``M4``/``M1p``).
        kind: Non-stationarity axis.
        role: ``"search"`` or ``"holdout"``.
        input_dim: Feature dimensionality (64 full digits, 49 cropped).
        n_classes: Label arity.
        n_tasks: Number of regime blocks.
        task_length: Online steps per block.
        hidden1: First hidden width of the micro MLP.
        hidden2: Second hidden width of the micro MLP.
        crop: Use the 7x7-cropped digit features.
    """

    name: str
    kind: str
    role: str
    input_dim: int
    n_classes: int
    n_tasks: int
    task_length: int
    hidden1: int
    hidden2: int
    crop: bool


MICRO_SUITE: dict[str, MicroTaskConfig] = {
    "M1": MicroTaskConfig(
        name="M1", kind="input_permutation", role="search",
        input_dim=64, n_classes=10, n_tasks=8, task_length=500,
        hidden1=32, hidden2=16, crop=False,
    ),
    "M2": MicroTaskConfig(
        name="M2", kind="label_permutation", role="search",
        input_dim=64, n_classes=10, n_tasks=8, task_length=500,
        hidden1=32, hidden2=16, crop=False,
    ),
    "M3": MicroTaskConfig(
        name="M3", kind="affine_drift", role="search",
        input_dim=64, n_classes=10, n_tasks=8, task_length=500,
        hidden1=32, hidden2=16, crop=False,
    ),
    "M4": MicroTaskConfig(
        name="M4", kind="permutation_affine", role="holdout",
        input_dim=64, n_classes=10, n_tasks=8, task_length=500,
        hidden1=32, hidden2=16, crop=False,
    ),
    "M1p": MicroTaskConfig(
        name="M1p", kind="input_permutation", role="holdout",
        input_dim=49, n_classes=10, n_tasks=12, task_length=300,
        hidden1=32, hidden2=16, crop=True,
    ),
}


@dataclass(frozen=True)
class MicroStream:
    """A materialized micro stream (numpy, host-side).

    Attributes:
        xs: ``f32[n_steps, input_dim]`` transformed observations.
        ys: ``i32[n_steps]`` (possibly remapped) labels.
        example_indices: ``i32[n_steps]`` base-dataset row of each step.
        config: The generating configuration.
        seed: The generating seed.
    """

    xs: np.ndarray
    ys: np.ndarray
    example_indices: np.ndarray
    config: MicroTaskConfig
    seed: int


@lru_cache(maxsize=2)
def _digits_cache(crop: bool) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits  # type: ignore[import-untyped]

    x_raw, y_raw = load_digits(return_X_y=True)
    x = (np.asarray(x_raw, dtype=np.float32) / 8.0) - 1.0  # 0..16 -> [-1, 1]
    y = np.asarray(y_raw, dtype=np.int32)
    if crop:
        x = x.reshape(-1, 8, 8)[:, :7, :7].reshape(-1, 49)
    return x, y


def load_digits_features(crop: bool) -> tuple[np.ndarray, np.ndarray]:
    """Digit features scaled to ``[-1, 1]`` (optionally 7x7-cropped) + labels."""
    x, y = _digits_cache(bool(crop))
    return x.copy(), y.copy()


def build_micro_stream(config: MicroTaskConfig, seed: int) -> MicroStream:
    """Materialize the deterministic stream for one ``(config, seed)``.

    Per task ``t`` the transform key is ``fold_in(fold_in(key(seed), t), axis)``
    so streams are reproducible and task-local; examples are drawn uniformly
    with replacement from the base dataset (1,797 rows << task lengths, so
    without-replacement sampling is not available at micro scale).
    """
    x_base, y_base = _digits_cache(config.crop)
    n_rows = x_base.shape[0]
    root = jr.key(np.uint32(seed))
    xs_blocks: list[np.ndarray] = []
    ys_blocks: list[np.ndarray] = []
    idx_blocks: list[np.ndarray] = []
    for task in range(config.n_tasks):
        task_key = jr.fold_in(root, task)
        k_examples, k_perm, k_scale, k_offset, k_labels = jr.split(task_key, 5)
        indices = np.asarray(
            jr.randint(k_examples, (config.task_length,), 0, n_rows)
        ).astype(np.int32)
        x_task = x_base[indices].copy()
        y_task = y_base[indices].copy()
        if config.kind in ("input_permutation", "permutation_affine"):
            perm = np.asarray(jr.permutation(k_perm, config.input_dim))
            x_task = x_task[:, perm]
        if config.kind in ("affine_drift", "permutation_affine"):
            log_scale = np.asarray(
                jr.uniform(
                    k_scale,
                    (config.input_dim,),
                    jnp.float32,
                    float(np.log(0.3)),
                    float(np.log(3.0)),
                )
            )
            offset = np.asarray(
                jr.uniform(k_offset, (config.input_dim,), jnp.float32, -0.5, 0.5)
            )
            x_task = x_task * np.exp(log_scale, dtype=np.float32) + offset
        if config.kind == "label_permutation":
            label_map = np.asarray(jr.permutation(k_labels, config.n_classes)).astype(
                np.int32
            )
            y_task = label_map[y_task]
        xs_blocks.append(np.asarray(x_task, dtype=np.float32))
        ys_blocks.append(np.asarray(y_task, dtype=np.int32))
        idx_blocks.append(indices)
    return MicroStream(
        xs=np.concatenate(xs_blocks, axis=0),
        ys=np.concatenate(ys_blocks, axis=0),
        example_indices=np.concatenate(idx_blocks, axis=0),
        config=config,
        seed=seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
