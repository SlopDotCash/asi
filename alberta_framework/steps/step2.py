# mypy: disable-error-code="call-arg"
"""Production Step 2 kernel.

The Step 2 production surface exposes the current packaged learner:
target-structure UPGD.  It is a single learner with online hidden-feature
utility, low-utility perturbation, ObGD-bounded updates, and vector-output
heads.  It is not a theorem of universal representation learning; it is the
current development-selected kernel for the supervised Step 2 acceptance
matrix.

For retained class-view memory, Step 2 also exposes a JAX fixed-budget
prototype memory and a packaged UPGD-memory learner that updates both the
differentiable UPGD path and the memory path every step. Their current
contracts are exercised in ``tests/test_prototype_memory.py`` and
``tests/test_upgd_memory.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._float32 import round_real_to_float32_with_ratio
from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.associative_memory import (
    AssociativeFeatureFamily,
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    run_associative_memory_arrays,
)
from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
)
from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
)
from alberta_framework.core.upgd import UPGDLearner, run_upgd_arrays
from alberta_framework.core.upgd_memory import UPGDMemoryConfig, UPGDMemoryLearner
from alberta_framework.streams.out_of_class import (
    CompositionalStream,
    FrequencyMismatchStream,
    OutOfClassPolynomialStream,
)

Step2StreamName = Literal["polynomial", "frequency", "compositional"]
Step2ReadoutMode = Literal[
    "linear_mse",
    "softmax_ce",
    "adaptive_simplex",
    "factorized_simplex",
    "adaptive_factorized_simplex",
    "two_timescale_simplex",
]
Step2HybridReadoutMode = Literal["linear_mse", "softmax_ce"]

_VALID_STEP2_STREAMS: frozenset[str] = frozenset(
    {"polynomial", "frequency", "compositional"}
)
_VALID_STEP2_READOUT_MODES: frozenset[str] = frozenset(
    {
        "linear_mse",
        "softmax_ce",
        "adaptive_simplex",
        "factorized_simplex",
        "adaptive_factorized_simplex",
        "two_timescale_simplex",
    }
)
_VALID_STEP2_LOSS_NORMALIZATIONS: frozenset[str] = frozenset(
    {"target_structure", "target_density"}
)
_STEP2_KERNEL_CONFIG_KEYS = frozenset(
    {
        "feature_dim",
        "n_heads",
        "hidden_sizes",
        "stream",
        "readout_mode",
        "step_size",
        "loss_normalization",
        "context_length",
        "noise_std",
    }
)
_STEP2_STRICT_DIGIT_CONFIG_KEYS = frozenset(
    {"n_heads", "hidden_sizes", "step_size"}
)
_STEP2_MEMORY_CONFIG_KEYS = frozenset(
    {
        "feature_dim",
        "n_classes",
        "slots_per_class",
        "update_rate",
        "novelty_threshold",
        "bandwidth",
    }
)
_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES: tuple[type, ...] = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
)


def _require_exact_keys(
    config_name: str,
    payload: dict[str, object],
    expected: frozenset[str],
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{config_name} payload keys must be exactly {sorted(expected)!r}"
        )


def finite_real_and_float32(name: str, value: object) -> tuple[Real, int, int, float]:
    """Return the original real, exact ratio, and finite binary32 rounding."""
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    real = cast(Real, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except (FloatingPointError, OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must narrow to a finite float32, got {value!r}") from None
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must narrow to a finite float32, got {value!r}")
    return real, numerator, denominator, narrowed


def canonical_float32_storage(value: object, narrowed: float) -> float:
    actual_type = type(value)
    if (actual_type is int or actual_type is float) and (
        bool(narrowed != 0.0) or value == 0
    ):
        return float(cast(int | float, value))
    return float(narrowed)


def _require_real(name: str, value: object) -> float:
    real, _, _, narrowed = finite_real_and_float32(name, value)
    return canonical_float32_storage(real, narrowed)


def _require_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real < 0.0
        or not real <= 1.0
        or numerator < 0
        or numerator > denominator
        or narrowed < 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_half_open_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real <= 0.0
        or not real <= 1.0
        or numerator <= 0
        or numerator > denominator
        or narrowed <= 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(
            f"{name} must be positive in float32 execution sink, got {value!r}"
        )
    return canonical_float32_storage(real, narrowed)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        number = int(cast(Integral, value))
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive, got {value!r}")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")
    return number


def _validate_step2_kernel_config(config: Step2KernelConfig) -> None:
    feature_dim = _require_int(
        "feature_dim", config.feature_dim, minimum=1, maximum=_INT32_MAX
    )
    n_heads = _require_int(
        "n_heads", config.n_heads, minimum=1, maximum=_INT32_MAX
    )
    if type(config.hidden_sizes) is not tuple:
        raise ValueError(
            f"hidden_sizes must be a tuple of integers, got {config.hidden_sizes!r}"
        )
    canonical_hidden: list[int] = []
    for h in config.hidden_sizes:
        canonical_hidden.append(
            _require_int(
                "hidden_sizes element", h, minimum=1, maximum=_INT32_MAX
            )
        )
    if type(config.stream) is not str or config.stream not in _VALID_STEP2_STREAMS:
        raise ValueError(
            f"unknown Step 2 stream field {config.stream!r}; "
            f"expected one of {sorted(_VALID_STEP2_STREAMS)}"
        )
    if config.stream == "polynomial" and feature_dim < 3:
        raise ValueError("feature_dim must be at least 3 for the polynomial stream")
    if (
        type(config.readout_mode) is not str
        or config.readout_mode not in _VALID_STEP2_READOUT_MODES
    ):
        raise ValueError(
            f"unknown Step 2 readout_mode field {config.readout_mode!r}; "
            f"expected one of {sorted(_VALID_STEP2_READOUT_MODES)}"
        )
    if (
        type(config.loss_normalization) is not str
        or config.loss_normalization not in _VALID_STEP2_LOSS_NORMALIZATIONS
    ):
        raise ValueError(
            f"unknown Step 2 loss_normalization field {config.loss_normalization!r}; "
            f"expected one of {sorted(_VALID_STEP2_LOSS_NORMALIZATIONS)}"
        )
    step_size = _require_nonnegative_real("step_size", config.step_size)
    context_length = _require_int(
        "context_length",
        config.context_length,
        minimum=1,
        maximum=_INT32_MAX,
    )
    noise_std = _require_nonnegative_real("noise_std", config.noise_std)
    object.__setattr__(config, "feature_dim", feature_dim)
    object.__setattr__(config, "n_heads", n_heads)
    object.__setattr__(config, "hidden_sizes", tuple(canonical_hidden))
    object.__setattr__(config, "step_size", step_size)
    object.__setattr__(config, "context_length", context_length)
    object.__setattr__(config, "noise_std", noise_std)


def _validate_step2_strict_digit_config(config: Step2StrictDigitReadoutConfig) -> None:
    n_heads = _require_int(
        "n_heads", config.n_heads, minimum=1, maximum=_INT32_MAX
    )
    if type(config.hidden_sizes) is not tuple:
        raise ValueError(
            f"hidden_sizes must be a tuple of integers, got {config.hidden_sizes!r}"
        )
    canonical_hidden: list[int] = []
    for h in config.hidden_sizes:
        canonical_hidden.append(
            _require_int(
                "hidden_sizes element", h, minimum=1, maximum=_INT32_MAX
            )
        )
    step_size = _require_nonnegative_real("step_size", config.step_size)
    object.__setattr__(config, "n_heads", n_heads)
    object.__setattr__(config, "hidden_sizes", tuple(canonical_hidden))
    object.__setattr__(config, "step_size", step_size)


def _validate_step2_memory_config(config: Step2MemoryConfig) -> None:
    feature_dim = _require_int(
        "feature_dim", config.feature_dim, minimum=1, maximum=_INT32_MAX
    )
    n_classes = _require_int(
        "n_classes", config.n_classes, minimum=2, maximum=_INT32_MAX
    )
    slots_per_class = _require_int(
        "slots_per_class",
        config.slots_per_class,
        minimum=1,
        maximum=_INT32_MAX,
    )
    update_rate = _require_half_open_unit_interval("update_rate", config.update_rate)
    novelty_threshold = _require_nonnegative_real(
        "novelty_threshold",
        config.novelty_threshold,
    )
    bandwidth = _require_positive_real("bandwidth", config.bandwidth)
    object.__setattr__(config, "feature_dim", feature_dim)
    object.__setattr__(config, "n_classes", n_classes)
    object.__setattr__(config, "slots_per_class", slots_per_class)
    object.__setattr__(config, "update_rate", update_rate)
    object.__setattr__(config, "novelty_threshold", novelty_threshold)
    object.__setattr__(config, "bandwidth", bandwidth)


@dataclass(frozen=True)
class Step2KernelConfig:
    """Config for the production Step 2 UPGD kernel."""

    feature_dim: int = 8
    n_heads: int = 3
    hidden_sizes: tuple[int, ...] = (32,)
    stream: Step2StreamName = "polynomial"
    readout_mode: Step2ReadoutMode = "linear_mse"
    step_size: float = 0.03
    loss_normalization: Literal["target_structure", "target_density"] = (
        "target_structure"
    )
    context_length: int = 128
    noise_std: float = 0.05

    def __post_init__(self) -> None:
        """Reject invalid parameters and canonicalize scalars."""
        _validate_step2_kernel_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2KernelConfig:
        """Reconstruct from :meth:`to_dict` output."""
        _require_exact_keys(cls.__name__, payload, _STEP2_KERNEL_CONFIG_KEYS)
        config = dict(payload)
        if isinstance(config["hidden_sizes"], list):
            config["hidden_sizes"] = tuple(cast(list[int], config["hidden_sizes"]))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2StrictDigitReadoutConfig:
    """Config for the strict one-branch digit/readout Step 2 learner.

    ``step_size=0.018`` is a frozen empirical calibration for the
    sklearn-digits stream family, not a derived value; see
    :meth:`UPGDLearner.step2_strict_digit_readout_default` for the full set
    of frozen constants and the two-timescale readout it configures.
    """

    n_heads: int = 10
    hidden_sizes: tuple[int, ...] = (64, 64)
    step_size: float = 0.018

    def __post_init__(self) -> None:
        """Reject invalid parameters and canonicalize scalars."""
        _validate_step2_strict_digit_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> Step2StrictDigitReadoutConfig:
        """Reconstruct from :meth:`to_dict` output."""
        _require_exact_keys(cls.__name__, payload, _STEP2_STRICT_DIGIT_CONFIG_KEYS)
        config = dict(payload)
        if isinstance(config["hidden_sizes"], list):
            config["hidden_sizes"] = tuple(cast(list[int], config["hidden_sizes"]))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2MemoryConfig:
    """Config for the production Step 2 retained-view memory."""

    feature_dim: int = 784
    n_classes: int = 10
    slots_per_class: int = 20
    update_rate: float = 0.3
    novelty_threshold: float = 0.08
    bandwidth: float = 0.01

    def __post_init__(self) -> None:
        """Reject invalid parameters and canonicalize scalars."""
        _validate_step2_memory_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2MemoryConfig:
        """Reconstruct from :meth:`to_dict` output."""
        _require_exact_keys(cls.__name__, payload, _STEP2_MEMORY_CONFIG_KEYS)
        return cls(**cast(Any, payload))


@dataclass(frozen=True)
class Step2AssociativeConfig:
    """Config for the Step 2 fast/slow associative sequence learner."""

    vocab_size: int = 16
    block_size: int = 8
    suffix_length: int = 4
    feature_family: AssociativeFeatureFamily = "token_suffix_pair"
    max_features: int = 512
    write_lr: float = 1.0
    retention: float = 0.80
    utility_lr: float = 0.10
    utility_decay: float = 0.995
    min_weight: float = 0.02
    max_weight: float = 8.0
    logit_scale: float = 4.0
    normalize_by_weight: bool = True
    adaptive_feature_family: bool = False
    adaptive_window: bool = False
    adaptive_budget: bool = False
    scope_lr: float = 0.05
    budget_lr: float = 0.05
    initial_budget_fraction: float = 0.5
    min_effective_budget: int = 1
    scope_logit_clip: float = 8.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2AssociativeConfig:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**cast(Any, payload))

    def to_core_config(self) -> AssociativeMemoryConfig:
        """Return the core associative-memory config."""
        return AssociativeMemoryConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            suffix_length=self.suffix_length,
            feature_family=self.feature_family,
            max_features=self.max_features,
            write_lr=self.write_lr,
            retention=self.retention,
            utility_lr=self.utility_lr,
            utility_decay=self.utility_decay,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            logit_scale=self.logit_scale,
            normalize_by_weight=self.normalize_by_weight,
            adaptive_feature_family=self.adaptive_feature_family,
            adaptive_window=self.adaptive_window,
            adaptive_budget=self.adaptive_budget,
            scope_lr=self.scope_lr,
            budget_lr=self.budget_lr,
            initial_budget_fraction=self.initial_budget_fraction,
            min_effective_budget=self.min_effective_budget,
            scope_logit_clip=self.scope_logit_clip,
        )


@dataclass(frozen=True)
class Step2HybridConfig:
    """Config for the production Step 2 UPGD plus memory learner.

    Fields mirror
    :class:`~alberta_framework.core.upgd_memory.UPGDMemoryConfig`
    one-for-one; see that class for per-field semantics.  The knobs fall
    into four groups:

    1. Output-head plasticity pressure — ``upgd_head_loss_pressure_*``
       (extra head step-size when the fast/slow loss ratio spikes) and
       ``upgd_head_repetition_*`` (extra head step-size while the target
       vector repeats).
    2. Prototype-memory geometry and allocation —
       ``slots_per_class``, ``memory_update_rate``,
       ``initial_novelty_threshold`` with its online adaptation
       (``novelty_adaptation_rate``, ``target_allocation_rate``), and
       ``memory_bandwidth``.
    3. Learned memory-vs-UPGD blending — the ``*_logit_*`` coefficients
       and ``reliability_decay``.
    4. The causal target-trace prior — ``target_trace_*``, applied only
       during updates so held-out prediction stays observation-based.
    """

    feature_dim: int = 784
    n_heads: int = 10
    hidden_sizes: tuple[int, ...] = (64,)
    readout_mode: Step2HybridReadoutMode = "softmax_ce"
    upgd_step_size: float = 0.03
    upgd_head_step_size_multiplier: float = 1.0
    upgd_head_bias_step_size_multiplier: float = 1.0
    upgd_head_loss_pressure_gate_ratio: float = 0.0
    upgd_head_loss_pressure_multiplier: float = 0.0
    upgd_head_loss_pressure_warmup_steps: int = 0
    upgd_head_repetition_multiplier: float = 0.0
    upgd_head_repetition_decay: float = 0.9
    upgd_head_repetition_delta_threshold: float = 0.05
    upgd_head_repetition_pressure_threshold: float = 0.0
    upgd_head_repetition_warmup_steps: int = 0
    slots_per_class: int = 20
    memory_update_rate: float = 0.3
    initial_novelty_threshold: float = 0.08
    memory_bandwidth: float = 0.01
    initial_memory_logit: float = 0.0
    memory_logit_step_size: float = 0.25
    confidence_logit_scale: float = 2.0
    reliability_logit_scale: float = 8.0
    reliability_decay: float = 0.98
    target_trace_blend_scale: float = 0.8
    target_trace_pressure_threshold: float = 0.5
    novelty_adaptation_rate: float = 0.02
    target_allocation_rate: float = 0.18

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2HybridConfig:
        """Reconstruct from :meth:`to_dict` output."""
        config = dict(payload)
        config["hidden_sizes"] = tuple(cast(list[int], config["hidden_sizes"]))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2TemporalContextConfig:
    """Config for the packaged phase-context UPGD stressor kernel.

    ``periods`` drives the sin/cos clock features of
    :class:`~alberta_framework.core.temporal_context.TemporalContextConfig`:
    each period contributes one sin/cos pair, and with phase products
    enabled (as :func:`make_step2_temporal_context` does) every pair is
    also multiplied against each input channel.  The 13-period grid spans
    drift timescales from 32 to 192 steps — denser at short periods
    (8-step spacing through 96) and coarser above.  It is a fixed
    hyperparameter grid, not a derived quantity.
    """

    feature_dim: int = 12
    n_heads: int = 1
    hidden_sizes: tuple[int, ...] = (64,)
    step_size: float = 0.03
    periods: tuple[float, ...] = (
        32.0,
        40.0,
        48.0,
        56.0,
        64.0,
        72.0,
        80.0,
        88.0,
        96.0,
        112.0,
        128.0,
        160.0,
        192.0,
    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["periods"] = list(self.periods)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2TemporalContextConfig:
        """Reconstruct from :meth:`to_dict` output."""
        config = dict(payload)
        config["hidden_sizes"] = tuple(cast(list[int], config["hidden_sizes"]))
        config["periods"] = tuple(cast(list[float], config["periods"]))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2SmokeResult:
    """Summary returned by :func:`run_step2_smoke`."""

    config: Step2KernelConfig
    steps: int
    seed: int
    final_window_mse: float
    metrics_shape: tuple[int, ...]
    finite: bool
    learner_config: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["metrics_shape"] = list(self.metrics_shape)
        return payload


@dataclass(frozen=True)
class Step2AssociativeSmokeResult:
    """Summary returned by :func:`run_step2_associative_smoke`."""

    config: Step2AssociativeConfig
    steps: int
    seed: int
    initial_window_nll: float
    final_window_nll: float
    metrics_shape: tuple[int, ...]
    finite: bool
    learner_config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["metrics_shape"] = list(self.metrics_shape)
        return payload


def make_step2_learner(config: Step2KernelConfig | None = None) -> UPGDLearner:
    """Create the packaged Step 2 target-structure UPGD learner."""
    cfg = config or Step2KernelConfig()
    return UPGDLearner.step2_default(
        n_heads=cfg.n_heads,
        hidden_sizes=cfg.hidden_sizes,
        loss_normalization=cfg.loss_normalization,
        readout_mode=cfg.readout_mode,
        step_size=cfg.step_size,
    )


def make_step2_strict_digit_readout_learner(
    config: Step2StrictDigitReadoutConfig | None = None,
) -> UPGDLearner:
    """Create the strict online-MSE digit/readout Step 2 learner.

    This is the heavier two-timescale simplex branch selected for
    sklearn-digits-style one-hot online classification streams.  The broad
    supervised default remains :func:`make_step2_learner`.
    """
    cfg = config or Step2StrictDigitReadoutConfig()
    return UPGDLearner.step2_strict_digit_readout_default(
        n_heads=cfg.n_heads,
        hidden_sizes=cfg.hidden_sizes,
        step_size=cfg.step_size,
    )


def make_step2_memory_learner(
    config: Step2MemoryConfig | None = None,
) -> PrototypeMemoryLearner:
    """Create the packaged Step 2 retained-view memory learner."""
    cfg = config or Step2MemoryConfig()
    return PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=cfg.feature_dim,
            n_classes=cfg.n_classes,
            slots_per_class=cfg.slots_per_class,
            update_rate=cfg.update_rate,
            novelty_threshold=cfg.novelty_threshold,
            bandwidth=cfg.bandwidth,
        )
    )


def make_step2_associative_learner(
    config: Step2AssociativeConfig | None = None,
) -> AssociativeMemoryLearner:
    """Create the Step 2 fast/slow associative sequence learner."""
    cfg = config or Step2AssociativeConfig()
    return AssociativeMemoryLearner(cfg.to_core_config())


def make_step2_hybrid_learner(
    config: Step2HybridConfig | None = None,
) -> UPGDMemoryLearner:
    """Create the Step 2 UPGD plus adaptive prototype-memory learner."""
    cfg = config or Step2HybridConfig()
    return UPGDMemoryLearner(
        UPGDMemoryConfig(
            feature_dim=cfg.feature_dim,
            n_heads=cfg.n_heads,
            hidden_sizes=cfg.hidden_sizes,
            readout_mode=cfg.readout_mode,
            upgd_step_size=cfg.upgd_step_size,
            upgd_head_step_size_multiplier=cfg.upgd_head_step_size_multiplier,
            upgd_head_bias_step_size_multiplier=(
                cfg.upgd_head_bias_step_size_multiplier
            ),
            upgd_head_loss_pressure_gate_ratio=(
                cfg.upgd_head_loss_pressure_gate_ratio
            ),
            upgd_head_loss_pressure_multiplier=(
                cfg.upgd_head_loss_pressure_multiplier
            ),
            upgd_head_loss_pressure_warmup_steps=(
                cfg.upgd_head_loss_pressure_warmup_steps
            ),
            upgd_head_repetition_multiplier=(
                cfg.upgd_head_repetition_multiplier
            ),
            upgd_head_repetition_decay=cfg.upgd_head_repetition_decay,
            upgd_head_repetition_delta_threshold=(
                cfg.upgd_head_repetition_delta_threshold
            ),
            upgd_head_repetition_pressure_threshold=(
                cfg.upgd_head_repetition_pressure_threshold
            ),
            upgd_head_repetition_warmup_steps=(
                cfg.upgd_head_repetition_warmup_steps
            ),
            slots_per_class=cfg.slots_per_class,
            memory_update_rate=cfg.memory_update_rate,
            initial_novelty_threshold=cfg.initial_novelty_threshold,
            memory_bandwidth=cfg.memory_bandwidth,
            initial_memory_logit=cfg.initial_memory_logit,
            memory_logit_step_size=cfg.memory_logit_step_size,
            confidence_logit_scale=cfg.confidence_logit_scale,
            reliability_logit_scale=cfg.reliability_logit_scale,
            reliability_decay=cfg.reliability_decay,
            target_trace_blend_scale=cfg.target_trace_blend_scale,
            target_trace_pressure_threshold=cfg.target_trace_pressure_threshold,
            novelty_adaptation_rate=cfg.novelty_adaptation_rate,
            target_allocation_rate=cfg.target_allocation_rate,
        )
    )


def make_step2_temporal_context(
    config: Step2TemporalContextConfig | None = None,
) -> TemporalContextFeaturizer:
    """Create the packaged causal phase-product context featurizer."""
    cfg = config or Step2TemporalContextConfig()
    return TemporalContextFeaturizer(
        TemporalContextConfig(
            input_dim=cfg.feature_dim,
            include_raw=True,
            include_ema=False,
            include_delta=False,
            include_phase_products=True,
            ema_decay=0.96,
            periods=cfg.periods,
        )
    )


def make_step2_temporal_learner(
    config: Step2TemporalContextConfig | None = None,
) -> UPGDLearner:
    """Create UPGD configured for temporal-context features."""
    cfg = config or Step2TemporalContextConfig()
    return UPGDLearner.step2_default(
        n_heads=cfg.n_heads,
        hidden_sizes=cfg.hidden_sizes,
        step_size=cfg.step_size,
    )


def make_step2_stream(
    config: Step2KernelConfig | None = None,
) -> OutOfClassPolynomialStream | FrequencyMismatchStream | CompositionalStream:
    """Construct a representative Step 2 stream for integration testing."""
    cfg = config or Step2KernelConfig()
    if cfg.stream == "polynomial":
        return OutOfClassPolynomialStream(
            feature_dim=cfg.feature_dim,
            n_tasks=cfg.n_heads,
            context_length=cfg.context_length,
            noise_std=cfg.noise_std,
        )
    if cfg.stream == "frequency":
        return FrequencyMismatchStream(
            feature_dim=cfg.feature_dim,
            n_tasks=cfg.n_heads,
            context_length=cfg.context_length,
            noise_std=cfg.noise_std,
        )
    if cfg.stream == "compositional":
        return CompositionalStream(
            feature_dim=cfg.feature_dim,
            n_tasks=cfg.n_heads,
            context_length=cfg.context_length,
            noise_std=cfg.noise_std,
        )
    msg = f"unknown Step 2 stream {cfg.stream!r}"
    raise ValueError(msg)


def collect_step2_arrays(
    stream: Any,
    *,
    steps: int,
    key: Array,
) -> tuple[Array, Array]:
    """Collect a small Step 2 stream into observation/target arrays.

    This helper is for smoke tests and downstream integration probes.  Canonical
    experiments use their dedicated runners so they can capture full metadata,
    baselines, and paired seed statistics.
    """
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    state = stream.init(key)
    observations = []
    targets = []
    for idx in range(steps):
        timestep, state = stream.step(state, jnp.array(idx))
        observations.append(timestep.observation)
        targets.append(timestep.target)
    return jnp.stack(observations), jnp.stack(targets)


def run_step2_smoke(
    config: Step2KernelConfig | None = None,
    *,
    steps: int = 128,
    seed: int = 0,
    final_window: int = 32,
) -> Step2SmokeResult:
    """Run a tiny deterministic Step 2 integration probe.

    The smoke probe verifies initialization, vector-target updates, finite
    utility/perturbation metrics, and config serialization.  It is not a
    canonical MLP comparison.
    """
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    seed = require_jax_seed(seed, name="seed")
    final_window = _require_int("final_window", final_window, minimum=1, maximum=steps)
    cfg = config or Step2KernelConfig()
    learner = make_step2_learner(cfg)
    stream = make_step2_stream(cfg)
    data_key, learner_key = jr.split(jr.key(seed))
    observations, targets = collect_step2_arrays(stream, steps=steps, key=data_key)
    state = learner.init(cfg.feature_dim, learner_key)
    result = run_upgd_arrays(learner, state, observations, targets)
    result.metrics.block_until_ready()
    window = result.metrics[-final_window:, 0]
    final_window_mse = float(jnp.mean(window))
    return Step2SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        final_window_mse=final_window_mse,
        metrics_shape=tuple(int(dim) for dim in result.metrics.shape),
        finite=bool(jnp.all(jnp.isfinite(result.metrics))),
        learner_config=learner.to_config(),
    )


def run_step2_associative_smoke(
    config: Step2AssociativeConfig | None = None,
    *,
    steps: int = 128,
    seed: int = 0,
    window: int = 32,
) -> Step2AssociativeSmokeResult:
    """Run a deterministic associative-memory integration probe.

    This is a package-quality smoke test for the sequence-memory path, not a
    replacement for the external sparse key-value recall protocols, which are
    not shipped in this fork. It repeats a small set of contexts so a healthy
    associative table should lower NLL over time.
    """
    cfg = config or Step2AssociativeConfig()
    steps = _require_int("steps", steps, minimum=2, maximum=_INT32_MAX)
    seed = require_jax_seed(seed, name="seed")
    window = _require_int("window", window, minimum=1, maximum=steps // 2)
    pattern_count = min(8, max(2, steps // 8))
    key = jr.key(seed)
    patterns = jr.randint(
        key,
        (pattern_count, cfg.block_size),
        minval=0,
        maxval=cfg.vocab_size,
        dtype=jnp.int32,
    )
    pattern_ids = jnp.arange(steps, dtype=jnp.int32) % pattern_count
    contexts = patterns[pattern_ids]
    labels_by_pattern = (
        patterns[:, -1] + 3 * patterns[:, -2] + patterns[:, 0]
    ) % cfg.vocab_size
    labels = labels_by_pattern[pattern_ids].astype(jnp.int32)
    learner = make_step2_associative_learner(cfg)
    state = learner.init()
    result = run_associative_memory_arrays(learner, state, contexts, labels)
    result.metrics.block_until_ready()
    losses = result.metrics[:, 0]
    initial_window_nll = float(jnp.mean(losses[:window]))
    final_window_nll = float(jnp.mean(losses[-window:]))
    return Step2AssociativeSmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        initial_window_nll=initial_window_nll,
        final_window_nll=final_window_nll,
        metrics_shape=tuple(int(dim) for dim in result.metrics.shape),
        finite=bool(
            jnp.all(jnp.isfinite(result.metrics))
            & jnp.all(jnp.isfinite(result.predictions))
        ),
        learner_config=learner.to_config(),
    )
