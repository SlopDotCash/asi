"""Production Step 1 kernel.

This module wraps the Step 1 research implementation in a narrow, stable API:

* canonical Alberta Plan Step 1 streams;
* public optimizers only, with no invented ``Auto`` alias;
* online normalizers used in the canonical ablations;
* a small smoke run suitable for integration tests and deployment probes.

This kernel runs a single configuration and is not an evidence generator:
paper-scale Step 1 claims require multi-seed optimizer/normalizer grid sweeps
(see :mod:`alberta_framework.utils.experiments` for the multi-seed machinery).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._float32 import round_real_to_float32
from alberta_framework.core.baseline_optimizers import NADALINE, AdaGain, Adam, RMSprop
from alberta_framework.core.learners import LinearLearner, run_learning_loop
from alberta_framework.core.normalizers import (
    EMANormalizer,
    Normalizer,
    StreamingBatchNormalizer,
    WelfordNormalizer,
)
from alberta_framework.core.optimizers import (
    IDBD,
    LMS,
    Autostep,
    AutostepGTDLambda,
)
from alberta_framework.streams.alberta_plan_step1 import (
    AlbertaPlanStep1Stream,
    XDistShiftStream,
)

Step1OptimizerName = Literal[
    "lms",
    "idbd",
    "autostep",
    "autostep_gtd",
    "adagain",
    "adam",
    "rmsprop",
    "nadaline",
]
Step1NormalizerName = Literal["none", "ema", "welford", "streaming_batch"]
Step1StreamName = Literal["alberta", "xdist_shift"]

_VALID_OPTIMIZERS: frozenset[str] = frozenset(
    {"lms", "idbd", "autostep", "autostep_gtd", "adagain", "adam", "rmsprop", "nadaline"}
)
_VALID_NORMALIZERS: frozenset[str] = frozenset({"none", "ema", "welford", "streaming_batch"})
_VALID_STREAMS: frozenset[str] = frozenset({"alberta", "xdist_shift"})


def _require_real(name: str, value: object) -> tuple[float, float]:
    """Return a JSON scalar and the value consumed by float32 JAX sinks."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        narrowed = round_real_to_float32(value)
    except (FloatingPointError, OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must narrow to a finite float32, got {value!r}") from None
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must narrow to a finite float32, got {value!r}")
    if not isinstance(value, (int, float, np.floating)):
        return narrowed, narrowed
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")

    # Preserve ordinary built-in/NumPy-float serialization when it narrows to
    # exactly the checked sink value.  Extended-precision inputs at a rounding
    # boundary must instead retain the direct float32 value: routing them
    # through binary64 can double-round a finite float32 to infinity.
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        renarrowed = np.asarray(number, dtype=np.float32)
    if not bool(np.array_equal(narrowed, renarrowed)):
        number = float(narrowed)
    return number, float(narrowed)


def _require_unit_interval(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if value < 0.0 or not value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    number, _ = _require_real(name, value)
    return number


def _require_nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    number, _ = _require_real(name, value)
    return number


def _require_positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    number, narrowed = _require_real(name, value)
    if narrowed <= 0.0:
        raise ValueError(f"{name} must remain positive in float32, got {value!r}")
    return number


# Exact trusted integer scalar types, compared by identity in _require_int.
# ``longlong``/``ulonglong`` are listed via their dtype codes because they can
# be distinct types from the fixed-width aliases on some platforms.
_TRUSTED_INT_TYPES: tuple[type, ...] = (
    int,
    *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")),
)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    exclusive_maximum: int | None = None,
) -> int:
    # Identity-only admission: an actual ``int`` subclass can override
    # ``__int__``/``__index__``/``__repr__`` with hostile hooks, so anything
    # that is not an exact trusted builtin/NumPy integer scalar type is
    # rejected before conversion, without interpolating the untrusted value.
    actual_type = type(value)
    if not any(actual_type is trusted_type for trusted_type in _TRUSTED_INT_TYPES):
        raise ValueError(f"{name} must be an integer of an exact trusted type")
    number: int = int(cast(Any, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive, got {value!r}")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if exclusive_maximum is not None and number >= exclusive_maximum:
        raise ValueError(f"{name} must be smaller than int32 max, got {value!r}")
    return number


def _validate_step1_config(config: Step1KernelConfig) -> None:
    feature_dim = _require_int("feature_dim", config.feature_dim, minimum=1)
    num_relevant = _require_int("num_relevant", config.num_relevant, minimum=1)
    if num_relevant > feature_dim:
        raise ValueError(
            f"num_relevant ({num_relevant}) must be <= feature_dim ({feature_dim})"
        )
    if not isinstance(config.optimizer, str) or config.optimizer.lower() not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"unknown Step 1 optimizer {config.optimizer!r}; "
            f"expected one of {sorted(_VALID_OPTIMIZERS)}"
        )
    if (
        not isinstance(config.normalizer, str)
        or config.normalizer.lower() not in _VALID_NORMALIZERS
    ):
        raise ValueError(
            f"unknown Step 1 normalizer {config.normalizer!r}; "
            f"expected one of {sorted(_VALID_NORMALIZERS)}"
        )
    if not isinstance(config.stream, str) or config.stream.lower() not in _VALID_STREAMS:
        raise ValueError(
            f"unknown Step 1 stream {config.stream!r}; "
            f"expected one of {sorted(_VALID_STREAMS)}"
        )
    step_size = _require_nonnegative_real("step_size", config.step_size)
    meta_step_size = _require_nonnegative_real("meta_step_size", config.meta_step_size)
    drift_rate_w = _require_nonnegative_real("drift_rate_w", config.drift_rate_w)
    drift_rate_b = _require_nonnegative_real("drift_rate_b", config.drift_rate_b)
    noise_std = _require_nonnegative_real("noise_std", config.noise_std)
    feature_std = _require_positive_real("feature_std", config.feature_std)
    ema_decay = _require_unit_interval("ema_decay", config.ema_decay)
    streaming_batch_momentum = _require_unit_interval(
        "streaming_batch_momentum",
        config.streaming_batch_momentum,
    )
    object.__setattr__(config, "feature_dim", feature_dim)
    object.__setattr__(config, "num_relevant", num_relevant)
    object.__setattr__(config, "optimizer", config.optimizer.lower())
    object.__setattr__(config, "normalizer", config.normalizer.lower())
    object.__setattr__(config, "stream", config.stream.lower())
    object.__setattr__(config, "step_size", step_size)
    object.__setattr__(config, "meta_step_size", meta_step_size)
    object.__setattr__(config, "drift_rate_w", drift_rate_w)
    object.__setattr__(config, "drift_rate_b", drift_rate_b)
    object.__setattr__(config, "noise_std", noise_std)
    object.__setattr__(config, "feature_std", feature_std)
    object.__setattr__(config, "ema_decay", ema_decay)
    object.__setattr__(config, "streaming_batch_momentum", streaming_batch_momentum)


@dataclass(frozen=True)
class Step1KernelConfig:
    """Config for the production Step 1 kernel.

    The default is deliberately conservative for daemon/integration use:
    Autostep plus EMA normalization on the canonical drifting Alberta stream.
    A single kernel configuration is never a canonical paper claim — those
    require the full optimizer/normalizer grid across multiple seeds.
    """

    feature_dim: int = 20
    num_relevant: int = 5
    optimizer: Step1OptimizerName = "autostep"
    normalizer: Step1NormalizerName = "ema"
    stream: Step1StreamName = "alberta"
    step_size: float = 0.01
    meta_step_size: float = 0.01
    drift_rate_w: float = 0.001
    drift_rate_b: float = 0.001
    noise_std: float = 1.0
    feature_std: float = 1.0
    ema_decay: float = 0.99
    streaming_batch_momentum: float = 0.99

    def __post_init__(self) -> None:
        """Reject invalid hyperparameters and canonicalize scalars."""
        _validate_step1_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step1KernelConfig:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**cast(Any, payload))


@dataclass(frozen=True)
class Step1SmokeResult:
    """Summary returned by :func:`run_step1_smoke`."""

    config: Step1KernelConfig
    steps: int
    seed: int
    final_window_mse: float
    metrics_shape: tuple[int, ...]
    finite: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["metrics_shape"] = list(self.metrics_shape)
        return payload


def make_step1_optimizer(config: Step1KernelConfig) -> Any:
    """Construct a public Step 1 optimizer from ``config``.

    ``Auto (Degris in prep.)`` is intentionally not accepted: no public update
    rule is available in the cited sources, so the production package exposes
    only reproducible optimizers.
    """
    name = config.optimizer.lower()
    if name == "lms":
        return LMS(step_size=config.step_size)
    if name == "idbd":
        return IDBD(
            initial_step_size=config.step_size,
            meta_step_size=config.meta_step_size,
        )
    if name == "autostep":
        return Autostep(
            initial_step_size=config.step_size,
            meta_step_size=config.meta_step_size,
        )
    if name == "autostep_gtd":
        # Autostep-for-GTD(lambda) per Kearney et al. 2019. The supervised
        # mode reduces to standard Autostep, providing a reproducible
        # implementation for Alberta Plan footnote 11 closure.
        return AutostepGTDLambda(
            initial_step_size=config.step_size,
            meta_step_size=config.meta_step_size,
        )
    if name == "adagain":
        return AdaGain(initial_step_size=config.step_size)
    if name == "adam":
        return Adam(step_size=config.step_size)
    if name == "rmsprop":
        return RMSprop(step_size=config.step_size)
    if name == "nadaline":
        return NADALINE(step_size=config.step_size)
    msg = f"unknown Step 1 optimizer {config.optimizer!r}"
    raise ValueError(msg)


def make_step1_normalizer(
    config: Step1KernelConfig,
) -> Normalizer[Any] | None:
    """Construct an online normalizer from ``config``."""
    name = config.normalizer.lower()
    if name == "none":
        return None
    if name == "ema":
        return EMANormalizer(decay=config.ema_decay)
    if name == "welford":
        return WelfordNormalizer()
    if name == "streaming_batch":
        return StreamingBatchNormalizer(momentum=config.streaming_batch_momentum)
    msg = f"unknown Step 1 normalizer {config.normalizer!r}"
    raise ValueError(msg)


def make_step1_stream(
    config: Step1KernelConfig,
) -> AlbertaPlanStep1Stream | XDistShiftStream:
    """Construct the configured Step 1 stream."""
    if config.stream == "alberta":
        return AlbertaPlanStep1Stream(
            feature_dim=config.feature_dim,
            num_relevant=config.num_relevant,
            drift_rate_w=config.drift_rate_w,
            drift_rate_b=config.drift_rate_b,
            noise_std=config.noise_std,
            feature_std=config.feature_std,
        )
    if config.stream == "xdist_shift":
        return XDistShiftStream(
            feature_dim=config.feature_dim,
            num_relevant=config.num_relevant,
            noise_std=config.noise_std,
            noise_in_target=config.noise_std > 0.0,
        )
    msg = f"unknown Step 1 stream {config.stream!r}"
    raise ValueError(msg)


def make_step1_learner(config: Step1KernelConfig | None = None) -> LinearLearner:
    """Create the production Step 1 learner."""
    cfg = config or Step1KernelConfig()
    return LinearLearner(
        optimizer=make_step1_optimizer(cfg),
        normalizer=make_step1_normalizer(cfg),
    )


def run_step1_smoke(
    config: Step1KernelConfig | None = None,
    *,
    steps: int = 256,
    seed: int = 0,
    final_window: int = 64,
) -> Step1SmokeResult:
    """Run a tiny deterministic Step 1 integration probe.

    The smoke probe is intentionally not a scientific claim.  It verifies that
    the production kernel can initialize, compile, update online, and return
    finite metrics.
    """
    steps = _require_int("steps", steps, minimum=1)
    final_window = _require_int("final_window", final_window, minimum=1)
    if final_window > steps:
        raise ValueError(
            f"final_window must be in [1, steps], got {final_window}"
        )
    cfg = config or Step1KernelConfig()
    learner = make_step1_learner(cfg)
    stream = make_step1_stream(cfg)
    loop_result = cast(
        tuple[Any, Array],
        run_learning_loop(
            learner,
            cast(Any, stream),
            num_steps=steps,
            key=jr.key(seed),
        ),
    )
    metrics = loop_result[1]
    metrics.block_until_ready()
    window = metrics[-final_window:, 0]
    final_window_mse = float(jnp.mean(window))
    return Step1SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        final_window_mse=final_window_mse,
        metrics_shape=tuple(int(dim) for dim in metrics.shape),
        finite=bool(jnp.all(jnp.isfinite(metrics))),
    )
