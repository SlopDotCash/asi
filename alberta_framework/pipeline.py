# mypy: disable-error-code="attr-defined,call-arg,no-any-return,unused-ignore"
"""Integrated Step 2 featurization, Step 3 prediction, and Step 4 control.

The pipeline composes the existing packaged pieces conservatively:

1. Step 1 optimizer components are reused by later learners; this pipeline
   does not run the Step 1 prediction benchmark.
2. Step 2 supplies feature augmentation in one of four modes: the lightweight
   temporal-context featurizer, the packaged nonlinear UPGD learner (whose
   penultimate hidden activations become the feature vector for downstream
   Step 3 and Step 4 learners), the associative-memory learner (whose
   next-token probability vector becomes the features), or raw identity
   passthrough.
3. Step 3 learns GVF/Horde predictions on those features. Cumulants are
   either supplied through a caller-provided callable or fall back to the
   observation-channel cumulant function used by the legacy smoke API.
4. Step 4 learns control on the same features, either as discrete SARSA
   (default) or as a Horde-backed actor-critic (``HordeActorCriticAgent``).

The API is intentionally narrow and transition-oriented.  It is suitable for
daemon smoke tests, downstream integration probes, and checkpointed online
state, while research-scale experiments should continue to use their dedicated
runners.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._scan_resources import ScanBudget, require_scan_steps
from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.associative_memory import (
    AssociativeFeatureFamily,
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    AssociativeMemoryState,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    HordeActorCriticState,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.optimizers import ObGDBounding
from alberta_framework.core.sarsa import SARSAState
from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
    TemporalContextState,
)
from alberta_framework.core.update_safety import floating_tree_is_finite
from alberta_framework.core.upgd import UPGDLearner, UPGDState
from alberta_framework.steps._float32_validation import (
    canonical_float32_storage,
    finite_real_and_float32,
)
from alberta_framework.steps.step3 import (
    Step3HordeConfig,
    init_step3_state,
    make_step3_horde,
    step3_predict,
)
from alberta_framework.steps.step4 import (
    Step4SARSAConfig,
    init_step4_state,
    make_step4_sarsa_agent,
    step4_update,
)

Step2Mode = Literal["temporal_context", "upgd", "associative", "identity"]
Step2UPGDPreset = Literal["default", "strict_digit_readout"]
Step2UPGDReadoutMode = Literal[
    "linear_mse",
    "softmax_ce",
    "adaptive_simplex",
    "factorized_simplex",
    "adaptive_factorized_simplex",
    "two_timescale_simplex",
]
ControlMode = Literal["sarsa", "horde_ac"]

CumulantFn = Callable[[Array, Array, Array], Array]
"""Caller-supplied cumulant function.

Signature: ``(observation, reward, terminated) -> Array(n_demons,)``.
"""

_INT32_MAX: int = 2**31 - 1
_MAX_CONFIG_SEQUENCE_LENGTH: int = 4096
# Public last-fit in tests is run_arrays length 2 and smoke steps=8.
# Origin scanned INT32-legal array lengths — hang, not leftover INT32 math.
_PIPELINE_SCAN_BUDGET = ScanBudget("Step 1-4 pipeline", maximum_steps=10_000)
_PIPELINE_SCAN_MAX: int = _PIPELINE_SCAN_BUDGET.maximum_steps

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

def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_exact_record(value: object, expected: type, *, name: str) -> None:
    if type(value) is not expected:
        raise ValueError(f"{name} must be an exact {expected.__name__}")


def _require_payload(
    value: object,
    *,
    name: str,
    allowed: frozenset[str],
    required: frozenset[str] | None = None,
) -> dict[str, object]:
    """Copy one JSON object without invoking mapping-subclass hooks."""
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact dictionary")
    payload = cast(dict[object, object], value)
    if any(type(key) is not str for key in payload):
        raise ValueError(f"{name} keys must be exact strings")
    keys = cast(set[str], set(payload))
    required_keys = allowed if required is None else required
    if not required_keys <= keys or not keys <= allowed:
        raise ValueError(f"{name} fields do not match the schema")
    return cast(dict[str, object], dict(payload))


def _require_bounded_tuple(name: str, value: object, *, nonempty: bool) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    values = cast(tuple[object, ...], value)
    if nonempty and not values:
        raise ValueError(f"{name} must contain at least one element")
    if len(values) > _MAX_CONFIG_SEQUENCE_LENGTH:
        raise ValueError(f"{name} must contain at most {_MAX_CONFIG_SEQUENCE_LENGTH} elements")
    return values


def _require_float32_resource(name: str, scalar_count: int) -> None:
    if scalar_count < 0 or scalar_count > _INT32_MAX or 4 * scalar_count > _INT32_MAX:
        raise ValueError(f"derived {name} float32 resources exceed signed-int32 bounds")


def _trusted_array_metadata(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Require trusted static array metadata without touching hostile objects."""
    actual_type = type(value)
    if not (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise TypeError(f"{name} must be a trusted array")
    try:
        actual_shape = tuple(value.shape)  # type: ignore[attr-defined]
        actual_dtype = jnp.dtype(value.dtype)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose trusted shape and dtype metadata") from error
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if actual_dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
    return cast(Array, value)


def _require_typed_key(name: str, value: object) -> Array:
    actual_type = type(value)
    if not (issubclass(actual_type, jax.Array) or issubclass(actual_type, jax.core.Tracer)):
        raise TypeError(f"{name} must be a scalar typed JAX PRNG key")
    try:
        shape = tuple(value.shape)  # type: ignore[attr-defined]
        words = jr.key_data(cast(Array, value))
        implementation = str(jr.key_impl(cast(Array, value)))
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar typed JAX PRNG key") from error
    if shape != () or words.shape != (2,) or words.dtype != jnp.uint32:
        raise TypeError(f"{name} must be a scalar typed JAX PRNG key")
    if implementation != "threefry2x32":
        raise ValueError(f"{name} must use Threefry2x32")
    return cast(Array, value)


def _require_bool(name: object, value: object) -> bool:
    """Require an actual builtin bool (``__class__`` spoofing is ignored)."""
    _require_exact_str("name", name)
    host_name = cast(str, name)
    if type(value) is not bool:
        raise ValueError(f"{host_name} must be a bool")
    return value  # type: ignore[return-value]


def _require_str_choice(
    name: object, value: object, choices: tuple[str, ...]
) -> str:
    """Require an actual builtin str drawn from ``choices``."""
    _require_exact_str("name", name)
    host_name = cast(str, name)
    if type(value) is not str or value not in choices:
        raise ValueError(f"unknown {host_name}")
    return value  # type: ignore[return-value]


def _require_real(name: object, value: object) -> float:
    _require_exact_str("name", name)
    host_name = cast(str, name)
    real, _, _, narrowed = finite_real_and_float32(host_name, value)
    return canonical_float32_storage(real, narrowed)


def _require_unit_interval(name: object, value: object) -> float:
    _require_exact_str("name", name)
    host_name = cast(str, name)
    real, numerator, denominator, narrowed = finite_real_and_float32(
        host_name, value
    )
    if (
        real < 0.0
        or not real <= 1.0
        or numerator < 0
        or numerator > denominator
        or narrowed < 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{host_name} must be in [0, 1]")
    return canonical_float32_storage(real, narrowed)


def _require_half_open_unit_interval(name: object, value: object) -> float:
    _require_exact_str("name", name)
    host_name = cast(str, name)
    real, numerator, denominator, narrowed = finite_real_and_float32(
        host_name, value
    )
    if (
        real <= 0.0
        or not real <= 1.0
        or numerator <= 0
        or numerator > denominator
        or narrowed <= 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{host_name} must be in (0, 1]")
    return canonical_float32_storage(real, narrowed)


def _require_half_open_zero_one_interval(name: object, value: object) -> float:
    _require_exact_str("name", name)
    host_name = cast(str, name)
    real, numerator, denominator, narrowed = finite_real_and_float32(
        host_name, value
    )
    if (
        real < 0.0
        or not real < 1.0
        or numerator < 0
        or numerator >= denominator
        or narrowed < 0.0
        or not narrowed < 1.0
    ):
        raise ValueError(f"{host_name} must be in [0, 1)")
    return canonical_float32_storage(real, narrowed)


def _require_nonnegative_real(name: object, value: object) -> float:
    _require_exact_str("name", name)
    host_name = cast(str, name)
    real, numerator, _, narrowed = finite_real_and_float32(host_name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{host_name} must be non-negative")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: object, value: object) -> float:
    _require_exact_str("name", name)
    host_name = cast(str, name)
    real, numerator, _, narrowed = finite_real_and_float32(host_name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{host_name} must be positive")
    return canonical_float32_storage(real, narrowed)


def _require_int(
    name: object,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    _require_exact_str("name", name)
    host_name = cast(str, name)
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{host_name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{host_name} must be positive")
        if minimum == 0:
            raise ValueError(f"{host_name} must be non-negative")
        raise ValueError(f"{host_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{host_name} must be <= {maximum}")
    return number


def _integer_associative_input(
    name: str,
    value: object,
    *,
    expected_shape: tuple[int, ...],
) -> Array:
    """Narrow associative indices without laundering floats or booleans.

    ``AssociativeMemoryLearner`` documents integer contexts and labels and
    validates that contract itself. Narrowing a caller's array to ``int32``
    before handing it over would defeat that validator and silently truncate
    an invalid input into a valid-looking one, so reject it here instead.
    """
    actual_type = type(value)
    trusted = (
        issubclass(actual_type, jax.core.Tracer)
        or actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
    )
    if not trusted:
        raise TypeError(f"{name} must be a trusted array")
    try:
        shape = tuple(value.shape)  # type: ignore[attr-defined]
        dtype = np.dtype(value.dtype)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} must expose trusted shape and dtype metadata") from error
    if shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    bounds = np.iinfo(cast(Any, dtype))
    int32_bounds = np.iinfo(np.int32)
    if bounds.min < int32_bounds.min or bounds.max > int32_bounds.max:
        raise ValueError(f"{name} integer dtype must be wholly representable as int32")
    return jnp.asarray(value, dtype=jnp.int32)


@dataclass(frozen=True)
class Step2FeatureConfig:
    """Config for the lightweight temporal-context Step 2 layer.

    This is the historical "raw + EMA + delta + phase products" featurizer
    retained for back-compatibility. New deployments should consider
    :class:`Step2UPGDConfig` for the packaged nonlinear Step 2 path.
    """

    observation_dim: int = 4
    include_raw: bool = True
    include_ema: bool = True
    include_delta: bool = True
    include_phase_products: bool = False
    ema_decay: float = 0.95
    periods: tuple[float, ...] = (32.0, 64.0)

    def __post_init__(self) -> None:
        """Validate observation and feature settings."""
        _require_exact_record(self, Step2FeatureConfig, name="config")
        observation_dim = _require_int(
            "observation_dim", self.observation_dim, minimum=1, maximum=_INT32_MAX
        )
        include_raw = _require_bool("include_raw", self.include_raw)
        include_ema = _require_bool("include_ema", self.include_ema)
        include_delta = _require_bool("include_delta", self.include_delta)
        include_phase_products = _require_bool(
            "include_phase_products",
            self.include_phase_products,
        )
        if not (include_raw or include_ema or include_delta):
            msg = "at least one of include_raw/include_ema/include_delta is required"
            raise ValueError(msg)
        ema_decay = _require_half_open_zero_one_interval("ema_decay", self.ema_decay)
        periods = _require_bounded_tuple("periods", self.periods, nonempty=False)
        canonical_periods = tuple(
            _require_positive_real("period", p) for p in periods
        )
        copies = int(include_raw) + int(include_ema) + int(include_delta)
        phase_dim = 2 * len(canonical_periods)
        output_dim = copies * observation_dim + phase_dim
        if include_phase_products:
            output_dim += phase_dim * observation_dim
        _require_float32_resource("temporal-context output", output_dim)
        _require_float32_resource("temporal-context state", observation_dim + 1)
        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "include_raw", include_raw)
        object.__setattr__(self, "include_ema", include_ema)
        object.__setattr__(self, "include_delta", include_delta)
        object.__setattr__(self, "include_phase_products", include_phase_products)
        object.__setattr__(self, "ema_decay", ema_decay)
        object.__setattr__(self, "periods", canonical_periods)

    @classmethod
    def identity(cls, observation_dim: int) -> Step2FeatureConfig:
        """Return a raw-observation feature config."""
        return cls(
            observation_dim=observation_dim,
            include_raw=True,
            include_ema=False,
            include_delta=False,
            periods=(),
        )

    def to_temporal_context_config(self) -> TemporalContextConfig:
        """Return the core Step 2 featurizer config."""
        return TemporalContextConfig(
            input_dim=self.observation_dim,
            include_raw=self.include_raw,
            include_ema=self.include_ema,
            include_delta=self.include_delta,
            include_phase_products=self.include_phase_products,
            ema_decay=self.ema_decay,
            periods=self.periods,
        )

    def output_dim(self) -> int:
        """Return the Step 2 feature dimensionality."""
        return self.to_temporal_context_config().output_dim()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["periods"] = list(self.periods)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2FeatureConfig:
        """Reconstruct from :meth:`to_dict` output."""
        fields = frozenset(cls.__dataclass_fields__)
        config = _require_payload(payload, name=cls.__name__, allowed=fields)
        raw_periods = config["periods"]
        if type(raw_periods) is not list:
            raise ValueError("periods must be an exact list")
        if len(cast(list[object], raw_periods)) > _MAX_CONFIG_SEQUENCE_LENGTH:
            raise ValueError(f"periods must contain at most {_MAX_CONFIG_SEQUENCE_LENGTH} elements")
        config["periods"] = tuple(cast(list[object], raw_periods))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2UPGDConfig:
    """Config for the packaged UPGD-backed Step 2 featurizer.

    The UPGD learner's penultimate hidden activations are exposed as the
    feature vector for downstream Step 3 and Step 4 learners. The number of
    UPGD heads is configurable; supervised targets may optionally be passed
    through :meth:`AlbertaPipeline.update` to drive UPGD learning. When no
    targets are supplied, UPGD operates as a representation extractor whose
    weights are unchanged and the hidden activations are propagated as-is.
    """

    observation_dim: int = 4
    n_heads: int = 1
    hidden_sizes: tuple[int, ...] = (32,)
    step_size: float = 0.03
    sparsity: float = 0.5
    use_layer_norm: bool = True
    learner_preset: Step2UPGDPreset = "default"
    loss_normalization: Literal["target_structure", "target_density"] = (
        "target_structure"
    )
    readout_mode: Step2UPGDReadoutMode = "linear_mse"

    def __post_init__(self) -> None:
        """Validate configuration."""
        _require_exact_record(self, Step2UPGDConfig, name="config")
        observation_dim = _require_int(
            "observation_dim", self.observation_dim, minimum=1, maximum=_INT32_MAX
        )
        n_heads = _require_int("n_heads", self.n_heads, minimum=1, maximum=_INT32_MAX)
        if type(self.hidden_sizes) is not tuple or not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one positive size")
        hidden_sizes = _require_bounded_tuple("hidden_sizes", self.hidden_sizes, nonempty=True)
        canonical_hidden = tuple(
            _require_int("hidden_sizes element", size, minimum=1, maximum=_INT32_MAX)
            for size in hidden_sizes
        )
        layer_sizes = (observation_dim, *canonical_hidden)
        parameter_scalars = sum(
            left * right + right for left, right in zip(layer_sizes, layer_sizes[1:])
        ) + n_heads * (canonical_hidden[-1] + 1)
        _require_float32_resource("UPGD parameter", parameter_scalars)
        step_size = _require_nonnegative_real("step_size", self.step_size)
        sparsity = _require_unit_interval("sparsity", self.sparsity)
        use_layer_norm = _require_bool("use_layer_norm", self.use_layer_norm)
        learner_preset = _require_str_choice(
            "learner_preset",
            self.learner_preset,
            ("default", "strict_digit_readout"),
        )
        loss_normalization = _require_str_choice(
            "loss_normalization",
            self.loss_normalization,
            ("target_structure", "target_density"),
        )
        readout_mode = _require_str_choice(
            "readout_mode",
            self.readout_mode,
            (
                "linear_mse",
                "softmax_ce",
                "adaptive_simplex",
                "factorized_simplex",
                "adaptive_factorized_simplex",
                "two_timescale_simplex",
            ),
        )
        if learner_preset == "strict_digit_readout" and (
            loss_normalization != "target_structure"
            or readout_mode != "two_timescale_simplex"
        ):
            msg = (
                "strict_digit_readout preset requires "
                "loss_normalization='target_structure' and "
                "readout_mode='two_timescale_simplex'"
            )
            raise ValueError(msg)
        if learner_preset == "strict_digit_readout" and (
            sparsity != 0.5 or not use_layer_norm
        ):
            msg = (
                "strict_digit_readout preset owns sparsity/use_layer_norm; "
                "use sparsity=0.5 and use_layer_norm=True"
            )
            raise ValueError(msg)
        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "n_heads", n_heads)
        object.__setattr__(self, "hidden_sizes", canonical_hidden)
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "sparsity", sparsity)
        object.__setattr__(self, "learner_preset", learner_preset)
        object.__setattr__(self, "loss_normalization", loss_normalization)
        object.__setattr__(self, "readout_mode", readout_mode)

    @classmethod
    def strict_digit_readout(
        cls,
        *,
        observation_dim: int = 64,
        n_heads: int = 10,
        hidden_sizes: tuple[int, ...] = (64, 64),
        step_size: float = 0.018,
    ) -> Step2UPGDConfig:
        """Return the packaged strict digit/readout Step 2 config."""
        return cls(
            observation_dim=observation_dim,
            n_heads=n_heads,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            learner_preset="strict_digit_readout",
            readout_mode="two_timescale_simplex",
        )

    def output_dim(self) -> int:
        """Penultimate-layer dimensionality used as features."""
        return self.hidden_sizes[-1]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2UPGDConfig:
        """Reconstruct from :meth:`to_dict` output."""
        fields = frozenset(cls.__dataclass_fields__)
        config = _require_payload(payload, name=cls.__name__, allowed=fields)
        raw_hidden = config["hidden_sizes"]
        if type(raw_hidden) is not list:
            raise ValueError("hidden_sizes must be an exact list")
        if len(cast(list[object], raw_hidden)) > _MAX_CONFIG_SEQUENCE_LENGTH:
            raise ValueError(
                f"hidden_sizes must contain at most {_MAX_CONFIG_SEQUENCE_LENGTH} elements"
            )
        config["hidden_sizes"] = tuple(cast(list[object], raw_hidden))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2AssociativePipelineConfig:
    """Config for associative Step 2 features in the end-to-end pipeline."""

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

    def __post_init__(self) -> None:
        """Validate integer context settings."""
        _require_exact_record(self, Step2AssociativePipelineConfig, name="config")
        vocab_size = _require_int("vocab_size", self.vocab_size, minimum=2, maximum=_INT32_MAX)
        block_size = _require_int("block_size", self.block_size, minimum=1, maximum=_INT32_MAX)
        suffix_length = _require_int(
            "suffix_length", self.suffix_length, minimum=2, maximum=block_size
        )
        max_features = _require_int(
            "max_features", self.max_features, minimum=1, maximum=_INT32_MAX
        )
        write_lr = _require_positive_real("write_lr", self.write_lr)
        retention = _require_unit_interval("retention", self.retention)
        utility_lr = _require_nonnegative_real("utility_lr", self.utility_lr)
        utility_decay = _require_unit_interval("utility_decay", self.utility_decay)
        min_weight = _require_positive_real("min_weight", self.min_weight)
        max_weight = _require_positive_real("max_weight", self.max_weight)
        if max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight")
        logit_scale = _require_positive_real("logit_scale", self.logit_scale)
        feature_family = _require_str_choice(
            "feature_family",
            self.feature_family,
            ("position_token", "suffix_pair", "token_suffix_pair"),
        )
        _require_bool("normalize_by_weight", self.normalize_by_weight)
        _require_bool("adaptive_feature_family", self.adaptive_feature_family)
        _require_bool("adaptive_window", self.adaptive_window)
        _require_bool("adaptive_budget", self.adaptive_budget)
        scope_lr = _require_nonnegative_real("scope_lr", self.scope_lr)
        budget_lr = _require_nonnegative_real("budget_lr", self.budget_lr)
        initial_budget_fraction = _require_half_open_unit_interval(
            "initial_budget_fraction", self.initial_budget_fraction
        )
        min_effective_budget = _require_int(
            "min_effective_budget", self.min_effective_budget, minimum=1, maximum=max_features
        )
        scope_logit_clip = _require_positive_real("scope_logit_clip", self.scope_logit_clip)
        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "suffix_length", suffix_length)
        object.__setattr__(self, "max_features", max_features)
        object.__setattr__(self, "write_lr", write_lr)
        object.__setattr__(self, "retention", retention)
        object.__setattr__(self, "utility_lr", utility_lr)
        object.__setattr__(self, "utility_decay", utility_decay)
        object.__setattr__(self, "min_weight", min_weight)
        object.__setattr__(self, "max_weight", max_weight)
        object.__setattr__(self, "logit_scale", logit_scale)
        object.__setattr__(self, "feature_family", feature_family)
        object.__setattr__(self, "scope_lr", scope_lr)
        object.__setattr__(self, "budget_lr", budget_lr)
        object.__setattr__(self, "initial_budget_fraction", initial_budget_fraction)
        object.__setattr__(self, "min_effective_budget", min_effective_budget)
        object.__setattr__(self, "scope_logit_clip", scope_logit_clip)
        # The core record owns the complete retained/transient accounting.
        self.to_core_config()

    def output_dim(self) -> int:
        """Return the associative probability-vector dimensionality."""
        return self.vocab_size

    def to_core_config(self) -> AssociativeMemoryConfig:
        """Return the core associative memory config."""
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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> Step2AssociativePipelineConfig:
        """Reconstruct from :meth:`to_dict` output."""
        fields = frozenset(cls.__dataclass_fields__)
        config = _require_payload(payload, name=cls.__name__, allowed=fields)
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class HordeActorCriticPipelineConfig:
    """Config wrapper for the Horde actor-critic Step 4 control."""

    n_actions: int = 2
    actor_step_size: float = 0.01
    actor_lamda: float = 0.9
    temperature: float = 1.0
    value_head_index: int = 0
    actor_obgd_kappa: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        _require_exact_record(self, HordeActorCriticPipelineConfig, name="config")
        n_actions = _require_int("n_actions", self.n_actions, minimum=1, maximum=_INT32_MAX)
        actor_step_size = _require_nonnegative_real("actor_step_size", self.actor_step_size)
        actor_lamda = _require_unit_interval("actor_lamda", self.actor_lamda)
        temperature = _require_positive_real("temperature", self.temperature)
        value_head_index = _require_int(
            "value_head_index", self.value_head_index, minimum=0, maximum=_INT32_MAX
        )
        actor_obgd_kappa = (
            _require_positive_real("actor_obgd_kappa", self.actor_obgd_kappa)
            if self.actor_obgd_kappa is not None
            else None
        )
        object.__setattr__(self, "n_actions", n_actions)
        object.__setattr__(self, "actor_step_size", actor_step_size)
        object.__setattr__(self, "actor_lamda", actor_lamda)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "value_head_index", value_head_index)
        object.__setattr__(self, "actor_obgd_kappa", actor_obgd_kappa)

    def to_horde_actor_critic_config(self) -> HordeActorCriticConfig:
        """Return the core actor-critic config."""
        return HordeActorCriticConfig(
            n_actions=self.n_actions,
            actor_step_size=self.actor_step_size,
            actor_lamda=self.actor_lamda,
            temperature=self.temperature,
            value_head_index=self.value_head_index,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HordeActorCriticPipelineConfig:
        """Reconstruct from :meth:`to_dict` output."""
        fields = frozenset(cls.__dataclass_fields__)
        config = _require_payload(payload, name=cls.__name__, allowed=fields)
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class AlbertaPipelineConfig:
    """Config for the Step 1-4 production pipeline.

    The ``step2`` and ``control`` fields select which Step 2 featurizer and
    Step 4 control mode the pipeline runs. Defaults preserve the legacy
    behavior (temporal-context features + SARSA control); set ``step2="upgd"``
    or ``control="horde_ac"`` to opt into the integrated Step 2/Step 4
    components.
    """

    features: Step2FeatureConfig = field(default_factory=Step2FeatureConfig)
    upgd: Step2UPGDConfig | None = None
    associative: Step2AssociativePipelineConfig | None = None
    horde: Step3HordeConfig = field(default_factory=Step3HordeConfig)
    control: Step4SARSAConfig = field(default_factory=Step4SARSAConfig)
    horde_ac: HordeActorCriticPipelineConfig | None = None
    step2: Step2Mode = "temporal_context"
    control_mode: ControlMode = "sarsa"

    def __post_init__(self) -> None:
        """Validate combinations of step2/control and required sub-configs."""
        _require_exact_record(self, AlbertaPipelineConfig, name="config")
        _require_exact_record(self.features, Step2FeatureConfig, name="features")
        _require_exact_record(self.horde, Step3HordeConfig, name="horde")
        _require_exact_record(self.control, Step4SARSAConfig, name="control")
        if self.upgd is not None:
            _require_exact_record(self.upgd, Step2UPGDConfig, name="upgd")
        if self.associative is not None:
            _require_exact_record(
                self.associative,
                Step2AssociativePipelineConfig,
                name="associative",
            )
        if self.horde_ac is not None:
            _require_exact_record(
                self.horde_ac,
                HordeActorCriticPipelineConfig,
                name="horde_ac",
            )
        _require_str_choice(
            "step2 mode",
            self.step2,
            ("temporal_context", "upgd", "associative", "identity"),
        )
        _require_str_choice("control_mode", self.control_mode, ("sarsa", "horde_ac"))
        if self.step2 == "upgd" and self.upgd is None:
            msg = "upgd config is required when step2='upgd'"
            raise ValueError(msg)
        if self.step2 == "associative" and self.associative is None:
            msg = "associative config is required when step2='associative'"
            raise ValueError(msg)
        if self.control_mode == "horde_ac" and self.horde_ac is None:
            msg = "horde_ac config is required when control_mode='horde_ac'"
            raise ValueError(msg)
        if self.control_mode == "horde_ac":
            ac = cast(HordeActorCriticPipelineConfig, self.horde_ac)
            if ac.value_head_index >= self.horde.n_demons:
                msg = (
                    "horde_ac.value_head_index must reference an existing "
                    f"horde demon (got {ac.value_head_index}, n_demons="
                    f"{self.horde.n_demons})"
                )
                raise ValueError(msg)
        _require_float32_resource("pipeline feature vector", self.feature_dim())

    def feature_dim(self) -> int:
        """Return the feature dimensionality passed to Step 3 and Step 4."""
        if self.step2 == "upgd":
            return cast(Step2UPGDConfig, self.upgd).output_dim()
        if self.step2 == "associative":
            return cast(Step2AssociativePipelineConfig, self.associative).output_dim()
        if self.step2 == "identity":
            return self.features.observation_dim
        return self.features.output_dim()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "features": self.features.to_dict(),
            "upgd": self.upgd.to_dict() if self.upgd is not None else None,
            "associative": (
                self.associative.to_dict() if self.associative is not None else None
            ),
            "horde": self.horde.to_dict(),
            "control": self.control.to_dict(),
            "horde_ac": (
                self.horde_ac.to_dict() if self.horde_ac is not None else None
            ),
            "step2": self.step2,
            "control_mode": self.control_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AlbertaPipelineConfig:
        """Reconstruct from :meth:`to_dict` output."""
        allowed = frozenset(
            {
                "features",
                "upgd",
                "associative",
                "horde",
                "control",
                "horde_ac",
                "step2",
                "control_mode",
            }
        )
        required = frozenset({"features", "horde", "control"})
        config = _require_payload(
            payload,
            name=cls.__name__,
            allowed=allowed,
            required=required,
        )
        upgd_payload = config.get("upgd")
        associative_payload = config.get("associative")
        horde_ac_payload = config.get("horde_ac")
        return cls(
            features=Step2FeatureConfig.from_dict(
                cast(dict[str, object], config["features"])
            ),
            upgd=Step2UPGDConfig.from_dict(cast(dict[str, object], upgd_payload))
            if upgd_payload is not None
            else None,
            associative=Step2AssociativePipelineConfig.from_dict(
                cast(dict[str, object], associative_payload)
            )
            if associative_payload is not None
            else None,
            horde=Step3HordeConfig.from_dict(
                cast(dict[str, object], config["horde"])
            ),
            control=Step4SARSAConfig.from_dict(
                cast(dict[str, object], config["control"])
            ),
            horde_ac=HordeActorCriticPipelineConfig.from_dict(
                cast(dict[str, object], horde_ac_payload)
            )
            if horde_ac_payload is not None
            else None,
            step2=cast(Step2Mode, config.get("step2", "temporal_context")),
            control_mode=cast(ControlMode, config.get("control_mode", "sarsa")),
        )


@chex.dataclass(frozen=True)
class AlbertaPipelineState:
    """Checkpoint-friendly immutable state for the Step 1-4 pipeline.

    ``feature_state`` stores the temporal-context state when ``step2`` is
    ``"temporal_context"``; otherwise it is None. ``upgd_state`` stores the
    UPGD learner state when ``step2`` is ``"upgd"``; otherwise it is None.
    ``associative_state`` stores the associative-memory state when ``step2``
    is ``"associative"``; otherwise it is None. ``control_state`` is either
    a SARSA state or a HordeActorCritic state depending on ``control_mode``.
    """

    feature_state: TemporalContextState | None
    upgd_state: UPGDState | None
    associative_state: AssociativeMemoryState | None
    horde_state: MultiHeadMLPState
    control_state: SARSAState | HordeActorCriticState
    last_features: Array
    step_count: Array


@chex.dataclass(frozen=True)
class AlbertaPipelineStepResult:
    """Result from one end-to-end transition update.

    ``q_values`` carries Q-values when ``control_mode == "sarsa"`` and the
    softmax policy when ``control_mode == "horde_ac"``. The ``action`` field
    is the action selected/sampled at the new observation.
    """

    state: AlbertaPipelineState
    features: Array
    horde_predictions: Array
    horde_td_errors: Array
    horde_td_targets: Array
    q_values: Array
    action: Array
    control_td_error: Array
    reward: Array


@chex.dataclass(frozen=True)
class AlbertaPipelineArrayResult:
    """Result from scanning the end-to-end pipeline over arrays."""

    state: AlbertaPipelineState
    features: Array
    horde_predictions: Array
    horde_td_errors: Array
    q_values: Array
    actions: Array
    control_td_errors: Array


@dataclass(frozen=True)
class AlbertaPipelineSmokeResult:
    """Summary returned by :func:`run_pipeline_smoke`."""

    config: AlbertaPipelineConfig
    steps: int
    seed: int
    feature_shape: tuple[int, ...]
    horde_predictions_shape: tuple[int, ...]
    q_values_shape: tuple[int, ...]
    actions_shape: tuple[int, ...]
    finite: bool

    def __post_init__(self) -> None:
        _require_exact_record(self, AlbertaPipelineSmokeResult, name="result")
        _require_exact_record(self.config, AlbertaPipelineConfig, name="config")
        object.__setattr__(
            self, "steps", _require_int("steps", self.steps, minimum=1, maximum=_INT32_MAX)
        )
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        object.__setattr__(self, "finite", _require_bool("finite", self.finite))
        expected = {
            "feature_shape": (self.steps, self.config.feature_dim()),
            "horde_predictions_shape": (self.steps, self.config.horde.n_demons),
            "q_values_shape": (
                self.steps,
                cast(HordeActorCriticPipelineConfig, self.config.horde_ac).n_actions
                if self.config.control_mode == "horde_ac"
                else self.config.control.n_actions,
            ),
            "actions_shape": (self.steps,),
        }
        for name, expected_shape in expected.items():
            raw = getattr(self, name)
            if type(raw) is not tuple or len(raw) != len(expected_shape):
                raise ValueError(f"{name} must equal the derived pipeline shape")
            shape = cast(tuple[object, ...], raw)
            if any(type(dim) is not int for dim in shape) or shape != expected_shape:
                raise ValueError(f"{name} must equal the derived pipeline shape")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["feature_shape"] = list(self.feature_shape)
        payload["horde_predictions_shape"] = list(self.horde_predictions_shape)
        payload["q_values_shape"] = list(self.q_values_shape)
        payload["actions_shape"] = list(self.actions_shape)
        return payload


def observation_channel_cumulant_fn(
    n_demons: int, observation_dim: int
) -> CumulantFn:
    """Return a cumulant function that maps demons to observation channels."""
    n_demons = _require_int("n_demons", n_demons, minimum=1, maximum=_INT32_MAX)
    observation_dim = _require_int(
        "observation_dim", observation_dim, minimum=1, maximum=_INT32_MAX
    )

    indices = jnp.arange(n_demons) % observation_dim

    def cumulant_fn(
        observation: Array, _reward: Array, _terminated: Array
    ) -> Array:
        obs_1d = jnp.atleast_1d(observation)
        return obs_1d[indices]

    return cumulant_fn


class AlbertaPipeline:
    """Composable Step 2 featurization + Step 3 Horde + Step 4 control.

    See :class:`AlbertaPipelineConfig` for selecting between temporal-context
    and UPGD Step 2 featurization, and between SARSA and HordeActorCritic
    Step 4 control. A caller-supplied ``cumulant_fn`` substitutes domain Step 3
    cumulants for the default observation-channel cumulants; passing
    ``cumulant_fn=None`` preserves the legacy smoke behavior for
    back-compatibility.
    """

    def __init__(
        self,
        config: AlbertaPipelineConfig | None = None,
        *,
        cumulant_fn: CumulantFn | None = None,
    ):
        """Construct all pipeline components from ``config``."""
        if config is None:
            self._config = AlbertaPipelineConfig()
        else:
            _require_exact_record(config, AlbertaPipelineConfig, name="config")
            self._config = config
        if cumulant_fn is not None and not callable(cumulant_fn):
            raise ValueError("cumulant_fn must be callable or None")

        if self._config.step2 == "temporal_context":
            self._featurizer: TemporalContextFeaturizer | None = (
                TemporalContextFeaturizer(
                    self._config.features.to_temporal_context_config()
                )
            )
        else:
            self._featurizer = None

        if self._config.step2 == "upgd":
            upgd_cfg = cast(Step2UPGDConfig, self._config.upgd)
            if upgd_cfg.learner_preset == "strict_digit_readout":
                self._upgd: UPGDLearner | None = (
                    UPGDLearner.step2_strict_digit_readout_default(
                        n_heads=upgd_cfg.n_heads,
                        hidden_sizes=upgd_cfg.hidden_sizes,
                        step_size=upgd_cfg.step_size,
                    )
                )
            else:
                self._upgd = UPGDLearner(
                    n_heads=upgd_cfg.n_heads,
                    hidden_sizes=upgd_cfg.hidden_sizes,
                    step_size=upgd_cfg.step_size,
                    bounder=ObGDBounding(kappa=0.5),
                    sparsity=upgd_cfg.sparsity,
                    use_layer_norm=upgd_cfg.use_layer_norm,
                    perturbation_sigma=1e-4,
                    perturbation_noise="rademacher",
                    utility_decay=0.995,
                    perturbation_beta=2.0,
                    perturbation_interval=16,
                    loss_normalization=upgd_cfg.loss_normalization,
                    readout_mode=upgd_cfg.readout_mode,
                    track_unit_utilities=False,
                    track_gradient_history=False,
                )
        else:
            self._upgd = None

        if self._config.step2 == "associative":
            assoc_cfg = cast(Step2AssociativePipelineConfig, self._config.associative)
            self._associative: AssociativeMemoryLearner | None = (
                AssociativeMemoryLearner(assoc_cfg.to_core_config())
            )
        else:
            self._associative = None

        self._horde = make_step3_horde(self._config.horde)

        self._control: HordeActorCriticAgent | Any
        if self._config.control_mode == "horde_ac":
            ac_cfg = cast(HordeActorCriticPipelineConfig, self._config.horde_ac)
            actor_bounder = (
                ObGDBounding(kappa=ac_cfg.actor_obgd_kappa)
                if ac_cfg.actor_obgd_kappa is not None
                else None
            )
            # HordeActorCritic requires the shared-trunk HordeLearner; the
            # mixed/independent routings are unsupported as a critic backend.
            if not isinstance(self._horde, HordeLearner):
                msg = (
                    "control_mode='horde_ac' requires Step 3 routing='shared'; "
                    f"got {type(self._horde).__name__}"
                )
                raise TypeError(msg)
            self._control = HordeActorCriticAgent(
                config=ac_cfg.to_horde_actor_critic_config(),
                critic=self._horde,
                actor_bounder=actor_bounder,
            )
        else:
            self._control = make_step4_sarsa_agent(
                self._config.control,
                prediction_demons=tuple(self._horde.horde_spec.demons),
            )

        observation_dim = self._observation_dim()
        self._cumulant_fn = (
            observation_channel_cumulant_fn(self._config.horde.n_demons, observation_dim)
            if cumulant_fn is None
            else cumulant_fn
        )

    def _observation_dim(self) -> int:
        if self._config.step2 == "upgd":
            return cast(Step2UPGDConfig, self._config.upgd).observation_dim
        if self._config.step2 == "associative":
            return cast(Step2AssociativePipelineConfig, self._config.associative).block_size
        return self._config.features.observation_dim

    def _observation_operand(self, name: str, value: object, *, batched: bool = False) -> Array:
        width = self._observation_dim()
        shape = (-1, width) if batched else (width,)
        if batched:
            actual_type = type(value)
            if not (
                actual_type is np.ndarray
                or issubclass(actual_type, jax.Array)
                or issubclass(actual_type, jax.core.Tracer)
            ):
                raise TypeError(f"{name} must be a trusted array")
            try:
                actual_shape = tuple(value.shape)  # type: ignore[attr-defined]
            except (AttributeError, TypeError) as error:
                raise TypeError(f"{name} must expose trusted shape metadata") from error
            if len(actual_shape) != 2 or actual_shape[1] != width:
                raise ValueError(f"{name} must have shape (steps, {width})")
            shape = actual_shape
        if self._config.step2 == "associative":
            return _integer_associative_input(name, value, expected_shape=shape)
        return _trusted_array_metadata(name, value, shape=shape, dtype=jnp.float32)

    def _state_contract(self, state: object) -> AlbertaPipelineState:
        _require_exact_record(state, AlbertaPipelineState, name="state")
        checked = cast(AlbertaPipelineState, state)
        _trusted_array_metadata(
            "state.last_features",
            checked.last_features,
            shape=(self.feature_dim,),
            dtype=jnp.float32,
        )
        _trusted_array_metadata("state.step_count", checked.step_count, shape=(), dtype=jnp.int32)
        if self._config.step2 == "temporal_context":
            if (
                checked.feature_state is None
                or checked.upgd_state is not None
                or checked.associative_state is not None
            ):
                raise ValueError("state Step 2 variants do not match pipeline config")
        elif self._config.step2 == "upgd":
            if (
                checked.feature_state is not None
                or checked.upgd_state is None
                or checked.associative_state is not None
            ):
                raise ValueError("state Step 2 variants do not match pipeline config")
        elif self._config.step2 == "associative":
            if (
                checked.feature_state is not None
                or checked.upgd_state is not None
                or checked.associative_state is None
            ):
                raise ValueError("state Step 2 variants do not match pipeline config")
        elif any(
            value is not None
            for value in (checked.feature_state, checked.upgd_state, checked.associative_state)
        ):
            raise ValueError("state Step 2 variants do not match pipeline config")
        return checked

    @property
    def config(self) -> AlbertaPipelineConfig:
        """Pipeline configuration."""
        return self._config

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality emitted by Step 2."""
        return self._config.feature_dim()

    @property
    def featurizer(self) -> TemporalContextFeaturizer | None:
        """Underlying temporal-context featurizer if configured."""
        return self._featurizer

    @property
    def upgd(self) -> UPGDLearner | None:
        """Underlying UPGD learner if configured."""
        return self._upgd

    @property
    def associative(self) -> AssociativeMemoryLearner | None:
        """Underlying associative memory learner if configured."""
        return self._associative

    @property
    def horde(self) -> Any:
        """Underlying Step 3 Horde learner."""
        return self._horde

    @property
    def control(self) -> Any:
        """Underlying Step 4 control agent (SARSA or HordeActorCritic)."""
        return self._control

    @property
    def cumulant_fn(self) -> CumulantFn:
        """Cumulant function used by Step 3."""
        return self._cumulant_fn

    def _features_from_observation(
        self,
        feature_state: TemporalContextState | None,
        upgd_state: UPGDState | None,
        associative_state: AssociativeMemoryState | None,
        observation: Array,
    ) -> tuple[
        TemporalContextState | None,
        UPGDState | None,
        AssociativeMemoryState | None,
        Array,
    ]:
        """Produce the Step 2 feature vector for an observation."""
        if self._config.step2 == "temporal_context":
            featurizer = cast(TemporalContextFeaturizer, self._featurizer)
            assert feature_state is not None
            new_feature_state, features = featurizer.step(feature_state, observation)
            return new_feature_state, upgd_state, associative_state, features
        if self._config.step2 == "upgd":
            upgd = cast(UPGDLearner, self._upgd)
            assert upgd_state is not None
            features = upgd._trunk_forward(  # noqa: SLF001
                upgd_state.trunk_params.weights,
                upgd_state.trunk_params.biases,
                observation,
                upgd._leaky_relu_slope,  # noqa: SLF001
                upgd._use_layer_norm,  # noqa: SLF001
            )
            return feature_state, upgd_state, associative_state, features
        if self._config.step2 == "associative":
            associative = cast(AssociativeMemoryLearner, self._associative)
            assert associative_state is not None
            prediction = associative.predict(
                associative_state,
                _integer_associative_input(
                    "observation",
                    observation,
                    expected_shape=(associative.config.block_size,),
                ),
            )
            return feature_state, upgd_state, associative_state, prediction.probabilities
        # identity
        return feature_state, upgd_state, associative_state, observation

    def init(self, key: Array, initial_observation: Array) -> AlbertaPipelineState:
        """Initialize learner state and prime control with the first observation."""
        key = _require_typed_key("key", key)
        initial_observation = self._observation_operand(
            "initial_observation", initial_observation
        )
        upgd_key, horde_key, control_key = jr.split(key, 3)

        feature_state: TemporalContextState | None = None
        upgd_state: UPGDState | None = None
        associative_state: AssociativeMemoryState | None = None
        observation_dim = self._observation_dim()

        if self._config.step2 == "temporal_context":
            featurizer = cast(TemporalContextFeaturizer, self._featurizer)
            feature_state, initial_features = featurizer.step(
                featurizer.init(),
                initial_observation,
            )
        elif self._config.step2 == "upgd":
            upgd = cast(UPGDLearner, self._upgd)
            upgd_state = upgd.init(observation_dim, upgd_key)
            initial_features = upgd._trunk_forward(  # noqa: SLF001
                upgd_state.trunk_params.weights,
                upgd_state.trunk_params.biases,
                initial_observation,
                upgd._leaky_relu_slope,  # noqa: SLF001
                upgd._use_layer_norm,  # noqa: SLF001
            )
        elif self._config.step2 == "associative":
            associative = cast(AssociativeMemoryLearner, self._associative)
            associative_state = associative.init()
            initial_features = associative.predict(
                associative_state,
                _integer_associative_input(
                    "initial_observation",
                    initial_observation,
                    expected_shape=(associative.config.block_size,),
                ),
            ).probabilities
        else:
            initial_features = initial_observation

        horde_state = init_step3_state(
            self._horde,
            feature_dim=self.feature_dim,
            key=horde_key,
        )

        control_state: SARSAState | HordeActorCriticState
        if self._config.control_mode == "horde_ac":
            ac = cast(HordeActorCriticAgent, self._control)
            ac_state = ac.init(self.feature_dim, control_key)
            ac_state, _action, _probs = ac.start(ac_state, initial_features)
            horde_state = ac_state.critic_state
            control_state = ac_state
        else:
            control_state = init_step4_state(
                self._control,
                feature_dim=self.feature_dim,
                key=control_key,
                initial_features=initial_features,
            )

        return AlbertaPipelineState(
            feature_state=feature_state,
            upgd_state=upgd_state,
            associative_state=associative_state,
            horde_state=horde_state,
            control_state=control_state,
            last_features=initial_features,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def predict(self, state: AlbertaPipelineState) -> tuple[Array, Array]:
        """Return Step 3 predictions and Step 4 control outputs.

        For SARSA control, the second element is the per-action Q-value
        vector. For HordeActorCritic control, it is the softmax action
        probability vector.
        """
        state = self._state_contract(state)
        horde_predictions = step3_predict(
            self._horde,
            state.horde_state,
            state.last_features,
        )
        if self._config.control_mode == "horde_ac":
            ac = cast(HordeActorCriticAgent, self._control)
            ac_state = cast(HordeActorCriticState, state.control_state)
            policy = ac.policy(ac_state, state.last_features)
            return horde_predictions, policy
        sarsa_state = cast(SARSAState, state.control_state)
        q_values = self._control.horde.predict(
            sarsa_state.learner_state,
            state.last_features,
        )[: self._config.control.n_actions]
        return horde_predictions, q_values

    def update(
        self,
        state: AlbertaPipelineState,
        observation: Array,
        reward: Array,
        terminated: Array,
        horde_cumulants: Array | None = None,
        upgd_targets: Array | None = None,
        associative_label: Array | None = None,
    ) -> AlbertaPipelineStepResult:
        """Advance every pipeline component by one transition.

        ``state.last_features`` represents the previous observation. The new
        raw ``observation`` is transformed by Step 2, then Step 3 and Step 4
        both update on the resulting transition.

        Args:
            state: Current pipeline state.
            observation: Next raw observation.
            reward: Scalar transition reward.
            terminated: Scalar termination flag (``0.0`` or ``1.0``).
            horde_cumulants: Optional explicit Step 3 cumulants of shape
                ``(n_demons,)``. When omitted, the configured cumulant
                function, or the default observation-channel cumulant function,
                is used.
            upgd_targets: Optional supervised targets of shape ``(n_heads,)``
                that drive UPGD learning when ``step2='upgd'``. NaN entries
                mark inactive heads. When omitted, UPGD weights stay frozen
                and the trunk acts as a pure feature extractor.
            associative_label: Optional integer next-token/class label that
                drives associative-memory writes when ``step2='associative'``.
        """
        state = self._state_contract(state)
        observation = self._observation_operand("observation", observation)
        reward = _trusted_array_metadata("reward", reward, shape=(), dtype=jnp.float32)
        terminated = _trusted_array_metadata(
            "terminated", terminated, shape=(), dtype=jnp.float32
        )
        if upgd_targets is not None and self._config.step2 != "upgd":
            raise ValueError("upgd_targets require step2='upgd'")
        if associative_label is not None and self._config.step2 != "associative":
            raise ValueError("associative_label requires step2='associative'")
        checked_upgd_targets: Array | None = None
        if upgd_targets is not None:
            upgd_cfg = cast(Step2UPGDConfig, self._config.upgd)
            checked_upgd_targets = _trusted_array_metadata(
                "upgd_targets",
                upgd_targets,
                shape=(upgd_cfg.n_heads,),
                dtype=jnp.float32,
            )
        checked_associative_label: Array | None = None
        if associative_label is not None:
            checked_associative_label = _integer_associative_input(
                "associative_label", associative_label, expected_shape=()
            )
        step2_update_applied = jnp.asarray(True, dtype=jnp.bool_)
        (
            new_feature_state,
            new_upgd_state,
            new_associative_state,
            features,
        ) = self._features_from_observation(
            state.feature_state,
            state.upgd_state,
            state.associative_state,
            observation,
        )
        if (
            self._config.step2 == "upgd"
            and upgd_targets is not None
            and new_upgd_state is not None
        ):
            upgd = cast(UPGDLearner, self._upgd)
            assert checked_upgd_targets is not None
            upgd_result = upgd.update(
                new_upgd_state, observation, checked_upgd_targets
            )
            new_upgd_state = upgd_result.state
            step2_update_applied = floating_tree_is_finite(new_upgd_state)
            features = upgd._trunk_forward(  # noqa: SLF001
                new_upgd_state.trunk_params.weights,
                new_upgd_state.trunk_params.biases,
                observation,
                upgd._leaky_relu_slope,  # noqa: SLF001
                upgd._use_layer_norm,  # noqa: SLF001
            )
        if (
            self._config.step2 == "associative"
            and associative_label is not None
            and new_associative_state is not None
        ):
            associative = cast(AssociativeMemoryLearner, self._associative)
            assoc_result = associative.update(
                new_associative_state,
                _integer_associative_input(
                    "observation",
                    observation,
                    expected_shape=(associative.config.block_size,),
                ),
                cast(Array, checked_associative_label),
            )
            new_associative_state = assoc_result.state
            features = assoc_result.predictions
            step2_update_applied = assoc_result.update_applied

        if horde_cumulants is None:
            horde_cumulants = self._cumulant_fn(observation, reward, terminated)
        horde_cumulants = _trusted_array_metadata(
            "horde_cumulants",
            horde_cumulants,
            shape=(self._config.horde.n_demons,),
            dtype=jnp.float32,
        )

        inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward)
            & ((terminated == 0.0) | (terminated == 1.0))
            & jnp.all(jnp.isfinite(horde_cumulants) | jnp.isnan(horde_cumulants))
        )
        if checked_upgd_targets is not None:
            inputs_valid = inputs_valid & jnp.all(
                jnp.isfinite(checked_upgd_targets) | jnp.isnan(checked_upgd_targets)
            )
        if self._config.step2 == "associative":
            associative_cfg = cast(
                Step2AssociativePipelineConfig, self._config.associative
            )
            inputs_valid = inputs_valid & jnp.all(
                (observation >= 0) & (observation < associative_cfg.vocab_size)
            )
            if checked_associative_label is not None:
                inputs_valid = inputs_valid & (
                    (checked_associative_label >= 0)
                    & (checked_associative_label < associative_cfg.vocab_size)
                )

        if self._config.control_mode == "horde_ac":
            ac = cast(HordeActorCriticAgent, self._control)
            ac_state = cast(HordeActorCriticState, state.control_state)
            ac_state = ac_state.replace(critic_state=state.horde_state)
            n_total_demons = self._horde.n_demons
            value_index = cast(
                HordeActorCriticPipelineConfig, self._config.horde_ac
            ).value_head_index
            aux_indices = jnp.array(
                [i for i in range(n_total_demons) if i != value_index],
                dtype=jnp.int32,
            )
            auxiliary_cumulants = horde_cumulants[aux_indices] if aux_indices.size else None
            ac_result = ac.update(
                ac_state,
                reward,
                features,
                auxiliary_cumulants=auxiliary_cumulants,
            )
            new_control_state: SARSAState | HordeActorCriticState = ac_result.state
            q_values_or_policy = ac_result.policy
            action_out = ac_result.action
            control_td_error = ac_result.td_error
            reward_out = jnp.asarray(reward, dtype=jnp.float32)
            # The actor-critic update already updated the critic for us;
            # we override horde_state to keep them in sync.
            new_horde_state = ac_result.critic_result.state
            horde_predictions = ac_result.critic_result.predictions
            horde_td_errors = ac_result.critic_result.td_errors
            horde_td_targets = ac_result.critic_result.td_targets
            components_applied = ac_result.update_applied
        else:
            # A terminal transition must not bootstrap: zero every demon's
            # discount on termination, as the horde_ac path and Step 4 already
            # do, instead of letting gamma > 0 demons look across the reset.
            horde_discounts = self._horde.horde_spec.gammas * (1.0 - terminated)
            horde_result = self._horde.update_with_discounts(
                state.horde_state,
                state.last_features,
                horde_cumulants,
                features,
                horde_discounts,
            )
            sarsa_state = cast(SARSAState, state.control_state)
            control_result = step4_update(
                self._control,
                sarsa_state,
                reward,
                features,
                terminated,
                prediction_cumulants=horde_cumulants,
            )
            new_control_state = control_result.state
            q_values_or_policy = control_result.q_values
            action_out = control_result.action
            control_td_error = control_result.td_error
            reward_out = control_result.reward
            new_horde_state = horde_result.state
            horde_predictions = horde_result.predictions
            horde_td_errors = horde_result.td_errors
            horde_td_targets = horde_result.td_targets
            components_applied = horde_result.update_applied & (control_result.action >= 0)

        proposed_state = AlbertaPipelineState(
            feature_state=new_feature_state,
            upgd_state=new_upgd_state,
            associative_state=new_associative_state,
            horde_state=new_horde_state,
            control_state=new_control_state,
            last_features=features,
            step_count=(
                jnp.minimum(state.step_count, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32))
                + jnp.asarray(1, dtype=jnp.int32)
            ),
        )
        transaction_applied = (
            inputs_valid
            & step2_update_applied
            & components_applied
            & floating_tree_is_finite(proposed_state)
        )
        next_state = jax.lax.cond(
            transaction_applied,
            lambda: proposed_state,
            lambda: state,
        )
        return AlbertaPipelineStepResult(
            state=next_state,
            features=jnp.where(transaction_applied, features, jnp.zeros_like(features)),
            horde_predictions=jnp.where(
                transaction_applied,
                horde_predictions,
                jnp.zeros_like(horde_predictions),
            ),
            horde_td_errors=jnp.where(
                transaction_applied,
                horde_td_errors,
                jnp.zeros_like(horde_td_errors),
            ),
            horde_td_targets=jnp.where(
                transaction_applied,
                horde_td_targets,
                jnp.zeros_like(horde_td_targets),
            ),
            q_values=jnp.where(
                transaction_applied,
                q_values_or_policy,
                jnp.zeros_like(q_values_or_policy),
            ),
            action=jnp.where(
                transaction_applied,
                action_out,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            control_td_error=jnp.where(
                transaction_applied,
                control_td_error,
                jnp.zeros_like(control_td_error),
            ),
            reward=jnp.where(
                transaction_applied,
                reward_out,
                jnp.zeros_like(reward_out),
            ),
        )

    def run_arrays(
        self,
        state: AlbertaPipelineState,
        observations: Array,
        rewards: Array,
        terminated: Array,
        horde_cumulants: Array,
        upgd_targets: Array | None = None,
        associative_labels: Array | None = None,
    ) -> AlbertaPipelineArrayResult:
        """Scan the pipeline over transition arrays.

        ``state`` should be initialized with the observation that precedes the
        first row in ``observations``. ``horde_cumulants`` is required here
        (the per-step callable variant is :meth:`update`); array runs use a
        fully resolved cumulant table for ``jax.lax.scan`` compatibility.
        """
        state = self._state_contract(state)
        observations = self._observation_operand("observations", observations, batched=True)
        try:
            steps = require_scan_steps(
                "observations length", observations.shape[0], _PIPELINE_SCAN_BUDGET
            )
        except ValueError:
            raise ValueError(
                f"observations must contain between 1 and {_PIPELINE_SCAN_MAX} steps"
            ) from None
        rewards = _trusted_array_metadata(
            "rewards", rewards, shape=(steps,), dtype=jnp.float32
        )
        terminated = _trusted_array_metadata(
            "terminated", terminated, shape=(steps,), dtype=jnp.float32
        )
        horde_cumulants = _trusted_array_metadata(
            "horde_cumulants",
            horde_cumulants,
            shape=(steps, self._config.horde.n_demons),
            dtype=jnp.float32,
        )
        if upgd_targets is not None and self._config.step2 != "upgd":
            raise ValueError("upgd_targets require step2='upgd'")
        if associative_labels is not None and self._config.step2 != "associative":
            raise ValueError("associative_labels require step2='associative'")
        if upgd_targets is None:
            upgd_targets_array = jnp.full(
                (steps, self._config.upgd.n_heads if self._config.upgd else 1),
                jnp.nan,
                dtype=jnp.float32,
            )
        else:
            upgd_cfg = cast(Step2UPGDConfig, self._config.upgd)
            upgd_targets_array = _trusted_array_metadata(
                "upgd_targets",
                upgd_targets,
                shape=(steps, upgd_cfg.n_heads),
                dtype=jnp.float32,
            )
        associative_labels_array = (
            _integer_associative_input(
                "associative_labels",
                associative_labels,
                expected_shape=(steps,),
            )
            if associative_labels is not None
            else jnp.zeros((observations.shape[0],), dtype=jnp.int32)
        )
        use_associative_labels = (
            self._config.step2 == "associative" and associative_labels is not None
        )
        n_actions = (
            cast(HordeActorCriticPipelineConfig, self._config.horde_ac).n_actions
            if self._config.control_mode == "horde_ac"
            else self._config.control.n_actions
        )
        input_scalars_per_step = (
            self._observation_dim()
            + 2
            + self._config.horde.n_demons
            + int(upgd_targets_array.shape[1])
            + 1
        )
        output_scalars_per_step = (
            self.feature_dim
            + 2 * self._config.horde.n_demons
            + n_actions
            + 2
        )
        _require_float32_resource(
            "pipeline array input/output",
            steps * (input_scalars_per_step + output_scalars_per_step),
        )

        def step_fn(
            carry: AlbertaPipelineState,
            inputs: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[AlbertaPipelineState, tuple[Array, Array, Array, Array, Array, Array]]:
            (
                obs_t,
                reward_t,
                terminated_t,
                cumulants_t,
                upgd_target_t,
                associative_label_t,
            ) = inputs
            result = self.update(
                carry,
                obs_t,
                reward_t,
                terminated_t,
                cumulants_t,
                upgd_target_t if self._config.step2 == "upgd" else None,
                associative_label_t if use_associative_labels else None,
            )
            return result.state, (
                result.features,
                result.horde_predictions,
                result.horde_td_errors,
                result.q_values,
                result.action,
                result.control_td_error,
            )

        final_state, outputs = jax.lax.scan(
            step_fn,
            state,
            (
                observations,
                rewards,
                terminated,
                horde_cumulants,
                upgd_targets_array,
                associative_labels_array,
            ),
        )
        (
            features,
            horde_predictions,
            horde_td_errors,
            q_values,
            actions,
            control_td_errors,
        ) = outputs
        return AlbertaPipelineArrayResult(
            state=final_state,
            features=features,
            horde_predictions=horde_predictions,
            horde_td_errors=horde_td_errors,
            q_values=q_values,
            actions=actions,
            control_td_errors=control_td_errors,
        )


def make_alberta_pipeline(
    config: AlbertaPipelineConfig | None = None,
    *,
    cumulant_fn: CumulantFn | None = None,
) -> AlbertaPipeline:
    """Create an end-to-end Alberta production pipeline."""
    return AlbertaPipeline(config, cumulant_fn=cumulant_fn)


def run_pipeline_smoke(
    config: AlbertaPipelineConfig | None = None,
    *,
    steps: int = 24,
    seed: int = 0,
) -> AlbertaPipelineSmokeResult:
    """Run a deterministic Step 1-4 pipeline smoke probe."""
    steps = require_scan_steps("steps", steps, _PIPELINE_SCAN_BUDGET)
    seed = require_jax_seed(seed, name="seed")
    if config is None:
        cfg = AlbertaPipelineConfig()
    else:
        _require_exact_record(config, AlbertaPipelineConfig, name="config")
        cfg = config
    pipeline = make_alberta_pipeline(cfg)

    observation_dim = pipeline._observation_dim()  # noqa: SLF001

    data_key, state_key = jr.split(jr.key(seed))
    if cfg.step2 == "associative" and cfg.associative is not None:
        observations = jr.randint(
            data_key,
            (steps + 1, observation_dim),
            minval=0,
            maxval=cfg.associative.vocab_size,
            dtype=jnp.int32,
        )
        rewards = jnp.tanh(observations[1:, 0].astype(jnp.float32))
        associative_labels = (
            observations[1:, -1] + 3 * observations[1:, -2] + observations[1:, 0]
        ) % cfg.associative.vocab_size
    else:
        observations = jr.normal(
            data_key,
            (steps + 1, observation_dim),
            dtype=jnp.float32,
        )
        rewards = jnp.tanh(observations[1:, 0])
        associative_labels = None
    terminated = jnp.zeros(steps, dtype=jnp.float32)
    cumulant_indices = jnp.arange(cfg.horde.n_demons) % observation_dim
    horde_cumulants = observations[1:, cumulant_indices].astype(jnp.float32)

    state = pipeline.init(state_key, observations[0])
    result = pipeline.run_arrays(
        state,
        observations[1:],
        rewards,
        terminated,
        horde_cumulants,
        associative_labels=associative_labels,
    )
    result.q_values.block_until_ready()

    finite_actions = (
        jnp.all(result.actions >= 0)
        & jnp.all(result.actions < cfg.horde_ac.n_actions)
        if cfg.control_mode == "horde_ac" and cfg.horde_ac is not None
        else jnp.all(result.actions >= 0) & jnp.all(result.actions < cfg.control.n_actions)
    )
    finite = bool(
        jnp.all(jnp.isfinite(result.features))
        & jnp.all(jnp.isfinite(result.horde_predictions))
        & jnp.all(jnp.isfinite(result.horde_td_errors))
        & jnp.all(jnp.isfinite(result.q_values))
        & jnp.all(jnp.isfinite(result.control_td_errors))
        & finite_actions
    )
    return AlbertaPipelineSmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        feature_shape=tuple(int(dim) for dim in result.features.shape),
        horde_predictions_shape=tuple(
            int(dim) for dim in result.horde_predictions.shape
        ),
        q_values_shape=tuple(int(dim) for dim in result.q_values.shape),
        actions_shape=tuple(int(dim) for dim in result.actions.shape),
        finite=finite,
    )
