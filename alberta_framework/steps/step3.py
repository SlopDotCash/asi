# mypy: disable-error-code="call-arg"
"""Public Step 3 Horde helpers.

This module packages the stable Step 3 surface for downstream use:

* given-feature GVF prediction through :class:`HordeLearner`;
* a causal array handoff from Step 2 constructed features to Horde inputs;
* a small smoke run for integration tests.

It intentionally does not claim general TD/GVF feature-discovery closure.
Research-scale evidence and open boundaries for Step 3 are tracked in
``docs/status.md``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.horde import (
    HordeLearner,
    HordeLearningResult,
    HordeUpdateResult,
    MixedHorde,
    MixedHordeLearningResult,
    MixedHordeState,
    run_horde_learning_loop,
    run_mixed_horde_learning_loop,
)
from alberta_framework.core.independent_demon_horde import (
    IndependentDemonHorde,
    IndependentDemonHordeLearningResult,
    IndependentDemonHordeState,
    run_independent_horde_learning_loop,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.normalizers import EMANormalizer, Normalizer
from alberta_framework.core.optimizers import ObGDBounding
from alberta_framework.core.types import DemonType, GVFSpec, TraceMode, create_horde_spec
from alberta_framework.steps._float32_validation import (
    canonical_float32_storage,
    finite_real_and_float32,
)
from alberta_framework.steps._smoke_record_validation import require_step_shape

Step3NormalizerName = Literal["none", "ema"]
Step3TraceModeName = Literal["accumulating", "replacing"]
Step3Horde = HordeLearner | IndependentDemonHorde | MixedHorde
Step3HordeState = MultiHeadMLPState | IndependentDemonHordeState | MixedHordeState
Step3LearningResult = (
    HordeLearningResult | IndependentDemonHordeLearningResult | MixedHordeLearningResult
)
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_ACTUAL_INT_TYPES = frozenset(
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
_FLOAT32_MIN_NORMAL = float.fromhex("0x1.0p-126")
_STEP3_CONFIG_FIELDS = frozenset(
    {
        "gammas",
        "lamdas",
        "hidden_sizes",
        "step_size",
        "use_obgd",
        "obgd_kappa",
        "normalizer",
        "sparsity",
        "use_layer_norm",
        "trace_mode",
        "routing",
    }
)
_MAX_SMOKE_CONFIG_NODES = 4096
_MAX_SMOKE_CONFIG_TEXT_BYTES = 64 * 1024


def _bounded_json_clone(value: object, *, path: str, budget: list[int]) -> Any:
    """Copy one bounded exact JSON tree without invoking user hooks."""
    budget[0] += 1
    if budget[0] > _MAX_SMOKE_CONFIG_NODES:
        raise ValueError(f"{path} exceeds the smoke config node limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value).bit_length() > 63:
            raise ValueError(f"{path} integer must fit signed 64-bit magnitude")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} number must be finite")
        return value
    if type(value) is str:
        if (
            len(value) > _MAX_SMOKE_CONFIG_TEXT_BYTES
            or len(value.encode("utf-8")) > _MAX_SMOKE_CONFIG_TEXT_BYTES
        ):
            raise ValueError(f"{path} text exceeds the smoke config byte limit")
        return value
    if type(value) is list:
        if len(value) > _MAX_SMOKE_CONFIG_NODES - budget[0]:
            raise ValueError(f"{path} exceeds the smoke config node limit")
        return [
            _bounded_json_clone(item, path=f"{path}[{index}]", budget=budget)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if len(value) > _MAX_SMOKE_CONFIG_NODES - budget[0]:
            raise ValueError(f"{path} exceeds the smoke config node limit")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} keys must be exact strings")
            cloned_key = _bounded_json_clone(key, path=f"{path} key", budget=budget)
            result[cloned_key] = _bounded_json_clone(
                item,
                path=f"{path}.{key}",
                budget=budget,
            )
        return result
    raise ValueError(f"{path} must contain only exact JSON values")


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_float32_matrix(name: str, value: object) -> Array:
    """Require trusted matrix metadata before any JAX coercion or hooks."""
    actual_type = type(value)
    if not (
        issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be a JAX array")
    array = cast(Array, value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if array.dtype != jnp.float32:
        raise ValueError(f"{name} must have dtype float32")
    return array


def _require_float32_vector(name: str, value: object) -> Array:
    """Require one trusted float32 vector without coercion."""
    actual_type = type(value)
    if not (
        issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be a JAX array")
    array = cast(Array, value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if array.dtype != jnp.float32:
        raise ValueError(f"{name} must have dtype float32")
    return array


def _require_typed_key(name: str, value: object) -> Array:
    """Require one scalar typed JAX PRNG key without legacy laundering."""
    actual_type = type(value)
    if not (
        issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be a typed JAX PRNG key")
    key = cast(Array, value)
    if key.shape != () or not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        key.dtype, jax.dtypes.prng_key
    ):
        raise ValueError(f"{name} must be a scalar typed JAX PRNG key")
    if str(jr.key_impl(key)) != "threefry2x32":
        raise ValueError(f"{name} must use Threefry2x32")
    return key


def _require_handoff_resources(
    *,
    steps: int,
    raw_dim: int,
    constructed_dim: int,
    n_demons: int,
) -> int:
    """Preflight exact derived handoff shapes and newly allocated bytes."""
    feature_dim = raw_dim + constructed_dim
    logical_scalars = steps * (2 * feature_dim + n_demons)
    allocated_bytes = 8 * steps * feature_dim
    if feature_dim > _INT32_MAX:
        raise ValueError("derived Step 3 feature_dim must fit signed int32")
    if logical_scalars > _INT32_MAX:
        raise ValueError("derived Step 3 handoff scalars must fit signed int32")
    if allocated_bytes > _INT32_MAX:
        raise ValueError("derived Step 3 handoff allocation bytes must fit signed int32")
    return feature_dim


def _require_smoke_resources(
    *,
    steps: int,
    raw_dim: int,
    constructed_dim: int,
    n_demons: int,
) -> None:
    """Preflight all arrays materialized by :func:`run_step3_smoke`."""
    _require_handoff_resources(
        steps=steps,
        raw_dim=raw_dim,
        constructed_dim=constructed_dim,
        n_demons=n_demons,
    )
    # raw observations; product columns plus their stack; cumulants; and the
    # concatenated current/next handoff matrices are simultaneously live.
    allocated_bytes = steps * (
        12 * raw_dim + 16 * constructed_dim + 4 * n_demons
    )
    if allocated_bytes > _INT32_MAX:
        raise ValueError("derived Step 3 smoke allocation bytes must fit signed int32")


Step3RoutingName = Literal["shared", "independent", "mixed"]


@dataclass(frozen=True)
class Step3HordeConfig:
    """Config for the public Step 3 given-feature Horde kernel.

    The default is a compact linear Horde with three prediction demons. Hidden
    layers may be enabled, but shared-trunk trace decay remains head-only via
    :class:`HordeLearner`; this helper does not implement nonlinear shared-trunk
    forward-view traces.
    """

    gammas: tuple[float, ...] = (0.0, 0.5, 0.9)
    lamdas: tuple[float, ...] = (0.0, 0.5, 0.8)
    hidden_sizes: tuple[int, ...] = ()
    step_size: float = 0.05
    use_obgd: bool = True
    obgd_kappa: float = 2.0
    normalizer: Step3NormalizerName = "none"
    sparsity: float = 0.0
    use_layer_norm: bool = True
    trace_mode: Step3TraceModeName = "accumulating"
    routing: Step3RoutingName = "shared"

    def __post_init__(self) -> None:
        """Reject illegal Horde scientific scalars and canonicalize reals."""
        _validate_horde_config(self)

    @property
    def n_demons(self) -> int:
        """Number of Step 3 GVF demons."""
        return len(self.gammas)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["gammas"] = list(self.gammas)
        payload["lamdas"] = list(self.lamdas)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step3HordeConfig:
        """Reconstruct from :meth:`to_dict` output."""
        if type(payload) is not dict:
            raise ValueError("Step 3 config must be an actual dict")
        if any(type(key) is not str for key in payload):
            raise ValueError("Step3HordeConfig keys must be exact strings")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("Step 3 config fields do not match its schema")
        for name in ("gammas", "lamdas", "hidden_sizes"):
            if type(payload[name]) is not list:
                raise ValueError(f"serialized {name} must be a JSON array")
        if any(type(value) is not float for value in cast(list[object], payload["gammas"])):
            raise ValueError("serialized gammas values must be JSON numbers")
        if any(type(value) is not float for value in cast(list[object], payload["lamdas"])):
            raise ValueError("serialized lamdas values must be JSON numbers")
        if any(type(value) is not int for value in cast(list[object], payload["hidden_sizes"])):
            raise ValueError("serialized hidden_sizes values must be JSON integers")
        for name in ("step_size", "obgd_kappa", "sparsity"):
            if type(payload[name]) is not float:
                raise ValueError(f"serialized {name} must be a JSON number")
        for name in ("use_obgd", "use_layer_norm"):
            if type(payload[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("normalizer", "trace_mode", "routing"):
            if type(payload[name]) is not str:
                raise ValueError(f"serialized {name} must be a JSON string")
        config = dict(payload)
        config["gammas"] = tuple(cast(list[float], config["gammas"]))
        config["lamdas"] = tuple(cast(list[float], config["lamdas"]))
        config["hidden_sizes"] = tuple(cast(list[int], config["hidden_sizes"]))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step3HandoffArrays:
    """Arrays needed by :func:`run_horde_learning_loop`.

    ``observations[t]`` is ``concat(raw_observations[t], constructed_features[t])``.
    ``next_observations[t]`` is the shifted augmented row for the same
    transition. Callers are responsible for constructing row ``t`` features
    causally, using only information available at time ``t``.
    """

    observations: Array
    cumulants: Array
    next_observations: Array

    def __post_init__(self) -> None:
        """Authenticate exact array identities and cross-field dimensions."""
        observations = _require_float32_matrix("observations", self.observations)
        cumulants = _require_float32_matrix("cumulants", self.cumulants)
        next_observations = _require_float32_matrix(
            "next_observations", self.next_observations
        )
        steps, feature_dim = observations.shape
        if steps < 1 or feature_dim < 1:
            raise ValueError("observations must have positive row and feature dimensions")
        if cumulants.shape[0] != steps or cumulants.shape[1] < 1:
            raise ValueError("cumulants must match observation rows and have demons")
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must match observations exactly")
        _require_handoff_resources(
            steps=steps,
            raw_dim=feature_dim,
            constructed_dim=0,
            n_demons=cumulants.shape[1],
        )

    @property
    def feature_dim(self) -> int:
        """Dimension of each augmented Horde observation."""
        return int(self.observations.shape[1])

    @property
    def n_demons(self) -> int:
        """Number of cumulant streams/demons."""
        return int(self.cumulants.shape[1])

    def to_dict(self) -> dict[str, object]:
        """Return shape metadata for logs and smoke tests."""
        return {
            "observations_shape": list(self.observations.shape),
            "cumulants_shape": list(self.cumulants.shape),
            "next_observations_shape": list(self.next_observations.shape),
            "feature_dim": self.feature_dim,
            "n_demons": self.n_demons,
        }


@dataclass(frozen=True)
class Step3SmokeResult:
    """Summary returned by :func:`run_step3_smoke`."""

    config: Step3HordeConfig
    steps: int
    seed: int
    final_window_mse: float
    per_demon_metrics_shape: tuple[int, ...]
    td_errors_shape: tuple[int, ...]
    finite: bool
    handoff: Step3HandoffArrays
    horde_config: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.config) is not Step3HordeConfig:
            raise TypeError("config must be an exact Step3HordeConfig")
        object.__setattr__(
            self, "steps", _require_int("steps", self.steps, minimum=1, maximum=_INT32_MAX)
        )
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        object.__setattr__(
            self,
            "final_window_mse",
            _require_nonnegative_real("final_window_mse", self.final_window_mse),
        )
        for name in ("per_demon_metrics_shape", "td_errors_shape"):
            object.__setattr__(
                self,
                name,
                require_step_shape(name, getattr(self, name), steps=self.steps),
            )
        object.__setattr__(self, "finite", _require_bool("finite", self.finite))
        if type(self.handoff) is not Step3HandoffArrays:
            raise TypeError("handoff must be an exact Step3HandoffArrays")
        if self.handoff.observations.shape[0] != self.steps:
            raise ValueError("handoff rows must match steps")
        if self.handoff.n_demons != self.config.n_demons:
            raise ValueError("handoff demons must match config")
        if self.per_demon_metrics_shape != (self.steps, self.config.n_demons, 3):
            raise ValueError("per_demon_metrics_shape must match steps and config demons")
        if self.td_errors_shape != (self.steps, self.config.n_demons):
            raise ValueError("td_errors_shape must match steps and config demons")
        if type(self.horde_config) is not dict:
            raise ValueError("horde_config must be an exact dict with exact string keys")
        horde_config = _bounded_json_clone(
            self.horde_config,
            path="horde_config",
            budget=[0],
        )
        expected_horde_config = make_step3_horde(self.config).to_config()
        if horde_config != expected_horde_config:
            raise ValueError("horde_config must match the exact Step3HordeConfig projection")
        object.__setattr__(self, "horde_config", horde_config)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "config": self.config.to_dict(),
            "steps": self.steps,
            "seed": self.seed,
            "final_window_mse": self.final_window_mse,
            "per_demon_metrics_shape": list(self.per_demon_metrics_shape),
            "td_errors_shape": list(self.td_errors_shape),
            "finite": self.finite,
            "handoff": self.handoff.to_dict(),
            "horde_config": self.horde_config,
        }


@chex.dataclass(frozen=True)
class Step3OneStepResult:
    """Result from one Step 3 transition."""

    state: Step3HordeState
    predictions: Array
    td_errors: Array
    td_targets: Array
    per_demon_metrics: Array


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
        raise ValueError(f"{name} must be in [0, 1]")
    return canonical_float32_storage(real, narrowed)


def _require_gvf_probability(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real < 0.0
        or not real <= 1.0
        or numerator < 0
        or numerator > denominator
        or narrowed < 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    if (
        (numerator != 0 and numerator << 126 < denominator)
        or (real != 0.0 and real < _FLOAT32_MIN_NORMAL)
        or (narrowed != 0.0 and narrowed < _FLOAT32_MIN_NORMAL)
    ):
        raise ValueError(f"{name} must be zero or a normal float32 value in [0, 1]")
    return canonical_float32_storage(real, narrowed)


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return canonical_float32_storage(real, narrowed)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    actual_type = type(value)
    if actual_type not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return number


def _require_positive_int(name: str, value: object) -> int:
    actual_type = type(value)
    if actual_type not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a positive integer")
    number = operator.index(cast(SupportsIndex, value))
    if number < 1:
        raise ValueError(f"{name} must be positive")
    if number > _INT32_MAX:
        raise ValueError(f"{name} must be at most int32 max")
    return number


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_choice(name: str, value: object, choices: Sequence[str]) -> str:
    if type(value) is not str or value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}")
    return value


def _validate_horde_config(config: Step3HordeConfig) -> None:
    if type(config) is not Step3HordeConfig:
        raise ValueError("config must be an exact Step3HordeConfig")
    for name in ("gammas", "lamdas", "hidden_sizes"):
        if type(getattr(config, name)) is not tuple:
            raise ValueError(f"{name} must be an actual tuple")
    if len(config.gammas) == 0:
        raise ValueError("Step 3 Horde must have at least one demon")
    if len(config.gammas) != len(config.lamdas):
        msg = (
            "gammas and lamdas must have the same length, "
            f"got {len(config.gammas)} and {len(config.lamdas)}"
        )
        raise ValueError(msg)
    gammas = tuple(_require_gvf_probability("gammas", value) for value in config.gammas)
    lamdas = tuple(_require_gvf_probability("lamdas", value) for value in config.lamdas)
    step_size = _require_nonnegative_real("step_size", config.step_size)
    sparsity = _require_unit_interval("sparsity", config.sparsity)
    obgd_kappa = _require_positive_real("obgd_kappa", config.obgd_kappa)
    hidden_sizes = tuple(
        _require_positive_int("hidden_sizes", size) for size in config.hidden_sizes
    )
    use_obgd = _require_bool("use_obgd", config.use_obgd)
    use_layer_norm = _require_bool("use_layer_norm", config.use_layer_norm)
    normalizer = _require_choice("normalizer", config.normalizer, ("none", "ema"))
    trace_mode = _require_choice(
        "trace_mode", config.trace_mode, ("accumulating", "replacing")
    )
    routing = _require_choice(
        "routing", config.routing, ("shared", "independent", "mixed")
    )
    object.__setattr__(config, "gammas", gammas)
    object.__setattr__(config, "lamdas", lamdas)
    object.__setattr__(config, "hidden_sizes", hidden_sizes)
    object.__setattr__(config, "step_size", step_size)
    object.__setattr__(config, "sparsity", sparsity)
    object.__setattr__(config, "obgd_kappa", obgd_kappa)
    object.__setattr__(config, "use_obgd", use_obgd)
    object.__setattr__(config, "use_layer_norm", use_layer_norm)
    object.__setattr__(config, "normalizer", normalizer)
    object.__setattr__(config, "trace_mode", trace_mode)
    object.__setattr__(config, "routing", routing)


def make_step3_normalizer(
    config: Step3HordeConfig,
) -> Normalizer[Any] | None:
    """Construct the configured Step 3 input normalizer."""
    _validate_horde_config(config)
    if config.normalizer == "none":
        return None
    if config.normalizer == "ema":
        return EMANormalizer()
    msg = "unknown Step 3 normalizer"
    raise ValueError(msg)


def make_step3_horde_spec(config: Step3HordeConfig | None = None) -> Any:
    """Create the GVF metadata used by the Step 3 Horde."""
    cfg = Step3HordeConfig() if config is None else config
    _validate_horde_config(cfg)
    demons = [
        GVFSpec(
            name=f"gvf_{idx}",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=lamda,
            cumulant_index=idx,
        )
        for idx, (gamma, lamda) in enumerate(zip(cfg.gammas, cfg.lamdas, strict=True))
    ]
    return create_horde_spec(demons)


def make_step3_horde(
    config: Step3HordeConfig | None = None,
) -> HordeLearner | IndependentDemonHorde | MixedHorde:
    """Create the public Step 3 given-feature Horde learner.

    Dispatches on ``config.routing``:

    - ``"shared"`` (default): :class:`HordeLearner` (shared trunk,
      head-only traces). Trunk gamma*lamda is forced to 0.
    - ``"independent"``: :class:`IndependentDemonHorde` (each demon owns
      its own MLP). Full per-parameter forward-view traces.
    - ``"mixed"``: :class:`MixedHorde`. Per-demon routing — demons with
      gamma*lamda=0 land on the shared path; demons with gamma*lamda>0
      land on the independent path. Eliminates the trunk-trace
      constraint while keeping memory cost low when most demons are
      single-step (gamma*lamda=0).
    """
    cfg = Step3HordeConfig() if config is None else config
    _validate_horde_config(cfg)
    bounder = ObGDBounding(kappa=cfg.obgd_kappa) if cfg.use_obgd else None
    common_kwargs: dict[str, Any] = {
        "horde_spec": make_step3_horde_spec(cfg),
        "hidden_sizes": cfg.hidden_sizes,
        "step_size": cfg.step_size,
        "bounder": bounder,
        "normalizer": make_step3_normalizer(cfg),
        "sparsity": cfg.sparsity,
        "use_layer_norm": cfg.use_layer_norm,
        "trace_mode": TraceMode(cfg.trace_mode),
    }
    if cfg.routing == "shared":
        return HordeLearner(**common_kwargs)
    if cfg.routing == "independent":
        return IndependentDemonHorde(**common_kwargs)
    if cfg.routing == "mixed":
        return MixedHorde(**common_kwargs)
    msg = "unknown Step 3 routing"
    raise ValueError(msg)


_STEP3_HORDE_STATES: dict[type, type] = {
    HordeLearner: MultiHeadMLPState,
    IndependentDemonHorde: IndependentDemonHordeState,
    MixedHorde: MixedHordeState,
}


def _require_step3_pair(horde: object, state: object) -> None:
    """Reject a horde/state pair that the Step 3 runtime cannot execute.

    Each routing mode owns exactly one state type. The learners do reject a
    foreign state, but only from inside a traced call, so the pair is checked
    here to keep the three public entry points rejecting identically and to
    surface the mismatch before any array validation runs.
    """
    expected = _STEP3_HORDE_STATES.get(type(horde))
    if expected is None:
        raise ValueError("horde must be an exact supported Step 3 Horde")
    if type(state) is not expected:
        raise ValueError("state must match the horde implementation")


def init_step3_state(
    horde: HordeLearner | IndependentDemonHorde | MixedHorde,
    *,
    feature_dim: int,
    key: Array,
) -> MultiHeadMLPState | IndependentDemonHordeState | MixedHordeState:
    """Initialize a Step 3 Horde state."""
    if type(horde) not in _STEP3_HORDE_STATES:
        raise ValueError("horde must be an exact supported Step 3 Horde")
    feature_dim = _require_positive_int("feature_dim", feature_dim)
    key = _require_typed_key("key", key)
    return horde.init(feature_dim, key)


def step3_predict(
    horde: HordeLearner | IndependentDemonHorde | MixedHorde,
    state: MultiHeadMLPState | IndependentDemonHordeState | MixedHordeState,
    features: Array,
) -> Array:
    """Return one prediction per Step 3 demon."""
    _require_step3_pair(horde, state)
    features = _require_float32_vector("features", features)
    return cast(Array, horde.predict(cast(Any, state), features))


def step3_update(
    horde: HordeLearner | IndependentDemonHorde | MixedHorde,
    state: MultiHeadMLPState | IndependentDemonHordeState | MixedHordeState,
    features: Array,
    cumulants: Array,
    next_features: Array,
) -> Step3OneStepResult:
    """Run one Step 3 Horde transition update."""
    _require_step3_pair(horde, state)
    features = _require_float32_vector("features", features)
    cumulants = _require_float32_vector("cumulants", cumulants)
    next_features = _require_float32_vector("next_features", next_features)
    if next_features.shape != features.shape:
        raise ValueError("next_features must match features")
    if cumulants.shape != (horde.n_demons,):
        raise ValueError("cumulants must match the configured demons")
    result: HordeUpdateResult = horde.update(
        cast(Any, state),
        features,
        cumulants,
        next_features,
    )
    return Step3OneStepResult(
        state=result.state,
        predictions=result.predictions,
        td_errors=result.td_errors,
        td_targets=result.td_targets,
        per_demon_metrics=result.per_demon_metrics,
    )


def run_step3_scan(
    horde: HordeLearner | IndependentDemonHorde | MixedHorde,
    state: MultiHeadMLPState | IndependentDemonHordeState | MixedHordeState,
    features: Array,
    cumulants: Array,
    next_features: Array,
) -> Step3LearningResult:
    """Run the Step 3 Horde over transition arrays."""
    if type(horde) not in (HordeLearner, IndependentDemonHorde, MixedHorde):
        raise TypeError("horde must be an exact HordeLearner or supported routed Horde")
    if type(state) not in (MultiHeadMLPState, IndependentDemonHordeState, MixedHordeState):
        raise TypeError("state must be an exact MultiHeadMLPState or routed Horde state")
    features = _require_float32_matrix("features", features)
    cumulants = _require_float32_matrix("cumulants", cumulants)
    next_features = _require_float32_matrix("next_features", next_features)
    steps = int(features.shape[0])
    feature_dim = int(features.shape[1])
    if not 1 <= steps <= _INT32_MAX:
        raise ValueError("features must contain between 1 and signed-int32 steps")
    if not 1 <= feature_dim <= _INT32_MAX:
        raise ValueError("features must have at least one feature column")
    if next_features.shape != features.shape:
        raise ValueError("next_features must match features")
    if cumulants.shape != (steps, horde.n_demons):
        raise ValueError("cumulants must match steps and configured demons")
    _require_handoff_resources(
        steps=steps,
        raw_dim=feature_dim,
        constructed_dim=0,
        n_demons=horde.n_demons,
    )
    if type(horde) is HordeLearner:
        shared_state = cast(MultiHeadMLPState, state)
        state_feature_dim = (
            shared_state.trunk_params.weights[0].shape[1]
            if shared_state.trunk_params.weights
            else shared_state.head_params.weights[0].shape[1]
        )
        if state_feature_dim != feature_dim:
            raise ValueError(
                f"state feature dimension ({state_feature_dim}) does not match "
                f"features ({feature_dim})"
            )
        if len(shared_state.head_params.weights) != horde.n_demons:
            raise ValueError(
                f"state demon count ({len(shared_state.head_params.weights)}) does not match "
                f"horde demon count ({horde.n_demons})"
            )
        return run_horde_learning_loop(horde, shared_state, features, cumulants, next_features)
    if type(horde) is IndependentDemonHorde:
        if not isinstance(state, IndependentDemonHordeState):
            raise TypeError("state must match the horde implementation")
        return run_independent_horde_learning_loop(horde, state, features, cumulants, next_features)
    if not isinstance(state, MixedHordeState):
        raise TypeError("state must match the horde implementation")
    return run_mixed_horde_learning_loop(
        cast(MixedHorde, horde), state, features, cumulants, next_features
    )


def build_step2_to_step3_arrays(
    raw_observations: Array,
    constructed_features: Array,
    cumulants: Array,
) -> Step3HandoffArrays:
    """Build causal Horde arrays from Step 2 constructed features.

    Args:
        raw_observations: Raw observations, shape ``(steps, raw_dim)``.
        constructed_features: Step 2 features available at the same time index,
            shape ``(steps, constructed_dim)``.
        cumulants: Per-demon cumulants for the transition starting at each row,
            shape ``(steps, n_demons)``.

    Returns:
        Augmented observations, cumulants, and shifted next observations for
        :func:`run_horde_learning_loop`.  The final row's next observation is
        a duplicate of the final observation (the true successor is
        unobserved), so the last transition is a synthetic self-loop whose
        TD target bootstraps on its own features — one boundary row out of
        ``steps``, kept so all arrays share the same length.
    """
    raw = _require_float32_matrix("raw_observations", raw_observations)
    constructed = _require_float32_matrix(
        "constructed_features", constructed_features
    )
    cums = _require_float32_matrix("cumulants", cumulants)
    steps = raw.shape[0]
    if steps < 1:
        raise ValueError("at least one transition is required")
    if raw.shape[1] < 1:
        raise ValueError("raw_observations must have at least one feature column")
    if cums.shape[1] < 1:
        raise ValueError("cumulants must have at least one demon column")
    if constructed.shape[0] != steps:
        msg = (
            "constructed_features must have the same number of rows as "
            f"raw_observations, got {constructed.shape[0]} and {steps}"
        )
        raise ValueError(msg)
    if cums.shape[0] != steps:
        msg = (
            "cumulants must have the same number of rows as raw_observations, "
            f"got {cums.shape[0]} and {steps}"
        )
        raise ValueError(msg)
    _require_handoff_resources(
        steps=steps,
        raw_dim=raw.shape[1],
        constructed_dim=constructed.shape[1],
        n_demons=cums.shape[1],
    )

    observations = jnp.concatenate([raw, constructed], axis=1)
    # Boundary row: the successor of the final observation is unobserved, so
    # the last next-observation duplicates the last observation (a synthetic
    # self-loop) to keep every array the same length.
    next_observations = jnp.concatenate([observations[1:], observations[-1:]], axis=0)
    return Step3HandoffArrays(
        observations=observations,
        cumulants=cums,
        next_observations=next_observations,
    )


def _synthetic_step2_features(raw_observations: Array, n_features: int) -> Array:
    """Create deterministic Step-2-style product features for smoke tests."""
    if n_features < 1:
        return jnp.zeros((raw_observations.shape[0], 0), dtype=jnp.float32)
    raw_dim = raw_observations.shape[1]
    cols = []
    for idx in range(n_features):
        left = idx % raw_dim
        right = (idx + 1) % raw_dim
        cols.append(raw_observations[:, left] * raw_observations[:, right])
    return jnp.stack(cols, axis=1)


def run_step3_smoke(
    config: Step3HordeConfig | None = None,
    *,
    steps: int = 128,
    seed: int = 0,
    final_window: int = 32,
    raw_feature_dim: int = 4,
    constructed_feature_dim: int = 3,
) -> Step3SmokeResult:
    """Run a tiny deterministic Step 3 Horde integration probe.

    The smoke probe verifies the Step 2-to-Horde array contract, Horde
    initialization, TD updates, finite diagnostics, and config serialization.
    It is not a feature-discovery or throughput claim.
    """
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    final_window = _require_int(
        "final_window", final_window, minimum=1, maximum=_INT32_MAX
    )
    raw_feature_dim = _require_int(
        "raw_feature_dim", raw_feature_dim, minimum=1, maximum=_INT32_MAX
    )
    constructed_feature_dim = _require_int(
        "constructed_feature_dim",
        constructed_feature_dim,
        minimum=0,
        maximum=_INT32_MAX,
    )
    seed = require_jax_seed(seed, name="seed")
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")

    cfg = Step3HordeConfig() if config is None else config
    _validate_horde_config(cfg)
    _require_smoke_resources(
        steps=steps,
        raw_dim=raw_feature_dim,
        constructed_dim=constructed_feature_dim,
        n_demons=cfg.n_demons,
    )
    data_key, learner_key = jr.split(jr.key(seed))
    raw_observations = jr.normal(data_key, (steps, raw_feature_dim))
    constructed_features = _synthetic_step2_features(
        raw_observations, constructed_feature_dim
    )
    n_demons = len(cfg.gammas)
    if constructed_feature_dim > 0:
        source = constructed_features
    else:
        source = raw_observations
    cumulants = jnp.stack(
        [source[:, idx % source.shape[1]] for idx in range(n_demons)],
        axis=1,
    )

    arrays = build_step2_to_step3_arrays(
        raw_observations,
        constructed_features,
        cumulants,
    )
    horde = make_step3_horde(cfg)
    state = horde.init(arrays.feature_dim, learner_key)
    result: Any
    if isinstance(horde, MixedHorde):
        result = run_mixed_horde_learning_loop(
            horde,
            state,  # type: ignore[arg-type]
            arrays.observations,
            arrays.cumulants,
            arrays.next_observations,
        )
    elif isinstance(horde, IndependentDemonHorde):
        result = run_independent_horde_learning_loop(
            horde,
            state,  # type: ignore[arg-type]
            arrays.observations,
            arrays.cumulants,
            arrays.next_observations,
        )
    else:
        result = run_horde_learning_loop(
            horde,
            state,  # type: ignore[arg-type]
            arrays.observations,
            arrays.cumulants,
            arrays.next_observations,
        )
    result.per_demon_metrics.block_until_ready()
    window = result.per_demon_metrics[-final_window:, :, 0]
    final_window_mse = float(jnp.nanmean(window))
    finite = bool(
        jnp.all(jnp.isfinite(result.per_demon_metrics))
        & jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(result.updates_applied)
        & jnp.all(result.head_updates_applied)
    )
    return Step3SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        final_window_mse=final_window_mse,
        per_demon_metrics_shape=tuple(int(dim) for dim in result.per_demon_metrics.shape),
        td_errors_shape=tuple(int(dim) for dim in result.td_errors.shape),
        finite=finite,
        handoff=arrays,
        horde_config=horde.to_config(),
    )
