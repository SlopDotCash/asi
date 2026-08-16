# mypy: disable-error-code="call-arg"
"""Stacked linear Horde: thousands of GVF demons as one batched TD update.

The existing Horde implementations (:mod:`~alberta_framework.core.horde`,
:mod:`~alberta_framework.core.independent_demon_horde`) unroll a Python loop
over demons inside ``jit``, so program size — and compile time and memory —
grow linearly with the demon count and throughput collapses well before the
"thousands of demons" regime the Horde architecture calls for (Sutton et al.
2011).  This module takes the other route for the linear case: every demon's
parameters live in one stacked array ``(n_demons, feature_dim)`` and the
entire Horde updates with a handful of batched array operations per step — no
per-demon loop exists anywhere, so program size is *constant* in the demon
count and the demon axis scales like any other array axis (and shards across
devices the same way).

Each demon d learns a linear GVF prediction ``v_d(x) = w_d @ x`` about the
shared behavior stream with its own question functions:

- cumulant ``c_d`` (selected from the observation/target channels or passed
  directly),
- pseudo-termination ``gamma_d``,
- trace decay ``lamda_d``,
- optional per-decision importance ratio ``rho`` for off-policy questions
  (per-decision IS with the ratio composed into the trace,
  ``z_d = rho * (gamma_d * lamda_d * z_d + x)``, matching the repo's
  post-fix off-policy convention).

The update is the textbook TD(lambda) per demon over shared features::

    delta_d = c_d + gamma_d * (w_d @ x') - (w_d @ x)
    z_d     = rho * (gamma_d * lamda_d * z_d + x)
    w_d    += alpha * delta_d * z_d

vectorized across all demons at once.  A ``NaN`` cumulant marks a demon
inactive for that step (its trace still decays; its weights freeze), matching
the NaN-masking convention of the loop-based hordes.
"""

import math
import operator
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework._float32 import round_real_to_float32_with_ratio

__all__ = [
    "StackedHordeConfig",
    "StackedHordeState",
    "StackedHordeUpdateResult",
    "StackedLinearHorde",
    "nexting_spec",
]

_FLOAT32_MIN_NORMAL = float.fromhex("0x1.0p-126")
_STEP_SIZE_ERROR = "step_size must be positive"
_NUMPY_STEP_SIZE_TYPES = tuple(
    np.dtype(dtype_code).type
    for dtype_code in (
        "b",
        "B",
        "h",
        "H",
        "i",
        "I",
        "l",
        "L",
        "q",
        "Q",
        "e",
        "f",
        "d",
        "g",
    )
)
_TRUSTED_STEP_SIZE_TYPES = (int, float, Fraction, *_NUMPY_STEP_SIZE_TYPES)


def _snapshot_exact_fraction(value: Fraction) -> Fraction:
    """Copy a structurally canonical exact Fraction without conversion hooks."""
    try:
        numerator = value.numerator
        denominator = value.denominator
    except AttributeError:
        raise ValueError(_STEP_SIZE_ERROR) from None
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError(_STEP_SIZE_ERROR)
    return Fraction(numerator, denominator)


def _require_positive_normal_float32_step_size(value: object) -> float:
    """Return a JSON-safe scalar whose float32 execution value is usable.

    JAX CPU execution flushes float32 subnormals to zero, so accepting a
    positive host value whose binary32 sink is zero or subnormal can create a
    Horde that reports applied updates while its weights never move. Only
    exact trusted scalar families are admitted, so user subclasses are
    refused before conversion hooks can run. Exact Fractions additionally
    require exact builtin-integer components and are copied before shared
    conversion. Exact-ratio conversion prevents overloaded host comparisons
    from hiding a negative value and avoids double-rounding supported
    third-party reals.
    """
    value_type = type(value)
    preserve_builtin_float = value_type is float
    if not any(value_type is trusted_type for trusted_type in _TRUSTED_STEP_SIZE_TYPES):
        raise ValueError(_STEP_SIZE_ERROR)
    real = (
        _snapshot_exact_fraction(cast(Fraction, value))
        if value_type is Fraction
        else cast(Real, value)
    )
    try:
        numerator, _, narrowed = round_real_to_float32_with_ratio(real)
    except (FloatingPointError, OverflowError, TypeError, ValueError):
        raise ValueError(_STEP_SIZE_ERROR) from None
    if numerator <= 0 or not math.isfinite(narrowed) or narrowed < _FLOAT32_MIN_NORMAL:
        raise ValueError(_STEP_SIZE_ERROR)
    # Preserve an already JSON-safe builtin float for payload compatibility.
    # Every other accepted Real is stored as the validated binary32 value.
    return cast(float, value) if preserve_builtin_float else narrowed


_INT32_MAX = 2**31 - 1
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


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


@dataclass(frozen=True)
class StackedHordeConfig:
    """Static configuration for :class:`StackedLinearHorde`.

    Attributes:
        n_demons: Number of GVF demons.
        feature_dim: Shared feature dimension.
        gammas: Per-demon pseudo-termination discounts, length ``n_demons``.
        lamdas: Per-demon trace decays, length ``n_demons``.
        cumulant_indices: Per-demon index into the cumulant-source vector
            handed to :meth:`StackedLinearHorde.update`.
        step_size: Shared TD step-size alpha. Its float32 execution value must
            be finite, positive, and normal; accepted supported non-builtin
            reals are canonicalized to a JSON-safe builtin float.
    """

    n_demons: int
    feature_dim: int
    gammas: tuple[float, ...]
    lamdas: tuple[float, ...]
    cumulant_indices: tuple[int, ...]
    step_size: float = 0.05

    def __post_init__(self) -> None:
        """Validate the configuration."""
        n_demons = _require_int32("n_demons", self.n_demons, minimum=1)
        feature_dim = _require_int32("feature_dim", self.feature_dim, minimum=1)

        for name, seq in (
            ("gammas", self.gammas),
            ("lamdas", self.lamdas),
            ("cumulant_indices", self.cumulant_indices),
        ):
            if len(seq) != n_demons:
                raise ValueError(f"{name} must have length n_demons={n_demons}")
        if any(not 0.0 <= g <= 1.0 for g in self.gammas):
            raise ValueError("every gamma must be in [0, 1]")
        if any(not 0.0 <= la <= 1.0 for la in self.lamdas):
            raise ValueError("every lamda must be in [0, 1]")

        cumulant_indices = tuple(
            _require_int32(f"cumulant_indices[{i}]", idx, minimum=0)
            for i, idx in enumerate(self.cumulant_indices)
        )

        object.__setattr__(self, "n_demons", n_demons)
        object.__setattr__(self, "feature_dim", feature_dim)
        object.__setattr__(self, "cumulant_indices", cumulant_indices)
        object.__setattr__(
            self,
            "step_size",
            _require_positive_normal_float32_step_size(self.step_size),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this config to a dictionary."""
        return {
            "type": "StackedHordeConfig",
            "n_demons": self.n_demons,
            "feature_dim": self.feature_dim,
            "gammas": list(self.gammas),
            "lamdas": list(self.lamdas),
            "cumulant_indices": list(self.cumulant_indices),
            "step_size": self.step_size,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StackedHordeConfig":
        """Reconstruct a config from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        for key in ("gammas", "lamdas", "cumulant_indices"):
            config[key] = tuple(config[key])
        return cls(**config)


def nexting_spec(
    feature_dim: int,
    cumulant_indices: tuple[int, ...],
    gammas: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99),
    lamda: float = 0.7,
    step_size: float = 0.05,
) -> StackedHordeConfig:
    """Build a nexting-style config: every cumulant at every timescale.

    Multi-timescale prediction of many sensor channels at once is the
    canonical Horde workload (Modayil, White & Sutton 2014, "nexting").  The
    returned config has ``len(cumulant_indices) * len(gammas)`` demons,
    ordered cumulant-major.

    Args:
        feature_dim: Shared feature dimension.
        cumulant_indices: Which cumulant channels to predict.
        gammas: Prediction timescales for each channel.
        lamda: Shared trace decay.
        step_size: Shared TD step-size.
    """
    idxs: list[int] = []
    gs: list[float] = []
    for c in cumulant_indices:
        for g in gammas:
            idxs.append(c)
            gs.append(g)
    n = len(idxs)
    return StackedHordeConfig(
        n_demons=n,
        feature_dim=feature_dim,
        gammas=tuple(gs),
        lamdas=(lamda,) * n,
        cumulant_indices=tuple(idxs),
        step_size=step_size,
    )


@chex.dataclass(frozen=True)
class StackedHordeState:
    """State for :class:`StackedLinearHorde`.

    Attributes:
        weights: Stacked demon weights, shape ``(n_demons, feature_dim)``.
        traces: Stacked eligibility traces, same shape.
        step_count: Number of updates applied.
    """

    weights: Float[Array, "n_demons feature_dim"]
    traces: Float[Array, "n_demons feature_dim"]
    step_count: Array


@chex.dataclass(frozen=True)
class StackedHordeUpdateResult:
    """Result of one stacked update.

    Attributes:
        state: Updated horde state.
        predictions: Pre-update predictions ``v_d(x)``, shape ``(n_demons,)``.
        td_errors: Per-demon TD errors (NaN where the demon was inactive).
    """

    state: StackedHordeState
    predictions: Float[Array, " n_demons"]
    td_errors: Float[Array, " n_demons"]
    head_updates_applied: Bool[Array, " n_demons"]
    update_applied: Bool[Array, ""]


class StackedLinearHorde:
    """Batched linear TD(lambda) over a stacked demon axis (module docstring)."""

    def __init__(self, config: StackedHordeConfig):
        """Initialize the horde around a static config."""
        self._config = config
        self._gammas = jnp.asarray(config.gammas, dtype=jnp.float32)
        self._lamdas = jnp.asarray(config.lamdas, dtype=jnp.float32)
        self._cumulant_idx = jnp.asarray(config.cumulant_indices, dtype=jnp.int32)
        self._max_cumulant_index = max(config.cumulant_indices)

    @property
    def config(self) -> StackedHordeConfig:
        """The static configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize this horde to a dictionary."""
        return {"type": "StackedLinearHorde", "config": self._config.to_config()}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StackedLinearHorde":
        """Reconstruct a horde from :meth:`to_config` output."""
        return cls(StackedHordeConfig.from_config(dict(config)["config"]))

    def init(self) -> StackedHordeState:
        """Initialize zero weights and traces."""
        cfg = self._config
        shape = (cfg.n_demons, cfg.feature_dim)
        return StackedHordeState(
            weights=jnp.zeros(shape, dtype=jnp.float32),
            traces=jnp.zeros(shape, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def predict(self, state: StackedHordeState, features: Array) -> Array:
        """All demons' predictions for one feature vector, shape ``(n_demons,)``."""
        return state.weights @ features

    def update(
        self,
        state: StackedHordeState,
        features: Array,
        next_features: Array,
        cumulant_source: Array,
        rho: Array | float = 1.0,
    ) -> StackedHordeUpdateResult:
        """One batched TD(lambda) step for every demon at once.

        Args:
            state: Current horde state.
            features: Feature vector ``x_t``, shape ``(feature_dim,)``.
            next_features: Feature vector ``x_{t+1}``.
            cumulant_source: Vector the demons draw their cumulants from via
                ``config.cumulant_indices`` (e.g. the next observation, or a
                dedicated signal vector).  A ``NaN`` entry deactivates every
                demon reading it for this step: the demon's weights freeze
                and its trace only decays.
            rho: Per-decision importance ratio of the behavior action under
                each demon's target policy.  Scalar (shared) or shape
                ``(n_demons,)``.  On-policy questions use 1.0.

        Returns:
            :class:`StackedHordeUpdateResult` with per-demon predictions and
            TD errors (TD errors are NaN for inactive demons).
        """
        cfg = self._config
        source_shape = jnp.shape(cumulant_source)
        # JAX clips out-of-range positive gather indices instead of raising,
        # so a short source would silently train demons on the wrong channel.
        # Shapes are static, so these checks also fire at jit/scan trace time.
        if len(source_shape) != 1:
            raise ValueError(f"cumulant_source must be rank-one, got shape {source_shape}")
        if source_shape[0] <= self._max_cumulant_index:
            raise ValueError(
                f"cumulant_source has {source_shape[0]} channels but "
                f"config.cumulant_indices requires index {self._max_cumulant_index}"
            )
        cumulants = cumulant_source[self._cumulant_idx]  # (n_demons,)
        requested = ~jnp.isnan(cumulants)
        active = jnp.isfinite(cumulants)
        rho_vec = jnp.broadcast_to(jnp.asarray(rho, dtype=jnp.float32), (cfg.n_demons,))
        rho_valid = jnp.isfinite(rho_vec)

        v = state.weights @ features  # (n_demons,)
        v_next = state.weights @ next_features
        # gamma=0 must not multiply an inf bootstrap (0*inf).
        bootstrap = jnp.where(self._gammas == 0.0, 0.0, self._gammas * v_next)
        raw_td_errors = cumulants + bootstrap - v
        prediction_valid = jnp.isfinite(v) & ((self._gammas == 0.0) | jnp.isfinite(v_next))
        channel_valid = active & rho_valid & prediction_valid & jnp.isfinite(raw_td_errors)
        safe_cumulants = jnp.where(channel_valid, cumulants, 0.0)
        td_errors = safe_cumulants + bootstrap - v

        # Per-decision IS with the ratio composed into the trace:
        # z_d = rho_d * (gamma_d * lamda_d * z_d + x).
        decay = (self._gammas * self._lamdas)[:, None]  # (n_demons, 1)
        carried = jnp.where(decay == 0.0, jnp.zeros_like(state.traces), decay * state.traces)
        accumulated = carried + features[None, :]
        # Inactive demons: trace decays but the current gradient is withheld.
        decayed_only = carried
        safe_rho = jnp.where(rho_valid, rho_vec, 0.0)
        proposed_traces = safe_rho[:, None] * jnp.where(active[:, None], accumulated, decayed_only)

        masked_delta = jnp.where(channel_valid, td_errors, 0.0)
        proposed_weights = state.weights + cfg.step_size * masked_delta[:, None] * proposed_traces
        source_weights_finite = jnp.all(jnp.isfinite(state.weights))
        traces_needed = decay[:, 0] != 0.0
        traces_usable = (~traces_needed) | jnp.all(jnp.isfinite(state.traces), axis=1)
        shared_inputs_valid = (
            jnp.all(jnp.isfinite(features))
            & jnp.all(jnp.isfinite(next_features))
            & source_weights_finite
        )
        candidate_finite = jnp.all(jnp.isfinite(proposed_weights), axis=1) & jnp.all(
            jnp.isfinite(proposed_traces), axis=1
        )
        head_updates_applied = (
            shared_inputs_valid & channel_valid & candidate_finite & traces_usable
        )
        inactive_trace_applied = (
            shared_inputs_valid
            & ~requested
            & rho_valid
            & prediction_valid
            & candidate_finite
            & traces_usable
        )
        accepted_channels = head_updates_applied | inactive_trace_applied
        new_weights = jnp.where(head_updates_applied[:, None], proposed_weights, state.weights)
        new_traces = jnp.where(accepted_channels[:, None], proposed_traces, state.traces)
        update_applied = jnp.any(accepted_channels)
        proposed_state = StackedHordeState(
            weights=new_weights,
            traces=new_traces,
            step_count=state.step_count + 1,
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        reported_td_errors = jnp.where(
            head_updates_applied,
            td_errors,
            jnp.where(
                requested,
                jnp.zeros_like(td_errors),
                jnp.full_like(td_errors, jnp.nan),
            ),
        )
        return StackedHordeUpdateResult(
            state=new_state,
            predictions=jnp.where(
                update_applied,
                jnp.where(
                    (requested & ~head_updates_applied) | ~prediction_valid,
                    jnp.zeros_like(v),
                    v,
                ),
                jnp.zeros_like(v),
            ),
            td_errors=reported_td_errors,
            head_updates_applied=head_updates_applied,
            update_applied=update_applied,
        )


def run_stacked_horde_scan(
    horde: StackedLinearHorde,
    state: StackedHordeState,
    features: Float[Array, "num_steps feature_dim"],
    cumulant_sources: Float[Array, "num_steps source_dim"],
    rhos: Float[Array, " num_steps"] | None = None,
) -> tuple[StackedHordeState, Float[Array, "num_steps n_demons"]]:
    """Scan the horde over transition arrays.

    ``features[t]`` and ``features[t+1]`` form each transition; the final
    row only provides a bootstrap target, so ``num_steps - 1`` updates run.

    Args:
        horde: The stacked horde.
        state: Initial state.
        features: Feature vectors per step.
        cumulant_sources: Cumulant-source vectors per transition (row ``t``
            is consumed by the ``t -> t+1`` update).
        rhos: Optional per-transition importance ratios (shared across
            demons); defaults to on-policy 1.0.

    Returns:
        ``(final_state, td_errors)`` with td_errors of shape
        ``(num_steps - 1, n_demons)``.
    """
    sources_shape = jnp.shape(cumulant_sources)
    if len(sources_shape) != 2:
        raise ValueError(f"cumulant_sources must be rank-two, got shape {sources_shape}")
    max_index = max(horde.config.cumulant_indices)
    if sources_shape[-1] <= max_index:
        raise ValueError(
            f"cumulant_sources has {sources_shape[-1]} channels but "
            f"config.cumulant_indices requires index {max_index}"
        )
    num_steps = features.shape[0]
    if rhos is None:
        rhos = jnp.ones((num_steps,), dtype=jnp.float32)

    def step_fn(
        carry: StackedHordeState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[StackedHordeState, Array]:
        x, x_next, c, rho = inputs
        result = horde.update(carry, x, x_next, c, rho)
        return result.state, result.td_errors

    final_state, td_errors = jax.lax.scan(
        step_fn,
        state,
        (features[:-1], features[1:], cumulant_sources[:-1], rhos[:-1]),
    )
    return final_state, td_errors
