"""Continual Backprop (CBP): per-unit utility tracking and replacement.

Implements Continual Backprop (Dohare, Hernandez-Garcia, Lan, Rahman,
Mahmood, Sutton 2024, "Loss of plasticity in deep continual learning",
*Nature* 632, pp. 768-774). CBP tracks a per-hidden-unit running utility
based on the magnitude of each unit's contribution to the loss
gradient and periodically re-initializes the lowest-utility units that
have aged past a maturity threshold. This prevents the gradual
"plasticity loss" observed when ordinary backprop is run for a long
time on a non-stationary stream.

This module provides:

* :class:`ContinualBackpropConfig` — plain-Python hyperparameters
  (``decay_rate``, ``replacement_rate``, ``maturity_threshold``,
  ``enabled``).
* :class:`ContinualBackpropState` — JAX-pytree state (per-layer
  utility EMAs and per-unit ages).
* :func:`init_cbp_state`, :func:`update_utility`, :func:`maybe_replace_units`
  — pure functional API used inside scan loops.
* :class:`CBPMultiHeadMLPLearner` — wrapper around
  :class:`MultiHeadMLPLearner` that exposes per-layer activations,
  performs the SGD update via :meth:`MultiHeadMLPLearner.update`, and
  then runs CBP utility tracking + replacement on the resulting state.

Design decision: CBP is implemented as a *wrapper* rather than baked
into :class:`MultiHeadMLPLearner` because the latter's JIT-compiled
update is already complex and does not expose intermediate activations.
The wrapper recomputes the per-layer activations from the (just-updated)
trunk parameters to drive the utility EMA and, when triggered, mutates
the trunk parameters in place to re-initialize replaced units.

Per-unit utility convention
---------------------------
For hidden unit ``i`` in layer ``l``, this implementation uses

    u_l[i] <- decay * u_l[i] + (1 - decay) * |a_l[i] * g_l[i]|

where ``a_l[i]`` is the post-activation of unit ``i`` and ``g_l[i]``
is the gradient of the loss w.r.t. that activation. This is the
contribution-based form that captures both how active a unit is and
how much that activation moves the loss. The paper's Eq. 2 expresses
contribution via outgoing-weight magnitudes; the gradient form here
is mathematically related (via the chain rule) and is what the
original Dohare implementation uses in practice.

Replacement
-----------
On every step, ``replacement_rate * num_mature_units`` fractional
replacements accrue per layer (units younger than ``maturity_threshold``
earn no budget), and at most one accumulated replacement is delivered per
step: the mature unit with the lowest utility in the layer is replaced. Replaced units have their
incoming weights re-drawn via :func:`sparse_init` and their outgoing
weights zeroed (so a freshly initialized unit does not destabilize the
prediction immediately). The unit's age and utility are reset to 0.

References
----------
- Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood,
  A. R., & Sutton, R. S. (2024). Loss of plasticity in deep continual
  learning. *Nature*, 632, 768-774.
"""

from __future__ import annotations

import functools
import operator
import time
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar_with_ratio,
)
from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.multi_head_learner import (
    MULTI_HEAD_MLP_STATE_SCHEMA,
    AnyOptimizer,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    MultiHeadMLPUpdateResult,
)
from alberta_framework.core.normalizers import Normalizer
from alberta_framework.core.optimizers import Bounder
from alberta_framework.core.types import TraceMode

_NUMPY_FLOAT_SCALAR_TYPES = frozenset((np.float16, np.float32, np.float64, np.longdouble))
_NUMPY_INTEGER_SCALAR_TYPES = frozenset(
    (
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
)
_CBP_CONFIG_FIELDS = frozenset(
    {"decay_rate", "replacement_rate", "maturity_threshold", "enabled"}
)
_CBP_MULTI_CONFIG_FIELDS = frozenset(
    {
        "type",
        "state_schema",
        "cbp_config",
        "n_heads",
        "hidden_sizes",
        "optimizer",
        "bounder",
        "normalizer",
        "head_optimizer",
        "sparsity",
        "leaky_relu_slope",
        "use_layer_norm",
        "gamma",
        "lamda",
        "per_head_gamma_lamda",
        "trace_mode",
        "utility_decay",
    }
)
_CBP_SINGLE_CONFIG_FIELDS = _CBP_MULTI_CONFIG_FIELDS - {
    "n_heads",
    "per_head_gamma_lamda",
}
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
_ACTUAL_FLOAT_TYPES: frozenset[type] = frozenset(
    {float, Fraction, *(np.dtype(c).type for c in ("e", "f", "d", "g"))}
)
_ALLOWED_REAL_TYPES: frozenset[type] = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _require_int(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer")
    return canonical


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an actual bool")
    return value


def _validated_config_float(
    name: str,
    value: object,
    *,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
    positive: bool = False,
) -> float:
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name,
        value,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
        positive=positive,
    )
    if numerator != 0 and abs(numerator) * (1 << 149) <= denominator:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _copy_mapping(payload: object, *, name: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} payload must be a mapping")
    try:
        data = dict(payload)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{name} payload could not be read") from exc
    for key in data:
        if type(key) is not str:
            raise ValueError(f"{name} payload has exact strings as keys")
    return data

# =============================================================================
# Config / state
# =============================================================================


@chex.dataclass(frozen=True)
class ContinualBackpropConfig:
    """Hyperparameters for Continual Backprop.

    Attributes:
        decay_rate: EMA decay for the per-unit utility estimate. Higher
            decay means slower-moving utility estimate (more inertia).
            Must be in ``[0, 1)``.
        replacement_rate: Fraction of hidden units per layer considered
            for replacement *per step*. Equivalent to ``rho`` in the
            paper. ``1e-4`` means roughly one unit per 10000 steps in
            a 100-unit layer. Must be in ``[0, 1]``.
        maturity_threshold: Minimum age (number of updates) before a
            unit can be replaced. Protects freshly initialized units.
            Must be ``>= 0``.
        enabled: Master switch. When ``False``, the wrapper behaves
            exactly like a plain :class:`MultiHeadMLPLearner` — no
            utility tracking, no replacement.
    """

    decay_rate: float = 0.99
    replacement_rate: float = 1e-4
    maturity_threshold: int = 100
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate and canonicalize configuration at its execution boundaries."""
        object.__setattr__(
            self,
            "decay_rate",
            _validated_config_float(
                "decay_rate",
                self.decay_rate,
                lower=0.0,
                upper=1.0,
                upper_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "replacement_rate",
            _validated_config_float(
                "replacement_rate",
                self.replacement_rate,
                lower=0.0,
                upper=1.0,
            ),
        )
        object.__setattr__(
            self,
            "maturity_threshold",
            _require_int("maturity_threshold", self.maturity_threshold, minimum=0),
        )
        object.__setattr__(self, "enabled", _require_bool("enabled", self.enabled))

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "decay_rate": self.decay_rate,
            "replacement_rate": self.replacement_rate,
            "maturity_threshold": self.maturity_threshold,
            "enabled": self.enabled,
        }

    @classmethod
    def from_config(cls, config: object) -> ContinualBackpropConfig:
        """Reconstruct from dict."""
        payload = _copy_mapping(config, name="ContinualBackpropConfig")
        if set(payload) != _CBP_CONFIG_FIELDS:
            raise ValueError("ContinualBackpropConfig fields do not match the schema")
        for name in ("decay_rate", "replacement_rate"):
            if type(payload[name]) is not float:
                raise ValueError(f"serialized {name} must be an exact JSON float")
        if type(payload["maturity_threshold"]) is not int:
            raise ValueError("serialized maturity_threshold must be an exact JSON integer")
        if type(payload["enabled"]) is not bool:
            raise ValueError("serialized enabled must be an exact JSON bool")
        return cls(**payload)  # type: ignore[arg-type]





@chex.dataclass(frozen=True)
class ContinualBackpropState:
    """Per-layer Continual Backprop state.

    Attributes:
        utilities: Per-hidden-unit utility EMA, one array per hidden
            layer. ``utilities[l]`` has shape ``(hidden_sizes[l],)``.
            Empty tuple when there are no hidden layers.
        ages: Per-hidden-unit age (number of updates since
            initialization or last replacement). Same shape as
            ``utilities``.
        replacement_accumulators: One scalar per hidden layer that
            accumulates the (fractional) number of replacements
            scheduled by ``replacement_rate * num_mature_units`` each
            step. Whenever an accumulator reaches 1.0 and a mature unit
            exists, a single unit in that layer is replaced and the
            accumulator is decremented by 1; it never carries more than
            one pending replacement.
            This makes ``replacement_rate << 1`` behave as expected in
            the JIT-compiled scan loop without integer arithmetic.
        rng_key: PRNG key used to draw fresh weights for replaced
            units. Split each step.
    """

    utilities: tuple[Float[Array, " hidden_dim"], ...]
    ages: tuple[Int[Array, " hidden_dim"], ...]
    replacement_accumulators: Float[Array, " n_layers"]
    rng_key: Array


# =============================================================================
# Pure-functional CBP API
# =============================================================================


def init_cbp_state(
    mlp_state: MultiHeadMLPState,
    hidden_sizes: tuple[int, ...],
    key: Array,
) -> ContinualBackpropState:
    """Initialize Continual Backprop state matching an existing MLP state.

    Args:
        mlp_state: An initialized :class:`MultiHeadMLPState`. Used only
            to validate that ``hidden_sizes`` matches the trunk shape.
        hidden_sizes: Hidden layer sizes from the learner constructor.
            Must match the trunk layer count and every layer's width.
        key: JAX random key for replacement weight sampling.

    Returns:
        Initial :class:`ContinualBackpropState` with zero utilities
        and ages.
    """
    n_layers = len(hidden_sizes)
    if n_layers != len(mlp_state.trunk_params.weights):
        msg = (
            f"hidden_sizes length ({n_layers}) does not match trunk weight "
            f"count ({len(mlp_state.trunk_params.weights)})."
        )
        raise ValueError(msg)
    for layer_idx, (width, weight) in enumerate(
        zip(hidden_sizes, mlp_state.trunk_params.weights, strict=True)
    ):
        if int(weight.shape[0]) != int(width):
            raise ValueError(
                f"hidden_sizes[{layer_idx}]={width} does not match trunk layer "
                f"{layer_idx} width ({weight.shape[0]})"
            )

    utilities = tuple(
        jnp.zeros(h, dtype=jnp.float32) for h in hidden_sizes
    )
    ages = tuple(
        jnp.zeros(h, dtype=jnp.int32) for h in hidden_sizes
    )
    accumulators = jnp.zeros(n_layers, dtype=jnp.float32)
    return ContinualBackpropState(  # type: ignore[call-arg]
        utilities=utilities,
        ages=ages,
        replacement_accumulators=accumulators,
        rng_key=key,
    )


def _trunk_layer_activations(
    weights: tuple[Array, ...],
    biases: tuple[Array, ...],
    observation: Array,
    leaky_relu_slope: float,
    use_layer_norm: bool,
) -> tuple[Array, ...]:
    """Compute the post-activation for each hidden layer.

    Mirrors :meth:`MultiHeadMLPLearner._trunk_forward` but returns the
    list of intermediate activations rather than only the final hidden
    representation.

    Args:
        weights: Trunk weight matrices.
        biases: Trunk bias vectors.
        observation: Input feature vector ``(feature_dim,)``.
        leaky_relu_slope: Negative slope for LeakyReLU.
        use_layer_norm: Whether to apply parameterless layer norm.

    Returns:
        Tuple of ``len(weights)`` arrays, each shape
        ``(hidden_sizes[l],)``, holding the post-activation of layer
        ``l``.
    """
    activations: list[Array] = []
    x = observation
    for i in range(len(weights)):
        x = weights[i] @ x + biases[i]
        if use_layer_norm:
            mean = jnp.mean(x)
            var = jnp.var(x)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = jnp.where(x >= 0, x, leaky_relu_slope * x)
        activations.append(x)
    return tuple(activations)


def update_utility(
    cbp_state: ContinualBackpropState,
    activations: tuple[Array, ...],
    activation_grads: tuple[Array, ...],
    decay_rate: float,
) -> ContinualBackpropState:
    """Update per-unit utility EMA from activations and their gradients.

    Computes, per hidden unit ``i`` in layer ``l``:

        u_l[i] <- decay * u_l[i] + (1 - decay) * |a_l[i] * g_l[i]|

    Also increments every unit's age by 1.

    Args:
        cbp_state: Current CBP state.
        activations: Per-layer post-activations. Each entry shape
            ``(hidden_sizes[l],)``.
        activation_grads: Per-layer gradients of the loss w.r.t. the
            post-activation. Same shapes as ``activations``.
        decay_rate: EMA decay (must be in ``[0, 1)``).

    Returns:
        New CBP state with updated utilities and ages. The
        ``replacement_accumulators`` and ``rng_key`` are unchanged.
    """
    n_layers = len(cbp_state.utilities)
    if len(activations) != n_layers or len(activation_grads) != n_layers:
        msg = (
            f"activations/activation_grads length must match utilities "
            f"length ({n_layers}); got {len(activations)} / "
            f"{len(activation_grads)}."
        )
        raise ValueError(msg)

    decay = jnp.asarray(decay_rate, dtype=jnp.float32)
    new_utilities: list[Array] = []
    new_ages: list[Array] = []
    for i in range(n_layers):
        contribution = jnp.abs(activations[i] * activation_grads[i])
        # Inf activation * a silent gradient is 0*inf = NaN. Hold the
        # previous finite utility so replacement does not treat NaN as
        # the lowest-utility unit.
        contribution_finite = jnp.isfinite(contribution)
        decayed = jnp.where(
            decay == 0.0,
            jnp.zeros_like(cbp_state.utilities[i]),
            decay * cbp_state.utilities[i],
        )
        u_new = decayed + (1.0 - decay) * contribution
        u_new = jnp.where(contribution_finite, u_new, cbp_state.utilities[i])
        new_utilities.append(u_new)
        new_ages.append(cbp_state.ages[i] + 1)

    return cbp_state.replace(  # type: ignore[attr-defined]
        utilities=tuple(new_utilities),
        ages=tuple(new_ages),
    )


def _select_replacement_index(
    utility: Array,
    age: Array,
    maturity_threshold: int,
) -> tuple[Array, Array]:
    """Pick the index of the lowest-utility mature unit, if any.

    A unit is "mature" iff ``age >= maturity_threshold``. Among mature
    units we pick the one with the smallest utility. If no mature unit
    exists, ``selected`` is ``-1`` (sentinel) and ``has_candidate`` is
    ``False``.

    Implementation: replace the utility of immature or non-finite units
    with ``+inf`` before taking ``argmin`` so they are never chosen.

    Args:
        utility: Per-unit utility array, shape ``(num_units,)``.
        age: Per-unit age array (same shape).
        maturity_threshold: Minimum age for eligibility.

    Returns:
        Tuple ``(index, has_candidate)`` where ``index`` is the chosen
        unit (or ``-1``) and ``has_candidate`` is a boolean scalar.
    """
    mature = age >= jnp.asarray(maturity_threshold, dtype=age.dtype)
    eligible = mature & jnp.isfinite(utility)
    masked_utility = jnp.where(eligible, utility, jnp.inf)
    has_candidate = jnp.any(eligible)
    idx = jnp.argmin(masked_utility)
    selected = jnp.where(has_candidate, idx, jnp.int32(-1))
    return selected.astype(jnp.int32), has_candidate


def _replace_one_unit(
    layer_idx: int,
    unit_idx: Array,
    has_candidate: Array,
    sparsity: float,
    mlp_state: MultiHeadMLPState,
    cbp_state: ContinualBackpropState,
    key: Array,
) -> tuple[MultiHeadMLPState, ContinualBackpropState]:
    """Re-initialize a single hidden unit in trunk layer ``layer_idx``.

    Operations (all guarded by ``has_candidate``):
    * Re-initialize the row ``unit_idx`` of ``trunk.weights[layer_idx]``
      via :func:`sparse_init`.
    * Reset ``trunk.biases[layer_idx][unit_idx]`` to 0.
    * Zero outgoing connections to this unit:
      - For trunks: column ``unit_idx`` of ``trunk.weights[layer_idx+1]``.
      - For the last hidden layer: column ``unit_idx`` of every head
        weight matrix.
    * Reset CBP ``utilities[layer_idx][unit_idx]`` and
      ``ages[layer_idx][unit_idx]`` to 0.

    Args:
        layer_idx: Hidden layer index (Python int, used in indexing
            tuples but not traced).
        unit_idx: Unit index inside the layer (traced int32 scalar).
        has_candidate: Boolean scalar; when False, the current unit is retained.
        sparsity: Sparsity fraction for the new incoming weights.
        mlp_state: Current learner state.
        cbp_state: Current CBP state.
        key: PRNG key for sampling new weights.

    Returns:
        Updated ``(mlp_state, cbp_state)``.
    """
    n_layers = len(mlp_state.trunk_params.weights)

    # ---- New incoming weights for unit `unit_idx` of layer `layer_idx`. ----
    weight_layer = mlp_state.trunk_params.weights[layer_idx]
    fan_out, fan_in = weight_layer.shape
    # Sample one full layer's worth of weights, then take row 0 as the
    # new row for the chosen unit. Cheap and JIT-stable.
    sampled = sparse_init(key, (1, fan_in), sparsity=sparsity)
    new_row = jnp.where(has_candidate, sampled[0], weight_layer[unit_idx])
    new_weight_layer = weight_layer.at[unit_idx].set(new_row)

    # Bias of replaced unit -> 0 (only when has_candidate).
    bias_layer = mlp_state.trunk_params.biases[layer_idx]
    new_bias_val = jnp.where(has_candidate, jnp.float32(0.0), bias_layer[unit_idx])
    new_bias_layer = bias_layer.at[unit_idx].set(new_bias_val)

    new_trunk_weights = list(mlp_state.trunk_params.weights)
    new_trunk_biases = list(mlp_state.trunk_params.biases)
    new_trunk_weights[layer_idx] = new_weight_layer
    new_trunk_biases[layer_idx] = new_bias_layer

    # ---- Zero outgoing weights from this unit. ----
    # If layer_idx is not the last hidden layer, zero column `unit_idx`
    # of trunk.weights[layer_idx+1]. Otherwise zero column `unit_idx`
    # of every head weight matrix.
    new_head_weights = list(mlp_state.head_params.weights)
    if layer_idx < n_layers - 1:
        next_w = new_trunk_weights[layer_idx + 1]
        zero_col = jnp.zeros(next_w.shape[0], dtype=next_w.dtype)
        new_col = jnp.where(has_candidate, zero_col, next_w[:, unit_idx])
        new_trunk_weights[layer_idx + 1] = next_w.at[:, unit_idx].set(new_col)
    else:
        # Last hidden layer -> heads.
        for h in range(len(new_head_weights)):
            head_w = new_head_weights[h]
            zero_col = jnp.zeros(head_w.shape[0], dtype=head_w.dtype)
            new_col = jnp.where(has_candidate, zero_col, head_w[:, unit_idx])
            new_head_weights[h] = head_w.at[:, unit_idx].set(new_col)

    new_trunk_params = mlp_state.trunk_params.replace(  # type: ignore[attr-defined]
        weights=tuple(new_trunk_weights),
        biases=tuple(new_trunk_biases),
    )
    new_head_params = mlp_state.head_params.replace(  # type: ignore[attr-defined]
        weights=tuple(new_head_weights),
    )

    new_mlp_state = mlp_state.replace(  # type: ignore[attr-defined]
        trunk_params=new_trunk_params,
        head_params=new_head_params,
    )

    # ---- Reset CBP utility & age for the replaced unit. ----
    util_layer = cbp_state.utilities[layer_idx]
    age_layer = cbp_state.ages[layer_idx]
    new_util_val = jnp.where(has_candidate, jnp.float32(0.0), util_layer[unit_idx])
    new_age_val = jnp.where(has_candidate, jnp.int32(0), age_layer[unit_idx])
    new_util_layer = util_layer.at[unit_idx].set(new_util_val)
    new_age_layer = age_layer.at[unit_idx].set(new_age_val)

    new_utilities = list(cbp_state.utilities)
    new_ages = list(cbp_state.ages)
    new_utilities[layer_idx] = new_util_layer
    new_ages[layer_idx] = new_age_layer
    new_cbp_state = cbp_state.replace(  # type: ignore[attr-defined]
        utilities=tuple(new_utilities),
        ages=tuple(new_ages),
    )

    return new_mlp_state, new_cbp_state


def maybe_replace_units(
    mlp_state: MultiHeadMLPState,
    cbp_state: ContinualBackpropState,
    config: ContinualBackpropConfig,
    sparsity: float,
) -> tuple[MultiHeadMLPState, ContinualBackpropState]:
    """Possibly replace one low-utility, mature unit per hidden layer.

    Uses ``replacement_accumulators`` to convert a fractional
    per-step replacement budget into an integer "replace this step?"
    decision. When ``config.enabled`` is False, the learner state is returned
    unchanged.

    Args:
        mlp_state: Current learner state.
        cbp_state: Current CBP state.
        config: CBP hyperparameters.
        sparsity: Sparsity fraction for newly initialized weights.

    Returns:
        Updated ``(mlp_state, cbp_state)``.
    """
    new_mlp_state, new_cbp_state, _replaced = replace_units_with_flags(
        mlp_state, cbp_state, config, sparsity
    )
    return new_mlp_state, new_cbp_state


def replace_units_with_flags(
    mlp_state: MultiHeadMLPState,
    cbp_state: ContinualBackpropState,
    config: ContinualBackpropConfig,
    sparsity: float,
) -> tuple[MultiHeadMLPState, ContinualBackpropState, Array]:
    """Run :func:`maybe_replace_units` and also return the per-layer replacement flags.

    Returns:
        ``(mlp_state, cbp_state, replaced)`` where ``replaced`` is a boolean
        array with one entry per hidden layer that is ``True`` exactly when
        that layer's gated replacement fired this step.
    """
    n_layers = len(cbp_state.utilities)
    no_replacements = jnp.zeros((n_layers,), dtype=jnp.bool_)
    if not config.enabled or n_layers == 0:
        return mlp_state, cbp_state, no_replacements

    rate = jnp.asarray(config.replacement_rate, dtype=jnp.float32)
    accum_arr = cbp_state.replacement_accumulators
    new_accum_list: list[Array] = []
    replaced_flags: list[Array] = []

    new_mlp_state = mlp_state
    new_cbp_state = cbp_state
    rng_key = cbp_state.rng_key
    for layer_idx in range(n_layers):
        # Add this step's fractional replacements to the accumulator, accrued
        # against the units that are actually eligible (mature) right now, as in
        # the reference GnT implementation; immature units earn no budget.
        eligible = new_cbp_state.ages[layer_idx] >= config.maturity_threshold
        n_eligible = jnp.sum(eligible).astype(jnp.float32)
        accum = accum_arr[layer_idx] + rate * n_eligible
        # Will we replace one unit this step?
        do_replace = accum >= 1.0
        # Pick lowest-utility mature unit from the *current* CBP state.
        unit_idx, has_candidate = _select_replacement_index(
            new_cbp_state.utilities[layer_idx],
            new_cbp_state.ages[layer_idx],
            config.maturity_threshold,
        )
        # Only replace if we both budgeted for it AND found a mature unit.
        gated = jnp.logical_and(do_replace, has_candidate)
        rng_key, subkey = jr.split(rng_key)
        new_mlp_state, new_cbp_state = _replace_one_unit(
            layer_idx,
            unit_idx,
            gated,
            sparsity,
            new_mlp_state,
            new_cbp_state,
            subkey,
        )
        # Decrement accumulator only if we actually replaced, and never carry
        # more than the one replacement a step can deliver.
        accum_after = jnp.minimum(jnp.where(gated, accum - 1.0, accum), 1.0)
        new_accum_list.append(accum_after)
        replaced_flags.append(gated)

    new_cbp_state = new_cbp_state.replace(  # type: ignore[attr-defined]
        replacement_accumulators=jnp.stack(new_accum_list),
        rng_key=rng_key,
    )
    return new_mlp_state, new_cbp_state, jnp.stack(replaced_flags)


# =============================================================================
# CBP wrapper: tracker + learner
# =============================================================================


@chex.dataclass(frozen=True)
class CBPMultiHeadMLPState:
    """Joint state for a CBPMultiHeadMLPLearner.

    Bundles the underlying :class:`MultiHeadMLPState` and the
    :class:`ContinualBackpropState` so they can flow together through
    a JIT-compiled scan loop.

    Attributes:
        mlp_state: Underlying multi-head MLP state.
        cbp_state: Continual Backprop tracker state.
    """

    mlp_state: MultiHeadMLPState
    cbp_state: ContinualBackpropState


@chex.dataclass(frozen=True)
class CBPUpdateResult:
    """Result of a single CBPMultiHeadMLPLearner update.

    Attributes:
        state: Updated joint state.
        predictions: Per-head predictions, shape ``(n_heads,)``.
        errors: Per-head errors. NaN for inactive heads.
        per_head_metrics: Per-head metrics from the underlying MLP
            update, shape ``(n_heads, 3)``.
        trunk_bounding_metric: Trunk bounding scalar from the
            underlying MLP update.
        replacements_made: Boolean array per hidden layer indicating
            whether a unit was replaced this step. Shape
            ``(n_hidden_layers,)``.
    """

    state: CBPMultiHeadMLPState
    predictions: Float[Array, " n_heads"]
    errors: Float[Array, " n_heads"]
    per_head_metrics: Float[Array, "n_heads 3"]
    trunk_bounding_metric: Float[Array, ""]
    replacements_made: Array


@chex.dataclass(frozen=True)
class CBPMLPState:
    """Joint state for a single-output CBP MLP learner."""

    multi_state: CBPMultiHeadMLPState


@chex.dataclass(frozen=True)
class CBPMLPUpdateResult:
    """Result of a single-output CBP MLP update."""

    state: CBPMLPState
    prediction: Float[Array, ""]
    error: Float[Array, ""]
    metrics: Float[Array, " 3"]
    trunk_bounding_metric: Float[Array, ""]
    replacements_made: Array


@chex.dataclass(frozen=True)
class ContinualBackpropTracker:
    """Convenience handle: config + sparsity + utility/replacement helpers.

    Decoupled from :class:`CBPMultiHeadMLPLearner` so callers can drive
    the tracker by hand against any compatible MLP-shaped state.

    Attributes:
        config: CBP hyperparameters.
        sparsity: Sparsity fraction for re-initialized incoming weights.
    """

    config: ContinualBackpropConfig
    sparsity: float = 0.9

    def __post_init__(self) -> None:
        """Reject bool/non-finite sparsity identities used at replacement."""
        object.__setattr__(
            self,
            "sparsity",
            _validated_config_float(
                "sparsity",
                self.sparsity,
                lower=0.0,
                upper=1.0,
            ),
        )


class CBPMultiHeadMLPLearner:
    """Multi-head MLP learner with Continual Backprop unit replacement.

    Wraps a :class:`MultiHeadMLPLearner`. Each :meth:`update` call:

    1. Runs the underlying SGD-style update via
       :meth:`MultiHeadMLPLearner.update`.
    2. Recomputes per-layer activations and their gradients on the
       *post-update* trunk parameters.
    3. Updates per-unit utility EMA and per-unit ages.
    4. If ``config.enabled``, possibly replaces one low-utility
       mature unit per hidden layer.

    Steps 2-4 are skipped when ``config.enabled`` is ``False``; the wrapper
    then matches plain :class:`MultiHeadMLPLearner` behavior.

    The learner is fully JIT-compatible: all branching is inside
    ``jnp.where``, no Python-level conditional compilation is needed.

    Attributes:
        learner: The underlying :class:`MultiHeadMLPLearner`.
        config: CBP hyperparameters.
        n_heads: Number of prediction heads.
    """

    def __init__(
        self,
        n_heads: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        cbp_config: ContinualBackpropConfig | None = None,
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        gamma: float = 0.0,
        lamda: float = 0.0,
        normalizer: Normalizer[Any] | None = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        per_head_gamma_lamda: tuple[float, ...] | None = None,
        trace_mode: TraceMode = TraceMode.ACCUMULATING,
        utility_decay: float = 0.99,
    ):
        """Initialize the CBP-augmented multi-head MLP learner.

        Args:
            n_heads: Number of prediction heads.
            hidden_sizes: Trunk hidden layer sizes (default two 128).
            cbp_config: CBP hyperparameters. When ``None``, defaults to
                ``ContinualBackpropConfig()`` (enabled, decay=0.99,
                rate=1e-4, maturity=100).
            optimizer: Optimizer for weight updates, forwarded to the
                underlying :class:`MultiHeadMLPLearner`.
            step_size: Base learning rate (used only when ``optimizer``
                is ``None``).
            bounder: Optional update bounder.
            gamma: Trunk discount factor for trace decay.
            lamda: Trunk eligibility trace decay parameter.
            normalizer: Optional online feature normalizer.
            sparsity: Fraction of weights zeroed at init AND at
                replacement (matches paper convention).
            leaky_relu_slope: LeakyReLU negative slope.
            use_layer_norm: Whether to apply parameterless layer norm.
            head_optimizer: Optional separate optimizer for the heads.
            per_head_gamma_lamda: Optional per-head trace decay.
            trace_mode: Eligibility trace mode.
            utility_decay: EMA decay for the underlying MLP's native
                hidden-unit utility diagnostics.
        """
        if cbp_config is None:
            cbp_config = ContinualBackpropConfig()
        elif type(cbp_config) is not ContinualBackpropConfig:
            raise ValueError("cbp_config must be an actual ContinualBackpropConfig or None")
        self._cbp_config = cbp_config
        if type(use_layer_norm) is not bool:
            raise ValueError("use_layer_norm must be an actual bool")
        sparsity = _validated_config_float(
            "sparsity",
            sparsity,
            lower=0.0,
            upper=1.0,
        )
        leaky_relu_slope = _validated_config_float(
            "leaky_relu_slope",
            leaky_relu_slope,
            lower=0.0,
        )
        step_size = _validated_config_float("step_size", step_size, positive=True)
        gamma = _validated_config_float("gamma", gamma, lower=0.0, upper=1.0)
        lamda = _validated_config_float("lamda", lamda, lower=0.0, upper=1.0)
        utility_decay = _validated_config_float(
            "utility_decay",
            utility_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope
        self._use_layer_norm = use_layer_norm

        self._learner = MultiHeadMLPLearner(
            n_heads=n_heads,
            hidden_sizes=hidden_sizes,
            optimizer=optimizer,
            step_size=step_size,
            bounder=bounder,
            gamma=gamma,
            lamda=lamda,
            normalizer=normalizer,
            sparsity=sparsity,
            leaky_relu_slope=leaky_relu_slope,
            use_layer_norm=use_layer_norm,
            head_optimizer=head_optimizer,
            per_head_gamma_lamda=per_head_gamma_lamda,
            trace_mode=trace_mode,
            utility_decay=utility_decay,
        )
        self._n_heads = self._learner.n_heads
        self._hidden_sizes = self._learner.hidden_sizes

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """The underlying :class:`MultiHeadMLPLearner`."""
        return self._learner

    @property
    def config(self) -> ContinualBackpropConfig:
        """CBP hyperparameters."""
        return self._cbp_config

    @property
    def n_heads(self) -> int:
        """Number of prediction heads."""
        return self._n_heads

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        """Hidden layer sizes."""
        return self._hidden_sizes

    def to_config(self) -> dict[str, Any]:
        """Serialize learner + CBP config to dict."""
        learner_cfg = self._learner.to_config()
        learner_cfg.pop("type", None)
        return {
            "type": "CBPMultiHeadMLPLearner",
            "cbp_config": self._cbp_config.to_config(),
            **learner_cfg,
        }

    @classmethod
    def from_config(cls, config: object) -> CBPMultiHeadMLPLearner:
        """Reconstruct from a config dict produced by :meth:`to_config`."""
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        payload = _copy_mapping(config, name="CBPMultiHeadMLPLearner")
        if set(payload) != _CBP_MULTI_CONFIG_FIELDS:
            raise ValueError("CBPMultiHeadMLPLearner config fields do not match the schema")
        config = payload
        if type(config["type"]) is not str or config["type"] != "CBPMultiHeadMLPLearner":
            raise ValueError("unexpected CBPMultiHeadMLPLearner config type")
        if (
            type(config["state_schema"]) is not str
            or config["state_schema"] != MULTI_HEAD_MLP_STATE_SCHEMA
        ):
            raise ValueError("unsupported MultiHeadMLP state schema")
        if type(config["hidden_sizes"]) is not list:
            raise ValueError("hidden_sizes must be a list")
        if (
            config["per_head_gamma_lamda"] is not None
            and type(config["per_head_gamma_lamda"]) is not list
        ):
            raise ValueError("per_head_gamma_lamda must be a list")
        if type(config["n_heads"]) is not int:
            raise ValueError("serialized n_heads must be an exact JSON integer")
        if any(type(width) is not int for width in config["hidden_sizes"]):
            raise ValueError("serialized hidden_sizes entries must be exact JSON integers")
        for name in ("sparsity", "leaky_relu_slope", "gamma", "lamda", "utility_decay"):
            if type(config[name]) is not float:
                raise ValueError(f"serialized {name} must be an exact JSON float")
        if type(config["use_layer_norm"]) is not bool:
            raise ValueError("serialized use_layer_norm must be an exact JSON bool")
        if config["per_head_gamma_lamda"] is not None and any(
            type(value) is not float for value in config["per_head_gamma_lamda"]
        ):
            raise ValueError(
                "serialized per_head_gamma_lamda entries must be exact JSON floats"
            )
        config = config.copy()
        config.pop("type")
        config.pop("state_schema")
        cbp_cfg_dict = config.pop("cbp_config")
        cbp_config = ContinualBackpropConfig.from_config(cbp_cfg_dict)

        optimizer = optimizer_from_config(config.pop("optimizer"))
        bounder_cfg = config.pop("bounder", None)
        bounder = bounder_from_config(bounder_cfg) if bounder_cfg is not None else None
        normalizer_cfg = config.pop("normalizer", None)
        normalizer = (
            normalizer_from_config(normalizer_cfg) if normalizer_cfg is not None else None
        )
        head_opt_cfg = config.pop("head_optimizer", None)
        head_optimizer = (
            optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None
        )

        per_head_gl = config.pop("per_head_gamma_lamda")
        if per_head_gl is not None:
            per_head_gl = tuple(per_head_gl)

        trace_mode_str = config.pop("trace_mode")
        if type(trace_mode_str) is not str:
            raise ValueError("trace_mode must be a string")
        trace_mode = TraceMode(trace_mode_str)

        raw_hidden = config.pop("hidden_sizes")
        return cls(
            n_heads=config.pop("n_heads"),
            hidden_sizes=tuple(raw_hidden),
            cbp_config=cbp_config,
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            head_optimizer=head_optimizer,
            per_head_gamma_lamda=per_head_gl,
            trace_mode=trace_mode,
            **config,
        )

    def init(self, feature_dim: int, key: Array) -> CBPMultiHeadMLPState:
        """Initialize joint MLP + CBP state.

        Args:
            feature_dim: Dimension of the input feature vector.
            key: JAX random key. Split between MLP weight init and
                CBP replacement key.

        Returns:
            Initial :class:`CBPMultiHeadMLPState`.
        """
        mlp_key, cbp_key = jr.split(key)
        mlp_state = self._learner.init(feature_dim, mlp_key)
        cbp_state = init_cbp_state(mlp_state, self._hidden_sizes, cbp_key)
        return CBPMultiHeadMLPState(  # type: ignore[call-arg]
            mlp_state=mlp_state,
            cbp_state=cbp_state,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: CBPMultiHeadMLPState, observation: Array) -> Array:
        """Per-head predictions; delegates to the underlying learner."""
        return self._learner.predict(state.mlp_state, observation)  # type: ignore[no-any-return]

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: CBPMultiHeadMLPState,
        observation: Array,
        targets: Array,
    ) -> CBPUpdateResult:
        """Run one CBP-augmented update step.

        Args:
            state: Current joint state.
            observation: Input feature vector ``(feature_dim,)``.
            targets: Per-head targets ``(n_heads,)``. NaN = inactive.

        Returns:
            :class:`CBPUpdateResult`.
        """
        # 1. Underlying SGD-style update.
        mlp_result: MultiHeadMLPUpdateResult = self._learner.update(
            state.mlp_state, observation, targets
        )

        n_layers = len(self._hidden_sizes)

        if not self._cbp_config.enabled or n_layers == 0:
            # CBP disabled OR linear baseline -> just pass through.
            replacements_made = jnp.zeros((max(n_layers, 1),), dtype=jnp.bool_)
            new_state = CBPMultiHeadMLPState(  # type: ignore[call-arg]
                mlp_state=mlp_result.state,
                cbp_state=state.cbp_state,
            )
            return CBPUpdateResult(  # type: ignore[call-arg]
                state=new_state,
                predictions=mlp_result.predictions,
                errors=mlp_result.errors,
                per_head_metrics=mlp_result.per_head_metrics,
                trunk_bounding_metric=mlp_result.trunk_bounding_metric,
                replacements_made=replacements_made,
            )

        # 2. Recompute per-layer activations + their gradients on the
        #    *post-update* parameters. We use the same loss as the
        #    underlying learner: 0.5 * sum_active (pred - target)^2,
        #    averaged over active heads. This gives meaningful
        #    per-activation gradients.
        post_state = mlp_result.state

        # Apply the same normalizer the learner uses (if any).
        obs_for_grad = observation
        if (
            self._learner.normalizer is not None
            and post_state.normalizer_state is not None
        ):
            obs_for_grad = self._learner.normalizer.normalize_only(
                post_state.normalizer_state, observation
            )

        active_mask = ~jnp.isnan(targets)
        safe_targets = jnp.where(active_mask, targets, 0.0)

        slope = self._leaky_relu_slope
        ln = self._use_layer_norm
        n_heads = self._n_heads

        def _loss_from_hidden(hidden: Array) -> Array:
            preds_list: list[Array] = []
            for i in range(n_heads):
                p = jnp.squeeze(
                    post_state.head_params.weights[i] @ hidden
                    + post_state.head_params.biases[i]
                )
                preds_list.append(p)
            preds = jnp.stack(preds_list)
            sq = (preds - safe_targets) ** 2
            sq_masked = jnp.where(active_mask, sq, 0.0)
            n_active = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
            return 0.5 * jnp.sum(sq_masked) / n_active

        def _loss_from_layer_activation(layer_idx: int, activation: Array) -> Array:
            hidden = activation
            for j in range(layer_idx + 1, n_layers):
                hidden = (
                    post_state.trunk_params.weights[j] @ hidden
                    + post_state.trunk_params.biases[j]
                )
                if ln:
                    mean = jnp.mean(hidden)
                    var = jnp.var(hidden)
                    hidden = (hidden - mean) / jnp.sqrt(var + 1e-5)
                hidden = jnp.where(hidden >= 0, hidden, slope * hidden)
            return _loss_from_hidden(hidden)

        activations = _trunk_layer_activations(
            post_state.trunk_params.weights,
            post_state.trunk_params.biases,
            obs_for_grad,
            slope,
            ln,
        )
        # `acts_grad` has the same shape as `activations`. Each layer's
        # activation must remain connected to the downstream trunk; taking
        # grad of a loss that consumes only `acts[-1]` would silently give
        # zero utility to every earlier hidden layer.
        acts_grad = tuple(
            jax.grad(functools.partial(_loss_from_layer_activation, i))(
                activations[i]
            )
            for i in range(n_layers)
        )

        # 3. EMA utility update + age increment.
        new_cbp_state = update_utility(
            state.cbp_state,
            activations,
            acts_grad,
            self._cbp_config.decay_rate,
        )

        # 4. Possibly replace low-utility mature units, reporting the exact
        #    gated decision per layer as the diagnostic.
        new_post_state, new_cbp_state, replacements_made = replace_units_with_flags(
            post_state,
            new_cbp_state,
            self._cbp_config,
            self._sparsity,
        )

        new_state = CBPMultiHeadMLPState(  # type: ignore[call-arg]
            mlp_state=new_post_state,
            cbp_state=new_cbp_state,
        )

        return CBPUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            predictions=mlp_result.predictions,
            errors=mlp_result.errors,
            per_head_metrics=mlp_result.per_head_metrics,
            trunk_bounding_metric=mlp_result.trunk_bounding_metric,
            replacements_made=replacements_made,
        )


class CBPMLPLearner:
    """Single-output MLP learner with Continual Backprop feature replacement.

    This is a thin single-head adapter over :class:`CBPMultiHeadMLPLearner`.
    It gives the standard Step 2 scalar-prediction MLP path the same
    per-hidden-unit utility tracking and low-utility unit replacement used by
    multi-head Horde-style learners.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 128),
        cbp_config: ContinualBackpropConfig | None = None,
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        gamma: float = 0.0,
        lamda: float = 0.0,
        normalizer: Normalizer[Any] | None = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        trace_mode: TraceMode = TraceMode.ACCUMULATING,
        utility_decay: float = 0.99,
    ) -> None:
        """Initialize the single-output CBP MLP learner."""
        self._learner = CBPMultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=hidden_sizes,
            cbp_config=cbp_config,
            optimizer=optimizer,
            step_size=step_size,
            bounder=bounder,
            gamma=gamma,
            lamda=lamda,
            normalizer=normalizer,
            sparsity=sparsity,
            leaky_relu_slope=leaky_relu_slope,
            use_layer_norm=use_layer_norm,
            head_optimizer=head_optimizer,
            trace_mode=trace_mode,
            utility_decay=utility_decay,
        )

    @property
    def learner(self) -> CBPMultiHeadMLPLearner:
        """Underlying one-head CBP multi-head learner."""
        return self._learner

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration."""
        cfg = self._learner.to_config()
        cfg["type"] = "CBPMLPLearner"
        cfg.pop("n_heads", None)
        cfg.pop("per_head_gamma_lamda", None)
        return cfg

    @classmethod
    def from_config(cls, config: object) -> CBPMLPLearner:
        """Reconstruct from a config dict produced by :meth:`to_config`."""
        payload = _copy_mapping(config, name="CBPMLPLearner")
        if set(payload) != _CBP_SINGLE_CONFIG_FIELDS:
            raise ValueError("CBPMLPLearner config fields do not match the schema")
        config = payload
        if type(config["type"]) is not str or config["type"] != "CBPMLPLearner":
            raise ValueError("unexpected CBPMLPLearner config type")
        config = config.copy()
        config["type"] = "CBPMultiHeadMLPLearner"
        config["n_heads"] = 1
        config["per_head_gamma_lamda"] = None
        rebuilt = CBPMultiHeadMLPLearner.from_config(config)
        instance = cls.__new__(cls)
        instance._learner = rebuilt
        return instance

    def init(self, feature_dim: int, key: Array) -> CBPMLPState:
        """Initialize single-output CBP MLP state."""
        return CBPMLPState(  # type: ignore[call-arg]
            multi_state=self._learner.init(feature_dim, key)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: CBPMLPState, observation: Array) -> Array:
        """Predict one scalar target."""
        return self._learner.predict(state.multi_state, observation)[0]

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: CBPMLPState,
        observation: Array,
        target: Array,
    ) -> CBPMLPUpdateResult:
        """Run one single-output CBP MLP update."""
        result = self._learner.update(
            state.multi_state,
            observation,
            jnp.reshape(jnp.asarray(target, dtype=jnp.float32), (1,)),
        )
        return CBPMLPUpdateResult(  # type: ignore[call-arg]
            state=CBPMLPState(multi_state=result.state),
            prediction=result.predictions[0],
            error=result.errors[0],
            metrics=result.per_head_metrics[0],
            trunk_bounding_metric=result.trunk_bounding_metric,
            replacements_made=result.replacements_made,
        )


# =============================================================================
# Loops
# =============================================================================


@chex.dataclass(frozen=True)
class CBPLearningResult:
    """Result of a CBP-MLP scan-based learning loop.

    Attributes:
        state: Final joint state.
        per_head_metrics: Per-step, per-head metrics, shape
            ``(num_steps, n_heads, 3)``.
        replacements_made: Per-step, per-layer replacement flags,
            shape ``(num_steps, n_hidden_layers)``.
    """

    state: CBPMultiHeadMLPState
    per_head_metrics: Float[Array, "num_steps n_heads 3"]
    replacements_made: Array


def _require_scan_array_metadata(name: str, array: object) -> tuple[int, ...]:
    actual_type = type(array)
    if not (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise TypeError(f"{name} must be a trusted array")
    try:
        shape = tuple(int(dim) for dim in array.shape)  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose trusted shape metadata") from error
    return shape


def _require_scan_resource(num_steps: int, n_heads: int, n_hidden_layers: int) -> None:
    output_scalars = num_steps * (3 * n_heads + n_hidden_layers)
    if output_scalars > _INT32_MAX or 4 * output_scalars > _INT32_MAX:
        raise ValueError("CBP learning loop scan result bytes must fit signed int32")


def run_cbp_learning_loop(
    learner: CBPMultiHeadMLPLearner,
    state: CBPMultiHeadMLPState,
    observations: Float[Array, "num_steps feature_dim"],
    targets: Float[Array, "num_steps n_heads"],
) -> CBPLearningResult:
    """Run a CBP-MLP learning loop via :func:`jax.lax.scan`.

    Args:
        learner: CBP-augmented multi-head MLP learner.
        state: Initial joint state.
        observations: Input observations ``(num_steps, feature_dim)``.
        targets: Per-head targets ``(num_steps, n_heads)``. NaN = inactive.

    Returns:
        :class:`CBPLearningResult` with the final state, per-step
        per-head metrics, and per-step per-layer replacement flags.
    """
    if type(learner) is not CBPMultiHeadMLPLearner:
        raise TypeError("learner must be an exact CBPMultiHeadMLPLearner")
    if type(state) is not CBPMultiHeadMLPState:
        raise TypeError("state must be an exact CBPMultiHeadMLPState")

    obs_shape = _require_scan_array_metadata("observations", observations)
    if len(obs_shape) != 2:
        raise ValueError("observations must have shape (num_steps, feature_dim)")
    num_steps = _require_int("scan sequence length", obs_shape[0], minimum=1)

    tgt_shape = _require_scan_array_metadata("targets", targets)
    if tgt_shape != (num_steps, learner.n_heads):
        raise ValueError(f"targets must have shape ({num_steps}, {learner.n_heads})")

    _require_scan_resource(num_steps, learner.n_heads, len(learner.hidden_sizes))

    obs_array = jnp.asarray(observations, dtype=jnp.float32)
    tgt_array = jnp.asarray(targets, dtype=jnp.float32)

    def step_fn(
        carry: CBPMultiHeadMLPState, inputs: tuple[Array, Array]
    ) -> tuple[CBPMultiHeadMLPState, tuple[Array, Array]]:
        obs, tgt = inputs
        result = learner.update(carry, obs, tgt)
        return result.state, (result.per_head_metrics, result.replacements_made)

    t0 = time.time()
    final_state, (per_head_metrics, replacements_made) = jax.lax.scan(
        step_fn, state, (obs_array, tgt_array)
    )
    elapsed = time.time() - t0
    final_mlp = final_state.mlp_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.mlp_state.uptime_s + elapsed
    )
    final_state = final_state.replace(mlp_state=final_mlp)  # type: ignore[attr-defined]

    return CBPLearningResult(  # type: ignore[call-arg]
        state=final_state,
        per_head_metrics=per_head_metrics,
        replacements_made=replacements_made,
    )
