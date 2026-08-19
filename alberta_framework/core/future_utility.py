"""Causal future-utility estimators for feature discovery.

The estimators here intentionally use only information available at the
current time step.  They are not hindsight ablations.  The main signal is a
one-step counterfactual: how much the current squared prediction error would
drop if a feature's output weight received the LMS update implied by the
current residual.
"""

import math
from fractions import Fraction
from numbers import Real
from typing import NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework._float32 import round_real_to_float32_with_ratio
from alberta_framework.core._float32_scalars import validated_float32_scalar

_ACTUAL_DECAY_TYPES = frozenset(
    {
        int,
        float,
        Fraction,
        *(np.dtype(code).type for code in "bBhHiIlLqQpefdg"),
    }
)


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Skip ``0 * inf`` so a disabled decay does not poison the next trace."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def canonical_float32_ema_decay(name: str, value: object) -> float:
    """Validate an exact EMA decay and canonicalize its float32 execution value.

    The exact-ratio checks retain domain facts that float32 narrowing can erase:
    a tiny negative must not become an accepted ``-0.0``, and a positive decay
    must not silently become the disabled-decay endpoint ``0.0``.
    """
    message = f"{name} must narrow to a finite float32 in [0, 1)"
    if type(value) not in _ACTUAL_DECAY_TYPES:
        raise ValueError(message)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(
            cast(Real, value)
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(message) from error
    exact_in_domain = 0 <= numerator < denominator
    positive_underflow = numerator > 0 and narrowed == 0.0
    if (
        not exact_in_domain
        or positive_underflow
        or not math.isfinite(narrowed)
        or not 0.0 <= narrowed < 1.0
    ):
        raise ValueError(message)
    return narrowed


class FutureUtilityEstimate(NamedTuple):
    """One-step utility estimate with an explicit transaction verdict."""

    reductions: Float[Array, "n_tasks n_features"]
    update_applied: Bool[Array, ""]


class ContributionTraceFutureUtilityEstimate(NamedTuple):
    """Contribution-trace estimate and an explicit transaction verdict."""

    reductions: Float[Array, "n_tasks n_features"]
    contribution_trace: Float[Array, "n_tasks n_features"]
    feature_energy_trace: Float[Array, " n_features"]
    update_applied: Bool[Array, ""]


class TraceFutureUtilityEstimate(NamedTuple):
    """Factored-trace estimate and an explicit transaction verdict."""

    reductions: Float[Array, "n_tasks n_features"]
    error_trace: Float[Array, " n_tasks"]
    feature_trace: Float[Array, " n_features"]
    feature_energy_trace: Float[Array, " n_features"]
    update_applied: Bool[Array, ""]


def trace_decay_from_half_life(half_life: float | Array) -> Float[Array, ""]:
    """Convert an eligibility half-life in steps to a trace decay.

    A half-life of ``0`` disables the trace.  Positive half-lives use
    ``0.5 ** (1 / half_life)`` so the trace contribution decays by half after
    that many time steps.

    Host scalars are validated before JAX conversion so ``True`` cannot become
    half-life ``1`` (decay ``0.5``) and non-finite values cannot become
    ``nan``/``1.0`` decays. JAX arrays and tracers keep the device formula.
    """
    if not isinstance(half_life, Array):
        half_life = validated_float32_scalar("half_life", half_life, lower=0.0)
    half_life_arr = jnp.asarray(half_life, dtype=jnp.float32)
    return jnp.where(
        half_life_arr <= 0.0,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.power(jnp.array(0.5, dtype=jnp.float32), 1.0 / half_life_arr),
    )


def one_step_output_loss_reduction_with_diagnostics(
    errors: Float[Array, " n_tasks"],
    feature_values: Float[Array, " n_features"],
    active_mask: Array,
    step_size_output: float | Array,
    active_count: float | Array,
) -> FutureUtilityEstimate:
    """Estimate utility, rejecting non-finite active inputs atomically.

    For output LMS,
    ``delta_w_ij = alpha * error_i * feature_j / active_count``.  If the same
    feature value recurred immediately, feature ``j`` would change prediction
    ``i`` by ``delta_y_ij = delta_w_ij * feature_j``.  The counterfactual
    reduction in ``0.5 * error_i ** 2`` is therefore
    ``error_i * delta_y_ij - 0.5 * delta_y_ij ** 2``.

    Negative reductions indicate overshoot and are clipped to zero because this
    function is used as a utility signal, not as a signed optimizer diagnostic.

    Args:
        errors: Current prediction residuals, with inactive tasks already zero.
        feature_values: Current active or candidate feature values.
        active_mask: Boolean mask of tasks observed at this step.
        step_size_output: Effective output-weight step size.
        active_count: Number of active tasks, lower-bounded by the caller.

    Returns:
        Non-negative per-task/per-feature predicted loss reductions with
        inactive tasks masked to zero.
    """
    step_size = jnp.asarray(step_size_output, dtype=jnp.float32)
    count = jnp.asarray(active_count, dtype=jnp.float32)
    delta_prediction = (
        step_size
        * errors[:, None]
        * (feature_values[None, :] ** 2)
        / jnp.maximum(count, 1.0)
    )
    reduction = errors[:, None] * delta_prediction - 0.5 * delta_prediction**2
    reduction = jnp.maximum(reduction, 0.0)
    proposed = jnp.where(active_mask[:, None], reduction, 0.0)
    inputs_valid = (
        jnp.all(jnp.where(active_mask, jnp.isfinite(errors), True))
        & jnp.all(jnp.isfinite(feature_values))
        & jnp.isfinite(step_size)
        & (step_size >= 0.0)
        & jnp.isfinite(count)
        & (count >= 0.0)
    )
    update_applied = inputs_valid & jnp.all(jnp.isfinite(proposed))
    return FutureUtilityEstimate(
        reductions=jnp.where(update_applied, proposed, jnp.zeros_like(proposed)),
        update_applied=update_applied,
    )


def one_step_output_loss_reduction(
    errors: Float[Array, " n_tasks"],
    feature_values: Float[Array, " n_features"],
    active_mask: Array,
    step_size_output: float | Array,
    active_count: float | Array,
) -> Float[Array, "n_tasks n_features"]:
    """Compatibility wrapper returning only the fail-closed reductions."""
    return one_step_output_loss_reduction_with_diagnostics(
        errors,
        feature_values,
        active_mask,
        step_size_output,
        active_count,
    ).reductions


def contribution_trace_output_loss_reduction_with_diagnostics(
    errors: Float[Array, " n_tasks"],
    feature_values: Float[Array, " n_features"],
    active_mask: Array,
    step_size_output: float | Array,
    active_count: float | Array,
    contribution_trace: Float[Array, "n_tasks n_features"],
    feature_energy_trace: Float[Array, " n_features"],
    trace_decay: float | Array,
) -> ContributionTraceFutureUtilityEstimate:
    """Estimate delayed usefulness from a TD(lambda)-style contribution trace.

    This variant traces the actual per-task/per-feature output contribution
    ``error_i * phi_j`` instead of maintaining separate residual and feature
    traces.  It is causal and temporally uniform: the trace is updated from
    the current sample and previous trace only, with no hindsight ablation or
    future labels.  With ``trace_decay == 0`` it is exactly the one-step LMS
    counterfactual in :func:`one_step_output_loss_reduction`.

    Args:
        errors: Current prediction residuals, with inactive tasks already zero.
        feature_values: Current active or candidate feature values.
        active_mask: Boolean mask of tasks observed at this step.
        step_size_output: Effective output-weight step size.
        active_count: Number of active tasks, lower-bounded by the caller.
        contribution_trace: Previous trace of ``error_i * phi_j``.
        feature_energy_trace: Previous discounted squared-feature trace.
        trace_decay: Discount applied to traces before adding the current step.

    Returns:
        ``(reductions, new_contribution_trace, new_feature_energy_trace)``.
    """
    decay = jnp.asarray(trace_decay, dtype=jnp.float32)
    step_size = jnp.asarray(step_size_output, dtype=jnp.float32)
    count = jnp.asarray(active_count, dtype=jnp.float32)

    active_errors = jnp.where(active_mask, errors, 0.0)
    decayed_contribution = _skip_zero_scale(decay, contribution_trace)
    new_contribution_trace = (
        decayed_contribution + active_errors[:, None] * feature_values[None, :]
    )
    new_contribution_trace = jnp.where(
        active_mask[:, None],
        new_contribution_trace,
        decayed_contribution,
    )
    new_feature_energy_trace = (
        _skip_zero_scale(decay, feature_energy_trace) + feature_values**2
    )

    delta_weight = (
        step_size
        * active_errors[:, None]
        * feature_values[None, :]
        / jnp.maximum(count, 1.0)
    )
    reduction = (
        delta_weight * new_contribution_trace
        - 0.5 * (delta_weight**2) * new_feature_energy_trace[None, :]
    )
    reduction = jnp.maximum(reduction, 0.0)
    proposed_reductions = jnp.where(active_mask[:, None], reduction, 0.0)
    inputs_valid = (
        jnp.all(jnp.where(active_mask, jnp.isfinite(errors), True))
        & jnp.all(jnp.isfinite(feature_values))
        & jnp.logical_or(
            decay == 0.0,
            jnp.all(jnp.isfinite(contribution_trace))
            & jnp.all(jnp.isfinite(feature_energy_trace)),
        )
        & jnp.isfinite(decay)
        & (decay >= 0.0)
        & (decay <= 1.0)
        & jnp.isfinite(step_size)
        & (step_size >= 0.0)
        & jnp.isfinite(count)
        & (count >= 0.0)
    )
    candidate_finite = (
        jnp.all(jnp.isfinite(proposed_reductions))
        & jnp.all(jnp.isfinite(new_contribution_trace))
        & jnp.all(jnp.isfinite(new_feature_energy_trace))
    )
    update_applied = inputs_valid & candidate_finite
    return ContributionTraceFutureUtilityEstimate(
        reductions=jnp.where(
            update_applied, proposed_reductions, jnp.zeros_like(proposed_reductions)
        ),
        contribution_trace=jnp.where(
            update_applied, new_contribution_trace, contribution_trace
        ),
        feature_energy_trace=jnp.where(
            update_applied, new_feature_energy_trace, feature_energy_trace
        ),
        update_applied=update_applied,
    )


def contribution_trace_output_loss_reduction(
    errors: Float[Array, " n_tasks"],
    feature_values: Float[Array, " n_features"],
    active_mask: Array,
    step_size_output: float | Array,
    active_count: float | Array,
    contribution_trace: Float[Array, "n_tasks n_features"],
    feature_energy_trace: Float[Array, " n_features"],
    trace_decay: float | Array,
) -> tuple[
    Float[Array, "n_tasks n_features"],
    Float[Array, "n_tasks n_features"],
    Float[Array, " n_features"],
]:
    """Compatibility wrapper returning fail-closed estimates and traces."""
    result = contribution_trace_output_loss_reduction_with_diagnostics(
        errors,
        feature_values,
        active_mask,
        step_size_output,
        active_count,
        contribution_trace,
        feature_energy_trace,
        trace_decay,
    )
    return result.reductions, result.contribution_trace, result.feature_energy_trace


def trace_output_loss_reduction_with_diagnostics(
    errors: Float[Array, " n_tasks"],
    feature_values: Float[Array, " n_features"],
    active_mask: Array,
    step_size_output: float | Array,
    active_count: float | Array,
    error_trace: Float[Array, " n_tasks"],
    feature_trace: Float[Array, " n_features"],
    feature_energy_trace: Float[Array, " n_features"],
    trace_decay: float | Array,
) -> TraceFutureUtilityEstimate:
    """Estimate temporally extended output-loss reduction with causal traces.

    The one-step estimator asks how much the current squared error would drop
    if the same feature value recurred immediately.  This variant keeps
    discounted traces of recent residuals, feature values, and feature energy,
    then evaluates the same current LMS output-weight update against that
    recurring-context proxy:

    ``sum_h decay^h e_{t-h} phi_{j,t-h}``.

    This is not a hindsight ablation and it does not peek at future samples.
    It is a causal provenance signal: features that repeatedly co-occur with
    the same residual direction receive more credit than one-step spikes.
    With ``trace_decay == 0`` it reduces exactly to
    :func:`one_step_output_loss_reduction`.

    Args:
        errors: Current prediction residuals, with inactive tasks already zero.
        feature_values: Current active or candidate feature values.
        active_mask: Boolean mask of tasks observed at this step.
        step_size_output: Effective output-weight step size.
        active_count: Number of active tasks, lower-bounded by the caller.
        error_trace: Previous discounted residual trace per task.
        feature_trace: Previous discounted feature-value trace.
        feature_energy_trace: Previous discounted squared-feature trace.
        trace_decay: Discount applied to traces before adding the current step.

    Returns:
        A tuple ``(reductions, new_error_trace, new_feature_trace,
        new_feature_energy_trace)``.  ``reductions`` is non-negative and masked
        for inactive tasks.
    """
    decay = jnp.asarray(trace_decay, dtype=jnp.float32)
    step_size = jnp.asarray(step_size_output, dtype=jnp.float32)
    count = jnp.asarray(active_count, dtype=jnp.float32)

    active_errors = jnp.where(active_mask, errors, 0.0)
    decayed_error = _skip_zero_scale(decay, error_trace)
    new_error_trace = decayed_error + active_errors
    new_error_trace = jnp.where(active_mask, new_error_trace, decayed_error)
    new_feature_trace = _skip_zero_scale(decay, feature_trace) + feature_values
    new_feature_energy_trace = (
        _skip_zero_scale(decay, feature_energy_trace) + feature_values**2
    )

    delta_weight = (
        step_size
        * active_errors[:, None]
        * feature_values[None, :]
        / jnp.maximum(count, 1.0)
    )
    recurring_error_feature = new_error_trace[:, None] * new_feature_trace[None, :]
    recurring_feature_energy = new_feature_energy_trace[None, :]
    reduction = (
        delta_weight * recurring_error_feature
        - 0.5 * (delta_weight**2) * recurring_feature_energy
    )
    reduction = jnp.maximum(reduction, 0.0)
    proposed_reductions = jnp.where(active_mask[:, None], reduction, 0.0)
    inputs_valid = (
        jnp.all(jnp.where(active_mask, jnp.isfinite(errors), True))
        & jnp.all(jnp.isfinite(feature_values))
        & jnp.logical_or(
            decay == 0.0,
            jnp.all(jnp.isfinite(error_trace))
            & jnp.all(jnp.isfinite(feature_trace))
            & jnp.all(jnp.isfinite(feature_energy_trace)),
        )
        & jnp.isfinite(decay)
        & (decay >= 0.0)
        & (decay <= 1.0)
        & jnp.isfinite(step_size)
        & (step_size >= 0.0)
        & jnp.isfinite(count)
        & (count >= 0.0)
    )
    candidate_finite = (
        jnp.all(jnp.isfinite(proposed_reductions))
        & jnp.all(jnp.isfinite(new_error_trace))
        & jnp.all(jnp.isfinite(new_feature_trace))
        & jnp.all(jnp.isfinite(new_feature_energy_trace))
    )
    update_applied = inputs_valid & candidate_finite
    return TraceFutureUtilityEstimate(
        reductions=jnp.where(
            update_applied, proposed_reductions, jnp.zeros_like(proposed_reductions)
        ),
        error_trace=jnp.where(update_applied, new_error_trace, error_trace),
        feature_trace=jnp.where(update_applied, new_feature_trace, feature_trace),
        feature_energy_trace=jnp.where(
            update_applied, new_feature_energy_trace, feature_energy_trace
        ),
        update_applied=update_applied,
    )


def trace_output_loss_reduction(
    errors: Float[Array, " n_tasks"],
    feature_values: Float[Array, " n_features"],
    active_mask: Array,
    step_size_output: float | Array,
    active_count: float | Array,
    error_trace: Float[Array, " n_tasks"],
    feature_trace: Float[Array, " n_features"],
    feature_energy_trace: Float[Array, " n_features"],
    trace_decay: float | Array,
) -> tuple[
    Float[Array, "n_tasks n_features"],
    Float[Array, " n_tasks"],
    Float[Array, " n_features"],
    Float[Array, " n_features"],
]:
    """Compatibility wrapper returning fail-closed estimates and traces."""
    result = trace_output_loss_reduction_with_diagnostics(
        errors,
        feature_values,
        active_mask,
        step_size_output,
        active_count,
        error_trace,
        feature_trace,
        feature_energy_trace,
        trace_decay,
    )
    return (
        result.reductions,
        result.error_trace,
        result.feature_trace,
        result.feature_energy_trace,
    )


def normalize_future_utility_signal(
    signal: Float[Array, " n_features"],
    ages: Array,
    second_moment: Float[Array, " n_features"],
    moment_decay: float | Array,
    utility_decay: float | Array,
    mode: str,
) -> tuple[Float[Array, " n_features"], Float[Array, " n_features"]]:
    """Apply optional causal uncertainty normalization to utility signals.

    Age correction is deliberately *not* applied to the signal entering the
    utility EMA.  Scaling every increment compounds correction already retained
    in the EMA.  Call :func:`bias_correct_future_utility` on the resulting raw
    EMA when it is ranked or reported instead.  ``"uncertainty"`` and
    ``"uncertainty_age"`` divide the current signal by an online RMS, favoring
    consistent usefulness over rare spikes.  The second moment is updated from
    the current signal only, so this remains causal.

    ``ages`` and ``utility_decay`` remain in this established helper surface so
    existing callers and serialized configurations remain compatible.
    """
    del ages, utility_decay
    decay = jnp.asarray(moment_decay, dtype=jnp.float32)
    new_second_moment = (
        _skip_zero_scale(decay, second_moment) + (1.0 - decay) * signal**2
    )
    normalized = signal

    if type(mode) is str and mode in {"uncertainty", "uncertainty_age"}:
        normalized = normalized / jnp.sqrt(new_second_moment + 1e-6)

    return normalized, new_second_moment


def bias_correct_future_utility(
    utilities: Float[Array, " n_features"],
    ages: Array,
    utility_decay: float | Array,
    mode: str,
) -> Float[Array, " n_features"]:
    """Return rank/report scores from a standard raw utility EMA.

    ``ages`` is the number of EMA updates since the slot's last reset.  For
    age-normalized modes, dividing the accumulated EMA by
    ``1 - utility_decay**ages`` removes only its initialization bias.  The raw
    EMA remains in learner state, preserving the existing fixed-shape state and
    checkpoint schemas.  Age-zero slots have no observations and retain their
    raw value (normally zero).
    """
    if type(mode) is not str or mode not in {"age", "uncertainty_age"}:
        return utilities

    age_count = jnp.maximum(ages.astype(jnp.float32), 0.0)
    decay = jnp.asarray(utility_decay, dtype=utilities.dtype)
    debias = 1.0 - jnp.power(decay, age_count)
    safe_debias = jnp.maximum(
        debias,
        jnp.asarray(1e-30, dtype=utilities.dtype),
    )
    corrected = utilities / safe_debias
    return jnp.where(age_count > 0.0, corrected, utilities)
