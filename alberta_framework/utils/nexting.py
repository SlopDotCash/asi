"""Multi-timescale nexting evaluation harness for GVF Horde predictions.

Implements the "nexting" evaluation protocol of Modayil, White, Sutton (2014),
"Multi-timescale nexting in a reinforcement learning robot": predict each
cumulant channel at multiple discount factors gamma, and compare each
prediction at time t against the empirical forward-view return computed
from the actual trajectory.

The empirical forward-view return for cumulant c with discount gamma is::

    G_t = sum_{k=0..} gamma^k * c_{t+k+1}

For finite trajectories this is computed by reverse-cumulative-sum::

    G_T = c_T
    G_t = c_{t+1} + gamma * G_{t+1}     for t < T

This module provides:
- ``forward_view_returns`` : compute G_t for a cumulant series at a fixed gamma
- ``multi_horizon_returns`` : compute G_t at several gammas in one pass
- ``per_horizon_rmse`` : aggregate per-step prediction-vs-return errors

These are the gold-standard targets that any temporal GVF prediction
algorithm should track. The harness is JAX-compatible so it can be used
inside scan loops or compared against scan-collected predictions.

References:
    Modayil, J., White, A., & Sutton, R.S. (2014).
    Multi-timescale nexting in a reinforcement learning robot.
    Adaptive Behavior 22(2), pp. 146-160.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

# Documented public trajectory last-fit (README / package ``__init__`` 10_000).
# Origin reverse-scanned T=20_000 with no reject.
_NEXTING_MAX_STEPS = 10_000
_NEXTING_MAX_HORIZONS = 16
_NEXTING_MAX_CHANNELS = 8


def _is_bool(value: object) -> bool:
    return type(value) is bool or type(value) is np.bool_


def _concrete_numeric_array(
    name: str,
    value: object,
    *,
    ndim: int,
) -> np.ndarray | None:
    """Return a concrete numeric host view, or ``None`` for a JAX tracer."""
    if isinstance(value, jax.core.Tracer):
        if value.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}-dimensional")
        if jnp.issubdtype(value.dtype, jnp.bool_):
            raise ValueError(f"{name} must not be boolean")
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real-valued")
    return array


def _require_host_discount[T](name: str, value: T, *, ndim: int) -> T:
    """Reject boolean / non-finite host discounts before they become 0/1 identities."""
    if _is_bool(value):
        raise ValueError(f"{name} must be a finite discount in [0, 1], not a boolean")
    array = _concrete_numeric_array(name, value, ndim=ndim)
    if array is None:
        return value
    if array.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite discount in [0, 1], not a boolean")
    if not bool(np.all(np.isfinite(array))) or not bool(
        np.all((array >= 0.0) & (array <= 1.0))
    ):
        raise ValueError(f"{name} must be a finite discount in [0, 1]")
    return value


def _require_leading_length(
    name: str,
    value: object,
    *,
    ndim: int,
    maximum: int,
) -> None:
    if isinstance(value, jax.core.Tracer):
        if value.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}-dimensional")
        return
    if isinstance(value, jax.Array):
        if value.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}-dimensional")
        length = int(value.shape[0])
    else:
        array = _concrete_numeric_array(name, value, ndim=ndim)
        if array is None:
            return
        length = int(array.shape[0])
    if length < 1 or length > maximum:
        raise ValueError(f"{name} length must be an integer in [1, {maximum}]")


def _require_host_finite_real[T](name: str, value: T) -> T:
    """Reject boolean / non-finite host scalars before they become 0/1 bootstraps."""
    if _is_bool(value):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    array = _concrete_numeric_array(name, value, ndim=0)
    if array is None:
        return value
    if array.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    if not bool(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real number")
    return value


def forward_view_returns(
    cumulants: Float[Array, " T"],
    gamma: float | Float[Array, ""],
    terminal_value: float = 0.0,
) -> Float[Array, " T"]:
    """Compute the forward-view return G_t for each step of a cumulant series.

    Uses the recursion ``G_t = c_{t+1} + gamma * G_{t+1}`` with terminal
    condition ``G_T = terminal_value`` (default 0). Implemented via
    ``jax.lax.scan`` over reversed indices for JIT efficiency.

    Args:
        cumulants: 1-D series of cumulants with length ``T`` -- the
            cumulant emitted at step ``t+1`` of a trajectory.
        gamma: Constant discount factor (or 0-D array).
        terminal_value: Value at the post-terminal step (default 0.0).

    Returns:
        Array of shape ``(T,)`` where index ``t`` is the forward-view
        return ``G_t = c_{t+1} + gamma * c_{t+2} + gamma^2 * c_{t+3} + ...``.
    """
    gamma = _require_host_discount("gamma", gamma, ndim=0)
    terminal_value = _require_host_finite_real("terminal_value", terminal_value)
    _require_leading_length(
        "cumulants", cumulants, ndim=1, maximum=_NEXTING_MAX_STEPS
    )
    # The computation dtype is promoted across all three operands so an
    # integer or boolean cumulant series yields float returns instead of
    # truncating gamma / terminal_value into the series dtype (which would
    # collapse the bootstrap to 0 and turn returns into a raw cumulant echo).
    result_dtype = jnp.promote_types(
        jnp.promote_types(cumulants.dtype, jnp.asarray(gamma).dtype),
        jnp.asarray(terminal_value).dtype,
    )
    if not jnp.issubdtype(result_dtype, jnp.floating):
        result_dtype = jnp.float32
    gamma_s = jnp.asarray(gamma, dtype=result_dtype)
    init = jnp.asarray(terminal_value, dtype=result_dtype)
    series = cumulants.astype(result_dtype)

    def step(carry: Array, c: Array) -> tuple[Array, Array]:
        # gamma=0 must not multiply an inf later return (0*inf).
        bootstrap = jnp.where(gamma_s == 0.0, jnp.zeros_like(carry), gamma_s * carry)
        new_carry = c + bootstrap
        return new_carry, new_carry

    _, returns_reversed = jax.lax.scan(step, init, series[::-1])
    return returns_reversed[::-1]


def multi_horizon_returns(
    cumulants: Float[Array, " T"],
    gammas: Float[Array, " H"],
    terminal_value: float = 0.0,
) -> Float[Array, "T H"]:
    """Compute forward-view returns for one cumulant at several discount factors.

    Equivalent to running ``forward_view_returns`` once per gamma but
    batched via ``jax.vmap``.

    Args:
        cumulants: 1-D series of cumulants, length ``T``.
        gammas: 1-D array of discount factors, length ``H``.
        terminal_value: Post-terminal G value (default 0.0).

    Returns:
        Array of shape ``(T, H)`` -- ``[t, h]`` is the forward-view return
        from step ``t`` at horizon ``gammas[h]``.
    """
    gammas = _require_host_discount("gammas", gammas, ndim=1)
    terminal_value = _require_host_finite_real("terminal_value", terminal_value)
    _require_leading_length(
        "cumulants", cumulants, ndim=1, maximum=_NEXTING_MAX_STEPS
    )
    _require_leading_length("gammas", gammas, ndim=1, maximum=_NEXTING_MAX_HORIZONS)

    def per_gamma(g: Array) -> Array:
        return forward_view_returns(cumulants, g, terminal_value=terminal_value)

    return jax.vmap(per_gamma, out_axes=1)(gammas)


def multi_channel_horizon_returns(
    cumulants: Float[Array, "T C"],
    gammas: Float[Array, " H"],
    terminal_value: float = 0.0,
) -> Float[Array, "T C H"]:
    """Compute forward-view returns for ``C`` cumulant channels at ``H`` horizons.

    Args:
        cumulants: 2-D array of cumulants, shape ``(T, C)``.
        gammas: 1-D discount factors, shape ``(H,)``.
        terminal_value: Post-terminal G value (default 0.0).

    Returns:
        Array of shape ``(T, C, H)`` of forward-view returns.
    """
    _require_leading_length(
        "cumulants", cumulants, ndim=2, maximum=_NEXTING_MAX_STEPS
    )
    if isinstance(cumulants, jax.Array) and not isinstance(cumulants, jax.core.Tracer):
        channel_count = int(cumulants.shape[1])
        if channel_count < 1 or channel_count > _NEXTING_MAX_CHANNELS:
            raise ValueError(
                "cumulants channel count must be an integer in "
                f"[1, {_NEXTING_MAX_CHANNELS}]"
            )
    _require_leading_length("gammas", gammas, ndim=1, maximum=_NEXTING_MAX_HORIZONS)
    # Vmap over channels: for each channel apply multi_horizon_returns
    def per_channel(c_series: Array) -> Array:
        return multi_horizon_returns(c_series, gammas, terminal_value=terminal_value)

    return jax.vmap(per_channel, in_axes=1, out_axes=1)(cumulants)


def _validate_rmse_inputs(
    predictions: Float[Array, "T H"],
    forward_returns: Float[Array, "T H"],
) -> int:
    if (
        predictions.ndim != 2
        or forward_returns.ndim != 2
        or predictions.shape != forward_returns.shape
        or 0 in predictions.shape
    ):
        raise ValueError(
            "predictions and forward_returns must be rank-2 arrays with identical "
            "nonempty shape (T, H) "
            f"(got predictions.shape={predictions.shape}, "
            f"forward_returns.shape={forward_returns.shape})"
        )
    return predictions.shape[0]


def _scaled_rmse_terms(errors: Array) -> tuple[Array, Array, Array]:
    """Return power-of-two scale, scaled errors, and their RMS."""
    scale = jnp.max(jnp.abs(errors), axis=0)
    _, exponent = jnp.frexp(scale)
    scaled_errors = jnp.ldexp(errors, -exponent)
    scaled_rmse = jnp.sqrt(jnp.mean(scaled_errors**2, axis=0))
    return exponent, scaled_errors, scaled_rmse


@jax.custom_jvp
def _finite_rmse(errors: Array) -> Array:
    """Compute finite-column RMSE without overflow or reciprocal underflow."""
    exponent, _, scaled_rmse = _scaled_rmse_terms(errors)
    return jnp.ldexp(scaled_rmse, exponent)


@_finite_rmse.defjvp
def _finite_rmse_jvp(
    primals: tuple[Array],
    tangents: tuple[Array],
) -> tuple[Array, Array]:
    """Differentiate in the scaled domain so huge finite gradients stay valid."""
    (errors,), (errors_dot,) = primals, tangents
    exponent, scaled_errors, scaled_rmse = _scaled_rmse_terms(errors)
    rmse = jnp.ldexp(scaled_rmse, exponent)
    # Zero is a valid subgradient for an all-zero column and avoids a 0 / 0.
    safe_scaled_rmse = jnp.where(
        scaled_rmse > 0.0,
        scaled_rmse,
        jnp.ones_like(scaled_rmse),
    )
    rmse_dot = jnp.mean(scaled_errors * errors_dot, axis=0) / safe_scaled_rmse
    return rmse, rmse_dot


def _sliding_sum(values: Array, window_size: int) -> Array:
    prefix = jnp.cumsum(
        jnp.concatenate(
            [jnp.zeros((1, values.shape[1]), dtype=values.dtype), values],
            axis=0,
        ),
        axis=0,
    )
    return prefix[window_size:] - prefix[:-window_size]


def _pad_running_window(values: Array, window_size: int) -> Array:
    pad = jnp.broadcast_to(values[0], (window_size - 1, values.shape[1]))
    return jnp.concatenate([pad, values], axis=0)


def _scaled_running_rmse_terms(
    errors: Array, window_size: int
) -> tuple[Array, Array, Array]:
    """Return one global power-of-two scale and stable sliding RMS terms."""
    scale = jnp.max(jnp.abs(errors), axis=0)
    _, exponent = jnp.frexp(scale)
    scaled_errors = jnp.ldexp(errors, -exponent)
    scaled_window = _sliding_sum(scaled_errors**2, window_size)
    scaled_running = jnp.sqrt(scaled_window / window_size)
    return exponent, scaled_errors, scaled_running


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _finite_running_rmse(errors: Array, window_size: int) -> Array:
    """Compute a finite sliding RMSE with a scale-free derivative."""
    exponent, _, scaled_running = _scaled_running_rmse_terms(errors, window_size)
    return _pad_running_window(jnp.ldexp(scaled_running, exponent), window_size)


@_finite_running_rmse.defjvp
def _finite_running_rmse_jvp(
    window_size: int,
    primals: tuple[Array],
    tangents: tuple[Array],
) -> tuple[Array, Array]:
    (errors,), (errors_dot,) = primals, tangents
    exponent, scaled_errors, scaled_running = _scaled_running_rmse_terms(
        errors, window_size
    )
    running = _pad_running_window(
        jnp.ldexp(scaled_running, exponent), window_size
    )
    safe_scaled_running = jnp.where(
        scaled_running > 0.0,
        scaled_running,
        jnp.ones_like(scaled_running),
    )
    numerator = _sliding_sum(scaled_errors * errors_dot, window_size) / window_size
    running_dot = _pad_running_window(
        numerator / safe_scaled_running,
        window_size,
    )
    return running, running_dot


def per_horizon_rmse(
    predictions: Float[Array, "T H"],
    forward_returns: Float[Array, "T H"],
    burn_in: int = 0,
) -> Float[Array, " H"]:
    """Per-horizon root-mean-squared error of predictions vs forward-view returns.

    Args:
        predictions: Predictions over time at each horizon, shape ``(T, H)``.
        forward_returns: Ground-truth forward-view returns, shape ``(T, H)``.
        burn_in: Exact built-in ``int`` number of initial steps to skip (helps
            when the learner has not yet warmed up). Defaults to 0. When JIT
            compiling a non-default value, close over it or mark ``burn_in``
            static with ``static_argnames``.

    Returns:
        Array of shape ``(H,)`` with RMSE per horizon.

    Raises:
        ValueError: If the arrays do not have identical nonempty ``(T, H)``
            shapes, or if ``burn_in`` is not an exact built-in ``int`` in
            ``[0, T)``.
    """
    if type(burn_in) is not int:
        raise ValueError("burn_in must be a built-in int")
    n_steps = _validate_rmse_inputs(predictions, forward_returns)
    if not 0 <= burn_in < n_steps:
        raise ValueError(
            f"burn_in must satisfy 0 <= burn_in < n_steps "
            f"(got burn_in={burn_in}, n_steps={n_steps})"
        )
    if burn_in:
        predictions = predictions[burn_in:]
        forward_returns = forward_returns[burn_in:]
    errors = predictions - forward_returns

    # Scale before squaring so large, finite float32 errors do not overflow.
    # Mask non-finite columns before every arithmetic operation, and divide by
    # one for all-zero columns.  This keeps debug/checkify modes free of
    # discarded overflow and 0/0 operations while retaining NaN/Inf diagnostics
    # in the final result.
    finite_columns = jnp.all(jnp.isfinite(errors), axis=0)
    safe_errors = jnp.where(finite_columns[None, :], errors, jnp.zeros_like(errors))
    # Arbitrary division by a near-max float32 scale can lower to a reciprocal
    # that underflows to zero.  ldexp performs exact power-of-two scaling and
    # keeps the normalized magnitudes in range without that reciprocal.
    stable_rmse = _finite_rmse(safe_errors)
    # A maximum absolute error carries the desired diagnostic: NaN if any
    # error is NaN, otherwise Inf if any error is infinite.  Deriving it from
    # the input avoids constructing discarded NaN/Inf literals on finite calls.
    nonfinite_rmse = jnp.max(jnp.abs(errors), axis=0)
    return jnp.where(finite_columns, stable_rmse, nonfinite_rmse)


def per_horizon_running_rmse(
    predictions: Float[Array, "T H"],
    forward_returns: Float[Array, "T H"],
    window_size: int = 100,
) -> Float[Array, "T H"]:
    """Per-horizon running RMSE over a sliding window.

    Useful for plotting how prediction accuracy evolves through a non-
    stationary stream (e.g. through Pavlovian phase transitions).

    Args:
        predictions: Shape ``(T, H)``.
        forward_returns: Shape ``(T, H)``.
        window_size: Exact built-in ``int`` length of the trailing average
            window. When JIT compiling a non-default value, close over it or
            mark ``window_size`` static with ``static_argnames``.

    Returns:
        Array of shape ``(T, H)``. The first ``window_size - 1`` rows are
        equal to ``running_rmse[window_size - 1]``.

    Raises:
        ValueError: If the arrays do not have identical nonempty ``(T, H)``
            shapes, or if ``window_size`` is not an exact built-in ``int`` in
            ``[1, T]``.
    """
    if type(window_size) is not int:
        raise ValueError("window_size must be a built-in int")
    n_steps = _validate_rmse_inputs(predictions, forward_returns)
    if not 1 <= window_size <= n_steps:
        raise ValueError(
            f"window_size must satisfy 1 <= window_size <= n_steps "
            f"(got window_size={window_size}, n_steps={n_steps})"
        )
    errors = predictions - forward_returns
    finite_columns = jnp.all(jnp.isfinite(errors), axis=0)
    safe_errors = jnp.where(finite_columns[None, :], errors, jnp.zeros_like(errors))
    stable_running = _finite_running_rmse(safe_errors, window_size)
    nonfinite = jnp.broadcast_to(
        jnp.max(jnp.abs(errors), axis=0),
        stable_running.shape,
    )
    return jnp.where(finite_columns[None, :], stable_running, nonfinite)
