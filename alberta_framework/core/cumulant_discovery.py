"""Surprise-driven cumulant discovery for Horde demons.

Background
==========
Horde (Sutton et al. 2011) requires a hand-curated set of GVF demons --
each demon's cumulant, gamma, lambda, and target policy are specified up
front. For Step 3 to scale beyond hand-engineered question sets, we need
**discovery**: a mechanism that proposes candidate cumulants, evaluates
their utility, keeps the best, and discards the rest.

This module implements the simplest practical discovery method --
**surprise-driven retention**. Candidates are random projections of the
input observation; each candidate has a "demon" attached as a
single-output ``LinearLearner``-style predictor; candidate utility is the
EMA of squared TD error (the "surprise"). Periodically we replace the
lowest-utility candidates with new random projections.

Why squared TD error? A TD error of zero means the demon has accurately
predicted its cumulant; the cumulant is therefore *not informative*
about future dynamics. Demons with persistent positive squared TD error
are predicting a signal that is hard to know. (Cf. White, Modayil &
Sutton 2014, "Surprise as an intrinsic motivation for hierarchical RL.")
Caveat: the converse selection also happens -- a cumulant that is
irreducible noise sustains high squared TD error forever, so this
utility cannot separate hard-but-learnable structure from unpredictable
noise. Distinguishing the two needs a learning-progress signal (error
*decrease* over time) rather than error magnitude.

Limitations
===========
This is intentionally minimal:
- Uses ``OffPolicyTDLinearLearner`` per candidate (linear, on-policy when
  rho=1) so the candidate predictor has no learned features beyond the
  raw observation.
- Random projections are the cheapest possible cumulant generator.
- Retains the K highest-utility candidates; it has no shadow or promotion
  logic like Step 2's interaction features.
- It does not implement Veeriah-style meta-gradient discovery: cumulant
  parameters are not optimized against a downstream task loss.

The output of discovery is a tuple ``(active_cumulants, utilities)`` that
can be plugged into a downstream Horde. Survival means only that a candidate
ranked highly under this module's surprise score; it is not evidence of
downstream utility.

Reference:
    White, A., Modayil, J., & Sutton, R.S. (2014). "Surprise as an
    intrinsic motivation for hierarchical reinforcement learning."
    Veeriah, V., et al. (2019). "Discovery of Useful Questions as
    Auxiliary Tasks." NeurIPS.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio

_INT32_MAX = 2**31 - 1
_MAX_PERSISTENT_STATE_BYTES = 256 * 1024 * 1024
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"))}
)
_ACTUAL_REAL_TYPES = _ACTUAL_INT_TYPES | frozenset(
    {float, *(np.dtype(code).type for code in "efdg")}
)


def _require_int32(name: str, value: object, *, minimum: int) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return canonical


def _require_float32(
    name: str,
    value: object,
    *,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
    positive: bool = False,
    preserve_nonzero: bool = False,
) -> float:
    if type(value) not in _ACTUAL_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    stored, numerator, _ = validated_float32_scalar_with_ratio(
        name,
        value,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
        positive=positive,
    )
    if preserve_nonzero and numerator != 0 and stored == 0.0:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _persistent_resources(raw_dim: int, n_candidates: int) -> dict[str, int]:
    """Exact retained-array budget, excluding compiler and transient buffers."""
    persistent_scalars = 2 * n_candidates * raw_dim + 3 * n_candidates + 2
    persistent_bytes = 4 * persistent_scalars
    if persistent_scalars > _INT32_MAX:
        raise ValueError("derived cumulant-discovery persistent scalars must fit signed int32")
    if persistent_bytes > _MAX_PERSISTENT_STATE_BYTES:
        raise ValueError("derived cumulant-discovery persistent state exceeds 256 MiB")
    return {
        "persistent_scalars": persistent_scalars,
        "persistent_bytes": persistent_bytes,
    }

# =============================================================================
# State
# =============================================================================


@chex.dataclass(frozen=True)
class CumulantDiscoveryState:
    """State for surprise-driven cumulant discovery.

    Attributes:
        projections: ``(n_candidates, raw_dim)`` random projection matrix
            -- each row defines one candidate cumulant ``c_i = w_i . obs``.
        weights: ``(n_candidates, raw_dim)`` per-candidate predictor weights.
        biases: ``(n_candidates,)`` per-candidate predictor biases.
        utility: ``(n_candidates,)`` EMA of squared TD error (surprise).
        ages: ``(n_candidates,)`` number of update steps since last reset.
        key: JAX random key for sampling fresh projections on replacement.
    """

    projections: Float[Array, "n_candidates raw_dim"]
    weights: Float[Array, "n_candidates raw_dim"]
    biases: Float[Array, " n_candidates"]
    utility: Float[Array, " n_candidates"]
    ages: Int[Array, " n_candidates"]
    key: PRNGKeyArray


# =============================================================================
# Discovery learner
# =============================================================================


class CumulantDiscovery:
    """Surprise-driven cumulant discovery.

    Maintains ``n_candidates`` parallel candidate cumulants (random
    linear projections of the observation) plus per-candidate linear
    value predictors. After each step, the predictor's TD error is the
    "surprise"; an EMA of squared surprise is the utility. Periodically,
    candidates with the lowest utility (subject to a maturity threshold)
    are replaced with fresh random projections.

    Args:
        raw_dim: Dimension of the raw observation
        n_candidates: Number of cumulant candidates to maintain
        decay_rate: Utility EMA decay (``utility := decay*utility +
            (1-decay)*surprise``). Default 0.99.
        replacement_rate: Per-step probability of replacing the
            lowest-utility candidate (eligible by maturity). Default
            ``5e-3`` (replace ~one candidate every 200 steps).
        maturity_threshold: Minimum age before a candidate may be
            replaced. Default 200 steps.
        predictor_step_size: Per-candidate linear predictor step size.
            Default 0.05.
        gamma: Pseudo-termination discount used for the demon's TD error
            (default 0.0 = supervised next-step prediction). Use
            ``gamma > 0`` to retain candidates whose temporal returns
            are surprising.
        enabled: If False, replacement never occurs (useful for
            ablations).
    """

    def __init__(
        self,
        raw_dim: int,
        n_candidates: int = 16,
        decay_rate: float = 0.99,
        replacement_rate: float = 5e-3,
        maturity_threshold: int = 200,
        predictor_step_size: float = 0.05,
        gamma: float = 0.0,
        enabled: bool = True,
    ):
        raw_dim = _require_int32("raw_dim", raw_dim, minimum=1)
        n_candidates = _require_int32("n_candidates", n_candidates, minimum=1)
        maturity_threshold = _require_int32(
            "maturity_threshold", maturity_threshold, minimum=0
        )
        _persistent_resources(raw_dim, n_candidates)
        decay_rate = _require_float32(
            "decay_rate",
            decay_rate,
            positive=True,
            upper=1.0,
            upper_inclusive=False,
        )
        replacement_rate = _require_float32(
            "replacement_rate",
            replacement_rate,
            lower=0.0,
            upper=1.0,
            preserve_nonzero=True,
        )
        predictor_step_size = _require_float32(
            "predictor_step_size", predictor_step_size, positive=True
        )
        gamma = _require_float32(
            "gamma", gamma, lower=0.0, upper=1.0, preserve_nonzero=True
        )
        if type(enabled) is not bool:
            raise ValueError("enabled must be an exact boolean")

        self._raw_dim = raw_dim
        self._n_candidates = n_candidates
        self._decay_rate = decay_rate
        self._replacement_rate = replacement_rate
        self._maturity_threshold = maturity_threshold
        self._predictor_step_size = predictor_step_size
        self._gamma = gamma
        self._enabled = enabled

    @staticmethod
    def _require_typed_key(name: str, value: object) -> Array:
        """Require a typed scalar Threefry key, never legacy uint32 words."""
        try:
            key = jnp.asarray(value)
            implementation = str(jr.key_impl(value))  # type: ignore[arg-type]
            words = jr.key_data(value)  # type: ignore[arg-type]
        except Exception as error:
            raise ValueError(f"{name} must be a typed scalar threefry2x32 key") from error
        if (
            key.shape != ()
            or implementation != "threefry2x32"
            or words.shape != (2,)
            or words.dtype != jnp.uint32
        ):
            raise ValueError(f"{name} must be a typed scalar threefry2x32 key")
        return key

    def _require_state_contract(self, state: object) -> CumulantDiscoveryState:
        if type(state) is not CumulantDiscoveryState:
            raise ValueError("state must be an actual CumulantDiscoveryState")
        canonical = state
        matrix_shape = (self._n_candidates, self._raw_dim)
        vector_shape = (self._n_candidates,)
        for name in ("projections", "weights"):
            leaf = getattr(canonical, name)
            if leaf.shape != matrix_shape or leaf.dtype != jnp.float32:
                raise ValueError(
                    f"state.{name} must have shape {matrix_shape} and dtype float32"
                )
        for name in ("biases", "utility"):
            leaf = getattr(canonical, name)
            if leaf.shape != vector_shape or leaf.dtype != jnp.float32:
                raise ValueError(
                    f"state.{name} must have shape {vector_shape} and dtype float32"
                )
        if canonical.ages.shape != vector_shape or canonical.ages.dtype != jnp.int32:
            raise ValueError(
                f"state.ages must have shape {vector_shape} and dtype int32"
            )
        self._require_typed_key("state.key", canonical.key)
        return canonical

    def _require_observation(self, name: str, value: object) -> Array:
        try:
            observation = jnp.asarray(value)
        except Exception as error:
            raise ValueError(f"{name} must be a readable float32 vector") from error
        if observation.shape != (self._raw_dim,) or observation.dtype != jnp.float32:
            raise ValueError(
                f"{name} must have shape ({self._raw_dim},) and dtype float32"
            )
        return observation

    @staticmethod
    def _state_values_valid(state: CumulantDiscoveryState) -> Array:
        return (
            jnp.all(jnp.isfinite(state.projections))
            & jnp.all(jnp.isfinite(state.weights))
            & jnp.all(jnp.isfinite(state.biases))
            & jnp.all(jnp.isfinite(state.utility))
            & jnp.all(state.utility >= 0.0)
            & jnp.all(state.ages >= 0)
        )

    @property
    def n_candidates(self) -> int:
        return self._n_candidates

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def persistent_resource_budget(self) -> dict[str, int]:
        """Exact persistent JAX-array envelope."""
        return _persistent_resources(self._raw_dim, self._n_candidates)

    def init(self, key: Array) -> CumulantDiscoveryState:
        """Initialize state with random projections, zero predictors,
        zero utility, zero ages."""
        key = self._require_typed_key("key", key)
        k_proj, k_state = jr.split(key)
        # Unit-norm random projections
        raw_proj = jr.normal(
            k_proj, (self._n_candidates, self._raw_dim), dtype=jnp.float32
        )
        norms = jnp.linalg.norm(raw_proj, axis=1, keepdims=True) + 1e-8
        projections = raw_proj / norms
        return CumulantDiscoveryState(  # type: ignore[call-arg]
            projections=projections,
            weights=jnp.zeros(
                (self._n_candidates, self._raw_dim), dtype=jnp.float32
            ),
            biases=jnp.zeros(self._n_candidates, dtype=jnp.float32),
            utility=jnp.zeros(self._n_candidates, dtype=jnp.float32),
            ages=jnp.zeros(self._n_candidates, dtype=jnp.int32),
            key=k_state,
        )

    def cumulants(
        self,
        state: CumulantDiscoveryState,
        observation: Float[Array, " raw_dim"],
    ) -> Float[Array, " n_candidates"]:
        """Compute the candidate cumulant values for an observation.

        In a GVF update, cumulants are transition signals: the update from
        ``s_t`` to ``s_{t+1}`` should use ``c_{t+1}``, so callers feeding a
        Horde should normally pass the *next* observation here.
        """
        state = self._require_state_contract(state)
        observation = self._require_observation("observation", observation)
        return state.projections @ observation

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: CumulantDiscoveryState,
        observation: Float[Array, " raw_dim"],
        next_observation: Float[Array, " raw_dim"],
    ) -> CumulantDiscoveryState:
        """Apply one update step.

        For each candidate i:
            cumulant_i = projections_i . next_obs
            V_i        = weights_i . obs + bias_i
            V_i_next   = weights_i . next_obs + bias_i
            delta_i    = cumulant_i + gamma * V_i_next - V_i
            weights_i += alpha * delta_i * obs
            bias_i    += alpha * delta_i
            utility_i := decay * utility_i + (1 - decay) * delta_i^2
            ages_i    += 1

        Args:
            state: Current discovery state
            observation: Current raw observation, ``s_t``.
            next_observation: Next raw observation, ``s_{t+1}``; candidate
                cumulants are evaluated on this observation to match the
                nexting/GVF convention ``G_t = c_{t+1} + gamma G_{t+1}``.

        Returns:
            Updated discovery state
        """
        state = self._require_state_contract(state)
        observation = self._require_observation("observation", observation)
        next_observation = self._require_observation(
            "next_observation", next_observation
        )
        alpha = jnp.asarray(self._predictor_step_size, dtype=jnp.float32)
        gamma = jnp.asarray(self._gamma, dtype=jnp.float32)
        decay = jnp.asarray(self._decay_rate, dtype=jnp.float32)

        # Per-candidate transition cumulant c_{t+1} and predictions.
        cumulants = state.projections @ next_observation  # (n,)
        v = state.weights @ observation + state.biases  # (n,)
        v_next = state.weights @ next_observation + state.biases  # (n,)

        td = cumulants + gamma * v_next - v  # (n,)

        # Predictor update (per-candidate semi-gradient TD step)
        # weights_i += alpha * td_i * obs
        proposed_weights = state.weights + alpha * td[:, None] * observation[None, :]
        proposed_biases = state.biases + alpha * td
        proposed_utility = decay * state.utility + (1.0 - decay) * (td**2)
        proposed_state = CumulantDiscoveryState(  # type: ignore[call-arg]
            projections=state.projections,
            weights=proposed_weights,
            biases=proposed_biases,
            utility=proposed_utility,
            ages=jnp.minimum(state.ages, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + 1,
            key=state.key,
        )
        # Inf next obs makes 0 @ inf = NaN in V(s') at zero init, then
        # alpha * nan * obs poisons every candidate. Hold the finite state.
        inputs_valid = jnp.all(jnp.isfinite(observation)) & jnp.all(
            jnp.isfinite(next_observation)
        )
        proposed_finite = (
            jnp.all(jnp.isfinite(proposed_weights))
            & jnp.all(jnp.isfinite(proposed_biases))
            & jnp.all(jnp.isfinite(proposed_utility))
        )
        return cast(
            CumulantDiscoveryState,
            jax.lax.cond(
                self._state_values_valid(state) & inputs_valid & proposed_finite,
                lambda: proposed_state,
                lambda: state,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def maybe_replace(
        self,
        state: CumulantDiscoveryState,
    ) -> CumulantDiscoveryState:
        """Possibly replace the lowest-utility eligible candidate with a
        fresh random projection.

        If ``enabled=False``, returns the input state unchanged.

        On each call, with probability ``replacement_rate``, the
        lowest-utility candidate that has reached maturity is replaced.
        Replacement: re-sample the projection row, zero the predictor
        weights/bias, reset utility to 0 and age to 0.
        """
        state = self._require_state_contract(state)
        if not self._enabled:
            return state

        k_trigger, k_proj, k_next = jr.split(state.key, 3)

        # Sample whether to attempt replacement
        trigger = jr.uniform(k_trigger, ()) < self._replacement_rate

        # Among mature candidates, find the one with lowest utility
        mature = state.ages >= self._maturity_threshold
        # Rank mature candidates by bias-corrected utility to avoid penalizing
        # freshly mature candidates whose raw utility EMA is still warming up.
        safe_ages = jnp.maximum(state.ages.astype(jnp.float32), 1.0)
        debias = 1.0 - jnp.power(jnp.asarray(self._decay_rate, dtype=jnp.float32), safe_ages)
        safe_debias = jnp.maximum(debias, jnp.asarray(1e-30, dtype=jnp.float32))
        ranking_utility = state.utility / safe_debias

        # Disqualify immature candidates by setting their utility to +inf
        masked_utility = jnp.where(
            mature, ranking_utility, jnp.full_like(ranking_utility, jnp.inf)
        )
        # If no candidates are mature, this index is arbitrary
        worst_idx = jnp.argmin(masked_utility)
        any_mature = jnp.any(mature)

        # Fresh projection
        raw_new = jr.normal(k_proj, (self._raw_dim,), dtype=jnp.float32)
        new_proj_row = raw_new / (jnp.linalg.norm(raw_new) + 1e-8)

        do_replace = trigger & any_mature

        new_projections = jnp.where(
            do_replace,
            state.projections.at[worst_idx].set(new_proj_row),
            state.projections,
        )
        new_weights = jnp.where(
            do_replace,
            state.weights.at[worst_idx].set(jnp.zeros(self._raw_dim, dtype=jnp.float32)),
            state.weights,
        )
        new_biases = jnp.where(
            do_replace,
            state.biases.at[worst_idx].set(jnp.float32(0.0)),
            state.biases,
        )
        new_utility = jnp.where(
            do_replace,
            state.utility.at[worst_idx].set(jnp.float32(0.0)),
            state.utility,
        )
        new_ages = jnp.where(
            do_replace,
            state.ages.at[worst_idx].set(jnp.int32(0)),
            state.ages,
        )

        candidate = CumulantDiscoveryState(  # type: ignore[call-arg]
            projections=new_projections,
            weights=new_weights,
            biases=new_biases,
            utility=new_utility,
            ages=new_ages,
            key=k_next,
        )
        commit = self._state_values_valid(state) & self._state_values_valid(candidate)
        return cast(
            CumulantDiscoveryState,
            jax.lax.cond(commit, lambda: candidate, lambda: state),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "CumulantDiscovery",
            "raw_dim": self._raw_dim,
            "n_candidates": self._n_candidates,
            "decay_rate": self._decay_rate,
            "replacement_rate": self._replacement_rate,
            "maturity_threshold": self._maturity_threshold,
            "predictor_step_size": self._predictor_step_size,
            "gamma": self._gamma,
            "enabled": self._enabled,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CumulantDiscovery:
        """Reconstruct from dict."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("config must be a readable mapping") from error
        if any(type(key) is not str for key in payload):
            raise ValueError("config keys must be strings")
        payload.pop("type", None)
        try:
            return cls(**payload)
        except TypeError as error:
            raise ValueError("config has missing or unsupported fields") from error
