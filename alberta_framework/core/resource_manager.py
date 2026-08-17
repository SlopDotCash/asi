"""Learned resource managers for continual feature/plasticity allocation.

The Alberta Plan calls for agents that decide where to spend limited
representation-building effort.  This module provides a small, JAX-friendly
resource manager that learns a causal allocation over discrete resource
policies from online losses.

The manager is intentionally generic: actions can represent generator choices,
replacement rates, perturbation schedules, expert policies, or any other
resource-consuming option.  It does not assume a particular learner.  Each
update receives the current action losses and optional resource costs, then
performs a discounted exponentiated-gradient update — Hedge / exponential
weights (Freund & Schapire 1997) with a preference-decay factor, plus an
optional Exp3-style importance-weighted variant for sampled-policy-only
feedback (Auer et al. 2002).  With ``n_contexts > 1`` the manager learns
separate allocations for externally supplied stream states or inferred
contexts.

References:
    Freund & Schapire (1997). "A Decision-Theoretic Generalization of
        On-Line Learning and an Application to Boosting." (Hedge)
    Auer, Cesa-Bianchi, Freund, & Schapire (2002). "The Nonstochastic
        Multiarmed Bandit Problem." (Exp3)
    Cesa-Bianchi & Lugosi (2006). "Prediction, Learning, and Games."
        (exponential-weights regret bounds)
"""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    safe_discrete_action,
    select_transaction,
)

_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")}
)
_ACTUAL_REAL_TYPES = _ACTUAL_INT_TYPES | frozenset(
    {float, *(np.dtype(code).type for code in "efdg")}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _validated_float32(
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
    preserve_nonzero: bool = False,
) -> float:
    """Validate one trusted concrete scalar in both host and float32 domains."""
    if type(value) not in _ACTUAL_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name,
        value,
        positive=positive,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )
    narrowed = float(np.float32(numerator / denominator))
    if preserve_nonzero and numerator != 0 and narrowed == 0.0:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _require_exact_tuple(name: str, value: object) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an exact tuple")
    return value


def _decode_sequence(name: str, value: object) -> tuple[Any, ...]:
    if type(value) is not list:
        raise ValueError(f"serialized {name} must be an exact JSON list")
    return tuple(value)


def _read_mapping(name: str, value: object) -> dict[str, Any]:
    if not issubclass(type(value), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        return dict(cast(Mapping[str, Any], value))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error


def _require_serialized_fields(
    payload: Mapping[str, Any],
    *,
    owner: str,
    expected: set[str],
    integer_fields: tuple[str, ...] = (),
    float_fields: tuple[str, ...] = (),
    string_fields: tuple[str, ...] = (),
) -> None:
    if set(payload) != expected:
        raise ValueError(f"serialized {owner} config fields do not match its schema")
    for name in integer_fields:
        if type(payload[name]) is not int:
            raise ValueError(f"serialized {name} must be a JSON integer")
    for name in float_fields:
        if type(payload[name]) is not float:
            raise ValueError(f"serialized {name} must be a JSON number")
    for name in string_fields:
        if type(payload[name]) is not str:
            raise ValueError(f"serialized {name} must be a JSON string")


def _require_manager_state_budget(name: str, n_contexts: int, n_choices: int) -> None:
    """Preflight persistent state and conservative update/select working sets."""
    state_scalars = 3 * n_contexts * n_choices + 1
    update_scalars = 3 * state_scalars + 40 * n_choices + 32
    if (
        state_scalars > _INT32_MAX
        or 4 * state_scalars > _INT32_MAX
        or update_scalars > _INT32_MAX
        or 4 * update_scalars > _INT32_MAX
    ):
        raise ValueError(f"{name} state exceeds the signed-int32 scalar/byte budget")


def _require_array_metadata(name: str, value: object, shape: tuple[int, ...], dtype: Any) -> None:
    try:
        actual_shape = tuple(value.shape)  # type: ignore[attr-defined]
        actual_dtype = jnp.dtype(value.dtype)  # type: ignore[attr-defined]
    except Exception as error:
        raise ValueError(f"{name} must expose array shape and dtype metadata") from error
    if actual_shape != shape or actual_dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} has an invalid shape or dtype")


def _require_vector(name: str, value: object, length: int, *, dtype: Any) -> Array:
    _require_array_metadata(name, value, (length,), dtype)
    return cast(Array, value)


def _skip_zero_scale(scale: float, value: Array) -> Array:
    """Keep finite multiplication exact while repairing zero-scaled poison."""
    product = scale * value
    if scale != 0.0:
        return product
    return jnp.where(jnp.isfinite(value), product, jnp.zeros_like(value))


def _recover_nonfinite_at_zero_scale(scale: float, value: Array) -> Array:
    """Repair only non-finite history that a zero scale is configured to forget."""
    if scale != 0.0:
        return value
    return jnp.where(~jnp.isfinite(value), jnp.zeros_like(value), value)


def _mask_unused_history(
    history: Array,
    context: Array,
    unused: Array,
) -> Array:
    """Zero only history entries that the candidate update does not consume."""
    current = history[context]
    recoverable = unused & ~jnp.isfinite(current)
    checked = jnp.where(recoverable, jnp.zeros_like(current), current)
    return history.at[context].set(checked)


def _validated_cost_weight(value: float) -> float:
    """Return a non-negative weight that stays finite and nonzero in float32."""
    return _validated_float32("cost_weight", value, lower=0.0, preserve_nonzero=True)


def _weighted_cost_terms(costs: Array, cost_weight: float) -> tuple[Array, Array]:
    """Return finite cost terms and whether each requested product is valid."""
    weight = jnp.asarray(cost_weight, dtype=jnp.float32)
    weighted = weight * costs
    used = weight != 0.0
    valid = (~used) | (jnp.isfinite(costs) & (costs >= 0.0) & jnp.isfinite(weighted))
    terms = jnp.where(used & valid, weighted, jnp.zeros_like(weighted))
    return terms, valid


def optimal_hedge_learning_rate(
    n_actions: int,
    horizon: int,
    loss_bound: float = 1.0,
) -> float:
    """Return the fixed-horizon Hedge rate for bounded losses.

    For losses in ``[0, loss_bound]`` and an undiscounted full-information
    exponential-weights update, ``sqrt(8 * ln(n) / (T * L^2))`` is the rate
    that minimizes the standard regret bound
    ``ln(n)/eta + eta * T * L^2 / 8`` (Freund & Schapire 1997;
    Cesa-Bianchi & Lugosi 2006, ch. 2).  For ``n_actions == 1`` the regret is
    identically zero, so the returned learning rate is ``0.0``.
    """
    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if loss_bound <= 0.0:
        raise ValueError("loss_bound must be positive")
    if n_actions == 1:
        return 0.0
    return math.sqrt(8.0 * math.log(n_actions) / (horizon * loss_bound**2))


def finite_candidate_hedge_regret_bound(
    n_actions: int,
    horizon: int,
    learning_rate: float,
    loss_bound: float = 1.0,
) -> float:
    """Bound static regret for finite full-information Hedge selection.

    For losses in ``[0, loss_bound]`` and update
    ``w_i <- w_i * exp(-learning_rate * loss_i)``, the cumulative mixture loss
    is at most the best fixed action's cumulative loss plus this value (the
    standard exponential-weights bound; Cesa-Bianchi & Lugosi 2006, ch. 2).

    This is a theorem for the selector abstraction. Discounting, forced
    exploration, partial feedback, nonstationary comparators, and heuristic
    promote/delete rules require separate terms and must not cite this helper
    as a proof.
    """
    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative")
    if loss_bound <= 0.0:
        raise ValueError("loss_bound must be positive")
    if n_actions == 1:
        return 0.0
    if learning_rate == 0.0:
        return math.inf
    return math.log(n_actions) / learning_rate + learning_rate * horizon * loss_bound**2 / 8.0


@chex.dataclass(frozen=True)
class LearnedResourceManagerState:
    """State for a contextual learned resource manager.

    Attributes:
        log_weights: Per-context action preferences, shape
            ``(n_contexts, n_actions)``.
        loss_ema: Per-context/action EMA of observed adjusted losses.
        action_counts: Per-context/action count of updates in which an action
            had a finite observed loss.
        step_count: Scalar update counter.
    """

    log_weights: Float[Array, " n_contexts n_actions"]
    loss_ema: Float[Array, " n_contexts n_actions"]
    action_counts: Float[Array, " n_contexts n_actions"]
    step_count: Array


@chex.dataclass(frozen=True)
class LearnedResourceManagerUpdateResult:
    """Result of one resource-manager update.

    Attributes:
        state: Updated manager state.
        weights: Pre-update action allocation for the selected context.
        adjusted_losses: Per-action loss plus resource cost.
        advantages: Baseline-relative advantage, positive for better actions.
    """

    state: LearnedResourceManagerState
    weights: Float[Array, " n_actions"]
    adjusted_losses: Float[Array, " n_actions"]
    advantages: Float[Array, " n_actions"]
    valid_actions: Bool[Array, " n_actions"]
    update_applied: Bool[Array, ""]


class LearnedResourceManager:
    """Contextual Hedge manager over discrete resource policies.

    The update is a discounted variant of exponential weights / Hedge
    (Freund & Schapire 1997).  At every time step, the manager emits a
    probability vector over resource
    actions.  After seeing the current losses, it shifts probability mass toward
    actions whose adjusted loss was lower than the manager's own allocation
    baseline.  Optional ``resource_costs`` let experiments encode a preference
    for cheaper plasticity when predictive losses are comparable.

    The update is causal and online:

    ``advantage_i = dot(weights, adjusted_losses) - adjusted_losses_i``

    ``log_weight_i <- discount * log_weight_i + learning_rate * advantage_i``

    Positive advantage means action ``i`` beat the current allocation.  Centering
    by the allocation baseline keeps the preferences numerically stable and
    makes uniform shifts in all losses irrelevant.
    """

    def __init__(
        self,
        n_actions: int,
        n_contexts: int = 1,
        learning_rate: float = 1.0,
        discount: float = 0.995,
        exploration: float = 0.0,
        loss_decay: float = 0.99,
        cost_weight: float = 0.0,
        advantage_clip: float = 10.0,
    ) -> None:
        """Initialize the resource manager.

        Args:
            n_actions: Number of discrete resource policies.
            n_contexts: Number of independent contexts/state bins.
            learning_rate: Exponentiated-gradient step size.
            discount: Preference decay in ``[0, 1]``.
            exploration: Uniform allocation floor in ``[0, 1)``.
            loss_decay: EMA decay for diagnostics.
            cost_weight: Multiplier on optional resource costs.
            advantage_clip: Absolute clip on centered advantages.

        Raises:
            ValueError: If any hyperparameter is outside its valid range.
        """
        n_actions = _require_int32("n_actions", n_actions, minimum=1)
        n_contexts = _require_int32("n_contexts", n_contexts, minimum=1)
        _require_manager_state_budget("LearnedResourceManager", n_contexts, n_actions)
        learning_rate = _validated_float32(
            "learning_rate", learning_rate, lower=0.0, preserve_nonzero=True
        )
        discount = _validated_float32(
            "discount", discount, lower=0.0, upper=1.0, preserve_nonzero=True
        )
        exploration = _validated_float32(
            "exploration",
            exploration,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
            preserve_nonzero=True,
        )
        loss_decay = _validated_float32(
            "loss_decay",
            loss_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
            preserve_nonzero=True,
        )
        validated_cost_weight = _validated_cost_weight(cost_weight)
        advantage_clip = _validated_float32("advantage_clip", advantage_clip, positive=True)

        self._n_actions = int(n_actions)
        self._n_contexts = int(n_contexts)
        self._learning_rate = learning_rate
        self._discount = discount
        self._exploration = exploration
        self._loss_decay = loss_decay
        self._cost_weight = validated_cost_weight
        self._advantage_clip = advantage_clip

    @property
    def n_actions(self) -> int:
        """Number of resource actions."""
        return self._n_actions

    @property
    def n_contexts(self) -> int:
        """Number of independent contexts."""
        return self._n_contexts

    def to_config(self) -> dict[str, Any]:
        """Serialize manager configuration."""
        return {
            "type": "LearnedResourceManager",
            "n_actions": self._n_actions,
            "n_contexts": self._n_contexts,
            "learning_rate": self._learning_rate,
            "discount": self._discount,
            "exploration": self._exploration,
            "loss_decay": self._loss_decay,
            "cost_weight": self._cost_weight,
            "advantage_clip": self._advantage_clip,
        }

    def fixed_candidate_regret_bound(
        self,
        horizon: int,
        loss_bound: float = 1.0,
    ) -> float:
        """Return the finite-action Hedge regret bound for this rate.

        This bound applies only to the undiscounted, no-exploration,
        full-information selector abstraction. The runtime manager can also be
        used with discounting, exploration floors, costs, ignored ``NaN`` losses,
        or context switches; those settings are causal and useful, but this
        static bound is no longer the complete statement.
        """
        return finite_candidate_hedge_regret_bound(
            self._n_actions,
            horizon,
            self._learning_rate,
            loss_bound=loss_bound,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LearnedResourceManager:
        """Reconstruct a manager from :meth:`to_config` output."""
        config = _read_mapping("config", config)
        if config.pop("type", None) != "LearnedResourceManager":
            raise ValueError("serialized LearnedResourceManager type is invalid")
        _require_serialized_fields(
            config,
            owner="LearnedResourceManager",
            expected={
                "n_actions",
                "n_contexts",
                "learning_rate",
                "discount",
                "exploration",
                "loss_decay",
                "cost_weight",
                "advantage_clip",
            },
            integer_fields=("n_actions", "n_contexts"),
            float_fields=(
                "learning_rate",
                "discount",
                "exploration",
                "loss_decay",
                "cost_weight",
                "advantage_clip",
            ),
        )
        try:
            return cls(**config)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("serialized LearnedResourceManager is invalid") from error

    def init(self) -> LearnedResourceManagerState:
        """Create an initial uniform-allocation state."""
        shape = (self._n_contexts, self._n_actions)
        return LearnedResourceManagerState(  # type: ignore[call-arg]
            log_weights=jnp.zeros(shape, dtype=jnp.float32),
            loss_ema=jnp.zeros(shape, dtype=jnp.float32),
            action_counts=jnp.zeros(shape, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _require_state_contract(self, state: LearnedResourceManagerState) -> None:
        if type(state) is not LearnedResourceManagerState:
            raise ValueError("state must be a LearnedResourceManagerState")
        shape = (self._n_contexts, self._n_actions)
        for name in ("log_weights", "loss_ema", "action_counts"):
            leaf = getattr(state, name)
            _require_array_metadata(f"state.{name}", leaf, shape, jnp.float32)
        _require_array_metadata("state.step_count", state.step_count, (), jnp.int32)

    def weights(
        self,
        state: LearnedResourceManagerState,
        context_id: Array | int = 0,
    ) -> Float[Array, " n_actions"]:
        """Return the current allocation for ``context_id``."""
        self._require_state_contract(state)
        return cast(Array, self._weights_jit(state, context_id))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _weights_jit(
        self,
        state: LearnedResourceManagerState,
        context_id: Array | int = 0,
    ) -> Float[Array, " n_actions"]:
        context, context_valid = safe_discrete_action(
            context_id,
            self._n_contexts,
        )
        logits = state.log_weights[context]
        logits = _recover_nonfinite_at_zero_scale(self._discount, logits)
        weights = jax.nn.softmax(logits)
        if self._exploration > 0.0:
            uniform = jnp.full_like(weights, 1.0 / float(self._n_actions))
            weights = (1.0 - self._exploration) * weights + self._exploration * uniform
        return jnp.where(
            context_valid,
            weights,
            jnp.full_like(weights, jnp.nan),
        )

    def update(
        self,
        state: LearnedResourceManagerState,
        losses: Float[Array, " n_actions"],
        context_id: Array | int = 0,
        resource_costs: Float[Array, " n_actions"] | None = None,
    ) -> LearnedResourceManagerUpdateResult:
        """Update preferences from current per-action losses.

        Args:
            state: Current manager state.
            losses: Per-action predictive losses. ``NaN`` entries are ignored.
            context_id: Context/state bin to update.
            resource_costs: Optional non-negative per-action costs.

        Returns:
            :class:`LearnedResourceManagerUpdateResult`.
        """
        self._require_state_contract(state)
        _require_vector("losses", losses, self._n_actions, dtype=jnp.float32)
        if resource_costs is not None:
            _require_vector(
                "resource_costs", resource_costs, self._n_actions, dtype=jnp.float32
            )
        return cast(
            LearnedResourceManagerUpdateResult,
            self._update_jit(state, losses, context_id, resource_costs),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: LearnedResourceManagerState,
        losses: Float[Array, " n_actions"],
        context_id: Array | int = 0,
        resource_costs: Float[Array, " n_actions"] | None = None,
    ) -> LearnedResourceManagerUpdateResult:
        context, context_valid = safe_discrete_action(
            context_id,
            self._n_contexts,
        )
        losses = _require_vector("losses", losses, self._n_actions, dtype=jnp.float32)
        finite_losses = jnp.isfinite(losses)
        safe_losses = jnp.where(finite_losses, losses, 0.0)
        costs = (
            jnp.zeros_like(safe_losses)
            if resource_costs is None
            else _require_vector(
                "resource_costs", resource_costs, self._n_actions, dtype=jnp.float32
            )
        )
        cost_terms, costs_valid = _weighted_cost_terms(costs, self._cost_weight)
        valid_actions = finite_losses & costs_valid
        adjusted = jnp.where(valid_actions, safe_losses + cost_terms, 0.0)

        weights = self._weights_jit(state, context)
        finite_weight_sum = jnp.maximum(jnp.sum(jnp.where(valid_actions, weights, 0.0)), 1e-12)
        masked_weights = jnp.where(valid_actions, weights / finite_weight_sum, 0.0)
        baseline = jnp.sum(masked_weights * adjusted)
        advantages = jnp.where(valid_actions, baseline - adjusted, 0.0)
        advantages = jnp.clip(
            advantages,
            -self._advantage_clip,
            self._advantage_clip,
        )

        old_context_logits = state.log_weights[context]
        new_context_logits = (
            _skip_zero_scale(self._discount, old_context_logits) + self._learning_rate * advantages
        )
        # Remove an arbitrary additive constant for numerical stability.
        new_context_logits = new_context_logits - jnp.mean(new_context_logits)
        new_log_weights = state.log_weights.at[context].set(new_context_logits)

        old_ema = state.loss_ema[context]
        new_ema = jnp.where(
            valid_actions,
            _skip_zero_scale(self._loss_decay, old_ema) + (1.0 - self._loss_decay) * adjusted,
            old_ema,
        )
        new_loss_ema = state.loss_ema.at[context].set(new_ema)
        new_counts = state.action_counts.at[context].add(valid_actions.astype(jnp.float32))

        candidate_state = LearnedResourceManagerState(  # type: ignore[call-arg]
            log_weights=new_log_weights,
            loss_ema=new_loss_ema,
            action_counts=new_counts,
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
        )
        checked_log_weights = _mask_unused_history(
            state.log_weights,
            context,
            jnp.asarray(self._discount == 0.0, dtype=jnp.bool_),
        )
        checked_loss_ema = _mask_unused_history(
            state.loss_ema,
            context,
            valid_actions & jnp.asarray(self._loss_decay == 0.0, dtype=jnp.bool_),
        )
        previous_checked = LearnedResourceManagerState(  # type: ignore[call-arg]
            log_weights=checked_log_weights,
            loss_ema=checked_loss_ema,
            action_counts=state.action_counts,
            step_count=state.step_count,
        )
        update_applied = (
            context_valid
            & (state.step_count >= 0)
            & floating_tree_is_finite(previous_checked)
            & floating_tree_is_finite(candidate_state)
            & jnp.all(jnp.isfinite(weights))
            & jnp.all(jnp.isfinite(adjusted))
            & jnp.all(jnp.isfinite(advantages))
        )
        return LearnedResourceManagerUpdateResult(  # type: ignore[call-arg]
            state=select_transaction(update_applied, candidate_state, state),
            weights=neutralize_array(update_applied, weights),
            adjusted_losses=neutralize_array(update_applied, adjusted),
            advantages=neutralize_array(update_applied, advantages),
            valid_actions=valid_actions & update_applied,
            update_applied=update_applied,
        )


@chex.dataclass(frozen=True)
class GeneratorMetaResourceManagerState:
    """State for generator-internal meta-resource allocation.

    Attributes:
        log_weights: Per-context generator-policy preferences.
        reward_ema: Per-context/policy EMA of observed provenance rewards.
        action_counts: Per-context/policy count of finite reward updates.
        step_count: Scalar update counter.
    """

    log_weights: Float[Array, " n_contexts n_policies"]
    reward_ema: Float[Array, " n_contexts n_policies"]
    action_counts: Float[Array, " n_contexts n_policies"]
    step_count: Array


@chex.dataclass(frozen=True)
class GeneratorMetaResourceDecision:
    """One causal generator-policy decision and its knobs."""

    action: Array
    weights: Float[Array, " n_policies"]
    op_id: Array
    parent_mode: Array
    replacement_multiplier: Array
    promotion_margin_multiplier: Array
    candidate_min_age_multiplier: Array
    imprint_scale: Array
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GeneratorMetaResourceUpdateResult:
    """Result of one generator meta-resource update."""

    state: GeneratorMetaResourceManagerState
    weights: Float[Array, " n_policies"]
    adjusted_rewards: Float[Array, " n_policies"]
    advantages: Float[Array, " n_policies"]
    valid_actions: Bool[Array, " n_policies"]
    update_applied: Bool[Array, ""]


class GeneratorMetaResourceManager:
    """Contextual Hedge manager for feature-generator internals.

    A policy is a bundle of generator-internal choices: operation type, parent
    sampling mode, replacement rate multiplier, promotion-margin multiplier,
    candidate refresh age multiplier, and residual-imprint scale.  The manager
    chooses one policy before candidate construction and later updates policy
    preferences from causal provenance rewards, such as the utility of active
    or candidate features built by each policy.
    """

    def __init__(
        self,
        policy_names: tuple[str, ...],
        op_ids: tuple[int, ...],
        parent_modes: tuple[int, ...],
        replacement_multipliers: tuple[float, ...],
        promotion_margin_multipliers: tuple[float, ...],
        candidate_min_age_multipliers: tuple[float, ...],
        imprint_scales: tuple[float, ...],
        n_contexts: int = 1,
        learning_rate: float = 1.0,
        discount: float = 0.995,
        exploration: float = 0.01,
        reward_decay: float = 0.99,
        cost_weight: float = 0.0,
        advantage_clip: float = 10.0,
        update_rule: str = "hedge",
        initial_preferences: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize a generator meta-resource manager.

        Args:
            policy_names: Stable names for generator policies.
            op_ids: Per-policy operation ids interpreted by the consumer.
            parent_modes: Per-policy parent-selection modes.
            replacement_multipliers: Per-policy replacement-rate multipliers.
            promotion_margin_multipliers: Per-policy promotion threshold
                multipliers; lower values are more aggressive.
            candidate_min_age_multipliers: Per-policy candidate refresh-age
                multipliers.
            imprint_scales: Per-policy residual-imprint scales.
            n_contexts: Number of independent context bins.
            learning_rate: Exponentiated-gradient step size.
            discount: Preference decay in ``[0, 1]``.
            exploration: Uniform action-probability floor in ``[0, 1)``.
            reward_decay: EMA decay for diagnostics.
            cost_weight: Multiplier on optional resource costs.
            advantage_clip: Absolute clip on centered rewards.
            update_rule: ``"hedge"`` updates all finite provenance scores;
                ``"exp3"`` applies an importance-weighted update to the
                sampled policy only (Exp3-style ``reward / probability``
                estimator; Auer et al. 2002).
            initial_preferences: Optional additive initial log-preferences.
        """
        policy_names = _require_exact_tuple("policy_names", policy_names)
        op_ids = _require_exact_tuple("op_ids", op_ids)
        parent_modes = _require_exact_tuple("parent_modes", parent_modes)
        replacement_multipliers = _require_exact_tuple(
            "replacement_multipliers", replacement_multipliers
        )
        promotion_margin_multipliers = _require_exact_tuple(
            "promotion_margin_multipliers", promotion_margin_multipliers
        )
        candidate_min_age_multipliers = _require_exact_tuple(
            "candidate_min_age_multipliers", candidate_min_age_multipliers
        )
        imprint_scales = _require_exact_tuple("imprint_scales", imprint_scales)
        if initial_preferences is not None:
            initial_preferences = _require_exact_tuple("initial_preferences", initial_preferences)
        n_policies = len(policy_names)
        if n_policies < 1:
            raise ValueError("at least one generator policy is required")
        lengths = {
            len(op_ids),
            len(parent_modes),
            len(replacement_multipliers),
            len(promotion_margin_multipliers),
            len(candidate_min_age_multipliers),
            len(imprint_scales),
        }
        if lengths != {n_policies}:
            raise ValueError("all generator policy tuples must have the same length")
        n_contexts = _require_int32("n_contexts", n_contexts, minimum=1)
        _require_manager_state_budget("GeneratorMetaResourceManager", n_contexts, n_policies)
        if any(type(name) is not str for name in policy_names):
            raise ValueError("policy_names elements must be exact strings")
        op_ids = tuple(_require_int32("op_ids element", op_id, minimum=0) for op_id in op_ids)
        parent_modes = tuple(
            _require_int32("parent_modes element", mode, minimum=0) for mode in parent_modes
        )
        learning_rate = _validated_float32(
            "learning_rate", learning_rate, lower=0.0, preserve_nonzero=True
        )
        discount = _validated_float32(
            "discount", discount, lower=0.0, upper=1.0, preserve_nonzero=True
        )
        exploration = _validated_float32(
            "exploration",
            exploration,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
            preserve_nonzero=True,
        )
        reward_decay = _validated_float32(
            "reward_decay",
            reward_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
            preserve_nonzero=True,
        )
        validated_cost_weight = _validated_cost_weight(cost_weight)
        advantage_clip = _validated_float32("advantage_clip", advantage_clip, positive=True)
        if type(update_rule) is not str or update_rule not in {"hedge", "exp3"}:
            raise ValueError("update_rule must be 'hedge' or 'exp3'")
        if initial_preferences is not None and len(initial_preferences) != n_policies:
            raise ValueError("initial_preferences must match policy_names length")
        replacement_multipliers = tuple(
            _validated_float32("replacement_multipliers element", value, positive=True)
            for value in replacement_multipliers
        )
        promotion_margin_multipliers = tuple(
            _validated_float32("promotion_margin_multipliers element", value, positive=True)
            for value in promotion_margin_multipliers
        )
        candidate_min_age_multipliers = tuple(
            _validated_float32("candidate_min_age_multipliers element", value, positive=True)
            for value in candidate_min_age_multipliers
        )
        imprint_scales = tuple(
            _validated_float32(
                "imprint_scales element", value, lower=0.0, preserve_nonzero=True
            )
            for value in imprint_scales
        )
        if initial_preferences is not None:
            initial_preferences = tuple(
                _validated_float32("initial_preferences element", value)
                for value in initial_preferences
            )
            if max(initial_preferences) - min(initial_preferences) > float(
                np.finfo(np.float32).max
            ):
                raise ValueError("initial_preferences span must remain finite in float32")

        self._policy_names = policy_names
        self._op_ids = tuple(int(value) for value in op_ids)
        self._parent_modes = tuple(int(value) for value in parent_modes)
        self._replacement_multipliers = replacement_multipliers
        self._promotion_margin_multipliers = promotion_margin_multipliers
        self._candidate_min_age_multipliers = candidate_min_age_multipliers
        self._imprint_scales = imprint_scales
        self._n_contexts = int(n_contexts)
        self._learning_rate = learning_rate
        self._discount = discount
        self._exploration = exploration
        self._reward_decay = reward_decay
        self._cost_weight = validated_cost_weight
        self._advantage_clip = advantage_clip
        self._update_rule = update_rule
        self._initial_preferences = (
            initial_preferences
            if initial_preferences is not None
            else tuple(0.0 for _ in range(n_policies))
        )

    @property
    def n_policies(self) -> int:
        """Number of generator policies."""
        return len(self._policy_names)

    @property
    def n_contexts(self) -> int:
        """Number of independent contexts."""
        return self._n_contexts

    def to_config(self) -> dict[str, Any]:
        """Serialize manager configuration."""
        return {
            "type": "GeneratorMetaResourceManager",
            "policy_names": list(self._policy_names),
            "op_ids": list(self._op_ids),
            "parent_modes": list(self._parent_modes),
            "replacement_multipliers": list(self._replacement_multipliers),
            "promotion_margin_multipliers": list(self._promotion_margin_multipliers),
            "candidate_min_age_multipliers": list(self._candidate_min_age_multipliers),
            "imprint_scales": list(self._imprint_scales),
            "n_contexts": self._n_contexts,
            "learning_rate": self._learning_rate,
            "discount": self._discount,
            "exploration": self._exploration,
            "reward_decay": self._reward_decay,
            "cost_weight": self._cost_weight,
            "advantage_clip": self._advantage_clip,
            "update_rule": self._update_rule,
            "initial_preferences": list(self._initial_preferences),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GeneratorMetaResourceManager:
        """Reconstruct a manager from :meth:`to_config` output."""
        config = _read_mapping("config", config)
        if config.pop("type", None) != "GeneratorMetaResourceManager":
            raise ValueError("serialized GeneratorMetaResourceManager type is invalid")
        _require_serialized_fields(
            config,
            owner="GeneratorMetaResourceManager",
            expected={
                "policy_names",
                "op_ids",
                "parent_modes",
                "replacement_multipliers",
                "promotion_margin_multipliers",
                "candidate_min_age_multipliers",
                "imprint_scales",
                "n_contexts",
                "learning_rate",
                "discount",
                "exploration",
                "reward_decay",
                "cost_weight",
                "advantage_clip",
                "update_rule",
                "initial_preferences",
            },
            integer_fields=("n_contexts",),
            float_fields=(
                "learning_rate",
                "discount",
                "exploration",
                "reward_decay",
                "cost_weight",
                "advantage_clip",
            ),
            string_fields=("update_rule",),
        )
        try:
            initial_preferences = config.pop("initial_preferences")
            decoded_policy_names = _decode_sequence("policy_names", config.pop("policy_names"))
            decoded_op_ids = _decode_sequence("op_ids", config.pop("op_ids"))
            decoded_parent_modes = _decode_sequence("parent_modes", config.pop("parent_modes"))
            if any(type(value) is not str for value in decoded_policy_names):
                raise ValueError("serialized policy_names elements must be JSON strings")
            if any(type(value) is not int for value in (*decoded_op_ids, *decoded_parent_modes)):
                raise ValueError("serialized policy ids and modes must be JSON integers")
            float_sequences: dict[str, tuple[Any, ...]] = {}
            for name in (
                "replacement_multipliers",
                "promotion_margin_multipliers",
                "candidate_min_age_multipliers",
                "imprint_scales",
            ):
                values = _decode_sequence(name, config.pop(name))
                if any(type(value) is not float for value in values):
                    raise ValueError(f"serialized {name} elements must be JSON numbers")
                float_sequences[name] = values
            decoded_initial = _decode_sequence("initial_preferences", initial_preferences)
            if any(type(value) is not float for value in decoded_initial):
                raise ValueError("serialized initial_preferences elements must be JSON numbers")
            return cls(
                policy_names=decoded_policy_names,
                op_ids=decoded_op_ids,
                parent_modes=decoded_parent_modes,
                replacement_multipliers=float_sequences["replacement_multipliers"],
                promotion_margin_multipliers=float_sequences[
                    "promotion_margin_multipliers"
                ],
                candidate_min_age_multipliers=float_sequences[
                    "candidate_min_age_multipliers"
                ],
                imprint_scales=float_sequences["imprint_scales"],
                initial_preferences=decoded_initial,
                **config,
            )
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("serialized GeneratorMetaResourceManager is invalid") from error

    def init(self) -> GeneratorMetaResourceManagerState:
        """Create an initial uniform-allocation state."""
        shape = (self._n_contexts, self.n_policies)
        initial = jnp.asarray(self._initial_preferences, dtype=jnp.float32)
        # Softmax is shift invariant; subtracting the maximum avoids an
        # overflowing reduction when every legal preference is near FLT_MAX.
        initial = initial - jnp.max(initial)
        log_weights = jnp.broadcast_to(initial, shape)
        return GeneratorMetaResourceManagerState(  # type: ignore[call-arg]
            log_weights=log_weights,
            reward_ema=jnp.zeros(shape, dtype=jnp.float32),
            action_counts=jnp.zeros(shape, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _require_state_contract(self, state: GeneratorMetaResourceManagerState) -> None:
        if type(state) is not GeneratorMetaResourceManagerState:
            raise ValueError("state must be a GeneratorMetaResourceManagerState")
        shape = (self._n_contexts, self.n_policies)
        for name in ("log_weights", "reward_ema", "action_counts"):
            leaf = getattr(state, name)
            _require_array_metadata(f"state.{name}", leaf, shape, jnp.float32)
        _require_array_metadata("state.step_count", state.step_count, (), jnp.int32)

    def weights(
        self,
        state: GeneratorMetaResourceManagerState,
        context_id: Array | int = 0,
    ) -> Float[Array, " n_policies"]:
        """Return the current policy allocation for ``context_id``."""
        self._require_state_contract(state)
        return cast(Array, self._weights_jit(state, context_id))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _weights_jit(
        self,
        state: GeneratorMetaResourceManagerState,
        context_id: Array | int = 0,
    ) -> Float[Array, " n_policies"]:
        context, context_valid = safe_discrete_action(
            context_id,
            self._n_contexts,
        )
        logits = state.log_weights[context]
        logits = _recover_nonfinite_at_zero_scale(self._discount, logits)
        weights = jax.nn.softmax(logits)
        if self._exploration > 0.0:
            uniform = jnp.full_like(weights, 1.0 / float(self.n_policies))
            weights = (1.0 - self._exploration) * weights + self._exploration * uniform
        return jnp.where(
            context_valid,
            weights,
            jnp.full_like(weights, jnp.nan),
        )

    def select(
        self,
        state: GeneratorMetaResourceManagerState,
        key: Array,
        context_id: Array | int = 0,
    ) -> GeneratorMetaResourceDecision:
        """Sample one policy and return the generator knobs it controls."""
        self._require_state_contract(state)
        return cast(GeneratorMetaResourceDecision, self._select_jit(state, key, context_id))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _select_jit(
        self,
        state: GeneratorMetaResourceManagerState,
        key: Array,
        context_id: Array | int = 0,
    ) -> GeneratorMetaResourceDecision:
        context, context_valid = safe_discrete_action(
            context_id,
            self._n_contexts,
        )
        weights = self._weights_jit(state, context)
        selection_valid = (
            context_valid & floating_tree_is_finite(state) & jnp.all(jnp.isfinite(weights))
        )
        action = jr.categorical(key, jnp.log(weights + 1e-8)).astype(jnp.int32)
        op_ids = jnp.asarray(self._op_ids, dtype=jnp.int32)
        parent_modes = jnp.asarray(self._parent_modes, dtype=jnp.int32)
        replacement = jnp.asarray(self._replacement_multipliers, dtype=jnp.float32)
        margins = jnp.asarray(
            self._promotion_margin_multipliers,
            dtype=jnp.float32,
        )
        ages = jnp.asarray(
            self._candidate_min_age_multipliers,
            dtype=jnp.float32,
        )
        imprints = jnp.asarray(self._imprint_scales, dtype=jnp.float32)
        return GeneratorMetaResourceDecision(  # type: ignore[call-arg]
            action=jnp.where(selection_valid, action, -1),
            weights=jnp.where(
                selection_valid,
                weights,
                jnp.full_like(weights, jnp.nan),
            ),
            op_id=jnp.where(selection_valid, op_ids[action], -1),
            parent_mode=jnp.where(selection_valid, parent_modes[action], -1),
            replacement_multiplier=jnp.where(
                selection_valid,
                replacement[action],
                jnp.asarray(jnp.nan, dtype=jnp.float32),
            ),
            promotion_margin_multiplier=jnp.where(
                selection_valid,
                margins[action],
                jnp.asarray(jnp.nan, dtype=jnp.float32),
            ),
            candidate_min_age_multiplier=jnp.where(
                selection_valid,
                ages[action],
                jnp.asarray(jnp.nan, dtype=jnp.float32),
            ),
            imprint_scale=jnp.where(
                selection_valid,
                imprints[action],
                jnp.asarray(jnp.nan, dtype=jnp.float32),
            ),
            valid=selection_valid,
        )

    def update(
        self,
        state: GeneratorMetaResourceManagerState,
        rewards: Float[Array, " n_policies"],
        context_id: Array | int = 0,
        finite_mask: Array | None = None,
        resource_costs: Float[Array, " n_policies"] | None = None,
        selected_action: Array | int | None = None,
        selected_probability: Array | float | None = None,
    ) -> GeneratorMetaResourceUpdateResult:
        """Update preferences from current per-policy rewards.

        ``NaN`` rewards, or entries masked out by ``finite_mask``, are ignored.
        Rewards are maximized, unlike :class:`LearnedResourceManager` losses.
        With ``update_rule="exp3"``, only ``selected_action`` receives an
        importance-weighted update.  This is useful when provenance rewards are
        sparse and the experiment wants explicit exploration credit.
        """
        self._require_state_contract(state)
        _require_vector("rewards", rewards, self.n_policies, dtype=jnp.float32)
        if finite_mask is not None:
            _require_vector("finite_mask", finite_mask, self.n_policies, dtype=jnp.bool_)
        if resource_costs is not None:
            _require_vector(
                "resource_costs", resource_costs, self.n_policies, dtype=jnp.float32
            )
        return cast(
            GeneratorMetaResourceUpdateResult,
            self._update_jit(
                state,
                rewards,
                context_id,
                finite_mask,
                resource_costs,
                selected_action,
                selected_probability,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: GeneratorMetaResourceManagerState,
        rewards: Float[Array, " n_policies"],
        context_id: Array | int = 0,
        finite_mask: Array | None = None,
        resource_costs: Float[Array, " n_policies"] | None = None,
        selected_action: Array | int | None = None,
        selected_probability: Array | float | None = None,
    ) -> GeneratorMetaResourceUpdateResult:
        context, context_valid = safe_discrete_action(
            context_id,
            self._n_contexts,
        )
        rewards = _require_vector("rewards", rewards, self.n_policies, dtype=jnp.float32)
        finite = jnp.isfinite(rewards)
        if finite_mask is not None:
            finite = finite & _require_vector(
                "finite_mask", finite_mask, self.n_policies, dtype=jnp.bool_
            )
        safe_rewards = jnp.where(finite, rewards, 0.0)
        costs = (
            jnp.zeros_like(safe_rewards)
            if resource_costs is None
            else _require_vector(
                "resource_costs", resource_costs, self.n_policies, dtype=jnp.float32
            )
        )
        cost_terms, costs_valid = _weighted_cost_terms(costs, self._cost_weight)
        finite = finite & costs_valid
        adjusted = jnp.where(finite, safe_rewards - cost_terms, 0.0)

        weights = self._weights_jit(state, context)
        finite_weight_sum = jnp.maximum(jnp.sum(jnp.where(finite, weights, 0.0)), 1e-12)
        masked_weights = jnp.where(finite, weights / finite_weight_sum, 0.0)
        baseline = jnp.sum(masked_weights * adjusted)
        selection_input_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._update_rule == "exp3":
            if selected_action is None:
                raise ValueError("selected_action is required for update_rule='exp3'")
            action, action_valid = safe_discrete_action(
                selected_action,
                self.n_policies,
            )
            raw_probability = (
                weights[action]
                if selected_probability is None
                else jnp.asarray(selected_probability, dtype=jnp.float32)
            )
            if raw_probability.shape != ():
                raise ValueError("selected_probability must be scalar")
            probability_valid = (
                jnp.all(jnp.isfinite(raw_probability))
                & jnp.all(raw_probability > 0.0)
                & jnp.all(raw_probability <= 1.0)
            )
            probability = jnp.maximum(jnp.where(probability_valid, raw_probability, 1.0), 1e-6)
            selection_input_valid = action_valid & probability_valid
            selected_finite = finite[action] & selection_input_valid
            reward_hat = jnp.where(
                selected_finite,
                adjusted[action] / probability,
                jnp.array(0.0, dtype=jnp.float32),
            )
            raw_advantages = jnp.zeros_like(adjusted).at[action].set(reward_hat)
            raw_advantages = raw_advantages - jnp.mean(raw_advantages)
            advantages = jnp.where(selected_finite, raw_advantages, 0.0)
        else:
            advantages = jnp.where(finite, adjusted - baseline, 0.0)
        advantages = jnp.clip(
            advantages,
            -self._advantage_clip,
            self._advantage_clip,
        )

        old_context_logits = state.log_weights[context]
        new_context_logits = (
            _skip_zero_scale(self._discount, old_context_logits) + self._learning_rate * advantages
        )
        new_context_logits = new_context_logits - jnp.mean(new_context_logits)
        new_log_weights = state.log_weights.at[context].set(new_context_logits)

        old_ema = state.reward_ema[context]
        new_ema = jnp.where(
            finite,
            _skip_zero_scale(self._reward_decay, old_ema) + (1.0 - self._reward_decay) * adjusted,
            old_ema,
        )
        new_reward_ema = state.reward_ema.at[context].set(new_ema)
        new_counts = state.action_counts.at[context].add(finite.astype(jnp.float32))

        candidate_state = GeneratorMetaResourceManagerState(  # type: ignore[call-arg]
            log_weights=new_log_weights,
            reward_ema=new_reward_ema,
            action_counts=new_counts,
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
        )
        checked_log_weights = _mask_unused_history(
            state.log_weights,
            context,
            jnp.asarray(self._discount == 0.0, dtype=jnp.bool_),
        )
        checked_reward_ema = _mask_unused_history(
            state.reward_ema,
            context,
            finite & jnp.asarray(self._reward_decay == 0.0, dtype=jnp.bool_),
        )
        previous_checked = GeneratorMetaResourceManagerState(  # type: ignore[call-arg]
            log_weights=checked_log_weights,
            reward_ema=checked_reward_ema,
            action_counts=state.action_counts,
            step_count=state.step_count,
        )
        update_applied = (
            context_valid
            & selection_input_valid
            & (state.step_count >= 0)
            & floating_tree_is_finite(previous_checked)
            & floating_tree_is_finite(candidate_state)
            & jnp.all(jnp.isfinite(weights))
            & jnp.all(jnp.isfinite(adjusted))
            & jnp.all(jnp.isfinite(advantages))
        )
        return GeneratorMetaResourceUpdateResult(  # type: ignore[call-arg]
            state=select_transaction(update_applied, candidate_state, state),
            weights=neutralize_array(update_applied, weights),
            adjusted_rewards=neutralize_array(update_applied, adjusted),
            advantages=neutralize_array(update_applied, advantages),
            valid_actions=finite & update_applied,
            update_applied=update_applied,
        )
