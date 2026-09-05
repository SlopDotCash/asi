"""Bounded C-CHAIN adaptation for the current online IPMNIST runner.

The implementation follows the mechanism in Tang et al. (ICML 2025,
arXiv:2506.00592v1) and the authors' public implementation at commit
``2f8bedfefb6a0a276d7709447a5cdb75cecfdaad``:

* an Adam task update is augmented by a cross-entropy penalty that keeps the
  current output distribution close to a recent parameter snapshot on a
  separate reference batch;
* the penalty coefficient can track a target task-loss/regularizer ratio;
* projective-only and orthogonal-only gradient components expose the two
  causal effects studied in the paper.

ASI's stream supplies one example per update rather than PPO minibatches.  A
fixed-size ring of *previous updates* therefore excludes the current training
example by construction (although a data identity can recur), and a static
50-update window replaces the paper code's host-side list of 50 outer-iteration
means.  These are protocol adaptations, not claims of reproducing the paper's
RL results.  The lane is permanently
nonpromoting; its receipt validator records every difference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core

from alberta_framework.benchmarks.upgd_ipmnist import (
    _PLASTICITY_LOSS_FLOOR,
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    LearnerUpdateResult,
    _make_adamw_learner,
    cross_entropy_loss,
    mlp_logits,
)
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    select_transaction,
)

PAPER_REVISION: Final = "arXiv:2506.00592v1"
OFFICIAL_REPOSITORY: Final = "https://github.com/bluecontra/C-CHAIN"
OFFICIAL_COMMIT: Final = "2f8bedfefb6a0a276d7709447a5cdb75cecfdaad"

REFERENCE_CAPACITY: Final = 32
COEFFICIENT_WINDOW: Final = 50
COEFFICIENT_DELAY: Final = 10
SNAPSHOT_WARMUP_UPDATES: Final = 2
MAX_NTK_EXAMPLES: Final = 4
NTK_ENERGY_THRESHOLD: Final = 0.99

_PARAMETER_KEYS: Final = frozenset(("w1", "b1", "w2", "b2", "w3", "b3"))
_HP_KEYS: Final = frozenset(
    (
        "step_size",
        "beta1",
        "beta2",
        "eps",
        "weight_decay",
        "reference_capacity",
        "coefficient_window",
        "coefficient_delay",
        "snapshot_warmup_updates",
        "churn_enabled",
        "adaptive_coefficient",
        "target_relative_loss_scale",
        "initial_coefficient",
        "gradient_component",
        "ntk_diagnostic_examples",
        "ntk_energy_threshold",
    )
)
_MAX_ARRAY_ELEMENTS: Final = 1_000_000
_MAX_NTK_JACOBIAN_ENVELOPE_BYTES: Final = 256 * 1024 * 1024
_INT32_MAX: Final = (1 << 31) - 1


@chex.dataclass(frozen=True, mappable_dataclass=False)
class CChainState:
    """Adam state, recent snapshot/reference data, and bounded diagnostics."""

    optimizer_state: dict[str, Any]
    snapshot_params: dict[str, Array]
    reference_examples: Array
    reference_count: Array
    reference_cursor: Array
    update_count: Array
    task_loss_window: Array
    churn_loss_window: Array
    coefficient_window_count: Array
    coefficient_window_cursor: Array
    coefficient: Array
    cumulative_probability_kl: Array
    cumulative_logit_mse: Array
    diagnostic_count: Array


def _exact_float(name: str, value: object, *, minimum: float = 0.0) -> float:
    if type(value) is not float or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be an exact finite float >= {minimum}")
    return value


def _cchain_hyperparameters(value: object) -> dict[str, float]:
    """Validate one exact, registry-owned C-CHAIN hyperparameter object."""
    if type(value) is not dict or frozenset(value) != _HP_KEYS:
        raise ValueError("C-CHAIN hyperparameters must contain the exact protocol keys")
    hp = {
        name: _exact_float(f"hyperparameters.{name}", cast(dict[str, object], value)[name])
        for name in _HP_KEYS
    }
    expected_adam = ADAMW_PROTOCOL_HYPERPARAMETERS
    if any(hp[name] != expected_adam[name] for name in expected_adam):
        raise ValueError("C-CHAIN must use the current matched AdamW control constants")
    exact_constants = {
        "reference_capacity": float(REFERENCE_CAPACITY),
        "coefficient_window": float(COEFFICIENT_WINDOW),
        "coefficient_delay": float(COEFFICIENT_DELAY),
        "snapshot_warmup_updates": float(SNAPSHOT_WARMUP_UPDATES),
        "initial_coefficient": 1.0,
        "ntk_diagnostic_examples": float(MAX_NTK_EXAMPLES),
        "ntk_energy_threshold": NTK_ENERGY_THRESHOLD,
    }
    if any(hp[name] != expected for name, expected in exact_constants.items()):
        raise ValueError("C-CHAIN hyperparameters drift from the audited adaptation")
    if hp["churn_enabled"] not in (0.0, 1.0):
        raise ValueError("churn_enabled must be exactly 0 or 1")
    if hp["adaptive_coefficient"] not in (0.0, 1.0):
        raise ValueError("adaptive_coefficient must be exactly 0 or 1")
    if hp["gradient_component"] not in (0.0, 1.0, 2.0):
        raise ValueError("gradient_component must encode full, orthogonal, or projective")
    if hp["churn_enabled"] == 0.0:
        if hp["adaptive_coefficient"] != 0.0 or hp["target_relative_loss_scale"] != 0.0:
            raise ValueError("the mechanism-off arm cannot adapt a churn coefficient")
    elif hp["adaptive_coefficient"] == 1.0:
        if hp["target_relative_loss_scale"] != 10_000.0:
            raise ValueError("adaptive C-CHAIN must use the official 10000 target scale")
    elif hp["target_relative_loss_scale"] != 0.0:
        raise ValueError("a fixed C-CHAIN coefficient must have target scale zero")
    return hp


def _array(value: object, *, name: str, rank: int | None = None) -> Array:
    actual_type = type(value)
    if actual_type is not np.ndarray and not issubclass(actual_type, (Array, core.Tracer)):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    array = jnp.asarray(value)
    if (
        array.size < 1
        or array.size > _MAX_ARRAY_ELEMENTS
        or not jnp.issubdtype(array.dtype, jnp.floating)
        or (rank is not None and array.ndim != rank)
    ):
        raise ValueError(f"{name} must be a bounded floating array of the required rank")
    if not isinstance(array, core.Tracer) and not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _params(value: object, *, name: str = "params") -> dict[str, Array]:
    if type(value) is not dict or frozenset(value) != _PARAMETER_KEYS:
        raise ValueError(f"{name} must be one exact protocol MLP tree")
    checked = {
        key: _array(array, name=f"{name}.{key}")
        for key, array in cast(dict[str, object], value).items()
    }
    w1, b1 = checked["w1"], checked["b1"]
    w2, b2 = checked["w2"], checked["b2"]
    w3, b3 = checked["w3"], checked["b3"]
    if (
        w1.ndim != 2
        or b1.shape != (w1.shape[1],)
        or w2.ndim != 2
        or w2.shape[0] != w1.shape[1]
        or b2.shape != (w2.shape[1],)
        or w3.ndim != 2
        or w3.shape[0] != w2.shape[1]
        or b3.shape != (w3.shape[1],)
        or any(array.dtype != jnp.dtype(jnp.float32) for array in checked.values())
    ):
        raise ValueError(f"{name} must match the float32 protocol MLP shapes")
    return checked


def _scalar_array(value: object, *, name: str, dtype: np.dtype[Any]) -> Array:
    actual_type = type(value)
    if not issubclass(actual_type, (Array, core.Tracer)):
        raise ValueError(f"{name} must be a scalar JAX array")
    array = cast(Array, value)
    if array.shape != () or array.dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} must be a scalar {jnp.dtype(dtype).name} JAX array")
    return array


def _masked_reference_objective(
    params: dict[str, Array],
    snapshot_params: dict[str, Array],
    reference_examples: Array,
    reference_count: Array,
) -> tuple[Array, tuple[Array, Array]]:
    """Policy-form C-CHAIN cross entropy plus zero-based churn telemetry."""
    target_logits = jax.vmap(mlp_logits, in_axes=(None, 0))(
        snapshot_params, reference_examples
    )
    current_logits = jax.vmap(mlp_logits, in_axes=(None, 0))(params, reference_examples)
    target_probabilities = jax.lax.stop_gradient(jax.nn.softmax(target_logits, axis=-1))
    current_log_probabilities = jax.nn.log_softmax(current_logits, axis=-1)
    per_example_cross_entropy = -jnp.sum(
        target_probabilities * current_log_probabilities, axis=-1
    )
    target_log_probabilities = jax.nn.log_softmax(target_logits, axis=-1)
    per_example_kl = jnp.sum(
        target_probabilities * (target_log_probabilities - current_log_probabilities),
        axis=-1,
    )
    per_example_logit_mse = jnp.mean(
        jnp.square(current_logits - target_logits), axis=-1
    )
    mask = (
        jnp.arange(reference_examples.shape[0], dtype=jnp.int32) < reference_count
    ).astype(jnp.float32)
    denominator = jnp.maximum(reference_count.astype(jnp.float32), 1.0)
    cross_entropy = jnp.sum(mask * per_example_cross_entropy) / denominator
    probability_kl = jnp.maximum(jnp.sum(mask * per_example_kl) / denominator, 0.0)
    logit_mse = jnp.sum(mask * per_example_logit_mse) / denominator
    return cross_entropy, (probability_kl, logit_mse)


def cchain_gradient_component(
    task_gradient: dict[str, Array],
    churn_gradient: dict[str, Array],
    *,
    component: str,
    epsilon: float = 1e-8,
) -> dict[str, Array]:
    """Return the full, projective, or orthogonal churn-gradient component."""
    if type(component) is not str or component not in ("full", "orthogonal", "projective"):
        raise ValueError("component must be full, orthogonal, or projective")
    _exact_float("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    task = _params(task_gradient, name="task_gradient")
    churn = _params(churn_gradient, name="churn_gradient")
    if any(task[name].shape != churn[name].shape for name in _PARAMETER_KEYS):
        raise ValueError("task and churn gradients must have identical shapes")
    if component == "full":
        return churn
    dot = sum(
        (jnp.vdot(churn[name], task[name]).real for name in sorted(_PARAMETER_KEYS)),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    task_energy = sum(
        (jnp.vdot(task[name], task[name]).real for name in sorted(_PARAMETER_KEYS)),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    scale = dot / jnp.maximum(task_energy, epsilon)
    projection = {name: scale * task[name] for name in _PARAMETER_KEYS}
    if component == "projective":
        return projection
    return {name: churn[name] - projection[name] for name in _PARAMETER_KEYS}


def _validated_state(
    state: object, params: dict[str, Array], hp: Mapping[str, float]
) -> CChainState:
    if type(state) is not CChainState:
        raise ValueError("state must be an exact CChainState")
    trusted = state
    snapshot = _params(trusted.snapshot_params, name="state.snapshot_params")
    if any(snapshot[name].shape != params[name].shape for name in _PARAMETER_KEYS):
        raise ValueError("snapshot parameters must match current parameters")
    examples = _array(trusted.reference_examples, name="state.reference_examples", rank=2)
    if examples.shape != (int(hp["reference_capacity"]), params["w1"].shape[0]):
        raise ValueError("reference buffer shape does not match the protocol")
    for name in (
        "reference_count",
        "reference_cursor",
        "update_count",
        "coefficient_window_count",
        "coefficient_window_cursor",
        "diagnostic_count",
    ):
        scalar = _scalar_array(
            getattr(trusted, name), name=f"state.{name}", dtype=np.dtype(np.int32)
        )
        if not isinstance(scalar, core.Tracer):
            integer = int(scalar)
            maximum = {
                "reference_count": int(hp["reference_capacity"]),
                "reference_cursor": int(hp["reference_capacity"]) - 1,
                "update_count": _INT32_MAX - 1,
                "coefficient_window_count": int(hp["coefficient_window"]),
                "coefficient_window_cursor": int(hp["coefficient_window"]) - 1,
                "diagnostic_count": _INT32_MAX - 1,
            }[name]
            if not 0 <= integer <= maximum:
                raise ValueError(f"state.{name} is outside the protocol domain")
    for name in (
        "coefficient",
        "cumulative_probability_kl",
        "cumulative_logit_mse",
    ):
        scalar = _scalar_array(
            getattr(trusted, name), name=f"state.{name}", dtype=np.dtype(np.float32)
        )
        if not isinstance(scalar, core.Tracer):
            numeric = float(scalar)
            minimum = 1.0 if name == "coefficient" else 0.0
            if not math.isfinite(numeric) or numeric < minimum:
                raise ValueError(f"state.{name} is outside the protocol domain")
    for name in ("task_loss_window", "churn_loss_window"):
        window = _array(getattr(trusted, name), name=f"state.{name}", rank=1)
        if window.shape != (int(hp["coefficient_window"]),):
            raise ValueError(f"state.{name} shape does not match the protocol")
    if type(trusted.optimizer_state) is not dict or frozenset(
        trusted.optimizer_state
    ) != _PARAMETER_KEYS:
        raise ValueError("optimizer_state must match the protocol parameter tree")
    return trusted


def make_cchain_learner(
    hyperparameters: Mapping[str, float],
) -> tuple[Any, Any]:
    """Build one C-CHAIN arm for :mod:`ipmnist_screening`."""
    if type(hyperparameters) is not dict:
        raise ValueError("C-CHAIN factory requires an exact hyperparameter dict")
    hp = _cchain_hyperparameters(hyperparameters)
    adam_hp = {name: hp[name] for name in ADAMW_PROTOCOL_HYPERPARAMETERS}
    adam_init, adam_step = _make_adamw_learner(adam_hp)
    capacity = int(hp["reference_capacity"])
    window_size = int(hp["coefficient_window"])
    warmup = int(hp["snapshot_warmup_updates"])
    delay = int(hp["coefficient_delay"])
    component = {0.0: "full", 1.0: "orthogonal", 2.0: "projective"}[
        hp["gradient_component"]
    ]

    def init_fn(params: dict[str, Array]) -> CChainState:
        checked = _params(params)
        if capacity > _MAX_ARRAY_ELEMENTS // checked["w1"].shape[0]:
            raise ValueError("C-CHAIN reference buffer exceeds the element limit")
        return CChainState(  # type: ignore[call-arg]
            optimizer_state=adam_init(checked),
            snapshot_params={name: value for name, value in checked.items()},
            reference_examples=jnp.zeros(
                (capacity, checked["w1"].shape[0]), dtype=jnp.float32
            ),
            reference_count=jnp.asarray(0, dtype=jnp.int32),
            reference_cursor=jnp.asarray(0, dtype=jnp.int32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
            task_loss_window=jnp.zeros(window_size, dtype=jnp.float32),
            churn_loss_window=jnp.zeros(window_size, dtype=jnp.float32),
            coefficient_window_count=jnp.asarray(0, dtype=jnp.int32),
            coefficient_window_cursor=jnp.asarray(0, dtype=jnp.int32),
            coefficient=jnp.asarray(hp["initial_coefficient"], dtype=jnp.float32),
            cumulative_probability_kl=jnp.asarray(0.0, dtype=jnp.float32),
            cumulative_logit_mse=jnp.asarray(0.0, dtype=jnp.float32),
            diagnostic_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def full_step(
        params: dict[str, Array], state: CChainState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], CChainState, tuple[Array, Array, Array]]:
        checked_params = _params(params)
        checked_state = _validated_state(state, checked_params, hp)
        example = _array(x, name="example", rank=1)
        if example.shape != (checked_params["w1"].shape[0],):
            raise ValueError("example shape does not match the protocol MLP")
        if not isinstance(y, (Array, core.Tracer)) or y.shape != () or y.dtype != jnp.int32:
            raise ValueError("label must be one scalar int32 JAX array")
        (task_loss, logits), task_gradient = jax.value_and_grad(
            cross_entropy_loss, has_aux=True
        )(checked_params, example, y)
        state_valid = (
            floating_tree_is_finite(checked_state)
            & (checked_state.reference_count >= 0)
            & (checked_state.reference_count <= capacity)
            & (checked_state.reference_cursor >= 0)
            & (checked_state.reference_cursor < capacity)
            & (checked_state.update_count >= 0)
            & (checked_state.update_count < _INT32_MAX)
            & (checked_state.coefficient_window_count >= 0)
            & (checked_state.coefficient_window_count <= window_size)
            & (checked_state.coefficient_window_cursor >= 0)
            & (checked_state.coefficient_window_cursor < window_size)
            & (checked_state.coefficient >= 1.0)
            & (checked_state.cumulative_probability_kl >= 0.0)
            & (checked_state.cumulative_logit_mse >= 0.0)
            & (checked_state.diagnostic_count >= 0)
            & (checked_state.diagnostic_count < _INT32_MAX)
        )
        safe_reference_count = jnp.clip(checked_state.reference_count, 0, capacity)
        safe_reference_cursor = jnp.clip(
            checked_state.reference_cursor, 0, capacity - 1
        )
        safe_update_count = jnp.clip(checked_state.update_count, 0, _INT32_MAX - 1)
        safe_window_count = jnp.clip(
            checked_state.coefficient_window_count, 0, window_size
        )
        safe_window_cursor = jnp.clip(
            checked_state.coefficient_window_cursor, 0, window_size - 1
        )
        safe_coefficient = jnp.where(
            jnp.isfinite(checked_state.coefficient)
            & (checked_state.coefficient >= 1.0),
            checked_state.coefficient,
            1.0,
        )
        active = state_valid & (safe_update_count >= warmup) & (safe_reference_count > 0)

        zero_gradient = jax.tree.map(jnp.zeros_like, checked_params)

        def measure_churn(
            operand: tuple[dict[str, Array], dict[str, Array], Array, Array]
        ) -> tuple[Array, Array, Array, dict[str, Array]]:
            current, snapshot, references, count = operand

            def objective(candidate: dict[str, Array]) -> tuple[Array, tuple[Array, Array]]:
                return _masked_reference_objective(
                    candidate, snapshot, references, count
                )

            (loss, (probability_kl, logit_mse)), gradient = jax.value_and_grad(
                objective, has_aux=True
            )(current)
            return loss, probability_kl, logit_mse, gradient

        churn_loss, probability_kl, logit_mse, churn_gradient = jax.lax.cond(
            active,
            measure_churn,
            lambda operand: (
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                zero_gradient,
            ),
            (
                checked_params,
                checked_state.snapshot_params,
                checked_state.reference_examples,
                safe_reference_count,
            ),
        )
        if hp["churn_enabled"] == 0.0:
            combined_gradient = task_gradient
        else:
            selected_churn = cchain_gradient_component(
                task_gradient,
                churn_gradient,
                component=component,
            )
            combined_gradient = {
                name: task_gradient[name] + safe_coefficient * selected_churn[name]
                for name in _PARAMETER_KEYS
            }

        update = adam_step(
            checked_params, checked_state.optimizer_state, combined_gradient, key
        )
        if type(update) is not LearnerUpdateResult:
            raise TypeError("the matched Adam learner returned an unsupported update")
        candidate_params = update.params

        buffer = checked_state.reference_examples.at[
            safe_reference_cursor
        ].set(example)
        reference_count = jnp.minimum(safe_reference_count + 1, capacity)
        reference_cursor = (safe_reference_cursor + 1) % capacity
        window_cursor = safe_window_cursor
        task_window_candidate = checked_state.task_loss_window.at[window_cursor].set(
            jnp.abs(task_loss)
        )
        churn_window_candidate = checked_state.churn_loss_window.at[window_cursor].set(
            churn_loss
        )
        task_window = jnp.where(
            active, task_window_candidate, checked_state.task_loss_window
        )
        churn_window = jnp.where(
            active, churn_window_candidate, checked_state.churn_loss_window
        )
        window_count = jnp.where(
            active,
            jnp.minimum(safe_window_count + 1, window_size),
            safe_window_count,
        )
        next_window_cursor = jnp.where(
            active, (window_cursor + 1) % window_size, window_cursor
        )
        denominator = jnp.maximum(window_count.astype(jnp.float32), 1.0)
        mean_task_loss = jnp.sum(task_window) / denominator
        mean_churn_loss = jnp.sum(churn_window) / denominator
        adaptive_candidate = jnp.maximum(
            hp["target_relative_loss_scale"]
            * mean_task_loss
            / (mean_churn_loss + hp["eps"]),
            1.0,
        )
        coefficient = jnp.where(
            active
            & (hp["adaptive_coefficient"] == 1.0)
            & (window_count >= delay)
            & jnp.isfinite(adaptive_candidate),
            adaptive_candidate,
            safe_coefficient,
        )
        candidate_state = CChainState(  # type: ignore[call-arg]
            optimizer_state=update.state,
            snapshot_params={name: value for name, value in checked_params.items()},
            reference_examples=buffer,
            reference_count=reference_count,
            reference_cursor=reference_cursor,
            update_count=safe_update_count + 1,
            task_loss_window=task_window,
            churn_loss_window=churn_window,
            coefficient_window_count=window_count,
            coefficient_window_cursor=next_window_cursor,
            coefficient=coefficient,
            cumulative_probability_kl=(
                checked_state.cumulative_probability_kl + probability_kl
            ),
            cumulative_logit_mse=checked_state.cumulative_logit_mse + logit_mse,
            diagnostic_count=checked_state.diagnostic_count + active.astype(jnp.int32),
        )
        transaction_valid = (
            update.update_applied
            & state_valid
            & floating_tree_is_finite(candidate_state)
            & jnp.isfinite(task_loss)
            & jnp.all(jnp.isfinite(logits))
        )
        new_params = select_transaction(transaction_valid, candidate_params, checked_params)
        new_state = select_transaction(transaction_valid, candidate_state, checked_state)
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        loss_after, _ = cross_entropy_loss(new_params, example, y)
        # Algebraically identical to the runner's metric, written directly so
        # this module does not import the screening registry and create a cycle.
        plasticity = jnp.clip(
            1.0 - loss_after / jnp.maximum(task_loss, _PLASTICITY_LOSS_FLOOR),
            0.0,
            1.0,
        )
        return new_params, new_state, (accuracy, task_loss, plasticity)

    return init_fn, full_step


def cchain_ntk_diagnostics(
    params: dict[str, Array],
    examples: Array,
    *,
    energy_threshold: float = NTK_ENERGY_THRESHOLD,
) -> dict[str, Array]:
    """Return bounded full-logit empirical-NTK rank and scale diagnostics.

    Each example/class logit is one Jacobian row.  This differs from the
    paper's policy-network diagnostic and is named explicitly in receipts.
    """
    checked = _params(params)
    batch = _array(examples, name="examples", rank=2)
    if batch.shape[1] != checked["w1"].shape[0]:
        raise ValueError("diagnostic examples do not match the protocol input width")
    if not 1 <= batch.shape[0] <= MAX_NTK_EXAMPLES:
        raise ValueError(f"diagnostics require 1..{MAX_NTK_EXAMPLES} examples")
    threshold = _exact_float(
        "energy_threshold", energy_threshold, minimum=np.nextafter(0.0, 1.0)
    )
    if threshold > 1.0:
        raise ValueError("energy_threshold must lie in (0, 1]")
    parameter_count = sum(array.size for array in checked.values())
    rows = batch.shape[0] * checked["b3"].shape[0]
    # Exact logical envelope for the Jacobian pytree plus its concatenated
    # matrix. Backend/compiler scratch allocations are deliberately not called
    # exact memory accounting.
    jacobian_envelope_bytes = (
        2 * rows * parameter_count * np.dtype(np.float32).itemsize
    )
    if jacobian_envelope_bytes > _MAX_NTK_JACOBIAN_ENVELOPE_BYTES:
        raise ValueError("empirical NTK Jacobian envelope exceeds 256 MiB")

    def all_logits(candidate: dict[str, Array]) -> Array:
        return jax.vmap(mlp_logits, in_axes=(None, 0))(candidate, batch).reshape(-1)

    jacobian_tree = jax.jacrev(all_logits)(checked)
    jacobian = jnp.concatenate(
        tuple(
            jacobian_tree[name].reshape(rows, -1) for name in sorted(_PARAMETER_KEYS)
        ),
        axis=1,
    )
    gram = jacobian @ jacobian.T
    eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(gram), 0.0)[::-1]
    total = jnp.sum(eigenvalues)
    cumulative = jnp.cumsum(eigenvalues)
    rank = jnp.where(
        total > 0.0,
        jnp.minimum(
            jnp.searchsorted(cumulative, threshold * total, side="left") + 1,
            rows,
        ),
        0,
    ).astype(jnp.int32)
    diagonal = jnp.diag(gram)
    off_diagonal = gram - jnp.diag(diagonal)
    off_count = max(rows * rows - rows, 1)
    return {
        "ntk_threshold_rank": rank,
        "ntk_off_diagonal_abs_mean": jnp.sum(jnp.abs(off_diagonal)) / off_count,
        "ntk_diagonal_mean": jnp.mean(diagonal),
    }


def cchain_host_diagnostics(
    params: dict[str, Array], state: CChainState
) -> dict[str, float]:
    """Materialize final run diagnostics for a host-side result receipt."""
    checked = _params(params)
    hp = cchain_hyperparameters(
        churn_enabled=1.0,
        adaptive_coefficient=1.0,
        target_relative_loss_scale=10_000.0,
        gradient_component=0.0,
    )
    trusted = _validated_state(state, checked, hp)
    reference_count = int(jax.device_get(trusted.reference_count))
    diagnostic_examples = min(reference_count, MAX_NTK_EXAMPLES)
    if diagnostic_examples < 1:
        raise ValueError("C-CHAIN diagnostics require at least one retained reference")
    ntk = cchain_ntk_diagnostics(
        checked, trusted.reference_examples[:diagnostic_examples]
    )
    count = int(jax.device_get(trusted.diagnostic_count))
    denominator = max(count, 1)
    return {
        "mean_probability_kl": float(
            jax.device_get(trusted.cumulative_probability_kl)
        )
        / denominator,
        "mean_logit_mse": float(jax.device_get(trusted.cumulative_logit_mse))
        / denominator,
        "final_coefficient": float(jax.device_get(trusted.coefficient)),
        "diagnostic_updates": float(count),
        "ntk_threshold_rank": float(jax.device_get(ntk["ntk_threshold_rank"])),
        "ntk_off_diagonal_abs_mean": float(
            jax.device_get(ntk["ntk_off_diagonal_abs_mean"])
        ),
        "ntk_diagonal_mean": float(jax.device_get(ntk["ntk_diagonal_mean"])),
        "ntk_examples": float(diagnostic_examples),
    }


def cchain_hyperparameters(
    *,
    churn_enabled: float,
    adaptive_coefficient: float,
    target_relative_loss_scale: float,
    gradient_component: float,
) -> dict[str, float]:
    """Build and validate one registry-ready C-CHAIN hyperparameter object."""
    value = {
        **ADAMW_PROTOCOL_HYPERPARAMETERS,
        "reference_capacity": float(REFERENCE_CAPACITY),
        "coefficient_window": float(COEFFICIENT_WINDOW),
        "coefficient_delay": float(COEFFICIENT_DELAY),
        "snapshot_warmup_updates": float(SNAPSHOT_WARMUP_UPDATES),
        "churn_enabled": churn_enabled,
        "adaptive_coefficient": adaptive_coefficient,
        "target_relative_loss_scale": target_relative_loss_scale,
        "initial_coefficient": 1.0,
        "gradient_component": gradient_component,
        "ntk_diagnostic_examples": float(MAX_NTK_EXAMPLES),
        "ntk_energy_threshold": NTK_ENERGY_THRESHOLD,
    }
    return _cchain_hyperparameters(value)
