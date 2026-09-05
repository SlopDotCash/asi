"""Protocol-extended replay and frozen-feature controls for IPMNIST.

These are permanently nonpromoting adaptations, not reproductions.  The
replay paper's Transformer is represented by a bounded label-attention
context plus an independently ablatable replay gradient.  RanDumb is adapted
as a frozen random Fourier extractor with an online linear head.  RanPAC is a
frozen random-ReLU projection with a recursive ridge readout.  PROL is exposed
only as an architecture proxy (frozen extractor, prompt/affine state, hard-soft
online updates): no pretrained vision checkpoint is imported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.benchmarks.upgd_ipmnist import (
    _PLASTICITY_LOSS_FLOOR,
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    LearnerUpdateResult,
    _make_adamw_learner,
    cross_entropy_loss,
    mlp_logits,
)
from alberta_framework.core.update_safety import floating_tree_is_finite, select_transaction

REPLAY_PAPER_REVISION: Final = "arXiv:2503.20018v1"
REPLAY_OFFICIAL_CODE: Final = "not-disclosed-as-of-2026-08-17"
RANDUMB_PAPER_REVISION: Final = "arXiv:2402.08823v3"
RANDUMB_COMMIT: Final = "14a51ee0c045bff642f6ffbfe481efa4d49a3033"
RANPAC_PAPER_REVISION: Final = "arXiv:2307.02251v3"
RANPAC_COMMIT: Final = "cf4b301d18b0c27db030f4371b72b768005ae58a"
PROL_PAPER_REVISION: Final = "arXiv:2507.12305v1"
PROL_COMMIT: Final = "bfff8418a4f603a24ae578f1e108bfac89af1e18"

REPLAY_CAPACITY: Final = 32
FROZEN_FEATURE_DIM: Final = 64
_PARAM_KEYS: Final = frozenset(("w1", "b1", "w2", "b2", "w3", "b3"))
_DOMAIN_RANDUMB: Final = 0x52414E44
_DOMAIN_RANPAC: Final = 0x52504143
_DOMAIN_PROL: Final = 0x50524F4C
_INT32_MAX: Final = (1 << 31) - 1


def _skip_zero_scale(scale: float, value: Array) -> Array:
    """Keep finite multiplication exact while repairing zero-scaled poison."""
    product = jnp.asarray(scale, dtype=value.dtype) * value
    if scale != 0.0:
        return product
    return jnp.where(jnp.isfinite(value), product, jnp.zeros_like(value))


@chex.dataclass(frozen=True, mappable_dataclass=False)
class ReplayContextState:
    optimizer_state: dict[str, Any]
    examples: Array
    labels: Array
    count: Array
    cursor: Array
    update_count: Array


@chex.dataclass(frozen=True, mappable_dataclass=False)
class FrozenFeatureState:
    projection: Array
    phase: Array
    inverse_covariance: Array
    readout: Array
    prompt: Array
    scale: Array
    shift: Array
    update_count: Array


def _key_from_params(params: dict[str, Array], domain: int) -> Array:
    bits = jax.lax.bitcast_convert_type(params["w1"].reshape(-1), jnp.uint32)
    key = jr.fold_in(jr.key(jnp.uint32(domain)), bits[0])
    return jr.fold_in(key, bits[-1])


def _metrics(logits: Array, logits_after: Array, y: Array) -> tuple[Array, Array, Array]:
    loss = -jax.nn.log_softmax(logits)[y]
    loss_after = -jax.nn.log_softmax(logits_after)[y]
    return (
        (jnp.argmax(logits) == y).astype(jnp.float32),
        loss,
        jnp.clip(1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0),
    )


def replay_hyperparameters(*, replay_update: float, context: float) -> dict[str, float]:
    if replay_update not in (0.0, 1.0) or context not in (0.0, 1.0):
        raise ValueError("replay_update and context must be exact binary floats")
    return {
        **ADAMW_PROTOCOL_HYPERPARAMETERS,
        "replay_capacity": float(REPLAY_CAPACITY),
        "replay_weight": replay_update,
        "context_weight": context,
        "context_temperature": 1.0,
    }


def make_replay_context_learner(
    hp: Mapping[str, float],
) -> tuple[Any, Any]:
    """Build a charged replay/context arm with exact Adam mechanism-off parity."""
    expected_keys = frozenset(
        {
            *ADAMW_PROTOCOL_HYPERPARAMETERS,
            "replay_capacity",
            "replay_weight",
            "context_weight",
            "context_temperature",
        }
    )
    if type(hp) is not dict or frozenset(hp) != expected_keys:
        raise ValueError("replay/context hyperparameters must match the exact protocol")
    if any(type(value) is not float or not math.isfinite(value) for value in hp.values()):
        raise ValueError("replay/context hyperparameters must be exact finite floats")
    if any(hp[name] != expected for name, expected in ADAMW_PROTOCOL_HYPERPARAMETERS.items()):
        raise ValueError("replay/context Adam constants drift from the matched control")
    if hp["replay_capacity"] != float(REPLAY_CAPACITY):
        raise ValueError("replay capacity drift")
    if hp["replay_weight"] not in (0.0, 1.0) or hp["context_weight"] not in (0.0, 1.0):
        raise ValueError("replay/context switches must be binary")
    if hp["context_temperature"] != 1.0:
        raise ValueError("context temperature drift")
    adam_hp = {name: hp[name] for name in ADAMW_PROTOCOL_HYPERPARAMETERS}
    adam_init, adam_step = _make_adamw_learner(adam_hp)

    def init_fn(params: dict[str, Array]) -> ReplayContextState:
        input_dim = params["w1"].shape[0]
        return ReplayContextState(  # type: ignore[call-arg]
            optimizer_state=adam_init(params),
            examples=jnp.zeros((REPLAY_CAPACITY, input_dim), dtype=jnp.float32),
            labels=jnp.zeros((REPLAY_CAPACITY,), dtype=jnp.int32),
            count=jnp.asarray(0, dtype=jnp.int32),
            cursor=jnp.asarray(0, dtype=jnp.int32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def contextual_logits(params: dict[str, Array], state: ReplayContextState, x: Array) -> Array:
        base = mlp_logits(params, x)
        safe_count = jnp.clip(state.count, 0, REPLAY_CAPACITY)
        mask = jnp.arange(REPLAY_CAPACITY, dtype=jnp.int32) < safe_count
        similarity = state.examples @ x / jnp.sqrt(jnp.float32(x.shape[0]))
        attention = jax.nn.softmax(jnp.where(mask, similarity, -1e30)) * mask
        attention = attention / jnp.maximum(jnp.sum(attention), 1.0)
        context_probability = attention @ jax.nn.one_hot(
            state.labels, base.shape[0], dtype=jnp.float32
        )
        context_logits = jnp.log(context_probability + 1e-6)
        enabled = (safe_count > 0).astype(jnp.float32) * hp["context_weight"]
        return cast(Array, base + enabled * context_logits)

    def step_fn(
        params: dict[str, Array], state: ReplayContextState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], ReplayContextState, tuple[Array, Array, Array]]:
        del key
        if type(state) is not ReplayContextState:
            raise ValueError("state must be an exact ReplayContextState")
        if (
            state.examples.shape != (REPLAY_CAPACITY, params["w1"].shape[0])
            or state.labels.shape != (REPLAY_CAPACITY,)
            or state.count.shape != ()
            or state.cursor.shape != ()
            or state.update_count.shape != ()
        ):
            raise ValueError("ReplayContextState shapes drift from the protocol")
        safe_count = jnp.clip(state.count, 0, REPLAY_CAPACITY)
        safe_cursor = jnp.clip(state.cursor, 0, REPLAY_CAPACITY - 1)
        safe_update_count = jnp.clip(state.update_count, 0, _INT32_MAX - 1)
        logits = contextual_logits(params, state, x)
        (_, _), task_gradient = jax.value_and_grad(cross_entropy_loss, has_aux=True)(params, x, y)
        mask = (jnp.arange(REPLAY_CAPACITY, dtype=jnp.int32) < safe_count).astype(jnp.float32)

        def replay_loss(candidate: dict[str, Array]) -> Array:
            logits_batch = jax.vmap(mlp_logits, in_axes=(None, 0))(candidate, state.examples)
            losses = -jax.nn.log_softmax(logits_batch)[jnp.arange(REPLAY_CAPACITY), state.labels]
            return jnp.sum(mask * losses) / jnp.maximum(safe_count.astype(jnp.float32), 1.0)

        replay_gradient = jax.grad(replay_loss)(params)
        combined = {
            name: task_gradient[name]
            + _skip_zero_scale(hp["replay_weight"], replay_gradient[name])
            for name in _PARAM_KEYS
        }
        update = adam_step(params, state.optimizer_state, combined, jr.key(0))
        if type(update) is not LearnerUpdateResult:
            raise TypeError("matched Adam returned an unsupported update")
        candidate_examples = state.examples.at[safe_cursor].set(x)
        candidate_labels = state.labels.at[safe_cursor].set(y)
        candidate_state = ReplayContextState(  # type: ignore[call-arg]
            optimizer_state=update.state,
            examples=candidate_examples,
            labels=candidate_labels,
            count=jnp.minimum(safe_count + 1, REPLAY_CAPACITY),
            cursor=(safe_cursor + 1) % REPLAY_CAPACITY,
            update_count=safe_update_count + 1,
        )
        valid = (
            update.update_applied
            & floating_tree_is_finite(state)
            & floating_tree_is_finite(candidate_state)
            & (state.count >= 0)
            & (state.count <= REPLAY_CAPACITY)
            & (state.cursor >= 0)
            & (state.cursor < REPLAY_CAPACITY)
            & (state.update_count >= 0)
            & (state.update_count < _INT32_MAX)
            & jnp.all(state.labels >= 0)
            & jnp.all(state.labels < params["b3"].shape[0])
        )
        new_params = select_transaction(valid, update.params, params)
        new_state = select_transaction(valid, candidate_state, state)
        logits_after = contextual_logits(new_params, new_state, x)
        return new_params, new_state, _metrics(logits, logits_after, y)

    return init_fn, step_fn


def frozen_hyperparameters(*, method: float, mechanism: float) -> dict[str, float]:
    if method not in (0.0, 1.0, 2.0) or mechanism not in (0.0, 1.0):
        raise ValueError("frozen method/mechanism encodings are outside the protocol")
    return {
        "method": method,
        "mechanism": mechanism,
        "feature_dim": float(FROZEN_FEATURE_DIM),
        "head_step_size": 1e-2,
        "ridge": 1.0,
        "hard_loss_threshold": 1.0,
        "soft_update_scale": 0.1,
        "external_pretraining_examples": 0.0,
        "external_pretraining_updates": 0.0,
        "noise_std": 0.0,
    }


def make_frozen_feature_learner(hp: Mapping[str, float]) -> tuple[Any, Any]:
    """Build RanDumb, RanPAC, or a PROL architecture-only proxy."""
    hp_keys = frozenset(frozen_hyperparameters(method=0.0, mechanism=0.0))
    if type(hp) is not dict or frozenset(hp) != hp_keys:
        raise ValueError("frozen-feature hyperparameters must match the exact keys")
    if any(type(value) is not float or not math.isfinite(value) for value in hp.values()):
        raise ValueError("frozen-feature hyperparameters must be exact finite floats")
    expected = frozen_hyperparameters(method=hp["method"], mechanism=hp["mechanism"])
    if dict(hp) != expected:
        raise ValueError("frozen-feature hyperparameters drift from the protocol")
    method = int(hp["method"])
    mechanism = hp["mechanism"]
    m = int(hp["feature_dim"])
    domain = (_DOMAIN_RANDUMB, _DOMAIN_RANPAC, _DOMAIN_PROL)[method]

    def init_fn(params: dict[str, Array]) -> FrozenFeatureState:
        input_dim = params["w1"].shape[0]
        n_classes = params["w3"].shape[1]
        key_projection, key_phase = jr.split(_key_from_params(params, domain))
        projection = jr.normal(key_projection, (m, input_dim), jnp.float32) / jnp.sqrt(
            jnp.float32(input_dim)
        )
        return FrozenFeatureState(  # type: ignore[call-arg]
            projection=projection,
            phase=jr.uniform(key_phase, (m,), jnp.float32, 0.0, 2.0 * math.pi),
            inverse_covariance=jnp.eye(m, dtype=jnp.float32) / hp["ridge"],
            readout=jnp.zeros((m, n_classes), dtype=jnp.float32),
            prompt=jnp.zeros((m,), dtype=jnp.float32),
            scale=jnp.ones((m,), dtype=jnp.float32),
            shift=jnp.zeros((m,), dtype=jnp.float32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def features(state: FrozenFeatureState, x: Array) -> Array:
        projected = state.projection @ x
        base = jnp.where(
            method == 0,
            jnp.sqrt(jnp.float32(2.0 / m)) * jnp.cos(projected + state.phase),
            jax.nn.relu(projected),
        )
        if method == 2:
            return cast(
                Array,
                state.scale * (base + mechanism * state.prompt) + mechanism * state.shift,
            )
        return base

    def step_fn(
        params: dict[str, Array], state: FrozenFeatureState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], FrozenFeatureState, tuple[Array, Array, Array]]:
        del key
        if type(state) is not FrozenFeatureState:
            raise ValueError("state must be an exact FrozenFeatureState")
        classes = params["b3"].shape[0]
        input_dim = params["w1"].shape[0]
        if (
            state.projection.shape != (m, input_dim)
            or state.phase.shape != (m,)
            or state.inverse_covariance.shape != (m, m)
            or state.readout.shape != (m, classes)
            or state.prompt.shape != (m,)
            or state.scale.shape != (m,)
            or state.shift.shape != (m,)
            or state.update_count.shape != ()
        ):
            raise ValueError("FrozenFeatureState shapes drift from the protocol")
        safe_update_count = jnp.clip(state.update_count, 0, _INT32_MAX - 1)
        phi = features(state, x)
        logits = state.readout.T @ phi
        target = jax.nn.one_hot(y, state.readout.shape[1], dtype=jnp.float32)
        error = target - jax.nn.softmax(logits)
        if method == 1:
            pp = state.inverse_covariance @ phi
            gain = pp / (1.0 + phi @ pp)
            readout = state.readout + jnp.outer(gain, target - logits)
            inverse = state.inverse_covariance - jnp.outer(gain, pp)
            prompt, scale, shift = state.prompt, state.scale, state.shift
        else:
            loss = -jax.nn.log_softmax(logits)[y]
            update_scale = jnp.where(
                (method == 2) & (loss <= hp["hard_loss_threshold"]),
                hp["soft_update_scale"],
                1.0,
            )
            readout = state.readout + hp["head_step_size"] * update_scale * jnp.outer(phi, error)
            inverse = state.inverse_covariance
            if method == 2:
                feature_gradient = state.readout @ (-error)
                # mechanism=0 is the declared mechanism-off reduction; skip
                # before ``0 * inf`` poisons prompt/scale/shift.
                feature_step = _skip_zero_scale(
                    mechanism * hp["head_step_size"],
                    update_scale * feature_gradient,
                )
                prompt = state.prompt - feature_step
                scale = state.scale - feature_step * (phi - state.shift)
                shift = state.shift - feature_step
            else:
                prompt, scale, shift = state.prompt, state.scale, state.shift
        candidate = FrozenFeatureState(  # type: ignore[call-arg]
            projection=state.projection,
            phase=state.phase,
            inverse_covariance=0.5 * (inverse + inverse.T),
            readout=readout,
            prompt=prompt,
            scale=scale,
            shift=shift,
            update_count=safe_update_count + 1,
        )
        valid = (
            floating_tree_is_finite(state)
            & floating_tree_is_finite(candidate)
            & (state.update_count >= 0)
            & (state.update_count < _INT32_MAX)
        )
        new_state = select_transaction(valid, candidate, state)
        logits_after = new_state.readout.T @ features(new_state, x)
        return params, new_state, _metrics(logits, logits_after, y)

    return init_fn, step_fn
