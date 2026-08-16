# mypy: disable-error-code="call-arg"
"""Unit contracts for the bounded recurrent latent world-model ensemble."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    EVIDENCE_LEVEL,
    SCIENTIFIC_PROMOTION_ALLOWED,
    RecurrentLatentDecisionCache,
    RecurrentLatentTransitionRecord,
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleConfig,
    RecurrentLatentWorldModelEnsembleState,
    _tree_l2_norm,
    load_recurrent_latent_world_model_ensemble_checkpoint,
    save_recurrent_latent_world_model_ensemble_checkpoint,
)

pytestmark = pytest.mark.unit

OBSERVATION = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
ACTION = jnp.asarray(1, dtype=jnp.int32)
BOOTSTRAP = jnp.asarray((0.75, 0.125), dtype=jnp.float32)


def _config(**overrides: Any) -> RecurrentLatentWorldModelEnsembleConfig:
    values: dict[str, Any] = {
        "observation_dim": 2,
        "n_actions": 2,
        "latent_dim": 3,
        "ensemble_size": 3,
        "learning_rate": 0.01,
        "bootstrap_probability": 0.8,
        "uncertainty_warmup_steps": 1,
        "max_updates": 8,
    }
    values.update(overrides)
    return RecurrentLatentWorldModelEnsembleConfig(**values)


def _transition(
    *,
    observation: jax.Array = OBSERVATION,
    action: jax.Array = ACTION,
    reward: float = 0.5,
    discount: float = 0.9,
    terminated: bool = False,
    truncated: bool = False,
    bootstrap_observation: jax.Array = BOOTSTRAP,
    next_decision_observation: jax.Array = BOOTSTRAP,
) -> RecurrentLatentTransitionRecord:
    return RecurrentLatentTransitionRecord(
        observation=observation,
        action=action,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(truncated, dtype=jnp.bool_),
        bootstrap_observation=bootstrap_observation,
        next_decision_observation=next_decision_observation,
    )


def _decision(
    model: RecurrentLatentWorldModelEnsemble,
    state: RecurrentLatentWorldModelEnsembleState,
    observation: jax.Array = OBSERVATION,
    action: jax.Array = ACTION,
) -> RecurrentLatentDecisionCache:
    return cast(
        RecurrentLatentDecisionCache,
        model.decide(state, model.start(state, observation), action),
    )


def _assert_tree_equal(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert cast(Any, left_tree) == right_tree
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jnp.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jr.key_data(left_leaf), jr.key_data(right_leaf))
        else:
            np.testing.assert_array_equal(left_leaf, right_leaf)


def _assert_tree_close(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert cast(Any, left_tree) == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jnp.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jr.key_data(left_leaf), jr.key_data(right_leaf))
        else:
            np.testing.assert_allclose(left_leaf, right_leaf, rtol=1.0e-6, atol=1.0e-6)


def test_config_roundtrip_is_strict_bounded_and_development_only() -> None:
    config = _config()
    restored = RecurrentLatentWorldModelEnsembleConfig.from_config(config.to_config())
    assert restored == config
    assert EVIDENCE_LEVEL == "L0"
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert config.target_dim == 4
    assert config.raw_output_dim == 8

    malformed = dict(config.to_config())
    malformed["task_id"] = 7
    with pytest.raises(ValueError, match="fields"):
        RecurrentLatentWorldModelEnsembleConfig.from_config(malformed)
    malformed = dict(config.to_config())
    malformed["learning_rate"] = "0.01"
    with pytest.raises(ValueError, match="real non-boolean"):
        RecurrentLatentWorldModelEnsembleConfig.from_config(malformed)
    with pytest.raises(ValueError, match="at least 2"):
        _config(ensemble_size=1)
    with pytest.raises(ValueError, match="warmup"):
        _config(uncertainty_warmup_steps=9)
    with pytest.raises(ValueError, match="variance_floor"):
        _config(variance_floor=2.0, max_variance=1.0)
    with pytest.raises(ValueError, match="bootstrap_probability"):
        _config(bootstrap_probability=1.0)


def test_initialization_is_distinct_fixed_width_and_exactly_accounted() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(0))
    assert bool(model.state_valid(state))
    assert not np.array_equal(
        state.member_parameters[0].mean_kernel,
        state.member_parameters[1].mean_kernel,
    )
    budget = model.resource_budget(state)
    assert budget.trainable_scalars_per_member == model.config.trainable_scalars_per_member
    assert budget.total_trainable_scalars == (
        model.config.ensemble_size * model.config.trainable_scalars_per_member
    )
    assert budget.persistent_state_bytes == model.config.state_nbytes
    assert budget.bootstrap_prng_keys == 1
    assert budget.bootstrap_prng_uint32_scalars == 2
    assert budget.member_gradient_candidates_per_event == model.config.ensemble_size
    assert budget.max_member_parameter_updates_per_event == model.config.ensemble_size
    assert budget.recurrent_advances_per_accepted_event == 1
    assert budget.replay_capacity == 0


_REAL_SCALAR_FIELDS = (
    "learning_rate",
    "variance_floor",
    "max_variance",
    "initialization_scale",
    "gradient_clip_norm",
    "max_raw_gradient_norm",
    "max_input_magnitude",
    "max_parameter_magnitude",
    "max_prediction_magnitude",
    "max_loss_magnitude",
    "bootstrap_probability",
)

_IN_DOMAIN_SCALARS = {
    "learning_rate": 0.01,
    "variance_floor": 1.0e-3,
    "max_variance": 100.0,
    "initialization_scale": 0.2,
    "gradient_clip_norm": 10.0,
    "max_raw_gradient_norm": 100_000.0,
    "max_input_magnitude": 1_000.0,
    "max_parameter_magnitude": 10_000.0,
    "max_prediction_magnitude": 10_000.0,
    "max_loss_magnitude": 100_000_000.0,
    "bootstrap_probability": 0.8,
}


class _FloatSpoof:
    """Not a Real at all, but reports ``float`` through ``__class__``."""

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return self._value


class _RaisingFloatSpoof:
    """A ``__class__`` spoof whose ``__float__`` hook raises when trusted."""

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        raise RuntimeError("untrusted __float__ hook executed")


@pytest.mark.parametrize("field", _REAL_SCALAR_FIELDS)
def test_config_rejects_objects_that_only_spoof_float_through_class(field: str) -> None:
    """An in-domain host value must not smuggle a non-Real type past validation."""
    spoof = _FloatSpoof(_IN_DOMAIN_SCALARS[field])
    with pytest.raises(ValueError, match="real non-boolean"):
        _config(**{field: spoof})


@pytest.mark.parametrize("field", _REAL_SCALAR_FIELDS)
def test_config_raising_spoofed_scalar_stays_a_value_error(field: str) -> None:
    """A spoof with a raising ``__float__`` must not leak its raw exception."""
    with pytest.raises(ValueError, match="real non-boolean"):
        _config(**{field: _RaisingFloatSpoof()})


def test_config_rejects_an_initial_variance_bias_outside_the_parameter_bound() -> None:
    """The deterministic variance-head initializer must fit inside max_parameter_magnitude."""
    expected = (
        r"^initial variance bias magnitude 5\.75646[0-9]* exceeds max_parameter_magnitude=1\.0$"
    )
    with pytest.raises(ValueError, match=expected):
        _config(max_parameter_magnitude=1.0)


def test_init_refuses_to_return_a_state_it_would_reject() -> None:
    """A bound that admits the deterministic bias can still be breached by drawn kernels."""
    model = RecurrentLatentWorldModelEnsemble(
        _config(initialization_scale=50.0, max_parameter_magnitude=6.0)
    )
    with pytest.raises(
        ValueError,
        match=r"^initialized parameters exceed max_parameter_magnitude=6\.0; "
        r"lower initialization_scale or raise the bound$",
    ):
        model.init(jr.key(0))


def test_init_state_is_valid_for_every_accepted_default_scale() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    for seed in range(4):
        assert bool(model.state_valid(model.init(jr.key(seed))))


def test_initial_variance_bias_bound_is_exact_at_its_float32_endpoint() -> None:
    """The bound is compared against the stored float32 bias, not its binary64 origin."""
    default = _config()
    sink = float(np.float32(abs(default.initial_variance_logit)))
    assert default.initial_variance_bias_magnitude == sink
    assert sink != abs(default.initial_variance_logit)

    endpoint = RecurrentLatentWorldModelEnsemble(_config(max_parameter_magnitude=sink))
    state = endpoint.init(jr.key(0))
    assert bool(endpoint.state_valid(state))
    stored = float(jnp.max(jnp.abs(state.member_parameters[0].variance_bias)))
    assert stored == sink

    below = float(np.nextafter(np.float32(sink), np.float32(0.0)))
    with pytest.raises(ValueError, match="initial variance bias magnitude"):
        _config(max_parameter_magnitude=below)


def test_init_is_a_host_side_entry_point() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(model.init)(jr.key(0))


def test_start_and_decide_are_read_only_predict_before_update_caches() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(0))
    original = jax.tree_util.tree_map(lambda value: value, state)
    start = model.start(state, OBSERVATION)
    decision = model.decide(state, start, ACTION)
    prediction = decision.prediction

    _assert_tree_equal(state, original)
    assert bool(start.valid)
    assert bool(decision.valid)
    np.testing.assert_array_equal(start.observation, OBSERVATION)
    np.testing.assert_array_equal(decision.observation, OBSERVATION)
    assert int(decision.action) == int(ACTION)
    assert prediction.member_raw_outputs.shape == (3, 8)
    assert prediction.member_mean_predictions.shape == (3, 4)
    assert prediction.member_next_hidden_states.shape == (3, 3)
    assert prediction.member_aleatoric_variances.shape == (3, 4)
    assert np.all(prediction.member_aleatoric_variances >= model.config.variance_floor)
    assert np.all(prediction.member_aleatoric_variances <= model.config.max_variance)
    assert np.all(prediction.member_continuations >= 0.0)
    assert np.all(prediction.member_continuations <= 1.0)
    assert bool(prediction.availability.prediction)
    assert not bool(prediction.warmup_ready)
    assert not bool(prediction.availability.epistemic)
    assert not bool(prediction.availability.aleatoric)
    assert float(prediction.aleatoric_uncertainty) > 0.0


def test_nonboundary_update_commits_one_recurrent_advance_and_masked_nll_updates() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(0))
    decision = _decision(model, state)
    result = model.update(state, decision, _transition())

    assert bool(result.diagnostics.applied)
    assert bool(result.diagnostics.recurrent_advanced_once)
    assert not bool(result.diagnostics.recurrent_reset)
    assert int(result.state.event_count) == 1
    assert int(result.state.recurrent_advance_count) == 1
    assert int(result.state.boundary_count) == 0
    np.testing.assert_array_equal(
        result.state.member_hidden_states,
        decision.prediction.member_next_hidden_states,
    )
    np.testing.assert_array_equal(
        result.targets,
        jnp.asarray((0.75, 0.125, 0.5, 0.9), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result.prediction.member_raw_outputs,
        decision.prediction.member_raw_outputs,
    )
    np.testing.assert_array_equal(result.next_start_cache.observation, BOOTSTRAP)
    np.testing.assert_array_equal(
        result.state.member_update_counts,
        result.bootstrap_mask.astype(jnp.int32),
    )
    assert np.any(result.bootstrap_mask)
    assert np.any(~np.asarray(result.bootstrap_mask))
    for index, applied in enumerate(np.asarray(result.bootstrap_mask)):
        before = state.member_parameters[index]
        after = result.state.member_parameters[index]
        if applied:
            assert not np.array_equal(before.variance_bias, after.variance_bias)
        else:
            _assert_tree_equal(before, after)
    assert bool(result.representation_gradient_available)
    assert np.all(np.isfinite(result.representation_gradient))

    next_decision = model.decide(result.state, result.next_start_cache, ACTION)
    assert bool(next_decision.prediction.warmup_ready)
    assert bool(next_decision.prediction.availability.epistemic)
    assert bool(next_decision.prediction.availability.aleatoric)


@pytest.mark.parametrize(
    ("terminated", "truncated", "discount"),
    [(True, False, 0.0), (False, True, 0.9)],
)
def test_boundary_uses_final_target_then_reset_observation(
    terminated: bool,
    truncated: bool,
    discount: float,
) -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(2))
    decision = _decision(model, state)
    final_observation = jnp.asarray((3.0, 4.0), dtype=jnp.float32)
    reset_observation = jnp.asarray((-7.0, 8.0), dtype=jnp.float32)
    result = model.update(
        state,
        decision,
        _transition(
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            bootstrap_observation=final_observation,
            next_decision_observation=reset_observation,
        ),
    )

    assert bool(result.diagnostics.applied)
    assert bool(result.diagnostics.recurrent_advanced_once)
    assert bool(result.diagnostics.recurrent_reset)
    np.testing.assert_array_equal(result.targets[:2], final_observation)
    np.testing.assert_array_equal(result.state.member_hidden_states, np.zeros((3, 3)))
    np.testing.assert_array_equal(result.next_start_cache.observation, reset_observation)
    assert int(result.state.boundary_count) == 1

    # The returned cache is exactly the reset-state/start cache, not a cache of
    # the final observation that supplied the learning target.
    fresh = model.start(result.state, reset_observation)
    _assert_tree_equal(result.next_start_cache, fresh)
    next_from_result = model.decide(result.state, result.next_start_cache, ACTION)
    next_from_fresh = model.decide(result.state, fresh, ACTION)
    _assert_tree_equal(next_from_result, next_from_fresh)


def test_off_boundary_reset_substitution_and_boundary_discount_errors_reject_atomically() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(3))
    decision = _decision(model, state)
    wrong_reset = jnp.asarray((-9.0, -9.0), dtype=jnp.float32)
    invalid_records = (
        _transition(next_decision_observation=wrong_reset),
        _transition(terminated=True, discount=0.9),
        _transition(truncated=True, discount=0.0),
    )
    for record in invalid_records:
        rejected = model.update(state, decision, record)
        assert bool(rejected.diagnostics.rejected)
        assert not bool(rejected.diagnostics.boundary_semantics_valid)
        _assert_tree_equal(rejected.state, state)
        assert not bool(rejected.prediction.availability.prediction)
        assert not bool(rejected.representation_gradient_available)


def test_member_gradient_validity_is_finiteness_only_not_a_raw_norm_ceiling() -> None:
    """Regression for issue #366.

    ``residual / variance`` scaling means a well-trained low-noise member can
    legally produce a per-member raw NLL gradient far above any fixed
    ``max_raw_gradient_norm`` ceiling on an ordinary in-bounds transition.
    ``gradient_clip_norm`` already bounds the committed step regardless of
    raw scale, so the finite-but-huge case must still apply, not reject.
    """
    model = RecurrentLatentWorldModelEnsemble(
        _config(gradient_clip_norm=1.0, max_raw_gradient_norm=5_000.0)
    )
    state = model.init(jr.key(20))
    decision = _decision(model, state)
    far_bootstrap = jnp.asarray((900.0, -900.0), dtype=jnp.float32)
    stopped_targets = jnp.concatenate(
        (far_bootstrap, jnp.asarray((0.5, 0.9), dtype=jnp.float32))
    )
    member_norms = []
    for index in range(model.config.ensemble_size):
        _, gradient = jax.value_and_grad(model._member_nll)(  # noqa: SLF001
            state.member_parameters[index],
            state.member_hidden_states[index],
            OBSERVATION,
            ACTION,
            stopped_targets,
        )
        member_norms.append(float(_tree_l2_norm(gradient)))

    # The whole point of the regression: the raw gradient genuinely, legally
    # exceeds the configured ceiling before this fix would have rejected it.
    assert all(norm > model.config.max_raw_gradient_norm for norm in member_norms)

    result = model.update(
        state,
        decision,
        _transition(
            bootstrap_observation=far_bootstrap,
            next_decision_observation=far_bootstrap,
        ),
    )
    assert bool(result.diagnostics.applied)
    assert bool(jnp.all(result.diagnostics.member_gradients_valid))
    assert bool(result.diagnostics.candidate_state_valid)


def test_late_numeric_rejection_returns_a_retryable_authoritative_cache() -> None:
    """A trusted off-boundary event may recover after a late numerical veto."""
    model = RecurrentLatentWorldModelEnsemble(
        _config(
            gradient_clip_norm=1.0,
            max_raw_gradient_norm=1.0,
            max_loss_magnitude=1.0e8,
        )
    )
    state = model.init(jr.key(21))
    decision = _decision(model, state)
    far_bootstrap = jnp.asarray((50.0, -50.0), dtype=jnp.float32)

    rejected = model.update(
        state,
        decision,
        _transition(
            bootstrap_observation=far_bootstrap,
            next_decision_observation=far_bootstrap,
        ),
    )
    assert bool(rejected.diagnostics.rejected)
    assert bool(rejected.diagnostics.state_valid)
    assert bool(rejected.diagnostics.cache_valid)
    assert bool(rejected.diagnostics.input_valid)
    assert bool(rejected.diagnostics.ownership_valid)
    assert bool(rejected.diagnostics.boundary_semantics_valid)
    assert bool(rejected.diagnostics.capacity_available)
    assert bool(rejected.diagnostics.cached_prediction_exact)
    assert bool(rejected.diagnostics.predictions_valid)
    assert not bool(rejected.diagnostics.representation_gradient_valid)
    _assert_tree_equal(rejected.state, state)

    assert bool(rejected.next_start_cache.valid)
    np.testing.assert_array_equal(rejected.next_start_cache.observation, far_bootstrap)
    np.testing.assert_array_equal(
        rejected.next_start_cache.owner_hidden_states, state.member_hidden_states
    )
    assert int(rejected.next_start_cache.owner_event_count) == int(state.event_count)

    # Prove the chain is genuinely un-poisoned: decide/update from the
    # recovered cache work normally, evaluated on their own merits.
    next_decision = model.decide(state, rejected.next_start_cache, ACTION)
    assert bool(next_decision.valid)
    recovered = model.update(
        state,
        next_decision,
        _transition(
            observation=far_bootstrap,
            bootstrap_observation=far_bootstrap,
            next_decision_observation=far_bootstrap,
        ),
    )
    assert bool(recovered.diagnostics.applied)


def test_stale_decision_cannot_launder_a_new_observation_into_an_owned_cache() -> None:
    """A locally valid cache flag is not authority to mint the next owner."""
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(22))
    decision = _decision(model, state)
    stale_decision = cast(Any, decision).replace(
        owner_event_count=decision.owner_event_count + jnp.int32(1)
    )
    untrusted_next = jnp.asarray((7.0, -8.0), dtype=jnp.float32)

    rejected = model.update(
        state,
        stale_decision,
        _transition(
            bootstrap_observation=untrusted_next,
            next_decision_observation=untrusted_next,
        ),
    )
    assert bool(rejected.diagnostics.state_valid)
    assert bool(rejected.diagnostics.cache_valid)
    assert bool(rejected.diagnostics.input_valid)
    assert not bool(rejected.diagnostics.ownership_valid)
    assert not bool(rejected.next_start_cache.valid)
    assert not bool(model.decide(state, rejected.next_start_cache, ACTION).valid)


def test_invalid_transition_or_boundary_cannot_mint_a_recovery_cache() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(23))
    decision = _decision(model, state)
    nonfinite_next = jnp.asarray((jnp.inf, 0.0), dtype=jnp.float32)
    wrong_reset = jnp.asarray((-9.0, -9.0), dtype=jnp.float32)

    invalid_input = model.update(
        state,
        decision,
        _transition(
            bootstrap_observation=nonfinite_next,
            next_decision_observation=nonfinite_next,
        ),
    )
    assert not bool(invalid_input.diagnostics.input_valid)
    assert not bool(invalid_input.next_start_cache.valid)

    invalid_boundary = model.update(
        state,
        decision,
        _transition(next_decision_observation=wrong_reset),
    )
    assert not bool(invalid_boundary.diagnostics.boundary_semantics_valid)
    assert not bool(invalid_boundary.next_start_cache.valid)


def test_tampered_cached_prediction_cannot_mint_a_recovery_cache() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(24))
    decision = _decision(model, state)
    tampered_prediction = cast(Any, decision.prediction).replace(
        mean_reward=decision.prediction.mean_reward + jnp.float32(1.0)
    )
    tampered_decision = cast(Any, decision).replace(prediction=tampered_prediction)

    rejected = model.update(state, tampered_decision, _transition())
    assert bool(rejected.diagnostics.ownership_valid)
    assert not bool(rejected.diagnostics.cached_prediction_exact)
    assert not bool(rejected.next_start_cache.valid)


def test_late_boundary_rejection_cannot_skip_the_required_recurrent_reset() -> None:
    model = RecurrentLatentWorldModelEnsemble(
        _config(
            gradient_clip_norm=1.0,
            max_raw_gradient_norm=1.0,
            max_loss_magnitude=1.0e8,
        )
    )
    state = model.init(jr.key(25))
    decision = _decision(model, state)
    far_bootstrap = jnp.asarray((50.0, -50.0), dtype=jnp.float32)
    reset_observation = jnp.asarray((-2.0, 3.0), dtype=jnp.float32)

    rejected = model.update(
        state,
        decision,
        _transition(
            discount=0.0,
            terminated=True,
            bootstrap_observation=far_bootstrap,
            next_decision_observation=reset_observation,
        ),
    )
    assert bool(rejected.diagnostics.boundary_semantics_valid)
    assert bool(rejected.diagnostics.cached_prediction_exact)
    assert not bool(rejected.diagnostics.representation_gradient_valid)
    assert not bool(rejected.next_start_cache.valid)


def test_state_invalid_rejection_still_returns_a_non_recoverable_cache() -> None:
    """A corrupt/invalid *state* must not be re-owned as if it were legal."""
    model = RecurrentLatentWorldModelEnsemble(_config())
    valid_state = model.init(jr.key(26))
    valid_decision = _decision(model, valid_state)
    corrupt_state = cast(Any, valid_state).replace(
        event_count=jnp.asarray(1, dtype=jnp.int32)
    )
    assert not bool(model.state_valid(corrupt_state))

    result = model.update(corrupt_state, valid_decision, _transition())
    assert not bool(result.diagnostics.state_valid)
    assert bool(result.diagnostics.rejected)
    assert not bool(result.next_start_cache.valid)


def test_exact_observation_action_and_cache_ownership_reject_stale_or_tampered_inputs() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(4))
    decision = _decision(model, state)
    accepted = model.update(state, decision, _transition())
    assert bool(accepted.diagnostics.applied)

    stale = model.update(accepted.state, decision, _transition())
    assert not bool(stale.diagnostics.ownership_valid)
    _assert_tree_equal(stale.state, accepted.state)

    mismatched_observation = model.update(
        state,
        decision,
        _transition(observation=jnp.asarray((0.25, -0.25), dtype=jnp.float32)),
    )
    assert not bool(mismatched_observation.diagnostics.ownership_valid)
    _assert_tree_equal(mismatched_observation.state, state)

    mismatched_action = model.update(
        state,
        decision,
        _transition(action=jnp.asarray(0, dtype=jnp.int32)),
    )
    assert not bool(mismatched_action.diagnostics.ownership_valid)
    _assert_tree_equal(mismatched_action.state, state)

    tampered_prediction = cast(Any, decision.prediction).replace(
        mean_reward=decision.prediction.mean_reward + 1.0
    )
    tampered_cache = cast(Any, decision).replace(prediction=tampered_prediction)
    tampered = model.update(state, tampered_cache, _transition())
    assert not bool(tampered.diagnostics.cached_prediction_exact)
    _assert_tree_equal(tampered.state, state)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.replace(reward=jnp.asarray(jnp.nan, dtype=jnp.float32)),
        lambda record: record.replace(
            bootstrap_observation=jnp.asarray((1.0e6, 0.0), dtype=jnp.float32)
        ),
        lambda record: record.replace(action=jnp.asarray(5, dtype=jnp.int32)),
    ],
)
def test_invalid_numeric_inputs_preserve_every_state_leaf_and_rng(
    mutate: Callable[[RecurrentLatentTransitionRecord], RecurrentLatentTransitionRecord],
) -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(5))
    decision = _decision(model, state)
    rejected = model.update(state, decision, mutate(_transition()))
    assert bool(rejected.diagnostics.rejected)
    assert not bool(rejected.diagnostics.input_valid) or not bool(
        rejected.diagnostics.ownership_valid
    )
    _assert_tree_equal(rejected.state, state)
    np.testing.assert_array_equal(
        jr.key_data(rejected.state.bootstrap_key),
        jr.key_data(state.bootstrap_key),
    )


def test_capacity_exhaustion_is_a_strict_noop() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config(max_updates=1, uncertainty_warmup_steps=0))
    initial = model.init(jr.key(6))
    first = model.update(initial, _decision(model, initial), _transition())
    assert bool(first.diagnostics.applied)
    second_decision = model.decide(first.state, first.next_start_cache, ACTION)
    second = model.update(
        first.state,
        second_decision,
        _transition(observation=BOOTSTRAP),
    )
    assert not bool(second.diagnostics.capacity_available)
    assert bool(second.diagnostics.rejected)
    _assert_tree_equal(second.state, first.state)


def test_dynamically_corrupt_state_is_an_atomic_noop_including_rng() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    valid_state = model.init(jr.key(61))
    valid_decision = _decision(model, valid_state)

    counter_corrupt = cast(Any, valid_state).replace(
        event_count=jnp.asarray(1, dtype=jnp.int32)
    )
    nan_parameters = cast(Any, valid_state.member_parameters[0]).replace(
        mean_bias=valid_state.member_parameters[0].mean_bias.at[0].set(jnp.nan)
    )
    nan_corrupt = cast(Any, valid_state).replace(
        member_parameters=(nan_parameters, *valid_state.member_parameters[1:])
    )
    overbound_parameters = cast(Any, valid_state.member_parameters[0]).replace(
        mean_bias=valid_state.member_parameters[0].mean_bias.at[0].set(
            model.config.max_parameter_magnitude + 1.0
        )
    )
    overbound_corrupt = cast(Any, valid_state).replace(
        member_parameters=(overbound_parameters, *valid_state.member_parameters[1:])
    )

    for corrupt_state in (counter_corrupt, nan_corrupt, overbound_corrupt):
        assert not bool(model.state_valid(corrupt_state))
        result = model.update(corrupt_state, valid_decision, _transition())
        assert not bool(result.diagnostics.state_valid)
        assert bool(result.diagnostics.rejected)
        _assert_tree_equal(result.state, corrupt_state)
        np.testing.assert_array_equal(
            jr.key_data(result.state.bootstrap_key),
            jr.key_data(corrupt_state.bootstrap_key),
        )
        assert int(result.state.event_count) == int(corrupt_state.event_count)
        assert int(result.state.recurrent_advance_count) == int(
            corrupt_state.recurrent_advance_count
        )


def test_representation_gradient_is_the_frozen_target_causal_nll_derivative() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(7))
    decision = _decision(model, state)
    result = model.update(state, decision, _transition())
    stopped_targets = jax.lax.stop_gradient(result.targets)

    def objective(observation: jax.Array) -> jax.Array:
        losses = [
            model._member_nll(  # noqa: SLF001 - this is the exact internal contract under test
                state.member_parameters[index],
                state.member_hidden_states[index],
                observation,
                ACTION,
                stopped_targets,
            )
            for index in range(model.config.ensemble_size)
        ]
        return jnp.mean(jnp.stack(losses))

    expected = jax.grad(objective)(OBSERVATION)
    np.testing.assert_allclose(result.representation_gradient, expected, rtol=1e-6, atol=1e-6)


def test_jit_and_scan_match_sequential_cache_state_and_outputs() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config(uncertainty_warmup_steps=0))
    initial = model.init(jr.key(8))
    initial_cache = model.start(initial, OBSERVATION)
    rewards = jnp.asarray((0.1, -0.2, 0.3), dtype=jnp.float32)
    next_observations = jnp.asarray(
        ((0.1, 0.2), (0.3, -0.4), (0.5, 0.6)), dtype=jnp.float32
    )

    def one_step(
        carry: tuple[RecurrentLatentWorldModelEnsembleState, Any],
        values: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[RecurrentLatentWorldModelEnsembleState, Any], tuple[jax.Array, ...]]:
        state, start_cache = carry
        reward, next_observation = values
        decision = model.decide(state, start_cache, ACTION)
        transition = RecurrentLatentTransitionRecord(
            observation=start_cache.observation,
            action=ACTION,
            reward=reward,
            discount=jnp.asarray(0.9, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            bootstrap_observation=next_observation,
            next_decision_observation=next_observation,
        )
        result = model.update(state, decision, transition)
        return (result.state, result.next_start_cache), (
            result.mean_negative_log_likelihood,
            result.representation_gradient,
            result.bootstrap_mask,
            result.diagnostics.applied,
        )

    scan_carry, scan_outputs = jax.jit(
        lambda state, cache: jax.lax.scan(
            one_step,
            (state, cache),
            (rewards, next_observations),
        )
    )(initial, initial_cache)

    sequential_carry: tuple[RecurrentLatentWorldModelEnsembleState, Any] = (
        initial,
        initial_cache,
    )
    sequential_outputs: list[tuple[jax.Array, ...]] = []
    for values in zip(rewards, next_observations, strict=True):
        sequential_carry, outputs = one_step(sequential_carry, values)
        sequential_outputs.append(outputs)
    stacked_outputs = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *sequential_outputs)

    _assert_tree_close(scan_carry, sequential_carry)
    _assert_tree_close(scan_outputs, stacked_outputs)


def test_digest_bound_checkpoint_roundtrip_preserves_exact_future_stream(tmp_path: Path) -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    initial = model.init(jr.key(9))
    first = model.update(initial, _decision(model, initial), _transition())
    checkpoint = tmp_path / "recurrent-ensemble.ckpt"
    save_recurrent_latent_world_model_ensemble_checkpoint(model, first.state, checkpoint)
    restored_model, restored_state = load_recurrent_latent_world_model_ensemble_checkpoint(
        checkpoint
    )
    assert restored_model.to_config() == model.to_config()
    assert restored_model.resource_budget(restored_state) == model.resource_budget(first.state)
    _assert_tree_equal(restored_state, first.state)

    next_record = _transition(observation=BOOTSTRAP)
    original_next = model.update(
        first.state,
        model.decide(first.state, first.next_start_cache, ACTION),
        next_record,
    )
    restored_start = restored_model.start(restored_state, BOOTSTRAP)
    restored_next = restored_model.update(
        restored_state,
        restored_model.decide(restored_state, restored_start, ACTION),
        next_record,
    )
    _assert_tree_equal(original_next, restored_next)


def test_checkpoint_restore_does_not_depend_on_an_unrelated_template_draw(
    tmp_path: Path,
) -> None:
    """A valid persisted draw must restore even when the default template draw is invalid."""
    model = RecurrentLatentWorldModelEnsemble(
        _config(initialization_scale=5.0, max_parameter_magnitude=8.0)
    )
    state = model.init(jr.key(3))
    assert bool(model.state_valid(state))
    with pytest.raises(ValueError, match="initialized parameters exceed"):
        model.init(jr.key(0))

    checkpoint = tmp_path / "seed-dependent-init.ckpt"
    save_recurrent_latent_world_model_ensemble_checkpoint(model, state, checkpoint)

    restored_model, restored_state = load_recurrent_latent_world_model_ensemble_checkpoint(
        checkpoint
    )
    assert restored_model.to_config() == model.to_config()
    _assert_tree_equal(restored_state, state)


def test_checkpoint_rejects_metadata_config_and_resource_tampering(tmp_path: Path) -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(91))
    source = tmp_path / "source.ckpt"
    save_recurrent_latent_world_model_ensemble_checkpoint(model, state, source)

    def tamper_copy(name: str) -> tuple[Path, Path, dict[str, Any]]:
        destination = tmp_path / name
        shutil.copytree(source, destination)
        metadata_path = destination / "metadata" / "metadata"
        payload = cast(dict[str, Any], json.loads(metadata_path.read_text(encoding="utf-8")))
        return destination, metadata_path, payload

    metadata_checkpoint, metadata_path, metadata = tamper_copy("metadata.ckpt")
    metadata["unregistered_claim"] = True
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metadata fields"):
        load_recurrent_latent_world_model_ensemble_checkpoint(metadata_checkpoint)

    config_checkpoint, config_path, config_metadata = tamper_copy("config.ckpt")
    model_config = cast(dict[str, Any], config_metadata["model_config"])
    nested_config = cast(dict[str, Any], model_config["config"])
    nested_config["task_id"] = 4
    canonical_config = json.dumps(
        model_config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    config_metadata["config_sha256"] = hashlib.sha256(canonical_config).hexdigest()
    config_path.write_text(
        json.dumps(config_metadata, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config fields"):
        load_recurrent_latent_world_model_ensemble_checkpoint(config_checkpoint)

    resource_checkpoint, resource_path, resource_metadata = tamper_copy("resource.ckpt")
    resource_budget = cast(dict[str, Any], resource_metadata["resource_budget"])
    resource_budget["persistent_state_bytes"] += 4
    resource_path.write_text(
        json.dumps(resource_metadata, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resource budget"):
        load_recurrent_latent_world_model_ensemble_checkpoint(resource_checkpoint)


def test_static_shape_and_dtype_contracts_fail_before_compiled_execution() -> None:
    model = RecurrentLatentWorldModelEnsemble(_config())
    state = model.init(jr.key(10))
    with pytest.raises(ValueError, match="shape"):
        model.start(state, jnp.asarray((1.0,), dtype=jnp.float32))
    start = model.start(state, OBSERVATION)
    with pytest.raises(ValueError, match="dtype"):
        model.decide(state, start, jnp.asarray(1.0, dtype=jnp.float32))
    decision = model.decide(state, start, ACTION)
    bad_record = cast(Any, _transition()).replace(
        terminated=jnp.asarray(0, dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="dtype"):
        model.update(state, decision, bad_record)


def test_transition_schema_contains_no_task_or_regime_channel() -> None:
    fields = {
        field.name
        for field in dataclasses.fields(cast(Any, RecurrentLatentTransitionRecord))
    }
    assert fields == {
        "observation",
        "action",
        "reward",
        "discount",
        "terminated",
        "truncated",
        "bootstrap_observation",
        "next_decision_observation",
    }
    assert not ({"task_id", "regime_id"} & fields)
