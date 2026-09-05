"""Fail-closed contracts for optimizer and feature-update numerical faults."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.baseline_optimizers import (
    NADALINE,
    AdaGain,
    Adam,
    RMSprop,
)
from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
)
from alberta_framework.core.feature_discovery import FixedBudgetFeatureLearner
from alberta_framework.core.learners import LinearLearner, TDLinearLearner
from alberta_framework.core.optimizers import (
    IDBD,
    LMS,
    TDIDBD,
    AdaptiveObGDBounding,
    AGCBounding,
    Autostep,
    AutoTDIDBD,
    ObGD,
    ObGDBounding,
)
from alberta_framework.core.swift_td import SwiftTD
from alberta_framework.core.upgd import UPGDLearner

_OBSERVATION = jnp.array([0.25, -0.5, 1.5], dtype=jnp.float32)
_NEXT_OBSERVATION = jnp.array([-0.1, 0.75, 0.2], dtype=jnp.float32)
_ERROR = jnp.array(0.7, dtype=jnp.float32)
_GAMMA = jnp.array(0.9, dtype=jnp.float32)


def _digest(*trees: object) -> str:
    """Hash exact array values, including typed PRNG-key payloads."""

    digest = hashlib.sha256()
    for tree in trees:
        for leaf in jax.tree.leaves(tree):
            try:
                array = np.asarray(leaf)
            except TypeError:
                array = np.asarray(jr.key_data(leaf))
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _assert_neutral_metrics(metrics: dict[str, jax.Array]) -> None:
    for value in metrics.values():
        chex.assert_tree_all_finite(value)
        chex.assert_trees_all_equal(value, jnp.zeros_like(value))


@pytest.mark.parametrize(
    ("factory", "expected_digest"),
    [
        (lambda: LMS(0.03), "d56899ad3455491c21a1622abceaa31245b3c3d16b9d25b9f074eedb7895b061"),
        (
            lambda: IDBD(0.02, 0.04),
            "6a5d5242d8f7a7c6f18bed4f08f75e426403d17b834bee6291effc214f75d032",
        ),
        (
            lambda: AdaGain(0.02, 0.003, 0.15),
            "f2117930ca67fa6fad629efeab92a0cb5eeaf002fceb06aea50dc1a74bab1f76",
        ),
        (
            lambda: Adam(0.004),
            "0ab5c9346be9f5eb32cef646094997fe96f8c2dcf8fde544a0c401d7b5666303",
        ),
        (
            lambda: RMSprop(0.005),
            "135a37f881adffee283dd489e8c51177ee3e4e5660ffe0a9314c4e91e6799784",
        ),
        (
            lambda: NADALINE(0.02),
            "b70b39d059e4f3ee568874f54eb0230172fa15c6cca3547147a32aadb56351b7",
        ),
    ],
)
def test_linear_optimizer_finite_one_step_is_bitwise_unchanged(
    factory: Callable[[], Any], expected_digest: str
) -> None:
    optimizer = factory()
    state = optimizer.init(3)
    result = optimizer.update(state, _ERROR, _OBSERVATION)

    assert bool(result.update_applied)
    assert _digest(result.weight_delta, result.bias_delta, result.new_state) == expected_digest


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LMS(0.03),
        lambda: IDBD(0.02, 0.04),
        lambda: AdaGain(0.02, 0.003, 0.15),
        lambda: Adam(0.004),
        lambda: RMSprop(0.005),
        lambda: NADALINE(0.02),
    ],
)
@pytest.mark.parametrize(
    ("error", "observation"),
    [
        (jnp.array(jnp.inf, dtype=jnp.float32), jnp.array([0.0, 1.0, 2.0])),
        (jnp.array(jnp.nan, dtype=jnp.float32), jnp.ones(3)),
        (jnp.array(1.0, dtype=jnp.float32), jnp.array([0.0, jnp.inf, 2.0])),
    ],
)
def test_linear_optimizer_invalid_transaction_is_exact_noop(
    factory: Callable[[], Any], error: jax.Array, observation: jax.Array
) -> None:
    optimizer = factory()
    state = optimizer.init(3)
    result = jax.jit(optimizer.update)(state, error, observation)

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(3, dtype=jnp.float32))
    chex.assert_trees_all_equal(result.bias_delta, jnp.array(0.0, dtype=jnp.float32))
    _assert_neutral_metrics(result.metrics)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LMS(),
        lambda: IDBD(),
        lambda: Autostep(),
        lambda: Adam(),
        lambda: RMSprop(),
    ],
)
def test_param_optimizer_invalid_gradient_is_exact_noop(
    factory: Callable[[], Any],
) -> None:
    optimizer = factory()
    state = optimizer.init_for_shape((2, 2))
    gradient = jnp.array([[1.0, jnp.inf], [0.0, 2.0]], dtype=jnp.float32)
    result = jax.jit(
        lambda current, value: optimizer.update_from_gradient_checked(
            current, value, error=jnp.array(1.0, dtype=jnp.float32)
        )
    )(state, gradient)

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.step, jnp.zeros_like(gradient))
    chex.assert_trees_all_equal(result.new_state, state)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LMS(),
        lambda: IDBD(),
        lambda: Autostep(),
        lambda: Adam(),
        lambda: RMSprop(),
    ],
)
def test_param_optimizer_legacy_api_remains_exact_pair(
    factory: Callable[[], Any],
) -> None:
    optimizer = factory()
    state = optimizer.init_for_shape((2, 2))
    gradient = jnp.array([[1.0, -0.5], [0.25, 2.0]], dtype=jnp.float32)

    legacy = optimizer.update_from_gradient(state, gradient, error=_ERROR)
    checked = optimizer.update_from_gradient_checked(state, gradient, error=_ERROR)

    assert type(legacy) is tuple
    assert len(legacy) == 2
    chex.assert_trees_all_equal(legacy[0], checked.step)
    chex.assert_trees_all_equal(legacy[1], checked.new_state)
    assert bool(checked.update_applied)


def test_idbd_finite_overflow_is_rejected_without_partial_state() -> None:
    optimizer = IDBD(initial_step_size=0.01, meta_step_size=0.01)
    state = optimizer.init(2)
    result = optimizer.update(
        state,
        jnp.array(1e30, dtype=jnp.float32),
        jnp.array([1e20, 1.0], dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(2, dtype=jnp.float32))


@pytest.mark.parametrize("factory", [lambda: IDBD(), lambda: Autostep()])
def test_linear_optimizer_rejects_finite_input_when_internal_meta_guard_fires(
    factory: Callable[[], Any],
) -> None:
    optimizer = factory()
    state = optimizer.init(2).replace(
        traces=jnp.full(2, 1e33, dtype=jnp.float32),
    )
    result = jax.jit(optimizer.update)(
        state,
        jnp.array(1e10, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(2, dtype=jnp.float32))
    chex.assert_trees_all_equal(result.bias_delta, jnp.array(0.0, dtype=jnp.float32))


@pytest.mark.parametrize(
    ("factory", "gradient", "error"),
    [
        (lambda: IDBD(), jnp.full(2, 1e10, dtype=jnp.float32), jnp.array(1.0)),
        (lambda: Autostep(), jnp.ones(2), jnp.array(1e10, dtype=jnp.float32)),
    ],
)
def test_checked_param_update_reports_internal_meta_guard(
    factory: Callable[[], Any], gradient: jax.Array, error: jax.Array
) -> None:
    optimizer = factory()
    state = optimizer.init_for_shape((2,)).replace(
        traces=jnp.full(2, 1e33, dtype=jnp.float32),
    )
    result = jax.jit(
        lambda current: optimizer.update_from_gradient_checked(
            current,
            gradient,
            error=error,
        )
    )(state)

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.step, jnp.zeros(2, dtype=jnp.float32))


@pytest.mark.parametrize(
    ("factory", "expected_digest"),
    # The TDIDBD/AutoTDIDBD digests cover the state tree, which now carries
    # ``previous_gamma``; the first-step weight/bias deltas are unchanged
    # (the seeded 1.0 is inert against zero initial traces).
    [
        (
            lambda: TDIDBD(0.02, 0.04, 0.5, True),
            "6b29599d10bb3cca3e2865e242c3a0dd7d48ac25962e6a83abcf96e7cf9c7e00",
        ),
        (
            lambda: TDIDBD(0.02, 0.04, 0.5, False),
            "6b29599d10bb3cca3e2865e242c3a0dd7d48ac25962e6a83abcf96e7cf9c7e00",
        ),
        (
            lambda: AutoTDIDBD(0.02, 0.04, 0.5, 50.0),
            "5169066bac6f67922563680b512a2b74baf82737078bdcc91b9ef17e262d1ed0",
        ),
        (
            lambda: SwiftTD(0.02, 0.04, 0.5, 0.1),
            "e078ebe5567c71d2616a477fc1537565147d703dcc72f2f8fafd668f4e9296b4",
        ),
    ],
)
def test_td_optimizer_finite_one_step_is_bitwise_unchanged(
    factory: Callable[[], Any], expected_digest: str
) -> None:
    optimizer = factory()
    state = optimizer.init(3)
    result = optimizer.update(state, _ERROR, _OBSERVATION, _NEXT_OBSERVATION, _GAMMA)

    assert bool(result.update_applied)
    assert _digest(result.weight_delta, result.bias_delta, result.new_state) == expected_digest


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TDIDBD(),
        lambda: AutoTDIDBD(),
        lambda: SwiftTD(initial_step_size=0.01),
    ],
)
@pytest.mark.parametrize(
    ("td_error", "observation", "gamma"),
    [
        (jnp.array(jnp.inf), jnp.ones(3), jnp.array(0.9)),
        (jnp.array(1.0), jnp.array([1.0, jnp.nan, 0.0]), jnp.array(0.9)),
        (jnp.array(1.0), jnp.ones(3), jnp.array(jnp.inf)),
    ],
)
def test_td_optimizer_invalid_transition_is_exact_noop(
    factory: Callable[[], Any],
    td_error: jax.Array,
    observation: jax.Array,
    gamma: jax.Array,
) -> None:
    optimizer = factory()
    state = optimizer.init(3)
    result = jax.jit(optimizer.update)(
        state, td_error, observation, _NEXT_OBSERVATION, gamma
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(3, dtype=jnp.float32))
    chex.assert_trees_all_equal(result.bias_delta, jnp.array(0.0, dtype=jnp.float32))
    _assert_neutral_metrics(result.metrics)


def test_tdidbd_rejects_when_internal_meta_guard_fires() -> None:
    optimizer = TDIDBD()
    state = optimizer.init(3).replace(
        h_traces=jnp.full(3, 1e33, dtype=jnp.float32),
    )
    result = jax.jit(optimizer.update)(
        state,
        jnp.array(1e10, dtype=jnp.float32),
        jnp.ones(3, dtype=jnp.float32),
        jnp.zeros(3, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(3, dtype=jnp.float32))


def test_swifttd_rejects_when_internal_meta_guard_fires() -> None:
    optimizer = SwiftTD(initial_step_size=0.01)
    state = optimizer.init(3).replace(
        p_traces=jnp.full(4, 1e33, dtype=jnp.float32),
    )
    result = jax.jit(optimizer.update)(
        state,
        jnp.array(1e10, dtype=jnp.float32),
        jnp.ones(3, dtype=jnp.float32),
        jnp.zeros(3, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(3, dtype=jnp.float32))


def test_swifttd_rejection_preserves_real_learner_weights() -> None:
    learner = TDLinearLearner(optimizer=SwiftTD(initial_step_size=0.01))
    state = learner.init(3)
    rejected = learner.update(
        state,
        _OBSERVATION,
        jnp.array(jnp.inf, dtype=jnp.float32),
        _NEXT_OBSERVATION,
        _GAMMA,
    )

    chex.assert_trees_all_equal(rejected.state, state)
    chex.assert_tree_all_finite(rejected.state)


def test_linear_learner_propagates_rejected_optimizer_transaction() -> None:
    learner = LinearLearner(optimizer=LMS(step_size=0.01))
    state = learner.init(3)
    rejected = learner.update(
        state,
        _OBSERVATION,
        jnp.array(jnp.inf, dtype=jnp.float32),
    )

    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, state)
    chex.assert_trees_all_equal(rejected.prediction, jnp.zeros_like(rejected.prediction))
    chex.assert_trees_all_equal(rejected.error, jnp.zeros_like(rejected.error))
    chex.assert_trees_all_equal(rejected.metrics, jnp.zeros_like(rejected.metrics))


def test_obgd_bounders_only_zero_actual_collapsed_infinities() -> None:
    steps = (jnp.array([jnp.inf, jnp.inf]), jnp.array(jnp.inf))
    params = tuple(jnp.ones_like(step) for step in steps)
    for bounder in (ObGDBounding(kappa=2.0), AdaptiveObGDBounding(kappa=2.0)):
        bounded, scale = jax.jit(bounder.bound)(steps, jnp.array(jnp.inf), params)
        assert float(scale) == 0.0
        chex.assert_trees_all_equal(bounded, tuple(jnp.zeros_like(step) for step in steps))

    clipped, fraction = jax.jit(AGCBounding(clip_factor=0.01).bound)(
        steps, jnp.array(jnp.inf), params
    )
    assert float(fraction) == 1.0
    chex.assert_trees_all_equal(clipped, tuple(jnp.zeros_like(step) for step in steps))

    dynamic, scale = UPGDLearner._obgd_bound_with_kappa(
        steps, jnp.array(jnp.inf), jnp.array(2.0)
    )
    assert float(scale) == 0.0
    chex.assert_trees_all_equal(dynamic, tuple(jnp.zeros_like(step) for step in steps))


def test_bounders_do_not_launder_arbitrary_nan() -> None:
    nan_step = (jnp.array([jnp.nan, 0.5]),)
    params = (jnp.ones(2),)

    bounded, _ = ObGDBounding(kappa=0.0).bound(nan_step, jnp.array(1.0), params)
    assert bool(jnp.isnan(bounded[0][0]))

    clipped, _ = AGCBounding(clip_factor=100.0).bound(
        nan_step, jnp.array(1.0), params
    )
    assert bool(jnp.isnan(clipped[0][0]))


def test_obgd_optimizer_rejects_infinite_public_error_atomically() -> None:
    optimizer = ObGD(step_size=1.0, kappa=2.0)
    state = optimizer.init(2)
    result = jax.jit(optimizer.update)(state, jnp.array(jnp.inf), jnp.ones(2))

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(2))
    chex.assert_trees_all_equal(result.bias_delta, jnp.array(0.0))
    chex.assert_trees_all_equal(result.new_state, state)
    _assert_neutral_metrics(result.metrics)


@pytest.mark.parametrize("kind", ["fixed", "compositional"])
@pytest.mark.parametrize(
    ("observation", "targets"),
    [
        (_OBSERVATION, jnp.array([jnp.inf, 0.5], dtype=jnp.float32)),
        (jnp.array([0.25, jnp.nan, 1.5]), jnp.array([0.4, -0.2])),
    ],
)
def test_feature_learner_invalid_transition_is_neutral_noop(
    kind: str, observation: jax.Array, targets: jax.Array
) -> None:
    if kind == "fixed":
        learner: Any = FixedBudgetFeatureLearner(
            n_features=5,
            n_tasks=2,
            candidate_count=2,
            replacement_interval=1,
            min_feature_age=0,
            candidate_min_age=0,
        )
    else:
        learner = CompositionalFeatureLearner(
            n_features=5,
            n_tasks=2,
            candidate_count=2,
            replacement_interval=1,
            min_feature_age=0,
            candidate_min_age=0,
        )
    state = learner.init(3, jr.key(5)).replace(birth_timestamp=0.0)
    result = learner.update(state, observation, targets)

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros(2, dtype=jnp.float32))
    chex.assert_trees_all_equal(result.errors, jnp.zeros(2, dtype=jnp.float32))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros(7, dtype=jnp.float32))
    assert int(result.replaced_slot) == -1
    assert int(result.promoted_candidate) == -1
    if kind == "compositional":
        assert not bool(result.curation_trace.has_event)
        assert int(result.curation_trace.logical_event_count) == 0


def test_compositional_feature_learner_rejects_nonfinite_context() -> None:
    learner = CompositionalFeatureLearner(
        n_features=5,
        n_tasks=2,
        candidate_count=2,
        replacement_interval=0,
    )
    state = learner.init(3, jr.key(13)).replace(birth_timestamp=0.0)
    result = learner.update(
        state,
        _OBSERVATION,
        jnp.array([0.4, -0.2], dtype=jnp.float32),
        context_id=jnp.array(jnp.nan, dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.metrics, jnp.zeros(7, dtype=jnp.float32))


def test_feature_learners_allow_nan_missing_target_and_report_commit() -> None:
    for learner in (
        FixedBudgetFeatureLearner(5, 2, candidate_count=2, replacement_interval=0),
        CompositionalFeatureLearner(5, 2, candidate_count=2, replacement_interval=0),
    ):
        state = learner.init(3, jr.key(3)).replace(birth_timestamp=0.0)
        result = learner.update(
            state,
            _OBSERVATION,
            jnp.array([0.4, jnp.nan], dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        assert bool(jnp.isnan(result.errors[1]))


def test_feature_finite_one_steps_are_bitwise_unchanged() -> None:
    fixed = FixedBudgetFeatureLearner(
        n_features=5,
        n_tasks=2,
        candidate_count=2,
        replacement_interval=0,
        use_obgd=True,
    )
    fixed_state = fixed.init(3, jr.key(7)).replace(birth_timestamp=0.0)
    fixed_result = fixed.update(
        fixed_state, _OBSERVATION, jnp.array([0.4, -0.2], dtype=jnp.float32)
    )
    assert bool(fixed_result.update_applied)
    assert _digest(
        fixed_result.state,
        fixed_result.predictions,
        fixed_result.errors,
        fixed_result.metrics,
        fixed_result.replaced_slot,
        fixed_result.promoted_candidate,
    ) == "9e314db98159056ad985d6b8b3047256ef8e503d084046758917774cc6655cee"

    compositional = CompositionalFeatureLearner(
        n_features=5,
        n_tasks=2,
        candidate_count=2,
        replacement_interval=0,
        use_obgd=True,
    )
    compositional_state = compositional.init(3, jr.key(11)).replace(
        birth_timestamp=0.0
    )
    compositional_result = compositional.update(
        compositional_state,
        _OBSERVATION,
        jnp.array([0.4, -0.2], dtype=jnp.float32),
    )
    assert bool(compositional_result.update_applied)
    # These event counters intentionally use int32 so they remain exact past
    # the float32 consecutive-integer limit.  The golden digest includes dtype.
    assert compositional_result.state.candidate_selector_action_counts.dtype == jnp.int32
    assert compositional_result.state.generator_resource_state.action_counts.dtype == jnp.int32
    assert _digest(
        compositional_result.state,
        compositional_result.predictions,
        compositional_result.errors,
        compositional_result.metrics,
        compositional_result.replaced_slot,
        compositional_result.promoted_candidate,
        compositional_result.curation_trace,
    ) == "cf84267749a5df5f6a0b499d1916d560c97b02648f070fe2ac0cc3eb1c66dfd1"
