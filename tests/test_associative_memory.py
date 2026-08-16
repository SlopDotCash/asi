# mypy: disable-error-code="attr-defined"
"""Tests for the Step 2 fixed-budget associative memory."""

import math

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    run_associative_memory_arrays,
)
from alberta_framework.steps.step2 import (
    Step2AssociativeConfig,
    make_step2_associative_learner,
    run_step2_associative_smoke,
)


def test_associative_config_roundtrip() -> None:
    config = AssociativeMemoryConfig(
        vocab_size=7,
        block_size=5,
        suffix_length=3,
        feature_family="token_suffix_pair",
        max_features=31,
    )

    restored = AssociativeMemoryConfig.from_config(config.to_config())

    assert restored == config
    learner = AssociativeMemoryLearner(restored)
    assert learner.max_active_features == 8
    chex.assert_shape(learner.init().keys, (31, 5))


def test_config_rejects_infinite_write_lr() -> None:
    with pytest.raises(ValueError, match="write_lr"):
        AssociativeMemoryLearner(
            AssociativeMemoryConfig(
                vocab_size=4,
                block_size=3,
                suffix_length=2,
                write_lr=float("inf"),
            )
        )


@pytest.mark.parametrize(
    ("field", "invalid_values"),
    [
        (
            "scope_lr",
            (float("nan"), float("inf"), float("-inf"), -0.01, True, "0.1"),
        ),
        (
            "budget_lr",
            (float("nan"), float("inf"), float("-inf"), -0.01, True, "0.1"),
        ),
        (
            "initial_budget_fraction",
            (float("nan"), float("inf"), float("-inf"), 0.0, 1.01, True, "0.5"),
        ),
        (
            "scope_logit_clip",
            (float("nan"), float("inf"), float("-inf"), 0.0, True, "8.0"),
        ),
        ("min_effective_budget", (float("nan"), 1.5, True, "1")),
        ("adaptive_feature_family", (0, 1, "yes")),
        ("adaptive_window", (0, 1, "yes")),
        ("adaptive_budget", (0, 1, "yes")),
    ],
)
def test_config_rejects_invalid_adaptive_scalars(
    field: str,
    invalid_values: tuple[object, ...],
) -> None:
    base = AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2)

    for invalid in invalid_values:
        payload = base.to_config()
        payload[field] = invalid
        with pytest.raises(ValueError, match=field):
            AssociativeMemoryLearner(AssociativeMemoryConfig.from_config(payload))


def test_silent_feature_does_not_turn_inf_value_into_nan() -> None:
    """Weight 0 times an inf stored row is 0*inf = NaN in the evidence sum."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init()
    poisoned = state.replace(
        values=state.values.at[0].set(jnp.full((4,), jnp.inf, dtype=jnp.float32))
    )
    context = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    raw = jnp.array([0.0], dtype=jnp.float32)[:, None] * poisoned.values[0][None, :]
    assert not bool(jnp.all(jnp.isfinite(raw)))

    prediction = learner.predict(poisoned, context)
    chex.assert_tree_all_finite(prediction.logits)
    chex.assert_tree_all_finite(prediction.probabilities)
    assert float(jnp.sum(prediction.probabilities)) == pytest.approx(1.0)

    result = learner.update(poisoned, context, jnp.asarray(1, dtype=jnp.int32))
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, poisoned)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros_like(result.predictions))
    chex.assert_trees_all_equal(result.logits, jnp.zeros_like(result.logits))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_corrupted_active_family_logits_remain_fail_visible() -> None:
    """An invalid active gate must not masquerade as a uniform valid gate."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=3,
            suffix_length=2,
            max_features=8,
            adaptive_feature_family=True,
        )
    )
    state = learner.init().replace(family_logits=jnp.full((2,), jnp.inf, dtype=jnp.float32))
    context = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    prediction = learner.predict(state, context)

    assert not bool(jnp.all(jnp.isfinite(prediction.family_probs)))
    result = learner.update(state, context, jnp.asarray(1, dtype=jnp.int32))
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize(
    "leaf",
    [
        "values",
        "utility",
        "counts",
        "prior",
        "family_logits",
        "window_logits",
        "budget_logit",
    ],
)
def test_update_rejects_nonfinite_source_state_leaf(leaf: str) -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=3,
            suffix_length=2,
            max_features=8,
            adaptive_feature_family=True,
            adaptive_window=True,
            adaptive_budget=True,
        )
    )
    state = learner.init()
    poison = jnp.asarray(jnp.nan, dtype=jnp.float32)
    if leaf == "values":
        state = state.replace(values=state.values.at[0, 0].set(poison))
    elif leaf == "utility":
        state = state.replace(utility=state.utility.at[0].set(poison))
    elif leaf == "counts":
        state = state.replace(counts=state.counts.at[0].set(poison))
    elif leaf == "prior":
        state = state.replace(prior=state.prior.at[0].set(poison))
    elif leaf == "family_logits":
        state = state.replace(family_logits=state.family_logits.at[0].set(poison))
    elif leaf == "window_logits":
        state = state.replace(window_logits=state.window_logits.at[0].set(poison))
    else:
        state = state.replace(budget_logit=poison)

    result = learner.update(
        state,
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_tree_all_finite((result.predictions, result.logits, result.metrics))
    chex.assert_trees_all_equal(result.predictions, jnp.zeros_like(result.predictions))
    chex.assert_trees_all_equal(result.logits, jnp.zeros_like(result.logits))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_array_runner_exposes_rejected_update_mask() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init().replace(
        counts=learner.init().counts.at[0].set(jnp.asarray(jnp.inf, dtype=jnp.float32))
    )
    contexts = jnp.asarray([[1, 2, 3], [2, 3, 0]], dtype=jnp.int32)
    labels = jnp.asarray([1, 2], dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, state, contexts, labels)

    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.zeros((contexts.shape[0],), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros_like(result.predictions))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_associative_prediction_is_before_write() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=5, block_size=4, suffix_length=3)
    )
    state = learner.init()
    context = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)

    before = learner.predict(state, context)
    result = learner.update(state, context, jnp.asarray(2, dtype=jnp.int32))

    assert bool(result.update_applied)
    chex.assert_trees_all_close(before.probabilities, jnp.full((5,), 0.2))
    chex.assert_trees_all_close(result.predictions, before.probabilities)
    assert int(result.state.step_count) == 1
    assert float(result.metrics[0]) == pytest.approx(math.log(5), abs=1e-5)


def test_associative_scope_controls_are_disabled_by_default() -> None:
    config = AssociativeMemoryConfig(
        vocab_size=5,
        block_size=4,
        suffix_length=3,
        max_features=32,
    )
    learner = AssociativeMemoryLearner(config)
    state = learner.init()
    context = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)

    result = learner.update(state, context, jnp.asarray(2, dtype=jnp.int32))
    prediction = learner.predict(result.state, context)

    assert not config.adaptive_feature_family
    assert not config.adaptive_window
    assert not config.adaptive_budget
    chex.assert_trees_all_close(result.state.family_logits, state.family_logits)
    chex.assert_trees_all_close(result.state.window_logits, state.window_logits)
    chex.assert_trees_all_close(result.state.budget_logit, state.budget_logit)
    chex.assert_trees_all_close(
        prediction.scope_weights,
        prediction.feature_mask.astype(jnp.float32),
    )
    assert float(prediction.effective_budget) == pytest.approx(config.max_features)


def test_associative_memory_learns_repeated_binding() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=6,
            block_size=4,
            suffix_length=3,
            max_features=64,
        )
    )
    context = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    contexts = jnp.tile(context[None, :], (32, 1))
    labels = jnp.full((32,), 5, dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    chex.assert_tree_all_finite((result.predictions, result.metrics))
    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.ones((contexts.shape[0],), dtype=jnp.bool_),
    )

    initial_nll = float(jnp.mean(result.metrics[:4, 0]))
    final_nll = float(jnp.mean(result.metrics[-4:, 0]))
    final_accuracy = float(jnp.mean(result.metrics[-4:, 1]))

    assert final_nll < initial_nll * 0.5
    assert final_accuracy == 1.0


def test_associative_memory_respects_fixed_budget() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=9,
            block_size=5,
            suffix_length=4,
            max_features=4,
        )
    )
    contexts = jnp.asarray(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [1, 3, 5, 7, 8],
        ],
        dtype=jnp.int32,
    )
    labels = jnp.asarray([1, 2, 3], dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    occupied = int(jnp.sum(result.state.counts > 0.0))

    assert occupied <= 4
    assert int(result.state.replacements) > 0


@pytest.mark.parametrize(
    "label",
    [
        -1,
        4,
        9999,
        1.9,
        float("nan"),
        float("inf"),
        1 + 2j,
        True,
        np.uint64(2**32),
        np.float64(1.00000001),
        np.asarray([1], dtype=np.int32),
        2**100,
    ],
)
def test_update_rejects_labels_outside_the_vocabulary_instead_of_clipping(
    label: object,
) -> None:
    """An out-of-domain label must be a rejected transaction, not a substituted class."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init()
    result = learner.update(
        state,
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
        label,  # type: ignore[arg-type]
    )
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


@pytest.mark.parametrize("label", [-1, 4, 9999])
def test_scan_update_rejects_out_of_vocabulary_traced_labels(label: int) -> None:
    """A traced out-of-vocabulary label is a rejected transaction, not a substituted class."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init()
    contexts = jnp.asarray([[1, 2, 3], [2, 3, 0]], dtype=jnp.int32)
    labels = jnp.full((2,), label, dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, state, contexts, labels)

    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.zeros((contexts.shape[0],), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros_like(result.predictions))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_scan_update_applies_only_the_valid_labels_of_a_mixed_stream() -> None:
    """Valid labels in a mixed stream keep training while invalid steps are neutralized."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init()
    contexts = jnp.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    labels = jnp.asarray([1, 4, 1], dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, state, contexts, labels)

    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.asarray([True, False, True], dtype=jnp.bool_),
    )
    assert float(result.state.prior[1]) > 0.0
    chex.assert_trees_all_equal(result.metrics[1], jnp.zeros_like(result.metrics[1]))


def test_all_invalid_label_stream_does_not_report_perfect_accuracy() -> None:
    """A stream whose every label is out of range must not publish trained-looking metrics."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init()
    steps = 20
    contexts = jnp.tile(jnp.asarray([[1, 2, 3]], dtype=jnp.int32), (steps, 1))
    labels = jnp.full((steps,), 4, dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, state, contexts, labels)

    assert not bool(result.updates_applied.any())
    chex.assert_trees_all_equal(result.state, state)
    assert float(result.metrics[-10:, 1].mean()) == 0.0
    assert float(result.metrics[-10:, 0].mean()) == 0.0


def test_update_accepts_every_in_vocabulary_label() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init()
    for label in range(4):
        result = learner.update(
            state, jnp.asarray([1, 2, 3], dtype=jnp.int32), jnp.asarray(label, dtype=jnp.int32)
        )
        assert bool(result.update_applied)
        assert float(result.state.prior[label]) > float(state.prior[label])


def test_evicted_row_is_reset_before_a_new_key_writes_into_it() -> None:
    """A slot reused by eviction must not carry the evicted key's values, utility, or counts."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=2,
            suffix_length=2,
            feature_family="position_token",
            max_features=2,
        )
    )
    state = learner.init()
    context_a = jnp.asarray([0, 1], dtype=jnp.int32)
    context_b = jnp.asarray([2, 3], dtype=jnp.int32)
    for _ in range(30):
        state = learner.update(state, context_a, jnp.asarray(3, dtype=jnp.int32)).state
    assert int(state.replacements) == 0

    state = learner.update(state, context_b, jnp.asarray(1, dtype=jnp.int32)).state
    assert int(state.replacements) > 0
    prediction = learner.predict(state, context_b)
    found = prediction.found > 0
    assert int(jnp.sum(found)) >= 1
    slots = prediction.indices[found]
    rows = state.values[slots]
    # B was written exactly once with label 1: no mass at A's label 3, one write's worth at 1
    chex.assert_trees_all_close(rows[:, 3], jnp.zeros((rows.shape[0],), dtype=jnp.float32))
    assert bool(jnp.all(rows[:, 1] > 0.0))
    assert int(jnp.argmax(prediction.logits)) == 1
    chex.assert_trees_all_close(state.counts[slots], jnp.ones((rows.shape[0],), dtype=jnp.float32))


def test_associative_adaptive_family_scope_prefers_useful_pairs() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=4,
            suffix_length=3,
            max_features=128,
            adaptive_feature_family=True,
            scope_lr=0.2,
        )
    )
    base_contexts = jnp.asarray(
        [
            [0, 0, 1, 2],
            [0, 0, 1, 3],
            [0, 0, 2, 2],
            [0, 0, 2, 3],
        ],
        dtype=jnp.int32,
    )
    base_labels = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    pattern_ids = jnp.arange(240, dtype=jnp.int32) % base_contexts.shape[0]
    contexts = base_contexts[pattern_ids]
    labels = base_labels[pattern_ids]

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    prediction = learner.predict(result.state, contexts[-1])

    assert float(result.state.family_logits[1]) > float(result.state.family_logits[0])
    assert float(prediction.family_probs[1]) > 0.80


def test_associative_adaptive_window_scope_prefers_useful_long_window() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=4,
            suffix_length=4,
            feature_family="suffix_pair",
            max_features=512,
            adaptive_window=True,
            scope_lr=0.2,
        )
    )
    contexts_list: list[list[int]] = []
    labels_list: list[int] = []
    for _ in range(3):
        for old_token in (1, 2):
            for middle_a in range(4):
                for middle_b in range(4):
                    for recent_token in range(4):
                        contexts_list.append(
                            [old_token, middle_a, middle_b, recent_token]
                        )
                        labels_list.append(old_token - 1)
    contexts = jnp.asarray(contexts_list, dtype=jnp.int32)
    labels = jnp.asarray(labels_list, dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    prediction = learner.predict(result.state, contexts[-1])

    assert float(result.state.window_logits[-1]) > float(result.state.window_logits[0])
    assert float(prediction.window_probs[-1]) > 0.80


def test_associative_adaptive_budget_expands_under_replacement_pressure() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=13,
            block_size=4,
            suffix_length=3,
            max_features=64,
            adaptive_budget=True,
            initial_budget_fraction=0.10,
            budget_lr=0.5,
        )
    )
    contexts = (
        jnp.arange(80 * 4, dtype=jnp.int32).reshape(80, 4)
        * jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    ) % 13
    labels = (contexts[:, 0] + 2 * contexts[:, 1] + 3 * contexts[:, 2]) % 13
    state = learner.init()
    initial_budget = learner.predict(state, contexts[0]).effective_budget

    result = run_associative_memory_arrays(learner, state, contexts, labels)
    final_budget = learner.predict(result.state, contexts[-1]).effective_budget

    assert int(result.state.replacements) > 0
    assert float(result.state.budget_logit) > float(state.budget_logit)
    assert float(final_budget) > float(initial_budget) + 10.0


def test_step2_associative_facade_smoke_and_roundtrip() -> None:
    config = Step2AssociativeConfig(
        vocab_size=8,
        block_size=5,
        suffix_length=3,
        max_features=128,
        adaptive_feature_family=True,
        adaptive_window=True,
        adaptive_budget=True,
        initial_budget_fraction=0.25,
    )
    restored = Step2AssociativeConfig.from_dict(config.to_dict())
    learner = make_step2_associative_learner(restored)

    assert learner.config == config.to_core_config()
    assert learner.config.adaptive_feature_family
    assert learner.config.adaptive_window
    assert learner.config.adaptive_budget
    assert learner.config.initial_budget_fraction == pytest.approx(0.25)

    result = run_step2_associative_smoke(config, steps=64, seed=0, window=16)
    assert result.finite
    assert result.metrics_shape == (64, 8)
    assert result.final_window_nll < result.initial_window_nll


_INVALID_ASSOCIATIVE_CONFIGS: tuple[dict[str, object], ...] = (
    {"vocab_size": 1, "block_size": 8},
    {"vocab_size": 0, "block_size": 8},
    {"vocab_size": -1, "block_size": 8},
    {"vocab_size": 2**31, "block_size": 8},
    {"vocab_size": True, "block_size": 8},
    {"vocab_size": "4", "block_size": 8},
    {"vocab_size": 4, "block_size": 0},
    {"vocab_size": 4, "block_size": -1},
    {"vocab_size": 4, "block_size": 2**31},
    {"vocab_size": 4, "block_size": True},
    {"vocab_size": 4, "block_size": 8, "suffix_length": 1},
    {"vocab_size": 4, "block_size": 8, "suffix_length": 9},
    {"vocab_size": 4, "block_size": 8, "suffix_length": 2**31},
    {"vocab_size": 4, "block_size": 8, "suffix_length": True},
    {"vocab_size": 4, "block_size": 8, "feature_family": "unknown_family"},
    {"vocab_size": 4, "block_size": 8, "max_features": 0},
    {"vocab_size": 4, "block_size": 8, "max_features": -1},
    {"vocab_size": 4, "block_size": 8, "max_features": 2**31},
    {"vocab_size": 4, "block_size": 8, "max_features": True},
    {"vocab_size": 4, "block_size": 8, "write_lr": 0.0},
    {"vocab_size": 4, "block_size": 8, "write_lr": -0.1},
    {"vocab_size": 4, "block_size": 8, "write_lr": 1e100},
    {"vocab_size": 4, "block_size": 8, "write_lr": float("nan")},
    {"vocab_size": 4, "block_size": 8, "write_lr": True},
    {"vocab_size": 4, "block_size": 8, "retention": -0.1},
    {"vocab_size": 4, "block_size": 8, "retention": 1.1},
    {"vocab_size": 4, "block_size": 8, "retention": 1e100},
    {"vocab_size": 4, "block_size": 8, "retention": float("nan")},
    {"vocab_size": 4, "block_size": 8, "retention": True},
    {"vocab_size": 4, "block_size": 8, "utility_lr": -0.1},
    {"vocab_size": 4, "block_size": 8, "utility_lr": 1e100},
    {"vocab_size": 4, "block_size": 8, "utility_lr": float("nan")},
    {"vocab_size": 4, "block_size": 8, "utility_lr": True},
    {"vocab_size": 4, "block_size": 8, "utility_decay": -0.1},
    {"vocab_size": 4, "block_size": 8, "utility_decay": 1.1},
    {"vocab_size": 4, "block_size": 8, "utility_decay": 1e100},
    {"vocab_size": 4, "block_size": 8, "utility_decay": float("nan")},
    {"vocab_size": 4, "block_size": 8, "utility_decay": True},
    {"vocab_size": 4, "block_size": 8, "min_weight": 0.0},
    {"vocab_size": 4, "block_size": 8, "min_weight": -0.1},
    {"vocab_size": 4, "block_size": 8, "min_weight": 1e100},
    {"vocab_size": 4, "block_size": 8, "min_weight": float("nan")},
    {"vocab_size": 4, "block_size": 8, "min_weight": True},
    {"vocab_size": 4, "block_size": 8, "max_weight": 0.0},
    {"vocab_size": 4, "block_size": 8, "max_weight": -0.1},
    {"vocab_size": 4, "block_size": 8, "max_weight": 1e100},
    {"vocab_size": 4, "block_size": 8, "max_weight": float("nan")},
    {"vocab_size": 4, "block_size": 8, "max_weight": True},
    {"vocab_size": 4, "block_size": 8, "min_weight": 1.0, "max_weight": 0.5},
    {"vocab_size": 4, "block_size": 8, "logit_scale": 0.0},
    {"vocab_size": 4, "block_size": 8, "logit_scale": -0.1},
    {"vocab_size": 4, "block_size": 8, "logit_scale": 1e100},
    {"vocab_size": 4, "block_size": 8, "logit_scale": float("nan")},
    {"vocab_size": 4, "block_size": 8, "logit_scale": True},
    {"vocab_size": 4, "block_size": 8, "normalize_by_weight": 1},
    {"vocab_size": 4, "block_size": 8, "min_effective_budget": 0},
    {"vocab_size": 4, "block_size": 8, "min_effective_budget": 2**31},
    {"vocab_size": 4, "block_size": 8, "min_effective_budget": 4097},
    {"vocab_size": 4, "block_size": 8, "min_effective_budget": True},
)


@pytest.mark.parametrize("kwargs", _INVALID_ASSOCIATIVE_CONFIGS)
def test_associative_memory_config_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AssociativeMemoryConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param((-1, 1), id="negative-ratio"),
        pytest.param((2, 1), id="above-unit-ratio"),
        pytest.param((-1, 2**200), id="negative-rounds-to-negative-zero"),
        pytest.param((2**200 + 1, 2**200), id="above-one-rounds-to-one"),
    ],
)
def test_associative_memory_rejects_adversarial_ratio_floats(
    ratio: tuple[int, int]
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    with pytest.raises(ValueError, match=r"retention must be in \[0, 1\]"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            retention=HiddenBoundaryFloat(0.5),
        )


def test_associative_memory_rejects_class_property_spoofing_float() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

    value = ClassSpoof()
    with pytest.raises(ValueError, match="must be a real number"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            write_lr=value,  # type: ignore[arg-type]
        )


def test_associative_memory_rejects_equality_spoofed_feature_family() -> None:
    class SpoofedFamily:
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash("token_suffix_pair")

    with pytest.raises(ValueError, match="feature_family"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            feature_family=SpoofedFamily(),  # type: ignore[arg-type]
        )


def test_associative_memory_rejects_spoofed_bool_flags() -> None:
    class SpoofedBool:
        @property
        def __class__(self) -> type[bool]:
            return bool

        def __bool__(self) -> bool:
            return True

    with pytest.raises(ValueError, match="normalize_by_weight"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            normalize_by_weight=SpoofedBool(),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="adaptive_feature_family"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            adaptive_feature_family=SpoofedBool(),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="adaptive_window"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            adaptive_window=SpoofedBool(),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="adaptive_budget"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            adaptive_budget=SpoofedBool(),  # type: ignore[arg-type]
        )


def test_associative_memory_rejects_spoofed_int_class_and_negative_ratios() -> None:
    class SpoofedIntFloat(float):
        @property
        def __class__(self) -> type[int]:
            return int

        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match="write_lr"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            write_lr=SpoofedIntFloat(0.5),
        )


def test_associative_memory_json_roundtrip() -> None:
    import json

    config = AssociativeMemoryConfig(
        vocab_size=8,
        block_size=16,
        suffix_length=4,
        feature_family="token_suffix_pair",
        max_features=256,
        write_lr=0.5,
        retention=0.9,
    )
    serialized = config.to_config()
    json_str = json.dumps(serialized)
    deserialized = json.loads(json_str)
    restored = AssociativeMemoryConfig.from_config(deserialized)

    assert restored == config
    assert restored.feature_family == "token_suffix_pair"
