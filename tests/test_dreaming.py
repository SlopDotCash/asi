from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.dreaming import (
    DreamBehaviorModelPrediction,
    DreamingConfig,
    DreamRolloutConfig,
    DreamSelectionConfig,
    DreamWorldModelPrediction,
    GuardedDreamer,
    RecentObservationBuffer,
    dream_one_step,
    dream_rollout,
    imagined_rollout_to_gvf_items,
    imagined_rollout_to_sarsa_items,
    imagined_transition_to_gvf_item,
    imagined_transition_to_supervised_item,
    init_dream_rollout_state,
    slice_imagined_transition,
)


class _ClassSpoof:
    def __init__(self, reported_type: type[object]) -> None:
        self._reported_type = reported_type

    @property
    def __class__(self) -> type[object]:  # type: ignore[override]
        return self._reported_type

    def __float__(self) -> float:
        raise RuntimeError("must not convert")

    def __int__(self) -> int:
        raise RuntimeError("must not convert")


def test_dream_configs_reject_class_spoofs_before_conversion() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        DreamingConfig(warmup_steps=_ClassSpoof(int))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_model_error"):
        DreamingConfig(max_model_error=_ClassSpoof(float))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stop_on_terminal"):
        DreamRolloutConfig(stop_on_terminal=_ClassSpoof(bool))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capacity"):
        RecentObservationBuffer(_ClassSpoof(int), 2)  # type: ignore[arg-type]


def test_dream_configs_validate_and_canonicalize_float32_sink_values() -> None:
    selection = DreamSelectionConfig(
        surprise_weight=np.float32(-0.5),
        utility_weight=Fraction(1, 4),
        min_confidence=np.float64(0.25),
    )
    assert selection.surprise_weight == -0.5
    assert selection.utility_weight == 0.25
    assert selection.min_confidence == 0.25
    assert all(
        type(value) is float
        for value in (
            selection.surprise_weight,
            selection.utility_weight,
            selection.min_confidence,
        )
    )
    with pytest.raises(ValueError, match="surprise_weight"):
        DreamSelectionConfig(surprise_weight=1e100)
    with pytest.raises(ValueError, match="min_confidence"):
        DreamSelectionConfig(min_confidence=Fraction(-1, 10**50))


@pytest.mark.parametrize("value", [2**31, True, 1.5])
def test_dream_count_fields_reject_non_int32_values(value: object) -> None:
    with pytest.raises(ValueError, match="max_items"):
        DreamSelectionConfig(max_items=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capacity"):
        RecentObservationBuffer(value, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observation_dim"):
        RecentObservationBuffer(2, value)  # type: ignore[arg-type]


def test_dream_count_fields_accept_int32_endpoint_and_numpy_ints() -> None:
    selection = DreamSelectionConfig(max_items=np.int32(3))
    buffer = RecentObservationBuffer(np.int32(2), np.int64(4))
    endpoint = DreamSelectionConfig(max_items=2**31 - 1)
    assert type(selection.max_items) is int
    assert type(buffer.capacity) is int
    assert type(buffer.observation_dim) is int
    assert endpoint.max_items == 2**31 - 1


def _assert_rollout_state_close(left, right) -> None:  # type: ignore[no-untyped-def]
    chex.assert_trees_all_close(
        (
            left.observation,
            jr.key_data(left.rng_key),
            left.active,
            left.cumulative_confidence,
            left.step_count,
        ),
        (
            right.observation,
            jr.key_data(right.rng_key),
            right.active,
            right.cumulative_confidence,
            right.step_count,
        ),
    )


def _assert_dream_outputs_finite(next_state, transition) -> None:  # type: ignore[no-untyped-def]
    chex.assert_tree_all_finite(
        (
            next_state.observation,
            jr.key_data(next_state.rng_key),
            next_state.active,
            next_state.cumulative_confidence,
            next_state.step_count,
            transition,
        )
    )


@chex.dataclass(frozen=True)
class MockWorldState:
    drift: jnp.ndarray
    reward_bias: jnp.ndarray
    confidence: jnp.ndarray
    model_error: jnp.ndarray


class MockWorldModel:
    def predict(
        self,
        state: MockWorldState,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        key: Any,
    ) -> DreamWorldModelPrediction:
        del key
        action_delta = 0.1 * jnp.asarray(action, dtype=jnp.float32)
        next_observation = observation + state.drift + action_delta
        reward = jnp.sum(next_observation) + state.reward_bias
        return DreamWorldModelPrediction(
            next_observation=next_observation,
            reward=reward,
            discount=jnp.array(0.9, dtype=jnp.float32),
            terminated=jnp.array(False),
            confidence=state.confidence,
            model_error=state.model_error,
        )


@chex.dataclass(frozen=True)
class DeterministicBehaviorState:
    action: jnp.ndarray


class DeterministicBehaviorModel:
    def sample_action(
        self,
        state: DeterministicBehaviorState,
        observation: jnp.ndarray,
        key: Any,
    ) -> DreamBehaviorModelPrediction:
        del observation, key
        return DreamBehaviorModelPrediction(
            action=state.action,
            action_probability=jnp.array(1.0, dtype=jnp.float32),
            log_probability=jnp.array(0.0, dtype=jnp.float32),
        )


@chex.dataclass(frozen=True)
class BernoulliBehaviorState:
    probability: jnp.ndarray


class BernoulliBehaviorModel:
    def sample_action(
        self,
        state: BernoulliBehaviorState,
        observation: jnp.ndarray,
        key: Any,
    ) -> DreamBehaviorModelPrediction:
        del observation
        action = jr.bernoulli(key, state.probability).astype(jnp.int32)
        probability = jnp.where(action == 1, state.probability, 1.0 - state.probability)
        return DreamBehaviorModelPrediction(
            action=action,
            action_probability=probability,
            log_probability=jnp.log(probability),
        )


def _world_state(confidence: float = 1.0, model_error: float = 0.0) -> MockWorldState:
    return MockWorldState(
        drift=jnp.array([0.5, -0.25], dtype=jnp.float32),
        reward_bias=jnp.array(0.25, dtype=jnp.float32),
        confidence=jnp.array(confidence, dtype=jnp.float32),
        model_error=jnp.array(model_error, dtype=jnp.float32),
    )


def test_one_step_dream_is_finite_and_does_not_mutate_model_state() -> None:
    world = MockWorldModel()
    behavior = DeterministicBehaviorModel()
    world_state = _world_state()
    behavior_state = DeterministicBehaviorState(action=jnp.array(1, dtype=jnp.int32))
    rollout_state = init_dream_rollout_state(
        jnp.array([1.0, 2.0], dtype=jnp.float32),
        jr.key(0),
    )

    next_state, transition = dream_one_step(
        world,
        world_state,
        behavior,
        behavior_state,
        rollout_state,
    )

    _assert_dream_outputs_finite(next_state, transition)
    chex.assert_trees_all_equal(world_state, _world_state())
    chex.assert_trees_all_equal(
        behavior_state,
        DeterministicBehaviorState(action=jnp.array(1, dtype=jnp.int32)),
    )
    assert bool(transition.valid)
    chex.assert_trees_all_close(
        transition.next_observation,
        jnp.array([1.6, 1.85], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(transition.reward, jnp.array(3.7, dtype=jnp.float32))


def test_rollout_is_reproducible_under_prng_keys() -> None:
    world = MockWorldModel()
    behavior = BernoulliBehaviorModel()
    world_state = _world_state()
    behavior_state = BernoulliBehaviorState(probability=jnp.array(0.35, dtype=jnp.float32))
    config = DreamRolloutConfig(rollout_horizon=5)
    initial_a = init_dream_rollout_state(jnp.array([0.0, 0.0], dtype=jnp.float32), jr.key(7))
    initial_b = init_dream_rollout_state(jnp.array([0.0, 0.0], dtype=jnp.float32), jr.key(7))

    rollout_a = dream_rollout(world, world_state, behavior, behavior_state, initial_a, config)
    rollout_b = dream_rollout(world, world_state, behavior, behavior_state, initial_b, config)

    _assert_rollout_state_close(rollout_a.state, rollout_b.state)
    chex.assert_trees_all_close(rollout_a.transitions, rollout_b.transitions)
    chex.assert_shape(rollout_a.transitions.reward, (5,))
    chex.assert_shape(rollout_a.transitions.next_observation, (5, 2))


def test_model_confidence_gating_marks_invalid_and_stops_rollout() -> None:
    world = MockWorldModel()
    behavior = DeterministicBehaviorModel()
    world_state = _world_state(confidence=0.2, model_error=0.8)
    behavior_state = DeterministicBehaviorState(action=jnp.array(0, dtype=jnp.int32))
    config = DreamRolloutConfig(
        rollout_horizon=3,
        confidence_threshold=0.9,
        max_model_error=0.1,
    )
    initial = init_dream_rollout_state(jnp.array([1.0, 1.0], dtype=jnp.float32), jr.key(11))

    rollout = dream_rollout(world, world_state, behavior, behavior_state, initial, config)

    assert not bool(rollout.transitions.valid[0])
    assert not bool(rollout.state.active)
    chex.assert_trees_all_close(
        rollout.state.observation,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    gvf_items = imagined_rollout_to_gvf_items(rollout)
    chex.assert_trees_all_close(gvf_items.weights, jnp.zeros((3,), dtype=jnp.float32))


def test_nonfinite_world_prediction_marks_the_dream_step_invalid_and_stops() -> None:
    """A NaN/inf model prediction must not become a valid, weight-1.0 imagined transition."""
    world = MockWorldModel()
    behavior = DeterministicBehaviorModel()
    world_state = MockWorldState(
        drift=jnp.array([jnp.nan, 0.0], dtype=jnp.float32),
        reward_bias=jnp.array(0.25, dtype=jnp.float32),
        confidence=jnp.array(1.0, dtype=jnp.float32),
        model_error=jnp.array(0.0, dtype=jnp.float32),
    )
    behavior_state = DeterministicBehaviorState(action=jnp.array(0, dtype=jnp.int32))
    config = DreamRolloutConfig(rollout_horizon=3, confidence_threshold=0.5, max_model_error=1.0)
    anchor = jnp.array([1.0, 1.0], dtype=jnp.float32)
    initial = init_dream_rollout_state(anchor, jr.key(11))

    rollout = dream_rollout(world, world_state, behavior, behavior_state, initial, config)

    assert not bool(jnp.any(rollout.transitions.valid))
    assert not bool(rollout.state.active)
    chex.assert_trees_all_close(rollout.state.observation, anchor)
    gvf_items = imagined_rollout_to_gvf_items(rollout)
    chex.assert_trees_all_close(gvf_items.weights, jnp.zeros((3,), dtype=jnp.float32))
    sarsa_items = imagined_rollout_to_sarsa_items(rollout)
    chex.assert_trees_all_close(sarsa_items.weights, jnp.zeros((3,), dtype=jnp.float32))
    transition = jax.tree.map(lambda leaf: leaf[0], rollout.transitions)
    item = imagined_transition_to_supervised_item(transition)
    gvf_item = imagined_transition_to_gvf_item(transition)
    assert float(item.weights) == 0.0
    for converted in (item, gvf_item, gvf_items, sarsa_items):
        for leaf in jax.tree.leaves(converted):
            assert bool(jnp.all(jnp.isfinite(leaf)))


class FiniteWorldModel:
    """World model that returns finite predictions regardless of the anchor."""

    def predict(
        self,
        state: Any,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        key: Any,
    ) -> DreamWorldModelPrediction:
        del state, action, key
        return DreamWorldModelPrediction(
            next_observation=jnp.zeros_like(observation),
            reward=jnp.array(1.0, dtype=jnp.float32),
            discount=jnp.array(0.9, dtype=jnp.float32),
            terminated=jnp.array(False),
            confidence=jnp.array(1.0, dtype=jnp.float32),
            model_error=jnp.array(0.0, dtype=jnp.float32),
        )


@pytest.mark.unit
def test_nonfinite_anchor_observation_marks_the_dream_step_invalid_and_stops() -> None:
    """A non-finite anchor observation must not ship as a valid, weight-1.0 item.

    ``GuardedDreamer.propose`` rejects a non-finite anchor observation with
    ``REJECT_NONFINITE``; the rollout path must mirror that gate even when the
    world model itself returns finite predictions, because the anchor is copied
    verbatim into ``ImaginedTransition.observation``.
    """
    behavior = DeterministicBehaviorModel()
    behavior_state = DeterministicBehaviorState(action=jnp.array(0, dtype=jnp.int32))
    config = DreamRolloutConfig(rollout_horizon=3, confidence_threshold=0.5, max_model_error=1.0)
    anchor = jnp.array([jnp.nan, 1.0], dtype=jnp.float32)
    initial = init_dream_rollout_state(anchor, jr.key(5))

    rollout = dream_rollout(FiniteWorldModel(), None, behavior, behavior_state, initial, config)

    assert not bool(jnp.any(rollout.transitions.valid))
    assert not bool(rollout.state.active)
    gvf_items = imagined_rollout_to_gvf_items(rollout)
    chex.assert_trees_all_close(gvf_items.weights, jnp.zeros((3,), dtype=jnp.float32))
    sarsa_items = imagined_rollout_to_sarsa_items(rollout)
    chex.assert_trees_all_close(sarsa_items.weights, jnp.zeros((3,), dtype=jnp.float32))
    transition = jax.tree.map(lambda leaf: leaf[0], rollout.transitions)
    item = imagined_transition_to_supervised_item(transition)
    gvf_item = imagined_transition_to_gvf_item(transition)
    assert float(item.weights) == 0.0
    for converted in (item, gvf_item, gvf_items, sarsa_items):
        for leaf in jax.tree.leaves(converted):
            assert bool(jnp.all(jnp.isfinite(leaf)))


@pytest.mark.unit
def test_nonfinite_imagined_action_marks_the_dream_step_invalid() -> None:
    """A non-finite sampled action must not ship as a valid training input."""

    class NaNActionBehaviorModel:
        def sample_action(
            self,
            state: Any,
            observation: jnp.ndarray,
            key: Any,
        ) -> DreamBehaviorModelPrediction:
            del state, observation, key
            return DreamBehaviorModelPrediction(
                action=jnp.array([jnp.nan], dtype=jnp.float32),
                action_probability=jnp.array(1.0, dtype=jnp.float32),
                log_probability=jnp.array(0.0, dtype=jnp.float32),
            )

    config = DreamRolloutConfig(rollout_horizon=2, confidence_threshold=0.5, max_model_error=1.0)
    initial = init_dream_rollout_state(jnp.array([1.0, 1.0], dtype=jnp.float32), jr.key(9))

    rollout = dream_rollout(
        FiniteWorldModel(), None, NaNActionBehaviorModel(), None, initial, config
    )

    assert not bool(jnp.any(rollout.transitions.valid))
    sarsa_items = imagined_rollout_to_sarsa_items(rollout)
    chex.assert_trees_all_close(sarsa_items.weights, jnp.zeros((2,), dtype=jnp.float32))
    for leaf in jax.tree.leaves(sarsa_items):
        assert bool(jnp.all(jnp.isfinite(leaf)))


@pytest.mark.parametrize("field", ["reward", "discount", "confidence", "model_error"])
def test_nonfinite_scalar_channels_mark_the_dream_step_invalid(field: str) -> None:
    class ScalarNaNWorldModel(MockWorldModel):
        def predict(self, state, observation, action, key):  # type: ignore[no-untyped-def]
            prediction = super().predict(state, observation, action, key)
            return dataclasses.replace(prediction, **{field: jnp.array(jnp.nan, dtype=jnp.float32)})

    initial = init_dream_rollout_state(jnp.array([1.0, 1.0], dtype=jnp.float32), jr.key(3))
    config = DreamRolloutConfig(rollout_horizon=2, confidence_threshold=0.5, max_model_error=1.0)
    rollout = dream_rollout(
        ScalarNaNWorldModel(),
        _world_state(),
        DeterministicBehaviorModel(),
        DeterministicBehaviorState(action=jnp.array(1, dtype=jnp.int32)),
        initial,
        config,
    )
    assert not bool(jnp.any(rollout.transitions.valid))


def test_training_item_conversions_have_expected_targets() -> None:
    world = MockWorldModel()
    behavior = DeterministicBehaviorModel()
    world_state = _world_state()
    behavior_state = DeterministicBehaviorState(action=jnp.array(1, dtype=jnp.int32))
    initial = init_dream_rollout_state(jnp.array([1.0, 2.0], dtype=jnp.float32), jr.key(13))
    _, transition = dream_one_step(world, world_state, behavior, behavior_state, initial)

    supervised = imagined_transition_to_supervised_item(
        transition,
        n_actions=2,
        target="reward_next_observation",
    )
    chex.assert_trees_all_close(
        supervised.inputs,
        jnp.array([1.0, 2.0, 0.0, 1.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        supervised.targets,
        jnp.array([3.7, 1.6, 1.85], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(supervised.weights, jnp.array(1.0, dtype=jnp.float32))

    gvf = imagined_transition_to_gvf_item(transition)
    chex.assert_trees_all_close(gvf.cumulants, jnp.array([3.7], dtype=jnp.float32))
    chex.assert_trees_all_close(gvf.discounts, jnp.array([0.9], dtype=jnp.float32))


def test_rollout_to_sarsa_items_shift_actions_and_mask_last_without_bootstrap() -> None:
    world = MockWorldModel()
    behavior = BernoulliBehaviorModel()
    world_state = _world_state()
    behavior_state = BernoulliBehaviorState(probability=jnp.array(0.5, dtype=jnp.float32))
    config = DreamRolloutConfig(rollout_horizon=4)
    initial = init_dream_rollout_state(jnp.array([0.0, 0.0], dtype=jnp.float32), jr.key(17))

    rollout = dream_rollout(world, world_state, behavior, behavior_state, initial, config)
    sarsa = imagined_rollout_to_sarsa_items(rollout)
    first = slice_imagined_transition(rollout.transitions, 0)

    chex.assert_shape(sarsa.actions, (4,))
    chex.assert_trees_all_equal(sarsa.next_actions[:-1], sarsa.actions[1:])
    chex.assert_trees_all_close(sarsa.weights[-1], jnp.array(0.0, dtype=jnp.float32))
    chex.assert_trees_all_close(sarsa.rewards[0], first.reward)

    bootstrapped = imagined_rollout_to_sarsa_items(
        rollout,
        bootstrap_action=jnp.array(1, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        bootstrapped.next_actions[-1],
        jnp.array(1, dtype=jnp.int32),
    )
    chex.assert_trees_all_close(bootstrapped.weights, rollout.transitions.valid.astype(jnp.float32))


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "max_model_error_ema",
        "max_uncertainty",
        "min_discount",
        "max_discount",
        "confidence_threshold",
        "max_model_error",
        "discount_floor",
    ],
)
def test_dreaming_config_rejects_nonfinite_float_fields(field: str) -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match=field):
            DreamingConfig(**{field: bad})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_model_error_ema", -0.5),
        ("max_uncertainty", -0.5),
        ("min_discount", -0.5),
        ("max_discount", -0.5),
        ("confidence_threshold", -0.5),
        ("max_model_error", -0.5),
        ("discount_floor", -0.5),
    ],
)
def test_dreaming_config_rejects_negative_float_fields(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        DreamingConfig(**{field: value})


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, 2.5, float("nan")])
def test_dreaming_config_rejects_invalid_rollout_horizon(bad: object) -> None:
    with pytest.raises(ValueError, match="rollout_horizon"):
        DreamingConfig(rollout_horizon=bad)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, True, False, 0.0, 1.5, float("nan")])
def test_dreaming_config_rejects_invalid_warmup_steps(bad: object) -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        DreamingConfig(warmup_steps=bad)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, 1, "yes", None, 1.0])
def test_dreaming_config_rejects_non_bool_stop_on_terminal(bad: object) -> None:
    with pytest.raises(ValueError, match="stop_on_terminal"):
        DreamingConfig(stop_on_terminal=bad)  # type: ignore[arg-type]


@pytest.mark.unit
def test_dreaming_config_accepts_valid_boundaries() -> None:
    config = DreamingConfig(
        warmup_steps=0,
        max_model_error_ema=0.0,
        max_uncertainty=0.0,
        min_discount=0.0,
        max_discount=0.0,
        rollout_horizon=1,
        confidence_threshold=0.0,
        max_model_error=0.0,
        discount_floor=0.0,
        stop_on_terminal=False,
    )
    assert config.rollout_horizon == 1
    assert config.max_discount == 0.0
    assert DreamingConfig(max_discount=None).max_discount is None


@pytest.mark.unit
def test_guarded_dreamer_rejects_nan_confidence_threshold_config() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        GuardedDreamer(DreamingConfig(confidence_threshold=float("nan")))


@pytest.mark.unit
def test_dreaming_config_from_config_revalidates() -> None:
    payload = DreamingConfig().to_config()
    payload["confidence_threshold"] = float("nan")
    with pytest.raises(ValueError, match="confidence_threshold"):
        DreamingConfig.from_config(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    ["confidence_threshold", "max_model_error", "discount_floor"],
)
def test_dream_rollout_config_rejects_nonfinite_float_fields(field: str) -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match=field):
            DreamRolloutConfig(**{field: bad})


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, 2.5, float("nan")])
def test_dream_rollout_config_rejects_invalid_rollout_horizon(bad: object) -> None:
    with pytest.raises(ValueError, match="rollout_horizon"):
        DreamRolloutConfig(rollout_horizon=bad)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, 1, "yes", None, 1.0])
def test_dream_rollout_config_rejects_non_bool_stop_on_terminal(bad: object) -> None:
    with pytest.raises(ValueError, match="stop_on_terminal"):
        DreamRolloutConfig(stop_on_terminal=bad)  # type: ignore[arg-type]
