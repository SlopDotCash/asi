# mypy: disable-error-code="call-arg,untyped-decorator"
"""Tests for the single UPGD plus prototype-memory Step 2 learner."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.upgd_memory import (
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    UPGDMemoryState,
    _normalize_simplex,
    run_upgd_memory_arrays,
)


def test_upgd_memory_init_predict_shapes() -> None:
    """Initial hybrid state should expose fixed prediction/metric shapes."""
    config = UPGDMemoryConfig(feature_dim=4, n_heads=3, hidden_sizes=(8,))
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(0))

    prediction = learner.predict(state, jnp.zeros(config.feature_dim))

    chex.assert_shape(prediction, (3,))
    chex.assert_tree_all_finite(prediction)
    chex.assert_tree_all_finite(state.memory_state)
    assert int(state.step_count) == 0
    assert learner.to_config()["type"] == "UPGDMemoryLearner"
    assert UPGDMemoryLearner.from_config(learner.to_config()).config == config


def test_upgd_memory_passes_head_plasticity_to_upgd() -> None:
    """Hybrid config should expose UPGD output-head plasticity controls."""
    config = UPGDMemoryConfig(
        feature_dim=4,
        n_heads=3,
        hidden_sizes=(8,),
        upgd_head_step_size_multiplier=2.0,
        upgd_head_bias_step_size_multiplier=3.0,
        upgd_head_loss_pressure_gate_ratio=1.2,
        upgd_head_loss_pressure_multiplier=1.5,
        upgd_head_loss_pressure_warmup_steps=7,
        upgd_head_repetition_multiplier=2.5,
        upgd_head_repetition_decay=0.8,
        upgd_head_repetition_delta_threshold=0.02,
        upgd_head_repetition_pressure_threshold=0.3,
        upgd_head_repetition_warmup_steps=5,
        target_trace_blend_scale=0.25,
        target_trace_pressure_threshold=0.4,
    )
    learner = UPGDMemoryLearner(config)
    upgd_config = learner.upgd.to_config()

    assert upgd_config["head_step_size_multiplier"] == 2.0
    assert upgd_config["head_bias_step_size_multiplier"] == 3.0
    assert upgd_config["head_loss_pressure_gate_ratio"] == 1.2
    assert upgd_config["head_loss_pressure_multiplier"] == 1.5
    assert upgd_config["head_loss_pressure_warmup_steps"] == 7
    assert upgd_config["head_repetition_multiplier"] == 2.5
    assert upgd_config["head_repetition_decay"] == 0.8
    assert upgd_config["head_repetition_delta_threshold"] == 0.02
    assert upgd_config["head_repetition_pressure_threshold"] == 0.3
    assert upgd_config["head_repetition_warmup_steps"] == 5
    assert UPGDMemoryConfig.from_config(config.to_config()) == config


def test_upgd_memory_target_trace_prior_is_causal() -> None:
    """Repeated targets should optionally bias prequential update predictions."""
    config = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        hidden_sizes=(4,),
        target_trace_blend_scale=0.5,
        target_trace_pressure_threshold=0.0,
    )
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(10))
    target = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    state = learner.update(state, observation, target).state
    state = learner.update(state, observation, target).state
    traced_prediction = learner.update(state, observation, target).predictions

    no_trace = UPGDMemoryLearner(
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            hidden_sizes=(4,),
            target_trace_blend_scale=0.0,
        )
    )
    no_trace_state = no_trace.init(jr.key(10))
    no_trace_state = no_trace.update(no_trace_state, observation, target).state
    no_trace_state = no_trace.update(no_trace_state, observation, target).state
    baseline_prediction = no_trace.update(
        no_trace_state,
        observation,
        target,
    ).predictions

    assert traced_prediction[1] > baseline_prediction[1]
    assert traced_prediction[1] > learner.predict(state, observation)[1]


def test_upgd_memory_updates_both_components() -> None:
    """One-hot targets should train UPGD and allocate memory slots."""
    config = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        hidden_sizes=(4,),
        slots_per_class=2,
        memory_logit_step_size=0.1,
        target_trace_blend_scale=0.0,
    )
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(1))
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    result = learner.update(state, jnp.asarray([1.0, -1.0], dtype=jnp.float32), target)

    chex.assert_shape(result.predictions, (2,))
    chex.assert_shape(result.metrics, (10,))
    assert int(result.state.step_count) == 1
    assert int(result.state.upgd_state.step_count) == 1
    assert int(result.state.memory_state.step_count) == 1
    assert int(jnp.sum(result.state.memory_state.counts > 0.0)) == 1
    assert float(result.metrics[0]) >= 0.0
    assert float(result.metrics[3]) == 0.0
    chex.assert_tree_all_finite(result.metrics)


def test_upgd_memory_scan_runner_is_jit_compatible() -> None:
    """Array runner should work under an outer JIT scan."""
    config = UPGDMemoryConfig(feature_dim=2, n_heads=2, hidden_sizes=(4,))
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(2))
    observations = jnp.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=jnp.float32,
    )

    @jax.jit
    def run(initial_state: UPGDMemoryState):
        return run_upgd_memory_arrays(learner, initial_state, observations, targets)

    result = run(state)

    chex.assert_shape(result.predictions, (4, 2))
    chex.assert_shape(result.metrics, (4, 10))
    assert int(result.state.step_count) == 4
    assert int(jnp.sum(result.state.memory_state.counts > 0.0)) == 2
    chex.assert_tree_all_finite(result.metrics)


def test_upgd_memory_novelty_threshold_adapts() -> None:
    """Runtime novelty threshold should move from the initial value."""
    config = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        hidden_sizes=(4,),
        slots_per_class=4,
        novelty_adaptation_rate=0.2,
        target_allocation_rate=0.0,
    )
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(3))
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    for value in (0.0, 1.0, 2.0):
        state = learner.update(
            state,
            jnp.asarray([value, value], dtype=jnp.float32),
            target,
        ).state

    assert float(jnp.exp(state.novelty_log_threshold)) > config.initial_novelty_threshold


def test_zero_reliability_decay_does_not_multiply_inf_ema() -> None:
    """reliability_decay=0 times an infinite EMA is NaN and would be held."""
    config = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        hidden_sizes=(4,),
        slots_per_class=2,
        reliability_decay=0.0,
    )
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(0)).replace(  # type: ignore[attr-defined]
        allocation_ema=jnp.asarray(jnp.inf, dtype=jnp.float32),
        upgd_loss_ema=jnp.asarray(jnp.inf, dtype=jnp.float32),
        memory_loss_ema=jnp.asarray(jnp.inf, dtype=jnp.float32),
        blended_loss_ema=jnp.asarray(jnp.inf, dtype=jnp.float32),
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = learner.update(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert bool(jnp.isfinite(result.state.allocation_ema))
    assert bool(jnp.isfinite(result.state.upgd_loss_ema))
    assert bool(jnp.isfinite(result.state.memory_loss_ema))
    assert bool(jnp.isfinite(result.state.blended_loss_ema))


def test_zero_reliability_decay_retains_finite_loss_history_in_gate() -> None:
    """A zero EMA decay must not discard finite losses used by the current gate."""
    config = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        hidden_sizes=(4,),
        slots_per_class=2,
        reliability_decay=0.0,
        confidence_logit_scale=0.0,
        reliability_logit_scale=2.0,
    )
    learner = UPGDMemoryLearner(config)
    initial = learner.init(jr.key(4))
    active_memory = initial.memory_state.replace(  # type: ignore[attr-defined]
        counts=jnp.ones_like(initial.memory_state.counts)
    )
    state = initial.replace(  # type: ignore[attr-defined]
        memory_state=active_memory,
        memory_logit=jnp.asarray(0.25, dtype=jnp.float32),
        upgd_loss_ema=jnp.asarray(1.5, dtype=jnp.float32),
        memory_loss_ema=jnp.asarray(0.5, dtype=jnp.float32),
    )

    gate = learner._blend_gate(
        state,
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
    )

    expected = jax.nn.sigmoid(jnp.asarray(2.25, dtype=jnp.float32))
    chex.assert_trees_all_close(gate, expected)


def test_zero_reliability_decay_does_not_relax_consumed_non_ema_state() -> None:
    """Only disabled EMA history is recoverable; a poisoned gate logit remains fatal."""
    config = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        hidden_sizes=(4,),
        slots_per_class=2,
        reliability_decay=0.0,
    )
    learner = UPGDMemoryLearner(config)
    state = learner.init(jr.key(5)).replace(  # type: ignore[attr-defined]
        memory_logit=jnp.asarray(jnp.inf, dtype=jnp.float32)
    )

    result = learner.update(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    # The wall-clock diagnostic is materialized through JIT at float32
    # precision; normalize it before asserting transactional rollback.
    expected = state.replace(  # type: ignore[attr-defined]
        upgd_state=state.upgd_state.replace(  # type: ignore[attr-defined]
            birth_timestamp=result.state.upgd_state.birth_timestamp
        )
    )
    chex.assert_trees_all_equal(result.state, expected)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros((2,), dtype=jnp.float32))


_INVALID_UPGD_MEMORY_CONFIGS: tuple[dict[str, object], ...] = (
    {"feature_dim": 0, "n_heads": 2},
    {"feature_dim": -1, "n_heads": 2},
    {"feature_dim": 2**31, "n_heads": 2},
    {"feature_dim": True, "n_heads": 2},
    {"feature_dim": "2", "n_heads": 2},
    {"feature_dim": 2, "n_heads": 1},
    {"feature_dim": 2, "n_heads": 0},
    {"feature_dim": 2, "n_heads": -1},
    {"feature_dim": 2, "n_heads": 2**31},
    {"feature_dim": 2, "n_heads": True},
    {"feature_dim": 2, "n_heads": 2, "hidden_sizes": (0,)},
    {"feature_dim": 2, "n_heads": 2, "hidden_sizes": (2**31,)},
    {"feature_dim": 2, "n_heads": 2, "hidden_sizes": (True,)},
    {"feature_dim": 2, "n_heads": 2, "readout_mode": "invalid_mode"},
    {"feature_dim": 2, "n_heads": 2, "upgd_step_size": 0.0},
    {"feature_dim": 2, "n_heads": 2, "upgd_step_size": -0.01},
    {"feature_dim": 2, "n_heads": 2, "upgd_step_size": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_step_size": float("nan")},
    {"feature_dim": 2, "n_heads": 2, "upgd_step_size": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_step_size_multiplier": 0.0},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_step_size_multiplier": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_step_size_multiplier": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_step_size_multiplier": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_bias_step_size_multiplier": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_bias_step_size_multiplier": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_bias_step_size_multiplier": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_gate_ratio": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_gate_ratio": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_gate_ratio": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_multiplier": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_multiplier": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_multiplier": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_warmup_steps": -1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_warmup_steps": 2**31},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_loss_pressure_warmup_steps": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_multiplier": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_multiplier": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_multiplier": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_decay": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_decay": 1.0},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_decay": 1.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_decay": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_decay": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_delta_threshold": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_delta_threshold": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_delta_threshold": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_pressure_threshold": -0.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_pressure_threshold": 1.0},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_pressure_threshold": 1.1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_pressure_threshold": 1e100},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_pressure_threshold": True},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_warmup_steps": -1},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_warmup_steps": 2**31},
    {"feature_dim": 2, "n_heads": 2, "upgd_head_repetition_warmup_steps": True},
    {"feature_dim": 2, "n_heads": 2, "slots_per_class": 0},
    {"feature_dim": 2, "n_heads": 2, "slots_per_class": -1},
    {"feature_dim": 2, "n_heads": 2, "slots_per_class": 2**31},
    {"feature_dim": 2, "n_heads": 2, "slots_per_class": True},
    {"feature_dim": 2, "n_heads": 2, "memory_update_rate": 0.0},
    {"feature_dim": 2, "n_heads": 2, "memory_update_rate": -0.1},
    {"feature_dim": 2, "n_heads": 2, "memory_update_rate": 1.1},
    {"feature_dim": 2, "n_heads": 2, "memory_update_rate": 1e100},
    {"feature_dim": 2, "n_heads": 2, "memory_update_rate": True},
    {"feature_dim": 2, "n_heads": 2, "initial_novelty_threshold": 0.0},
    {"feature_dim": 2, "n_heads": 2, "initial_novelty_threshold": -0.1},
    {"feature_dim": 2, "n_heads": 2, "initial_novelty_threshold": 1e100},
    {"feature_dim": 2, "n_heads": 2, "initial_novelty_threshold": True},
    {"feature_dim": 2, "n_heads": 2, "memory_bandwidth": 0.0},
    {"feature_dim": 2, "n_heads": 2, "memory_bandwidth": -0.1},
    {"feature_dim": 2, "n_heads": 2, "memory_bandwidth": 1e100},
    {"feature_dim": 2, "n_heads": 2, "memory_bandwidth": True},
    {"feature_dim": 2, "n_heads": 2, "initial_memory_logit": 1e100},
    {"feature_dim": 2, "n_heads": 2, "initial_memory_logit": True},
    {"feature_dim": 2, "n_heads": 2, "memory_logit_step_size": -0.1},
    {"feature_dim": 2, "n_heads": 2, "memory_logit_step_size": 1e100},
    {"feature_dim": 2, "n_heads": 2, "memory_logit_step_size": True},
    {"feature_dim": 2, "n_heads": 2, "confidence_logit_scale": 1e100},
    {"feature_dim": 2, "n_heads": 2, "confidence_logit_scale": True},
    {"feature_dim": 2, "n_heads": 2, "reliability_logit_scale": 1e100},
    {"feature_dim": 2, "n_heads": 2, "reliability_logit_scale": True},
    {"feature_dim": 2, "n_heads": 2, "reliability_decay": -0.1},
    {"feature_dim": 2, "n_heads": 2, "reliability_decay": 1.0},
    {"feature_dim": 2, "n_heads": 2, "reliability_decay": 1.1},
    {"feature_dim": 2, "n_heads": 2, "reliability_decay": 1e100},
    {"feature_dim": 2, "n_heads": 2, "reliability_decay": True},
    {"feature_dim": 2, "n_heads": 2, "target_trace_blend_scale": -0.1},
    {"feature_dim": 2, "n_heads": 2, "target_trace_blend_scale": 1.1},
    {"feature_dim": 2, "n_heads": 2, "target_trace_blend_scale": 1e100},
    {"feature_dim": 2, "n_heads": 2, "target_trace_blend_scale": True},
    {"feature_dim": 2, "n_heads": 2, "target_trace_pressure_threshold": -0.1},
    {"feature_dim": 2, "n_heads": 2, "target_trace_pressure_threshold": 1.0},
    {"feature_dim": 2, "n_heads": 2, "target_trace_pressure_threshold": 1.1},
    {"feature_dim": 2, "n_heads": 2, "target_trace_pressure_threshold": 1e100},
    {"feature_dim": 2, "n_heads": 2, "target_trace_pressure_threshold": True},
    {"feature_dim": 2, "n_heads": 2, "novelty_adaptation_rate": -0.1},
    {"feature_dim": 2, "n_heads": 2, "novelty_adaptation_rate": 1e100},
    {"feature_dim": 2, "n_heads": 2, "novelty_adaptation_rate": True},
    {"feature_dim": 2, "n_heads": 2, "target_allocation_rate": -0.1},
    {"feature_dim": 2, "n_heads": 2, "target_allocation_rate": 1.1},
    {"feature_dim": 2, "n_heads": 2, "target_allocation_rate": 1e100},
    {"feature_dim": 2, "n_heads": 2, "target_allocation_rate": True},
    {"feature_dim": 2, "n_heads": 2, "min_novelty_threshold": 0.0},
    {"feature_dim": 2, "n_heads": 2, "min_novelty_threshold": -0.1},
    {"feature_dim": 2, "n_heads": 2, "min_novelty_threshold": 1e100},
    {"feature_dim": 2, "n_heads": 2, "min_novelty_threshold": True},
    {"feature_dim": 2, "n_heads": 2, "max_novelty_threshold": 0.0},
    {"feature_dim": 2, "n_heads": 2, "max_novelty_threshold": -0.1},
    {"feature_dim": 2, "n_heads": 2, "max_novelty_threshold": 1e100},
    {"feature_dim": 2, "n_heads": 2, "max_novelty_threshold": True},
    {"feature_dim": 2, "n_heads": 2, "min_novelty_threshold": 0.5, "max_novelty_threshold": 0.1},
)


@pytest.mark.parametrize("kwargs", _INVALID_UPGD_MEMORY_CONFIGS)
def test_upgd_memory_config_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UPGDMemoryConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden_sizes": ()},
        {"target_allocation_rate": 1.0},
        {"min_novelty_threshold": 0.25, "max_novelty_threshold": 0.25},
    ],
)
def test_upgd_memory_preserves_legal_boundary_configs(
    overrides: dict[str, object],
) -> None:
    config = UPGDMemoryConfig(feature_dim=2, n_heads=2, **overrides)
    assert config.target_allocation_rate <= 1.0
    assert config.min_novelty_threshold <= config.max_novelty_threshold


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param((-1, 1), id="negative-ratio"),
        pytest.param((2, 1), id="above-unit-ratio"),
        pytest.param((-1, 2**200), id="negative-rounds-to-negative-zero"),
        pytest.param((2**200 + 1, 2**200), id="above-one-rounds-to-one"),
    ],
)
def test_upgd_memory_rejects_adversarial_ratio_floats(
    ratio: tuple[int, int]
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    with pytest.raises(ValueError, match="memory_update_rate"):
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            memory_update_rate=HiddenBoundaryFloat(0.5),
        )


def test_upgd_memory_rejects_class_property_spoofing_float() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

    value = ClassSpoof()
    with pytest.raises(ValueError, match="finite real"):
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            upgd_step_size=value,  # type: ignore[arg-type]
        )


def test_upgd_memory_rejects_spoofed_tuple_container() -> None:
    class SpoofedTuple(list):
        @property
        def __class__(self) -> type[tuple]:
            return tuple

    with pytest.raises(TypeError, match="hidden_sizes"):
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            hidden_sizes=SpoofedTuple([64]),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="hidden_sizes"):
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            hidden_sizes=[64],  # type: ignore[arg-type]
        )


def test_upgd_memory_rejects_equality_spoofed_readout_mode() -> None:
    class SpoofedReadoutMode:
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash("softmax_ce")

    with pytest.raises(TypeError, match="readout_mode"):
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            readout_mode=SpoofedReadoutMode(),  # type: ignore[arg-type]
        )


def test_upgd_memory_rejects_spoofed_int_class_and_negative_ratios() -> None:
    class SpoofedIntFloat(float):
        @property
        def __class__(self) -> type[int]:
            return int

        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match="upgd_step_size"):
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            upgd_step_size=SpoofedIntFloat(0.5),
        )


def test_upgd_memory_json_roundtrip() -> None:
    import json

    config = UPGDMemoryConfig(
        feature_dim=4,
        n_heads=3,
        hidden_sizes=(16, 8),
        readout_mode="softmax_ce",
        upgd_step_size=0.05,
    )
    serialized = config.to_dict()
    json_str = json.dumps(serialized)
    deserialized = json.loads(json_str)
    restored = UPGDMemoryConfig.from_dict(deserialized)

    assert restored == config
    assert restored.hidden_sizes == (16, 8)
    assert restored.readout_mode == "softmax_ce"


def test_upgd_memory_rejects_hostile_integral_subclasses() -> None:
    class LieInt(int):
        def __int__(self) -> int:
            return 4

    defaults: dict[str, object] = {"feature_dim": 2, "n_heads": 2}
    for field in (
        "feature_dim",
        "n_heads",
        "upgd_head_loss_pressure_warmup_steps",
        "upgd_head_repetition_warmup_steps",
        "slots_per_class",
    ):
        with pytest.raises(ValueError, match=field):
            UPGDMemoryConfig(**{**defaults, field: LieInt(-1)})  # type: ignore[arg-type]


def test_upgd_memory_preserves_legal_closed_endpoints() -> None:
    allocation_endpoint = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        target_allocation_rate=1.0,
    )
    fixed_threshold = UPGDMemoryConfig(
        feature_dim=2,
        n_heads=2,
        min_novelty_threshold=0.5,
        max_novelty_threshold=0.5,
    )

    assert allocation_endpoint.target_allocation_rate == 1.0
    assert fixed_threshold.min_novelty_threshold == 0.5
    assert fixed_threshold.max_novelty_threshold == 0.5


@pytest.mark.parametrize(
    ("label", "prediction"),
    (
        # Reachable: UPGDLearner initializes previous_targets to zeros, so the
        # target-trace blend mixes against this until the first target arrives.
        ("zero mass", [0.0, 0.0, 0.0]),
        ("wholly negative", [-1.0, -2.0, -3.0]),
        # A single entry large enough to dominate the float32 sum makes every
        # ratio underflow to zero.
        ("underflowing spread", [1e38, 1.0, 1.0]),
        # Positive totals below the old 1e-12 denominator floor were divided by
        # the floor rather than themselves, so this normalized to 0.75.
        ("mass below the old floor", [7.5e-13, 0.0, 0.0]),
        ("far below the old floor", [1e-30, 0.0, 0.0]),
    ),
)
def test_normalize_simplex_returns_a_simplex_for_degenerate_mass(
    label: str,
    prediction: list[float],
) -> None:
    """The helper's name is its contract; every branch must sum to one."""
    normalized = _normalize_simplex(jnp.asarray(prediction, dtype=jnp.float32))

    chex.assert_tree_all_finite(normalized)
    assert float(jnp.min(normalized)) >= 0.0, label
    chex.assert_trees_all_close(jnp.sum(normalized), 1.0, atol=1e-5)


def test_normalize_simplex_leaves_ordinary_inputs_unchanged() -> None:
    """Well-formed inputs must not move."""
    chex.assert_trees_all_close(
        _normalize_simplex(jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)),
        jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32),
        atol=1e-6,
    )
    chex.assert_trees_all_close(
        _normalize_simplex(jnp.asarray([-1.0, 2.0, 1.0], dtype=jnp.float32)),
        jnp.asarray([0.0, 2.0 / 3.0, 1.0 / 3.0], dtype=jnp.float32),
        atol=1e-6,
    )


def test_target_trace_blend_preserves_mass_before_the_first_target() -> None:
    """Blending a simplex against the initial zero trace must not lose mass.

    ``previous_targets`` starts as zeros, and the blend is
    ``(1 - trace_gate) * prediction + trace_gate * trace_prediction``. When the
    trace normalized to a zero vector the blended mass collapsed to
    ``1 - trace_gate`` -- up to 80% of the distribution gone at the default
    ``target_trace_blend_scale``.
    """
    prediction = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)
    trace_prediction = _normalize_simplex(jnp.zeros(3, dtype=jnp.float32))

    for trace_gate in (0.0, 0.25, 0.5, 0.8, 1.0):
        blended = (1.0 - trace_gate) * prediction + trace_gate * trace_prediction
        chex.assert_trees_all_close(jnp.sum(blended), 1.0, atol=1e-5)
