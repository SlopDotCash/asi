# mypy: disable-error-code="call-arg,untyped-decorator"
"""Tests for lightweight working-memory predictive-state features."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    WorkingMemoryState,
    transform_working_memory_arrays,
)


def test_working_memory_config_roundtrip_and_shapes() -> None:
    config = WorkingMemoryConfig(
        observation_dim=3,
        action_dim=2,
        reward_dim=1,
        observation_decay_rates=(0.5, 0.9),
        action_decay_rates=(0.25,),
        reward_decay_rates=(0.0, 0.8),
        include_innovations=True,
    )
    memory = WorkingMemoryFeaturizer(config)
    restored = WorkingMemoryFeaturizer.from_config(memory.to_config())
    state = restored.init()

    assert restored.config == config
    assert restored.feature_dim() == config.feature_dim()
    chex.assert_shape(state.observation_traces, (2, 3))
    chex.assert_shape(state.action_traces, (1, 2))
    chex.assert_shape(state.reward_traces, (2, 1))
    chex.assert_shape(state.last_gate, (3,))


def test_working_memory_config_from_config_accepts_exact_lists_or_tuples() -> None:
    class HostileList(list[float]):
        def __iter__(self) -> Iterator[float]:
            raise AssertionError("invalid list subclass must not be iterated")

    class HostileTuple(tuple):
        pass

    base = WorkingMemoryConfig(observation_dim=2, action_dim=1).to_config()
    keys = (
        "observation_decay_rates",
        "action_decay_rates",
        "reward_decay_rates",
    )
    invalid_values: tuple[object, ...] = (
        "0.5",
        np.asarray([0.5]),
        HostileList([0.5]),
        HostileTuple((0.5,)),
        0.5,
    )
    for key in keys:
        for value in invalid_values:
            payload = {**base, key: value}
            with pytest.raises(ValueError, match=key):
                WorkingMemoryConfig.from_config(payload)  # type: ignore[arg-type]

    restored = WorkingMemoryConfig.from_config(base)
    assert type(restored.observation_decay_rates) is tuple
    assert type(restored.action_decay_rates) is tuple
    assert type(restored.reward_decay_rates) is tuple

    tuple_payload = {
        **base,
        "observation_decay_rates": tuple(base["observation_decay_rates"]),
        "action_decay_rates": tuple(base["action_decay_rates"]),
        "reward_decay_rates": tuple(base["reward_decay_rates"]),
    }
    assert WorkingMemoryConfig.from_config(tuple_payload) == WorkingMemoryConfig.from_config(
        base
    )


def test_working_memory_trace_decay_and_feature_causality() -> None:
    config = WorkingMemoryConfig(
        observation_dim=1,
        action_dim=0,
        reward_dim=0,
        observation_decay_rates=(0.5,),
        include_current_observation=False,
        include_current_action=False,
        include_current_reward=False,
    )
    memory = WorkingMemoryFeaturizer(config)
    state = memory.init()

    state1, features1 = memory.step(
        state,
        jnp.asarray([2.0]),
        memory.zero_action(),
        memory.zero_reward(),
    )
    state2, features2 = memory.step(
        state1,
        jnp.asarray([0.0]),
        memory.zero_action(),
        memory.zero_reward(),
    )

    chex.assert_trees_all_close(features1, jnp.asarray([0.0]))
    chex.assert_trees_all_close(state1.observation_traces[0], jnp.asarray([1.0]))
    chex.assert_trees_all_close(features2, jnp.asarray([1.0]))
    chex.assert_trees_all_close(state2.observation_traces[0], jnp.asarray([0.5]))


def test_working_memory_reset_semantics() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(observation_dim=2, action_dim=1, reward_dim=1)
    )
    state, _ = memory.step(
        memory.init(),
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([0.5]),
    )
    reset = memory.reset()

    assert int(state.step_count) == 1
    assert int(reset.step_count) == 0
    chex.assert_trees_all_close(reset.observation_traces, jnp.zeros_like(reset.observation_traces))
    chex.assert_trees_all_close(reset.action_traces, jnp.zeros_like(reset.action_traces))
    chex.assert_trees_all_close(reset.reward_traces, jnp.zeros_like(reset.reward_traces))


def test_working_memory_action_and_reward_are_included() -> None:
    config = WorkingMemoryConfig(
        observation_dim=1,
        action_dim=3,
        reward_dim=1,
        observation_decay_rates=(),
        action_decay_rates=(0.0,),
        reward_decay_rates=(0.0,),
        include_current_observation=False,
        include_current_action=False,
        include_current_reward=False,
    )
    memory = WorkingMemoryFeaturizer(config)
    action = jnp.asarray([0.0, 1.0, 0.0])
    reward = jnp.asarray([2.5])

    state, features0 = memory.step(memory.init(), jnp.asarray([0.0]), action, reward)
    _, features1 = memory.step(state, jnp.asarray([0.0]), jnp.zeros(3), jnp.zeros(1))

    chex.assert_trees_all_close(features0, jnp.zeros(4))
    chex.assert_trees_all_close(features1, jnp.asarray([0.0, 1.0, 0.0, 2.5]))


def test_working_memory_gated_update_can_hold_traces() -> None:
    config = WorkingMemoryConfig(
        observation_dim=1,
        action_dim=0,
        reward_dim=0,
        observation_decay_rates=(0.0,),
        include_current_observation=False,
        include_current_action=False,
        include_current_reward=False,
    )
    memory = WorkingMemoryFeaturizer(config)
    state = memory.init()
    state1 = memory.update(state, jnp.asarray([1.0]), memory.zero_action(), memory.zero_reward())
    held = memory.update(
        state1,
        jnp.asarray([5.0]),
        memory.zero_action(),
        memory.zero_reward(),
        external_gate=0.0,
    )

    chex.assert_trees_all_close(held.observation_traces, state1.observation_traces)
    chex.assert_trees_all_close(held.last_gate, jnp.zeros(3))


def test_working_memory_scan_and_jit_compatibility() -> None:
    config = WorkingMemoryConfig(
        observation_dim=2,
        action_dim=2,
        reward_dim=1,
        observation_decay_rates=(0.5, 0.9),
        include_innovations=True,
    )
    memory = WorkingMemoryFeaturizer(config)
    observations = jnp.arange(20, dtype=jnp.float32).reshape(10, 2) / 10.0
    action_ids = jnp.arange(10) % 2
    actions = jax.nn.one_hot(action_ids, 2)
    rewards = jnp.linspace(-1.0, 1.0, 10).reshape(10, 1)

    @jax.jit
    def run(initial_state: WorkingMemoryState):
        return transform_working_memory_arrays(
            memory,
            observations,
            actions,
            rewards,
            state=initial_state,
        )

    final_state, features = run(memory.init())

    chex.assert_shape(features, (10, config.feature_dim()))
    chex.assert_tree_all_finite(features)
    chex.assert_tree_all_finite(final_state)
    assert int(final_state.step_count) == 10


def test_working_memory_step_reports_rejection_with_neutral_features_and_recovers() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.0,),
            include_current_observation=False,
            include_current_action=False,
            include_current_reward=False,
        )
    )
    state = memory.update(
        memory.init(),
        jnp.asarray([2.0]),
        memory.zero_action(),
        memory.zero_reward(),
    )

    checked_update = memory.update_checked(
        state,
        jnp.asarray([jnp.inf]),
        memory.zero_action(),
        memory.zero_reward(),
    )
    assert not bool(checked_update.update_applied)
    chex.assert_trees_all_equal(checked_update.state, state)

    result = memory.step(
        state,
        jnp.asarray([jnp.inf]),
        memory.zero_action(),
        memory.zero_reward(),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.features, jnp.zeros((1,), dtype=jnp.float32))
    legacy_state, legacy_features = result
    chex.assert_trees_all_equal(legacy_state, result.state)
    chex.assert_trees_all_equal(legacy_features, result.features)

    recovered = memory.step(
        result.state,
        jnp.asarray([3.0]),
        memory.zero_action(),
        memory.zero_reward(),
    )
    assert bool(recovered.update_applied)
    assert int(recovered.state.step_count) == int(state.step_count) + 1


def test_working_memory_step_rejects_finite_post_apply_overflow() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.5,),
            include_current_observation=False,
            include_current_action=False,
            include_current_reward=False,
        )
    )
    limit = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    state = memory.init().replace(observation_traces=-limit.reshape((1, 1)))

    rejected = memory.step(
        state,
        jnp.asarray([limit]),
        memory.zero_action(),
        memory.zero_reward(),
    )

    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, state)
    chex.assert_trees_all_equal(rejected.features, jnp.zeros((1,), dtype=jnp.float32))
    recovered = memory.step(
        rejected.state,
        jnp.asarray([-limit]),
        memory.zero_action(),
        memory.zero_reward(),
    )
    assert bool(recovered.update_applied)
    chex.assert_tree_all_finite(recovered.state)


def test_working_memory_jit_reports_gate_only_rejection_and_rolls_back() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.5,),
            include_current_observation=False,
            include_current_action=False,
            include_current_reward=False,
        )
    )
    state = memory.update(
        memory.init(),
        jnp.asarray([2.0]),
        memory.zero_action(),
        memory.zero_reward(),
    )

    compiled_step = jax.jit(
        lambda current, gate: memory.step(
            current,
            jnp.asarray([1.0]),
            memory.zero_action(),
            memory.zero_reward(),
            gate,
        )
    )
    rejected = compiled_step(state, jnp.asarray(jnp.nan, dtype=jnp.float32))

    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, state)
    chex.assert_trees_all_equal(rejected.features, jnp.zeros((1,), dtype=jnp.float32))


def test_working_memory_array_result_exposes_per_step_transaction_mask() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.0,),
            include_current_observation=False,
            include_current_action=False,
            include_current_reward=False,
        )
    )
    observations = jnp.asarray([[1.0], [jnp.inf], [2.0], [3.0]], dtype=jnp.float32)
    actions = jnp.zeros((4, 0), dtype=jnp.float32)
    rewards = jnp.zeros((4, 0), dtype=jnp.float32)
    gates = jnp.asarray([1.0, 1.0, jnp.nan, 1.0], dtype=jnp.float32)

    result = jax.jit(
        lambda: transform_working_memory_arrays(
            memory,
            observations,
            actions,
            rewards,
            external_gates=gates,
        )
    )()

    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.asarray([True, False, False, True]),
    )
    chex.assert_trees_all_equal(result.features[:, 0], jnp.asarray([0.0, 0.0, 0.0, 1.0]))
    assert int(result.state.step_count) == 2
    chex.assert_trees_all_close(result.state.observation_traces, jnp.asarray([[3.0]]))
    legacy_state, legacy_features = result
    chex.assert_trees_all_equal(legacy_state, result.state)
    chex.assert_trees_all_equal(legacy_features, result.features)


def test_working_memory_diagnostics_are_finite() -> None:
    memory = WorkingMemoryFeaturizer(WorkingMemoryConfig(observation_dim=2, action_dim=1))
    state = memory.update(
        memory.init(),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([1.0]),
        jnp.asarray([0.5]),
    )
    diagnostics = memory.diagnostics(state)

    assert int(diagnostics.step_count) == 1
    assert float(diagnostics.trace_energy) > 0.0
    assert float(diagnostics.effective_dimension) > 0.0
    chex.assert_tree_all_finite(diagnostics)


def test_working_memory_delayed_action_positive_control() -> None:
    config = WorkingMemoryConfig(
        observation_dim=1,
        action_dim=2,
        reward_dim=0,
        observation_decay_rates=(),
        action_decay_rates=(0.0,),
        reward_decay_rates=(),
        include_current_observation=False,
        include_current_action=False,
        include_current_reward=False,
    )
    memory = WorkingMemoryFeaturizer(config)
    observations = jnp.zeros((8, 1), dtype=jnp.float32)
    action_ids = jnp.asarray([0, 1, 1, 0, 1, 0, 0, 1])
    actions = jax.nn.one_hot(action_ids, 2)
    rewards = jnp.zeros((8, 0), dtype=jnp.float32)
    _, features = transform_working_memory_arrays(memory, observations, actions, rewards)

    delayed_first_action = jnp.concatenate(
        [jnp.asarray([0.0]), actions[:-1, 0]],
        axis=0,
    )
    memory_prediction = features[:, 0]
    raw_prediction = jnp.zeros_like(delayed_first_action)
    memory_mse = jnp.mean((memory_prediction - delayed_first_action) ** 2)
    raw_mse = jnp.mean((raw_prediction - delayed_first_action) ** 2)

    chex.assert_trees_all_close(memory_mse, 0.0)
    assert float(memory_mse) < float(raw_mse)


def _minimal_config(**overrides: object) -> WorkingMemoryConfig:
    payload: dict[str, object] = {
        "observation_dim": 2,
        "action_dim": 0,
        "reward_dim": 0,
        "observation_decay_rates": (0.5,),
        "action_decay_rates": (),
        "reward_decay_rates": (),
        "include_current_action": False,
        "include_current_reward": False,
    }
    payload.update(overrides)
    return WorkingMemoryConfig(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_dim", True),
        ("observation_dim", 1.5),
        ("observation_dim", 0),
        ("observation_dim", 2**31),
        ("action_dim", True),
        ("action_dim", -1),
        ("action_dim", 2**31),
        ("reward_dim", True),
        ("reward_dim", -1),
        ("reward_dim", 2**31),
    ],
)
def test_config_rejects_invalid_int32_dimensions(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _minimal_config(**{field: value})


def test_config_rejects_hostile_integral_subclasses() -> None:
    class LieInt(int):
        def __int__(self) -> int:
            return 2

    for field in ("observation_dim", "action_dim", "reward_dim"):
        with pytest.raises(ValueError, match=field):
            _minimal_config(**{field: LieInt(1)})


def test_config_canonicalizes_supported_numpy_dimensions() -> None:
    config = _minimal_config(
        observation_dim=np.longlong(2),
        action_dim=np.ulonglong(1),
        reward_dim=np.int32(0),
    )
    assert type(config.observation_dim) is int
    assert type(config.action_dim) is int
    assert type(config.reward_dim) is int


@pytest.mark.parametrize(
    "rates",
    [[0.5], 0.5, (False,), (1.0,), (-0.1,), (1e100,)],
)
def test_config_rejects_invalid_decay_containers_and_float32_values(
    rates: object,
) -> None:
    with pytest.raises(ValueError, match="observation_decay_rates"):
        _minimal_config(observation_decay_rates=rates)


def test_config_rejects_tuple_and_bool_class_spoofs() -> None:
    class TupleSubclass(tuple):
        pass

    class BoolSpoof:
        @property
        def __class__(self) -> type[bool]:
            return bool

    with pytest.raises(ValueError, match="observation_decay_rates"):
        _minimal_config(observation_decay_rates=TupleSubclass((0.5,)))
    for field in (
        "include_current_observation",
        "include_current_action",
        "include_current_reward",
        "include_traces",
        "include_innovations",
        "gated_update",
    ):
        with pytest.raises(ValueError, match=field):
            _minimal_config(**{field: BoolSpoof()})


@pytest.mark.parametrize(
    ("field", "ratio"),
    [
        ("observation_decay_rates", (-1, 2**200)),
        ("observation_decay_rates", (2**200 - 1, 2**200)),
        ("gate_threshold", (-1, 2**200)),
        ("gate_temperature", (-1, 2**200)),
    ],
)
def test_config_rejects_hostile_exact_float_ratios(
    field: str,
    ratio: tuple[int, int],
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    value: object = HiddenBoundaryFloat(0.5)
    if field.endswith("decay_rates"):
        value = (value,)
    with pytest.raises(ValueError, match=field):
        _minimal_config(**{field: value})


def test_config_canonicalizes_float32_scalars_and_round_trips() -> None:
    config = _minimal_config(
        observation_decay_rates=(
            Fraction(1, 10),
            np.float64(0.25),
            Fraction(1, 10**400),
        ),
        gate_threshold=Fraction(1, 10),
        gate_temperature=np.float64(0.5),
    )
    assert config.observation_decay_rates == (float(np.float32(0.1)), 0.25, 0.0)
    assert type(config.gate_threshold) is float
    assert config.gate_threshold == float(np.float32(0.1))
    assert type(config.gate_temperature) is float
    assert WorkingMemoryConfig.from_config(config.to_config()) == config


def test_config_rejects_derived_feature_dimension_above_int32() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        WorkingMemoryConfig(observation_dim=2**31 - 1)


def test_config_accepts_exact_int32_max_derived_feature_dimension() -> None:
    config = WorkingMemoryConfig(
        observation_dim=2**31 - 1,
        action_dim=0,
        reward_dim=0,
        observation_decay_rates=(),
        action_decay_rates=(),
        reward_decay_rates=(),
        include_current_action=False,
        include_current_reward=False,
        include_traces=False,
    )

    assert config.feature_dim() == 2**31 - 1


@pytest.mark.parametrize("method", ["features", "update_checked", "step"])
@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("observation", (2, 1)),
        ("observation", (1, 2)),
        ("observation", ()),
        ("action", (1, 1)),
        ("action", ()),
        ("reward", (1, 1)),
        ("reward", ()),
    ],
)
def test_vector_entry_points_reject_same_size_wrong_shapes(
    method: str,
    field: str,
    shape: tuple[int, ...],
) -> None:
    memory = WorkingMemoryFeaturizer(
        _minimal_config(action_dim=1, reward_dim=1, include_current_action=True)
    )
    values = {
        "observation": jnp.ones((2,), dtype=jnp.float32),
        "action": jnp.ones((1,), dtype=jnp.float32),
        "reward": jnp.ones((1,), dtype=jnp.float32),
    }
    values[field] = jnp.ones(shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match="vector must have shape"):
        getattr(memory, method)(
            memory.init(),
            values["observation"],
            values["action"],
            values["reward"],
        )


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("observations", (3,)),
        ("observations", (3, 1)),
        ("actions", (3,)),
        ("actions", (3, 2)),
        ("rewards", (3,)),
        ("rewards", (3, 2)),
        ("actions", (2, 1)),
        ("rewards", (4, 1)),
        ("external_gates", (3, 1)),
        ("external_gates", (2,)),
    ],
)
def test_batch_transform_rejects_wrong_shapes(
    field: str,
    shape: tuple[int, ...],
) -> None:
    memory = WorkingMemoryFeaturizer(
        _minimal_config(action_dim=1, reward_dim=1, include_current_action=True)
    )
    values: dict[str, object] = {
        "observations": jnp.ones((3, 2), dtype=jnp.float32),
        "actions": jnp.ones((3, 1), dtype=jnp.float32),
        "rewards": jnp.ones((3, 1), dtype=jnp.float32),
        "external_gates": jnp.ones((3,), dtype=jnp.float32),
    }
    values[field] = jnp.ones(shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match=field if field != "external_gates" else "external_gates"):
        transform_working_memory_arrays(
            memory,
            values["observations"],  # type: ignore[arg-type]
            values["actions"],  # type: ignore[arg-type]
            values["rewards"],  # type: ignore[arg-type]
            external_gates=values["external_gates"],  # type: ignore[arg-type]
        )


def test_zero_width_vectors_and_batches_preserve_exact_shapes() -> None:
    memory = WorkingMemoryFeaturizer(_minimal_config())
    state = memory.init()
    features = memory.features(
        state,
        jnp.ones((2,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
    )
    result = transform_working_memory_arrays(
        memory,
        jnp.ones((3, 2), dtype=jnp.float32),
        jnp.empty((3, 0), dtype=jnp.float32),
        jnp.empty((3, 0), dtype=jnp.float32),
    )
    chex.assert_shape(features, (memory.feature_dim(),))
    chex.assert_shape(result.features, (3, memory.feature_dim()))


@pytest.mark.parametrize("field", [
    "observation_decay_rates",
    "action_decay_rates",
    "reward_decay_rates",
])
@pytest.mark.parametrize(
    "rates",
    [
        (float("nan"),),
        (0.5, float("nan")),
        (float("inf"),),
        (float("-inf"),),
        (1.0,),
        (-0.1,),
    ],
)
def test_decay_rates_must_be_finite_and_in_unit_half_open_interval(
    field: str, rates: tuple[float, ...]
) -> None:
    extras: dict[str, object] = {field: rates}
    if field != "observation_decay_rates":
        extras["observation_decay_rates"] = (0.5,)
    with pytest.raises(ValueError, match=field):
        WorkingMemoryFeaturizer(_minimal_config(**extras))


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_gate_temperature_must_be_finite_and_positive(temperature: float) -> None:
    with pytest.raises(ValueError, match="gate_temperature"):
        WorkingMemoryFeaturizer(
            _minimal_config(gated_update=True, gate_temperature=temperature)
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_gate_threshold_must_be_finite_and_nonnegative(threshold: float) -> None:
    with pytest.raises(ValueError, match="gate_threshold"):
        WorkingMemoryFeaturizer(
            _minimal_config(gated_update=True, gate_threshold=threshold)
        )


def test_legal_finite_decay_endpoints_still_construct_and_update() -> None:
    memory = WorkingMemoryFeaturizer(
        _minimal_config(
            observation_decay_rates=(0.0, 0.5, 0.99),
            gate_temperature=1.0,
            gate_threshold=0.0,
        )
    )
    checked = memory.update_checked(
        memory.init(),
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        memory.zero_action(),
        memory.zero_reward(),
    )
    assert bool(checked.update_applied)
    chex.assert_tree_all_finite(checked.state)


def test_zero_decay_does_not_multiply_inf_traces() -> None:
    """decay=0 is a full replace; inf leftover traces must not become NaN."""
    memory = WorkingMemoryFeaturizer(
        _minimal_config(observation_decay_rates=(0.0,), gated_update=False)
    )
    obs = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
    state = memory.update_checked(
        memory.init(),
        obs,
        memory.zero_action(),
        memory.zero_reward(),
    ).state
    state = state.replace(  # type: ignore[attr-defined]
        observation_traces=jnp.full_like(state.observation_traces, jnp.inf),
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = memory.update_checked(
        state,
        obs,
        memory.zero_action(),
        memory.zero_reward(),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_close(result.state.observation_traces, obs[None, :])
