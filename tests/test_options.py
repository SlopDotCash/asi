"""Tests for STOMP checkpoint payloads and state migration (core/options.py)."""

import dataclasses
from types import MappingProxyType

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.options import (
    DISPATCH_OWNER_BASE_PRIMITIVE,
    STOMP_OPTION_MODEL_EXPANSION_FIELDS,
    STOMP_STATE_EXPANSION_FIELDS,
    STOMP_STATE_LIFETIME_FIELDS,
    STOMPAgent,
    STOMPConfig,
    STOMPSpecArrays,
    STOMPState,
    SubtaskSpec,
    _differential_q_update,
    _differential_semidp_q_update,
    _stomp_direct_array_scalars,
    load_stomp_state_with_migration,
    replace_dispatched_primitive_action,
    stomp_state_to_checkpoint_payload,
)

OBS_DIM = 3
N_PRIMITIVE = 2
N_OPTIONS = 2


@pytest.mark.unit
class TestSubtaskSpecValidation:
    """SubtaskSpec must reject non-finite thresholds and degenerate scales."""

    def test_valid_spec_accepted(self) -> None:
        spec = SubtaskSpec(feature_index=0, threshold=0.5, pseudo_reward_scale=2.0)
        assert spec.threshold == 0.5
        assert spec.pseudo_reward_scale == 2.0

    @pytest.mark.parametrize("threshold", [float("nan"), float("inf"), 0.0, -1.0])
    def test_rejects_bad_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            SubtaskSpec(feature_index=0, threshold=threshold)

    @pytest.mark.parametrize("scale", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
    def test_rejects_bad_pseudo_reward_scale(self, scale: float) -> None:
        with pytest.raises(ValueError, match="pseudo_reward_scale"):
            SubtaskSpec(feature_index=0, pseudo_reward_scale=scale)


def _agent() -> STOMPAgent:
    return STOMPAgent(
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0),
                SubtaskSpec(feature_index=1),
            ),
            observation_dim=OBS_DIM,
            n_primitive_actions=N_PRIMITIVE,
        )
    )


def _state() -> STOMPState:
    agent = _agent()
    state = agent.init(jr.key(0))
    return agent.start(state, jnp.array([0.5, -0.25, 1.0], dtype=jnp.float32))


def _old_format_payload(state: STOMPState) -> dict:
    """Synthesize a pre-expansion checkpoint payload from a current state."""
    payload = stomp_state_to_checkpoint_payload(state)
    for name in STOMP_STATE_EXPANSION_FIELDS:
        del payload[name]
    for name in STOMP_STATE_LIFETIME_FIELDS:
        del payload[name]
    for name in STOMP_OPTION_MODEL_EXPANSION_FIELDS:
        del payload["option_models"][name]
    return payload


class TestStompCheckpointRoundTrip:
    """Current-format payloads must round-trip losslessly."""

    def test_new_format_round_trips(self) -> None:
        state = _state()

        restored = load_stomp_state_with_migration(stomp_state_to_checkpoint_payload(state))

        chex.assert_trees_all_equal(restored, state)

    def test_migrated_state_round_trips(self) -> None:
        state = _state()
        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        restored = load_stomp_state_with_migration(stomp_state_to_checkpoint_payload(migrated))

        chex.assert_trees_all_equal(restored, migrated)


class TestStompStateMigration:
    """Pre-expansion payloads gain zero-filled expansion fields."""

    def test_old_format_fills_option_model_zeros(self) -> None:
        state = _state()

        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        for name in STOMP_OPTION_MODEL_EXPANSION_FIELDS:
            value = getattr(migrated.option_models, name)
            chex.assert_shape(value, (N_OPTIONS,))
            assert value.dtype == jnp.float32
            assert bool(jnp.all(value == 0.0))

    def test_old_format_fills_scalar_accumulator_zeros(self) -> None:
        state = _state()

        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        for name in STOMP_STATE_EXPANSION_FIELDS:
            value = getattr(migrated, name)
            chex.assert_shape(value, ())
            assert value.dtype == jnp.float32
            assert float(value) == 0.0

    def test_old_format_preserves_shared_fields(self) -> None:
        state = _state()

        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        chex.assert_trees_all_equal(migrated.base_learner_state, state.base_learner_state)
        chex.assert_trees_all_equal(migrated.option_policies, state.option_policies)
        chex.assert_trees_all_equal(
            migrated.option_models.cumreward_ema, state.option_models.cumreward_ema
        )
        chex.assert_trees_all_equal(
            migrated.option_models.discount_ema, state.option_models.discount_ema
        )
        chex.assert_trees_all_equal(migrated.step_count, state.step_count)
        chex.assert_trees_all_equal(migrated.executing_option, state.executing_option)

    def test_migrated_state_matches_fresh_init_priors(self) -> None:
        """Filled defaults equal a fresh agent's expansion-field priors."""
        agent = _agent()
        fresh = agent.init(jr.key(1))
        migrated = load_stomp_state_with_migration(_old_format_payload(_state()))

        for name in STOMP_OPTION_MODEL_EXPANSION_FIELDS:
            chex.assert_trees_all_equal(
                getattr(migrated.option_models, name),
                getattr(fresh.option_models, name),
            )


class TestStompMigrationFailsClosed:
    """Anything other than the two known templates must be rejected."""

    def test_missing_required_state_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        del payload["base_average_reward"]

        with pytest.raises(ValueError, match="base_average_reward"):
            load_stomp_state_with_migration(payload)

    def test_missing_required_option_model_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        del payload["option_models"]["cumreward_ema"]

        with pytest.raises(ValueError, match="cumreward_ema"):
            load_stomp_state_with_migration(payload)

    def test_unknown_state_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        payload["not_a_field"] = jnp.array(0.0, dtype=jnp.float32)

        with pytest.raises(ValueError, match="not_a_field"):
            load_stomp_state_with_migration(payload)

    def test_unknown_option_model_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        payload["option_models"]["not_a_field"] = jnp.zeros(N_OPTIONS, dtype=jnp.float32)

        with pytest.raises(ValueError, match="not_a_field"):
            load_stomp_state_with_migration(payload)

    def test_incomplete_option_policies_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        del payload["option_policies"]["traces"]

        with pytest.raises(ValueError, match="option-policy"):
            load_stomp_state_with_migration(payload)

    def test_non_mapping_option_models_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        payload["option_models"] = 3.0

        with pytest.raises(ValueError, match="option_models"):
            load_stomp_state_with_migration(payload)


def test_primitive_only_stomp_has_typed_empty_option_state_and_base_only_updates() -> None:
    specs = STOMPSpecArrays.from_specs([])
    chex.assert_shape(specs.feature_indices, (0,))
    chex.assert_shape(specs.thresholds, (0,))
    chex.assert_shape(specs.pseudo_reward_scales, (0,))
    chex.assert_shape(specs.max_option_steps, (0,))
    assert specs.feature_indices.dtype == jnp.int32
    assert specs.thresholds.dtype == jnp.float32
    assert specs.pseudo_reward_scales.dtype == jnp.float32
    assert specs.max_option_steps.dtype == jnp.int32
    assert specs.to_list() == []

    config = STOMPConfig(
        subtask_specs=(),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        epsilon_base=0.0,
        option_planning_backups_per_step=0,
    )
    agent = STOMPAgent(config)
    state = agent.init(jr.key(44))
    chex.assert_shape(state.option_policies.q_weights, (0, N_PRIMITIVE, OBS_DIM))
    chex.assert_shape(state.option_models.next_state_weights, (0, OBS_DIM, OBS_DIM))
    assert bool(agent.state_valid(state))

    observation = jnp.asarray((0.25, -0.5, 1.0), dtype=jnp.float32)
    started = agent.start(state, observation)
    assert int(started.executing_option) == -1
    assert int(started.base_last_action) == int(started.last_primitive_action)
    assert 0 <= int(started.last_primitive_action) < N_PRIMITIVE
    assert bool(agent.state_valid(started))
    masked_start = agent.start_with_extended_action_mask(
        state,
        observation,
        jnp.ones((N_PRIMITIVE,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(masked_start.state, started)
    chex.assert_trees_all_equal(masked_start.primitive_action, started.last_primitive_action)

    restored = load_stomp_state_with_migration(stomp_state_to_checkpoint_payload(started))
    chex.assert_trees_all_equal(restored, started)

    result = agent.update(
        started,
        jnp.asarray(0.75, dtype=jnp.float32),
        jnp.asarray((-0.1, 0.4, 0.8), dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.executing_option) == -1
    assert not bool(result.option_terminated)
    assert float(result.pseudo_reward) == 0.0
    assert float(result.option_importance_ratio) == 0.0
    assert int(result.planning_backups) == 0
    assert int(result.nested_updates_required) == 1
    assert int(result.nested_updates_applied) == 1
    assert bool(agent.state_valid(result.state))

    replacement = replace_dispatched_primitive_action(
        result.state,
        result.state.base_last_obs,
        jnp.asarray(1 - int(result.primitive_action), dtype=jnp.int32),
    )
    assert bool(replacement.decision.applied)
    assert int(replacement.decision.owner) == DISPATCH_OWNER_BASE_PRIMITIVE
    assert int(replacement.state.executing_option) == -1

    with pytest.raises(ValueError, match="primitive-only STOMP"):
        STOMPConfig(
            subtask_specs=(),
            observation_dim=OBS_DIM,
            n_primitive_actions=N_PRIMITIVE,
            option_planning_backups_per_step=1,
        )


def test_differential_q_infinite_reward_does_not_poison_weights() -> None:
    """Inf reward is 0*inf = NaN on silent features and unused actions."""
    n_actions, dim = 2, 2
    q = jnp.zeros((n_actions, dim), dtype=jnp.float32)
    z = jnp.zeros((n_actions, dim), dtype=jnp.float32)
    rbar = jnp.array(0.0, dtype=jnp.float32)
    obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    action = jnp.int32(0)
    nxt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    kw = dict(step_size=0.1, avg_reward_step_size=0.01, trace_decay=0.0, n_actions=n_actions)

    new_q, new_z, new_rbar, td_error, update_applied = _differential_q_update(
        q, z, rbar, obs, action, jnp.array(jnp.inf, dtype=jnp.float32), nxt, **kw
    )
    chex.assert_trees_all_close(new_q, q)
    chex.assert_trees_all_close(new_z, z)
    chex.assert_trees_all_close(new_rbar, rbar)
    assert float(td_error) == 0.0
    assert not bool(update_applied)

    recovered_q, _, recovered_rbar, _, recovered_applied = _differential_q_update(
        new_q, new_z, new_rbar, obs, action, jnp.array(1.0, dtype=jnp.float32), nxt, **kw
    )
    chex.assert_tree_all_finite(recovered_q)
    chex.assert_tree_all_finite(recovered_rbar)
    assert bool(recovered_applied)


def test_semidp_q_infinite_reward_does_not_poison_weights() -> None:
    n_actions, dim = 2, 2
    q = jnp.zeros((n_actions, dim), dtype=jnp.float32)
    z = jnp.zeros((n_actions, dim), dtype=jnp.float32)
    rbar = jnp.array(0.0, dtype=jnp.float32)
    obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    action = jnp.int32(0)
    nxt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    kw = dict(
        step_size=0.1,
        avg_reward_step_size=0.01,
        trace_decay=0.0,
        n_actions=n_actions,
        baseline_mass=jnp.array(1.0, dtype=jnp.float32),
        discount=jnp.array(1.0, dtype=jnp.float32),
    )

    new_q, new_z, new_rbar, td_error, update_applied = _differential_semidp_q_update(
        q, z, rbar, obs, action, jnp.array(jnp.inf, dtype=jnp.float32), nxt, **kw
    )
    chex.assert_trees_all_close(new_q, q)
    chex.assert_trees_all_close(new_z, z)
    chex.assert_trees_all_close(new_rbar, rbar)
    assert float(td_error) == 0.0
    assert not bool(update_applied)


@pytest.mark.unit
class TestStompConfigScalarValidation:
    """STOMPConfig must reject non-finite/out-of-range scalar hyperparameters (#523)."""

    _STEP_SIZE_FIELDS = (
        "base_step_size",
        "base_avg_reward_step_size",
        "option_step_size",
        "option_avg_reward_step_size",
        "option_model_step_size",
    )
    _UNIT_INTERVAL_FIELDS = (
        "base_trace_decay",
        "option_trace_decay",
        "option_model_decay",
        "epsilon_base",
        "epsilon_option",
    )

    @staticmethod
    def _build(**overrides: float) -> STOMPConfig:
        return STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=OBS_DIM,
            n_primitive_actions=N_PRIMITIVE,
            **overrides,
        )

    @pytest.mark.parametrize("name", _STEP_SIZE_FIELDS)
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.05])
    def test_rejects_invalid_step_sizes(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=name):
            self._build(**{name: value})

    @pytest.mark.parametrize("name", _UNIT_INTERVAL_FIELDS)
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.5, 2.0])
    def test_rejects_invalid_unit_interval_scalars(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=name):
            self._build(**{name: value})

    @pytest.mark.parametrize("name", _STEP_SIZE_FIELDS)
    @pytest.mark.parametrize("value", [0.0, 0.5])
    def test_accepts_valid_step_sizes(self, name: str, value: float) -> None:
        """Zero step size stays accepted: it is the supported learning-freeze boundary."""
        config = self._build(**{name: value})
        assert getattr(config, name) == value

    @pytest.mark.parametrize("name", _UNIT_INTERVAL_FIELDS)
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_unit_interval_boundaries(self, name: str, value: float) -> None:
        config = self._build(**{name: value})
        assert getattr(config, name) == value

    def test_from_config_enforces_scalar_validation(self) -> None:
        payload = self._build().to_config()
        payload["base_step_size"] = float("nan")
        with pytest.raises(ValueError, match="base_step_size"):
            STOMPConfig.from_config(payload)


def test_subtask_spec_scalar_validation() -> None:
    with pytest.raises(ValueError, match="feature_index"):
        SubtaskSpec(feature_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_index"):
        SubtaskSpec(feature_index=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_option_steps"):
        SubtaskSpec(feature_index=0, max_option_steps=True)  # type: ignore[arg-type]

    spec = SubtaskSpec(
        feature_index=np.int32(1),
        max_option_steps=np.int64(10),
    )
    assert type(spec.feature_index) is int
    assert type(spec.max_option_steps) is int
    assert spec.feature_index == 1
    assert spec.max_option_steps == 10


def test_stomp_config_integer_validation() -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        STOMPConfig(observation_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_primitive_actions"):
        STOMPConfig(n_primitive_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_hidden_sizes"):
        STOMPConfig(base_hidden_sizes=(True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="option_planning_backups_per_step"):
        STOMPConfig(option_planning_backups_per_step=True)  # type: ignore[arg-type]

    cfg = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=np.int32(4),
        n_primitive_actions=np.int64(2),
        base_hidden_sizes=(np.int32(16), np.int64(8)),
        option_planning_backups_per_step=np.uint16(2),
    )
    assert type(cfg.observation_dim) is int
    assert type(cfg.n_primitive_actions) is int
    assert type(cfg.base_hidden_sizes[0]) is int
    assert type(cfg.base_hidden_sizes[1]) is int
    assert type(cfg.option_planning_backups_per_step) is int
    assert cfg.observation_dim == 4
    assert cfg.n_primitive_actions == 2
    assert cfg.base_hidden_sizes == (16, 8)
    assert cfg.option_planning_backups_per_step == 2


def test_stomp_closes_float32_schema_and_direct_resource_boundaries() -> None:
    spec = SubtaskSpec(
        feature_index=0,
        threshold=np.float64(0.5),
        pseudo_reward_scale=np.float32(1.0),
    )
    assert type(spec.threshold) is float
    assert type(spec.pseudo_reward_scale) is float
    with pytest.raises(ValueError, match="threshold"):
        SubtaskSpec(feature_index=0, threshold=1.0e100)

    last_legal_observation_dim = (2**29 - 1 - 22) // 4
    STOMPConfig(observation_dim=last_legal_observation_dim, n_primitive_actions=1)
    with pytest.raises(ValueError, match="direct array bytes"):
        STOMPConfig(observation_dim=last_legal_observation_dim + 1, n_primitive_actions=1)

    measured_config = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0), SubtaskSpec(feature_index=1)),
        observation_dim=3,
        n_primitive_actions=2,
        base_hidden_sizes=(4,),
    )
    measured_agent = STOMPAgent(measured_config)
    array_bytes = sum(
        int(leaf.nbytes)
        for leaf in jax.tree_util.tree_leaves(
            (measured_agent.init(jr.key(0)), measured_agent.spec_arrays)
        )
        if hasattr(leaf, "nbytes")
    )
    array_bytes += sum(
        int(getattr(measured_agent.spec_arrays, field.name).nbytes)
        for field in dataclasses.fields(STOMPSpecArrays)
    )
    assert 4 * _stomp_direct_array_scalars(measured_config) == array_bytes


@pytest.mark.unit
def test_stomp_from_config_preserves_mapping_partial_and_tuple_compatibility() -> None:
    class MappingSpoof:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            return dict

        def __iter__(self) -> object:
            raise AssertionError("iteration hook executed")

        def __repr__(self) -> str:
            raise AssertionError("repr hook executed")

    config = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=2,
    )
    payload = config.to_config()
    payload["subtask_specs"] = tuple(payload["subtask_specs"])
    payload["base_hidden_sizes"] = tuple(payload["base_hidden_sizes"])
    payload["type"] = "historical-marker"
    assert STOMPConfig.from_config(MappingProxyType(payload)) == config

    partial = STOMPConfig.from_config(
        {"subtask_specs": ({"feature_index": 0},), "observation_dim": 2}
    )
    assert partial.subtask_specs == (SubtaskSpec(feature_index=0),)
    assert partial.option_planning_backups_per_step == 0

    with pytest.raises(ValueError, match="mapping"):
        STOMPConfig.from_config(MappingSpoof())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="STOMPConfig"):
        STOMPConfig.from_config({"unexpected_field": 1})

    for field, value in (
        ("subtask_specs", "not-a-sequence"),
        ("base_hidden_sizes", "not-a-sequence"),
    ):
        invalid = config.to_config()
        invalid[field] = value
        with pytest.raises(ValueError, match=field):
            STOMPConfig.from_config(invalid)
