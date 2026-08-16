# mypy: disable-error-code="attr-defined,call-arg"
"""Exact long-horizon transactions for Intelligence Amplification."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.intelligence_amplification import (
    EXO_CEREBELLUM_LIFETIME_COUNTER_DELTA_NBYTES,
    EXO_CEREBELLUM_LIFETIME_COUNTER_NBYTES,
    IA_LIFETIME_COUNTER_DELTA_NBYTES,
    IA_LIFETIME_COUNTER_NBYTES,
    RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_DELTA_NBYTES,
    RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_NBYTES,
    ExoCerebellumAgent,
    ExoCerebellumConfig,
    ExoCerebellumState,
    IAAgent,
    IAConfig,
    IAState,
    RecommendationProtocolConfig,
    RecommendationProtocolState,
    exo_cerebellum_lifetime_counter_nbytes,
    ia_lifetime_counter_nbytes,
    init_recommendation_protocol_state,
    measure_exo_cerebellum_state_nbytes,
    measure_ia_state_nbytes,
    measure_ia_wrapper_state_nbytes,
    measure_recommendation_protocol_state_nbytes,
    migrate_legacy_exo_cerebellum_state,
    migrate_legacy_ia_state,
    migrate_legacy_recommendation_protocol_state,
    recommendation_protocol_lifetime_counter_nbytes,
    recommendation_protocol_state_is_valid,
    update_recommendation_protocol,
)
from alberta_framework.core.oak import OaKConfig, measure_oak_state_nbytes
from alberta_framework.core.options import STOMPConfig, SubtaskSpec

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBS = jnp.asarray((1.0, 0.0), dtype=jnp.float32)


def _config() -> IAConfig:
    return IAConfig(
        cerebellum=ExoCerebellumConfig(
            n_demons=2,
            obs_dim=2,
            step_size=0.05,
        ),
        cortex=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=1.0e6,
                        max_option_steps=16,
                    ),
                ),
                observation_dim=2,
                n_primitive_actions=2,
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        ),
    )


def _telemetry(words: tuple[int, int]) -> int:
    high, low = words
    return low if high == 0 and low <= _INT32_MAX else _INT32_MAX


def _started(agent: IAAgent) -> IAState:
    return agent.start(agent.init(jr.key(7)), _OBS)


def _with_primitive_clock(
    state: IAState,
    words: tuple[int, int],
) -> IAState:
    exact = jnp.asarray(words, dtype=jnp.uint32)
    telemetry = jnp.asarray(_telemetry(words), dtype=jnp.int32)
    stomp = state.cortex_state.stomp_state.replace(
        step_count=telemetry,
        step_words=exact,
    )
    cortex = state.cortex_state.replace(
        stomp_state=stomp,
        step_count=telemetry,
        step_words=exact,
    )
    cerebellum = state.cerebellum_state.replace(
        step_count=telemetry,
        step_words=exact,
    )
    return cast(
        IAState,
        state.replace(
            cerebellum_state=cerebellum,
            cortex_state=cortex,
            step_count=telemetry,
            step_words=exact,
        ),
    )


def _update(agent: IAAgent, state: IAState) -> Any:
    return agent.update(
        state,
        _OBS,
        jnp.asarray(0.25, dtype=jnp.float32),
        _OBS,
        partner_action=jnp.asarray(0, dtype=jnp.int32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
    )


def _without_host_timing(state: IAState) -> IAState:
    """Normalize nested MultiHead host metadata for JIT state comparisons."""

    learner = state.cortex_state.stomp_state.base_learner_state.replace(
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    stomp = state.cortex_state.stomp_state.replace(base_learner_state=learner)
    cortex = state.cortex_state.replace(stomp_state=stomp)
    return cast(IAState, state.replace(cortex_state=cortex))


def _protocol_state(
    *,
    accepted: tuple[int, int],
    rejected: tuple[int, int],
    step: tuple[int, int],
    ema: float = 0.5,
) -> RecommendationProtocolState:
    return RecommendationProtocolState(
        accepted_count=jnp.asarray(_telemetry(accepted), dtype=jnp.int32),
        rejected_count=jnp.asarray(_telemetry(rejected), dtype=jnp.int32),
        acceptance_ema=jnp.asarray(ema, dtype=jnp.float32),
        step_count=jnp.asarray(_telemetry(step), dtype=jnp.int32),
        accepted_words=jnp.asarray(accepted, dtype=jnp.uint32),
        rejected_words=jnp.asarray(rejected, dtype=jnp.uint32),
        step_words=jnp.asarray(step, dtype=jnp.uint32),
    )


def test_cerebellum_eager_jit_and_scan_cross_low_word_carry() -> None:
    agent = ExoCerebellumAgent(ExoCerebellumConfig(n_demons=2, obs_dim=2))
    source = cast(
        ExoCerebellumState,
        agent.init().replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=jnp.asarray((5, _UINT32_MAX), dtype=jnp.uint32),
        ),
    )

    with jax.disable_jit():
        eager = agent.update_result(source, _OBS, _OBS)
    compiled = jax.jit(agent.update_result)(source, _OBS, _OBS)

    def scan_step(
        state: ExoCerebellumState,
        _: jax.Array,
    ) -> tuple[ExoCerebellumState, jax.Array]:
        result = agent.update_result(state, _OBS, _OBS)
        return result.state, result.post_step_words

    scanned, trace = jax.lax.scan(
        scan_step,
        source,
        jnp.zeros((1,), dtype=jnp.int32),
    )
    expected = jnp.asarray((6, 0), dtype=jnp.uint32)
    assert bool(eager.update_applied)
    assert bool(compiled.update_applied)
    chex.assert_trees_all_equal(eager.state.step_words, expected)
    chex.assert_trees_all_equal(compiled.state.step_words, expected)
    chex.assert_trees_all_equal(scanned.step_words, expected)
    chex.assert_trees_all_equal(trace[0], expected)
    assert int(eager.state.step_count) == _INT32_MAX


def test_ia_eager_jit_and_scan_cross_low_word_carry() -> None:
    agent = IAAgent(_config())
    source = _with_primitive_clock(_started(agent), (7, _UINT32_MAX - 1))

    with jax.disable_jit():
        eager = _update(agent, source)
    compiled = jax.jit(_update, static_argnums=(0,))(agent, source)
    scanned = agent.scan(
        source,
        jnp.stack((_OBS, _OBS)),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.stack((_OBS, _OBS)),
        partner_actions=jnp.zeros((2,), dtype=jnp.int32),
        discounts=jnp.ones((2,), dtype=jnp.float32),
    )

    first = jnp.asarray((7, _UINT32_MAX), dtype=jnp.uint32)
    final = jnp.asarray((8, 0), dtype=jnp.uint32)
    assert bool(eager.update_applied)
    assert bool(compiled.update_applied)
    chex.assert_trees_all_equal(eager.state.step_words, first)
    chex.assert_trees_all_equal(compiled.state.step_words, first)
    chex.assert_trees_all_equal(scanned.post_step_words[0], first)
    chex.assert_trees_all_equal(scanned.state.step_words, final)
    assert bool(jnp.all(scanned.updates_applied))
    chex.assert_trees_all_equal(
        scanned.state.step_words,
        scanned.state.cerebellum_state.step_words,
    )
    chex.assert_trees_all_equal(
        scanned.state.step_words,
        scanned.state.cortex_state.step_words,
    )
    chex.assert_trees_all_equal(
        scanned.state.step_words,
        scanned.state.cortex_state.stomp_state.step_words,
    )


def test_ia_telemetry_saturates_while_exact_identity_advances() -> None:
    agent = IAAgent(_config())
    source = _with_primitive_clock(_started(agent), (0, _INT32_MAX))

    result = _update(agent, source)

    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.cerebellum_state.step_count) == _INT32_MAX
    assert int(result.state.cortex_state.step_count) == _INT32_MAX
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((0, _INT32_MAX + 1), dtype=jnp.uint32),
    )


def test_ia_start_consumes_no_primitive_identity_and_is_atomic() -> None:
    agent = IAAgent(_config())
    source = agent.init(jr.key(11))

    started = agent.start(source, _OBS)
    rejected = agent.start(source, jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32))

    assert bool(agent.state_is_valid(started))
    chex.assert_trees_all_equal(started.step_words, jnp.zeros((2,), dtype=jnp.uint32))
    chex.assert_trees_all_equal(
        started.step_words,
        started.cerebellum_state.step_words,
    )
    chex.assert_trees_all_equal(started.step_words, started.cortex_state.step_words)
    chex.assert_trees_all_equal(rejected, source)


def test_ia_impossible_child_history_and_nonfinite_input_roll_back() -> None:
    agent = IAAgent(_config())
    valid = _with_primitive_clock(_started(agent), (0, 5))
    misaligned = cast(
        IAState,
        valid.replace(
            cerebellum_state=valid.cerebellum_state.replace(
                step_count=jnp.asarray(4, dtype=jnp.int32),
                step_words=jnp.asarray((0, 4), dtype=jnp.uint32),
            )
        ),
    )

    impossible = _update(agent, misaligned)
    nonfinite = agent.update(valid, _OBS, jnp.asarray(jnp.nan), _OBS)

    assert not bool(impossible.source_state_valid)
    assert not bool(impossible.child_clocks_aligned)
    assert not bool(impossible.update_applied)
    chex.assert_trees_all_equal(impossible.state, misaligned)
    assert not bool(nonfinite.input_valid)
    assert not bool(nonfinite.update_applied)
    chex.assert_trees_all_equal(nonfinite.state, valid)


def test_ia_invalid_traced_action_rolls_back_bit_exactly() -> None:
    agent = IAAgent(_config())
    source = _started(agent)
    compiled = jax.jit(
        lambda action: agent.update(
            source,
            _OBS,
            jnp.asarray(0.0, dtype=jnp.float32),
            _OBS,
            partner_action=action,
        )
    )

    result = compiled(jnp.asarray(-1, dtype=jnp.int32))

    assert not bool(result.input_valid)
    assert not bool(result.cortex_update_applied)
    assert not bool(result.update_applied)
    assert not bool(jnp.isfinite(result.cortex_td_error))
    chex.assert_trees_all_equal(
        _without_host_timing(result.state),
        _without_host_timing(source),
    )


def test_all_ones_cerebellum_ia_and_protocol_are_terminal() -> None:
    cerebellum = ExoCerebellumAgent(ExoCerebellumConfig(n_demons=2, obs_dim=2))
    cerebellum_terminal = cast(
        ExoCerebellumState,
        cerebellum.init().replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
        ),
    )
    cerebellum_result = cerebellum.update_result(cerebellum_terminal, _OBS, _OBS)
    chex.assert_trees_all_equal(cerebellum_result.state, cerebellum_terminal)
    assert not bool(cerebellum_result.lifetime_capacity_available)
    assert not bool(cerebellum_result.update_applied)

    ia = IAAgent(_config())
    ia_terminal = _with_primitive_clock(
        _started(ia),
        (_UINT32_MAX, _UINT32_MAX),
    )
    ia_result = _update(ia, ia_terminal)
    chex.assert_trees_all_equal(ia_result.state, ia_terminal)
    assert not bool(ia_result.lifetime_capacity_available)
    assert not bool(ia_result.update_applied)

    protocol_terminal = _protocol_state(
        accepted=(_UINT32_MAX, _UINT32_MAX),
        rejected=(0, 0),
        step=(_UINT32_MAX, _UINT32_MAX),
    )
    protocol_result = update_recommendation_protocol(
        RecommendationProtocolConfig(),
        protocol_terminal,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(recommendation_protocol_state_is_valid(protocol_terminal))
    assert not bool(protocol_result.lifetime_capacity_available)
    assert not bool(protocol_result.update_applied)
    chex.assert_trees_all_equal(protocol_result.state, protocol_terminal)


def test_protocol_exact_partition_crosses_int32_and_low_word_boundaries() -> None:
    config = RecommendationProtocolConfig(acceptance_ema_decay=0.5)
    saturated = _protocol_state(
        accepted=(0, _INT32_MAX),
        rejected=(0, 0),
        step=(0, _INT32_MAX),
    )
    after_rejection = update_recommendation_protocol(
        config,
        saturated,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(after_rejection.update_applied)
    assert int(after_rejection.state.step_count) == _INT32_MAX
    assert int(after_rejection.state.accepted_count) == _INT32_MAX
    assert int(after_rejection.state.rejected_count) == 1
    chex.assert_trees_all_equal(
        after_rejection.state.step_words,
        jnp.asarray((0, _INT32_MAX + 1), dtype=jnp.uint32),
    )

    carry_source = _protocol_state(
        accepted=(2, _UINT32_MAX - 1),
        rejected=(0, 0),
        step=(2, _UINT32_MAX - 1),
    )

    def scan_step(
        state: RecommendationProtocolState,
        _: jax.Array,
    ) -> tuple[RecommendationProtocolState, jax.Array]:
        result = update_recommendation_protocol(
            config,
            state,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        return result.state, result.update_applied

    scanned, applied = jax.lax.scan(
        scan_step,
        carry_source,
        jnp.zeros((2,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        scanned.step_words,
        jnp.asarray((3, 0), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(scanned.accepted_words, scanned.step_words)
    assert bool(jnp.all(applied))
    assert bool(recommendation_protocol_state_is_valid(scanned))


def test_protocol_impossible_partition_rolls_back_under_jit() -> None:
    impossible = _protocol_state(
        accepted=(0, 2),
        rejected=(0, 2),
        step=(0, 5),
    )
    compiled = jax.jit(update_recommendation_protocol, static_argnums=(0,))

    result = compiled(
        RecommendationProtocolConfig(),
        impossible,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )

    assert not bool(result.source_state_valid)
    assert not bool(result.exact_partition_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, impossible)


def test_migrations_are_strict_and_reject_ambiguous_legacy_counters() -> None:
    config = _config()
    agent = IAAgent(config)
    current = _update(agent, _started(agent)).state
    legacy_cerebellum = {
        "weights": current.cerebellum_state.weights,
        "step_count": current.cerebellum_state.step_count,
    }
    migrated_cerebellum = migrate_legacy_exo_cerebellum_state(
        legacy_cerebellum,
        config=config.cerebellum,
    )
    migrated_ia = migrate_legacy_ia_state(
        {
            "cerebellum_state": legacy_cerebellum,
            "cortex_state": current.cortex_state,
            "step_count": current.step_count,
        },
        config=config,
    )
    migrated_protocol = migrate_legacy_recommendation_protocol_state(
        {
            "accepted_count": jnp.asarray(2, dtype=jnp.int32),
            "rejected_count": jnp.asarray(3, dtype=jnp.int32),
            "acceptance_ema": jnp.asarray(0.4, dtype=jnp.float32),
            "step_count": jnp.asarray(5, dtype=jnp.int32),
        }
    )

    chex.assert_trees_all_equal(
        migrated_cerebellum.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert bool(agent.state_is_valid(migrated_ia))
    assert bool(recommendation_protocol_state_is_valid(migrated_protocol))
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_exo_cerebellum_state(
            {
                "weights": current.cerebellum_state.weights,
                "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            },
            config=config.cerebellum,
        )
    with pytest.raises(ValueError, match="partition"):
        migrate_legacy_recommendation_protocol_state(
            {
                "accepted_count": jnp.asarray(1, dtype=jnp.int32),
                "rejected_count": jnp.asarray(1, dtype=jnp.int32),
                "acceptance_ema": jnp.asarray(0.5, dtype=jnp.float32),
                "step_count": jnp.asarray(3, dtype=jnp.int32),
            }
        )


def test_resource_accounting_declares_exact_byte_deltas() -> None:
    agent = IAAgent(_config())
    state = agent.init(jr.key(13))
    protocol = init_recommendation_protocol_state()
    cerebellum_bytes = measure_exo_cerebellum_state_nbytes(state.cerebellum_state)
    wrapper_bytes = measure_ia_wrapper_state_nbytes(state)

    assert EXO_CEREBELLUM_LIFETIME_COUNTER_NBYTES == 12
    assert EXO_CEREBELLUM_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert IA_LIFETIME_COUNTER_NBYTES == 60
    assert IA_LIFETIME_COUNTER_DELTA_NBYTES == 16
    assert RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_NBYTES == 36
    assert RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_DELTA_NBYTES == 24
    assert exo_cerebellum_lifetime_counter_nbytes() == 12
    assert ia_lifetime_counter_nbytes() == 60
    assert recommendation_protocol_lifetime_counter_nbytes() == 36
    assert cerebellum_bytes == int(state.cerebellum_state.weights.nbytes) + 12
    assert wrapper_bytes == cerebellum_bytes + 12
    assert measure_ia_state_nbytes(state) == (
        wrapper_bytes + measure_oak_state_nbytes(state.cortex_state)
    )
    assert measure_recommendation_protocol_state_nbytes(protocol) == 40
    assert int(protocol.accepted_words.nbytes) * 3 == (
        RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_DELTA_NBYTES
    )


def test_malformed_shapes_and_dtypes_fail_before_mutation() -> None:
    cerebellum = ExoCerebellumAgent(ExoCerebellumConfig(n_demons=2, obs_dim=2))
    source = cerebellum.init()
    with pytest.raises(ValueError, match="observation must have shape"):
        cerebellum.update_result(source, jnp.zeros((1,), dtype=jnp.float32), _OBS)
    with pytest.raises(TypeError, match="integer dtypes"):
        update_recommendation_protocol(
            RecommendationProtocolConfig(),
            init_recommendation_protocol_state(),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        )
    np.testing.assert_array_equal(
        np.asarray(source.step_words),
        np.zeros((2,), dtype=np.uint32),
    )


@pytest.mark.parametrize(
    "step_size",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        1.0e100,
        1.0e-100,
        True,
        np.bool_(True),
        np.asarray(0.05),
    ],
)
def test_cerebellum_config_rejects_values_outside_its_float32_sink(
    step_size: object,
) -> None:
    with pytest.raises(ValueError, match="step_size"):
        ExoCerebellumConfig(step_size=step_size)


def test_cerebellum_config_rounds_exact_reals_once_and_updates_with_that_sink() -> None:
    midpoint_above = Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**60)
    expected = float(np.nextafter(np.float32(1.0), np.float32(np.inf)))

    config = ExoCerebellumConfig(n_demons=1, obs_dim=1, step_size=midpoint_above)
    assert type(config.step_size) is float
    assert config.step_size == expected
    assert config.to_config()["step_size"] == expected

    agent = ExoCerebellumAgent(config)
    result = agent.update_result(
        agent.init(),
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert float(result.state.weights[0, 0]) == expected


def test_cerebellum_config_preserves_valid_builtin_float_serialization() -> None:
    config = ExoCerebellumConfig(step_size=0.05)
    assert type(config.step_size) is float
    assert config.step_size == 0.05
    assert config.to_config()["step_size"] == 0.05
