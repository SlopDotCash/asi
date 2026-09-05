"""Tests for the observation-causal Forager cognitive-map variant.

The agent under test (:mod:`alberta_framework.benchmarks.causal_map_forager`)
learns a compact world model online from egocentric field-of-view
observations only — a relative toroidal map, per-channel reward statistics,
and respawn-schedule estimates — with no privileged environment state.  The
suite covers the map/estimator mechanisms (observation integration, interval
and sample merging, saturating counters), the cost-aware routing and safety
grids, action selection and retry/respawn timing, RNG contracts, state
schema round-trips and validation, and exact-parity runs against the shared
Forager host runner on tiny fake environments.

Tests are deliberately white-box: they import ~15 private ``_helpers`` from
the module under test to pin numeric behavior at the mechanism level (the
public API alone cannot distinguish, e.g., interval-merge edge cases), so
renaming module internals is expected to require touching this file.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import heapq
import inspect
import json
import math
import pickle
import textwrap
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any, Final, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.causal_map_forager as causal_map_module
from alberta_framework.benchmarks.causal_map_forager import (
    _DIRECTION_STEPS,
    CAUSAL_MAP_STATE_SCHEMA,
    CAUSAL_MAP_VARIANT_KIND,
    CausalMapForagerAgent,
    CausalMapForagerConfig,
    _choose_action,
    _cost_aware_route_grid,
    _empty_state,
    _estimated_respawn_delay,
    _integrate_observation,
    _merge_channel_interval_bounds,
    _merge_exact_channel_samples,
    _retry_delay,
    _saturating_add_int32,
    causal_map_rng_contract,
    causal_map_start,
    causal_map_state_from_dict,
    causal_map_state_to_dict,
    causal_map_step,
    causal_map_variant_spec,
    run_causal_map_forager_seeds,
    validate_causal_map_state,
)
from alberta_framework.benchmarks.forager import (
    ForagerBenchmarkConfig,
    ForagerEnvConfig,
    _run_forager_host,
    run_forager,
)

pytestmark = pytest.mark.integration


def _observation(
    *objects: tuple[int, int, int],
    aperture: int = 9,
    channels: int = 3,
) -> jax.Array:
    """Return one-hot image; object tuples are ``(dy, dx, channel)``."""
    image = jnp.zeros((aperture, aperture, channels), dtype=jnp.float32)
    center = aperture // 2
    for dy, dx, channel in objects:
        image = image.at[center + dy, center + dx, channel].set(1.0)
    return image


class _CausalFakeForagax:
    """Tiny valid-observation environment for exact runner comparisons."""

    default_params = None

    def reset(self, key: Any, params: Any) -> tuple[jax.Array, jax.Array]:
        del key, params
        return _observation(aperture=3, channels=2), jnp.asarray(0, jnp.int32)

    def step(
        self,
        key: Any,
        state: jax.Array,
        action: jax.Array,
        params: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Mapping[str, Any]]:
        del key, params
        reward = jnp.where(
            action == (state % 4),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(-0.25, dtype=jnp.float32),
        )
        next_state = state + 1
        return (
            _observation(aperture=3, channels=2),
            next_state,
            reward,
            jnp.asarray(False),
            {"biome_regret": jnp.abs(reward)},
        )


class _HostCausalMapAgent(CausalMapForagerAgent):
    """Subclass bypasses exact-type scan dispatch for host parity."""


class _InvalidResetCausalFakeForagax(_CausalFakeForagax):
    def reset(self, key: Any, params: Any) -> tuple[jax.Array, jax.Array]:
        observation, state = super().reset(key, params)
        return observation.at[0, 0, 0].set(jnp.nan), state


class _InvalidStepCausalFakeForagax(_CausalFakeForagax):
    def step(
        self,
        key: Any,
        state: jax.Array,
        action: jax.Array,
        params: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Mapping[str, Any]]:
        observation, next_state, reward, done, info = super().step(
            key,
            state,
            action,
            params,
        )
        return observation.at[0, 0, 0].set(jnp.nan), next_state, reward, done, info


class _RngSensitiveCausalFakeForagax(_CausalFakeForagax):
    def step(
        self,
        key: Any,
        state: jax.Array,
        action: jax.Array,
        params: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Mapping[str, Any]]:
        del action, params
        reward = jr.uniform(key, (), dtype=jnp.float32)
        return (
            _observation(aperture=3, channels=2),
            state + 1,
            reward,
            jnp.asarray(False),
            {"biome_regret": reward},
        )


def _fake_make(self: ForagerEnvConfig) -> tuple[_CausalFakeForagax, None]:
    del self
    return _CausalFakeForagax(), None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("world_shape", (0, 15), "positive"),
        ("optimistic_unknown_reward", math.nan, "finite"),
        ("initial_retry_delay", 0, "positive integer"),
        ("maximum_retry_exponent", -1, "non-negative integer"),
        ("maximum_exact_interval_width", 1, "must be zero"),
        ("exploration_probability", 1.1, r"\[0, 1\]"),
        ("arrival_aware_readiness", 1, "must be a boolean"),
        ("respawn_safety_quantile", 0.49, r"\[0.5, 1\)"),
        ("respawn_safety_factor", 0.0, "positive"),
        ("distance_cost", 1e100, "positive"),
        ("tie_break_scale", 1e-100, "positive"),
        ("reverse_action_penalty", -0.1, "non-negative"),
        ("one_hot_tolerance", math.inf, "positive"),
        ("one_hot_tolerance", 0.5, "strictly between"),
        (
            "one_hot_tolerance",
            math.nextafter(0.5, 0.0),
            "strictly between",
        ),
    ],
)
def test_config_rejects_invalid_values(field: str, value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CausalMapForagerConfig(**{field: value})


def test_config_rejects_invalid_shape_type_and_canonicalizes_numpy_scalars() -> None:
    with pytest.raises(ValueError, match="world_shape"):
        CausalMapForagerConfig(world_shape=15)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="world_shape"):
        CausalMapForagerConfig(world_shape=[15, 15])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at most 4096 total cells"):
        CausalMapForagerConfig(world_shape=(65, 65))
    assert CausalMapForagerConfig(world_shape=(64, 64)).world_shape == (64, 64)

    config = CausalMapForagerConfig(
        initial_retry_delay=np.int64(10),
        maximum_retry_exponent=np.int64(31),
        distance_cost=np.float32(0.125),
    )
    assert type(config.initial_retry_delay) is int
    assert type(config.maximum_retry_exponent) is int
    assert type(config.distance_cost) is float
    assert int(_retry_delay(jnp.asarray(31, dtype=jnp.int32), config)) > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_cost", float(np.finfo(np.float32).max) / 16.0),
        ("retry_penalty", float(np.finfo(np.float32).max) / 16.0),
        ("visit_penalty", float(np.finfo(np.float32).max) / 16.0),
        ("respawn_safety_factor", float(np.finfo(np.float32).max) / 16.0),
    ],
)
def test_config_rejects_finite_terms_whose_combined_scores_can_overflow(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="overflow float32 planner arithmetic"):
        CausalMapForagerConfig(**cast(Any, {field: value}))


def test_config_rejects_respawn_factor_below_float32_one() -> None:
    below_one = float(
        np.nextafter(np.float32(1.0), np.float32(0.0), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="at least 1.0 after float32 conversion"):
        CausalMapForagerConfig(respawn_safety_factor=below_one)


@pytest.mark.parametrize(
    "field",
    [
        "optimistic_unknown_reward",
        "negative_reward_threshold",
        "reward_observation_epsilon",
        "respawn_safety_quantile",
        "respawn_safety_factor",
        "distance_cost",
        "reverse_action_penalty",
        "visit_penalty",
        "retry_penalty",
        "tie_break_scale",
        "exploration_probability",
        "one_hot_tolerance",
    ],
)
def test_config_rejects_boolean_numeric_values(field: str) -> None:
    with pytest.raises(ValueError):
        CausalMapForagerConfig(**{field: True})


def test_config_round_trip_fingerprint_and_variant_spec() -> None:
    config = CausalMapForagerConfig(
        optimistic_unknown_reward=3.0,
        respawn_safety_quantile=0.9,
    )
    restored = CausalMapForagerConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.fingerprint() == config.fingerprint()
    spec = causal_map_variant_spec(config)
    assert spec["kind"] == CAUSAL_MAP_VARIANT_KIND
    assert spec["privileged"] is False
    assert spec["prng_impl"] == "threefry2x32"
    assert spec["jax_threefry_partitionable"] is bool(
        jax.config.jax_threefry_partitionable
    )
    rng_contract = causal_map_rng_contract()
    assert rng_contract["prng_impl"] == "threefry2x32"
    assert rng_contract["jax_threefry_partitionable"] is bool(
        jax.config.jax_threefry_partitionable
    )
    assert "impl=prng_impl" in rng_contract["root"]
    assert spec["config_sha256"] == config.fingerprint()
    assert config.to_dict()["arrival_aware_readiness"] is True
    assert spec["config"]["arrival_aware_readiness"] is True
    assert (
        CausalMapForagerConfig(arrival_aware_readiness=False).fingerprint()
        != config.fingerprint()
    )
    with pytest.raises(ValueError, match="unknown"):
        CausalMapForagerConfig.from_dict({**config.to_dict(), "object_delay": 300})
    with pytest.raises(ValueError, match="respawn_quantile_z"):
        CausalMapForagerConfig.from_dict(
            {**config.to_dict(), "respawn_quantile_z": 999.0}
        )
    with pytest.raises(ValueError, match="mapping"):
        CausalMapForagerConfig.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CausalMapForagerConfig"):
        causal_map_variant_spec({})  # type: ignore[arg-type]


def test_agent_rejects_falsey_or_wrong_config_instead_of_defaulting() -> None:
    for invalid in ({}, False, 0):
        with pytest.raises(TypeError, match="CausalMapForagerConfig"):
            CausalMapForagerAgent(invalid)  # type: ignore[arg-type]


def test_metadata_declares_arrival_and_exploration_scheduler_semantics() -> None:
    config = CausalMapForagerConfig(
        exploration_probability=0.25,
        arrival_aware_readiness=True,
    )
    world_model = CausalMapForagerAgent(config).metadata()["world_model"]
    assert world_model["arrival_aware_readiness"] is True
    assert "route_step_distance - 1" in world_model["arrival_readiness_semantics"]
    assert "never broadens" in world_model["exploration_probability_semantics"]
    assert "lexicographic" in world_model["negative_avoidance"]
    assert "never impassable" in world_model["negative_route_semantics"]

    decision_world_model = CausalMapForagerAgent(
        dataclasses.replace(config, arrival_aware_readiness=False)
    ).metadata()["world_model"]
    assert decision_world_model["arrival_aware_readiness"] is False
    assert decision_world_model["arrival_readiness_semantics"] == (
        "ready_step <= step_count"
    )


def test_start_infers_aperture_channels_and_builds_relative_map() -> None:
    config = CausalMapForagerConfig()
    state, action = causal_map_start(
        _observation((1, 0, 2), (0, -2, 1)),
        config,
        7,
    )
    assert state.reward_sum.shape == (3,)
    assert int(state.cell_channel[1, 0]) == 2
    assert int(state.cell_channel[0, 13]) == 1
    assert tuple(np.asarray(state.position)) == (0, 0)
    assert int(action) == 0


def test_agent_rejects_non_one_hot_or_even_aperture() -> None:
    agent = CausalMapForagerAgent()
    with pytest.raises(ValueError, match="odd aperture of at least 3"):
        agent.start(jnp.zeros((4, 4, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="aperture 1 cannot causally identify"):
        agent.start(jnp.zeros((1, 1, 3), dtype=jnp.float32))
    non_binary = jnp.zeros((3, 3, 2), dtype=jnp.float32).at[0, 0, 0].set(0.5)
    with pytest.raises(ValueError, match="binary one-hot"):
        agent.start(non_binary)
    multi_hot = jnp.ones((3, 3, 2), dtype=jnp.float32)
    with pytest.raises(ValueError, match="one-hot"):
        agent.start(multi_hot)
    with pytest.raises(ValueError, match="uint32"):
        CausalMapForagerAgent(seed=2**32)
    agent = CausalMapForagerAgent()
    agent.start(_observation(channels=2))
    with pytest.raises(ValueError, match="channel count changed"):
        agent.step(0.0, _observation(channels=3))


def test_pure_start_and_step_reject_invalid_observation_values_eager_and_jit() -> None:
    config = CausalMapForagerConfig()
    valid = _observation(aperture=3, channels=2)
    state, _ = causal_map_start(valid, config, 0)
    invalid: tuple[jax.Array, ...] = (
        valid.at[0, 0, 0].set(jnp.nan),
        valid.at[0, 0, 0].set(0.5),
        valid.at[0, 0, 0].set(2.0),
        valid.at[0, 0, :].set(1.0),
    )
    compiled_start = jax.jit(lambda image: causal_map_start(image, config, 0))
    compiled_step = jax.jit(
        lambda current, image: causal_map_step(current, 0.0, image, config)
    )
    for image in invalid:
        with pytest.raises(Exception, match="observation must be finite"):
            jax.block_until_ready(  # type: ignore[no-untyped-call]
                causal_map_start(image, config, 0)
            )
        with pytest.raises(Exception, match="observation must be finite"):
            jax.block_until_ready(  # type: ignore[no-untyped-call]
                compiled_start(image)
            )
        with pytest.raises(Exception, match="observation must be finite"):
            jax.block_until_ready(  # type: ignore[no-untyped-call]
                causal_map_step(state, 0.0, image, config)
            )
        with pytest.raises(Exception, match="observation must be finite"):
            jax.block_until_ready(  # type: ignore[no-untyped-call]
                compiled_step(state, image)
            )


@pytest.mark.parametrize("complex_value", (1j, 1.0 + 1.0j))
def test_host_and_pure_paths_reject_complex_observations_before_cast(
    complex_value: complex,
) -> None:
    config = CausalMapForagerConfig()
    complex_observation = jnp.zeros((3, 3, 2), dtype=jnp.complex64).at[0, 0, 0].set(
        complex_value
    )
    with pytest.raises(ValueError, match="real numeric dtype"):
        CausalMapForagerAgent(config).start(complex_observation)
    with pytest.raises(ValueError, match="real numeric dtype"):
        causal_map_start(complex_observation, config, 0)
    with pytest.raises(ValueError, match="real numeric dtype"):
        jax.jit(lambda image: causal_map_start(image, config, 0))(
            complex_observation
        )

    valid = _observation(aperture=3, channels=2)
    state, _ = causal_map_start(valid, config, 0)
    with pytest.raises(ValueError, match="real numeric dtype"):
        causal_map_step(state, 0.0, complex_observation, config)
    with pytest.raises(ValueError, match="real numeric dtype"):
        jax.jit(
            lambda current, image: causal_map_step(current, 0.0, image, config)
        )(state, complex_observation)


@pytest.mark.parametrize(
    "seed",
    (
        True,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
    ),
)
def test_pure_start_rejects_invalid_seed_eager_and_jit(seed: Any) -> None:
    config = CausalMapForagerConfig()
    observation = _observation(aperture=3, channels=2)
    with pytest.raises(Exception, match="seed must be (?:one non-bool )?uint32-compatible"):
        jax.block_until_ready(  # type: ignore[no-untyped-call]
            causal_map_start(observation, config, seed)
        )
    compiled = jax.jit(
        lambda value: causal_map_start(observation, config, value)
    )
    with pytest.raises(Exception, match="seed must be (?:one non-bool )?uint32-compatible"):
        jax.block_until_ready(compiled(seed))  # type: ignore[no-untyped-call]


def test_pure_start_accepts_full_uint32_seed_range_eager_and_jit() -> None:
    config = CausalMapForagerConfig()
    observation = _observation(aperture=3, channels=2)
    seed = jnp.asarray(np.iinfo(np.uint32).max, dtype=jnp.uint32)
    eager_state, _ = causal_map_start(observation, config, seed)
    compiled_state, _ = jax.jit(
        lambda value: causal_map_start(observation, config, value)
    )(seed)
    assert int(eager_state.initial_seed) == np.iinfo(np.uint32).max
    chex.assert_trees_all_equal(eager_state, compiled_state)


@pytest.mark.parametrize("seed", (np.int64(-1), np.uint64(2**32)))
def test_pure_start_rejects_numpy_seed_aliases_before_jax_cast(seed: Any) -> None:
    with pytest.raises(ValueError, match="seed must be one non-bool uint32-compatible"):
        causal_map_start(_observation(aperture=3), CausalMapForagerConfig(), seed)


def test_pure_step_rejects_channel_count_change_and_invalid_reward_eager_and_jit() -> None:
    config = CausalMapForagerConfig()
    observation = _observation(aperture=3, channels=2)
    state, _ = causal_map_start(observation, config, 0)
    changed_channels = _observation(aperture=3, channels=3)
    with pytest.raises(ValueError, match="channel count changed"):
        causal_map_step(state, 0.0, changed_channels, config)
    with pytest.raises(ValueError, match="channel count changed"):
        jax.jit(lambda image: causal_map_step(state, 0.0, image, config))(
            changed_channels
        )

    compiled = jax.jit(
        lambda current, reward: causal_map_step(
            current,
            reward,
            observation,
            config,
        )
    )
    for reward in (jnp.asarray(jnp.nan), jnp.asarray(jnp.inf)):
        with pytest.raises(Exception, match="reward must be one finite"):
            jax.block_until_ready(  # type: ignore[no-untyped-call]
                causal_map_step(state, reward, observation, config)
            )
        with pytest.raises(Exception, match="reward must be one finite"):
            jax.block_until_ready(  # type: ignore[no-untyped-call]
                compiled(state, reward)
            )
    with pytest.raises(ValueError, match="one finite real scalar"):
        causal_map_step(state, jnp.asarray((1.0,)), observation, config)


def test_reward_sign_and_magnitude_are_learned_from_own_transition() -> None:
    config = CausalMapForagerConfig()
    state, action = causal_map_start(_observation((1, 0, 1)), config, 2)
    assert int(action) == 0
    assert np.all(np.asarray(state.reward_count) == 0)
    state, _, diagnostics = causal_map_step(
        state,
        jnp.asarray(-3.5, dtype=jnp.float32),
        _observation(),
        config,
    )
    assert bool(diagnostics.learned_reward)
    assert float(state.reward_sum[1]) == pytest.approx(-3.5)
    assert int(state.reward_count[1]) == 1
    assert int(state.cell_collection_step[1, 0]) == 1
    assert int(state.cell_ready_step[1, 0]) == 11


def test_reward_accumulation_overflow_is_rejected_eager_and_jit() -> None:
    config = CausalMapForagerConfig()
    observation = _observation((1, 0, 0), aperture=3, channels=1)
    state, action = causal_map_start(observation, config, 0)
    assert int(action) == 0
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    state, _, _ = causal_map_step(state, maximum, observation, config)

    with pytest.raises(Exception, match="reward accumulation must remain finite"):
        jax.block_until_ready(  # type: ignore[no-untyped-call]
            causal_map_step(state, maximum, observation, config)
        )
    compiled = jax.jit(
        lambda current: causal_map_step(current, maximum, observation, config)
    )
    with pytest.raises(Exception, match="reward accumulation must remain finite"):
        jax.block_until_ready(compiled(state))  # type: ignore[no-untyped-call]


def test_zero_reward_without_collection_does_not_corrupt_channel_mean() -> None:
    config = CausalMapForagerConfig()
    state, action = causal_map_start(_observation((1, 0, 0)), config, 0)
    assert int(action) == 0
    state, _, diagnostics = causal_map_step(state, jnp.asarray(0.0), _observation(), config)
    assert not bool(diagnostics.learned_reward)
    assert int(state.reward_count[0]) == 0
    assert int(state.cell_collection_step[1, 0]) == -1


def test_dead_reckoning_uses_public_action_order_and_wraps() -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(last_action=jnp.asarray(2, dtype=jnp.int32))
    state, _, _ = causal_map_step(state, jnp.asarray(0.0), _observation(), config)
    assert tuple(np.asarray(state.position)) == (0, 14)
    state = state._replace(last_action=jnp.asarray(3, dtype=jnp.int32))
    state, _, _ = causal_map_step(state, jnp.asarray(0.0), _observation(), config)
    assert tuple(np.asarray(state.position)) == (14, 14)


def test_negative_channel_cells_are_not_selected_when_safe_move_exists() -> None:
    config = CausalMapForagerConfig(tie_break_scale=1e-8)
    state, _ = causal_map_start(
        _observation((1, 0, 0), (0, 2, 1)),
        config,
        0,
    )
    state = state._replace(
        reward_sum=state.reward_sum.at[0].set(-1.0).at[1].set(5.0),
        reward_count=state.reward_count.at[0].set(1).at[1].set(1),
    )
    state, action = _choose_action(state, config)
    assert int(action) != 0
    destination = np.asarray(state.last_target_position)
    assert tuple(destination) != (0, 1)


@pytest.mark.parametrize("corrupt_value", [math.nan, math.inf])
def test_nonfinite_candidate_scores_fail_closed_to_finite_target(
    corrupt_value: float,
) -> None:
    config = CausalMapForagerConfig(
        exploration_probability=0.0,
        tie_break_scale=1e-8,
    )
    state, _ = causal_map_start(
        _observation((1, 0, 1), (0, 1, 2)),
        config,
        0,
    )
    state = state._replace(
        reward_sum=(
            state.reward_sum.at[1].set(corrupt_value).at[2].set(1.0)
        ),
        reward_count=state.reward_count.at[1].set(1).at[2].set(1),
    )
    planned, action = _choose_action(state, config)
    assert int(action) == 1
    assert tuple(np.asarray(planned.last_target_position)) == (1, 0)


def _routing_state(
    config: CausalMapForagerConfig,
    *,
    target: tuple[int, int] | None,
    negatives: tuple[tuple[int, int], ...],
) -> Any:
    state = _empty_state(2, config, 0)
    state = state._replace(
        visit_count=state.visit_count.at[0, 0].set(1),
        reward_sum=state.reward_sum.at[0].set(-1.0).at[1].set(5.0),
        reward_count=state.reward_count.at[0].set(1).at[1].set(1),
    )
    cell_channel = state.cell_channel
    cell_active = state.cell_active
    last_seen = state.cell_last_seen_step
    for x, y in negatives:
        cell_channel = cell_channel.at[y, x].set(0)
        last_seen = last_seen.at[y, x].set(0)
    if target is not None:
        x, y = target
        cell_channel = cell_channel.at[y, x].set(1)
        cell_active = cell_active.at[y, x].set(True)
        last_seen = last_seen.at[y, x].set(0)
    return state._replace(
        cell_channel=cell_channel,
        cell_active=cell_active,
        cell_last_seen_step=last_seen,
    )


def _host_cost_aware_route_grid(
    source: tuple[int, int],
    negative_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent lexicographic Dijkstra oracle for public traversable cells."""
    height, width = negative_cells.shape
    infinity = height * width + 1
    risks = np.full((height, width), infinity, dtype=np.int32)
    distances = np.full((height, width), infinity, dtype=np.int32)
    source_x, source_y = source
    risks[source_y, source_x] = 0
    distances[source_y, source_x] = 0
    queue: list[tuple[int, int, int, int]] = [(0, 0, source_x, source_y)]
    while queue:
        risk, distance, x, y = heapq.heappop(queue)
        if (risk, distance) != (int(risks[y, x]), int(distances[y, x])):
            continue
        for dx, dy in _DIRECTION_STEPS:
            neighbor_x = (x + dx) % width
            neighbor_y = (y + dy) % height
            candidate = (
                risk + int(negative_cells[neighbor_y, neighbor_x]),
                distance + 1,
            )
            incumbent = (
                int(risks[neighbor_y, neighbor_x]),
                int(distances[neighbor_y, neighbor_x]),
            )
            if candidate < incumbent:
                risks[neighbor_y, neighbor_x], distances[neighbor_y, neighbor_x] = (
                    candidate
                )
                heapq.heappush(
                    queue,
                    (candidate[0], candidate[1], neighbor_x, neighbor_y),
                )
    return risks, distances


@pytest.mark.parametrize("world_shape", ((1, 1), (2, 3), (3, 3)))
def test_cost_aware_route_grid_matches_lexicographic_dijkstra_under_jit_vmap(
    world_shape: tuple[int, int],
) -> None:
    config = CausalMapForagerConfig(world_shape=world_shape)
    height, width = world_shape
    cell_count = height * width
    masks: list[np.ndarray] = []
    sources: list[tuple[int, int]] = []
    expected_risks: list[np.ndarray] = []
    expected_distances: list[np.ndarray] = []
    for bits in range(1 << cell_count):
        negative = np.asarray(
            [(bits >> index) & 1 for index in range(cell_count)],
            dtype=np.bool_,
        ).reshape(world_shape)
        for source_y in range(height):
            for source_x in range(width):
                source = (source_x, source_y)
                expected_risk, expected_distance = _host_cost_aware_route_grid(
                    source,
                    negative,
                )
                masks.append(negative)
                sources.append(source)
                expected_risks.append(expected_risk)
                expected_distances.append(expected_distance)

    actual_risks, actual_distances = jax.jit(
        jax.vmap(
            lambda source, mask: _cost_aware_route_grid(source, mask, config)
        )
    )(
        jnp.asarray(sources, dtype=jnp.int32),
        jnp.asarray(np.stack(masks), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(np.asarray(actual_risks), np.stack(expected_risks))
    np.testing.assert_array_equal(
        np.asarray(actual_distances),
        np.stack(expected_distances),
    )


def test_safe_toroidal_routing_uses_wrap_and_routes_around_barriers() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=0.0,
        tie_break_scale=1e-8,
    )
    wrap_state = _routing_state(config, target=(4, 0), negatives=())
    _, wrap_action = _choose_action(wrap_state, config)
    assert int(wrap_action) == 3

    barrier_state = _routing_state(
        config,
        target=(2, 0),
        negatives=((1, 0), (4, 0)),
    )
    negative_mask = barrier_state.cell_channel == 0
    crossing_risk, distances = _cost_aware_route_grid(
        jnp.asarray((2, 0), dtype=jnp.int32),
        negative_mask,
        config,
    )
    assert int(crossing_risk[0, 0]) == 0
    assert int(distances[0, 0]) == 4
    planned, barrier_action = _choose_action(barrier_state, config)
    assert int(barrier_action) in (0, 2)
    destination_x, destination_y = map(int, np.asarray(planned.last_target_position))
    assert not bool(negative_mask[destination_y, destination_x])


def test_safe_grid_marks_enclosure_but_cost_route_can_cross_with_explicit_fallback() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=0.0,
        tie_break_scale=1e-8,
    )
    ring = ((2, 1), (3, 2), (2, 3), (1, 2))
    enclosed = _routing_state(config, target=(2, 2), negatives=ring)
    negative_mask = enclosed.cell_channel == 0
    crossing_risk, crossing_distance = _cost_aware_route_grid(
        jnp.asarray((0, 0), dtype=jnp.int32),
        negative_mask,
        config,
    )
    assert int(crossing_risk[2, 2]) == 1
    assert int(crossing_distance[2, 2]) == 4
    planned, action = _choose_action(enclosed, config)
    assert 0 <= int(action) < 4
    destination_x, destination_y = map(int, np.asarray(planned.last_target_position))
    assert not bool(negative_mask[destination_y, destination_x])

    all_neighbor_cells = ((0, 1), (1, 0), (0, 4), (4, 0))
    trapped = _routing_state(
        config,
        target=None,
        negatives=all_neighbor_cells,
    )
    planned, action = _choose_action(trapped, config)
    assert 0 <= int(action) < 4
    destination_x, destination_y = map(int, np.asarray(planned.last_target_position))
    assert bool((trapped.cell_channel == 0)[destination_y, destination_x])


def _negative_crossing_state(
    config: CausalMapForagerConfig,
    *,
    sealed: bool,
    remote_reward: float = 10.0,
    local_reward: float = 1.0,
) -> Any:
    """Return an observed map with remote/local rewards and learned deathcaps."""
    state = _empty_state(3, config, 41)
    negative_positions = (
        tuple((1, y) for y in range(config.height))
        + tuple((config.width - 1, y) for y in range(config.height))
        if sealed
        else ((1, 0),)
    )
    channels = state.cell_channel
    last_absent = jnp.zeros(config.world_shape, dtype=jnp.int32)
    for x, y in negative_positions:
        channels = channels.at[y, x].set(0)
    channels = channels.at[0, 2].set(1).at[1, 0].set(2)
    active = state.cell_active.at[0, 2].set(True).at[1, 0].set(True)
    last_absent = last_absent.at[0, 2].set(-1).at[1, 0].set(-1)
    return state._replace(
        cell_channel=channels,
        cell_active=active,
        cell_last_seen_step=jnp.zeros(config.world_shape, dtype=jnp.int32),
        cell_last_absent_step=last_absent,
        reward_sum=(
            state.reward_sum.at[0]
            .set(-1.0)
            .at[1]
            .set(remote_reward)
            .at[2]
            .set(local_reward)
        ),
        reward_count=jnp.ones((3,), dtype=jnp.int32),
    )


def test_cost_route_avoids_deathcap_when_clean_detour_exists() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=0.0,
        distance_cost=0.01,
        tie_break_scale=1e-8,
    )
    state = _negative_crossing_state(config, sealed=False)
    planned, action = _choose_action(state, config)
    assert int(action) == 3
    destination_x, destination_y = map(int, np.asarray(planned.last_target_position))
    assert int(state.cell_channel[destination_y, destination_x]) != 0


def test_profitable_target_can_cross_minimum_deathcap_barrier_eager_jit_vmap() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=0.0,
        distance_cost=0.01,
        tie_break_scale=1e-8,
    )
    clean_detour = _negative_crossing_state(config, sealed=False)
    sealed = _negative_crossing_state(config, sealed=True)

    def choose(value: Any) -> jax.Array:
        return _choose_action(value, config)[1]

    assert int(choose(clean_detour)) == 3
    assert int(choose(sealed)) == 1
    assert int(jax.jit(choose)(sealed)) == 1
    batch = jax.tree.map(
        lambda left, right: jnp.stack((left, right)),
        clean_detour,
        sealed,
    )
    np.testing.assert_array_equal(
        np.asarray(jax.jit(jax.vmap(choose))(batch)),
        np.asarray((3, 1)),
    )

    planned, _ = _choose_action(sealed, config)
    destination_x, destination_y = map(int, np.asarray(planned.last_target_position))
    assert int(sealed.cell_channel[destination_y, destination_x]) == 0


def test_negative_entry_price_prefers_safe_local_reward_when_crossing_is_not_worth_it() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=0.0,
        distance_cost=0.01,
        tie_break_scale=1e-8,
    )
    state = _negative_crossing_state(
        config,
        sealed=True,
        remote_reward=1.0,
        local_reward=1.0,
    )
    planned, action = _choose_action(state, config)
    assert int(action) == 0
    assert tuple(np.asarray(planned.last_target_position)) == (0, 1)


def _readiness_routing_state(
    config: CausalMapForagerConfig,
    *,
    step_count: int,
    pending_ready_step: int,
) -> Any:
    """Build a fully observed planner state with opposed pending/active targets."""
    state = _empty_state(3, config, 29)
    pending_x = 3
    active_x = config.width - 1
    return state._replace(
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        cell_channel=(
            state.cell_channel.at[0, pending_x].set(1).at[0, active_x].set(2)
        ),
        cell_active=state.cell_active.at[0, active_x].set(True),
        cell_collection_step=state.cell_collection_step.at[0, pending_x].set(1),
        cell_ready_step=(
            state.cell_ready_step.at[0, pending_x].set(pending_ready_step)
        ),
        # Every cell has genuinely been observed, so exploration_probability
        # cannot mask which exploitation target passed the readiness gate.
        cell_last_seen_step=jnp.full(
            config.world_shape,
            step_count,
            dtype=jnp.int32,
        ),
        cell_last_absent_step=jnp.full(
            config.world_shape,
            step_count,
            dtype=jnp.int32,
        ),
        reward_sum=state.reward_sum.at[1].set(10.0).at[2].set(1.0),
        reward_count=state.reward_count.at[1].set(1).at[2].set(1),
    )


def test_arrival_readiness_boundary_matches_pre_entry_reward_order_eager_jit_vmap() -> None:
    config = CausalMapForagerConfig(
        world_shape=(7, 7),
        exploration_probability=1.0,
        arrival_aware_readiness=True,
        distance_cost=0.01,
        tie_break_scale=1e-8,
    )
    # The pending high-value target is three moves east.  The entry action is
    # evaluated from step 12, so ready_step=12 is collectible on arrival while
    # ready_step=13 is one public transition too late.  The active lower-value
    # target immediately west makes the rejected case unambiguous.
    on_boundary = _readiness_routing_state(
        config,
        step_count=10,
        pending_ready_step=12,
    )
    one_step_late = _readiness_routing_state(
        config,
        step_count=10,
        pending_ready_step=13,
    )

    def choose(value: Any) -> jax.Array:
        return _choose_action(value, config)[1]

    assert int(choose(on_boundary)) == 1
    assert int(choose(one_step_late)) == 3
    assert int(jax.jit(choose)(on_boundary)) == 1
    assert int(jax.jit(choose)(one_step_late)) == 3

    batch = jax.tree.map(
        lambda left, right: jnp.stack((left, right)),
        on_boundary,
        one_step_late,
    )
    actions = jax.jit(jax.vmap(choose))(batch)
    np.testing.assert_array_equal(np.asarray(actions), np.asarray((1, 3)))


def test_arrival_readiness_can_be_disabled_for_exact_decision_time_semantics() -> None:
    arrival_config = CausalMapForagerConfig(
        world_shape=(7, 7),
        exploration_probability=0.0,
        arrival_aware_readiness=True,
        distance_cost=0.01,
        tie_break_scale=1e-8,
    )
    decision_config = dataclasses.replace(
        arrival_config,
        arrival_aware_readiness=False,
    )
    state = _readiness_routing_state(
        arrival_config,
        step_count=10,
        pending_ready_step=12,
    )
    assert int(_choose_action(state, arrival_config)[1]) == 1
    assert int(_choose_action(state, decision_config)[1]) == 3


def test_arrival_readiness_saturates_near_int32_lifetime_instead_of_wrapping() -> None:
    config = CausalMapForagerConfig(
        world_shape=(7, 7),
        exploration_probability=0.0,
        arrival_aware_readiness=True,
        distance_cost=0.01,
        tie_break_scale=1e-8,
    )
    maximum = np.iinfo(np.int32).max
    state = _readiness_routing_state(
        config,
        step_count=maximum - 1,
        pending_ready_step=maximum,
    )
    eager = _choose_action(state, config)[1]
    compiled = jax.jit(lambda value: _choose_action(value, config)[1])(state)
    assert int(eager) == 1
    assert int(compiled) == 1


def test_exploration_probability_never_redirects_after_reachable_map_is_observed() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=1.0,
        tie_break_scale=1e-8,
    )
    state = _empty_state(2, config, 7)
    visits = jnp.full(config.world_shape, 100, dtype=jnp.int32).at[0, 4].set(0)
    state = state._replace(
        cell_channel=state.cell_channel.at[0, 1].set(1),
        cell_active=state.cell_active.at[0, 1].set(True),
        cell_last_seen_step=jnp.zeros(config.world_shape, dtype=jnp.int32),
        cell_last_absent_step=jnp.zeros(config.world_shape, dtype=jnp.int32),
        visit_count=visits,
        reward_sum=state.reward_sum.at[1].set(5.0),
        reward_count=state.reward_count.at[1].set(1),
    )
    eager_state, eager_action = _choose_action(state, config)
    compiled_state, compiled_action = jax.jit(
        lambda value: _choose_action(value, config)
    )(state)
    assert int(eager_action) == 1
    assert int(compiled_action) == 1
    chex.assert_trees_all_equal(eager_state, compiled_state)


def test_exploration_can_cross_deathcap_ring_to_genuinely_unobserved_cell() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=1.0,
        tie_break_scale=1e-8,
    )
    state = _empty_state(2, config, 7)
    ring = ((2, 1), (3, 2), (2, 3), (1, 2))
    channels = state.cell_channel.at[0, 1].set(1)
    for x, y in ring:
        channels = channels.at[y, x].set(0)
    last_seen = jnp.zeros(config.world_shape, dtype=jnp.int32).at[2, 2].set(-1)
    state = state._replace(
        cell_channel=channels,
        cell_active=state.cell_active.at[0, 1].set(True),
        cell_last_seen_step=last_seen,
        cell_last_absent_step=jnp.zeros(config.world_shape, dtype=jnp.int32),
        reward_sum=state.reward_sum.at[0].set(-1.0).at[1].set(5.0),
        reward_count=state.reward_count.at[0].set(1).at[1].set(1),
    )
    # Public deathcaps are costly but traversable, so the ring can no longer
    # make its unobserved center permanently unreachable.  With exploration
    # forced, the planner begins the minimum-one-deathcap route instead of
    # exploiting the adjacent known reward.
    assert int(_choose_action(state, config)[1]) == 0
    assert int(jax.jit(lambda value: _choose_action(value, config)[1])(state)) == 0


def test_no_target_after_full_coverage_keeps_global_safe_coverage_fail_safe() -> None:
    config = CausalMapForagerConfig(
        world_shape=(5, 5),
        exploration_probability=1.0,
        tie_break_scale=1e-8,
    )
    state = _empty_state(1, config, 7)
    visits = jnp.full(config.world_shape, 100, dtype=jnp.int32).at[0, 4].set(0)
    state = state._replace(
        cell_last_seen_step=jnp.zeros(config.world_shape, dtype=jnp.int32),
        cell_last_absent_step=jnp.zeros(config.world_shape, dtype=jnp.int32),
        visit_count=visits,
    )
    planned, action = _choose_action(state, config)
    assert int(action) == 3
    assert int(planned.last_target_channel) == -1


def test_seeded_persistent_coverage_discovers_remote_region_despite_local_target() -> None:
    config = CausalMapForagerConfig(
        world_shape=(7, 7),
        exploration_probability=1.0,
        tie_break_scale=1e-6,
    )
    directions = np.asarray(((0, 1), (1, 0), (0, -1), (-1, 0)))
    objects = {(1, 0): 0, (3, 3): 1}

    def observation(position: tuple[int, int]) -> jax.Array:
        image = np.zeros((3, 3, 2), dtype=np.float32)
        for row, dy in enumerate((-1, 0, 1)):
            for col, dx in enumerate((-1, 0, 1)):
                coordinate = (
                    (position[0] + dx) % config.width,
                    (position[1] + dy) % config.height,
                )
                if coordinate in objects:
                    image[row, col, objects[coordinate]] = 1.0
        return jnp.asarray(image)

    def run(seed: int) -> tuple[Any, tuple[int, ...]]:
        true_position = np.asarray((0, 0), dtype=np.int32)
        state, action = causal_map_start(observation((0, 0)), config, seed)
        actions: list[int] = [int(action)]
        transition = jax.jit(
            lambda current, image: causal_map_step(
                current,
                jnp.asarray(0.0, dtype=jnp.float32),
                image,
                config,
            )
        )
        for _ in range(40):
            true_position = np.mod(
                true_position + directions[int(action)],
                np.asarray((config.width, config.height)),
            )
            state, action, _ = transition(
                state,
                observation((int(true_position[0]), int(true_position[1]))),
            )
            actions.append(int(action))
        return state, tuple(actions)

    first_state, first_actions = run(23)
    second_state, second_actions = run(23)
    assert first_actions == second_actions
    chex.assert_trees_all_equal(first_state, second_state)
    assert int(first_state.cell_last_seen_step[3, 3]) >= 0
    assert np.any(np.asarray(first_state.cell_last_seen_step)[:, :3] >= 0)
    assert np.any(np.asarray(first_state.cell_last_seen_step)[:, 4:] >= 0)


def test_retry_backoff_is_generic_and_exponential() -> None:
    config = CausalMapForagerConfig(initial_retry_delay=10)
    state, _ = causal_map_start(_observation((1, 0, 2)), config, 0)
    state = state._replace(
        last_action=jnp.asarray(0, dtype=jnp.int32),
        last_target_channel=jnp.asarray(2, dtype=jnp.int32),
        last_target_position=jnp.asarray((0, 1), dtype=jnp.int32),
        last_target_expected_active=jnp.asarray(True),
        cell_active=state.cell_active.at[1, 0].set(False),
        cell_collection_step=state.cell_collection_step.at[1, 0].set(0),
        cell_ready_step=state.cell_ready_step.at[1, 0].set(1),
    )
    state, _, diagnostics = causal_map_step(
        state,
        jnp.asarray(0.0),
        _observation(),
        config,
    )
    assert bool(diagnostics.retry_miss)
    assert int(state.cell_retry_count[1, 0]) == 1
    assert int(state.cell_ready_step[1, 0]) == 21


def test_visible_due_absence_backs_off_once_and_is_not_targeted() -> None:
    config = CausalMapForagerConfig(
        initial_retry_delay=10,
        exploration_probability=0.0,
        tie_break_scale=1e-8,
    )
    state, _ = causal_map_start(_observation((1, 0, 1)), config, 0)
    state = state._replace(
        step_count=jnp.asarray(1, dtype=jnp.int32),
        cell_active=state.cell_active.at[1, 0].set(False),
        cell_collection_step=state.cell_collection_step.at[1, 0].set(0),
        cell_ready_step=state.cell_ready_step.at[1, 0].set(1),
    )

    state, _, visible_misses = _integrate_observation(
        state,
        _observation(),
        config,
    )
    assert int(visible_misses) == 1
    assert int(state.cell_retry_count[1, 0]) == 1
    assert int(state.cell_ready_step[1, 0]) == 21
    planned, _ = _choose_action(state, config)
    assert tuple(np.asarray(planned.last_target_position)) != (0, 1)

    state = state._replace(step_count=jnp.asarray(2, dtype=jnp.int32))
    state, _, repeated_misses = _integrate_observation(
        state,
        _observation(),
        config,
    )
    assert int(repeated_misses) == 0
    assert int(state.cell_retry_count[1, 0]) == 1
    assert int(state.cell_ready_step[1, 0]) == 21


def test_retry_arithmetic_saturates_without_zero_or_wraparound() -> None:
    config = CausalMapForagerConfig(
        initial_retry_delay=2,
        maximum_retry_delay=np.iinfo(np.int32).max,
        maximum_retry_exponent=31,
    )
    delay = _retry_delay(jnp.asarray(31, dtype=jnp.int32), config)
    assert int(delay) == np.iinfo(np.int32).max
    assert int(delay) >= 1
    saturated = _saturating_add_int32(
        jnp.asarray(np.iinfo(np.int32).max - 3, dtype=jnp.int32),
        jnp.asarray(10, dtype=jnp.int32),
    )
    assert int(saturated) == np.iinfo(np.int32).max
    with pytest.raises(ValueError, match="positive integer"):
        CausalMapForagerConfig(
            maximum_retry_delay=np.iinfo(np.int32).max + 1,
        )


def test_transition_rejects_a_valid_saturated_lifetime_eager_and_jit() -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation((1, 0, 1)), config, 0)
    maximum = np.iinfo(np.int32).max
    state = state._replace(
        step_count=jnp.asarray(maximum, dtype=jnp.int32),
        visit_count=jnp.zeros_like(state.visit_count).at[0, 0].set(maximum),
        cell_last_seen_step=state.cell_last_seen_step.at[0, 0].set(maximum),
        cell_last_absent_step=state.cell_last_absent_step.at[0, 0].set(maximum),
    )
    validate_causal_map_state(state, config)

    with pytest.raises(Exception, match="step_count is saturated"):
        causal_map_step(state, jnp.asarray(0.0), _observation(), config)

    compiled = jax.jit(
        lambda value: causal_map_step(
            value,
            jnp.asarray(0.0),
            _observation(),
            config,
        )
    )
    with pytest.raises(Exception, match="step_count is saturated"):
        compiled(state)[0].step_count.block_until_ready()


def test_reappearance_interval_updates_channel_schedule_without_type_prior() -> None:
    config = CausalMapForagerConfig(
        respawn_safety_quantile=0.5,
        respawn_safety_factor=1.0,
    )
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(
        step_count=jnp.asarray(301, dtype=jnp.int32),
        position=jnp.asarray((4, 5), dtype=jnp.int32),
        cell_channel=state.cell_channel.at[5, 4].set(1),
        cell_collection_step=state.cell_collection_step.at[5, 4].set(1),
        cell_ready_step=state.cell_ready_step.at[5, 4].set(11),
        cell_last_absent_step=state.cell_last_absent_step.at[5, 4].set(300),
    )
    state, learned, visible_misses = _integrate_observation(
        state,
        _observation((0, 0, 1)),
        config,
    )
    assert int(learned) == 1
    assert int(visible_misses) == 0
    assert int(state.respawn_interval_count[1]) == 1
    assert int(state.respawn_interval_lower_floor[1]) == 300
    assert int(state.respawn_interval_lower_remainder[1]) == 0
    assert int(state.respawn_interval_upper_floor[1]) == 300
    assert int(state.respawn_interval_upper_remainder[1]) == 0
    assert int(state.respawn_exact_count[1]) == 1
    assert int(state.respawn_exact_floor[1]) == 300
    assert int(state.respawn_exact_remainder[1]) == 0
    assert float(state.respawn_exact_mean[1]) == pytest.approx(300.0)
    assert int(_estimated_respawn_delay(state, jnp.asarray(1), config)) == 300
    assert int(state.cell_collection_step[5, 4]) == -1


def test_censored_interval_estimator_preserves_online_identification_bounds() -> None:
    count, lower_floor, lower_remainder, upper_floor, upper_remainder = (
        _merge_channel_interval_bounds(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray((1, 1), dtype=jnp.int32),
        jnp.asarray((2, 101), dtype=jnp.int32),
        jnp.asarray((300, 300), dtype=jnp.int32),
        jnp.asarray((True, True)),
        )
    )
    assert int(count[1]) == 2
    assert (int(lower_floor[1]), int(lower_remainder[1])) == (51, 1)
    assert (int(upper_floor[1]), int(upper_remainder[1])) == (300, 0)
    # One compatible latent population is (100, 250), whose empirical mean
    # must remain inside the endpoint-mean identification interval.
    latent_mean = (100.0 + 250.0) / 2.0
    exact_lower = float(lower_floor[1]) + float(lower_remainder[1]) / float(count[1])
    exact_upper = float(upper_floor[1]) + float(upper_remainder[1]) / float(count[1])
    assert exact_lower <= latent_mean <= exact_upper

    config = CausalMapForagerConfig(respawn_safety_quantile=0.5)
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(
        respawn_interval_count=count,
        respawn_interval_lower_floor=lower_floor,
        respawn_interval_lower_remainder=lower_remainder,
        respawn_interval_upper_floor=upper_floor,
        respawn_interval_upper_remainder=upper_remainder,
    )
    # Scheduling uses the conservative upper endpoint.  In particular, it
    # never treats either lower bound (2 or 101) as an exact respawn draw.
    assert int(_estimated_respawn_delay(state, jnp.asarray(1), config)) == 300


def test_exact_rational_interval_bound_is_outward_after_65_537_samples() -> None:
    sample_count = 65_537
    channels = jnp.zeros((sample_count,), dtype=jnp.int32)
    lower = jnp.full((sample_count,), 300, dtype=jnp.int32).at[0].set(299)
    upper = jnp.full((sample_count,), 300, dtype=jnp.int32).at[0].set(301)
    mask = jnp.ones((sample_count,), dtype=jnp.bool_)
    zero = jnp.zeros((1,), dtype=jnp.int32)

    def merge() -> tuple[jax.Array, ...]:
        return _merge_channel_interval_bounds(
            zero,
            zero,
            zero,
            zero,
            zero,
            channels,
            lower,
            upper,
            mask,
        )

    eager = merge()
    compiled = jax.jit(merge)()
    chex.assert_trees_all_equal(eager, compiled)
    count, lower_floor, lower_remainder, upper_floor, upper_remainder = eager
    assert int(count[0]) == sample_count
    assert (int(lower_floor[0]), int(lower_remainder[0])) == (299, 65_536)
    assert (int(upper_floor[0]), int(upper_remainder[0])) == (300, 1)

    config = CausalMapForagerConfig(respawn_safety_quantile=0.5)
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(
        respawn_interval_count=count,
        respawn_interval_lower_floor=lower_floor,
        respawn_interval_lower_remainder=lower_remainder,
        respawn_interval_upper_floor=upper_floor,
        respawn_interval_upper_remainder=upper_remainder,
    )
    assert int(_estimated_respawn_delay(state, jnp.asarray(0), config)) == 301
    compiled_delay = jax.jit(
        lambda value: _estimated_respawn_delay(value, jnp.asarray(0), config)
    )(state)
    assert int(compiled_delay) == 301


def test_visible_censored_reappearance_retains_both_bounds_separately() -> None:
    config = CausalMapForagerConfig(respawn_safety_quantile=0.5)
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(
        step_count=jnp.asarray(301, dtype=jnp.int32),
        position=jnp.asarray((4, 5), dtype=jnp.int32),
        cell_channel=state.cell_channel.at[5, 4].set(1),
        cell_collection_step=state.cell_collection_step.at[5, 4].set(1),
        cell_ready_step=state.cell_ready_step.at[5, 4].set(11),
        cell_last_absent_step=state.cell_last_absent_step.at[5, 4].set(101),
    )
    state, learned, _ = _integrate_observation(
        state,
        _observation((0, 0, 1)),
        config,
    )
    assert int(learned) == 1
    assert int(state.respawn_interval_count[1]) == 1
    assert int(state.respawn_interval_lower_floor[1]) == 101
    assert int(state.respawn_interval_lower_remainder[1]) == 0
    assert int(state.respawn_interval_upper_floor[1]) == 300
    assert int(state.respawn_interval_upper_remainder[1]) == 0
    assert int(state.respawn_exact_count[1]) == 0
    assert int(_estimated_respawn_delay(state, jnp.asarray(1), config)) == 300


def test_immediate_recollection_captures_reappearance_before_pending_overwrite() -> None:
    config = CausalMapForagerConfig(respawn_safety_quantile=0.5)
    state = _empty_state(2, config, 0)
    state = state._replace(
        step_count=jnp.asarray(10, dtype=jnp.int32),
        last_action=jnp.asarray(0, dtype=jnp.int32),
        last_target_channel=jnp.asarray(1, dtype=jnp.int32),
        last_target_position=jnp.asarray((0, 1), dtype=jnp.int32),
        last_target_expected_active=jnp.asarray(True),
        cell_channel=state.cell_channel.at[1, 0].set(1),
        cell_collection_step=state.cell_collection_step.at[1, 0].set(1),
        cell_ready_step=state.cell_ready_step.at[1, 0].set(5),
        cell_last_seen_step=(
            state.cell_last_seen_step.at[0, 0].set(10).at[1, 0].set(10)
        ),
        cell_last_absent_step=(
            state.cell_last_absent_step.at[0, 0].set(10).at[1, 0].set(10)
        ),
        visit_count=state.visit_count.at[0, 0].set(11),
        reward_sum=state.reward_sum.at[1].set(1.0),
        reward_count=state.reward_count.at[1].set(1),
    )
    validate_causal_map_state(state, config)

    state, _, diagnostics = causal_map_step(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        _observation(aperture=3, channels=2),
        config,
    )
    assert int(diagnostics.learned_respawn) == 1
    assert int(state.respawn_interval_count[1]) == 1
    assert int(state.respawn_interval_lower_floor[1]) == 10
    assert int(state.respawn_interval_lower_remainder[1]) == 0
    assert int(state.respawn_interval_upper_floor[1]) == 10
    assert int(state.respawn_interval_upper_remainder[1]) == 0
    assert int(state.respawn_exact_count[1]) == 1
    assert int(state.respawn_exact_floor[1]) == 10
    assert int(state.respawn_exact_remainder[1]) == 0
    assert float(state.respawn_exact_mean[1]) == pytest.approx(10.0)
    # The old collection was learned before the new collection replaced it.
    assert int(state.cell_collection_step[1, 0]) == 11
    assert int(state.reward_count[1]) == 2
    validate_causal_map_state(state, config)


def test_respawn_safety_quantile_uses_observed_variance() -> None:
    config = CausalMapForagerConfig(
        respawn_safety_quantile=0.9,
        respawn_safety_factor=1.0,
    )
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(
        respawn_interval_count=state.respawn_interval_count.at[0].set(2),
        respawn_interval_lower_floor=state.respawn_interval_lower_floor.at[0].set(100),
        respawn_interval_upper_floor=state.respawn_interval_upper_floor.at[0].set(100),
        respawn_exact_count=state.respawn_exact_count.at[0].set(2),
        respawn_exact_floor=state.respawn_exact_floor.at[0].set(100),
        respawn_exact_mean=state.respawn_exact_mean.at[0].set(100.0),
        respawn_exact_m2=state.respawn_exact_m2.at[0].set(200.0),
    )
    delay = int(_estimated_respawn_delay(state, jnp.asarray(0), config))
    expected = math.ceil(100.0 + config.respawn_quantile_z * math.sqrt(200.0))
    assert delay == expected


def test_respawn_delay_clips_before_unsafe_float32_to_int32_conversion() -> None:
    maximum = np.iinfo(np.int32).max
    config = CausalMapForagerConfig(
        respawn_safety_quantile=0.5,
        respawn_safety_factor=2.0,
        maximum_respawn_delay=maximum,
    )
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(
        respawn_interval_count=state.respawn_interval_count.at[0].set(1),
        respawn_interval_lower_floor=state.respawn_interval_lower_floor.at[0].set(1),
        respawn_interval_upper_floor=(
            state.respawn_interval_upper_floor.at[0].set(1_500_000_000)
        ),
    )
    assert int(_estimated_respawn_delay(state, jnp.asarray(0), config)) == maximum


def test_exact_respawn_safety_statistics_can_only_raise_interval_schedule() -> None:
    config = CausalMapForagerConfig(respawn_safety_quantile=0.5)
    state, _ = causal_map_start(_observation(), config, 0)
    state = state._replace(
        respawn_interval_count=state.respawn_interval_count.at[1].set(3),
        respawn_interval_lower_floor=state.respawn_interval_lower_floor.at[1].set(10),
        respawn_interval_upper_floor=state.respawn_interval_upper_floor.at[1].set(20),
        respawn_exact_count=state.respawn_exact_count.at[1].set(2),
        respawn_exact_floor=state.respawn_exact_floor.at[1].set(10),
        respawn_exact_mean=state.respawn_exact_mean.at[1].set(10.0),
    )
    assert int(_estimated_respawn_delay(state, jnp.asarray(1), config)) == 20
    state = state._replace(
        respawn_exact_floor=state.respawn_exact_floor.at[1].set(30),
        respawn_exact_mean=state.respawn_exact_mean.at[1].set(30.0)
    )
    assert int(_estimated_respawn_delay(state, jnp.asarray(1), config)) == 30


def test_state_serialization_round_trip_and_validation() -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation((1, 0, 1)), config, 19)
    payload = causal_map_state_to_dict(state, config)
    assert payload["schema"] == CAUSAL_MAP_STATE_SCHEMA
    assert payload["schema"].endswith(".v5")
    assert payload["prng_impl"] == "threefry2x32"
    assert payload["jax_threefry_partitionable"] is bool(
        jax.config.jax_threefry_partitionable
    )
    assert payload["fields"]["initial_seed"] == 19
    restored = causal_map_state_from_dict(payload, config)
    chex.assert_trees_all_equal(state, restored)
    validate_causal_map_state(restored, config, observation_channels=3)

    corrupt = dict(payload)
    corrupt["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="configuration"):
        causal_map_state_from_dict(corrupt, config)
    bad_state = state._replace(cell_channel=state.cell_channel.at[0, 0].set(99))
    with pytest.raises(ValueError, match="channel"):
        validate_causal_map_state(bad_state, config)

    legacy = copy.deepcopy(payload)
    legacy["schema"] = "alberta.forager_causal_map_state.v4"
    with pytest.raises(ValueError, match="unsupported"):
        causal_map_state_from_dict(legacy, config)


def test_field_absent_v5_checkpoint_requires_explicit_legacy_readiness_policy() -> None:
    legacy_config = CausalMapForagerConfig(arrival_aware_readiness=False)
    state, _ = causal_map_start(
        _observation((1, 0, 1)),
        legacy_config,
        19,
    )
    payload = causal_map_state_to_dict(state, legacy_config)
    del payload["config"]["arrival_aware_readiness"]
    encoded = json.dumps(
        payload["config"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    payload["config_sha256"] = hashlib.sha256(encoded).hexdigest()

    restored = causal_map_state_from_dict(payload, legacy_config)
    chex.assert_trees_all_equal(state, restored)
    with pytest.raises(ValueError, match="different configuration"):
        causal_map_state_from_dict(
            payload,
            dataclasses.replace(legacy_config, arrival_aware_readiness=True),
        )


def test_checkpoint_continuation_is_exact_with_bound_prng_implementation() -> None:
    config = CausalMapForagerConfig()
    source = CausalMapForagerAgent(config, seed=23)
    source.start(_observation((1, 0, 1)))
    for _ in range(3):
        source.step(0.0, _observation())

    payload = source.state_dict()
    assert payload["prng_impl"] == "threefry2x32"
    restored = CausalMapForagerAgent(config, seed=23)
    restored.load_state_dict(payload)
    chex.assert_trees_all_equal(source.state, restored.state)

    for _ in range(10):
        assert source.step(0.0, _observation()) == restored.step(
            0.0,
            _observation(),
        )
        chex.assert_trees_all_equal(source.state, restored.state)


@pytest.mark.parametrize("arrival_aware_readiness", [False, True])
def test_agent_pickle_preserves_scheduler_config_and_exact_continuation(
    arrival_aware_readiness: bool,
) -> None:
    config = CausalMapForagerConfig(
        exploration_probability=0.75,
        arrival_aware_readiness=arrival_aware_readiness,
    )
    source = CausalMapForagerAgent(config, seed=31)
    source.start(_observation((1, 0, 1)))
    source.step(0.0, _observation())

    restored = pickle.loads(pickle.dumps(source))
    assert isinstance(restored, CausalMapForagerAgent)
    assert restored.config == config
    assert restored.config.fingerprint() == config.fingerprint()
    chex.assert_trees_all_equal(source.state, restored.state)

    for _ in range(3):
        assert source.step(0.0, _observation()) == restored.step(
            0.0,
            _observation(),
        )
        chex.assert_trees_all_equal(source.state, restored.state)


def test_checkpoint_rejects_cross_mode_threefry_partitionable_restore() -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation(), config, 23)
    payload = causal_map_state_to_dict(state, config)
    recorded = bool(jax.config.jax_threefry_partitionable)
    assert payload["jax_threefry_partitionable"] is recorded
    with jax.threefry_partitionable(not recorded):
        with pytest.raises(ValueError, match="jax_threefry_partitionable mode"):
            causal_map_state_from_dict(payload, config)
        assert causal_map_rng_contract()["jax_threefry_partitionable"] is not recorded
        assert (
            CausalMapForagerAgent(config).metadata()["jax_threefry_partitionable"]
            is not recorded
        )


def test_state_bound_threefry_mode_rejects_midrun_drift_eager_and_jit() -> None:
    config = CausalMapForagerConfig()
    observation = _observation()
    state, _ = causal_map_start(observation, config, 23)
    agent = CausalMapForagerAgent(config, seed=23)
    agent.start(observation)
    recorded = bool(state.jax_threefry_partitionable)
    assert recorded is bool(jax.config.jax_threefry_partitionable)

    with jax.threefry_partitionable(not recorded):
        with pytest.raises(ValueError, match="mode does not match runtime"):
            validate_causal_map_state(state, config)
        with pytest.raises(ValueError, match="mode does not match runtime"):
            causal_map_state_to_dict(state, config)
        with pytest.raises(ValueError, match="mode does not match runtime"):
            agent.state_dict()
        with pytest.raises(ValueError, match="mode does not match runtime"):
            agent.metadata()
        with pytest.raises(Exception, match="mode does not match runtime"):
            jax.block_until_ready(  # type: ignore[no-untyped-call]
                causal_map_step(state, 0.0, observation, config)
            )
        compiled = jax.jit(
            lambda current: causal_map_step(current, 0.0, observation, config)
        )
        with pytest.raises(Exception, match="mode does not match runtime"):
            jax.block_until_ready(compiled(state))  # type: ignore[no-untyped-call]


def test_checkpoint_cross_checks_state_bound_and_top_level_threefry_modes() -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation(), config, 23)
    payload = causal_map_state_to_dict(state, config)
    payload["fields"]["jax_threefry_partitionable"] = not payload[
        "jax_threefry_partitionable"
    ]
    with pytest.raises(ValueError, match="state-bound and top-level"):
        causal_map_state_from_dict(payload, config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("dtypes"), "top-level"),
        (
            lambda payload: payload.__setitem__("prng_impl", "rbg"),
            "PRNG implementation",
        ),
        (
            lambda payload: payload.__setitem__("jax_threefry_partitionable", 1),
            "jax_threefry_partitionable mode must be boolean",
        ),
        (
            lambda payload: payload["dtypes"].__setitem__("step_count", "float32"),
            "dtype table",
        ),
        (
            lambda payload: payload["fields"].__setitem__(
                "last_target_channel",
                999,
            ),
            "last_target_channel",
        ),
        (
            lambda payload: payload["fields"].__setitem__("step_count", [0]),
            "scalar",
        ),
        (
            lambda payload: payload["fields"].__setitem__("last_action", 1.9),
            "int32",
        ),
        (
            lambda payload: payload["fields"].__setitem__(
                "last_target_expected_active",
                "false",
            ),
            "bool",
        ),
        (
            lambda payload: payload.__setitem__("unexpected", 1),
            "top-level",
        ),
        (
            lambda payload: payload["fields"].__setitem__(
                "last_target_position",
                [15, 0],
            ),
            "outside",
        ),
        (
            lambda payload: payload["fields"]["cell_channel"].__setitem__(
                0,
                tuple(payload["fields"]["cell_channel"][0]),
            ),
            "JSON arrays",
        ),
        (
            lambda payload: payload["config"].__setitem__(
                "world_shape",
                tuple(payload["config"]["world_shape"]),
            ),
            "raw JSON values",
        ),
    ],
)
def test_state_deserialization_rejects_malformed_payload_without_coercion(
    mutation: Any,
    message: str,
) -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation((1, 0, 1)), config, 19)
    payload = copy.deepcopy(causal_map_state_to_dict(state, config))
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        causal_map_state_from_dict(payload, config)


def test_state_validation_rejects_dtype_and_cross_field_corruption() -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation((1, 0, 1)), config, 0)
    with pytest.raises(ValueError, match="last_action must have dtype int32"):
        validate_causal_map_state(
            state._replace(last_action=jnp.asarray(1.0, dtype=jnp.float32)),
            config,
        )
    with pytest.raises(ValueError, match="PRNG implementation"):
        validate_causal_map_state(
            state._replace(rng_key=jr.key(0, impl="rbg")),
            config,
        )
    with pytest.raises(ValueError, match="typed JAX PRNG key"):
        validate_causal_map_state(
            state._replace(rng_key=jr.key_data(state.rng_key)),
            config,
        )
    inconsistent = state._replace(
        cell_collection_step=state.cell_collection_step.at[1, 0].set(0),
    )
    with pytest.raises(ValueError, match="collection timestamps"):
        validate_causal_map_state(inconsistent, config)
    retry_without_collection = state._replace(
        cell_retry_count=state.cell_retry_count.at[1, 0].set(1),
    )
    with pytest.raises(ValueError, match="retry counts require"):
        validate_causal_map_state(retry_without_collection, config)
    inactive_with_active_timestamp = state._replace(
        cell_active=state.cell_active.at[1, 0].set(False),
    )
    with pytest.raises(ValueError, match="cell_active must exactly match"):
        validate_causal_map_state(inactive_with_active_timestamp, config)

    advanced = state
    for _ in range(2):
        advanced, _, _ = causal_map_step(
            advanced,
            jnp.asarray(0.0, dtype=jnp.float32),
            _observation(),
            config,
        )
    inconsistent_populations = advanced._replace(
        reward_sum=advanced.reward_sum.at[0].set(2.0),
        reward_count=advanced.reward_count.at[0].set(2),
        respawn_interval_count=advanced.respawn_interval_count.at[0].set(2),
        respawn_interval_lower_floor=advanced.respawn_interval_lower_floor.at[0].set(1),
        respawn_interval_lower_remainder=(
            advanced.respawn_interval_lower_remainder.at[0].set(1)
        ),
        respawn_interval_upper_floor=advanced.respawn_interval_upper_floor.at[0].set(2),
        respawn_interval_upper_remainder=(
            advanced.respawn_interval_upper_remainder.at[0].set(0)
        ),
        respawn_exact_count=advanced.respawn_exact_count.at[0].set(2),
        respawn_exact_floor=advanced.respawn_exact_floor.at[0].set(1),
        respawn_exact_mean=advanced.respawn_exact_mean.at[0].set(1.0),
        respawn_exact_m2=advanced.respawn_exact_m2.at[0].set(0.0),
    )
    with pytest.raises(ValueError, match="identical rational"):
        validate_causal_map_state(inconsistent_populations, config)
    inconsistent_exact_mean = advanced._replace(
        reward_sum=advanced.reward_sum.at[0].set(2.0),
        reward_count=advanced.reward_count.at[0].set(2),
        respawn_interval_count=advanced.respawn_interval_count.at[0].set(2),
        respawn_interval_lower_floor=advanced.respawn_interval_lower_floor.at[0].set(1),
        respawn_interval_lower_remainder=(
            advanced.respawn_interval_lower_remainder.at[0].set(1)
        ),
        respawn_interval_upper_floor=advanced.respawn_interval_upper_floor.at[0].set(1),
        respawn_interval_upper_remainder=(
            advanced.respawn_interval_upper_remainder.at[0].set(1)
        ),
        respawn_exact_count=advanced.respawn_exact_count.at[0].set(2),
        respawn_exact_floor=advanced.respawn_exact_floor.at[0].set(1),
        respawn_exact_mean=advanced.respawn_exact_mean.at[0].set(1.0),
        respawn_exact_m2=advanced.respawn_exact_m2.at[0].set(0.0),
    )
    with pytest.raises(ValueError, match="exact-sample rational"):
        validate_causal_map_state(inconsistent_exact_mean, config)
    negative_m2 = advanced._replace(
        reward_sum=advanced.reward_sum.at[0].set(2.0),
        reward_count=advanced.reward_count.at[0].set(2),
        respawn_interval_count=advanced.respawn_interval_count.at[0].set(2),
        respawn_interval_lower_floor=advanced.respawn_interval_lower_floor.at[0].set(1),
        respawn_interval_lower_remainder=(
            advanced.respawn_interval_lower_remainder.at[0].set(1)
        ),
        respawn_interval_upper_floor=advanced.respawn_interval_upper_floor.at[0].set(1),
        respawn_interval_upper_remainder=(
            advanced.respawn_interval_upper_remainder.at[0].set(1)
        ),
        respawn_exact_count=advanced.respawn_exact_count.at[0].set(2),
        respawn_exact_floor=advanced.respawn_exact_floor.at[0].set(1),
        respawn_exact_remainder=advanced.respawn_exact_remainder.at[0].set(1),
        respawn_exact_mean=advanced.respawn_exact_mean.at[0].set(1.5),
        respawn_exact_m2=(
            advanced.respawn_exact_m2.at[0].set(-np.finfo(np.float32).tiny)
        ),
    )
    with pytest.raises(ValueError, match="non-negative"):
        validate_causal_map_state(negative_m2, config)


def test_state_validation_binds_selected_action_target_and_causal_counts() -> None:
    config = CausalMapForagerConfig()
    state, _ = causal_map_start(_observation((1, 0, 1)), config, 0)
    with pytest.raises(ValueError, match="destination of last_action"):
        validate_causal_map_state(
            state._replace(
                last_target_position=jnp.asarray((7, 7), dtype=jnp.int32),
            ),
            config,
        )
    with pytest.raises(ValueError, match="destination map cell"):
        validate_causal_map_state(
            state._replace(
                last_target_channel=jnp.asarray(0, dtype=jnp.int32),
            ),
            config,
        )
    with pytest.raises(ValueError, match="destination state"):
        validate_causal_map_state(
            state._replace(
                last_target_expected_active=jnp.asarray(
                    not bool(state.last_target_expected_active),
                ),
            ),
            config,
        )
    with pytest.raises(ValueError, match="visit_count total"):
        validate_causal_map_state(
            state._replace(visit_count=jnp.zeros_like(state.visit_count)),
            config,
        )
    displaced = state._replace(
        position=jnp.asarray((7, 7), dtype=jnp.int32),
        last_target_position=jnp.asarray((8, 7), dtype=jnp.int32),
        last_target_channel=jnp.asarray(-1, dtype=jnp.int32),
        last_target_expected_active=jnp.asarray(False),
    )
    with pytest.raises(ValueError, match="current position"):
        validate_causal_map_state(displaced, config)
    with pytest.raises(ValueError, match="total reward_count"):
        validate_causal_map_state(
            state._replace(
                step_count=jnp.asarray(1, dtype=jnp.int32),
                visit_count=state.visit_count.at[0, 0].set(2),
                reward_count=jnp.ones_like(state.reward_count),
            ),
            config,
        )
    collected, _, _ = causal_map_step(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        _observation(),
        config,
    )
    validate_causal_map_state(collected, config)
    unaccounted_reappearance = collected._replace(
        respawn_interval_count=(
            collected.respawn_interval_count.at[1].set(1)
        ),
        respawn_interval_lower_floor=(
            collected.respawn_interval_lower_floor.at[1].set(1)
        ),
        respawn_interval_upper_floor=(
            collected.respawn_interval_upper_floor.at[1].set(1)
        ),
    )
    with pytest.raises(ValueError, match="reappearances plus pending"):
        validate_causal_map_state(unaccounted_reappearance, config)

    duplicate_pending_timestamp = collected._replace(
        cell_channel=collected.cell_channel.at[5, 5].set(1),
        cell_collection_step=collected.cell_collection_step.at[5, 5].set(1),
        cell_ready_step=collected.cell_ready_step.at[5, 5].set(11),
        cell_last_seen_step=collected.cell_last_seen_step.at[5, 5].set(1),
        cell_last_absent_step=collected.cell_last_absent_step.at[5, 5].set(1),
    )
    with pytest.raises(ValueError, match="pending collection timestamps"):
        validate_causal_map_state(duplicate_pending_timestamp, config)

    advanced, _, _ = causal_map_step(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        _observation(),
        config,
    )
    invalid_m2 = advanced._replace(
        reward_sum=advanced.reward_sum.at[0].set(1.0),
        reward_count=advanced.reward_count.at[0].set(1),
        respawn_interval_count=advanced.respawn_interval_count.at[0].set(1),
        respawn_interval_lower_floor=advanced.respawn_interval_lower_floor.at[0].set(1),
        respawn_interval_upper_floor=advanced.respawn_interval_upper_floor.at[0].set(1),
        respawn_exact_count=advanced.respawn_exact_count.at[0].set(1),
        respawn_exact_floor=advanced.respawn_exact_floor.at[0].set(1),
        respawn_exact_mean=advanced.respawn_exact_mean.at[0].set(1.0),
        respawn_exact_m2=advanced.respawn_exact_m2.at[0].set(100.0),
    )
    with pytest.raises(ValueError, match="M2 must be zero"):
        validate_causal_map_state(invalid_m2, config)


def test_checkpoint_binds_initial_seed_to_agent() -> None:
    source = CausalMapForagerAgent(seed=0)
    source.start(_observation((1, 0, 1)))
    payload = source.state_dict()
    assert payload["fields"]["initial_seed"] == 0
    destination = CausalMapForagerAgent(seed=1)
    with pytest.raises(ValueError, match="initial_seed"):
        destination.load_state_dict(payload)


def test_exact_interval_merge_saturates_count_without_wraparound() -> None:
    maximum = np.iinfo(np.int32).max
    result = _merge_channel_interval_bounds(
        jnp.asarray((maximum,), dtype=jnp.int32),
        jnp.asarray((5,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((7,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((9,), dtype=jnp.int32),
        jnp.asarray((True,)),
    )
    expected = (
        jnp.asarray((maximum,), dtype=jnp.int32),
        jnp.asarray((5,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((7,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(result, expected)


def test_exact_sample_merge_is_stable_canonical_and_chunk_invariant() -> None:
    samples = jnp.asarray(
        1_000_000 + np.arange(81, dtype=np.int32),
        dtype=jnp.int32,
    )
    channels = jnp.zeros((81,), dtype=jnp.int32)
    mask = jnp.ones((81,), dtype=jnp.bool_)
    zero_i = jnp.zeros((1,), dtype=jnp.int32)
    zero_f = jnp.zeros((1,), dtype=jnp.float32)

    def merge(
        count: jax.Array,
        floor: jax.Array,
        remainder: jax.Array,
        mean: jax.Array,
        m2: jax.Array,
        sample_values: jax.Array,
        channel_values: jax.Array,
        sample_mask: jax.Array,
    ) -> tuple[jax.Array, ...]:
        return _merge_exact_channel_samples(
            count,
            floor,
            remainder,
            mean,
            m2,
            channel_values,
            sample_values,
            sample_mask,
        )

    eager = merge(zero_i, zero_i, zero_i, zero_f, zero_f, samples, channels, mask)
    compiled = jax.jit(merge)(
        zero_i,
        zero_i,
        zero_i,
        zero_f,
        zero_f,
        samples,
        channels,
        mask,
    )
    first_chunk = merge(
        zero_i,
        zero_i,
        zero_i,
        zero_f,
        zero_f,
        samples[:40],
        channels[:40],
        mask[:40],
    )
    chunked = merge(
        *first_chunk,
        samples[40:],
        channels[40:],
        mask[40:],
    )
    expected = (
        jnp.asarray((81,), dtype=jnp.int32),
        jnp.asarray((1_000_040,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1_000_040.0,), dtype=jnp.float32),
        jnp.asarray((44_280.0,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(eager, compiled, chunked, expected)

    next_sample = jnp.asarray((1_000_081,), dtype=jnp.int32)
    continued = merge(
        *eager,
        next_sample,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((True,), dtype=jnp.bool_),
    )
    one_shot = merge(
        zero_i,
        zero_i,
        zero_i,
        zero_f,
        zero_f,
        jnp.concatenate((samples, next_sample)),
        jnp.zeros((82,), dtype=jnp.int32),
        jnp.ones((82,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(continued, one_shot)


def test_exact_81_reappearance_production_path_validates_and_continues() -> None:
    config = CausalMapForagerConfig()
    samples = jnp.asarray(
        1_000_000 + np.arange(81, dtype=np.int32),
        dtype=jnp.int32,
    )
    step_count = 1_000_081
    visible_slice = (slice(3, 12), slice(3, 12))
    collection_steps = (step_count - samples).reshape((9, 9))
    state = _empty_state(1, config, 17)
    state = state._replace(
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        position=jnp.asarray((7, 7), dtype=jnp.int32),
        last_action=jnp.asarray(0, dtype=jnp.int32),
        last_target_channel=jnp.asarray(0, dtype=jnp.int32),
        last_target_position=jnp.asarray((7, 8), dtype=jnp.int32),
        last_target_expected_active=jnp.asarray(True),
        cell_channel=state.cell_channel.at[visible_slice].set(0),
        cell_collection_step=(
            state.cell_collection_step.at[visible_slice].set(collection_steps)
        ),
        cell_ready_step=(
            state.cell_ready_step.at[visible_slice].set(collection_steps + 1)
        ),
        cell_last_seen_step=(
            state.cell_last_seen_step.at[visible_slice].set(step_count - 1)
        ),
        cell_last_absent_step=(
            state.cell_last_absent_step.at[visible_slice].set(step_count - 1)
        ),
        visit_count=state.visit_count.at[7, 7].set(step_count + 1),
        reward_sum=state.reward_sum.at[0].set(81.0),
        reward_count=state.reward_count.at[0].set(81),
    )
    observation = jnp.ones((9, 9, 1), dtype=jnp.float32)
    eager_state, eager_learned, eager_misses = _integrate_observation(
        state,
        observation,
        config,
    )
    compiled_state, compiled_learned, compiled_misses = jax.jit(
        lambda value: _integrate_observation(value, observation, config)
    )(state)
    chex.assert_trees_all_equal(
        (eager_state, eager_learned, eager_misses),
        (compiled_state, compiled_learned, compiled_misses),
    )
    assert int(eager_learned) == 81
    assert int(eager_misses) == 0
    assert int(eager_state.respawn_exact_count[0]) == 81
    assert int(eager_state.respawn_exact_floor[0]) == 1_000_040
    assert int(eager_state.respawn_exact_remainder[0]) == 0
    assert float(eager_state.respawn_exact_mean[0]) == 1_000_040.0
    assert float(eager_state.respawn_exact_m2[0]) == 44_280.0
    validate_causal_map_state(eager_state, config)

    restored = causal_map_state_from_dict(
        causal_map_state_to_dict(eager_state, config),
        config,
    )
    collection_observation = observation.at[4, 4, 0].set(0.0)

    def continue_run(value: Any) -> Any:
        value, _, _ = causal_map_step(
            value,
            jnp.asarray(1.0, dtype=jnp.float32),
            collection_observation,
            config,
        )
        validate_causal_map_state(value, config)
        value, _, _ = causal_map_step(
            value,
            jnp.asarray(0.0, dtype=jnp.float32),
            observation,
            config,
        )
        validate_causal_map_state(value, config)
        return value

    uninterrupted = continue_run(eager_state)
    checkpointed = continue_run(restored)
    chex.assert_trees_all_equal(uninterrupted, checkpointed)
    assert int(uninterrupted.respawn_exact_count[0]) == 82
    assert int(uninterrupted.respawn_exact_floor[0]) == 987_844
    assert int(uninterrupted.respawn_exact_remainder[0]) == 33


@pytest.mark.parametrize(
    "reward",
    [
        True,
        math.nan,
        math.inf,
        "1",
        [1.0],
        1.0 + 0.0j,
        np.complex64(1.0),
        1e100,
        10**1_000,
    ],
)
def test_host_agent_rejects_non_scalar_or_nonfinite_reward(reward: Any) -> None:
    config = CausalMapForagerConfig()
    agent = CausalMapForagerAgent(config, seed=0)
    agent.start(_observation((1, 0, 1)))
    with pytest.raises(ValueError, match="finite (?:real|float32) scalar"):
        agent.step(reward, _observation())


def test_pure_transition_is_jittable_finite_and_deterministic() -> None:
    config = CausalMapForagerConfig()
    observation = _observation((1, 0, 2))
    empty = _observation()
    state, action = causal_map_start(observation, config, 11)
    transition = jax.jit(lambda value: causal_map_step(value, 30.0, empty, config))
    first = transition(state)
    second = transition(state)
    chex.assert_trees_all_equal(first, second)
    assert int(action) == 0
    assert int(first[0].step_count) == 1
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(first[0])
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )


def test_vmap_transition_matches_independent_lanes_exactly() -> None:
    config = CausalMapForagerConfig()
    observation = _observation((1, 0, 1), (0, 1, 2))
    seeds = jnp.asarray((3, 7), dtype=jnp.uint32)
    batched_state, batched_action = jax.jit(
        jax.vmap(lambda seed: causal_map_start(observation, config, seed))
    )(seeds)
    batched_next = jax.jit(
        jax.vmap(
            lambda state, reward: causal_map_step(
                state,
                reward,
                _observation(),
                config,
            )
        )
    )(batched_state, jnp.asarray((1.0, -1.0), dtype=jnp.float32))
    for lane, seed in enumerate((3, 7)):
        state, action = causal_map_start(observation, config, seed)
        expected = causal_map_step(
            state,
            jnp.asarray((1.0, -1.0)[lane], dtype=jnp.float32),
            _observation(),
            config,
        )
        chex.assert_trees_all_equal(
            jax.tree_util.tree_map(lambda value: value[lane], batched_state),
            state,
        )
        assert int(batched_action[lane]) == int(action)
        chex.assert_trees_all_equal(
            jax.tree_util.tree_map(lambda value: value[lane], batched_next),
            expected,
        )


def test_agent_metadata_is_explicitly_nonprivileged_and_context_independent() -> None:
    observation = _observation((1, 0, 0))
    ordinary = CausalMapForagerAgent(seed=4)
    with_context = CausalMapForagerAgent(seed=4)
    assert ordinary.start(observation, None) == with_context.start(
        observation,
        cast(Any, object()),
    )
    assert ordinary.step(1.0, _observation(), None) == with_context.step(
        1.0,
        _observation(),
        cast(Any, object()),
    )
    chex.assert_trees_all_equal(ordinary.state, with_context.state)
    metadata = ordinary.metadata()
    assert metadata["privileged"] is False
    assert metadata["state_schema"] == CAUSAL_MAP_STATE_SCHEMA
    assert metadata["prng_impl"] == "threefry2x32"
    assert metadata["jax_threefry_partitionable"] is bool(
        jax.config.jax_threefry_partitionable
    )
    assert "identification interval" in metadata["world_model"]["respawn_model"]
    assert "maximum delay ceiling" in metadata["world_model"]["respawn_model"]
    assert metadata["world_model"]["maximum_map_cells"] == 4096
    assert metadata["nonprivilege_contract"]["context_consumed"] is False
    assert "global position" in metadata["nonprivilege_contract"]["forbidden_inputs"]
    assert "SOTA claim" in metadata["status_claim"]


def _result_signature(result: Any) -> tuple[Any, ...]:
    return (
        result.agent,
        result.privileged,
        result.seed,
        result.steps,
        result.total_reward,
        result.mean_reward,
        result.final_window_mean_reward,
        result.final_ewm_reward,
        result.mean_ewm_reward,
        result.fov_last_10pct_ema_auc,
        result.mean_biome_regret,
        result.final_biome_regret,
        result.curve_steps,
        result.curve_ewm_reward,
        result.curve_window_reward,
        result.environment,
        result.metric_contract,
    )


@pytest.mark.slow
def test_scan_host_chunk_and_batch_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    environment = ForagerEnvConfig.paper_field_of_view()
    agent_config = CausalMapForagerConfig()
    benchmark_a = ForagerBenchmarkConfig(
        environment=environment,
        steps=11,
        seed=3,
        record_every=3,
        final_window=5,
        jax_chunk_size=4,
    )
    benchmark_b = dataclasses.replace(benchmark_a, jax_chunk_size=7)

    scan_agent_a = CausalMapForagerAgent(agent_config, seed=3)
    scan_a = run_forager(scan_agent_a, benchmark_a)
    scan_agent_b = CausalMapForagerAgent(agent_config, seed=3)
    scan_b = run_forager(scan_agent_b, benchmark_b)
    host_agent = _HostCausalMapAgent(agent_config, seed=3)
    host = _run_forager_host(host_agent, benchmark_a)
    left = _result_signature(scan_a)
    right = _result_signature(scan_b)
    assert left[:4] == right[:4]
    np.testing.assert_allclose(left[4:12], right[4:12], rtol=0.0, atol=1e-14)
    assert left[12:] == right[12:]
    host_signature = _result_signature(host)
    assert left[:4] == host_signature[:4]
    np.testing.assert_allclose(left[4:12], host_signature[4:12], rtol=0.0, atol=1e-7)
    assert left[12] == host_signature[12]
    np.testing.assert_allclose(left[13], host_signature[13], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(left[14], host_signature[14], rtol=0.0, atol=1e-7)
    assert left[15:] == host_signature[15:]
    chex.assert_trees_all_equal(scan_agent_a.state, scan_agent_b.state)
    chex.assert_trees_all_equal(scan_agent_a.state, host_agent.state)

    batch_vmap = run_causal_map_forager_seeds(
        agent_config,
        benchmark_a,
        (3, 7),
        mode="vmap",
    )
    batch_strict = run_causal_map_forager_seeds(
        agent_config,
        benchmark_a,
        (3, 7),
        mode="strict",
    )
    for left, right in zip(batch_vmap, batch_strict, strict=True):
        assert _result_signature(left) == _result_signature(right)
    reordered = run_causal_map_forager_seeds(
        agent_config,
        benchmark_a,
        (7, 3),
        mode="vmap",
    )
    signatures_by_seed = {
        result.seed: _result_signature(result) for result in batch_vmap
    }
    for result in reordered:
        assert _result_signature(result) == signatures_by_seed[result.seed]
    assert _result_signature(batch_vmap[0]) == _result_signature(scan_a)
    assert batch_vmap[0].agent_metadata["runner"]["full_reward_history_retained"] is False


def test_causal_runner_environment_rng_is_independent_of_default_prng_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _RngSensitiveCausalFakeForagax()
    monkeypatch.setattr(ForagerEnvConfig, "make", lambda self: (environment, None))
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=7,
        final_window=3,
        jax_chunk_size=3,
    )
    config = CausalMapForagerConfig()
    ordinary_agent = CausalMapForagerAgent(config, seed=0)
    ordinary = run_forager(ordinary_agent, benchmark)
    current_default = str(jax.config.jax_default_prng_impl)
    alternate_default = "rbg" if current_default != "rbg" else "threefry2x32"
    with jax.default_prng_impl(alternate_default):
        alternate_agent = CausalMapForagerAgent(config, seed=0)
        alternate = run_forager(alternate_agent, benchmark)

    assert _result_signature(ordinary) == _result_signature(alternate)
    chex.assert_trees_all_equal(ordinary_agent.state, alternate_agent.state)
    for result in (ordinary, alternate):
        assert result.agent_metadata["environment_prng_impl"] == "threefry2x32"
        assert result.agent_metadata["runner"]["environment_prng_impl"] == "threefry2x32"


def test_runner_validates_final_state_before_trace_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    events: list[str] = []

    def reject_final_state(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        events.append("validate")
        raise ValueError("synthetic invalid final state")

    def record_abort(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        events.append("abort")

    def record_finalize(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        del args, kwargs
        events.append("finalize")
        return ()

    monkeypatch.setattr(causal_map_module, "validate_causal_map_state", reject_final_state)
    monkeypatch.setattr(causal_map_module, "_abort_reward_trace_sinks", record_abort)
    monkeypatch.setattr(causal_map_module, "_finalize_reward_trace_sinks", record_finalize)
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=1,
        final_window=1,
        jax_chunk_size=1,
    )
    with pytest.raises(ValueError, match="synthetic invalid final state"):
        run_forager(CausalMapForagerAgent(seed=0), benchmark)
    assert events == ["validate", "abort"]


@pytest.mark.parametrize(
    "environment",
    (_InvalidResetCausalFakeForagax(), _InvalidStepCausalFakeForagax()),
)
def test_runner_aborts_trace_sinks_on_initialization_or_compiled_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    environment: _CausalFakeForagax,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", lambda self: (environment, None))
    events: list[str] = []
    monkeypatch.setattr(
        causal_map_module,
        "_abort_reward_trace_sinks",
        lambda sinks: events.append("abort"),
    )
    monkeypatch.setattr(
        causal_map_module,
        "_finalize_reward_trace_sinks",
        lambda sinks: events.append("finalize"),
    )
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=1,
        final_window=1,
        jax_chunk_size=1,
    )
    with pytest.raises(Exception, match="observation must be finite"):
        run_causal_map_forager_seeds(
            CausalMapForagerConfig(),
            benchmark,
            (0,),
        )
    assert events == ["abort"]


@pytest.mark.parametrize("failure_kind", ("conversion", "metrics"))
def test_runner_aborts_unsealed_trace_after_host_chunk_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    events: list[str] = []

    class RecordingSink:
        def append(self, rewards: np.ndarray, biome_regrets: np.ndarray) -> None:
            assert rewards.shape == biome_regrets.shape == (1,)
            events.append("append")

        def finalize(self) -> Mapping[str, Any]:
            events.append("finalize")
            return {}

        def abort(self) -> None:
            events.append("abort")

    sink = RecordingSink()
    if failure_kind == "conversion":
        original_conversion = causal_map_module._host_metric_array
        conversion_calls = 0

        def fail_second_chunk_conversion(*args: Any, **kwargs: Any) -> np.ndarray:
            nonlocal conversion_calls
            conversion_calls += 1
            if conversion_calls == 5:
                raise TypeError("synthetic host conversion failure")
            return original_conversion(*args, **kwargs)

        monkeypatch.setattr(
            causal_map_module,
            "_host_metric_array",
            fail_second_chunk_conversion,
        )
        expected_message = "synthetic host conversion failure"
        expected_events = ["append", "abort"]
    else:
        original_add = causal_map_module._LaneMetrics.add
        metric_calls = 0

        def fail_second_metric_update(self: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal metric_calls
            metric_calls += 1
            if metric_calls == 2:
                raise RuntimeError("synthetic metric accumulation failure")
            original_add(self, *args, **kwargs)

        monkeypatch.setattr(causal_map_module._LaneMetrics, "add", fail_second_metric_update)
        expected_message = "synthetic metric accumulation failure"
        expected_events = ["append", "append", "abort"]

    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=2,
        final_window=1,
        jax_chunk_size=1,
    )
    with pytest.raises(Exception, match=expected_message):
        run_causal_map_forager_seeds(
            CausalMapForagerConfig(),
            benchmark,
            (0,),
            reward_trace_sink_factory=lambda seed, steps: sink,
        )
    assert events == expected_events


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("step_count", "step_count does not match requested horizon"),
        ("initial_seed", "initial_seed does not match requested lane"),
    ),
)
def test_runner_binds_final_state_to_requested_horizon_and_lane_before_finalization(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    original_make_scan_chunk = causal_map_module._make_scan_chunk

    def make_corrupting_scan(*args: Any, **kwargs: Any) -> Any:
        scan = original_make_scan_chunk(*args, **kwargs)

        def corrupt(carry: Any, active_steps: jax.Array) -> Any:
            next_carry, outputs = scan(carry, active_steps)
            env_state, env_key, agent_state, action = next_carry
            if field == "step_count":
                agent_state = agent_state._replace(
                    step_count=jnp.asarray(0, dtype=jnp.int32)
                )
            else:
                agent_state = agent_state._replace(
                    initial_seed=agent_state.initial_seed + jnp.asarray(1, jnp.uint32)
                )
            return (env_state, env_key, agent_state, action), outputs

        return corrupt

    monkeypatch.setattr(causal_map_module, "_make_scan_chunk", make_corrupting_scan)
    monkeypatch.setattr(causal_map_module, "validate_causal_map_state", lambda *args: None)
    events: list[str] = []
    monkeypatch.setattr(
        causal_map_module,
        "_abort_reward_trace_sinks",
        lambda sinks: events.append("abort"),
    )
    monkeypatch.setattr(
        causal_map_module,
        "_finalize_reward_trace_sinks",
        lambda sinks: events.append("finalize"),
    )
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=1,
        final_window=1,
        jax_chunk_size=1,
    )
    with pytest.raises(ValueError, match=message):
        run_causal_map_forager_seeds(
            CausalMapForagerConfig(),
            benchmark,
            (0,),
        )
    assert events == ["abort"]


def test_runner_checks_state_bound_threefry_mode_immediately_before_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    events: list[str] = []

    def require_mode(expected: bool) -> None:
        assert expected is bool(jax.config.jax_threefry_partitionable)
        raise RuntimeError("synthetic Threefry mode drift")

    monkeypatch.setattr(
        causal_map_module,
        "_require_host_threefry_mode",
        require_mode,
    )
    monkeypatch.setattr(
        causal_map_module,
        "_abort_reward_trace_sinks",
        lambda sinks: events.append("abort"),
    )

    def finalize(sinks: Any) -> tuple[Any, ...]:
        events.append("finalize")
        return ()

    monkeypatch.setattr(causal_map_module, "_finalize_reward_trace_sinks", finalize)
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=1,
        final_window=1,
        jax_chunk_size=1,
    )
    with pytest.raises(RuntimeError, match="synthetic Threefry mode drift"):
        run_causal_map_forager_seeds(
            CausalMapForagerConfig(),
            benchmark,
            (0,),
        )
    assert events == ["abort"]


def test_finalize_mode_flip_cannot_relabel_state_bound_runner_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    recorded = bool(jax.config.jax_threefry_partitionable)

    def finalize(sinks: Any) -> tuple[Any, ...]:
        jax.config.update("jax_threefry_partitionable", not recorded)
        return ()

    monkeypatch.setattr(causal_map_module, "_finalize_reward_trace_sinks", finalize)
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=1,
        final_window=1,
        jax_chunk_size=1,
    )
    try:
        result = run_causal_map_forager_seeds(
            CausalMapForagerConfig(),
            benchmark,
            (0,),
        )[0]
    finally:
        jax.config.update("jax_threefry_partitionable", recorded)
    assert result.agent_metadata["jax_threefry_partitionable"] is recorded


def test_post_finalize_result_construction_failure_aborts_sinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    events: list[str] = []
    monkeypatch.setattr(
        causal_map_module,
        "_finalize_reward_trace_sinks",
        lambda sinks: events.append("finalize") or (),
    )
    monkeypatch.setattr(
        causal_map_module,
        "_abort_reward_trace_sinks",
        lambda sinks: events.append("abort"),
    )

    def reject_result(**kwargs: Any) -> Any:
        raise RuntimeError("synthetic post-finalize constructor failure")

    monkeypatch.setattr(causal_map_module, "_build_causal_map_result", reject_result)
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=1,
        final_window=1,
        jax_chunk_size=1,
    )
    with pytest.raises(RuntimeError, match="post-finalize constructor failure"):
        run_causal_map_forager_seeds(
            CausalMapForagerConfig(),
            benchmark,
            (0,),
        )
    assert events == ["finalize", "abort"]


@pytest.mark.parametrize(
    "environment",
    [
        ForagerEnvConfig.paper_relearning(),
        ForagerEnvConfig.paper_unending(),
        dataclasses.replace(
            ForagerEnvConfig.paper_field_of_view(),
            reward_delay=1,
        ),
        dataclasses.replace(
            ForagerEnvConfig.paper_field_of_view(),
            observation_type="rgb",
        ),
        dataclasses.replace(
            ForagerEnvConfig.paper_field_of_view(),
            require_exact_version=False,
        ),
        ForagerEnvConfig.paper_field_of_view(aperture_size=1),
    ],
)
def test_runner_rejects_out_of_scope_environment_contract(
    monkeypatch: pytest.MonkeyPatch,
    environment: ForagerEnvConfig,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    benchmark = ForagerBenchmarkConfig(environment=environment, steps=2)
    with pytest.raises(ValueError):
        run_forager(CausalMapForagerAgent(seed=0), benchmark)


def test_runner_preflight_defensively_rejects_even_aperture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    environment = ForagerEnvConfig.paper_field_of_view(aperture_size=3)
    # ForagerEnvConfig itself rejects this at construction.  Corrupt a frozen
    # instance to prove the runner independently fails closed before make().
    object.__setattr__(environment, "aperture_size", 4)
    benchmark = ForagerBenchmarkConfig(environment=environment, steps=2)
    with pytest.raises(ValueError, match="odd centered aperture"):
        run_forager(CausalMapForagerAgent(seed=0), benchmark)


def test_installed_foragax_ready_observation_precedes_collectible_transition() -> None:
    """Public observations/rewards pin the arrival-readiness off-by-one."""
    pytest.importorskip("foragax")
    from foragax.env import Biome, ForagaxEnv
    from foragax.objects import DefaultForagaxObject

    food = DefaultForagaxObject(
        name="deterministic_food",
        reward=1.0,
        collectable=True,
        regen_delay=(2, 2),
        color=(1, 2, 3),
    )
    env = ForagaxEnv(
        size=(1, 1),
        aperture_size=-1,
        objects=(food,),
        biomes=(
            Biome(
                object_frequencies=(1.0,),
                start=(0, 0),
                stop=(1, 1),
            ),
        ),
        deterministic_spawn=True,
        observation_type="color",
    )
    params = env.default_params
    key = jr.key(91)
    key, reset_key = jr.split(key)
    observation, env_state = env.reset(reset_key, params)
    assert float(observation[0, 0, 0]) == 1.0

    rewards: list[float] = []
    visible: list[bool] = []
    for _ in range(5):
        key, step_key = jr.split(key)
        observation, env_state, reward, _, _ = env.step(
            step_key,
            env_state,
            jnp.asarray(0, dtype=jnp.int32),
            params,
        )
        rewards.append(float(reward))
        visible.append(bool(observation[0, 0, 0]))

    # Collection occurs first.  On step four the respawn is visible in the
    # returned public observation, but its transition reward is still zero;
    # only the action selected from that ready state collects on step five.
    assert rewards == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert visible == [False, False, False, True, False]


def test_installed_public_deathcap_is_costly_but_traversable() -> None:
    pytest.importorskip("foragax")
    from foragax.env import Biome, ForagaxEnv
    from foragax.objects import LARGE_DEATHCAP

    assert LARGE_DEATHCAP.collectable is True
    assert LARGE_DEATHCAP.blocking is False
    env = ForagaxEnv(
        size=(3, 1),
        aperture_size=-1,
        objects=(LARGE_DEATHCAP,),
        biomes=(
            Biome(
                object_frequencies=(1.0,),
                start=(0, 0),
                stop=(3, 1),
            ),
        ),
        deterministic_spawn=True,
        observation_type="color",
    )
    params = env.default_params
    key = jr.key(123)
    key, reset_key = jr.split(key)
    _, env_state = env.reset(reset_key, params)
    initial_position = np.asarray(env_state.pos, dtype=np.int32)
    key, step_key = jr.split(key)
    _, next_state, reward, _, _ = env.step(
        step_key,
        env_state,
        jnp.asarray(1, dtype=jnp.int32),
        params,
    )
    expected_position = np.mod(initial_position + np.asarray((1, 0)), (3, 1))
    np.testing.assert_array_equal(np.asarray(next_state.pos), expected_position)
    assert float(reward) == -1.0


def test_installed_foragax_relative_map_projection_matches_public_mechanics() -> None:
    pytest.importorskip("foragax")
    config = CausalMapForagerConfig(exploration_probability=0.0)
    env, params = ForagerEnvConfig.paper_field_of_view(aperture_size=3).make()
    env_key = jr.key(0)
    env_key, reset_key = jr.split(env_key)
    observation, env_state = env.reset(reset_key, params)
    origin = np.asarray(env_state.pos, dtype=np.int32)
    state, action = causal_map_start(observation, config, 0)
    step_function = jax.jit(env.step)

    for _ in range(5):
        env_key, step_key = jr.split(env_key)
        observation, env_state, reward, done, _ = step_function(
            step_key,
            env_state,
            action,
            params,
        )
        state, action, _ = causal_map_step(state, reward, observation, config)
        actual_relative = np.mod(
            np.asarray(env_state.pos, dtype=np.int32) - origin,
            np.asarray((config.width, config.height), dtype=np.int32),
        )
        np.testing.assert_array_equal(np.asarray(state.position), actual_relative)

        image = np.asarray(observation)
        active = image.sum(axis=-1) > 0.5
        channels = image.argmax(axis=-1)
        radius = image.shape[0] // 2
        position = np.asarray(state.position)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                x = (int(position[0]) + col - radius) % config.width
                y = (int(position[1]) + row - radius) % config.height
                assert bool(state.cell_active[y, x]) == bool(active[row, col])
                if active[row, col]:
                    assert int(state.cell_channel[y, x]) == int(channels[row, col])
        assert not bool(done)


def test_installed_foragax_batch_modes_and_seed_order_are_invariant() -> None:
    pytest.importorskip("foragax")
    config = CausalMapForagerConfig()
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=3,
        seed=0,
        record_every=2,
        final_window=3,
        jax_chunk_size=2,
    )
    batched = run_causal_map_forager_seeds(
        config,
        benchmark,
        (0, 1),
        mode="vmap",
    )
    strict = run_causal_map_forager_seeds(
        config,
        benchmark,
        (0, 1),
        mode="strict",
    )
    reordered = run_causal_map_forager_seeds(
        config,
        benchmark,
        (1, 0),
        mode="vmap",
    )
    expected = {result.seed: _result_signature(result) for result in batched}
    assert {
        result.seed: _result_signature(result) for result in strict
    } == expected
    assert {
        result.seed: _result_signature(result) for result in reordered
    } == expected


def test_seed_batch_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    benchmark = ForagerBenchmarkConfig(
        environment=ForagerEnvConfig.paper_field_of_view(),
        steps=2,
    )
    config = CausalMapForagerConfig()
    with pytest.raises(ValueError, match="non-empty"):
        run_causal_map_forager_seeds(config, benchmark, ())
    with pytest.raises(ValueError, match="unique"):
        run_causal_map_forager_seeds(config, benchmark, (1, 1))
    with pytest.raises(ValueError, match="uint32"):
        run_causal_map_forager_seeds(config, benchmark, (-1,))
    for invalid in (True, 1.5, "1"):
        with pytest.raises(ValueError, match="without coercion"):
            run_causal_map_forager_seeds(
                config,
                benchmark,
                cast(Any, (invalid,)),
            )
    with pytest.raises(ValueError, match="mode"):
        run_causal_map_forager_seeds(config, benchmark, (1,), mode=cast(Any, "bad"))
    with pytest.raises(ValueError, match="steps must be an integer"):
        dataclasses.replace(benchmark, steps=cast(Any, True))
    unsafe = dataclasses.replace(benchmark, steps=np.iinfo(np.int32).max)
    with pytest.raises(ValueError, match="steps must be"):
        run_causal_map_forager_seeds(config, unsafe, (1,))


# ---------------------------------------------------------------------------
# Architecture-independent respawn quantile
# ---------------------------------------------------------------------------

# The exact ``respawn_quantile_z`` for every quantile the published
# matched-current protocol uses.  These are the values the sealed campaign was
# built from -- they appear in plaintext in the sealed configuration payloads
# under ``outputs/forager/matched_current_qualification_2c3b214c_v1`` -- and are
# pinned in hex so that a one-ULP drift cannot hide behind repr rounding.
_REFERENCE_RESPAWN_QUANTILE_Z: Final = {
    0.5: "0x0.0p+0",
    0.75: "0x1.5956b87528a49p-1",
    0.9: "0x1.4813c36e26d34p+0",
}


@pytest.mark.parametrize("quantile", sorted(_REFERENCE_RESPAWN_QUANTILE_Z))
def test_respawn_quantile_z_matches_the_published_reference_bits(
    quantile: float,
) -> None:
    """``respawn_quantile_z`` must be bit-exact on every host architecture.

    The value feeds ``fingerprint()``, hence ``protocol_sha256`` and
    ``schedule_sha256``, so a single ULP of drift changes the published
    identity of a sealed campaign.  ``statistics.NormalDist().inv_cdf`` is a C
    accelerator compiled with floating-point contraction enabled, which makes
    that identity depend on whether the host has a fused multiply-add.  Pin the
    bits, not the repr.
    """

    config = CausalMapForagerConfig(respawn_safety_quantile=quantile)
    assert config.respawn_quantile_z.hex() == _REFERENCE_RESPAWN_QUANTILE_Z[quantile]


def test_respawn_quantile_z_is_not_the_contracted_neighbour() -> None:
    """Pin the drift direction, so a near-miss cannot pass as close enough.

    A contracted evaluation lands on the adjacent representable double rather
    than somewhere far away, so any assertion phrased as closeness would accept
    the defect.  Name the wrong value explicitly instead.
    """

    reference = float.fromhex(_REFERENCE_RESPAWN_QUANTILE_Z[0.9])
    contracted = math.nextafter(reference, 0.0)
    assert reference != contracted

    observed = CausalMapForagerConfig(respawn_safety_quantile=0.9).respawn_quantile_z
    assert observed == reference
    assert observed != contracted


@pytest.mark.parametrize(
    "quantile",
    [
        0.5,
        0.6,
        0.6886660381825473,  # measured worst-case divergence, 9 ULP
        0.75,
        0.837392,
        0.9,
        0.925,
        0.95,
        0.99,
        0.999,
        0.999999,
        1.0 - 2.0**-40,
    ],
)
def test_deterministic_inv_cdf_tracks_the_standard_library_structurally(
    quantile: float,
) -> None:
    """Catch structural divergence across the whole admissible domain.

    ``respawn_safety_quantile`` is validated into ``[0.5, 1.0)``, so the AS241
    tail branches are reachable even though the published protocol only uses
    the central one.  This holds the reimplementation to the shape of the
    reference across that range -- a wrong branch cut, a transposed
    numerator/denominator, or a mistyped exponent all show up here.

    The tolerance is deliberately loose and is calibrated, not guessed: the
    accelerator contracts and this implementation does not, so the two
    legitimately disagree.  Enumerating five million consecutive doubles
    through the worst region puts the maximum at **9 ULP** (at ``p =
    0.6886660381825473``; the divergence histogram there is
    ``{0: 1769290, ..., 7: 3272, 8: 315, 9: 4}``), so the bound below carries
    the project's ``>=2x`` margin over the measured worst case.  A tight bound
    is therefore impossible against this reference, which is exactly why
    fine-grained coefficient integrity is pinned separately in
    :func:`test_deterministic_inv_cdf_coefficients_are_pinned` rather than
    inferred from agreement here.
    """

    observed = causal_map_module._deterministic_inv_cdf(quantile)  # noqa: SLF001
    expected = NormalDist().inv_cdf(quantile)
    assert observed == pytest.approx(expected, abs=20.0 * math.ulp(abs(expected) or 1.0))


def test_deterministic_inv_cdf_negates_below_the_median() -> None:
    """Cover the ``q < 0`` sign flip, which no other test reaches.

    This pins branch structure, not the coefficients.  Oddness holds
    bit-exactly for *any* coefficient values, so it cannot detect a
    transcription error -- on the central branch it follows from the trailing
    ``* q`` factor, and on the tail branches from the explicit ``x = -x`` sign
    flip over a ``p``-symmetric intermediate.  Only the ``0.99``/``0.01`` pair
    below reaches that flip; it is here because the negative-``q`` path is
    otherwise unexecuted.
    """

    for quantile in (0.6, 0.75, 0.9, 0.99):
        upper = causal_map_module._deterministic_inv_cdf(quantile)  # noqa: SLF001
        lower = causal_map_module._deterministic_inv_cdf(1.0 - quantile)  # noqa: SLF001
        assert lower == -upper


_AS241_SOURCE_CONSTANTS: Final = (
    0.5,
    0.425,
    5.0,
    0.0,
    0.180625,
    1.0,
    0.0,
    1.0,
    1.6,
    1.4234371107496835,
    1.0,
    5.0,
    6.657904643501103,
    1.0,
    3.3871328727963665,
    42.31333070160091,
    4.630337846156546,
    2.053191626637759,
    5.463784911164114,
    0.599832206555888,
    133.14166789178438,
    687.1870074920579,
    5.769497221460691,
    1.6763848301838038,
    1.7848265399172913,
    0.1369298809227358,
    1971.5909503065513,
    5394.196021424751,
    3.6478483247632045,
    0.6897673349851,
    0.29656057182850487,
    0.014875361290850615,
    13731.69376550946,
    21213.794301586597,
    1.2704582524523684,
    0.14810397642748008,
    0.026532189526576124,
    0.0007868691311456133,
    45921.95393154987,
    39307.89580009271,
    0.2417807251774506,
    0.015198666563616457,
    0.0012426609473880784,
    1.8463183175100548e-05,
    67265.7709270087,
    28729.085735721943,
    0.022723844989269184,
    0.0005475938084995345,
    2.7115555687434876e-05,
    1.421511758316446e-07,
    33430.57558358813,
    5226.495278852854,
    0.0007745450142783414,
    1.0507500716444169e-09,
    2.0103343992922881e-07,
    2.0442631033899397e-15,
    2509.0809287301227,
)


def test_deterministic_inv_cdf_coefficients_are_pinned() -> None:
    """Freeze every literal in the quantile, to the bit.

    The differential test above cannot resolve small coefficient drift,
    because the accelerator it compares against is itself up to 7 ULP away.
    Most single-ULP perturbations of these constants do change the function's
    output somewhere in ``[0.5, 1.0)`` -- the exact count rises with probe
    density, so no single figure is meaningful -- and every one of them stays
    inside any tolerance loose enough to accommodate the accelerator.  So
    agreement with the standard library is not sufficient evidence that the
    transcription is intact.

    Read the literals straight out of the source and compare them against a
    frozen tuple.  The values were established by fuzzing this implementation
    against CPython's own pure-Python ``_normal_dist_inv_cdf`` over 1,999,999
    points with zero mismatches; this test's job is to keep them that way.
    """

    source = textwrap.dedent(inspect.getsource(causal_map_module._deterministic_inv_cdf))  # noqa: SLF001
    observed = tuple(
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    )
    assert observed == _AS241_SOURCE_CONSTANTS

def test_causal_map_forager_config_accepts_all_numpy_integer_families() -> None:
    from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig
    for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"):
        dtype_type = np.dtype(code).type
        cfg = CausalMapForagerConfig(initial_retry_delay=dtype_type(1))
        assert cfg.initial_retry_delay == 1
