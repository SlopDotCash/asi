"""Tests for the Forager/Foragax benchmark integration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.official_foragax as official_foragax_module
from alberta_framework.benchmarks.forager import (
    FORAGAX_INSTALL_TREE_SHA256,
    AlbertaForagerAgent,
    AlbertaForagerConfig,
    ForagerAgentContext,
    ForagerBenchmarkConfig,
    ForagerEnvConfig,
    ForagerFeatureConfig,
    ForagerFeatureEncoder,
    ForagerRunResult,
    OracleSearchForagerAgent,
    RandomForagerAgent,
    _agent_key,
    bootstrap_mean_interval,
    foragax_install_tree_sha256,
    forager_metric_contract,
    paper_baselines,
    paper_protocol,
    paper_reference_targets,
    run_alberta_forager_seeds,
    run_forager,
    summarize_forager_runs,
)
from alberta_framework.benchmarks.forager_results import (
    OfficialForagaxRunSpec,
    build_forager_comparison_report,
    import_official_foragax_npz,
    paired_forager_comparison,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


class _FakeForagax:
    """Small Gymnax-like continuing environment usable inside ``lax.scan``."""

    default_params = None

    def reset(self, key: Any, params: Any) -> tuple[Any, Any]:
        del key, params
        return jnp.zeros((3, 3, 2), dtype=jnp.float32), jnp.asarray(0, jnp.int32)

    def step(
        self,
        key: Any,
        state: Any,
        action: Any,
        params: Any,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        del key, params
        next_state = state + jnp.asarray(1, dtype=jnp.int32)
        reward = jnp.where(
            action == (state % 4),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(-0.25, dtype=jnp.float32),
        )
        observation = jnp.full(
            (3, 3, 2),
            next_state.astype(jnp.float32) / 10.0,
            dtype=jnp.float32,
        )
        info = {"biome_regret": jnp.abs(reward)}
        return observation, next_state, reward, jnp.asarray(False), info


class _TrackingPolicy:
    def __init__(self, *, privileged: bool = False) -> None:
        self._privileged = privileged
        self.contexts: list[ForagerAgentContext | None] = []

    @property
    def name(self) -> str:
        return "tracking"

    @property
    def privileged(self) -> bool:
        return self._privileged

    def start(
        self,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        del observation
        self.contexts.append(context)
        return 0

    def step(
        self,
        reward: float,
        observation: Any,
        context: ForagerAgentContext | None = None,
    ) -> int:
        del reward, observation
        self.contexts.append(context)
        return 0

    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "privileged": self.privileged,
            "config": {"kind": "tracking"},
        }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"aperture_size": True}, "aperture_size"),
        ({"aperture_size": 3.0}, "aperture_size"),
        ({"reward_delay": True}, "reward_delay must be an integer"),
        ({"random_shift_max_steps": 1.5}, "must be an integer"),
        ({"require_exact_version": 1}, "must be a boolean"),
        ({"env_id": False}, "env_id"),
        (
            {"extra_kwargs": {"reward_delay": 4}},
            "duplicates explicit environment fields",
        ),
        ({"extra_kwargs": {"bad": math.nan}}, "finite JSON"),
    ],
)
def test_environment_config_rejects_lossy_or_ambiguous_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ForagerEnvConfig(**kwargs)


def test_environment_config_defensively_copies_extra_kwargs() -> None:
    supplied = {"return_hint": True, "nested": {"value": 1}}
    config = ForagerEnvConfig(extra_kwargs=supplied)
    supplied["nested"]["value"] = 9

    assert config.extra_kwargs == {
        "nested": {"value": 1},
        "return_hint": True,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_hint": 1},
        {"reward_trace_decays": (True,)},
        {"reward_trace_decays": [0.9]},
        {"reward_trace_decays": (1e100,)},
        {"reward_scale": True},
        {"reward_scale": 1e-100},
    ],
)
def test_feature_config_rejects_boolean_numeric_coercion(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ForagerFeatureConfig(**kwargs)


def test_feature_encoder_rejects_falsey_nonconfig_instead_of_defaulting() -> None:
    with pytest.raises(TypeError, match="ForagerFeatureConfig"):
        ForagerFeatureEncoder({})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": True},
        {"actor_epsilon": False},
        {"temperature": True},
        {"sparsity": False},
        {"td_error_normalizer_decay": True},
        {"td_error_clip": True},
        {"recurrent_input_scale": True},
        {"recurrent_scale": False},
        {"recurrent_update_bias": True},
        {"actor_hidden_sizes": [64]},
        {"temperature": 1e-100},
        {"recurrent_update_bias": 1e100},
        {"features": {}},
    ],
)
def test_alberta_config_rejects_boolean_numeric_coercion(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        AlbertaForagerConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 1.5},
        {"seed": True},
        {"seed": 1.5},
        {"seed": "1"},
        {"seed": 2**31},
        {"ewm_decay": True},
        {"record_every": True},
        {"final_window": 1.5},
        {"jax_chunk_size": False},
    ],
)
def test_benchmark_config_rejects_lossy_numeric_coercion(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ForagerBenchmarkConfig(**kwargs)


class _SpoofedFloat:
    """Mimics ``float`` via ``__class__`` to defeat ``isinstance`` checks."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return 0.5

    def __le__(self, other: float) -> bool:
        return 0.5 <= other

    def __lt__(self, other: float) -> bool:
        return 0.5 < other


def test_benchmark_config_rejects_class_spoofed_ewm_decay() -> None:
    with pytest.raises(ValueError, match="ewm_decay"):
        ForagerBenchmarkConfig(ewm_decay=_SpoofedFloat())


def test_benchmark_config_accepts_numpy_float64_ewm_decay() -> None:
    config = ForagerBenchmarkConfig(ewm_decay=np.float64(0.5))
    assert config.ewm_decay == 0.5
    assert type(config.ewm_decay) is float


class _FloatSubclass(float):
    """A real float subtype whose later arithmetic remains user-controlled."""

    def __mul__(self, other: object) -> float:
        raise AssertionError("custom arithmetic must never reach a benchmark")


def test_benchmark_config_rejects_user_defined_float_subclass() -> None:
    with pytest.raises(ValueError, match="ewm_decay"):
        ForagerBenchmarkConfig(ewm_decay=_FloatSubclass(0.5))


def test_benchmark_config_rejects_unrepresentable_builtin_integer() -> None:
    with pytest.raises(ValueError, match="ewm_decay must be finite"):
        ForagerBenchmarkConfig(ewm_decay=10**10000)


def test_benchmark_chunk_is_bounded_by_requested_lifetime() -> None:
    config = ForagerBenchmarkConfig(steps=7, jax_chunk_size=10_000)

    assert config.jax_chunk_size == 7
    assert config.to_dict()["jax_chunk_size"] == 7


@pytest.mark.parametrize("seed", [True, 1.9, "1", 2**31])
def test_all_in_tree_agents_reject_noncanonical_seeds(seed: Any) -> None:
    with pytest.raises(ValueError):
        AlbertaForagerAgent(seed=seed)
    with pytest.raises(ValueError):
        RandomForagerAgent(seed=seed)
    with pytest.raises(ValueError):
        OracleSearchForagerAgent(seed=seed)


@pytest.mark.parametrize("seed", [True, 1.9, "1", 2**31])
def test_batched_runner_rejects_noncanonical_seeds_before_environment(
    seed: Any,
) -> None:
    with pytest.raises(ValueError):
        run_alberta_forager_seeds(
            AlbertaForagerConfig(),
            ForagerBenchmarkConfig(steps=1),
            (seed,),
        )


def test_bootstrap_rejects_boolean_counts_and_seed() -> None:
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_mean_interval((1.0, 2.0), resamples=True)
    with pytest.raises(ValueError, match="bootstrap seed"):
        bootstrap_mean_interval((1.0, 2.0), seed=True)


class _HostAlberta(AlbertaForagerAgent):
    """Subclass forces the public host lifecycle instead of the exact-type scan."""


def _fake_make(self: ForagerEnvConfig) -> tuple[_FakeForagax, None]:
    del self
    return _FakeForagax(), None


def _result(
    agent: str,
    seed: int,
    value: float,
    *,
    config_id: str = "same",
    ewm_decay: float = 0.9,
) -> ForagerRunResult:
    return ForagerRunResult(
        agent=agent,
        privileged=False,
        seed=seed,
        steps=10,
        total_reward=value * 10,
        mean_reward=value,
        final_window_mean_reward=value,
        final_ewm_reward=value,
        mean_ewm_reward=value,
        fov_last_10pct_ema_auc=value,
        mean_biome_regret=0.0,
        final_biome_regret=0.0,
        curve_steps=(1, 10),
        curve_ewm_reward=(value, value),
        curve_window_reward=(value, value),
        duration_s=1.0,
        frames_per_second=10.0,
        environment={"env_id": "fake", "aperture_size": 9},
        metric_contract=forager_metric_contract(
            ewm_decay=ewm_decay,
            final_window=10,
            record_every=10,
            steps=10,
        ),
        agent_metadata={"config": {"id": config_id}},
    )


def _official_environment_provenance(
    environment: ForagerEnvConfig,
    *,
    install_tree_sha256: str = "a" * 64,
) -> dict[str, Any]:
    return {
        "semantic": {
            "preset": environment.preset,
            "env_id": environment.resolved_env_id,
            "aperture_size": environment.aperture_size,
            "observation_type": environment.resolved_observation_type,
            "reward_delay": environment.reward_delay,
            "random_shift_max_steps": environment.random_shift_max_steps,
            "extra_kwargs": dict(environment.extra_kwargs),
        },
        "implementation": {
            "distribution": "continual-foragax",
            "package": "foragax",
            "version": "0.55.0",
            "direct_url": {
                "url": "https://example.invalid/continual-foragax.whl",
                "archive_info": {"hash": "sha256=" + "b" * 64},
            },
            "install_tree_hash_scheme": "relative-path+size+bytes-v1",
            "install_tree_sha256": install_tree_sha256,
        },
    }


def _current_foragax_result(
    value: float,
    *,
    install_tree_sha256: str,
    schedule: str = "dedicated_environment_split_chain_v1",
    self_declared_runtime_profile: bool = False,
) -> ForagerRunResult:
    base = _result("alberta_horde_ac", 0, value)
    environment = {
        "preset": "relearning",
        "env_id": "ForagaxSquareWaveTwoBiome-v11",
        "aperture_size": 9,
        "observation_type": "color",
        "reward_delay": 0,
        "random_shift_max_steps": 0,
        "extra_kwargs": {},
        "require_exact_version": True,
        "foragax_distribution": "continual-foragax",
        "foragax_required_version": "0.55.0",
        "foragax_installed_version": "0.55.0",
        "foragax_install_tree_hash_scheme": "relative-path+size+bytes-v1",
        "foragax_expected_install_tree_sha256": install_tree_sha256,
        "foragax_installed_tree_sha256": install_tree_sha256,
    }
    schedule_sha256 = hashlib.sha256(
        json.dumps(
            {
                "schema_version": "alberta.environment_rng_schedule.v1",
                "identity": schedule,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    metadata = {
        "name": "alberta_horde_ac",
        "seed": 0,
        "config": {"id": "matched-current-runtime"},
        "environment_rng_schedule": schedule,
        "environment_rng_schedule_sha256": schedule_sha256,
    }
    if self_declared_runtime_profile:
        metadata.update(
            {
                "environment_runtime_profile_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "schema_version": (
                                "test.environment_runtime_profile.v1"
                            ),
                            "install_tree_sha256": install_tree_sha256,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "runtime_profile_id": "test-immutable-runtime-v1",
            }
        )
    return dataclasses.replace(
        base,
        environment=environment,
        agent_metadata=metadata,
    )


def test_feature_encoder_is_causal_and_handles_dict_observation() -> None:
    config = ForagerFeatureConfig(
        reward_trace_decays=(0.5, 0.9),
        reward_scale=2.0,
    )
    encoder = ForagerFeatureEncoder(config)
    image = jnp.arange(12, dtype=jnp.float32).reshape((2, 2, 3)) / 12
    observation = {"image": image, "hint": jnp.asarray([0.0, 1.0])}

    initial = encoder.init()
    encoded = encoder.encode(observation, initial)
    assert encoded.shape == (12 + 3 + 2 + 4 + 1 + 2,)
    np.testing.assert_allclose(encoded[-7:-3], np.zeros((4,)))

    advanced = encoder.advance(initial, action=2, reward=2.0)
    assert advanced.last_action == 2
    assert advanced.last_reward == pytest.approx(1.0)
    assert advanced.reward_traces == pytest.approx((0.5, 0.1))
    next_encoded = encoder.encode(observation, advanced)
    np.testing.assert_allclose(next_encoded[-7:-3], [0.0, 0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="reward_trace_decays"):
        ForagerFeatureConfig(reward_trace_decays=(math.nan,))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("actor_hidden_sizes", (64.0,), "positive integer widths"),
        ("critic_hidden_sizes", (0,), "positive integer widths"),
        ("gamma", math.nan, "gamma must be finite"),
        ("actor_lamda", -0.1, "actor_lamda"),
        ("critic_lamda", 1.1, "critic_lamda"),
        ("actor_epsilon", math.inf, "actor_epsilon must be finite"),
        ("td_error_normalizer_decay", 1.0, "td_error_normalizer_decay"),
        ("td_error_clip", math.nan, "td_error_clip"),
        ("actor_gradient_clip_norm", 0.0, "actor_gradient_clip_norm"),
        ("freeze_after_steps", 1.5, "non-negative integer"),
    ],
)
def test_alberta_forager_config_rejects_invalid_values(
    field_name: str,
    value: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AlbertaForagerConfig(**{field_name: value})


def test_agent_rng_namespace_is_disjoint_from_environment() -> None:
    for seed in range(4):
        _, environment_subkey = jr.split(jr.key(seed))
        _, agent_subkey = jr.split(_agent_key(seed))
        assert not np.array_equal(
            np.asarray(jr.key_data(environment_subkey)),
            np.asarray(jr.key_data(agent_subkey)),
        )


def test_host_runner_withholds_privileged_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    config = ForagerBenchmarkConfig(
        steps=4,
        record_every=2,
        final_window=2,
    )
    ordinary = _TrackingPolicy(privileged=False)
    ordinary_result = run_forager(ordinary, config)
    assert all(context is None for context in ordinary.contexts)
    assert ordinary_result.steps == 4
    assert math.isfinite(ordinary_result.mean_ewm_reward)

    privileged = _TrackingPolicy(privileged=True)
    run_forager(privileged, config)
    assert all(context is not None for context in privileged.contexts)


@pytest.mark.parametrize("action", [True, 1.9, "1", [1]])
def test_host_runner_rejects_noninteger_actions_before_coercion(
    action: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)

    class BadActionPolicy(_TrackingPolicy):
        def start(
            self,
            observation: Any,
            context: ForagerAgentContext | None = None,
        ) -> Any:
            del observation, context
            return action

    with pytest.raises(ValueError, match="scalar integers"):
        run_forager(
            BadActionPolicy(),
            ForagerBenchmarkConfig(steps=1),
        )


def test_in_tree_policy_seed_must_match_environment_seed() -> None:
    with pytest.raises(ValueError, match="policy seed"):
        run_forager(
            RandomForagerAgent(seed=1),
            ForagerBenchmarkConfig(steps=1, seed=2),
        )


def test_run_forager_rejects_falsey_nonconfig_instead_of_defaulting() -> None:
    with pytest.raises(TypeError, match="ForagerBenchmarkConfig"):
        run_forager(_TrackingPolicy(), {})  # type: ignore[arg-type]


def test_host_runner_freezes_policy_identity_and_metadata_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)

    class ChangingPrivilegePolicy(_TrackingPolicy):
        def step(
            self,
            reward: float,
            observation: Any,
            context: ForagerAgentContext | None = None,
        ) -> int:
            self._privileged = True
            return super().step(reward, observation, context)

    with pytest.raises(ValueError, match="changed during the run"):
        run_forager(
            ChangingPrivilegePolicy(),
            ForagerBenchmarkConfig(steps=1),
        )

    class NonJsonMetadataPolicy(_TrackingPolicy):
        def metadata(self) -> Mapping[str, Any]:
            return {"name": self.name, "config": {"bad": math.nan}}

    with pytest.raises(ValueError, match="finite JSON"):
        run_forager(
            NonJsonMetadataPolicy(),
            ForagerBenchmarkConfig(steps=1),
        )


def test_compiled_alberta_and_random_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    benchmark = ForagerBenchmarkConfig(
        steps=7,
        seed=3,
        record_every=3,
        final_window=4,
        jax_chunk_size=3,
    )
    alberta = AlbertaForagerAgent(
        AlbertaForagerConfig(
            actor_hidden_sizes=(4,),
            critic_hidden_sizes=(4,),
            features=ForagerFeatureConfig(reward_trace_decays=(0.9,)),
        ),
        seed=3,
    )
    alberta_result = run_forager(alberta, benchmark)
    random_result = run_forager(RandomForagerAgent(seed=3), benchmark)

    assert alberta_result.curve_steps == (1, 3, 6, 7)
    assert alberta_result.agent_metadata["runner"]["kind"] == "jax_scan"
    assert random_result.agent_metadata["runner"]["kind"] == "jax_scan"
    assert alberta._updates == 7
    assert 0 <= alberta._feature_state.last_action < 4
    assert math.isfinite(alberta.last_td_error)
    assert math.isfinite(alberta_result.mean_ewm_reward)


def test_batched_alberta_runner_matches_independent_seed_trajectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    agent_config = AlbertaForagerConfig(
        actor_hidden_sizes=(4,),
        critic_hidden_sizes=(4,),
        features=ForagerFeatureConfig(reward_trace_decays=(0.5,)),
    )
    benchmark = ForagerBenchmarkConfig(
        steps=7,
        record_every=3,
        final_window=4,
        jax_chunk_size=4,
    )
    seeds = (3, 7)

    batched = run_alberta_forager_seeds(
        agent_config,
        benchmark,
        seeds,
        mode="vmap",
    )
    independent = tuple(
        run_forager(
            AlbertaForagerAgent(agent_config, seed=seed),
            benchmark.with_seed(seed),
        )
        for seed in seeds
    )

    assert tuple(run.seed for run in batched) == seeds
    for batch_run, single_run in zip(batched, independent):
        assert batch_run.total_reward == single_run.total_reward
        assert batch_run.curve_steps == single_run.curve_steps
        np.testing.assert_allclose(
            batch_run.curve_ewm_reward,
            single_run.curve_ewm_reward,
        )
        np.testing.assert_allclose(
            batch_run.curve_window_reward,
            single_run.curve_window_reward,
        )
        assert batch_run.fov_last_10pct_ema_auc == pytest.approx(single_run.fov_last_10pct_ema_auc)
        assert batch_run.agent_metadata["runner"]["kind"] == "jax_batched_scan"


def test_alberta_host_and_scan_lifecycles_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    agent_config = AlbertaForagerConfig(
        actor_hidden_sizes=(4,),
        critic_hidden_sizes=(4,),
        features=ForagerFeatureConfig(reward_trace_decays=(0.5, 0.9)),
    )
    benchmark = ForagerBenchmarkConfig(
        steps=10,
        seed=7,
        record_every=2,
        final_window=4,
        jax_chunk_size=4,
    )
    host_agent = _HostAlberta(agent_config, seed=7)
    scan_agent = AlbertaForagerAgent(agent_config, seed=7)
    host_result = run_forager(host_agent, benchmark)
    scan_result = run_forager(scan_agent, benchmark)

    assert host_result.total_reward == scan_result.total_reward
    np.testing.assert_allclose(
        host_result.curve_ewm_reward,
        scan_result.curve_ewm_reward,
        rtol=1e-6,
        atol=1e-7,
    )
    assert host_agent._last_action == scan_agent._last_action
    assert host_agent._feature_state.last_action == scan_agent._feature_state.last_action
    np.testing.assert_allclose(
        host_agent._feature_state.reward_traces,
        scan_agent._feature_state.reward_traces,
        rtol=1e-6,
        atol=1e-7,
    )
    assert host_agent.last_td_error == pytest.approx(
        scan_agent.last_td_error,
        rel=1e-6,
        abs=1e-7,
    )


def test_alberta_host_and_scan_match_across_freeze_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    agent_config = AlbertaForagerConfig(
        actor_hidden_sizes=(4,),
        critic_hidden_sizes=(4,),
        freeze_after_steps=3,
        features=ForagerFeatureConfig(reward_trace_decays=(0.5, 0.9)),
    )
    benchmark = ForagerBenchmarkConfig(
        steps=8,
        seed=11,
        record_every=2,
        final_window=4,
        jax_chunk_size=3,
    )
    host_agent = _HostAlberta(agent_config, seed=11)
    scan_agent = AlbertaForagerAgent(agent_config, seed=11)

    host_result = run_forager(host_agent, benchmark)
    scan_result = run_forager(scan_agent, benchmark)

    assert host_result.total_reward == scan_result.total_reward
    assert host_agent._updates == scan_agent._updates == 3
    assert host_agent._last_action == scan_agent._last_action
    np.testing.assert_allclose(
        host_agent._feature_state.reward_traces,
        scan_agent._feature_state.reward_traces,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        jr.key_data(host_agent._state.rng_key),
        jr.key_data(scan_agent._state.rng_key),
    )
    chex.assert_trees_all_close(
        host_agent._state.replace(rng_key=jnp.zeros((2,), dtype=jnp.uint32)),
        scan_agent._state.replace(rng_key=jnp.zeros((2,), dtype=jnp.uint32)),
        rtol=1e-6,
        atol=1e-7,
    )


def test_alberta_scan_is_chunk_size_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    agent_config = AlbertaForagerConfig(
        actor_hidden_sizes=(4,),
        critic_hidden_sizes=(4,),
        features=ForagerFeatureConfig(reward_trace_decays=(0.5,)),
    )
    base = ForagerBenchmarkConfig(
        steps=11,
        seed=13,
        record_every=3,
        final_window=5,
        jax_chunk_size=3,
    )
    first_agent = AlbertaForagerAgent(agent_config, seed=13)
    second_agent = AlbertaForagerAgent(agent_config, seed=13)

    first = run_forager(first_agent, base)
    second = run_forager(
        second_agent,
        dataclasses.replace(base, jax_chunk_size=7),
    )

    assert first.total_reward == second.total_reward
    assert first.curve_steps == second.curve_steps
    np.testing.assert_allclose(first.curve_ewm_reward, second.curve_ewm_reward)
    np.testing.assert_allclose(first.curve_window_reward, second.curve_window_reward)
    assert first.fov_last_10pct_ema_auc == pytest.approx(second.fov_last_10pct_ema_auc)
    np.testing.assert_array_equal(
        jr.key_data(first_agent._state.rng_key),
        jr.key_data(second_agent._state.rng_key),
    )
    chex.assert_trees_all_close(
        first_agent._state.replace(rng_key=jnp.zeros((2,), dtype=jnp.uint32)),
        second_agent._state.replace(rng_key=jnp.zeros((2,), dtype=jnp.uint32)),
    )


def test_official_npz_import_matches_adjusted_ewm(tmp_path: Path) -> None:
    path = tmp_path / "0.npz"
    rewards = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.savez_compressed(path, rewards=rewards, biome_regret=np.arange(4))
    result = import_official_foragax_npz(
        OfficialForagaxRunSpec(
            agent="RTU-PPO",
            seed=0,
            path=path,
            expected_steps=4,
        ),
        ewm_decay=0.5,
        record_every=2,
        final_window=2,
    )

    expected_ewm = np.asarray([1.0, 5 / 3, 17 / 7, 49 / 15])
    assert result.mean_reward == pytest.approx(2.5)
    assert result.final_window_mean_reward == pytest.approx(3.5)
    assert result.final_ewm_reward == pytest.approx(expected_ewm[-1])
    assert result.mean_ewm_reward == pytest.approx(float(np.mean(expected_ewm)))
    assert result.fov_last_10pct_ema_auc == pytest.approx(0.001)
    assert result.agent_metadata["archive_sha256"]
    assert result.agent_metadata["official_foragax_evidence"] is None
    assert result.agent_metadata["attestation_state"] == "unattested"
    assert "protocol_attested" not in result.agent_metadata

    with pytest.raises(ValueError, match="protocol attestation"):
        paired_forager_comparison([_result("alberta_horde_ac", 0, 1.0)], [result])

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        OfficialForagaxRunSpec(
            agent="Search Oracle",
            seed=0,
            path=path,
            privileged=True,
            config_path="paper/config/Search-Oracle.json",
            expected_steps=4,
            protocol_attested=True,
        )


def test_official_npz_import_rejects_nonfinite_ewm_decay(tmp_path: Path) -> None:
    path = tmp_path / "0.npz"
    np.savez_compressed(path, rewards=np.ones((4,), dtype=np.float32))

    with pytest.raises(ValueError, match="ewm_decay must be a finite number"):
        import_official_foragax_npz(
            OfficialForagaxRunSpec(agent="DQN", seed=0, path=path),
            ewm_decay=math.nan,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ewm_decay": True}, "ewm_decay must be a finite number"),
        ({"ewm_decay": _SpoofedFloat()}, "ewm_decay must be a finite number"),
        ({"ewm_decay": 10**10000}, "ewm_decay must be a finite number"),
        ({"ewm_decay": 1.0}, "ewm_decay must be a finite number"),
        ({"ewm_decay": -0.1}, "ewm_decay must be a finite number"),
        ({"ewm_decay": "0.5"}, "ewm_decay must be a finite number"),
        ({"record_every": True}, "record_every must be a positive integer"),
        ({"record_every": 0}, "record_every must be a positive integer"),
        ({"record_every": 1.5}, "record_every must be a positive integer"),
        ({"record_every": "2"}, "record_every must be a positive integer"),
        ({"final_window": True}, "final_window must be a positive integer"),
        ({"final_window": 0}, "final_window must be a positive integer"),
        ({"final_window": 1.5}, "final_window must be a positive integer"),
        ({"final_window": "2"}, "final_window must be a positive integer"),
    ],
)
def test_official_npz_import_rejects_spoofed_and_out_of_range_numeric_params(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    path = tmp_path / "0.npz"
    np.savez_compressed(path, rewards=np.ones((4,), dtype=np.float32))

    with pytest.raises(ValueError, match=match):
        import_official_foragax_npz(
            OfficialForagaxRunSpec(agent="DQN", seed=0, path=path),
            **kwargs,
        )


def test_official_npz_import_accepts_finite_endpoint_values(tmp_path: Path) -> None:
    path = tmp_path / "0.npz"
    np.savez_compressed(path, rewards=np.ones((4,), dtype=np.float32))

    result = import_official_foragax_npz(
        OfficialForagaxRunSpec(agent="DQN", seed=0, path=path, expected_steps=4),
        ewm_decay=0.0,
        record_every=1,
        final_window=1,
    )

    assert result.mean_reward == pytest.approx(1.0)


def test_official_npz_import_accepts_numpy_float64_ewm_decay(tmp_path: Path) -> None:
    path = tmp_path / "0.npz"
    np.savez_compressed(path, rewards=np.ones((4,), dtype=np.float32))

    result = import_official_foragax_npz(
        OfficialForagaxRunSpec(agent="DQN", seed=0, path=path, expected_steps=4),
        ewm_decay=np.float64(0.5),
        record_every=1,
        final_window=1,
    )

    assert result.mean_reward == pytest.approx(1.0)


def test_official_npz_import_rejects_user_defined_float_subclass(tmp_path: Path) -> None:
    class FloatSubclass(float):
        def __float__(self) -> float:
            raise AssertionError("custom conversion must not reach the importer")

    path = tmp_path / "0.npz"
    np.savez_compressed(path, rewards=np.ones((4,), dtype=np.float32))

    with pytest.raises(ValueError, match="ewm_decay must be a finite number"):
        import_official_foragax_npz(
            OfficialForagaxRunSpec(agent="DQN", seed=0, path=path, expected_steps=4),
            ewm_decay=FloatSubclass(0.5),
            record_every=1,
            final_window=1,
        )


def test_protocol_attestation_cannot_be_minted_by_constructing_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "0.npz"
    np.savez_compressed(path, rewards=np.ones((4,), dtype=np.float32))
    evidence = official_foragax_module.VerifiedOfficialForagaxEvidence(
        manifest_path=tmp_path / "forged-manifest.json",
        manifest_sha256="1" * 64,
        manifest_kind="official_foragax_single",
        trust_descriptor_id="forged",
        trust_descriptor_sha256="2" * 64,
        profile_id="forged",
        profile_sha256="3" * 64,
        artifact_identities_sha256="4" * 64,
        endorsement_descriptor_id="forged",
        endorsement_descriptor_sha256="5" * 64,
        endorsement_sha256="6" * 64,
    )
    with pytest.raises(ValueError, match="does not reverify"):
        OfficialForagaxRunSpec(
            agent="DQN",
            seed=0,
            path=path,
            attestation_evidence=evidence,
        )


def test_official_import_rejects_mislabelled_hash_symlink_and_lossy_arrays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "0.npz"
    np.savez_compressed(path, rewards=np.ones((4,), dtype=np.float32))
    environment = ForagerEnvConfig.paper_relearning()
    mislabelled = _official_environment_provenance(environment)
    mislabelled["semantic"]["env_id"] = "ForagaxBig-v5"
    with pytest.raises(ValueError, match="semantics do not match"):
        OfficialForagaxRunSpec(
            agent="DQN",
            seed=0,
            path=path,
            environment=environment,
            environment_provenance=mislabelled,
        )

    with pytest.raises(ValueError, match="official archive SHA-256"):
        import_official_foragax_npz(
            OfficialForagaxRunSpec(
                agent="DQN",
                seed=0,
                path=path,
                expected_archive_sha256="f" * 64,
            )
        )

    sealed = OfficialForagaxRunSpec(agent="DQN", seed=0, path=path)
    outside = tmp_path.parent / "outside-official.npz"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="import archive is not a regular file"):
        import_official_foragax_npz(sealed)
    path.unlink()
    path.write_bytes(outside.read_bytes())

    for name, rewards in (
        ("squeezed", np.ones((1, 4), dtype=np.float32)),
        ("boolean", np.ones((4,), dtype=np.bool_)),
        ("complex", np.ones((4,), dtype=np.complex64)),
    ):
        malformed = tmp_path / f"{name}.npz"
        np.savez_compressed(malformed, rewards=rewards)
        with pytest.raises(ValueError, match="one-dimensional|real numeric"):
            import_official_foragax_npz(
                OfficialForagaxRunSpec(
                    agent="DQN",
                    seed=0,
                    path=malformed,
                )
            )


def test_pairing_requires_matching_foragax_rng_schedule() -> None:
    candidate = _current_foragax_result(
        1.0,
        install_tree_sha256="a" * 64,
    )
    matched = _current_foragax_result(
        0.5,
        install_tree_sha256="a" * 64,
    )
    comparison = paired_forager_comparison(
        [candidate],
        [matched],
        bootstrap_resamples=10,
    )
    assert comparison.seeds == (0,)
    assert comparison.mean_difference == pytest.approx(0.5)

    different_schedule = _current_foragax_result(
        0.5,
        install_tree_sha256="a" * 64,
        schedule="shared_agent_environment_rng_v1",
    )
    with pytest.raises(ValueError, match="different environment RNG schedules"):
        paired_forager_comparison([candidate], [different_schedule])

    self_declared_profile = _current_foragax_result(
        0.5,
        install_tree_sha256="a" * 64,
        self_declared_runtime_profile=True,
    )
    with pytest.raises(
        ValueError,
        match="host Forager results may not self-declare",
    ):
        paired_forager_comparison([candidate], [self_declared_profile])

    partial_profile_metadata = dict(matched.agent_metadata)
    partial_profile_metadata["runtime_profile_id"] = (
        "foragax-current-gpu-a"
    )
    partial_profile = dataclasses.replace(
        matched,
        agent_metadata=partial_profile_metadata,
    )
    with pytest.raises(
        ValueError,
        match="host Forager results may not self-declare",
    ):
        paired_forager_comparison([candidate], [partial_profile])

    relabelled_schedule_metadata = dict(matched.agent_metadata)
    relabelled_schedule_metadata["environment_rng_schedule_sha256"] = "f" * 64
    relabelled_schedule = dataclasses.replace(
        matched,
        agent_metadata=relabelled_schedule_metadata,
    )
    with pytest.raises(ValueError, match="schedule digest does not verify"):
        paired_forager_comparison([candidate], [relabelled_schedule])


def test_npz_import_computes_historical_fov_statistic_from_raw_rewards(
    tmp_path: Path,
) -> None:
    path = tmp_path / "0.npz"
    rewards = np.ones((1_000,), dtype=np.float32)
    np.savez_compressed(path, rewards=rewards)

    result = import_official_foragax_npz(
        OfficialForagaxRunSpec(
            agent="DQN",
            seed=0,
            path=path,
            expected_steps=1_000,
        ),
        record_every=100,
        final_window=100,
    )

    # The historical collector initializes the unadjusted EMA to zero and
    # samples reward indices 0, 100, ..., 900. With ten samples, the final
    # floor-tail 10% is therefore the sample after reward index 900.
    assert result.fov_last_10pct_ema_auc == pytest.approx(1.0 - 0.999**901)
    assert result.metric_contract["fov_last_10pct_ema_auc"] == {
        "ema_decay": 0.999,
        "bias_correction": False,
        "initial_value": 0.0,
        "subsample_every_steps": 100,
        "subsample_first_reward": True,
        "tail_fraction_of_sampled_curve": 0.1,
    }
    assert result.agent_metadata["result_source"] == "official_foragax_agents_npz"


def test_summary_and_paired_comparison_fail_closed() -> None:
    candidate = [_result("alberta_horde_ac", 0, 1.0), _result("alberta_horde_ac", 1, 2.0)]
    baseline = [_result("RTU-PPO", 0, 0.5), _result("RTU-PPO", 1, 1.0)]
    comparison = paired_forager_comparison(
        candidate,
        baseline,
        bootstrap_resamples=100,
    )
    assert comparison.mean_difference == pytest.approx(0.75)
    report = build_forager_comparison_report(
        candidate + baseline,
        bootstrap_resamples=100,
    )
    assert report.paired_comparisons[0].baseline == "RTU-PPO"

    with pytest.raises(ValueError, match="unique seeds"):
        summarize_forager_runs([candidate[0], candidate[0]])
    with pytest.raises(ValueError, match="configuration"):
        summarize_forager_runs(
            [candidate[0], _result("alberta_horde_ac", 1, 2.0, config_id="different")]
        )
    incompatible_metric = _result("alberta_horde_ac", 1, 2.0, ewm_decay=0.5)
    with pytest.raises(ValueError, match="metric contract"):
        summarize_forager_runs([candidate[0], incompatible_metric])
    incompatible_baseline = [
        _result("RTU-PPO", 0, 0.5, ewm_decay=0.5),
        _result("RTU-PPO", 1, 1.0, ewm_decay=0.5),
    ]
    with pytest.raises(ValueError, match="metric contract"):
        paired_forager_comparison(candidate, incompatible_baseline)
    endorsed = _result("official", 0, 1.0)
    unattested = _result("official", 1, 1.0)
    endorsed.agent_metadata.update(
        result_source="official_foragax_agents_npz",
        attestation_state="protocol_attested",
        official_foragax_evidence={"endorsement_sha256": "a" * 64},
    )
    unattested.agent_metadata.update(
        result_source="official_foragax_agents_npz",
        attestation_state="unattested",
        official_foragax_evidence=None,
    )
    with pytest.raises(ValueError, match="configuration"):
        summarize_forager_runs([endorsed, unattested])
    with pytest.raises(ValueError, match="finite"):
        summarize_forager_runs([_result("bad", 0, math.nan)])
    with pytest.raises(ValueError, match="unsupported"):
        summarize_forager_runs(
            [candidate[0]],
            metric="not_a_metric",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="seed"):
        summarize_forager_runs(
            [dataclasses.replace(candidate[0], seed=True)],
        )
    malformed_environment = dataclasses.replace(
        candidate[0],
        environment={"env_id": object()},
    )
    with pytest.raises(ValueError, match="finite JSON"):
        summarize_forager_runs([malformed_environment])
    malformed_provenance = dataclasses.replace(
        candidate[0],
        agent_metadata={"config": {"bad": math.nan}},
    )
    with pytest.raises(ValueError, match="finite JSON"):
        summarize_forager_runs([malformed_provenance])


def test_paper_protocols_and_reference_labels() -> None:
    fov = paper_protocol("field_of_view")
    square = paper_protocol("relearning")
    big = paper_protocol("unending")
    assert (fov.tuning_steps, fov.tuning_seeds, fov.evaluation_steps) == (
        10_000,
        5,
        500_000,
    )
    assert fov.primary_metric == "fov_last_10pct_ema_auc"
    assert square.hidden_switch_interval_steps == 250_000
    assert square.primary_metric == "mean_ewm_reward"
    assert square.frozen_ablation_after_steps == 5_000_000
    assert big.ewm_decay == pytest.approx(0.99999)
    assert big.primary_metric == "final_ewm_reward"

    search = next(item for item in paper_baselines("relearning") if item.name == "Search Oracle")
    assert not search.in_tree_implementation
    relearning_names = {item.name for item in paper_baselines("relearning")}
    assert {
        "DQN + L2",
        "DQN + S&P",
        "PT-DQN",
        "PPO + L2",
        "PPO + S&P",
    } <= relearning_names
    fov_baselines = paper_baselines("field_of_view")
    assert "Search Nearest" in {item.name for item in fov_baselines}
    assert "github.com/steventango/forager-agents/tree/696b3a06" in fov_baselines[0].source
    for preset in ("field_of_view", "relearning", "unending"):
        assert all(
            item.official_config_path is not None
            for item in paper_baselines(preset)
            if not item.in_tree_implementation
        )
    with pytest.raises(ValueError, match="cannot be imported as raw Foragax NPZ"):
        OfficialForagaxRunSpec(
            agent="DQN",
            seed=0,
            path=Path("unused.npz"),
            environment=ForagerEnvConfig.paper_field_of_view(),
        )
    rtu = next(item for item in paper_reference_targets("relearning") if item.method == "RTU-PPO")
    assert rtu.central_estimate == pytest.approx(1.3)
    assert rtu.precision == "figure_digitized_approximation"


@pytest.mark.parametrize(
    "environment",
    [
        ForagerEnvConfig.paper_field_of_view(),
        ForagerEnvConfig.paper_relearning(),
        ForagerEnvConfig.paper_unending(),
    ],
    ids=("field_of_view", "relearning", "unending"),
)
def test_official_foragax_api_smoke(environment: ForagerEnvConfig) -> None:
    pytest.importorskip("foragax.registry")
    assert foragax_install_tree_sha256() == FORAGAX_INSTALL_TREE_SHA256
    env, params = environment.make()
    observation, state = env.reset(jr.key(0), params)
    initial_features = ForagerFeatureEncoder().encode(
        observation,
        ForagerFeatureEncoder().init(),
    )
    assert initial_features.ndim == 1
    observation, _, reward, done, info = env.step(
        jr.key(1),
        state,
        jnp.asarray(0, dtype=jnp.int32),
        params,
    )
    next_features = ForagerFeatureEncoder().encode(
        observation,
        ForagerFeatureEncoder().init(),
    )
    assert next_features.shape == initial_features.shape
    assert reward.shape == ()
    assert not bool(done)
    assert "biome_regret" in info
