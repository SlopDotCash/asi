"""Focused contracts for the reconstructed mutable NumPy Forager lane."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.historical_forager import (
    HISTORICAL_FORAGER_EMA_DECAY,
    HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256,
    HistoricalForagerArtifactError,
    HistoricalForagerContractError,
    HistoricalForagerPairingIdentity,
    HistoricalForagerRunConfig,
    HistoricalUpdateKernel,
    assert_historical_artifacts_pairable,
    development_historical_environment_adapter,
    historical_artifact_pairing_identity,
    historical_fov_metrics,
    run_historical_forager,
    validate_historical_forager_artifact,
    verify_historical_environment_factory,
)
from alberta_framework.benchmarks.historical_forager_provenance import (
    CURRENT_FORAGAX_055_FAMILY_ID,
    HISTORICAL_FORAGER_FAMILY_ID,
    HISTORICAL_FORAGER_PROVENANCE_SHA256,
    HistoricalForagerFamilyMismatchError,
    HistoricalForagerProvenanceError,
    assert_historical_family_pairing,
    historical_forager_provenance,
    validate_historical_forager_provenance,
)

pytestmark = pytest.mark.integration


class _FakeHistoricalEnvironment:
    def __init__(
        self,
        rewards: list[Any],
        events: list[tuple[Any, ...]],
        observations: list[object],
        *,
        info: Mapping[str, Any] | None = None,
        fail_at: int | None = None,
    ) -> None:
        self.rewards = rewards
        self.events = events
        self.observations = observations
        self.info = {} if info is None else info
        self.fail_at = fail_at
        self.offset = 0
        self.start_calls = 0

    def start(self) -> object:
        self.start_calls += 1
        self.events.append(("environment.start",))
        return self.observations[0]

    def step(self, action: int) -> tuple[Any, object, bool, Mapping[str, Any]]:
        self.events.append(("environment.step", self.offset, action))
        if self.fail_at == self.offset:
            raise RuntimeError("synthetic interruption")
        reward = self.rewards[self.offset]
        self.offset += 1
        return reward, self.observations[self.offset], False, self.info


def _fake_lane(
    tmp_path: Path,
    rewards: list[Any],
    *,
    info: Mapping[str, Any] | None = None,
    fail_at: int | None = None,
    action_for: Callable[[int], Any] | None = None,
) -> tuple[
    Any,
    HistoricalUpdateKernel[int],
    HistoricalForagerRunConfig,
    list[tuple[Any, ...]],
    list[object],
    _FakeHistoricalEnvironment,
]:
    events: list[tuple[Any, ...]] = []
    observations = [object() for _ in range(len(rewards) + 1)]
    environment = _FakeHistoricalEnvironment(
        rewards,
        events,
        observations,
        info=info,
        fail_at=fail_at,
    )
    factory_calls = 0

    def factory(seed: int, aperture_size: int) -> _FakeHistoricalEnvironment:
        nonlocal factory_calls
        factory_calls += 1
        events.append(("factory", seed, aperture_size, factory_calls))
        return environment

    def start_kernel(observation: Any) -> tuple[int, Any]:
        assert observation is observations[0]
        events.append(("kernel.start", observation))
        return 0, 1 if action_for is None else action_for(0)

    def update_kernel(state: int, reward: Any, observation: Any) -> tuple[int, Any]:
        assert reward is rewards[state]
        assert observation is observations[state + 1]
        events.append(("kernel.update", state, reward, observation))
        next_state = state + 1
        action = (next_state + 1) % 4 if action_for is None else action_for(next_state)
        return next_state, action

    adapter = development_historical_environment_adapter(factory)
    kernel = HistoricalUpdateKernel(
        name="fake_streaming_kernel",
        start_kernel=start_kernel,
        update_kernel=update_kernel,
        metadata={"implementation": "deterministic_test_double"},
    )
    config = HistoricalForagerRunConfig(
        seed=7,
        steps=len(rewards),
        aperture_size=9,
        output_directory=tmp_path / "run",
        allow_unverified_development_adapter=True,
    )
    return adapter, kernel, config, events, observations, environment


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def test_provenance_records_exact_reconstruction_and_unattested_resolution() -> None:
    provenance = historical_forager_provenance()

    assert HISTORICAL_FORAGER_PROVENANCE_SHA256 == (
        "0ec63f3628c13222b24983f0ac2ac025c9e8ca7d7c8e4a10ae62373336d09b01"
    )
    assert provenance["family_id"] == HISTORICAL_FORAGER_FAMILY_ID
    assert provenance["environment_resolution_attested"] is False
    assert provenance["agents"]["commit"] == ("696b3a06fbd0dc72407556b039d219e704ec6992")
    assert provenance["agents"]["tree"] == ("4936577cba549a3ffb4dec69bff722360c52f8be")
    assert provenance["agents"]["archive_sha256"] == (
        "a66ee0f7dd565dd64f5959520587e7121e5195d2814ef83504d8fc2b341d4803"
    )
    assert provenance["agents"]["environment_dependency_revision_pinned"] is False
    assert provenance["environment"]["commit"] == ("d140bdb3c51c7b6747d0588078ca97a67b55a8e1")
    assert provenance["environment"]["tree"] == ("0eb78e64b34cce3222215ebee3b94de2d83d41ce")
    assert provenance["environment"]["archive_sha256"] == (
        "2b7caf0a83b741404a88dfbb427f34f92e822d25901b4c9a71667d6e24cf14dd"
    )
    assert provenance["environment"]["reconstructed_wheel_sha256"] == (
        "9fcf134767a73337d36d6dec9c25721da68fd1b9587ea4c4299a3cdc00fc2020"
    )
    assert provenance["environment"]["reconstructed_wheel_is_historical_attestation"] is False
    assert (
        len([name for name in provenance["environment"]["files"] if name.startswith("forager/")])
        == 15
    )
    assert hashlib.sha256(_canonical_json(provenance)).hexdigest() == (
        HISTORICAL_FORAGER_PROVENANCE_SHA256
    )
    validate_historical_forager_provenance(provenance)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("environment_resolution_attested", True),
        lambda value: value["environment"].__setitem__("commit", "0" * 40),
        lambda value: value["agents"].__setitem__("extra_claim", True),
        lambda value: value["environment"].pop("tree"),
    ],
)
def test_provenance_tampering_fails_closed(mutate: Callable[[dict[str, Any]], Any]) -> None:
    provenance = historical_forager_provenance()
    mutate(provenance)
    with pytest.raises(HistoricalForagerProvenanceError, match="differs"):
        validate_historical_forager_provenance(provenance)


def test_cross_family_pairing_is_rejected_before_metrics() -> None:
    assert_historical_family_pairing(
        HISTORICAL_FORAGER_FAMILY_ID,
        HISTORICAL_FORAGER_FAMILY_ID,
    )
    with pytest.raises(HistoricalForagerFamilyMismatchError, match="pair only"):
        assert_historical_family_pairing(
            HISTORICAL_FORAGER_FAMILY_ID,
            CURRENT_FORAGAX_055_FAMILY_ID,
        )
    with pytest.raises(HistoricalForagerFamilyMismatchError, match="pair only"):
        assert_historical_family_pairing("historical_numpy_forager_corrected", "unknown")


def test_runner_preserves_exact_rlglue_order_and_raw_values(tmp_path: Path) -> None:
    rewards = [np.int64(1), np.float32(-1), np.int32(30), np.float64(0.5)]
    adapter, kernel, config, events, observations, environment = _fake_lane(
        tmp_path,
        rewards,
    )

    execution = run_historical_forager(adapter, kernel, config)

    assert environment.start_calls == 1
    assert environment.offset == len(rewards)
    assert execution.final_kernel_state == len(rewards)
    assert execution.next_action == 1
    assert events == [
        ("factory", 7, 9, 1),
        ("environment.start",),
        ("kernel.start", observations[0]),
        ("environment.step", 0, 1),
        ("kernel.update", 0, rewards[0], observations[1]),
        ("environment.step", 1, 2),
        ("kernel.update", 1, rewards[1], observations[2]),
        ("environment.step", 2, 3),
        ("kernel.update", 2, rewards[2], observations[3]),
        ("environment.step", 3, 0),
        ("kernel.update", 3, rewards[3], observations[4]),
    ]

    manifest = validate_historical_forager_artifact(config.output_directory)
    raw = np.load(config.output_directory / "rewards.npy", allow_pickle=False)
    assert raw.dtype.str == "<f8"
    assert np.array_equal(raw, np.asarray(rewards, dtype=np.float64))
    assert manifest["family_id"] == HISTORICAL_FORAGER_FAMILY_ID
    assert manifest["environment_resolution_attested"] is False
    assert manifest["pairable_with_current_foragax"] is False
    assert manifest["environment_adapter"]["source_preflight_verified"] is False
    assert manifest["metric_contract"]["biome_regret"] == {
        "available": False,
        "synthesized": False,
    }
    assert manifest["reward_sidecar"]["biome_regret_present"] is False
    assert "mean_biome_regret" not in manifest["metrics"]
    assert manifest["runtime"]["runtime_is_historical_attestation"] is False
    assert set(path.name for path in config.output_directory.iterdir()) == {
        "result.json",
        "rewards.npy",
    }


def test_exact_unadjusted_ema_subsample_and_last_tenth_mean() -> None:
    rewards = np.asarray(
        [30.0 if index % 137 == 0 else (-1.0 if index % 29 == 0 else 1.0) for index in range(1001)],
        dtype=np.float64,
    )
    ema = 0.0
    samples: list[float] = []
    for index, reward in enumerate(rewards):
        ema = HISTORICAL_FORAGER_EMA_DECAY * ema + (1.0 - HISTORICAL_FORAGER_EMA_DECAY) * float(
            reward
        )
        if index % 100 == 0:
            samples.append(ema)
    expected = float(np.mean(np.asarray(samples[int(0.9 * len(samples)) :], dtype=np.float64)))

    metrics = historical_fov_metrics(rewards)

    assert metrics["ema_sample_count"] == 11
    assert metrics["ema_tail_start_index"] == 9
    assert metrics["ema_tail_sample_count"] == 2
    assert metrics["fov_last_10pct_ema_auc"] == expected
    assert metrics["final_unadjusted_ema"] == ema
    assert metrics["total_reward"] == float(sum(float(value) for value in rewards))


def test_development_adapter_requires_explicit_opt_in(tmp_path: Path) -> None:
    adapter, kernel, config, events, _, _ = _fake_lane(tmp_path, [1.0])
    config = dataclasses.replace(config, allow_unverified_development_adapter=False)

    with pytest.raises(HistoricalForagerContractError, match="explicit"):
        run_historical_forager(adapter, kernel, config)

    assert events == []
    assert not config.output_directory.exists()


def test_nonempty_info_is_rejected_and_never_reaches_kernel(tmp_path: Path) -> None:
    adapter, kernel, config, events, _, _ = _fake_lane(
        tmp_path,
        [1.0],
        info={"biome_regret": 0.0},
    )

    with pytest.raises(HistoricalForagerContractError, match="biome_regret is unavailable"):
        run_historical_forager(adapter, kernel, config)

    assert not any(event[0] == "kernel.update" for event in events)
    assert not config.output_directory.exists()


class _SpoofedFloat:
    """Mimics ``float`` via ``__class__`` to defeat ``isinstance`` checks."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return 0.5


def test_class_spoofed_reward_is_rejected_and_never_reaches_kernel(
    tmp_path: Path,
) -> None:
    adapter, kernel, config, events, _, _ = _fake_lane(tmp_path, [_SpoofedFloat()])

    with pytest.raises(HistoricalForagerContractError, match="reward must be a real scalar"):
        run_historical_forager(adapter, kernel, config)

    assert not any(event[0] == "kernel.update" for event in events)
    assert not config.output_directory.exists()


class _HostileFloatSubclass(float):
    def __float__(self) -> float:
        raise RuntimeError("hostile conversion")


def test_hostile_real_conversion_is_translated_and_never_reaches_kernel(
    tmp_path: Path,
) -> None:
    adapter, kernel, config, events, _, _ = _fake_lane(
        tmp_path, [_HostileFloatSubclass(0.5)]
    )

    with pytest.raises(
        HistoricalForagerContractError,
        match="reward must be a finite real scalar",
    ):
        run_historical_forager(adapter, kernel, config)

    assert not any(event[0] == "kernel.update" for event in events)
    assert not config.output_directory.exists()


@pytest.mark.parametrize("bad_action", [True, 1.0, -1, 4, np.asarray([1])])
def test_invalid_actions_fail_closed_without_partial_artifact(
    tmp_path: Path,
    bad_action: Any,
) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(
        tmp_path,
        [1.0],
        action_for=lambda _: bad_action,
    )

    with pytest.raises(HistoricalForagerContractError, match="action"):
        run_historical_forager(adapter, kernel, config)

    assert not config.output_directory.exists()


def test_interruption_cannot_publish_a_complete_artifact(tmp_path: Path) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(
        tmp_path,
        [1.0, 2.0, 3.0],
        fail_at=1,
    )

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_historical_forager(adapter, kernel, config)

    assert not config.output_directory.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(tmp_path, [1.0])
    config.output_directory.mkdir()
    marker = config.output_directory / "owned.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(HistoricalForagerArtifactError, match="overwrite"):
        run_historical_forager(adapter, kernel, config)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_artifact_tampering_and_extra_files_fail_closed(tmp_path: Path) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(tmp_path, [1.0, -1.0, 30.0])
    run_historical_forager(adapter, kernel, config)
    reward_path = config.output_directory / "rewards.npy"
    reward_path.chmod(0o600)
    encoded = bytearray(reward_path.read_bytes())
    encoded[-1] ^= 1
    reward_path.write_bytes(encoded)
    reward_path.chmod(0o444)
    with pytest.raises(HistoricalForagerArtifactError, match="SHA-256|metrics"):
        validate_historical_forager_artifact(config.output_directory)

    reward_path.chmod(0o600)
    reward_path.write_bytes(np.asarray([1.0, -1.0, 30.0], dtype="<f8").tobytes())
    extra = config.output_directory / "biome-regret.npy"
    extra.write_bytes(b"forbidden")
    with pytest.raises(HistoricalForagerArtifactError, match="exactly"):
        validate_historical_forager_artifact(config.output_directory)


def test_manifest_family_and_provenance_tampering_fail_closed(tmp_path: Path) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(tmp_path, [1.0])
    run_historical_forager(adapter, kernel, config)
    result_path = config.output_directory / "result.json"
    result_path.chmod(0o600)
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["family_id"] = CURRENT_FORAGAX_055_FAMILY_ID
    manifest["environment_resolution_attested"] = True
    result_path.write_bytes(_canonical_json(manifest) + b"\n")
    result_path.chmod(0o444)

    with pytest.raises(HistoricalForagerArtifactError, match="identity"):
        validate_historical_forager_artifact(config.output_directory)


def test_boolean_sidecar_dimensions_cannot_alias_integer_horizon(tmp_path: Path) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(tmp_path, [1.0])
    run_historical_forager(adapter, kernel, config)
    result_path = config.output_directory / "result.json"
    result_path.chmod(0o600)
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["reward_sidecar"]["steps"] = True
    manifest["reward_sidecar"]["shape"] = [True]
    result_path.write_bytes(_canonical_json(manifest) + b"\n")
    result_path.chmod(0o444)

    with pytest.raises(HistoricalForagerArtifactError, match="sidecar metadata"):
        validate_historical_forager_artifact(config.output_directory)


def test_runtime_inventory_claim_cannot_be_flipped(tmp_path: Path) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(tmp_path, [1.0])
    run_historical_forager(adapter, kernel, config)
    result_path = config.output_directory / "result.json"
    result_path.chmod(0o600)
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["runtime"]["matches_audited_compatibility_runtime"] = not manifest["runtime"][
        "matches_audited_compatibility_runtime"
    ]
    result_path.write_bytes(_canonical_json(manifest) + b"\n")
    result_path.chmod(0o444)

    with pytest.raises(HistoricalForagerArtifactError, match="runtime compatibility"):
        validate_historical_forager_artifact(config.output_directory)


def test_pairing_identity_requires_same_seed_horizon_geometry_and_family(tmp_path: Path) -> None:
    adapter, kernel, config, _, _, _ = _fake_lane(tmp_path, [1.0, 0.0])
    run_historical_forager(adapter, kernel, config)
    identity = historical_artifact_pairing_identity(config.output_directory)

    assert_historical_artifacts_pairable(identity, identity)
    with pytest.raises(HistoricalForagerContractError, match="identical provenance"):
        assert_historical_artifacts_pairable(identity, dataclasses.replace(identity, seed=8))
    current = dataclasses.replace(identity, family_id=CURRENT_FORAGAX_055_FAMILY_ID)
    with pytest.raises(HistoricalForagerFamilyMismatchError, match="pair only"):
        assert_historical_artifacts_pairable(identity, current)
    dishonest = HistoricalForagerPairingIdentity(
        family_id=HISTORICAL_FORAGER_FAMILY_ID,
        provenance_sha256="0" * 64,
        seed=identity.seed,
        aperture_size=identity.aperture_size,
        steps=identity.steps,
        semantic_contract_sha256=identity.semantic_contract_sha256,
        environment_adapter_mode=identity.environment_adapter_mode,
        runtime_sha256=identity.runtime_sha256,
    )
    with pytest.raises(HistoricalForagerContractError, match="canonical provenance"):
        assert_historical_artifacts_pairable(identity, dishonest)
    with pytest.raises(HistoricalForagerContractError, match="canonical provenance"):
        assert_historical_artifacts_pairable(dishonest, dishonest)
    invalid_coordinates = dataclasses.replace(identity, steps=0)
    with pytest.raises(HistoricalForagerContractError, match="run coordinates"):
        assert_historical_artifacts_pairable(invalid_coordinates, invalid_coordinates)
    verified_mode = dataclasses.replace(
        identity,
        environment_adapter_mode="golden_verified_read_only_source",
    )
    with pytest.raises(HistoricalForagerContractError, match="verification mode"):
        assert_historical_artifacts_pairable(identity, verified_mode)


def test_jitted_pure_kernel_seam_runs_without_algorithm_specific_adapter(tmp_path: Path) -> None:
    rewards = [1.0, -1.0, 30.0]
    observations = [np.asarray([index], dtype=np.float32) for index in range(4)]
    events: list[tuple[Any, ...]] = []
    environment = _FakeHistoricalEnvironment(rewards, events, list(observations))

    def factory(seed: int, aperture: int) -> _FakeHistoricalEnvironment:
        del seed, aperture
        return environment

    @jax.jit
    def start_kernel(observation: Any) -> tuple[Any, Any]:
        return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(observation[0] % 4, dtype=jnp.int32)

    @jax.jit
    def update_kernel(state: Any, reward: Any, observation: Any) -> tuple[Any, Any]:
        next_state = state + jnp.asarray(1, dtype=jnp.int32)
        action = (jnp.asarray(observation[0], dtype=jnp.int32) + next_state) % 4
        return next_state, action

    execution = run_historical_forager(
        development_historical_environment_adapter(factory),
        HistoricalUpdateKernel(
            name="jitted_test_kernel",
            start_kernel=start_kernel,
            update_kernel=update_kernel,
        ),
        HistoricalForagerRunConfig(
            seed=0,
            steps=3,
            output_directory=tmp_path / "jitted-run",
            allow_unverified_development_adapter=True,
        ),
    )

    assert int(execution.final_kernel_state) == 3
    assert 0 <= execution.next_action < 4


def test_trusted_preflight_rejects_temporary_source_before_factory_execution(
    tmp_path: Path,
) -> None:
    called = False

    def factory(seed: int, aperture: int) -> Any:
        nonlocal called
        called = True
        raise AssertionError((seed, aperture))

    with pytest.raises(HistoricalForagerContractError, match="temporary storage"):
        verify_historical_environment_factory(factory, trusted_source_root=tmp_path)

    assert called is False
    assert len(HISTORICAL_FORAGER_GOLDEN_TRACE_SHA256) == 64


def test_kernel_metadata_cannot_override_environment_claims() -> None:
    with pytest.raises(HistoricalForagerContractError, match="reserved"):
        HistoricalUpdateKernel(
            name="dishonest",
            start_kernel=lambda observation: (None, 0),
            update_kernel=lambda state, reward, observation: (None, 0),
            metadata={"environment_resolution_attested": True},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"steps": True},
        {"steps": 0},
        {"steps": 100_000_001},
        {"aperture_size": 2},
        {"allow_unverified_development_adapter": 1},
    ],
)
def test_run_configuration_is_strict_and_bounded(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "seed": 0,
        "steps": 1,
        "output_directory": tmp_path / "run",
    }
    values.update(kwargs)
    with pytest.raises(HistoricalForagerContractError):
        HistoricalForagerRunConfig(**values)


def test_metric_rejects_nonfinite_or_unbounded_shapes() -> None:
    with pytest.raises(HistoricalForagerContractError):
        historical_fov_metrics(np.asarray([[1.0]]))
    with pytest.raises(HistoricalForagerContractError):
        historical_fov_metrics(np.asarray([math.nan]))
