"""Unit coverage for the historical Forager reconstruction module.

Targets the pure contract functions and validation gates: canonical JSON
bytes, mapping copies with byte limits, metric/semantic contracts, reward
and action validation, step-transition contract, kernel output, trace
digests, adapter construction gate, and factory source read-only checks.
"""

import json
import pathlib

import numpy as np
import pytest

from alberta_framework.benchmarks.historical_forager import (
    HISTORICAL_FORAGER_FAMILY_ID,
    HistoricalEnvironmentAdapter,
    HistoricalForagerContractError,
    _canonical_json_bytes,
    _finite_reward,
    _historical_transition,
    _json_mapping_copy,
    _kernel_output,
    _require_environment,
    _trace_digest,
    _validated_action,
    development_historical_environment_adapter,
    historical_forager_metric_contract,
    historical_forager_runtime_identity,
    historical_forager_semantic_contract,
)


class _StubEnv:
    def __init__(self, reward: float = 1.0):
        self._reward = reward

    def start(self) -> None:
        pass

    def step(self, action: int) -> tuple[float, np.ndarray, bool, dict]:
        return self._reward, np.zeros(2, dtype=np.float64), False, {}


def test_canonical_json_bytes_sort_keys() -> None:
    encoded = _canonical_json_bytes({"b": 1, "a": 2})
    assert json.loads(encoded) == {"a": 2, "b": 1}
    # Deterministic ordering regardless of insertion.
    assert _canonical_json_bytes({"b": 1, "a": 2}) == _canonical_json_bytes({"a": 2, "b": 1})


def test_canonical_json_bytes_rejects_nonfinite() -> None:
    with pytest.raises(HistoricalForagerContractError, match="finite canonical JSON"):
        _canonical_json_bytes({"x": float("nan")})


def test_json_mapping_copy_detaches_and_bounds() -> None:
    original = {"key": [1, 2, 3]}
    copied = _json_mapping_copy(original, name="test", maximum_bytes=1024)
    assert copied == original
    assert copied is not original
    copied["key"].append(4)
    assert original["key"] == [1, 2, 3]


def test_json_mapping_copy_rejects_oversize() -> None:
    with pytest.raises(HistoricalForagerContractError, match="byte limit"):
        _json_mapping_copy({"x": "a" * 5000}, name="big", maximum_bytes=100)


def test_json_mapping_copy_rejects_non_mapping() -> None:
    with pytest.raises(HistoricalForagerContractError, match="must be a mapping"):
        _json_mapping_copy([1, 2], name="list")


def test_metric_contract_is_nonpromoting() -> None:
    contract = historical_forager_metric_contract()
    assert contract["raw_rewards"]["present"] is True
    assert contract["biome_regret"]["available"] is False
    assert contract["biome_regret"]["synthesized"] is False


def test_semantic_contract_matches_family() -> None:
    contract = historical_forager_semantic_contract()
    assert contract["environment_family_id"] == HISTORICAL_FORAGER_FAMILY_ID
    assert contract["environment_reset_calls"] == 0
    assert contract["kernel_updates_per_transition"] == 1


def test_runtime_identity_records_host_without_attesting() -> None:
    identity = historical_forager_runtime_identity()
    assert identity["binding"] == "host_inventory_recorded_not_immutable"
    assert identity["runtime_is_historical_attestation"] is False
    assert identity["schema_version"].startswith("alberta.historical_numpy_forager.runtime")
    assert isinstance(identity["numpy"], str)


def test_require_environment_accepts_start_step() -> None:
    env = _StubEnv()
    assert _require_environment(env) is env


def test_require_environment_rejects_without_step() -> None:
    class NoStep:
        def start(self) -> None:
            pass

    with pytest.raises(HistoricalForagerContractError, match="start\\(\\) and step\\(\\)"):
        _require_environment(NoStep())


def test_finite_reward_accepts_real_scalar() -> None:
    assert _finite_reward(0.5) == 0.5
    assert _finite_reward(3) == 3.0
    assert _finite_reward(np.float64(1.25)) == 1.25


def test_finite_reward_rejects_bool_nan_and_strings() -> None:
    with pytest.raises(HistoricalForagerContractError):
        _finite_reward(True)  # bool is not a real scalar
    with pytest.raises(HistoricalForagerContractError, match="finite"):
        _finite_reward(float("inf"))
    with pytest.raises(HistoricalForagerContractError):
        _finite_reward("1.0")


def test_historical_transition_contract() -> None:
    reward, obs = _historical_transition(_StubEnv(0.5), 2)
    assert reward == 0.5
    assert obs.shape == (2,)


def test_historical_transition_rejects_terminal() -> None:
    class TerminalEnv:
        def start(self) -> None:
            pass

        def step(self, action: int) -> tuple[float, np.ndarray, bool, dict]:
            return 1.0, np.zeros(2), True, {}

    with pytest.raises(HistoricalForagerContractError, match="continuing"):
        _historical_transition(TerminalEnv(), 0)


def test_historical_transition_rejects_nonempty_info() -> None:
    class InfoEnv:
        def start(self) -> None:
            pass

        def step(self, action: int) -> tuple[float, np.ndarray, bool, dict]:
            return 1.0, np.zeros(2), False, {"extra": 1}

    with pytest.raises(HistoricalForagerContractError, match="empty info"):
        _historical_transition(InfoEnv(), 0)


def test_validated_action_range() -> None:
    assert _validated_action(0) == 0
    assert _validated_action(np.int32(3)) == 3
    assert _validated_action(np.array(2)) == 2
    with pytest.raises(HistoricalForagerContractError, match="\\[0, 3\\]"):
        _validated_action(4)
    with pytest.raises(HistoricalForagerContractError, match="\\[0, 3\\]"):
        _validated_action(-1)
    with pytest.raises(HistoricalForagerContractError, match="integer scalar"):
        _validated_action(1.5)
    with pytest.raises(HistoricalForagerContractError, match="integer scalar"):
        _validated_action(np.array([1, 2]))


def test_kernel_output_contract() -> None:
    state = {"weights": np.zeros(4)}
    out_state, action = _kernel_output((state, 1), phase="test")
    assert out_state is state
    assert action == 1
    with pytest.raises(HistoricalForagerContractError, match="exactly \\(state, action\\)"):
        _kernel_output((1, 2, 3), phase="test")
    with pytest.raises(HistoricalForagerContractError, match="exactly \\(state, action\\)"):
        _kernel_output("not-a-tuple", phase="test")


def test_trace_digest_deterministic_and_shape_sensitive() -> None:
    obs = np.zeros((2, 2), dtype=np.float64)
    rewards = np.array([1.0, 2.0])
    digest = _trace_digest(obs, rewards)
    assert _trace_digest(obs, rewards) == digest
    # Different reward values change the digest.
    assert _trace_digest(obs, np.array([2.0, 1.0])) != digest
    # Different shape changes the digest.
    assert _trace_digest(np.zeros((2, 2)), rewards) == digest
    assert _trace_digest(np.zeros((4, 1)), rewards) != digest


def test_development_adapter_is_unverified() -> None:
    adapter = development_historical_environment_adapter(_StubEnv)
    assert adapter.verified is False
    assert adapter.mode == "development_unverified_factory"
    assert adapter.golden_trace_sha256 is None
    assert adapter.to_dict()["trusted_source_asserted"] is False


def test_development_adapter_constructs_environment() -> None:
    adapter = development_historical_environment_adapter(
        lambda seed, aperture: _StubEnv()
    )
    env = adapter.construct(seed=0, aperture_size=4)
    assert callable(getattr(env, "step", None))


def test_adapter_rejects_direct_construction() -> None:
    with pytest.raises(HistoricalForagerContractError, match="construct adapters with"):
        HistoricalEnvironmentAdapter(
            factory=_StubEnv,
            mode="development_unverified_factory",
            golden_trace_sha256=None,
            stale_cache_seed_1_verified=False,
            source_inventory_verified=False,
            trusted_source_root=None,
            _token=object(),  # wrong token
        )
