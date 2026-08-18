"""Hostile string identity gates for causal-map serialized state validation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.benchmarks.causal_map_forager import (
    CausalMapForagerConfig,
    causal_map_start,
    causal_map_state_from_dict,
    causal_map_state_to_dict,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile str equality hook executed")

    def __ne__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile str inequality hook executed")

    def __hash__(self) -> int:
        return str.__hash__(self)


class _LyingStr(str):
    """Claims equality with everything without raising."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return str.__hash__(self)


def _observation() -> jnp.ndarray:
    image = jnp.zeros((9, 9, 3), dtype=jnp.float32)
    return image.at[4, 4, 1].set(1.0)


def _valid_payload(config: CausalMapForagerConfig) -> dict[str, object]:
    state, _ = causal_map_start(_observation(), config, 19)
    return causal_map_state_to_dict(state, config)


class TestSerializedStateStringIdentity:
    def test_valid_payload_round_trips(self) -> None:
        config = CausalMapForagerConfig()
        payload = _valid_payload(config)
        causal_map_state_from_dict(payload, config)

    def test_schema_subclass_is_rejected_without_hooks(self) -> None:
        config = CausalMapForagerConfig()
        payload = _valid_payload(config)
        _HostileStr.calls = 0
        payload["schema"] = _HostileStr(payload["schema"])
        with pytest.raises(ValueError, match="unsupported causal-map state schema"):
            causal_map_state_from_dict(payload, config)
        assert _HostileStr.calls == 0

    def test_lying_schema_subclass_cannot_spoof_the_gate(self) -> None:
        config = CausalMapForagerConfig()
        payload = _valid_payload(config)
        payload["schema"] = _LyingStr("alberta.wrong_schema.v0")
        with pytest.raises(ValueError, match="unsupported causal-map state schema"):
            causal_map_state_from_dict(payload, config)

    def test_prng_impl_subclass_is_rejected_without_hooks(self) -> None:
        config = CausalMapForagerConfig()
        payload = _valid_payload(config)
        _HostileStr.calls = 0
        payload["prng_impl"] = _HostileStr(payload["prng_impl"])
        with pytest.raises(
            ValueError, match="PRNG implementation does not match"
        ):
            causal_map_state_from_dict(payload, config)
        assert _HostileStr.calls == 0

    def test_config_sha256_subclass_is_rejected_without_hooks(self) -> None:
        config = CausalMapForagerConfig()
        payload = _valid_payload(config)
        _HostileStr.calls = 0
        payload["config_sha256"] = _HostileStr(payload["config_sha256"])
        with pytest.raises(
            ValueError, match="embeds a different configuration"
        ):
            causal_map_state_from_dict(payload, config)
        assert _HostileStr.calls == 0
