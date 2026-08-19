from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.streams.gauntlet import (
    GauntletConfig,
    LifetimeGauntletStream,
)

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile lt must not run")

    def __mod__(self, other: object) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile mod must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_gauntlet_rejects_hostile_int_before_range() -> None:
    hostile = _HostileInt(8)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="relevant_dim must be an even integer"):
        GauntletConfig(relevant_dim=hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    hostile_seg = _HostileInt(3000)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="segment_length must be positive"):
        GauntletConfig(segment_length=hostile_seg)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    # bool must be rejected without dispatch
    with pytest.raises(ValueError, match="relevant_dim must be an even integer"):
        GauntletConfig(relevant_dim=True)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_gauntlet_rejects_hostile_float_before_float() -> None:
    hostile = _HostileFloat(0.1)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="noise_std must be a finite real number"):
        GauntletConfig(noise_std=hostile)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0

    hostile_int = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="noise_std must be a finite real number"):
        GauntletConfig(noise_std=hostile_int)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    # bool subclass
    with pytest.raises(ValueError, match="noise_std must be a finite real number"):
        GauntletConfig(noise_std=True)  # type: ignore[arg-type]


def test_gauntlet_benign_still_works() -> None:
    cfg = GauntletConfig()
    assert cfg.relevant_dim == 8
    assert cfg.noise_std == 0.1
    # benign exact int/float
    cfg2 = GauntletConfig(
        relevant_dim=4,
        noise_std=0.2,
        feature_std=1.0,
        scale_factor=5.0,
        drift_rate=0.02,
        context_noise_std=0.1,
    )
    assert cfg2.relevant_dim == 4
    # finite check

    with pytest.raises(ValueError, match="must be finite"):
        GauntletConfig(noise_std=float("inf"))  # type: ignore[arg-type]


def test_gauntlet_hostile_not_in_repr() -> None:
    hostile = _HostileInt(8)
    _HostileInt.calls = 0
    try:
        GauntletConfig(relevant_dim=hostile)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "_HostileInt" not in str(exc)
        assert _HostileInt.calls == 0
    else:
        raise AssertionError("should have raised")


def test_lifetime_gauntlet_from_config_roundtrip() -> None:
    stream = LifetimeGauntletStream(scale_cycle_period=4)
    cfg = stream.to_config()
    restored = LifetimeGauntletStream.from_config(cfg)
    assert restored.scale_cycle_period == 4
    assert restored.config.relevant_dim == stream.config.relevant_dim
    assert restored.config.noise_std == stream.config.noise_std


def test_lifetime_gauntlet_from_config_rejects_invalid_containers() -> None:
    with pytest.raises(ValueError, match="exact dictionary"):
        LifetimeGauntletStream.from_config([("type", "LifetimeGauntletStream")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact dictionary"):
        LifetimeGauntletStream.from_config("not_a_dict")  # type: ignore[arg-type]


def test_lifetime_gauntlet_from_config_rejects_non_string_keys() -> None:
    valid = LifetimeGauntletStream().to_config()
    bad_keys: dict[Any, Any] = dict(valid)
    bad_keys[123] = "invalid_key"
    with pytest.raises(ValueError, match="keys must be exact strings"):
        LifetimeGauntletStream.from_config(bad_keys)  # type: ignore[arg-type]


def test_lifetime_gauntlet_from_config_rejects_invalid_schema_fields() -> None:
    valid = LifetimeGauntletStream().to_config()
    with pytest.raises(ValueError, match="fields are invalid"):
        LifetimeGauntletStream.from_config({**valid, "extra": 1})
    missing = dict(valid)
    del missing["scale_cycle_period"]
    with pytest.raises(ValueError, match="fields are invalid"):
        LifetimeGauntletStream.from_config(missing)


def test_lifetime_gauntlet_from_config_rejects_unsupported_type_or_schema() -> None:
    valid = LifetimeGauntletStream().to_config()
    with pytest.raises(ValueError, match="type is unsupported"):
        LifetimeGauntletStream.from_config({**valid, "type": "WrongStream"})
    with pytest.raises(ValueError, match="config schema is unsupported"):
        LifetimeGauntletStream.from_config({**valid, "config_schema": "wrong.schema"})
    with pytest.raises(ValueError, match="state schema is unsupported"):
        LifetimeGauntletStream.from_config({**valid, "state_schema": "wrong.schema"})


def test_lifetime_gauntlet_from_config_rejects_invalid_nested_gauntlet_config() -> None:
    valid = LifetimeGauntletStream().to_config()
    with pytest.raises(ValueError, match="nested config must be an exact dictionary"):
        LifetimeGauntletStream.from_config({**valid, "gauntlet_config": "not_a_dict"})
    raw_gauntlet = dict(valid["gauntlet_config"])
    raw_gauntlet_bad_keys: dict[Any, Any] = dict(raw_gauntlet)
    raw_gauntlet_bad_keys[123] = "bad"
    with pytest.raises(ValueError, match="nested config keys must be exact strings"):
        LifetimeGauntletStream.from_config({**valid, "gauntlet_config": raw_gauntlet_bad_keys})
    with pytest.raises(ValueError, match="nested config fields are invalid"):
        LifetimeGauntletStream.from_config(
            {**valid, "gauntlet_config": {**raw_gauntlet, "extra": 1}}
        )


def test_lifetime_gauntlet_from_config_rejects_invalid_scale_period_type() -> None:
    valid = LifetimeGauntletStream().to_config()
    with pytest.raises(ValueError, match="scale_cycle_period must be an exact integer"):
        LifetimeGauntletStream.from_config({**valid, "scale_cycle_period": True})
    with pytest.raises(ValueError, match="scale_cycle_period must be an exact integer"):
        LifetimeGauntletStream.from_config({**valid, "scale_cycle_period": 3.0})
