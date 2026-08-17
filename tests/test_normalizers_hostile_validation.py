"""Hostile validation for core/normalizers trust-boundary."""

import numpy as np
import pytest

from alberta_framework.core.normalizers import (
    NORMALIZER_STATE_SCHEMA,
    EMANormalizer,
    StreamingBatchNormalizer,
    WelfordNormalizer,
    migrate_legacy_normalizer_state,
    normalizer_from_config,
    normalizer_state_nbytes_formula,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileFloat.__repr__ must not be called")


def test_rejects_string_subclass_for_normalizer_type() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        normalizer_state_nbytes_formula(
            _StringSubclass("EMANormalizer"), 4  # type: ignore[arg-type]
        )


def test_hostile_str_for_normalizer_type_without_repr_leak() -> None:
    evil = _EvilStr("EMANormalizer")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        normalizer_state_nbytes_formula(evil, 4)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int_for_feature_dim() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        normalizer_state_nbytes_formula("EMANormalizer", True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="feature_dim") as exc:
        normalizer_state_nbytes_formula(
            "EMANormalizer", _HostileInt(4)  # type: ignore[arg-type]
        )
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_does_not_invoke_hostile_feature_dim_when_type_is_evil() -> None:
    evil = _EvilStr("EMANormalizer")
    hostile = _HostileInt(4)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        normalizer_state_nbytes_formula(evil, hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_rejects_hostile_str_for_migrate_without_repr() -> None:
    evil = _EvilStr("EMANormalizer")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        migrate_legacy_normalizer_state({}, normalizer_type=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_rejects_string_subclass_for_type_name_via_from_config() -> None:
    cfg = EMANormalizer().to_config()
    cfg["type"] = _StringSubclass("EMANormalizer")  # type: ignore[assignment]
    # Hits estimator_schema branch? No, but hits Unknown? Actually EMANormalizer type
    # with StringSubclass will still be looked up via _NORMALIZER_REGISTRY.get
    # which uses exact string match - will miss and go to final gate.
    # But our final gate now checks _require_exact_str before ValueError.
    # So it should raise "must be an exact string" for type_name.
    # For Welford case with estimator_schema we test separately.
    # Use a config that triggers the final Unknown branch.
    cfg2 = {"type": _StringSubclass("Bogus"), "state_schema": NORMALIZER_STATE_SCHEMA}
    with pytest.raises(ValueError, match="must be an exact string"):
        normalizer_from_config(cfg2)  # type: ignore[arg-type]


def test_hostile_str_for_type_name_without_repr_via_from_config() -> None:
    evil = _EvilStr("Bogus")
    cfg = {"type": evil, "state_schema": NORMALIZER_STATE_SCHEMA}
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        normalizer_from_config(cfg)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_unsupported_schema_without_repr() -> None:
    evil = _EvilStr("bad-schema")
    cfg = EMANormalizer().to_config()
    cfg["state_schema"] = evil  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported normalizer state schema") as exc:
        normalizer_from_config(cfg)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)

    # estimator_schema generic too
    cfg2 = WelfordNormalizer().to_config()
    cfg2["estimator_schema"] = _EvilStr("bad")  # type: ignore[assignment]
    with pytest.raises(
        ValueError, match="Unsupported Welford estimator schema"
    ) as exc2:
        normalizer_from_config(cfg2)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_rejects_mismatched_estimator_semantics_without_repr() -> None:
    cfg = EMANormalizer(decay=0.9).to_config()
    cfg["estimator_semantics"] = "bogus-mismatch"
    with pytest.raises(ValueError, match="does not match") as exc:
        normalizer_from_config(cfg)
    assert "!r" not in str(exc.value)
    assert "bogus-mismatch" in str(exc.value)
    # ensure quoted
    assert "'bogus-mismatch'" in str(exc.value)

    # EvilStr for estimator_semantics must be blocked via exact-string gate
    evil = _EvilStr("bogus")
    cfg2 = EMANormalizer(decay=0.9).to_config()
    cfg2["estimator_semantics"] = evil  # type: ignore[assignment]
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        normalizer_from_config(cfg2)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_rejects_unknown_type_without_repr() -> None:
    with pytest.raises(ValueError, match="Unknown normalizer type") as exc:
        normalizer_state_nbytes_formula("BogusType", 4)
    assert "!r" not in str(exc.value)
    assert "BogusType" in str(exc.value)
    assert "'BogusType'" in str(exc.value)

    with pytest.raises(ValueError, match="Unknown normalizer type") as exc2:
        normalizer_from_config(
            {"type": "BogusType", "state_schema": NORMALIZER_STATE_SCHEMA}
        )
    assert "!r" not in str(exc2.value)


def test_valid_configs_still_pass() -> None:
    # canonical normalizers
    for ctor in (EMANormalizer, WelfordNormalizer, StreamingBatchNormalizer):
        n = ctor()  # type: ignore[operator]
        state = n.init(4)
        assert state.mean.shape == (4,)
    # valid formulas
    assert normalizer_state_nbytes_formula("EMANormalizer", 4) == 48
    assert normalizer_state_nbytes_formula("WelfordNormalizer", 4) == 60
    # valid from_config roundtrip
    cfg = EMANormalizer(decay=0.5).to_config()
    restored = normalizer_from_config(cfg)
    assert isinstance(restored, EMANormalizer)
    cfg2 = WelfordNormalizer().to_config()
    restored2 = normalizer_from_config(cfg2)
    assert isinstance(restored2, WelfordNormalizer)


def test_numpy_scalars_pass() -> None:
    # int families
    for int_type in (int, np.int32, np.int64, np.uint32):
        assert normalizer_state_nbytes_formula("EMANormalizer", int_type(4)) == 48
    # valid string still passes
    n = EMANormalizer()
    s = n.init(np.int32(2))
    assert s.mean.shape == (2,)


def test_hostile_float_feature_dim_rejected_without_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="feature_dim") as exc:
        normalizer_state_nbytes_formula(
            "EMANormalizer", _HostileFloat(4.0)  # type: ignore[arg-type]
        )
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)
