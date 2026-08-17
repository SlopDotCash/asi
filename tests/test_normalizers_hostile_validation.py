"""Hostile validation for normalizers trust boundary."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.core.normalizers import (
    _require_exact_str,
    migrate_legacy_normalizer_state,
    normalizer_from_config,
    normalizer_state_nbytes_formula,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_nbytes_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("EMANormalizer")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        normalizer_state_nbytes_formula(evil, 4)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_nbytes_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        normalizer_state_nbytes_formula(_StringSubclass("EMANormalizer"), 4)


def test_nbytes_sanitized_without_repr() -> None:
    with pytest.raises(ValueError, match="Unknown normalizer type") as exc:
        normalizer_state_nbytes_formula("bad_type", 4)
    assert "!r" not in str(exc.value)
    assert "bad_type" in str(exc.value)
    assert "'" in str(exc.value)


def test_nbytes_valid() -> None:
    assert normalizer_state_nbytes_formula("EMANormalizer", 4) == 8 * 4 + 16
    assert normalizer_state_nbytes_formula("WelfordNormalizer", 4) == 12 * 4 + 12


def test_migrate_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("EMANormalizer")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        migrate_legacy_normalizer_state({}, normalizer_type=evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_migrate_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        migrate_legacy_normalizer_state({}, normalizer_type=_StringSubclass("EMANormalizer"))


def test_migrate_sanitized() -> None:
    with pytest.raises(ValueError, match="Unknown normalizer type") as exc:
        migrate_legacy_normalizer_state({}, normalizer_type="bad")
    assert "!r" not in str(exc.value)


def test_from_config_rejects_evil_type_without_hooks() -> None:
    evil = _EvilStr("EMANormalizer")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        normalizer_from_config({"type": evil})
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_from_config_rejects_subclass_type() -> None:
    with pytest.raises(ValueError, match="exact string"):
        normalizer_from_config({"type": _StringSubclass("EMANormalizer")})


def test_from_config_rejects_evil_schema_without_hooks() -> None:
    evil = _EvilStr("bad.schema")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        normalizer_from_config({"type": "EMANormalizer", "state_schema": evil})
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_from_config_sanitized_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported normalizer state schema") as exc:
        normalizer_from_config({"type": "EMANormalizer", "state_schema": "bad.schema"})
    assert "!r" not in str(exc.value)
    assert "bad.schema" in str(exc.value)


def test_from_config_rejects_evil_estimator_schema() -> None:
    evil = _EvilStr("bad.estimator")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        normalizer_from_config({"type": "WelfordNormalizer", "estimator_schema": evil})
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_from_config_rejects_evil_semantics_without_hooks() -> None:
    evil = _EvilStr("bad.semantics")
    _EvilStr.calls = 0
    # Use valid type but invalid semantics triggers mismatch branch
    with pytest.raises(ValueError, match="exact string|does not match") as exc:
        normalizer_from_config({"type": "EMANormalizer", "estimator_semantics": evil})
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_from_config_sanitized_semantics_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match") as exc:
        normalizer_from_config({"type": "EMANormalizer", "estimator_semantics": "bad.semantics"})
    assert "!r" not in str(exc.value)
    assert "bad.semantics" in str(exc.value)


def test_from_config_hostile_int_type_without_repr() -> None:
    evil = _HostileInt(123)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        normalizer_from_config({"type": evil})
    assert _HostileInt.calls == 0


def test_from_config_unknown_type_sanitized() -> None:
    with pytest.raises(ValueError, match="Unknown normalizer type") as exc:
        normalizer_from_config({"type": "BadType"})
    assert "!r" not in str(exc.value)
    assert "BadType" in str(exc.value)


def test_from_config_valid() -> None:
    from alberta_framework.core.normalizers import EMANormalizer, WelfordNormalizer

    n = normalizer_from_config(EMANormalizer().to_config())
    assert type(n).__name__ == "EMANormalizer"
    n2 = normalizer_from_config(WelfordNormalizer().to_config())
    assert type(n2).__name__ == "WelfordNormalizer"


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/core/normalizers.py").read_text()
    assert "!r" not in text
