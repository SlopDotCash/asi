"""Hostile validation for benchmarks/historical_forager trust-boundary (additional)."""

import pathlib
import sys
import types

import pytest

from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerArtifactError,
    HistoricalForagerContractError,
    _require_exact_str,
    _strict_json_object,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


class _EvilStrNoHash(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


def test_require_exact_str_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("forager"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("forager.foo")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("name", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_strict_json_duplicate_key_sanitized(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "dup.json"
    p.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    p.chmod(0o444)
    with pytest.raises(HistoricalForagerArtifactError, match="duplicate JSON key") as exc:
        _strict_json_object(p)
    assert "!r" not in str(exc.value)
    assert "'a'" in str(exc.value)


def test_strict_json_duplicate_key_hostile_gate_before_hash(tmp_path: pathlib.Path) -> None:
    evil = _EvilStr("a")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("a"))  # type: ignore[arg-type]


def test_strict_json_nonstandard_constant_sanitized(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "const.json"
    p.write_text('{"a": NaN}', encoding="utf-8")
    p.chmod(0o444)
    with pytest.raises(HistoricalForagerArtifactError, match="non-standard JSON constant") as exc:
        _strict_json_object(p)
    assert "!r" not in str(exc.value)
    assert "'NaN'" in str(exc.value) or "NaN" in str(exc.value)


def test_strict_json_nonfinite_number_sanitized(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "num.json"
    p.write_text('{"a": 1e999}', encoding="utf-8")
    p.chmod(0o444)
    with pytest.raises(HistoricalForagerArtifactError, match="non-finite JSON number") as exc:
        _strict_json_object(p)
    assert "!r" not in str(exc.value)
    assert "'1e999'" in str(exc.value)


def test_loaded_module_hostile_without_repr_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from alberta_framework.benchmarks import historical_forager as hf

    evil_name = _EvilStrNoHash("forager.evil")
    fake_mod = types.ModuleType(evil_name)  # type: ignore[arg-type]
    fake_mod.__file__ = str(tmp_path / "nonexistent.py")
    monkeypatch.setitem(sys.modules, evil_name, fake_mod)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        hf._require_loaded_forager_modules_read_only_non_tmp()
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_loaded_module_valid_name_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from alberta_framework.benchmarks import historical_forager as hf

    valid_name = "forager.evil_valid"
    fake_mod = types.ModuleType(valid_name)
    fake_mod.__file__ = str(tmp_path / "nonexistent_valid.py")
    monkeypatch.setitem(sys.modules, valid_name, fake_mod)
    with pytest.raises(HistoricalForagerContractError, match="has no stable source file") as exc:
        hf._require_loaded_forager_modules_read_only_non_tmp()
    assert "!r" not in str(exc.value)
    assert f"'{valid_name}'" in str(exc.value)


def test_loaded_module_string_subclass_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from alberta_framework.benchmarks import historical_forager as hf

    sub_name = _StringSubclass("forager.sub")
    fake_mod = types.ModuleType(sub_name)  # type: ignore[arg-type]
    fake_mod.__file__ = str(tmp_path / "nonexistent2.py")
    monkeypatch.setitem(sys.modules, sub_name, fake_mod)  # type: ignore[arg-type]
    with pytest.raises(
        (HistoricalForagerContractError, ValueError),
        match="must be an exact string|has no stable",
    ):
        hf._require_loaded_forager_modules_read_only_non_tmp()


def test_missing_artifact_path_name_sanitized(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "missing.json"
    evil = _EvilStr("missing.json")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("path.name", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    with pytest.raises(HistoricalForagerArtifactError, match="missing artifact file") as exc2:
        _strict_json_object(missing)
    assert "!r" not in str(exc2.value)
    assert "'missing.json'" in str(exc2.value)


def test_not_canonical_path_name_sanitized(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(HistoricalForagerArtifactError, match="is not canonical") as exc:
        _strict_json_object(p)
    assert "!r" not in str(exc.value)
    assert "'bad.json'" in str(exc.value)


def test_installed_member_name_sanitized_via_require() -> None:
    evil = _EvilStr("forager/member.py")
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", evil)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("x"))  # type: ignore[arg-type]


def test_valid_strict_json_still_passes(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "valid.json"
    import json as _json

    payload = {"b": 1, "a": 2}
    canonical = _json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    p.write_bytes(canonical.encode("utf-8"))
    p.chmod(0o444)
    obj, raw = _strict_json_object(p)
    assert obj == {"a": 2, "b": 1}
    assert raw == canonical.encode("utf-8")
    assert _require_exact_str("key", "ok") == "ok"
