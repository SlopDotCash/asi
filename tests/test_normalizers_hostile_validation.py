"""Hostile validation for normalizers trust boundary."""

from __future__ import annotations

import pathlib
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    migrate_legacy_multi_head_mlp_state,
)
from alberta_framework.core.normalizers import (
    EMANormalizer,
    StreamingBatchNormalizer,
    WelfordNormalizer,
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


class _HostileComparableStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileComparableStr.__eq__ must not be called")

    def __hash__(self) -> int:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileComparableStr.__hash__ must not be called")


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


def _legacy_normalizer_fields(type_name: str) -> dict[str, Any]:
    """Build one well-formed pre-v2 normalizer field manifest."""
    fields: dict[str, Any] = {
        "mean": jnp.zeros(2, dtype=jnp.float32),
        "var": jnp.ones(2, dtype=jnp.float32),
        "sample_count": jnp.asarray(0.0, dtype=jnp.float32),
    }
    extra = {
        "EMANormalizer": "decay",
        "StreamingBatchNormalizer": "momentum",
        "WelfordNormalizer": "p",
    }[type_name]
    fields[extra] = (
        jnp.zeros(2, dtype=jnp.float32)
        if extra == "p"
        else jnp.asarray(0.99, dtype=jnp.float32)
    )
    return fields


@pytest.mark.parametrize(
    ("type_name", "normalizer"),
    [
        ("EMANormalizer", EMANormalizer(decay=0.99)),
        ("StreamingBatchNormalizer", StreamingBatchNormalizer(momentum=0.99)),
        ("WelfordNormalizer", WelfordNormalizer()),
    ],
)
def test_migrate_keeps_well_formed_legacy_state(type_name: str, normalizer: Any) -> None:
    state = migrate_legacy_normalizer_state(
        _legacy_normalizer_fields(type_name),
        normalizer_type=type_name,
    )
    assert bool(normalizer.counter_status(state).update_available)
    assert normalizer.normalize_only(state, jnp.ones(2)).tolist() == pytest.approx(
        [1.0, 1.0], rel=1e-4
    )


_FLOAT32 = jnp.float32


@pytest.mark.parametrize(
    ("type_name", "name", "value", "error"),
    [
        ("EMANormalizer", "mean", jnp.asarray([jnp.nan, 0.0], dtype=_FLOAT32), ValueError),
        ("EMANormalizer", "mean", jnp.asarray([jnp.inf, 0.0], dtype=_FLOAT32), ValueError),
        ("EMANormalizer", "mean", jnp.asarray(0.0, dtype=_FLOAT32), TypeError),
        ("EMANormalizer", "mean", jnp.ones(2, dtype=jnp.complex64), TypeError),
        ("EMANormalizer", "mean", jnp.ones(2, dtype=jnp.int32), TypeError),
        ("EMANormalizer", "var", jnp.ones(3, dtype=_FLOAT32), TypeError),
        ("EMANormalizer", "var", jnp.asarray([-1.0, 1.0], dtype=_FLOAT32), ValueError),
        ("EMANormalizer", "var", jnp.asarray([jnp.inf, 1.0], dtype=_FLOAT32), ValueError),
        ("EMANormalizer", "decay", jnp.asarray(5.0, dtype=_FLOAT32), ValueError),
        ("EMANormalizer", "decay", jnp.asarray(-0.5, dtype=_FLOAT32), ValueError),
        ("EMANormalizer", "decay", jnp.asarray(jnp.nan, dtype=_FLOAT32), ValueError),
        ("EMANormalizer", "decay", jnp.full((1,), 0.99, dtype=_FLOAT32), TypeError),
        ("StreamingBatchNormalizer", "momentum", jnp.asarray(5.0, dtype=_FLOAT32), ValueError),
        ("WelfordNormalizer", "p", jnp.asarray([jnp.nan, 0.0], dtype=_FLOAT32), ValueError),
        ("WelfordNormalizer", "p", jnp.asarray([-1.0, 0.0], dtype=_FLOAT32), ValueError),
        ("WelfordNormalizer", "p", jnp.zeros(3, dtype=_FLOAT32), TypeError),
    ],
)
def test_migrate_rejects_malformed_legacy_statistics(
    type_name: str,
    name: str,
    value: Any,
    error: type[Exception],
) -> None:
    fields = {**_legacy_normalizer_fields(type_name), name: value}

    with pytest.raises(error, match=f"legacy normalizer {name}"):
        migrate_legacy_normalizer_state(fields, normalizer_type=type_name)


def test_multi_head_migration_rejects_non_finite_legacy_normalizer() -> None:
    """The STOMP legacy path reaches the normalizer through this learner migration."""
    learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), normalizer=EMANormalizer())
    state = learner.init(feature_dim=2, key=jr.key(0))
    learner_fields = {
        name: getattr(state, name)
        for name in type(state).__dataclass_fields__
        if name != "step_words"
    }
    learner_fields["normalizer_state"] = {
        **_legacy_normalizer_fields("EMANormalizer"),
        "mean": jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32),
    }

    with pytest.raises(ValueError, match="legacy normalizer mean"):
        migrate_legacy_multi_head_mlp_state(learner_fields)


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


@pytest.mark.parametrize(
    ("normalizer_type", "field", "value"),
    [
        ("EMANormalizer", "state_schema", "bad.schema"),
        ("WelfordNormalizer", "estimator_schema", "bad.estimator"),
        ("EMANormalizer", "estimator_semantics", "bad.semantics"),
    ],
)
def test_from_config_gates_strings_before_comparison_or_hashing(
    normalizer_type: str, field: str, value: str
) -> None:
    hostile = _HostileComparableStr(value)
    _HostileComparableStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        normalizer_from_config({"type": normalizer_type, field: hostile})
    assert _HostileComparableStr.calls == 0


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
