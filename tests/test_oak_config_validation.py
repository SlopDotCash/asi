"""Mutation-resistant host validation for OaK and option-keyboard configs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest

from alberta_framework.core.oak import (
    KEYBOARD_CHORD_LEARNER_CONFIG_SCHEMA,
    OAK_CONFIG_SCHEMA,
    KeyboardChordLearnerConfig,
    OaKConfig,
)
from alberta_framework.core.options import STOMPConfig

_INTEGER_DTYPES = tuple(np.dtype(code).type for code in "bBhHiIlLqQ")
_REAL_DTYPES = tuple(np.dtype(code).type for code in "efdg")
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MAX_KEYBOARD_OPTIONS = ((2**32 - 1) - 8) // 4


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


class _StringSubclass(str):
    pass


class _RaisingInt(int):
    def __index__(self) -> int:
        raise AssertionError("hostile index hook ran")

    def __repr__(self) -> str:
        raise AssertionError("hostile repr hook ran")


class _RaisingFloat(float):
    def __float__(self) -> float:
        raise AssertionError("hostile float hook ran")

    def __repr__(self) -> str:
        raise AssertionError("hostile repr hook ran")


class _RaisingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise AssertionError("hostile mapping hook ran")

    def __iter__(self) -> Iterator[str]:
        yield "type"

    def __len__(self) -> int:
        return 1


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_integer_families_are_canonicalized(dtype: type[np.generic]) -> None:
    oak = OaKConfig(min_steps_before_curation=dtype(3))  # type: ignore[arg-type]
    keyboard = KeyboardChordLearnerConfig(n_options=dtype(3))  # type: ignore[arg-type]
    assert type(oak.min_steps_before_curation) is int
    assert type(keyboard.n_options) is int


@pytest.mark.parametrize("value", [True, 1.0, _IntSubclass(1), "1"])
def test_integer_spoofs_are_rejected_without_hooks(value: object) -> None:
    with pytest.raises(ValueError, match="min_steps_before_curation"):
        OaKConfig(min_steps_before_curation=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_options"):
        KeyboardChordLearnerConfig(n_options=value)  # type: ignore[arg-type]


def test_hostile_integer_subclass_is_rejected_without_hooks() -> None:
    value = _RaisingInt(1)
    with pytest.raises(ValueError, match="min_steps_before_curation"):
        OaKConfig(min_steps_before_curation=value)
    with pytest.raises(ValueError, match="n_options"):
        KeyboardChordLearnerConfig(n_options=value)


def test_integer_endpoints_and_derived_keyboard_allocation_bound() -> None:
    assert OaKConfig(min_steps_before_curation=2**64 - 1).min_steps_before_curation == 2**64 - 1
    with pytest.raises(ValueError, match="min_steps_before_curation"):
        OaKConfig(min_steps_before_curation=2**64)
    cfg = KeyboardChordLearnerConfig(n_options=_MAX_KEYBOARD_OPTIONS)
    assert cfg.n_options * 4 + 8 <= 2**32 - 1
    with pytest.raises(ValueError, match="n_options"):
        KeyboardChordLearnerConfig(n_options=_MAX_KEYBOARD_OPTIONS + 1)


@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_real_dtype_families_are_canonicalized(dtype: type[np.generic]) -> None:
    half: Any = dtype(0.5)
    quarter: Any = dtype(0.25)
    tenth: Any = dtype(0.1)
    two: Any = dtype(2.0)
    oak = OaKConfig(utility_ema_decay=half, curation_threshold=quarter)
    keyboard = KeyboardChordLearnerConfig(
        n_options=2,
        step_size=tenth,
        baseline_decay=half,
        l2_penalty=tenth,
        max_norm=two,
    )
    assert type(oak.utility_ema_decay) is float
    assert all(
        type(value) is float
        for value in (
            keyboard.step_size,
            keyboard.baseline_decay,
            keyboard.l2_penalty,
            keyboard.max_norm,
        )
    )


@pytest.mark.parametrize("value", [True, _FloatSubclass(0.5), 1j, "0.5"])
def test_real_spoofs_are_rejected_without_hooks(value: object) -> None:
    with pytest.raises(ValueError, match="utility_ema_decay"):
        OaKConfig(utility_ema_decay=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step_size"):
        KeyboardChordLearnerConfig(n_options=2, step_size=value)  # type: ignore[arg-type]


def test_hostile_float_subclass_is_rejected_without_hooks() -> None:
    value = _RaisingFloat(0.5)
    with pytest.raises(ValueError, match="utility_ema_decay"):
        OaKConfig(utility_ema_decay=value)
    with pytest.raises(ValueError, match="step_size"):
        KeyboardChordLearnerConfig(n_options=2, step_size=value)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), 2 * _FLOAT32_MAX, 10**1000]
)
def test_all_float_fields_reject_nonfinite_or_overflow(value: float) -> None:
    for field in ("utility_ema_decay", "curation_threshold"):
        with pytest.raises(ValueError, match=field):
            OaKConfig(**{field: value})  # type: ignore[arg-type]
    for field in ("step_size", "baseline_decay", "l2_penalty", "max_norm"):
        with pytest.raises(ValueError, match=field):
            KeyboardChordLearnerConfig(n_options=2, **{field: value})


def test_float_endpoints_subnormals_and_negative_zero() -> None:
    oak = OaKConfig(utility_ema_decay=1.0, curation_threshold=_FLOAT32_MAX)
    assert oak.utility_ema_decay == 1.0
    for field in ("utility_ema_decay", "curation_threshold"):
        with pytest.raises(ValueError, match=field):
            OaKConfig(**{field: _FLOAT32_TINY / 2})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="baseline_decay"):
        KeyboardChordLearnerConfig(n_options=2, baseline_decay=1.0)
    for field in ("step_size", "l2_penalty", "max_norm"):
        with pytest.raises(ValueError, match=field):
            KeyboardChordLearnerConfig(n_options=2, **{field: _FLOAT32_TINY / 2})
    oak_zero = OaKConfig(curation_threshold=-0.0)
    keyboard_zero = KeyboardChordLearnerConfig(n_options=2, step_size=-0.0, l2_penalty=-0.0)
    assert np.signbit(oak_zero.curation_threshold) is np.bool_(False)
    assert not np.signbit(keyboard_zero.step_size)
    assert not np.signbit(keyboard_zero.l2_penalty)


def test_exact_nested_config_and_serialization_contracts() -> None:
    class _STOMPSubclass(STOMPConfig):
        pass

    with pytest.raises(ValueError, match="exact STOMPConfig"):
        OaKConfig(stomp=_STOMPSubclass())

    oak = OaKConfig()
    oak_payload = oak.to_config()
    assert oak_payload["schema"] == OAK_CONFIG_SCHEMA
    assert OaKConfig.from_config(oak_payload) == oak
    legacy = dict(oak_payload)
    legacy.pop("schema")
    legacy.pop("min_steps_before_curation")
    assert OaKConfig.from_config(legacy).min_steps_before_curation == 0

    keyboard = KeyboardChordLearnerConfig(n_options=2)
    keyboard_payload = keyboard.to_config()
    assert keyboard_payload["schema"] == KEYBOARD_CHORD_LEARNER_CONFIG_SCHEMA
    assert KeyboardChordLearnerConfig.from_config(keyboard_payload) == keyboard

    for payload, loader in (
        (oak_payload, OaKConfig.from_config),
        (keyboard_payload, KeyboardChordLearnerConfig.from_config),
    ):
        for field in ("schema", "type"):
            hostile = dict(payload)
            hostile[field] = _StringSubclass(str(payload[field]))
            with pytest.raises(ValueError):
                loader(hostile)
        extra = dict(payload)
        extra["unexpected"] = 1
        with pytest.raises(ValueError, match="fields"):
            loader(extra)
        with pytest.raises(ValueError, match="readable mapping"):
            loader(_RaisingMapping())
