"""Hostile-safe validation for timing utilities."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import alberta_framework.utils.timing as timing
from alberta_framework.utils.timing import Timer, format_duration


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")

    def __float__(self) -> float:
        type(self).calls += 1
        raise RuntimeError("float hook")


class _StringSubclass(str):
    pass


def test_format_rejects_bool() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(True)
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(np.bool_(True))


def test_format_rejects_hostile_float_without_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(_HostileFloat(1.0))
    assert _HostileFloat.calls == 0


def test_format_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(_StringSubclass("1.0"))


def test_format_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(float("nan"))
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(float("inf"))


def test_format_rejects_negative() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(-1.0)
    with pytest.raises(ValueError, match="formatting bound"):
        format_duration(1.0e13)
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(Fraction(10**10_000, 1))


def test_format_valid_cases() -> None:
    assert format_duration(0.5) == "0.50s"
    assert format_duration(90.5) == "1m 30.50s"
    assert format_duration(3665) == "1h 1m 5.00s"
    assert format_duration(Fraction(1, 2)) == "0.50s"
    assert format_duration(np.float64(1.0)) == "1.00s"


def test_timer_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        Timer(name=_StringSubclass("op"))


def test_timer_does_not_invoke_hostile_name_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

        def __str__(self) -> str:  # pragma: no cover
            raise RuntimeError("str hook")

    with pytest.raises(ValueError, match="exact string"):
        Timer(name=EvilStr("op"))


def test_timer_rejects_verbose_not_bool() -> None:
    with pytest.raises(ValueError, match="built-in bool"):
        Timer(name="op", verbose=1)
    with pytest.raises(ValueError, match="built-in bool"):
        Timer(name="op", verbose=_StringSubclass("true"))


def test_timer_repr_does_not_invoke_hostile_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

    evil = EvilStr("op")
    # Timer should reject EvilStr at construction, so repr not needed
    with pytest.raises(ValueError, match="exact string"):
        Timer(name=evil)
    # Valid timer repr should not use !r
    t = Timer(name="good")
    t.duration = 1.23
    r = repr(t)
    assert "good" in r
    assert "duration" in r


def test_timer_valid() -> None:
    t = Timer(name="ok", verbose=False)
    assert t.name == "ok"
    assert t.verbose is False
    with t:
        pass
    assert t.duration >= 0


def test_format_rejects_hostile_int() -> None:
    class HostileInt(int):
        def __repr__(self) -> str:  # pragma: no cover
            raise AssertionError("repr hook")

    with pytest.raises(ValueError, match="finite real number"):
        format_duration(HostileInt(1))


def test_format_rejects_hostile_class_spoof_without_hook() -> None:
    class HostileClass:
        @property
        def __class__(self) -> type:
            raise AssertionError("class hook executed")

    with pytest.raises(ValueError, match="finite real number"):
        format_duration(HostileClass())


def test_timer_rejects_nonfinite_and_hostile_clock_samples_without_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing.time, "perf_counter", lambda: float("nan"))
    with pytest.raises(ValueError, match="perf_counter"):
        Timer(verbose=False).__enter__()

    _HostileFloat.calls = 0
    monkeypatch.setattr(timing.time, "perf_counter", lambda: _HostileFloat(1.0))
    with pytest.raises(ValueError, match="perf_counter"):
        Timer(verbose=False).__enter__()
    assert _HostileFloat.calls == 0


def test_timer_rejects_decreasing_clock_and_closes_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter([10.0, 11.0, 10.5])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(samples))
    timer = Timer(verbose=False)
    timer.__enter__()
    assert timer.elapsed() == 1.0
    with pytest.raises(ValueError, match="monotonic"):
        timer.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match="active"):
        timer.elapsed()


def test_timer_invalid_exit_sample_preserves_block_exception_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter([10.0, 9.0])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(samples))
    timer = Timer(verbose=False)
    with pytest.raises(KeyError, match="authoritative"):
        with timer:
            raise KeyError("authoritative")
    assert timer._active is False


def test_timer_static_validation_failure_cleans_up_and_preserves_block_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 10.0)
    timer = Timer(verbose=False)
    with pytest.raises(KeyError, match="authoritative"):
        with timer:
            timer.name = _StringSubclass("mutated")  # type: ignore[assignment]
            raise KeyError("authoritative")
    assert timer._active is False

    timer = Timer(verbose=False)
    with pytest.raises(ValueError, match="exact string"):
        with timer:
            timer.name = _StringSubclass("mutated")  # type: ignore[assignment]
    assert timer._active is False


def test_timer_reporting_failure_does_not_mask_block_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter([1.0, 2.0])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(samples))

    def broken_reporter(message: str) -> None:
        del message
        raise RuntimeError("report failed")

    with pytest.raises(KeyError, match="authoritative"):
        with Timer(print_fn=broken_reporter):
            raise KeyError("authoritative")


def test_timer_rejects_reentry_and_sample_counter_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 1.0)
    timer = Timer(verbose=False)
    timer.__enter__()
    with pytest.raises(RuntimeError, match="already active"):
        timer.__enter__()
    timer._sample_count = 2_147_483_647
    with pytest.raises(ValueError, match="sample count"):
        timer.elapsed()


def test_timer_does_not_probe_falsey_callback_truthiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

    class Callback:
        def __bool__(self) -> bool:
            raise AssertionError("truthiness hook executed")

        def __call__(self, message: str) -> None:
            messages.append(message)

    samples = iter([1.0, 2.0])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(samples))
    with Timer("op", print_fn=Callback()):
        pass
    assert messages == ["op completed in 1.00s"]


def test_timer_declares_telemetry_only_semantics() -> None:
    assert "telemetry-only" in (Timer.__doc__ or "")
    assert "scientific evidence" in (Timer.__doc__ or "")
