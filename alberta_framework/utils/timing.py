"""Timing utilities for measuring and reporting experiment durations.

This module provides a simple Timer context manager for measuring execution time
and formatting durations in a human-readable format.

Examples
--------
```python
from alberta_framework.utils.timing import Timer

with Timer("Training"):
    # run training code
    pass
# Output: Training completed in 1.23s

# Or capture the duration:
with Timer("Experiment") as t:
    # run experiment
    pass
print(f"Took {t.duration:.2f} seconds")
```
"""

import math
import time
from collections.abc import Callable
from fractions import Fraction
from types import TracebackType

import numpy as np

_ACTUAL_INT_TYPES: frozenset[type] = frozenset(
    {int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")}
)
_ACTUAL_FLOAT_TYPES: frozenset[type] = frozenset(
    {
        float,
        Fraction,
        np.dtype("e").type,
        np.dtype("f").type,
        np.dtype("d").type,
        np.dtype("g").type,
    }
)
_ALLOWED_REAL_TYPES: frozenset[type] = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES
_MAX_DURATION_SECONDS = 1.0e12
_MAX_TIMER_SAMPLES = 2_147_483_647


def _has_exact_type(value: object, allowed: frozenset[type]) -> bool:
    """Match a concrete type without invoking an untrusted metaclass hook."""
    actual_type = type(value)
    return any(actual_type is allowed_type for allowed_type in allowed)


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_finite_real(name: str, value: object) -> float:
    if type(value) is bool or type(value) is np.bool_:
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    if not _has_exact_type(value, _ALLOWED_REAL_TYPES):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _read_perf_counter() -> float:
    sample = time.perf_counter()
    if type(sample) is not float or not math.isfinite(sample) or sample < 0.0:
        raise ValueError("perf_counter must return a non-negative finite built-in float")
    return sample


def format_duration(seconds: object) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1.23s", "2m 30.5s", or "1h 5m 30s"

    Examples
    --------
    ```python
    format_duration(0.5)   # Returns: '0.50s'
    format_duration(90.5)  # Returns: '1m 30.50s'
    format_duration(3665)  # Returns: '1h 1m 5.00s'
    ```
    """
    sec = _require_finite_real("seconds", seconds)
    if sec < 0:
        raise ValueError("seconds must be a finite real number")
    if sec > _MAX_DURATION_SECONDS:
        raise ValueError("seconds exceeds the telemetry formatting bound")
    rounded_seconds = round(sec, 2)
    if rounded_seconds < 60:
        return f"{rounded_seconds:.2f}s"
    elif rounded_seconds < 3600:
        minutes = int(rounded_seconds // 60)
        secs = rounded_seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(rounded_seconds // 3600)
        remaining = rounded_seconds % 3600
        minutes = int(remaining // 60)
        secs = remaining % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


class Timer:
    """Context manager for timing code execution.

    Measures monotonic process-local telemetry for a block and optionally
    prints it. Timing is telemetry-only: this utility does not qualify timing
    protocols or create performance/scientific evidence.

    Attributes:
        name: Description of what is being timed
        duration: Elapsed time in seconds (available after context exits)
        start_time: Timestamp when timing started
        end_time: Timestamp when timing ended

    Examples
    --------
    ```python
    with Timer("Training loop"):
        for i in range(1000):
            pass
    # Output: Training loop completed in 0.01s

    # Silent timing (no print):
    with Timer("Silent", verbose=False) as t:
        time.sleep(0.1)
    print(f"Elapsed: {t.duration:.2f}s")
    # Output: Elapsed: 0.10s

    # Custom print function:
    with Timer("Custom", print_fn=lambda msg: print(f">> {msg}")):
        pass
    # Output: >> Custom completed in 0.00s
    ```
    """

    def __init__(
        self,
        name: object = "Operation",
        verbose: object = True,
        print_fn: Callable[[str], None] | None = None,
    ):
        """Initialize the timer.

        Args:
            name: Description of the operation being timed
            verbose: Whether to print the duration when done
            print_fn: Custom print function (defaults to built-in print)
        """
        validated_name = _require_exact_str("name", name)
        if type(verbose) is not bool:
            raise ValueError("verbose must be a built-in bool")
        if print_fn is not None and not callable(print_fn):
            raise ValueError("print_fn must be callable")
        self.name: str = validated_name
        self.verbose: bool = verbose
        self.print_fn = print if print_fn is None else print_fn
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration: float = 0.0
        self._last_sample: float = 0.0
        self._sample_count: int = 0
        self._active: bool = False

    def _validate_static_contract(self) -> None:
        _require_exact_str("name", self.name)
        if type(self.verbose) is not bool:
            raise ValueError("verbose must be a built-in bool")
        if not callable(self.print_fn):
            raise ValueError("print_fn must be callable")

    def _record_sample(self) -> float:
        if type(self._sample_count) is not int or not 0 <= self._sample_count < _MAX_TIMER_SAMPLES:
            raise ValueError("timer sample count is outside the signed-int32 bound")
        sample = _read_perf_counter()
        if sample < self._last_sample:
            raise ValueError("perf_counter samples must be monotonic")
        self._last_sample = sample
        self._sample_count += 1
        return sample

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self._validate_static_contract()
        if self._active:
            raise RuntimeError("Timer cannot be entered while already active")
        self._last_sample = 0.0
        self._sample_count = 0
        self.start_time = self._record_sample()
        self.end_time = 0.0
        self.duration = 0.0
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the timer and report completion or failure, never both."""
        try:
            self._validate_static_contract()
            if not self._active:
                raise RuntimeError("Timer cannot exit before it is entered")
            end_time = self._record_sample()
            duration = end_time - self.start_time
            if not math.isfinite(duration) or not 0.0 <= duration <= _MAX_DURATION_SECONDS:
                raise ValueError("timer duration is outside the finite telemetry bound")
            self.end_time = end_time
            self.duration = duration
        except BaseException:
            if exc_type is None:
                raise
            return
        finally:
            self._active = False

        if self.verbose:
            try:
                formatted = format_duration(self.duration)
                if exc_type is None:
                    self.print_fn(f"{self.name} completed in {formatted}")
                else:
                    self.print_fn(f"{self.name} failed after {formatted}")
            except BaseException:
                if exc_type is None:
                    raise

    def elapsed(self) -> float:
        """Get elapsed time since timer started (can be called during execution).

        Returns:
            Elapsed time in seconds
        """
        self._validate_static_contract()
        if not self._active:
            raise RuntimeError("elapsed requires an active Timer")
        start_time = _require_finite_real("start_time", self.start_time)
        if start_time < 0.0:
            raise ValueError("start_time must be non-negative")
        elapsed = self._record_sample() - start_time
        if not 0.0 <= elapsed <= _MAX_DURATION_SECONDS:
            raise ValueError("elapsed timing is outside the finite telemetry bound")
        return elapsed

    def __repr__(self) -> str:
        """Return string representation."""
        self._validate_static_contract()
        duration = _require_finite_real("duration", self.duration)
        if not 0.0 <= duration <= _MAX_DURATION_SECONDS:
            raise ValueError("duration is outside the finite telemetry bound")
        if duration > 0:
            return f"Timer(name={repr(self.name)}, duration={duration:.2f}s)"
        return f"Timer(name={repr(self.name)})"
