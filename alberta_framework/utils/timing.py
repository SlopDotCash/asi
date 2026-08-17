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

_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)
_ACTUAL_FLOAT_TYPES = frozenset(
    {
        float,
        Fraction,
        np.dtype("e").type,
        np.dtype("f").type,
        np.dtype("d").type,
        np.dtype("g").type,
    }
)
_ALLOWED_REAL_TYPES = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


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

    Measures wall-clock time for a block of code and optionally prints
    the duration when the block completes.

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
        _require_exact_str("name", name)
        if type(verbose) is not bool:
            raise ValueError("verbose must be a built-in bool")
        if print_fn is not None and not callable(print_fn):
            raise ValueError("print_fn must be callable")
        self.name = name
        self.verbose = verbose
        self.print_fn = print_fn or print
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the timer and report completion or failure, never both."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        if self.verbose:
            formatted = format_duration(self.duration)
            if exc_type is None:
                self.print_fn(f"{self.name} completed in {formatted}")
            else:
                self.print_fn(f"{self.name} failed after {formatted}")

    def elapsed(self) -> float:
        """Get elapsed time since timer started (can be called during execution).

        Returns:
            Elapsed time in seconds
        """
        return time.perf_counter() - self.start_time

    def __repr__(self) -> str:
        """Return string representation."""
        if self.duration > 0:
            return f"Timer(name={self.name}, duration={self.duration:.2f}s)"
        return f"Timer(name={self.name})"
