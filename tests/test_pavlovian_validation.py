"""Unit tests for the pavlovian stream's scalar validation gates.

Locks the type gates in ``_require_finite_real``,
``_require_nonnegative_float32``, and ``_require_unit_interval`` against
``__class__``-spoofed non-real scalars (issue #600). ``isinstance`` consults
the overridable ``__class__`` attribute, so an object whose ``__class__``
property returns ``float`` used to sail through validation into
``ClassicalConditioningStream``'s internal state.
"""

from __future__ import annotations

import pytest

from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    PavlovianPhase,
    partial_reinforcement_scenario,
)

pytestmark = pytest.mark.unit


class _FloatSpoof:
    """Not a Real at all, but reports ``float`` through ``__class__``."""

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return self._value


class _RaisingFloatSpoof:
    """A ``__class__`` spoof whose ``__float__`` hook raises when trusted."""

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        raise RuntimeError("untrusted __float__ hook executed")


def _valid_phase(**overrides: object) -> PavlovianPhase:
    fields: dict[str, object] = {
        "name": "acq",
        "n_steps": 10,
        "cs_active": (0,),
        "compound_index": -1,
        "cs_us_contingency": 1.0,
    }
    fields.update(overrides)
    return PavlovianPhase(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("spoof", [_FloatSpoof(0.5), _RaisingFloatSpoof()])
def test_construct_rejects_class_spoofed_distractor_prob(spoof: object) -> None:
    """An in-domain host value must not smuggle a non-Real type past the gate."""
    with pytest.raises(ValueError, match="distractor_prob must be in"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            distractor_prob=spoof,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("spoof", [_FloatSpoof(0.05), _RaisingFloatSpoof()])
def test_construct_rejects_class_spoofed_noise_std(spoof: object) -> None:
    """The local noise_std gate must reject spoofs itself, not lean on float32 rounding."""
    with pytest.raises(ValueError, match="noise_std must be a non-negative finite real"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            noise_std=spoof,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("spoof", [_FloatSpoof(1.0), _RaisingFloatSpoof()])
def test_construct_rejects_class_spoofed_phase_contingency(spoof: object) -> None:
    """A spoofed cs_us_contingency on a phase must fail stream construction."""
    with pytest.raises(ValueError, match="cs_us_contingency must be in"):
        ClassicalConditioningStream(
            phases=(_valid_phase(cs_us_contingency=spoof),),
        )


@pytest.mark.parametrize("spoof", [_FloatSpoof(0.5), _RaisingFloatSpoof()])
def test_partial_reinforcement_rejects_class_spoofed_p(spoof: object) -> None:
    """The partial-reinforcement helper shares the same hardened unit-interval gate."""
    with pytest.raises(ValueError, match="p must be in"):
        partial_reinforcement_scenario(p=spoof)  # type: ignore[arg-type]


def test_spoofed_scalars_never_leak_the_raw_float_hook_exception() -> None:
    """A spoof with a raising ``__float__`` must surface the documented ValueError."""
    for kwargs in (
        {"distractor_prob": _RaisingFloatSpoof()},
        {"noise_std": _RaisingFloatSpoof()},
    ):
        with pytest.raises(ValueError):
            ClassicalConditioningStream(
                phases=(_valid_phase(),),
                **kwargs,  # type: ignore[arg-type]
            )
