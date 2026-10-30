"""A positive synthetic-stream scale must not narrow silently to binary32 zero.

``_require_finite_nonnegative_float32`` narrows each host value to the float32
value the step kernel actually executes. A positive request below the binary32
rounding boundary narrowed to ``0.0``, so the constructor accepted
``drift_rate=1e-46`` and then ran a perfectly stationary stream, accepted
``noise_std=1e-46`` and then added no noise, and accepted ``feature_std=1e-46``
and then emitted identically zero observations. The recorded configuration
disagreed with the executed stream in every case.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.streams.synthetic import (
    AbruptChangeStream,
    DynamicScaleShiftStream,
    HiddenStateAR2Stream,
    RandomWalkStream,
    ScaleDriftStream,
    SuttonExperiment1Stream,
)

# Positive in binary64, but rounds to +0.0 in binary32.
_COLLAPSING = 1e-46
# The smallest positive binary32 value: representable, never a collapse.
_SMALLEST_SUBNORMAL = float(np.nextafter(np.float32(0.0), np.float32(1.0)))


def _hidden_ar2(**kwargs: float) -> object:
    return HiddenStateAR2Stream(feature_dim=4, visible_dim=2, **kwargs)


def _dynamic_scale_shift(**kwargs: float) -> object:
    return DynamicScaleShiftStream(feature_dim=4, **kwargs)


def _abrupt_change(**kwargs: float) -> object:
    return AbruptChangeStream(feature_dim=4, **kwargs)


def _scale_drift(**kwargs: float) -> object:
    return ScaleDriftStream(feature_dim=4, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["drift_rate", "noise_std", "feature_std"])
def test_random_walk_rejects_scales_that_narrow_to_zero(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RandomWalkStream(feature_dim=4, **{field: _COLLAPSING})  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ctor", "field"),
    [
        (_hidden_ar2, "innovation_std"),
        (_hidden_ar2, "target_noise_std"),
        (SuttonExperiment1Stream, "noise_std"),
        (SuttonExperiment1Stream, "bias_drift_rate"),
        (_dynamic_scale_shift, "noise_std"),
        (_abrupt_change, "feature_std"),
        (_scale_drift, "noise_std"),
    ],
)
def test_sibling_streams_reject_scales_that_narrow_to_zero(ctor, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        ctor(**{field: _COLLAPSING})


@pytest.mark.unit
def test_random_walk_accepts_exact_zero_scales() -> None:
    stream = RandomWalkStream(feature_dim=4, drift_rate=0.0, noise_std=0.0, feature_std=0.0)
    assert stream.feature_dim == 4


@pytest.mark.unit
@pytest.mark.parametrize("field", ["drift_rate", "noise_std", "feature_std"])
def test_random_walk_accepts_the_smallest_representable_positive_scale(field: str) -> None:
    stream = RandomWalkStream(feature_dim=4, **{field: _SMALLEST_SUBNORMAL})
    assert stream.feature_dim == 4
