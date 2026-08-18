"""Leftover-identity gates for step-size and normalizer tracking configs."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.types import (
    NormalizerTrackingConfig,
    StepSizeTrackingConfig,
)


def test_step_size_tracking_config_rejects_leftover_identities() -> None:
    """Tracking intervals and include_bias must not keep leftover bool/float identities."""

    with pytest.raises(ValueError, match="interval"):
        StepSizeTrackingConfig(interval=True)
    with pytest.raises(ValueError, match="interval"):
        StepSizeTrackingConfig(interval=False)
    with pytest.raises(ValueError, match="interval"):
        StepSizeTrackingConfig(interval=np.bool_(True))
    with pytest.raises(ValueError, match="interval"):
        StepSizeTrackingConfig(interval=1.0)
    with pytest.raises(ValueError, match="interval"):
        StepSizeTrackingConfig(interval=float("nan"))
    with pytest.raises(ValueError, match="interval"):
        StepSizeTrackingConfig(interval=-1)
    with pytest.raises(ValueError, match="include_bias"):
        StepSizeTrackingConfig(interval=10, include_bias=1)
    with pytest.raises(ValueError, match="include_bias"):
        StepSizeTrackingConfig(interval=10, include_bias=0)
    with pytest.raises(ValueError, match="include_bias"):
        StepSizeTrackingConfig(interval=10, include_bias=np.bool_(False))

    legal = StepSizeTrackingConfig(interval=10, include_bias=False)
    assert legal.interval == 10
    assert legal.include_bias is False
    assert type(legal.interval) is int
    assert type(legal.include_bias) is bool
    zero = StepSizeTrackingConfig(interval=0)
    assert zero.interval == 0
    assert zero.include_bias is True


def test_normalizer_tracking_config_rejects_leftover_identities() -> None:
    """Normalizer recording intervals must not keep leftover bool/float identities."""

    with pytest.raises(ValueError, match="interval"):
        NormalizerTrackingConfig(interval=True)
    with pytest.raises(ValueError, match="interval"):
        NormalizerTrackingConfig(interval=False)
    with pytest.raises(ValueError, match="interval"):
        NormalizerTrackingConfig(interval=float("nan"))
    with pytest.raises(ValueError, match="interval"):
        NormalizerTrackingConfig(interval=-1)

    legal = NormalizerTrackingConfig(interval=0)
    assert legal.interval == 0
    assert type(legal.interval) is int
