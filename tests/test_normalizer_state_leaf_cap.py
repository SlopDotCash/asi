"""Reject oversized host pytrees before measuring normalizer persist bytes."""

from __future__ import annotations

import pytest

from alberta_framework.core.normalizers import (
    _MAX_NORMALIZER_STATE_LEAVES,
    EMANormalizer,
    measure_normalizer_state_nbytes,
)

pytestmark = pytest.mark.unit


def test_measure_nbytes_rejects_oversized_host_tuple() -> None:
    assert _MAX_NORMALIZER_STATE_LEAVES == 4096
    with pytest.raises(
        ValueError,
        match=r"normalizer state length must be an integer in \[0, 4096\]",
    ):
        measure_normalizer_state_nbytes((0,) * (_MAX_NORMALIZER_STATE_LEAVES + 1))


def test_measure_nbytes_still_counts_ema_state() -> None:
    state = EMANormalizer().init(4)
    assert measure_normalizer_state_nbytes(state) > 0
