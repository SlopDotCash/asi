"""Protocol ceilings for public bootstrap resample counts.

Documented last-fit in tests is n_bootstrap=500; the function default is 10_000.
Origin handed unbounded counts to range(n_bootstrap) with no last-fit reject.
"""

from __future__ import annotations

import pytest

from alberta_framework.utils.statistics import _BOOTSTRAP_MAX_COUNT, bootstrap_ci


def test_documented_protocol_ceiling() -> None:
    assert _BOOTSTRAP_MAX_COUNT == 10_000


@pytest.mark.parametrize("value", [10**12, 2**31 - 1, 10_001])
def test_rejects_oversized_bootstrap_counts(value: int) -> None:
    with pytest.raises(ValueError, match="n_bootstrap count must be"):
        bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=value)
