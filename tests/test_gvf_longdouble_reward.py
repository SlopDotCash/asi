"""Regression coverage for #2283: GVF terminal_reward must accept every numpy
floating family, including np.longdouble.

The gate `_ACTUAL_REAL_SCALAR_TYPES` hand-enumerated float16/32/64 and
omitted longdouble, so `terminal_reward=np.longdouble(1.5)` was rejected
while its sibling fields `gamma`/`lamda` (checked via issubclass(Real))
accepted it, and while the float32 narrowing helper beneath the gate supports
it. The fix derives the set from dtype codes ("efdg"), matching the
_ACTUAL_FLOAT_TYPES precedent in _float32.py.
"""

import numpy as np

from alberta_framework import DemonType, GVFSpec
from alberta_framework.core.types import _ACTUAL_REAL_SCALAR_TYPES


def _base() -> dict:
    return {
        "name": "d",
        "demon_type": DemonType.PREDICTION,
        "gamma": 0.0,
        "lamda": 0.0,
        "cumulant_index": 0,
    }


def test_derived_set_accepts_every_float_dtype_code() -> None:
    for code in "efdg":
        t = np.dtype(code).type
        assert t in _ACTUAL_REAL_SCALAR_TYPES, f"{code} ({t}) missing"


def test_terminal_reward_accepts_longdouble() -> None:
    spec = GVFSpec(**_base(), terminal_reward=np.longdouble(1.5))
    assert spec.terminal_reward == 1.5


def test_terminal_reward_accepts_all_numpy_floats() -> None:
    for value in (np.float16(1.5), np.float32(1.5), np.float64(1.5), np.longdouble(1.5)):
        spec = GVFSpec(**_base(), terminal_reward=value)
        assert spec.terminal_reward == 1.5


def test_sibling_fields_accept_longdouble_too() -> None:
    # The issue's reproduction: gamma/lamda already accepted longdouble via
    # the issubclass(Real) path; terminal_reward must agree.
    base = _base()
    base.update(
        gamma=np.longdouble(0.5),
        lamda=np.longdouble(0.5),
        terminal_reward=np.longdouble(1.5),
    )
    spec = GVFSpec(**base)
    assert spec.gamma == 0.5
    assert spec.lamda == 0.5
    assert spec.terminal_reward == 1.5


def test_set_matches_dtype_family() -> None:
    expected = frozenset({float, int, *(np.dtype(code).type for code in "efdg")})
    assert _ACTUAL_REAL_SCALAR_TYPES == expected
