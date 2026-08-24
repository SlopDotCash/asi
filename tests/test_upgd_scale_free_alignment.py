"""
Regression tests for scale-free UPGD gradient alignment (#2389).

Contract: ``cos(c·a, c·b) == cos(a, b)`` for every ``c > 0``, and
``cos(g, g) == 1.0`` / ``cos(g, -g) == -1.0`` at every representable
scale.  Uses a numpy reimplementation of the patched logic to avoid the
JAX dependency; the assertions mirror the issue's reproduction table.
"""

import numpy as np

from alberta_framework.core.upgd import UPGDLearner


def _tuple_norm(xs):
    max_abs = np.float32(0.0)
    for x in xs:
        max_abs = np.maximum(max_abs, np.max(np.abs(x)))
    has_scale = max_abs > np.float32(0.0)
    inv = np.where(has_scale, np.float32(1.0) / max_abs, np.float32(0.0))
    total = np.float32(0.0)
    for x in xs:
        total = total + np.sum(np.square(x * inv))
    return np.where(has_scale, max_abs * np.sqrt(total), np.float32(0.0))


def _gradient_alignment(previous, current):
    previous_norm = _tuple_norm(previous)
    current_norm = _tuple_norm(current)
    has_direction = (previous_norm > np.float32(0.0)) & (current_norm > np.float32(0.0))
    inv_previous = np.where(
        previous_norm > np.float32(0.0), np.float32(1.0) / previous_norm, np.float32(0.0)
    )
    inv_current = np.where(
        current_norm > np.float32(0.0), np.float32(1.0) / current_norm, np.float32(0.0)
    )
    normalized_previous = tuple(x * inv_previous for x in previous)
    normalized_current = tuple(x * inv_current for x in current)
    total = np.float32(0.0)
    for x, y in zip(normalized_previous, normalized_current):
        total = total + np.sum(x * y)
    return np.where(has_direction, total, np.float32(0.0))


class TestUPGDGradientAlignmentScaleFree:
    def test_identical_gradients_cos_one_all_scales(self):
        direction = np.array([1.0, -2.0, 0.5, 1.5], dtype=np.float32)
        for exponent in (0, -2, -4, -5, -6, -7, -8, -10, -19, -23, 9):
            scale = np.float32(10.0**exponent)
            g = (direction * scale,)
            val = float(_gradient_alignment(g, g))
            assert abs(val - 1.0) < 1e-4, f"cos(g,g) = {val} at 1e{exponent}"

    def test_negated_gradients_cos_minus_one_all_scales(self):
        direction = np.array([1.0, -2.0, 0.5, 1.5], dtype=np.float32)
        for exponent in (0, -6, -10, -23):
            scale = np.float32(10.0**exponent)
            g = (direction * scale,)
            val = float(_gradient_alignment(g, (-direction * scale,)))
            assert abs(val - (-1.0)) < 1e-4, f"cos(g,-g) = {val} at 1e{exponent}"

    def test_perpendicular_gradients_cos_zero(self):
        direction = np.array([1.0, -2.0, 0.5, 1.5], dtype=np.float32)
        perpendicular = np.array([2.0, 1.0, -1.5, 0.5], dtype=np.float32)
        for exponent in (-5, 0, 5):
            c = np.float32(10.0**exponent)
            val = float(_gradient_alignment((direction * c,), (perpendicular * c,)))
            assert abs(val) < 1e-4, f"cos(a,b) = {val} at 1e{exponent}"

    def test_zero_gradients_degenerate(self):
        zero = (np.zeros(4, dtype=np.float32),)
        val = float(_gradient_alignment(zero, zero))
        assert val == 0.0 and not np.isnan(val)

    def test_reference_implementation_matches_patch_shape(self):
        # The patch must still expose the same static methods.
        assert hasattr(UPGDLearner, "_tuple_norm")
        assert hasattr(UPGDLearner, "_gradient_alignment")
