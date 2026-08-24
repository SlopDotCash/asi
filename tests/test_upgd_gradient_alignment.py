"""Regression coverage for UPGD gradient alignment scale invariance
(issue #2389): identical gradients must report cosine 1.0 at every
magnitude — not 0.0 below 1e-6 and not nan above 1e9."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.upgd import UPGDLearner


class TestGradientAlignmentScaleFree:
    @pytest.mark.parametrize(
        "scale",
        [1e-10, 1e-8, 1e-6, 1.0, 1e3, 1e6, 1e9, 1e12],
    )
    def test_identical_gradients_report_cosine_one(self, scale: float) -> None:
        base = jnp.array([[1.0, -2.0, 0.5], [0.25, 1.0, -1.0]], dtype=jnp.float32)
        prev = (base * scale,)
        curr = (base * scale,)
        cosine = UPGDLearner._gradient_alignment(prev, curr)
        assert jnp.isfinite(cosine)
        assert float(cosine) == pytest.approx(1.0, abs=1e-4), (
            f"scale {scale}: identical gradients reported {float(cosine)}"
        )

    @pytest.mark.parametrize(
        "scale",
        [1e-10, 1.0, 1e12],
    )
    def test_orthogonal_gradients_report_zero(self, scale: float) -> None:
        a = jnp.array([[1.0, 0.0]], dtype=jnp.float32) * scale
        b = jnp.array([[0.0, 1.0]], dtype=jnp.float32) * scale
        cosine = UPGDLearner._gradient_alignment((a,), (b,))
        assert jnp.isfinite(cosine)
        assert abs(float(cosine)) < 1e-3

    def test_opposite_gradients_report_minus_one(self) -> None:
        a = jnp.array([[2.0, -1.0]], dtype=jnp.float32)
        cosine = UPGDLearner._gradient_alignment((a,), (-a,))
        assert float(cosine) == pytest.approx(-1.0, abs=1e-4)

    def test_zero_gradient_returns_zero(self) -> None:
        a = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        zero = jnp.zeros_like(a)
        cosine = UPGDLearner._gradient_alignment((a,), (zero,))
        assert float(cosine) == 0.0
