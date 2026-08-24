"""Regression coverage for flad_noise_component scale invariance
(issue #2393): a subnormal gradient whose squared norm underflows to zero
must still have its component removed from the perturbation — not silently
returned unchanged with valid=True."""

from __future__ import annotations

import jax.numpy as jnp

from alberta_framework.evaluation.optimizer_geometry import (
    flad_noise_component_transaction,
)


class TestFladNoiseScaleInvariance:
    def test_subnormal_gradient_component_still_removed(self) -> None:
        # Delta has an O(1) component along a subnormal gradient; the old
        # code's squared norm underflowed to 0 -> active=False -> the whole
        # gradient component was silently kept.
        delta = jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)
        grad = jnp.array(
            [2e-20, 1e-20, -1e-20, 0.5e-20], dtype=jnp.float32
        )
        noise, valid = flad_noise_component_transaction(delta, grad)
        assert bool(valid)
        # 去除梯度方向分量后：noise ⊥ grad（内积≈0）
        inner = jnp.vdot(noise, grad).real
        assert abs(float(inner)) < 1e-3, f"gradient component not removed: {inner}"

    def test_zero_gradient_returns_perturbation(self) -> None:
        delta = jnp.array([1.0, -2.0], dtype=jnp.float32)
        grad = jnp.zeros((2,), dtype=jnp.float32)
        noise, valid = flad_noise_component_transaction(delta, grad)
        assert bool(valid)
        assert jnp.allclose(noise, delta)

    def test_normal_gradient_unchanged_behavior(self) -> None:
        delta = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
        grad = jnp.array([2.0, 1.0, -1.0], dtype=jnp.float32)
        noise, valid = flad_noise_component_transaction(delta, grad)
        assert bool(valid)
        inner = jnp.vdot(noise, grad).real
        assert abs(float(inner)) < 1e-4

    def test_scale_invariance_of_result(self) -> None:
        # 结果不依赖梯度的绝对尺度
        delta = jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)
        grad_small = jnp.array(
            [2e-20, 1e-20, -1e-20, 0.5e-20], dtype=jnp.float32
        )
        grad_big = grad_small * 1e10
        noise_small, _ = flad_noise_component_transaction(delta, grad_small)
        noise_big, _ = flad_noise_component_transaction(delta, grad_big)
        assert jnp.allclose(noise_small, noise_big, rtol=1e-3)
