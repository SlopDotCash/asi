"""Regression coverage for spectral_matrix_sign normalization
(issues #2379/#2391): matrices with sub-1e-12 Frobenius norm must not be
laundered into a numerically zero "valid" sign, and the exact zero matrix
keeps its exact zero."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.evaluation.optimizer_geometry import (
    spectral_matrix_sign,
    spectral_matrix_sign_transaction,
)


class TestSpectralMatrixSignNormalization:
    def test_subnormal_matrix_not_laundered(self) -> None:
        # Nonzero float32 matrix whose entries are all subnormal: the old
        # 1e-12 floor produced a numerically zero sign with valid=True.
        matrix = jnp.array(
            [[1e-30, 0.0], [0.0, 1e-30]], dtype=jnp.float32
        )
        sign, valid = spectral_matrix_sign_transaction(matrix)
        assert bool(valid)
        # 非零矩阵的 sign 不能是零矩阵（修复后输出 7.59e-30 非零）
        assert jnp.abs(sign).sum() > 0.0
        # 用严格相等验证非零（allclose 默认 atol 会把小值当零）
        assert not jnp.array_equal(sign, jnp.zeros_like(sign))

    def test_zero_matrix_keeps_exact_zero(self) -> None:
        matrix = jnp.zeros((2, 2), dtype=jnp.float32)
        sign, valid = spectral_matrix_sign_transaction(matrix)
        assert bool(valid)
        assert jnp.allclose(sign, jnp.zeros_like(sign))

    def test_normal_matrix_unchanged_behavior(self) -> None:
        matrix = jnp.array([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float32)
        sign = spectral_matrix_sign(matrix)
        # 对角正矩阵的 sign 近似单位阵
        assert jnp.allclose(sign, jnp.eye(2), atol=0.5)

    def test_small_norm_still_orthogonalizes(self) -> None:
        # 1e-13 范数（低于旧 floor）—— sign 应该是正交的（接近单位阵）
        matrix = jnp.array([[1e-13, 0.0], [0.0, 1e-13]], dtype=jnp.float32)
        sign, valid = spectral_matrix_sign_transaction(matrix)
        assert bool(valid)
        assert jnp.abs(sign).sum() > 0.0
