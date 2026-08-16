"""Tests for sliced isotropic Gaussian regularization."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.sigreg import (
    SIGRegConfig,
    epps_pulley_gaussian_statistic,
    sample_sigreg_directions,
    sigreg_diagnostics,
    sliced_sigreg_loss,
)


def test_sigreg_config_roundtrip_and_direction_shapes() -> None:
    config = SIGRegConfig(n_projections=7, kernel_width=1.5)
    restored = SIGRegConfig.from_config(config.to_config())
    assert restored == config

    directions = sample_sigreg_directions(jr.key(0), latent_dim=5, config=config)

    chex.assert_shape(directions, (7, 5))
    chex.assert_trees_all_close(
        jnp.linalg.norm(directions, axis=1),
        jnp.ones((7,), dtype=jnp.float32),
        atol=1.0e-5,
    )


def test_epps_pulley_statistic_penalizes_collapsed_samples() -> None:
    gaussian = jr.normal(jr.key(1), (128,), dtype=jnp.float32)
    collapsed = jnp.zeros((128,), dtype=jnp.float32)

    gaussian_loss = epps_pulley_gaussian_statistic(gaussian)
    collapsed_loss = epps_pulley_gaussian_statistic(collapsed)

    assert float(collapsed_loss) > float(gaussian_loss)


@pytest.mark.parametrize(
    "kernel_width",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "1.0",
        None,
        1.0e-30,
        1.0e30,
        10**400,
        jnp.asarray([1.0]),
    ],
)
def test_epps_pulley_rejects_invalid_static_kernel_width(kernel_width: object) -> None:
    samples = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="kernel_width must be positive and finite"):
        epps_pulley_gaussian_statistic(
            samples,
            kernel_width=kernel_width,  # type: ignore[arg-type]
        )


def test_epps_pulley_dynamic_jit_keeps_valid_width_and_signals_invalid_widths() -> None:
    samples = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    compiled = jax.jit(lambda width: epps_pulley_gaussian_statistic(samples, kernel_width=width))

    eager = epps_pulley_gaussian_statistic(samples, kernel_width=1.0)
    chex.assert_trees_all_close(compiled(jnp.asarray(1.0)), eager, atol=1.0e-6)
    for invalid in (0.0, -1.0, float("nan"), float("inf"), 1.0e-30, 1.0e30):
        with pytest.raises(
            jax.errors.JaxRuntimeError,
            match="kernel_width must be positive and finite",
        ):
            compiled(jnp.asarray(invalid)).block_until_ready()


def test_epps_pulley_accepts_supported_concrete_real_scalars() -> None:
    samples = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)

    for width in (1, 10**10, np.int64(1), np.float64(0.75)):
        assert bool(
            jnp.isfinite(
                epps_pulley_gaussian_statistic(
                    samples,
                    kernel_width=width,  # type: ignore[arg-type]
                )
            )
        )


def test_epps_pulley_vmap_preserves_valid_widths_and_signals_invalid_lanes() -> None:
    samples = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    mapped = jax.jit(
        jax.vmap(lambda width: epps_pulley_gaussian_statistic(samples, kernel_width=width))
    )
    valid_widths = jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float32)
    expected = jnp.asarray(
        [
            epps_pulley_gaussian_statistic(samples, kernel_width=float(width))
            for width in valid_widths
        ]
    )
    chex.assert_trees_all_close(mapped(valid_widths), expected, atol=1.0e-6)

    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match="kernel_width must be positive and finite",
    ):
        mapped(jnp.asarray([0.5, 0.0, 2.0], dtype=jnp.float32)).block_until_ready()


def test_sliced_sigreg_dynamic_jit_keeps_valid_width_and_signals_invalid_width() -> None:
    embeddings = jnp.asarray(
        [[-1.0, 0.5], [0.0, -0.5], [1.0, 1.5]],
        dtype=jnp.float32,
    )
    directions = jnp.eye(2, dtype=jnp.float32)
    compiled_sliced = jax.jit(
        lambda width: sliced_sigreg_loss(
            embeddings,
            directions,
            kernel_width=width,
        )
    )

    eager = sliced_sigreg_loss(embeddings, directions, kernel_width=1.0)
    chex.assert_trees_all_close(compiled_sliced(jnp.asarray(1.0)), eager, atol=1.0e-6)

    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match="kernel_width must be positive and finite",
    ):
        compiled_sliced(jnp.asarray(0.0)).block_until_ready()


def test_sigreg_config_rejects_invalid_static_numeric_contracts() -> None:
    for bad_width in (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
        "1.0",
        None,
        1.0e-30,
        1.0e30,
        10**400,
        jnp.asarray(1.0),
    ):
        with pytest.raises(ValueError):
            SIGRegConfig(kernel_width=bad_width)  # type: ignore[arg-type]
    for bad_eps in (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
        "1.0",
        None,
        1.0e-50,
        1.0e50,
        10**400,
        jnp.asarray(1.0),
    ):
        with pytest.raises(ValueError):
            SIGRegConfig(eps=bad_eps)  # type: ignore[arg-type]
    for bad_count in (0, -1, True, 1.0, "1", [1]):
        with pytest.raises(ValueError):
            SIGRegConfig(n_projections=bad_count)  # type: ignore[arg-type]


def test_sigreg_config_normalizes_supported_real_scalars() -> None:
    config = SIGRegConfig(
        n_projections=3,
        kernel_width=np.float64(0.75),
        eps=np.float32(1.0e-6),  # type: ignore[arg-type]
    )

    assert type(config.kernel_width) is float
    assert type(config.eps) is float


def test_sliced_sigreg_penalizes_shifted_and_collapsed_embeddings() -> None:
    key = jr.key(2)
    z_key, dir_key = jr.split(key)
    gaussian = jr.normal(z_key, (96, 6), dtype=jnp.float32)
    shifted = 2.0 + 0.2 * gaussian
    collapsed = jnp.zeros_like(gaussian)
    directions = sample_sigreg_directions(
        dir_key,
        latent_dim=6,
        config=SIGRegConfig(n_projections=16),
    )

    gaussian_loss = sliced_sigreg_loss(gaussian, directions)
    shifted_loss = sliced_sigreg_loss(shifted, directions)
    collapsed_loss = sliced_sigreg_loss(collapsed, directions)

    assert float(shifted_loss) > float(gaussian_loss)
    assert float(collapsed_loss) > float(gaussian_loss)


def test_sigreg_diagnostics_are_finite() -> None:
    config = SIGRegConfig(n_projections=8)
    embeddings = jr.normal(jr.key(3), (32, 4), dtype=jnp.float32)
    directions = sample_sigreg_directions(jr.key(4), latent_dim=4, config=config)

    diagnostics = sigreg_diagnostics(embeddings, directions, config)

    chex.assert_tree_all_finite(diagnostics)
    chex.assert_shape(diagnostics.loss, ())


def test_epps_pulley_rejects_nonfinite_samples() -> None:
    """Two inf samples make inf-inf diffs, so the kernel statistic is NaN."""
    samples = jnp.asarray([jnp.inf, jnp.inf], dtype=jnp.float32)

    assert not bool(jnp.isfinite(jnp.mean(samples[:, None] - samples[None, :])))
    with pytest.raises(ValueError, match="samples must be finite"):
        epps_pulley_gaussian_statistic(samples)


def test_sliced_sigreg_rejects_inf_embedding_times_silent_direction() -> None:
    """Inf embedding @ a zero direction coordinate is 0*inf = NaN in the slice."""
    embeddings = jnp.asarray([[jnp.inf, 1.0], [0.0, -0.5]], dtype=jnp.float32)
    directions = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)

    assert not bool(jnp.isfinite(embeddings @ directions.T).all())
    with pytest.raises(ValueError, match="embeddings must be finite"):
        sliced_sigreg_loss(embeddings, directions)


def test_sliced_sigreg_rejects_nonfinite_directions() -> None:
    embeddings = jnp.asarray([[-1.0, 0.5], [0.0, -0.5]], dtype=jnp.float32)
    directions = jnp.asarray([[jnp.inf, 0.0]], dtype=jnp.float32)

    with pytest.raises(ValueError, match="directions must be finite"):
        sliced_sigreg_loss(embeddings, directions)


def test_sigreg_diagnostics_reject_nonfinite_embeddings() -> None:
    embeddings = jnp.asarray([[jnp.nan, 0.0]], dtype=jnp.float32)
    directions = jnp.eye(2, dtype=jnp.float32)

    with pytest.raises(ValueError, match="embeddings must be finite"):
        sigreg_diagnostics(embeddings, directions)


def test_epps_pulley_compiled_finite_check_preserves_valid_values_and_rejects_nan() -> None:
    valid = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    compiled = jax.jit(epps_pulley_gaussian_statistic)

    chex.assert_trees_all_close(
        compiled(valid),
        epps_pulley_gaussian_statistic(valid),
        atol=1.0e-6,
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match="samples must be finite"):
        compiled(valid.at[1].set(jnp.nan)).block_until_ready()


def test_sliced_sigreg_compiled_vmap_rejects_nonfinite_lane() -> None:
    valid = jnp.asarray([[-1.0, 0.5], [0.0, -0.5]], dtype=jnp.float32)
    directions = jnp.eye(2, dtype=jnp.float32)
    batches = jnp.stack((valid, valid.at[0, 0].set(jnp.inf)))
    compiled = jax.jit(jax.vmap(sliced_sigreg_loss, in_axes=(0, None)))

    with pytest.raises(jax.errors.JaxRuntimeError, match="embeddings must be finite"):
        compiled(batches, directions).block_until_ready()


def test_epps_pulley_rejects_samples_that_overflow_float32_narrowing() -> None:
    with jax.enable_x64():
        samples = jnp.asarray([1.0e300, 1.0e300], dtype=jnp.float64)

        with pytest.raises(ValueError, match="samples must be finite"):
            epps_pulley_gaussian_statistic(samples)


def test_sliced_sigreg_rejects_nonfinite_derived_projections() -> None:
    maximum = jnp.finfo(jnp.float32).max
    embeddings = jnp.asarray([[maximum, maximum], [maximum, maximum]])
    directions = jnp.asarray([[1.0, 1.0]], dtype=jnp.float32)

    assert bool(jnp.isfinite(embeddings).all())
    assert bool(jnp.isfinite(directions).all())
    with pytest.raises(ValueError, match="projected samples must be finite"):
        sliced_sigreg_loss(embeddings, directions)


def test_sigreg_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_projections"):
        SIGRegConfig(n_projections=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_projections"):
        SIGRegConfig(n_projections=32.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_projections"):
        SIGRegConfig(n_projections=0)


def test_sigreg_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = SIGRegConfig(
        n_projections=np.int32(64),
        kernel_width=np.float32(1.5),
        eps=np.float32(1e-7),
    )
    assert type(cfg.n_projections) is int
    assert type(cfg.kernel_width) is float
    assert type(cfg.eps) is float
    assert cfg.n_projections == 64
