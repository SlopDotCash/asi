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


@pytest.mark.parametrize(
    "integer_type",
    sorted(
        {np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")},
        key=lambda value: value.__name__,
    ),
)
def test_sigreg_accepts_every_numpy_integer_family(integer_type) -> None:
    config = SIGRegConfig(n_projections=integer_type(2))
    assert type(config.n_projections) is int
    directions = sample_sigreg_directions(jr.key(0), integer_type(3), config)
    assert directions.shape == (2, 3)


def test_sigreg_rejects_integer_spoofs_without_hooks_or_repr() -> None:
    class HostileInt(int):
        def __index__(self):
            raise AssertionError("subclass hook executed")

        def __repr__(self):
            raise AssertionError("repr executed")

    with pytest.raises(ValueError, match="n_projections"):
        SIGRegConfig(n_projections=HostileInt(2))
    with pytest.raises(ValueError, match="latent_dim"):
        sample_sigreg_directions(jr.key(0), HostileInt(2))


def test_sigreg_config_schema_is_exact() -> None:
    config = SIGRegConfig(n_projections=np.int32(3))
    payload = config.to_config()
    assert SIGRegConfig.from_config(payload) == config
    with pytest.raises(ValueError, match="actual dict"):
        SIGRegConfig.from_config(type("DictSubclass", (dict,), {})(payload))
    with pytest.raises(ValueError, match="schema"):
        SIGRegConfig.from_config({**payload, "unknown": 1})
    with pytest.raises(ValueError, match="type"):
        SIGRegConfig.from_config({**payload, "type": "wrong"})


def test_sigreg_config_float_hooks_are_not_run() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self):
            type(self).calls += 1
            raise RuntimeError("hostile ratio")

        def __repr__(self):
            raise AssertionError("repr executed")

    with pytest.raises(ValueError, match="kernel_width"):
        SIGRegConfig(kernel_width=HostileFloat(1.0))
    assert HostileFloat.calls == 0


def test_sigreg_preflights_direction_and_pairwise_resources_without_allocating() -> None:
    with pytest.raises(ValueError, match="directions"):
        SIGRegConfig(n_projections=(2**31 - 1) // 4 + 1)
    config = SIGRegConfig(n_projections=2)
    with pytest.raises(ValueError, match="directions"):
        sample_sigreg_directions(jr.key(0), (2**31 - 1) // 4, config)

    samples = jnp.zeros((23_171,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="pairwise workspace"):
        epps_pulley_gaussian_statistic(samples)

    embeddings = jnp.zeros((1_000, 1), dtype=jnp.float32)
    directions = jnp.ones((600, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="projected pairwise workspace"):
        sliced_sigreg_loss(embeddings, directions)


def test_sigreg_rejects_mismatched_or_empty_representation_shapes() -> None:
    with pytest.raises(ValueError, match="directions must have shape"):
        sliced_sigreg_loss(jnp.ones((2, 3)), jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="at least one sample"):
        sliced_sigreg_loss(jnp.ones((0, 3)), jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="non-empty"):
        epps_pulley_gaussian_statistic(jnp.asarray([], dtype=jnp.float32))


def test_sigreg_diagnostics_std_is_finite_at_float32_extremes() -> None:
    maximum = np.finfo(np.float32).max
    embeddings = jnp.asarray([[maximum], [-maximum]], dtype=jnp.float32)
    directions = jnp.ones((1, 1), dtype=jnp.float32)
    assert not bool(jnp.isfinite(jnp.std(embeddings, axis=0)[0]))
    diagnostics = sigreg_diagnostics(embeddings, directions, SIGRegConfig(n_projections=1))
    chex.assert_tree_all_finite(diagnostics)


def _bhep_reference_statistic(samples: np.ndarray, kernel_width: float) -> float:
    """Independent float64 reimplementation of the closed-form BHEP statistic.

    Transcribes ``BHEP_{n,beta}`` exactly as given in Henze, N. (2021), "Tests
    for multivariate normality -- a critical review with emphasis on
    weighted L^2-statistics" (arXiv:2004.07332), Section 2.1:

        BHEP_{n,beta} = (1/n) * sum_{j,k} exp(-beta^2 ||Y_j - Y_k||^2 / 2)
            - 2 / (1 + beta^2)^(d/2) * sum_j exp(-beta^2 ||Y_j||^2 / (2 (1 + beta^2)))
            + n / (1 + 2 beta^2)^(d/2)

    for the univariate case ``d = 1``, citing that Baringhaus, L. & Henze, N.
    (1988), "A Consistent Test for Multivariate Normality Based on the
    Empirical Characteristic Function," studied the special case ``beta =
    1``. ``epps_pulley_gaussian_statistic`` uses ``kernel_width = 1 / beta``
    and reports ``BHEP_{n,beta} / n`` (a per-sample normalization suited to
    use as a bounded loss rather than a raw n-scaled test statistic), clipped
    at zero for floating-point safety.
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.shape[0]
    beta = 1.0 / kernel_width
    diffs = x[:, None] - x[None, :]
    term_a = np.sum(np.exp(-(beta**2) * diffs**2 / 2.0)) / n
    term_b = (2.0 / (1.0 + beta**2) ** 0.5) * np.sum(
        np.exp(-(beta**2) * x**2 / (2.0 * (1.0 + beta**2)))
    )
    term_c = n / (1.0 + 2.0 * beta**2) ** 0.5
    return float(max((term_a - term_b + term_c) / n, 0.0))


@pytest.mark.parametrize("kernel_width", [1.0, 0.75, 2.0])
def test_epps_pulley_matches_closed_form_bhep_statistic(kernel_width: float) -> None:
    """Cross-check against the published closed form, not just relative ordering.

    ``kernel_width=1.0`` is beta=1, the exact case Baringhaus & Henze (1988)
    studied; the other widths exercise the general beta dependence given in
    Henze (2021, arXiv:2004.07332, Sec. 2.1). See
    ``_bhep_reference_statistic`` for the transcribed formula.
    """
    samples = np.asarray([-1.5, 0.3, 2.0, -0.7], dtype=np.float64)
    expected = _bhep_reference_statistic(samples, kernel_width)

    actual = epps_pulley_gaussian_statistic(
        jnp.asarray(samples, dtype=jnp.float32), kernel_width=kernel_width
    )

    assert float(actual) == pytest.approx(expected, abs=1.0e-6)


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
    compiled = jax.jit(
        lambda width: epps_pulley_gaussian_statistic(samples, kernel_width=width)
    )

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
        jax.vmap(
            lambda width: epps_pulley_gaussian_statistic(samples, kernel_width=width)
        )
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
    for bad_count in (0, -1, True, 1.0):
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
