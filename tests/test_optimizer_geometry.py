import jax.numpy as jnp
import numpy as np

from alberta_framework.evaluation.optimizer_geometry import (
    GEOMETRY_PROTOCOL,
    flad_noise_component,
    orthogonal_correction,
    spectral_matrix_sign,
)


def test_orthogonal_correction_removes_protected_direction() -> None:
    update = jnp.array([2.0, 3.0])
    basis = jnp.array([[1.0, 0.0]])
    np.testing.assert_allclose(orthogonal_correction(update, basis), [0.0, 3.0])
    np.testing.assert_array_equal(orthogonal_correction(update, jnp.zeros((0, 2))), update)


def test_spectral_matrix_sign_has_bounded_singular_values() -> None:
    result = spectral_matrix_sign(jnp.diag(jnp.array([3.0, 0.5])), steps=5)
    assert float(jnp.linalg.svd(result, compute_uv=False)[0]) <= 1.1


def test_flad_removes_gradient_aligned_component() -> None:
    gradient = jnp.array([1.0, 0.0])
    perturbation = jnp.array([2.0, 4.0])
    np.testing.assert_allclose(flad_noise_component(perturbation, gradient), [0.0, 4.0])
    np.testing.assert_array_equal(flad_noise_component(perturbation, jnp.zeros(2)), perturbation)


def test_protocol_is_small_matrix_first_and_nonpromoting() -> None:
    assert GEOMETRY_PROTOCOL["paper_revisions"] == (
        "arXiv:2605.08949v2",
        "arXiv:2606.10406v1",
        "arXiv:2601.07636v1",
    )
    assert GEOMETRY_PROTOCOL["stage"] == "small_streaming_matrix_pre_ipmnist"
    assert GEOMETRY_PROTOCOL["scientific_promotion_allowed"] is False
