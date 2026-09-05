import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation.optimizer_geometry import (
    FROZEN_GEOMETRY_CONFIG,
    GEOMETRY_PROTOCOL,
    GEOMETRY_RESULT_SCHEMA,
    canonical_streaming_matrix_result_bytes,
    flad_noise_component,
    flad_noise_component_transaction,
    muon_ogd_dual_update,
    muon_ogd_dual_update_transaction,
    orthogonal_correction,
    retain_streaming_matrix_result,
    run_streaming_matrix_evaluation,
    spectral_matrix_sign,
    spectral_matrix_sign_transaction,
    validate_streaming_matrix_result,
)


def _paper_ns5(matrix: jax.Array, *, steps: int) -> jax.Array:
    value = matrix / jnp.maximum(jnp.linalg.norm(matrix), jnp.asarray(1e-12, matrix.dtype))
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.T
    for _ in range(steps):
        value = 1.5 * value - 0.5 * (value @ value.T) @ value
    return value.T if transposed else value


def test_orthogonal_correction_removes_protected_direction() -> None:
    update = jnp.array([2.0, 3.0])
    basis = jnp.array([[1.0, 0.0]])
    np.testing.assert_allclose(orthogonal_correction(update, basis), [0.0, 3.0])
    np.testing.assert_array_equal(orthogonal_correction(update, jnp.zeros((0, 2))), update)


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2)])
def test_spectral_matrix_sign_matches_pinned_muon_ogd_ns5(shape: tuple[int, int]) -> None:
    matrix = jnp.arange(1, shape[0] * shape[1] + 1, dtype=jnp.float32).reshape(shape)
    np.testing.assert_allclose(
        spectral_matrix_sign(matrix, steps=5), _paper_ns5(matrix, steps=5), rtol=1e-6
    )


def test_spectral_matrix_sign_zero_and_jit_paths() -> None:
    zero = jnp.zeros((3, 2), dtype=jnp.float32)
    np.testing.assert_array_equal(spectral_matrix_sign(zero), zero)
    jitted = jax.jit(spectral_matrix_sign)
    np.testing.assert_allclose(jitted(jnp.eye(2, dtype=jnp.float32)), jnp.eye(2), rtol=1e-5)


def test_muon_ogd_empty_constraints_reduces_exactly_to_ns5() -> None:
    matrix = jnp.array([[2.0, 1.0], [0.5, -1.0]], dtype=jnp.float32)
    update, dual = muon_ogd_dual_update(
        matrix,
        jnp.zeros((0, 2, 2), dtype=jnp.float32),
        jnp.zeros((0,), dtype=jnp.float32),
        dual_learning_rate=0.25,
        dual_steps=2,
    )
    np.testing.assert_array_equal(dual, jnp.zeros((0,), dtype=jnp.float32))
    np.testing.assert_allclose(update, spectral_matrix_sign(matrix), rtol=1e-6)


def test_flad_zero_gradient_is_primal_and_derivative_safe() -> None:
    perturbation = jnp.array([2.0, 4.0])
    zero = jnp.zeros(2)
    np.testing.assert_array_equal(flad_noise_component(perturbation, zero), perturbation)
    delta_jacobian, gradient_jacobian = jax.jacrev(flad_noise_component, argnums=(0, 1))(
        perturbation, zero
    )
    assert bool(jnp.all(jnp.isfinite(delta_jacobian)))
    assert bool(jnp.all(jnp.isfinite(gradient_jacobian)))
    np.testing.assert_array_equal(delta_jacobian, jnp.eye(2))


def test_flad_jit_removes_gradient_aligned_component() -> None:
    result = jax.jit(flad_noise_component)(jnp.array([2.0, 4.0]), jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(result, [0.0, 4.0])


def test_streaming_matrix_evaluation_is_frozen_matched_and_nonpromoting() -> None:
    result = run_streaming_matrix_evaluation()
    assert result["schema"] == GEOMETRY_RESULT_SCHEMA
    assert result["config"] == dict(FROZEN_GEOMETRY_CONFIG)
    assert GEOMETRY_PROTOCOL["stage"] == "frozen_small_streaming_matrix_pre_ipmnist"
    assert GEOMETRY_PROTOCOL["scientific_promotion_allowed"] is False
    assert GEOMETRY_PROTOCOL["persistent_numeric_bytes_accounting_required"] is True
    assert GEOMETRY_PROTOCOL["aggregate_working_set_bytes_claimed"] is False
    arms = result["arms"]
    assert isinstance(arms, list)
    assert len(arms) == 6
    for arm in arms:
        resources = arm["resources"]
        assert resources["observations"] == FROZEN_GEOMETRY_CONFIG["updates"]
        assert resources["updates"] == FROZEN_GEOMETRY_CONFIG["updates"]
        assert resources["data_steps"] == FROZEN_GEOMETRY_CONFIG["updates"]
        assert resources["persistent_numeric_bytes"] > 0
        assert resources["timing_qualified"] is False
    assert result["policy"] == {
        "status": "development-only-nonpromoting",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
    }
    assert set(result["identity"]) == {
        "source_sha256",
        "plan_sha256",
        "runtime",
        "consistency_not_attestation",
    }
    assert len(result["identity"]["source_sha256"]) == 64
    assert len(result["identity"]["plan_sha256"]) == 64
    assert result["identity"]["runtime"]["packages"].keys() == {"jax", "jaxlib", "numpy"}
    assert result["identity"]["consistency_not_attestation"] is True
    validate_streaming_matrix_result(json.loads(json.dumps(result)))


def test_geometry_result_has_canonical_bytes_and_exclusive_retention(tmp_path: Path) -> None:
    result = run_streaming_matrix_evaluation()
    encoded = canonical_streaming_matrix_result_bytes(result)
    assert encoded == canonical_streaming_matrix_result_bytes(json.loads(encoded))
    destination = retain_streaming_matrix_result(result, repository_root=tmp_path)
    assert destination.parent == tmp_path / "outputs/optimizer_geometry/development.v1"
    assert destination.read_bytes() == encoded
    with pytest.raises(FileExistsError):
        retain_streaming_matrix_result(result, repository_root=tmp_path)


def test_geometry_retention_rejects_namespace_symlink(tmp_path: Path) -> None:
    result = run_streaming_matrix_evaluation()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        retain_streaming_matrix_result(result, repository_root=tmp_path)
    assert not (outside / "optimizer_geometry").exists()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("config", "seed"), 1),
        (("config", "updates"), 8.0),
        (("policy", "scientific_promotion_allowed"), True),
        (("arms", 0, "resources", "updates"), 7),
        (("arms", 0, "metrics", "final_target_mse"), 0.0),
        (("comparisons", 0, "outcome"), "unexpected"),
        (("identity", "runtime", "machine"), "different-machine"),
    ],
)
def test_streaming_matrix_validator_rejects_tampering(
    path: tuple[object, ...], replacement: object
) -> None:
    result = copy.deepcopy(run_streaming_matrix_evaluation())
    target: object = result
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)


def test_streaming_matrix_validator_admits_exact_builtin_containers_before_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

        def __getitem__(self, key: object) -> object:
            self.calls += 1
            raise AssertionError("must not index")

    hostile = HostileDict()
    with pytest.raises(ValueError, match="string-keyed mapping"):
        validate_streaming_matrix_result(hostile)
    assert hostile.calls == 0


def test_streaming_matrix_validator_rejects_hostile_exact_dict_key_without_hooks() -> None:
    class HostileKey(str):
        calls = 0

        def __hash__(self) -> int:
            self.calls += 1
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            return super().__eq__(other)

    key = HostileKey("schema")
    hostile = {key: GEOMETRY_RESULT_SCHEMA}
    key.calls = 0
    with pytest.raises(ValueError, match="string-keyed"):
        validate_streaming_matrix_result(hostile)
    assert key.calls == 0


@pytest.mark.parametrize("field", ["protocol", "arms"])
def test_streaming_matrix_validator_rejects_nested_container_subclasses_without_hooks(
    field: str,
) -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

    class HostileList(list[object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

    result = run_streaming_matrix_evaluation()
    hostile: HostileDict | HostileList
    if field == "protocol":
        hostile = HostileDict()
    else:
        hostile = HostileList()
    result[field] = hostile
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)
    assert hostile.calls == 0


def test_streaming_matrix_validator_rejects_oversized_exact_containers_before_iteration() -> None:
    result = run_streaming_matrix_evaluation()
    result["arms"] = [None] * 10_000
    with pytest.raises(ValueError, match="arms"):
        validate_streaming_matrix_result(result)
    result = run_streaming_matrix_evaluation()
    result["protocol"]["unexpected"] = [None] * 10_000  # type: ignore[index]
    with pytest.raises(ValueError, match="protocol"):
        validate_streaming_matrix_result(result)


def test_streaming_matrix_validator_rejects_scalar_subclasses_without_equality_hooks() -> None:
    class HostileStr(str):
        calls = 0

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    class HostileFloat(float):
        calls = 0

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    result = run_streaming_matrix_evaluation()
    hostile_string = HostileStr("unexpected")
    result["schema"] = hostile_string
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)
    assert hostile_string.calls == 0

    result = run_streaming_matrix_evaluation()
    hostile_number = HostileFloat(0.0)
    result["arms"][0]["metrics"]["final_target_mse"] = hostile_number  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)
    assert hostile_number.calls == 0


def test_geometry_primitives_are_outer_jit_safe() -> None:
    corrected = jax.jit(orthogonal_correction)(jnp.array([1.0, 2.0]), jnp.array([[1.0, 0.0]]))
    np.testing.assert_allclose(corrected, [0.0, 2.0])
    invalid = jax.jit(flad_noise_component)(jnp.array([jnp.nan]), jnp.ones(1))
    assert bool(jnp.all(jnp.isnan(invalid)))
    safe, valid = jax.jit(flad_noise_component_transaction)(
        jnp.array([jnp.nan]), jnp.ones(1)
    )
    np.testing.assert_array_equal(safe, jnp.zeros(1))
    assert not bool(valid)


def test_geometry_rejects_empty_flad_and_hostile_array() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        flad_noise_component(jnp.zeros(0), jnp.zeros(0))
    with pytest.raises(ValueError, match="non-empty"):
        orthogonal_correction(jnp.zeros(0), jnp.zeros((0, 0)))

    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        spectral_matrix_sign(hostile)  # type: ignore[arg-type]
    assert hostile.calls == 0


def test_geometry_float32_overflow_is_invalid_not_laundered() -> None:
    maximum = np.float32(np.finfo(np.float32).max)
    matrix, matrix_valid = jax.jit(spectral_matrix_sign_transaction)(jnp.full((2, 2), maximum))
    assert bool(jnp.all(jnp.isfinite(matrix)))
    assert not bool(matrix_valid)
    component, component_valid = jax.jit(flad_noise_component_transaction)(
        jnp.full((2,), maximum), jnp.full((2,), maximum)
    )
    assert bool(jnp.all(jnp.isfinite(component)))
    assert not bool(component_valid)
    update, dual, dual_valid = jax.jit(
        lambda value: muon_ogd_dual_update_transaction(
            value,
            jnp.ones((1, 2, 2), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            dual_learning_rate=0.25,
            dual_steps=2,
        )
    )(jnp.full((2, 2), maximum))
    assert bool(jnp.all(jnp.isfinite(update)))
    assert bool(jnp.all(jnp.isfinite(dual)))
    assert not bool(dual_valid)


def _floor_probe_matrix() -> jax.Array:
    return jnp.array([[2.0, 1.0], [0.5, -1.0]], dtype=jnp.float32)


@pytest.mark.parametrize("scale", [1e-13, 1e-15, 1e-20, 1e-25, 1e-30])
def test_spectral_matrix_sign_is_scale_invariant_below_the_frobenius_floor(scale: float) -> None:
    reference = _floor_probe_matrix()
    tiny = reference * jnp.asarray(scale, dtype=jnp.float32)
    sign, valid = spectral_matrix_sign_transaction(tiny, steps=5)
    assert bool(valid)
    np.testing.assert_allclose(sign, spectral_matrix_sign(reference, steps=5), rtol=1e-6)


@pytest.mark.parametrize("exponent", [-120, -80, -45, 40])
def test_spectral_matrix_sign_is_exactly_invariant_to_power_of_two_rescaling(
    exponent: int,
) -> None:
    reference = _floor_probe_matrix()
    np.testing.assert_array_equal(
        spectral_matrix_sign(jnp.ldexp(reference, exponent), steps=5),
        spectral_matrix_sign(reference, steps=5),
    )


def test_spectral_matrix_sign_below_the_floor_stays_orthogonal() -> None:
    tiny = _floor_probe_matrix() * jnp.asarray(1e-20, dtype=jnp.float32)
    assert 0.0 <= float(jnp.linalg.norm(tiny)) < 1e-12
    assert float(jnp.max(jnp.abs(tiny))) > 0.0
    sign, valid = spectral_matrix_sign_transaction(tiny, steps=5)
    assert bool(valid)
    singular = np.linalg.svd(np.asarray(sign, dtype=np.float64), compute_uv=False)
    np.testing.assert_allclose(singular, np.ones_like(singular), rtol=1e-3)


def test_spectral_matrix_sign_gradient_is_finite_below_the_frobenius_floor() -> None:
    tiny = _floor_probe_matrix() * jnp.asarray(1e-25, dtype=jnp.float32)
    jacobian = jax.jacrev(spectral_matrix_sign)(tiny)
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_jit_spectral_matrix_sign_is_scale_invariant_below_the_floor() -> None:
    reference = _floor_probe_matrix()
    jitted = jax.jit(spectral_matrix_sign)
    tiny = reference * jnp.asarray(1e-25, dtype=jnp.float32)
    np.testing.assert_allclose(jitted(tiny), jitted(reference), rtol=1e-6)


def test_muon_ogd_empty_constraints_reduces_to_ns5_below_the_floor() -> None:
    reference = _floor_probe_matrix()
    tiny = reference * jnp.asarray(1e-25, dtype=jnp.float32)
    update, dual = muon_ogd_dual_update(
        tiny,
        jnp.zeros((0, 2, 2), dtype=jnp.float32),
        jnp.zeros((0,), dtype=jnp.float32),
        dual_learning_rate=0.25,
        dual_steps=2,
    )
    np.testing.assert_array_equal(dual, jnp.zeros((0,), dtype=jnp.float32))
    np.testing.assert_allclose(update, spectral_matrix_sign(reference), rtol=1e-6)


def test_spectral_matrix_sign_zero_matrix_stays_exactly_zero_and_valid() -> None:
    zero = jnp.zeros((3, 2), dtype=jnp.float32)
    sign, valid = spectral_matrix_sign_transaction(zero, steps=5)
    np.testing.assert_array_equal(sign, zero)
    assert bool(valid)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_spectral_matrix_sign_narrow_dtype_below_the_floor_is_orthogonal(
    dtype: jnp.dtype,
) -> None:
    tiny = (_floor_probe_matrix() * jnp.asarray(2.0**-20, dtype=jnp.float32)).astype(dtype)
    sign, valid = spectral_matrix_sign_transaction(tiny, steps=5)
    assert bool(valid)
    assert sign.dtype == tiny.dtype
    singular = np.linalg.svd(np.asarray(sign, dtype=np.float64), compute_uv=False)
    np.testing.assert_allclose(singular, np.ones_like(singular), rtol=1e-2)


def test_geometry_runner_rejects_invalid_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "alberta_framework.evaluation.optimizer_geometry._frozen_stream",
        lambda: jnp.full((8, 3, 2), jnp.nan, dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="transaction"):
        run_streaming_matrix_evaluation()


def test_flad_noise_component_scale_freedom() -> None:
    delta = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    grad = np.array([2.0, 1.0, -1.0, 0.5], dtype=np.float32)

    # Reference projection in float64
    d_f64 = np.asarray(delta, dtype=np.float64)
    g_f64 = np.asarray(grad, dtype=np.float64)
    ref = d_f64 - g_f64 * (float(g_f64 @ d_f64) / float(g_f64 @ g_f64))

    for scale in (1.0, 1e-10, 1e-20, 1e-30):
        scaled_grad = jnp.asarray(grad * np.float32(scale))
        safe, valid = jax.jit(flad_noise_component_transaction)(jnp.asarray(delta), scaled_grad)
        assert bool(valid)
        np.testing.assert_allclose(np.asarray(safe, dtype=np.float64), ref, rtol=1e-5, atol=1e-5)

    # Zero gradient returns delta bit-identically
    safe_zero, valid_zero = jax.jit(flad_noise_component_transaction)(
        jnp.asarray(delta), jnp.zeros_like(scaled_grad)
    )
    assert bool(valid_zero)
    np.testing.assert_array_equal(safe_zero, jnp.asarray(delta))
