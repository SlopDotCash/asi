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


FLAD_PERTURBATION = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
FLAD_GRADIENT = np.array([2.0, 1.0, -1.0, 0.5], dtype=np.float32)


def _flad_reference(perturbation: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Project the perturbation away from the gradient in float64."""
    delta = perturbation.astype(np.float64)
    direction = gradient.astype(np.float64)
    squared_norm = float(direction @ direction)
    if squared_norm == 0.0:
        return delta
    return delta - direction * (float(direction @ delta) / squared_norm)


@pytest.mark.parametrize("scale", [1e-20, 1e-25, 1e-30])
def test_flad_underflowing_squared_norm_still_removes_the_component(scale: float) -> None:
    gradient = FLAD_GRADIENT * np.float32(scale)
    assert float(jnp.vdot(jnp.asarray(gradient), jnp.asarray(gradient)).real) == 0.0
    safe, valid = jax.jit(flad_noise_component_transaction)(
        jnp.asarray(FLAD_PERTURBATION), jnp.asarray(gradient)
    )
    assert bool(valid)
    np.testing.assert_allclose(
        np.asarray(safe, dtype=np.float64),
        _flad_reference(FLAD_PERTURBATION, gradient),
        atol=1e-6,
    )


@pytest.mark.parametrize("scale", [1e19, 1e20, 1e30, 1.7e38])
def test_flad_overflowing_squared_norm_is_answered_from_the_rescaled_gradient(
    scale: float,
) -> None:
    gradient = FLAD_GRADIENT * np.float32(scale)
    assert not bool(jnp.isfinite(jnp.vdot(jnp.asarray(gradient), jnp.asarray(gradient)).real))
    safe, valid = jax.jit(flad_noise_component_transaction)(
        jnp.asarray(FLAD_PERTURBATION), jnp.asarray(gradient)
    )
    assert bool(valid)
    np.testing.assert_allclose(
        np.asarray(safe, dtype=np.float64),
        _flad_reference(FLAD_PERTURBATION, gradient),
        atol=1e-6,
    )


def test_flad_unrepresentable_coefficient_stays_invalid() -> None:
    maximum = np.float32(np.finfo(np.float32).max)
    safe, valid = jax.jit(flad_noise_component_transaction)(
        jnp.full((4,), maximum), jnp.full((4,), maximum)
    )
    assert bool(jnp.all(jnp.isfinite(safe)))
    assert not bool(valid)


@pytest.mark.parametrize("scale", [1e-38, 2.938736e-39])
def test_flad_subnormal_gradient_entries_are_left_at_their_own_scale(scale: float) -> None:
    gradient = FLAD_GRADIENT * np.float32(scale)
    magnitudes = np.abs(gradient).view(np.uint32) & np.uint32(0x7FFFFFFF)
    assert bool(np.any((magnitudes > 0) & (magnitudes < np.uint32(0x00800000))))
    safe, valid = jax.jit(flad_noise_component_transaction)(
        jnp.asarray(FLAD_PERTURBATION), jnp.asarray(gradient)
    )
    assert bool(valid)
    np.testing.assert_array_equal(safe, jnp.asarray(FLAD_PERTURBATION))


@pytest.mark.parametrize(
    ("scale", "expected_words"),
    [
        (1.0, [0x3F2E147B, 0xC00A3D71, 0x3F28F5C2, 0x403AE148]),
        (1e-10, [0x3F2E147B, 0xC00A3D71, 0x3F28F5C2, 0x403AE148]),
        (1e-19, [0x3F1E79E8, 0xC00C30C3, 0x3F30C30C, 0x4039E79E]),
    ],
)
def test_flad_usable_squared_norm_answers_are_bit_for_bit_unchanged(
    scale: float, expected_words: list[int]
) -> None:
    """Pin the answers recorded before the rescale was introduced.

    A gradient whose squared norm is already a positive finite number is answered
    from that divisor, so its result must not be re-rounded onto the grid the
    rescaled inner products would land on. The 1e-19 row keeps the precision it
    loses to a divisor assembled from subnormal partial products.
    """
    direction = jnp.asarray(FLAD_GRADIENT * np.float32(scale))
    squared_norm = jnp.vdot(direction, direction).real
    assert bool(squared_norm > 0.0) and bool(jnp.isfinite(squared_norm))
    safe, valid = jax.jit(flad_noise_component_transaction)(
        jnp.asarray(FLAD_PERTURBATION), direction
    )
    assert bool(valid)
    words = np.asarray(safe, dtype=np.float32).view(np.uint32)
    assert [int(word) for word in words] == expected_words


def test_flad_zero_gradient_reduces_to_the_perturbation_bit_for_bit() -> None:
    delta = jnp.asarray(FLAD_PERTURBATION)
    safe, valid = jax.jit(flad_noise_component_transaction)(
        delta, jnp.zeros(4, dtype=jnp.float32)
    )
    assert bool(valid)
    np.testing.assert_array_equal(safe, delta)


def test_flad_bfloat16_underflow_is_repaired_within_its_own_precision() -> None:
    gradient = jnp.asarray(FLAD_GRADIENT * np.float32(1e-20), dtype=jnp.bfloat16)
    delta = jnp.asarray(FLAD_PERTURBATION, dtype=jnp.bfloat16)
    assert float(jnp.vdot(gradient, gradient).real) == 0.0
    safe, valid = jax.jit(flad_noise_component_transaction)(delta, gradient)
    assert bool(valid)
    reference = _flad_reference(
        np.asarray(delta, dtype=np.float32), np.asarray(gradient, dtype=np.float32)
    )
    np.testing.assert_allclose(np.asarray(safe, dtype=np.float64), reference, atol=1e-2)


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


def test_geometry_runner_rejects_invalid_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "alberta_framework.evaluation.optimizer_geometry._frozen_stream",
        lambda: jnp.full((8, 3, 2), jnp.nan, dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="transaction"):
        run_streaming_matrix_evaluation()
