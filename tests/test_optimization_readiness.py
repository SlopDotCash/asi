from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from alberta_framework.evaluation.optimization_readiness import (
    OPTIMIZATION_READINESS_PROTOCOL,
    energy_rank,
    estimate_appendix_c1_optimization_readiness,
    estimate_optimization_readiness,
    validate_development_result,
    validate_matched_development_results,
)


def test_readiness_matches_paper_empirical_equations() -> None:
    gradients = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    report = estimate_optimization_readiness(
        loss=2.0,
        full_validation_gradient=np.asarray([1.0, 0.0]),
        batch_gradients=gradients,
    )
    assert report.gradient_squared_norm == pytest.approx(1.0)
    assert report.expected_batch_gradient_squared_norm == pytest.approx(5.0)
    assert report.gradient_strength == pytest.approx(0.5)
    assert report.gradient_reliability == pytest.approx(0.2)
    assert report.optimization_readiness == pytest.approx(0.1)
    assert report.gradient_norm == pytest.approx(1.0)


def test_appendix_c1_mode_checks_observable_contract_without_claiming_independence() -> None:
    gradients = np.ones((128, 2), dtype=np.float64)
    report = estimate_appendix_c1_optimization_readiness(
        loss=1.0,
        full_validation_gradient=np.ones(2, dtype=np.float64),
        batch_gradients=gradients,
        full_validation_observations=10_000,
        mini_batch_size=4,
        sampling_provenance=(
            "caller_reported_independent_with_replacement_not_verified_from_gradients"
        ),
    )
    assert report.batch_count == 128

    with pytest.raises(ValueError, match="128 mini-batch gradient rows"):
        estimate_appendix_c1_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.ones(2, dtype=np.float64),
            batch_gradients=gradients[:127],
            full_validation_observations=10_000,
            mini_batch_size=4,
            sampling_provenance=(
                "caller_reported_independent_with_replacement_not_verified_from_gradients"
            ),
        )

    with pytest.raises(ValueError, match="10,000 validation"):
        estimate_appendix_c1_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.ones(2, dtype=np.float64),
            batch_gradients=gradients,
            full_validation_observations=9_999,
            mini_batch_size=4,
            sampling_provenance=(
                "caller_reported_independent_with_replacement_not_verified_from_gradients"
            ),
        )
    with pytest.raises(ValueError, match="mini-batch size 4"):
        estimate_appendix_c1_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.ones(2, dtype=np.float64),
            batch_gradients=gradients,
            full_validation_observations=10_000,
            mini_batch_size=1,
            sampling_provenance=(
                "caller_reported_independent_with_replacement_not_verified_from_gradients"
            ),
        )
    with pytest.raises(ValueError, match="explicit unverified label"):
        estimate_appendix_c1_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.ones(2, dtype=np.float64),
            batch_gradients=gradients,
            full_validation_observations=10_000,
            mini_batch_size=4,
            sampling_provenance="independence_proven",
        )


def test_generic_equation_helper_remains_explicitly_sampling_agnostic() -> None:
    report = estimate_optimization_readiness(
        loss=1.0,
        full_validation_gradient=np.ones(1),
        batch_gradients=np.ones((1, 1)),
    )
    assert report.batch_count == 1


def test_zero_and_mechanism_off_reductions() -> None:
    zero = estimate_optimization_readiness(
        loss=0.0,
        full_validation_gradient=np.zeros(3, dtype=np.float64),
        batch_gradients=np.zeros((2, 3), dtype=np.float64),
    )
    assert zero.optimization_readiness == 0.0
    strength_only = estimate_optimization_readiness(
        loss=2.0,
        full_validation_gradient=np.asarray([2.0]),
        batch_gradients=np.asarray([[1.0], [3.0]]),
        include_reliability=False,
    )
    assert strength_only.optimization_readiness == strength_only.gradient_strength


def test_energy_rank_baseline_and_exact_threshold() -> None:
    matrix = np.diag([3.0, 1.0])
    assert energy_rank(matrix, threshold=0.9) == 1
    assert energy_rank(matrix, threshold=0.99) == 2
    assert energy_rank(np.zeros((2, 2)), threshold=0.99) == 0
    rounding_case = np.random.default_rng(7).normal(size=(10, 10))
    assert energy_rank(rounding_case, threshold=1.0) == 10


@pytest.mark.parametrize("loss", [-1.0, float("nan"), True])
def test_readiness_rejects_invalid_loss(loss: object) -> None:
    with pytest.raises(ValueError, match="loss"):
        estimate_optimization_readiness(
            loss=loss,  # type: ignore[arg-type]
            full_validation_gradient=np.ones(2),
            batch_gradients=np.ones((2, 2)),
        )


def test_readiness_rejects_mismatched_parameter_axes() -> None:
    with pytest.raises(ValueError, match="parameter axis"):
        estimate_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.ones(3),
            batch_gradients=np.ones((2, 2)),
        )


def test_readiness_rejects_nonrepresentable_squared_norms() -> None:
    with pytest.raises(ValueError, match="finite float64"):
        estimate_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.asarray([1e308]),
            batch_gradients=np.asarray([[1e308]]),
        )


def test_batch_gradient_mean_survives_large_but_representable_row_norms() -> None:
    """A representable batch mean must not be fabricated from an overflowed sum.

    Summing 128 rows whose individual squared norms are near the float64 ceiling
    overflows, and dividing infinity by the batch count keeps it infinite.  The
    reliability ratio then silently collapses to ``0.0`` while every gated
    output stays finite, so the receipt reports zero readiness for a real
    gradient.
    """
    row_norm = 6.5e153
    batch_gradients = np.tile(np.full(4, row_norm), (128, 1))
    with np.errstate(over="ignore"):
        unscaled_mean = float(np.mean(np.square(batch_gradients).sum(axis=1)))
    assert not np.isfinite(unscaled_mean)

    readiness = estimate_optimization_readiness(
        loss=1.0,
        full_validation_gradient=np.ones(4),
        batch_gradients=batch_gradients,
    )
    expected = 4.0 * row_norm * row_norm
    assert np.isfinite(readiness.expected_batch_gradient_squared_norm)
    assert readiness.expected_batch_gradient_squared_norm == pytest.approx(expected, rel=1e-12)
    assert readiness.gradient_reliability > 0.0
    assert readiness.optimization_readiness > 0.0


def test_energy_rank_is_scale_stable() -> None:
    assert energy_rank(np.diag([9e307, 1e307]), threshold=0.99) == 2


def test_protocol_is_prospective_and_nonpromoting() -> None:
    assert OPTIMIZATION_READINESS_PROTOCOL["paper_revision"] == "arXiv:2605.09044v1"
    assert OPTIMIZATION_READINESS_PROTOCOL["population_gradient_estimator"] == (
        "full_validation_set_gradient"
    )
    assert OPTIMIZATION_READINESS_PROTOCOL["reference_reliability_batch_count"] == 128
    assert OPTIMIZATION_READINESS_PROTOCOL["reference_reliability_batch_size"] == 4
    assert OPTIMIZATION_READINESS_PROTOCOL["execution_protocol_required"] is True
    assert OPTIMIZATION_READINESS_PROTOCOL["diagnostics"] == (
        "optimization_readiness",
        "gradient_norm",
        "representation_energy_rank_0.99",
        "curvature_energy_rank_0.99",
        "parameter_norm",
    )


def _result_payload(*, arm_id: str = "candidate") -> dict[str, object]:
    return {
        "schema": "asi.optimization-readiness.development-result.v1",
        "comparison_id": "or-dev-001",
        "arm_id": arm_id,
        "protocol": {
            "schema": "asi.optimization-readiness.protocol.v1",
            "seed": 7,
            "checkpoint": "checkpoint-0042",
            "task": "ipmnist-permutation-3",
            "updates": 128,
            "observations": 1_291_024,
            "full_validation_observations": 10_000,
            "mini_batch_size": 4,
            "diagnostic_batch_count": 128,
            "future_gain_steps": 1,
            "future_gain_rollout_count": 128,
            "future_gain_batch_size": 4,
            "future_gain_step_size": 1e-3,
            "parameter_count": 16,
            "sampling_provenance": (
                "caller_reported_independent_with_replacement_not_verified_from_gradients"
            ),
            "allowed_boundary_information": ["task_start"],
            "allowed_task_information": [
                "current_validation_inputs",
                "current_validation_labels",
            ],
        },
        "resources": {
            "schema": "asi.optimization-readiness.resources.v1",
            "persistent_bytes": 4096,
            "peak_working_set_bytes": 129 * 16 * 8,
            "environment_steps": 0,
            "data_steps": 1_291_024,
            "model_queries": 1_291_024,
            "parameter_updates": 128,
            "timing_seconds": 1.25,
            "timing_is_telemetry_only": True,
        },
        "metrics": {
            "optimization_readiness": 0.25,
            "gradient_norm": 0.5,
            "representation_energy_rank_0_99": 8,
            "curvature_energy_rank_0_99": 5,
            "parameter_norm": 12.0,
            "future_relative_loss_reduction": -0.1,
        },
        "reported_outcome": "rejected",
        "reported_outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def test_strict_result_and_resource_receipt_validation() -> None:
    receipt = validate_development_result(_result_payload())
    assert receipt.reported_outcome == "rejected"
    assert receipt.resources.data_steps == 1_291_024
    assert receipt.protocol.allowed_boundary_information == ("task_start",)

    payload = _result_payload()
    resources = payload["resources"]
    assert isinstance(resources, dict)
    resources["unaccounted_gpu_queries"] = 1
    with pytest.raises(ValueError, match="resources keys must be exactly"):
        validate_development_result(payload)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("resources", "persistent_bytes"), -1, "persistent_bytes"),
        (("resources", "timing_seconds"), float("nan"), "timing_seconds"),
        (("resources", "timing_is_telemetry_only"), False, "telemetry"),
        (("scientific_promotion_allowed",), True, "scientific_promotion_allowed"),
        (("reported_outcome_retained",), False, "reported_outcome_retained"),
    ],
)
def test_result_validator_fails_closed(
    path: tuple[str, ...], value: object, match: str
) -> None:
    payload = _result_payload()
    target = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_development_result(payload)


def test_matched_result_validation_enforces_all_axes() -> None:
    candidate = _result_payload()
    control = _result_payload(arm_id="control")
    receipts = validate_matched_development_results([candidate, control])
    assert len(receipts) == 2

    mismatched = deepcopy(control)
    protocol = mismatched["protocol"]
    assert isinstance(protocol, dict)
    protocol["seed"] = 8
    with pytest.raises(ValueError, match="matched protocol axes"):
        validate_matched_development_results([candidate, mismatched])

    mismatched_budget = deepcopy(control)
    resources = mismatched_budget["resources"]
    assert isinstance(resources, dict)
    resources["persistent_bytes"] = 4097
    with pytest.raises(ValueError, match="resource budgets"):
        validate_matched_development_results([candidate, mismatched_budget])


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("protocol", "observations", 1_291_023, "pinned diagnostic and rollout"),
        ("protocol", "mini_batch_size", 5, "Appendix C.1"),
        ("protocol", "diagnostic_batch_count", 127, "Appendix C.1"),
        ("protocol", "updates", 129, "future-gain steps times rollout count"),
        ("resources", "data_steps", 1_291_023, "environment_steps plus data_steps"),
        ("resources", "model_queries", 1_291_023, "model_queries"),
        ("resources", "parameter_updates", 127, "parameter_updates"),
        ("resources", "peak_working_set_bytes", 16_511, "peak_working_set_bytes"),
    ],
)
def test_resource_receipt_enforces_live_cross_field_identities(
    section: str, field: str, value: object, match: str
) -> None:
    payload = _result_payload()
    nested = payload[section]
    assert isinstance(nested, dict)
    nested[field] = value
    with pytest.raises(ValueError, match=match):
        validate_development_result(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("representation_energy_rank_0_99", 10_001, "validation row count"),
        ("curvature_energy_rank_0_99", 17, "parameter_count"),
        ("future_relative_loss_reduction", 1.0001, "cannot exceed one"),
    ],
)
def test_reported_metrics_respect_mathematical_domains(
    field: str, value: object, match: str
) -> None:
    payload = _result_payload()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics[field] = value
    with pytest.raises(ValueError, match=match):
        validate_development_result(payload)


def test_receipt_boundaries_reject_hostile_runtime_classes_without_hooks() -> None:
    class HostileDict(dict[str, object]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("mapping iteration hook must not run")

    class HostileList(list[object]):
        def __len__(self) -> int:
            raise AssertionError("list length hook must not run")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("list iteration hook must not run")

    class HostileString(str):
        def __bool__(self) -> bool:
            raise AssertionError("string truth hook must not run")

        def __eq__(self, other: object) -> bool:
            del other
            raise AssertionError("string equality hook must not run")

        def encode(self, *args: object, **kwargs: object) -> bytes:
            del args, kwargs
            raise AssertionError("string encode hook must not run")

    class HostileInt(int):
        def __lt__(self, other: object) -> bool:
            del other
            raise AssertionError("integer comparison hook must not run")

        def __le__(self, other: object) -> bool:
            del other
            raise AssertionError("integer comparison hook must not run")

    with pytest.raises(ValueError, match="exact dict"):
        validate_development_result(HostileDict(_result_payload()))
    with pytest.raises(ValueError, match="exact list or tuple"):
        validate_matched_development_results(HostileList())

    payload = _result_payload()
    payload["schema"] = HostileString(str(payload["schema"]))
    with pytest.raises(ValueError, match="exact string"):
        validate_development_result(payload)

    payload = _result_payload()
    protocol = payload["protocol"]
    assert isinstance(protocol, dict)
    protocol["seed"] = HostileInt(7)
    with pytest.raises(ValueError, match="bounded nonnegative built-in int"):
        validate_development_result(payload)

    payload = _result_payload()
    protocol = payload["protocol"]
    assert isinstance(protocol, dict)
    protocol["allowed_task_information"] = HostileList(["label"])
    with pytest.raises(ValueError, match="exact list"):
        validate_development_result(payload)


def test_receipt_payloads_and_scalars_are_bounded() -> None:
    payload = _result_payload()
    payload["comparison_id"] = "x" * 4_097
    with pytest.raises(ValueError, match="bounded non-empty string"):
        validate_development_result(payload)

    payload = _result_payload()
    resources = payload["resources"]
    assert isinstance(resources, dict)
    resources["persistent_bytes"] = 256 * 1024 * 1024 + 1
    with pytest.raises(ValueError, match="persistent_bytes"):
        validate_development_result(payload)

    with pytest.raises(ValueError, match="bounded count"):
        validate_matched_development_results([_result_payload()] * 257)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed_boundary_information", ["future_task_boundary"]),
        ("allowed_task_information", ["held_out_answers"]),
    ],
)
def test_receipt_rejects_information_outside_frozen_permissions(
    field: str, value: list[str]
) -> None:
    payload = _result_payload()
    protocol = payload["protocol"]
    assert isinstance(protocol, dict)
    protocol[field] = value
    with pytest.raises(ValueError, match=field):
        validate_development_result(payload)


def test_numeric_payloads_are_bounded_before_conversion_or_svd() -> None:
    oversized = np.broadcast_to(np.zeros(1, dtype=np.float64), (67_108_865,))
    with pytest.raises(ValueError, match="bounded"):
        estimate_optimization_readiness(
            loss=1.0,
            full_validation_gradient=oversized,
            batch_gradients=np.ones((1, 1)),
        )
    with pytest.raises(ValueError, match="bounded"):
        energy_rank(oversized.reshape(1, -1))


def test_protocol_records_estimator_differences_and_nonpromotion() -> None:
    assert OPTIMIZATION_READINESS_PROTOCOL["paper_revision"] == "arXiv:2605.09044v1"
    assert (
        OPTIMIZATION_READINESS_PROTOCOL["official_code_revision"]
        == "none-cited-in-arxiv-v1-as-of-2026-08-17"
    )
    assert (
        OPTIMIZATION_READINESS_PROTOCOL["estimator"]
        == "generic-equation-helper-plus-appendix-c.1-declared-mode"
    )
    assert OPTIMIZATION_READINESS_PROTOCOL["asi_protocol_differences"]
    assert OPTIMIZATION_READINESS_PROTOCOL["development_only"] is True
    assert OPTIMIZATION_READINESS_PROTOCOL["scientific_promotion_allowed"] is False
    assert OPTIMIZATION_READINESS_PROTOCOL["completed_result_exists"] is False
    assert OPTIMIZATION_READINESS_PROTOCOL["outcome_retention_required"] is True
    assert OPTIMIZATION_READINESS_PROTOCOL["resource_receipts_are_authenticated"] is False
    assert OPTIMIZATION_READINESS_PROTOCOL["metrics_are_recomputed_by_this_module"] is False
    assert OPTIMIZATION_READINESS_PROTOCOL["outcomes_are_derived_by_this_module"] is False
