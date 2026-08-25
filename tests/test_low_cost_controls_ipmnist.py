"""Contract tests for the issue #1566 low-cost activation comparator lane."""

import dataclasses
import math

import numpy as np
import pytest

from alberta_framework.benchmarks.low_cost_controls_ipmnist import (
    ARM_IDS,
    FROZEN_SEEDS,
    SCHEMA,
    LowCostArmResult,
    LowCostCatalogEntry,
    LowCostResult,
    _preactivation_width,
    qualification_gates,
    run_comparator,
)


def _fixture(rows: int = 32):
    rng = np.random.default_rng(0)
    images = rng.random((rows, 784), dtype=np.float32)
    labels = rng.integers(0, 10, size=rows).astype(np.int32)
    return images, labels


def _arm(result, arm_id):
    return next(arm for arm in result.arms if arm.arm_id == arm_id)


class TestHarnessNeutrality:
    def test_aid_off_reproduces_the_relu_control_exactly(self) -> None:
        # AID at relu_probability=1 *is* ReLU, so this arm must trace the control
        # exactly. Any divergence means the harness itself, not the mechanism,
        # moved the numbers -- which would invalidate every other comparison.
        images, labels = _fixture()
        result = run_comparator(images, labels)
        control = _arm(result, "sgd_current_control")
        aid_off = _arm(result, "aid_off")
        assert aid_off.task_accuracy == control.task_accuracy
        assert aid_off.task_loss == control.task_loss
        assert aid_off.effective_rank == control.effective_rank
        assert aid_off.persistent_bytes == control.persistent_bytes

    def test_an_enabled_mechanism_actually_changes_the_trace(self) -> None:
        # Guards the converse: if every arm agreed, the lane would be measuring
        # nothing at all.
        images, labels = _fixture()
        result = run_comparator(images, labels)
        control = _arm(result, "sgd_current_control")
        assert _arm(result, "smooth_leaky").task_loss != control.task_loss
        assert _arm(result, "deep_fourier").task_loss != control.task_loss


class TestCapacityAccounting:
    def test_deep_fourier_halves_preactivations_to_match_output_width(self) -> None:
        images, labels = _fixture()
        result = run_comparator(images, labels)
        control = _arm(result, "sgd_current_control")
        fourier = _arm(result, "deep_fourier")
        assert fourier.preactivation_width * 2 == control.preactivation_width
        # The parameter delta is real and must be reported, not concealed.
        assert fourier.persistent_bytes < control.persistent_bytes

    def test_disabled_fourier_keeps_the_control_geometry(self) -> None:
        images, labels = _fixture()
        result = run_comparator(images, labels)
        assert (
            _arm(result, "deep_fourier_off").preactivation_width
            == _arm(result, "sgd_current_control").preactivation_width
        )

    def test_odd_hidden_width_is_refused_for_the_fourier_arm(self) -> None:
        with pytest.raises(ValueError, match="even hidden width"):
            _preactivation_width("deep_fourier", 7)


class TestCatalogProvenance:
    def test_default_catalog_validates(self) -> None:
        LowCostCatalogEntry().validate()

    def test_unlicensed_official_code_cannot_be_marked_copied(self) -> None:
        entry = LowCostCatalogEntry(smooth_leaky_official_code_copied=True)
        with pytest.raises(ValueError, match="unlicensed repository"):
            entry.validate()

    def test_located_official_code_cannot_be_erased(self) -> None:
        entry = LowCostCatalogEntry(smooth_leaky_official_code_available=False)
        with pytest.raises(ValueError, match="official code must stay recorded"):
            entry.validate()

    def test_absent_official_code_cannot_be_invented(self) -> None:
        entry = LowCostCatalogEntry(aid_official_code_available=True)
        with pytest.raises(ValueError, match="explicitly absent"):
            entry.validate()

    def test_paper_identity_drift_is_refused(self) -> None:
        entry = LowCostCatalogEntry(aid_paper_identity="ICML-2025:not-the-paper")
        with pytest.raises(ValueError, match="identity drift"):
            entry.validate()

    def test_promotion_cannot_be_enabled(self) -> None:
        entry = LowCostCatalogEntry(scientific_promotion_allowed=True)
        with pytest.raises(ValueError, match="nonpromoting"):
            entry.validate()


class TestResultContract:
    def test_every_arm_is_reported_in_declared_order(self) -> None:
        images, labels = _fixture()
        result = run_comparator(images, labels)
        assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
        assert result.schema == SCHEMA

    def test_a_missing_arm_is_refused(self) -> None:
        images, labels = _fixture()
        result = run_comparator(images, labels)
        with pytest.raises(ValueError, match="every declared arm"):
            LowCostResult(
                schema=SCHEMA,
                profile_id=result.profile_id,
                seed=result.seed,
                dataset_sha256=result.dataset_sha256,
                catalog=LowCostCatalogEntry(),
                arms=result.arms[:-1],
            )

    def test_reordered_arms_are_refused(self) -> None:
        images, labels = _fixture()
        result = run_comparator(images, labels)
        with pytest.raises(ValueError, match="arm order"):
            LowCostResult(
                schema=SCHEMA,
                profile_id=result.profile_id,
                seed=result.seed,
                dataset_sha256=result.dataset_sha256,
                catalog=LowCostCatalogEntry(),
                arms=tuple(reversed(result.arms)),
            )

    def test_reconstructed_result_revalidates_nested_arms(self) -> None:
        images, labels = _fixture()
        result = run_comparator(images, labels)
        object.__setattr__(
            result.arms[0],
            "task_accuracy",
            (math.nan, *result.arms[0].task_accuracy[1:]),
        )

        with pytest.raises(ValueError, match="diagnostic curves"):
            dataclasses.replace(result)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"seed": True}, "frozen development seed"),
            ({"seed": FROZEN_SEEDS[0] - 1}, "frozen development seed"),
            ({"dataset_sha256": "not-a-digest"}, "dataset identity"),
        ],
    )
    def test_result_rejects_malformed_schedule_identity(
        self, changes: dict[str, object], message: str
    ) -> None:
        images, labels = _fixture()
        result = run_comparator(images, labels)

        with pytest.raises(ValueError, match=message):
            dataclasses.replace(result, **changes)

    def test_arm_binds_mechanism_flag_to_roster(self) -> None:
        with pytest.raises(ValueError, match="mechanism flag"):
            LowCostArmResult(
                arm_id="sgd_current_control",
                mechanism_enabled=True,
                preactivation_width=8,
                task_accuracy=(0.5,),
                task_loss=(1.0,),
                dead_unit_fraction=(0.0,),
                effective_rank=(1.0,),
                persistent_bytes=1,
                parameter_updates=1,
                model_queries=1,
                elapsed_ns=0,
            )

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("task_accuracy", (1.1,), "probability curves"),
            ("dead_unit_fraction", (-0.1,), "probability curves"),
            ("task_loss", (-0.1,), "nonnegative"),
            ("effective_rank", (-0.1,), "nonnegative"),
            ("task_loss", (1.0, 2.0), "equal lengths"),
        ],
    )
    def test_arm_curves_are_bounded_and_equal_length(
        self, field: str, value: tuple[float, ...], message: str
    ) -> None:
        fields = {
            "arm_id": "sgd_current_control",
            "mechanism_enabled": False,
            "preactivation_width": 8,
            "task_accuracy": (0.5,),
            "task_loss": (1.0,),
            "dead_unit_fraction": (0.0,),
            "effective_rank": (1.0,),
            "persistent_bytes": 1,
            "parameter_updates": 1,
            "model_queries": 1,
            "elapsed_ns": 0,
        }
        fields[field] = value

        with pytest.raises(ValueError, match=message):
            LowCostArmResult(**fields)

    def test_unknown_profile_is_refused(self) -> None:
        images, labels = _fixture()
        with pytest.raises(ValueError, match="unknown profile"):
            run_comparator(images, labels, profile_id="not-a-profile")

    def test_gates_declare_each_mechanism_off_reduction(self) -> None:
        gates = qualification_gates()
        assert gates["development_only"] is True
        assert gates["scientific_promotion_allowed"] is False
        assert set(gates["mechanism_off_reductions"]) == {
            "smooth_leaky_off",
            "aid_off",
            "deep_fourier_off",
        }
