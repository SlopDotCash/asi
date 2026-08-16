"""Strict toy checks for the development-only recurring-IPMNIST diagnostic."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alberta_framework.evaluation.recurring_ipmnist_retention import (
    RecurringIPMNISTPhase,
    RecurringIPMNISTProtocol,
    RecurringIPMNISTTrace,
    SentinelProbeBinding,
    SentinelProbeSnapshot,
    build_recurring_ipmnist_retention_report,
)

pytestmark = pytest.mark.unit


def _sha(character: str) -> str:
    return character * 64


def _protocol() -> RecurringIPMNISTProtocol:
    return RecurringIPMNISTProtocol(
        protocol_id="tests.recurring-ipmnist-retention.v1",
        phases=(
            RecurringIPMNISTPhase(
                phase_index=0,
                start_step=0,
                length=4,
                permutation_id="permutation-a.v1",
                exposure_index=0,
            ),
            RecurringIPMNISTPhase(
                phase_index=1,
                start_step=4,
                length=4,
                permutation_id="permutation-b.v1",
                exposure_index=0,
            ),
            RecurringIPMNISTPhase(
                phase_index=2,
                start_step=8,
                length=4,
                permutation_id="permutation-a.v1",
                exposure_index=1,
            ),
        ),
        sentinel_bindings=(
            SentinelProbeBinding(
                permutation_id="permutation-a.v1",
                permutation_sha256=_sha("a"),
                sentinel_set_id="sentinel-a.v1",
                sentinel_set_sha256=_sha("c"),
                sentinel_case_count=4,
            ),
            SentinelProbeBinding(
                permutation_id="permutation-b.v1",
                permutation_sha256=_sha("b"),
                sentinel_set_id="sentinel-b.v1",
                sentinel_set_sha256=_sha("d"),
                sentinel_case_count=4,
            ),
        ),
        relearning_window=2,
    )


def test_protocol_rejects_aggregate_trace_and_probe_workload_overflow() -> None:
    protocol = _protocol()
    too_many = (2**31 - 1) // 3 + 1
    oversized_bindings = (
        dataclasses.replace(protocol.sentinel_bindings[0], sentinel_case_count=too_many),
        protocol.sentinel_bindings[1],
    )
    with pytest.raises(ValueError, match="sentinel probe evaluations"):
        dataclasses.replace(protocol, sentinel_bindings=oversized_bindings)

    long_phase = 400_000_000
    with pytest.raises(ValueError, match="trace scalar count"):
        dataclasses.replace(
            protocol,
            phases=(
                dataclasses.replace(protocol.phases[0], length=long_phase),
                dataclasses.replace(
                    protocol.phases[1], start_step=long_phase, length=long_phase
                ),
                dataclasses.replace(
                    protocol.phases[2], start_step=2 * long_phase, length=long_phase
                ),
            ),
        )


def _trace(*, retaining: bool) -> RecurringIPMNISTTrace:
    revisit = (1.0, 1.0, 1.0, 1.0) if retaining else (0.0, 0.0, 1.0, 1.0)
    return RecurringIPMNISTTrace(
        pre_update_online_accuracy=(
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            *revisit,
        ),
        # Deliberately identical in both traces: same-example one-step loss
        # improvement cannot substitute for a frozen sentinel retention probe.
        post_update_one_step_plasticity=(
            1.0,
            1.0,
            0.2,
            0.1,
            1.0,
            0.5,
            0.2,
            0.1,
            1.0,
            1.0,
            0.2,
            0.1,
        ),
    )


def _snapshots(
    protocol: RecurringIPMNISTProtocol,
    *,
    retaining: bool,
) -> tuple[SentinelProbeSnapshot, ...]:
    # Required order is: A@A0, A@B0, B@B0, A@A1, B@A1.
    correct_counts = (4, 4 if retaining else 1, 3, 4, 3)
    state_hashes = (_sha("1"), _sha("2"), _sha("2"), _sha("3"), _sha("3"))
    return tuple(
        SentinelProbeSnapshot.from_requirement(
            requirement,
            learner_state_sha256_before=state_hash,
            learner_state_sha256_after=state_hash,
            correctness=(True,) * count + (False,) * (requirement.sentinel_case_count - count),
        )
        for requirement, count, state_hash in zip(
            protocol.required_probe_snapshots,
            correct_counts,
            state_hashes,
            strict=True,
        )
    )


def test_retaining_and_forgetting_traces_separate_retention_from_plasticity() -> None:
    protocol = _protocol()
    retaining = build_recurring_ipmnist_retention_report(
        protocol=protocol,
        trace=_trace(retaining=True),
        sentinel_snapshots=_snapshots(protocol, retaining=True),
    )
    forgetting = build_recurring_ipmnist_retention_report(
        protocol=protocol,
        trace=_trace(retaining=False),
        sentinel_snapshots=_snapshots(protocol, retaining=False),
    )

    assert retaining.recurrence.acquisition_end_sentinel_accuracy == 1.0
    assert retaining.recurrence.pre_revisit_sentinel_accuracy == 1.0
    assert retaining.recurrence.peak_to_revisit_forgetting == 0.0
    assert retaining.recurrence.relearning_savings_mistakes == 2
    assert retaining.recurrence.relearning_savings_accuracy == pytest.approx(1.0)

    assert forgetting.recurrence.acquisition_end_sentinel_accuracy == 1.0
    assert forgetting.recurrence.pre_revisit_sentinel_accuracy == 0.25
    assert forgetting.recurrence.retention_change_from_acquisition == -0.75
    assert forgetting.recurrence.peak_to_revisit_forgetting == 0.75
    assert forgetting.recurrence.revisit_end_sentinel_accuracy == 1.0
    assert forgetting.recurrence.revisit_recovery == 0.75
    assert forgetting.recurrence.relearning_savings_mistakes == 0

    assert retaining.recurrence.revisit_leading_one_step_plasticity == 1.0
    assert forgetting.recurrence.revisit_leading_one_step_plasticity == 1.0
    assert (
        retaining.recurrence.revisit_leading_one_step_plasticity
        == forgetting.recurrence.revisit_leading_one_step_plasticity
    )


def test_protocol_binds_aba_identity_exposures_and_keeps_trace_task_id_free() -> None:
    protocol = _protocol()
    assert tuple(field.name for field in dataclasses.fields(RecurringIPMNISTTrace)) == (
        "pre_update_online_accuracy",
        "post_update_one_step_plasticity",
    )
    assert tuple((phase.permutation_id, phase.exposure_index) for phase in protocol.phases) == (
        ("permutation-a.v1", 0),
        ("permutation-b.v1", 0),
        ("permutation-a.v1", 1),
    )
    assert tuple(
        (item.phase_index, item.permutation_id, item.exposure_index)
        for item in protocol.required_probe_snapshots
    ) == (
        (0, "permutation-a.v1", 0),
        (1, "permutation-a.v1", 0),
        (1, "permutation-b.v1", 0),
        (2, "permutation-a.v1", 1),
        (2, "permutation-b.v1", 0),
    )

    config = protocol.to_config()
    assert config["learner_visible_trace_fields"] == [
        "pre_update_online_accuracy",
        "post_update_one_step_plasticity",
    ]
    assert config["evaluator_only_fields"] == [
        "phase_index",
        "permutation_id",
        "permutation_sha256",
        "exposure_index",
        "sentinel_set_id",
        "sentinel_set_sha256",
        "sentinel_correctness",
    ]

    bad_final = dataclasses.replace(protocol.phases[2], exposure_index=0)
    with pytest.raises(ValueError, match="exposure_index"):
        dataclasses.replace(protocol, phases=(*protocol.phases[:2], bad_final))


def test_missing_reordered_or_malformed_sentinel_snapshots_fail_closed() -> None:
    protocol = _protocol()
    trace = _trace(retaining=True)
    snapshots = _snapshots(protocol, retaining=True)

    with pytest.raises(ValueError, match="exact required order"):
        build_recurring_ipmnist_retention_report(
            protocol=protocol,
            trace=trace,
            sentinel_snapshots=snapshots[:-1],
        )
    with pytest.raises(ValueError, match="exact required order"):
        build_recurring_ipmnist_retention_report(
            protocol=protocol,
            trace=trace,
            sentinel_snapshots=(snapshots[1], snapshots[0], *snapshots[2:]),
        )

    wrong_binding = dataclasses.replace(snapshots[1], sentinel_set_sha256=_sha("e"))
    with pytest.raises(ValueError, match="does not match its frozen requirement"):
        build_recurring_ipmnist_retention_report(
            protocol=protocol,
            trace=trace,
            sentinel_snapshots=(snapshots[0], wrong_binding, *snapshots[2:]),
        )

    short_probe = dataclasses.replace(snapshots[1], correctness=(True, False))
    with pytest.raises(ValueError, match="case count"):
        build_recurring_ipmnist_retention_report(
            protocol=protocol,
            trace=trace,
            sentinel_snapshots=(snapshots[0], short_probe, *snapshots[2:]),
        )

    with pytest.raises(ValueError, match="must not mutate learner state"):
        dataclasses.replace(snapshots[0], learner_state_sha256_after=_sha("f"))


def test_sentinel_probes_at_one_checkpoint_must_share_one_frozen_state() -> None:
    protocol = _protocol()
    snapshots = _snapshots(protocol, retaining=True)
    drifted = dataclasses.replace(
        snapshots[2], learner_state_sha256_before=_sha("9"), learner_state_sha256_after=_sha("9")
    )
    with pytest.raises(ValueError, match="one frozen state"):
        build_recurring_ipmnist_retention_report(
            protocol=protocol,
            trace=_trace(retaining=True),
            sentinel_snapshots=(*snapshots[:2], drifted, *snapshots[3:]),
        )


def test_sentinel_probes_at_different_checkpoints_must_use_distinct_frozen_states() -> None:
    """Re-scoring one learner state at every boundary would report zero forgetting."""
    protocol = _protocol()
    one_state = _sha("7")
    snapshots = tuple(
        dataclasses.replace(
            snapshot,
            learner_state_sha256_before=one_state,
            learner_state_sha256_after=one_state,
        )
        for snapshot in _snapshots(protocol, retaining=True)
    )
    with pytest.raises(
        ValueError,
        match=r"^sentinel probes at different checkpoints must use distinct frozen states; "
        r"checkpoint steps \[4, 8, 12\] all declare one learner state$",
    ):
        build_recurring_ipmnist_retention_report(
            protocol=protocol,
            trace=_trace(retaining=True),
            sentinel_snapshots=snapshots,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("sentinel_set_sha256", "A and B must bind distinct sentinel set digests"),
        ("sentinel_set_id", "A and B must bind distinct sentinel set identities"),
    ],
)
def test_protocol_rejects_a_and_b_sharing_one_sentinel_set(field: str, message: str) -> None:
    """Distinct permutations transform the sentinel inputs, so shared sets cannot be genuine."""
    protocol = _protocol()
    a_binding, b_binding = protocol.sentinel_bindings
    shared = dataclasses.replace(b_binding, **{field: getattr(a_binding, field)})
    with pytest.raises(ValueError, match=f"^{message}$"):
        dataclasses.replace(protocol, sentinel_bindings=(a_binding, shared))


def test_report_is_explicitly_threshold_free_development_only_and_nonpromoting() -> None:
    protocol = _protocol()
    report = build_recurring_ipmnist_retention_report(
        protocol=protocol,
        trace=_trace(retaining=True),
        sentinel_snapshots=_snapshots(protocol, retaining=True),
    )
    payload = report.to_config()
    assert payload["development_status"] == "development-only-not-assessed"
    assert payload["assessment_status"] == "not-assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert payload["retention_claimed"] is False
    assert payload["catastrophic_forgetting_absence_claimed"] is False


@pytest.mark.parametrize("bad_accuracy", [-0.1, 0.5, 1.1, float("nan")])
def test_trace_requires_binary_pre_update_outcomes(bad_accuracy: float) -> None:
    with pytest.raises(ValueError, match="binary"):
        RecurringIPMNISTTrace(
            pre_update_online_accuracy=(bad_accuracy,),
            post_update_one_step_plasticity=(0.2,),
        )


def test_recurring_ipmnist_dataclasses_reject_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="phase_index"):
        RecurringIPMNISTPhase(
            phase_index=True,  # type: ignore[arg-type]
            start_step=0,
            length=4,
            permutation_id="permutation-a.v1",
            exposure_index=0,
        )
    with pytest.raises(ValueError, match="sentinel_case_count"):
        SentinelProbeBinding(
            permutation_id="permutation-a.v1",
            permutation_sha256=_sha("a"),
            sentinel_set_id="sentinel-a.v1",
            sentinel_set_sha256=_sha("2"),
            sentinel_case_count=2.5,  # type: ignore[arg-type]
        )


def test_recurring_ipmnist_dataclasses_accept_and_canonicalize_numpy_integers() -> None:
    binding = SentinelProbeBinding(
        permutation_id="permutation-a.v1",
        permutation_sha256=_sha("a"),
        sentinel_set_id="sentinel-a.v1",
        sentinel_set_sha256=_sha("2"),
        sentinel_case_count=np.int32(4),
    )
    assert type(binding.sentinel_case_count) is int
    assert binding.sentinel_case_count == 4

    phase = RecurringIPMNISTPhase(
        phase_index=np.int32(0),
        start_step=np.int64(0),
        length=np.uint16(4),
        permutation_id="permutation-a.v1",
        exposure_index=np.uint8(0),
    )
    assert type(phase.phase_index) is int
    assert type(phase.start_step) is int
    assert type(phase.length) is int
    assert type(phase.exposure_index) is int
    assert phase.length == 4

    snapshot = SentinelProbeSnapshot(
        phase_index=np.int8(0),
        checkpoint_step=np.uint16(4),
        permutation_id="permutation-a.v1",
        permutation_sha256=_sha("a"),
        exposure_index=np.int64(0),
        sentinel_set_id="sentinel-a.v1",
        sentinel_set_sha256=_sha("2"),
        learner_state_sha256_before=_sha("3"),
        learner_state_sha256_after=_sha("3"),
        correctness=(True,),
    )
    assert type(snapshot.phase_index) is int
    assert type(snapshot.checkpoint_step) is int
    assert type(snapshot.exposure_index) is int
    assert type(snapshot.to_config()["checkpoint_step"]) is int


def test_recurring_ipmnist_phase_preflights_derived_stop_step() -> None:
    legal = RecurringIPMNISTPhase(
        phase_index=0,
        start_step=2**31 - 2,
        length=1,
        permutation_id="permutation-a.v1",
        exposure_index=0,
    )
    assert legal.stop_step == 2**31 - 1

    with pytest.raises(ValueError, match="stop_step"):
        RecurringIPMNISTPhase(
            phase_index=0,
            start_step=2**31 - 1,
            length=1,
            permutation_id="permutation-a.v1",
            exposure_index=0,
        )
