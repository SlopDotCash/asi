"""Seed-domain contract for the scale-robust pair-feature evaluation entrypoint.

``run_scale_robust_feature_evaluation`` is the executable protocol behind the
registered ``scale_robust_pair_features`` evidence claim.  Its ``seeds``
argument used to be canonicalized with a bare ``int(seed)`` call plus manual
non-empty/unique/non-negative checks -- silently truncating floats, silently
coercing ``bool``/numeric-string/other numeric types, and never bounding a
seed to the JAX uint32 scalar-key domain.  ``jax.random.key`` reduces an
out-of-range seed modulo ``2**32``, so two distinct declared seeds could
silently execute the identical random stream while being recorded as
different identities in the evidence artifact.

These tests exercise only the validation boundary, which runs before any JAX
computation starts, so they stay fast: a rejection must be raised before the
(expensive) protocol would ever begin.
"""

from __future__ import annotations

import json

import pytest

from alberta_framework._seed_validation import JAX_KEY_SEED_MAX
from alberta_framework.evaluation.scale_robust_feature import (
    CONDITION_PRIMARY,
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    ConditionSeedRecord,
    PhaseWindowRecord,
    ScaleRobustFeatureReport,
    run_scale_robust_feature_evaluation,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import _phase_payload


class _ProtocolStartedError(Exception):
    """Raised by the patched stream constructor once validation has passed."""


@pytest.fixture(autouse=True)
def _fail_if_protocol_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any post-validation progress fail loudly and cheaply.

    ``run_scale_robust_feature_evaluation`` constructs a ``GauntletStream``
    immediately after validating ``seeds``.  Patching the constructor to
    raise turns "validation silently passed" into a clear test failure
    instead of a multi-minute JAX run.
    """

    import alberta_framework.evaluation.scale_robust_feature as module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise _ProtocolStartedError("seed validation passed; protocol would have started")

    monkeypatch.setattr(module, "GauntletStream", _boom)


@pytest.mark.parametrize(
    "bad_seeds",
    [
        pytest.param((True,), id="bool"),
        pytest.param((False,), id="bool_false"),
        pytest.param((5.9,), id="float_would_silently_truncate"),
        pytest.param((5.0,), id="float_exact_value"),
        pytest.param(("5",), id="numeric_string"),
        pytest.param((JAX_KEY_SEED_MAX + 1,), id="above_uint32_domain"),
        pytest.param((JAX_KEY_SEED_MAX + 1 + (1 << 40),), id="far_above_uint32_domain"),
        pytest.param((-1,), id="negative"),
    ],
)
def test_rejects_non_canonical_seed(bad_seeds: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_scale_robust_feature_evaluation(seeds=bad_seeds)  # type: ignore[arg-type]


def test_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_scale_robust_feature_evaluation(seeds=())


def test_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_scale_robust_feature_evaluation(seeds=(5, 5))


def test_rejects_noncanonical_seed_container_before_hooks() -> None:
    class ListSubclass(list[int]):
        pass

    class TupleSubclass(tuple[int, ...]):
        pass

    class SequenceSpoof:
        @property  # type: ignore[misc]
        def __class__(self) -> type:
            return tuple

        def __iter__(self) -> object:
            raise AssertionError("iteration hook executed")

        def __len__(self) -> int:
            raise AssertionError("length hook executed")

        def __repr__(self) -> str:
            raise AssertionError("repr hook executed")

    for bad_seeds in (ListSubclass([1]), TupleSubclass((1,)), SequenceSpoof()):
        with pytest.raises(ValueError, match="actual list or tuple"):
            run_scale_robust_feature_evaluation(seeds=bad_seeds)  # type: ignore[arg-type]


def test_out_of_range_seed_would_alias_an_in_range_seed() -> None:
    """Document the exact corruption the domain bound prevents.

    Without the ``[0, 2**32 - 1]`` bound, seed ``2**32`` would construct the
    identical JAX key as seed ``0`` -- two declared identities, one actual
    random stream.
    """
    import jax.random as jr

    aliasing_seed = JAX_KEY_SEED_MAX + 1  # == 2**32, wraps to 0 mod 2**32
    assert bool(jr.key_data(jr.key(0)).tolist() == jr.key_data(jr.key(aliasing_seed)).tolist())
    with pytest.raises(ValueError):
        run_scale_robust_feature_evaluation(seeds=(aliasing_seed,))


def test_canonical_evidence_and_development_seeds_pass_validation() -> None:
    """The frozen seed schedules validate cleanly (protocol start is stubbed)."""
    with pytest.raises(_ProtocolStartedError):
        run_scale_robust_feature_evaluation(seeds=EVIDENCE_SEEDS)
    with pytest.raises(_ProtocolStartedError):
        run_scale_robust_feature_evaluation(seeds=DEVELOPMENT_SEEDS)


def _legal_phase(**overrides: object) -> PhaseWindowRecord:
    payload: dict[str, object] = {
        "phase_index": 0,
        "phase_name": "scale",
        "step_count": 1,
        "nonfinite_steps": 0,
        "phase_squared_error_sum": 1.0,
        "early_squared_error_sum": 1.0,
        "early_count": 1,
        "tail_squared_error_sum": 1.0,
        "tail_count": 1,
        "asymptotic_squared_error_sum": 1.0,
        "asymptotic_count": 1,
    }
    payload.update(overrides)
    return PhaseWindowRecord(**payload)  # type: ignore[arg-type]


def test_phase_window_record_rejects_bool_and_nonfinite_identities() -> None:
    """Public measurement records must not keep leftover bool/NaN identities."""

    with pytest.raises(ValueError, match="phase_index"):
        _legal_phase(phase_index=True)
    with pytest.raises(ValueError, match="step_count"):
        _legal_phase(step_count=False)
    with pytest.raises(ValueError, match="phase_squared_error_sum"):
        _legal_phase(phase_squared_error_sum=float("nan"))
    with pytest.raises(ValueError, match="early_squared_error_sum"):
        _legal_phase(early_squared_error_sum=float("inf"))

    legal = _legal_phase()
    dumped = json.dumps(_phase_payload(legal), allow_nan=False)
    assert '"phase_index": 0' in dumped
    assert '"phase_index": true' not in dumped
    assert _legal_phase(phase_squared_error_sum=None).phase_squared_error_sum is None


def test_condition_seed_and_report_reject_leftover_identities() -> None:
    with pytest.raises(ValueError, match="seed"):
        ConditionSeedRecord(
            seed=True,  # type: ignore[arg-type]
            condition=CONDITION_PRIMARY,
            phases=(),
            end_segment_5_active_pairs=(),
            end_segment_7_active_pairs=(),
            final_active_pairs=(),
        )
    with pytest.raises(ValueError, match="end_segment_5_active_pairs"):
        ConditionSeedRecord(
            seed=8,
            condition=CONDITION_PRIMARY,
            phases=(),
            end_segment_5_active_pairs=((True, 1),),  # type: ignore[arg-type]
            end_segment_7_active_pairs=(),
            final_active_pairs=(),
        )
    with pytest.raises(ValueError, match="seeds"):
        ScaleRobustFeatureReport(
            seeds=(True,),  # type: ignore[arg-type]
            records=(),
            memory_by_condition={},
            wall_time_seconds_by_condition={CONDITION_PRIMARY: 1.0},
        )
    with pytest.raises(ValueError, match="wall_time_seconds_by_condition"):
        ScaleRobustFeatureReport(
            seeds=(8,),
            records=(),
            memory_by_condition={},
            wall_time_seconds_by_condition={CONDITION_PRIMARY: float("nan")},
        )

    report = ScaleRobustFeatureReport(
        seeds=(8,),
        records=(),
        memory_by_condition={},
        wall_time_seconds_by_condition={CONDITION_PRIMARY: 1.25},
    )
    assert report.seeds == (8,)
    assert report.wall_time_seconds_by_condition[CONDITION_PRIMARY] == 1.25


@pytest.mark.parametrize(
    "field",
    (
        "phase_squared_error_sum",
        "early_squared_error_sum",
        "tail_squared_error_sum",
        "asymptotic_squared_error_sum",
    ),
)
def test_phase_window_record_rejects_negative_squared_error_sums(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _legal_phase(**{field: -0.001})


def test_scale_report_rejects_negative_wall_time() -> None:
    with pytest.raises(ValueError, match="wall_time_seconds_by_condition"):
        ScaleRobustFeatureReport(
            seeds=(8,),
            records=(),
            memory_by_condition={},
            wall_time_seconds_by_condition={CONDITION_PRIMARY: -0.001},
        )
