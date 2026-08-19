"""Scan sequence-length ceiling for public Horde learning-loop scans (hang guard).

``run_horde_learning_loop``, ``run_mixed_horde_learning_loop``, and
``run_horde_learning_loop_final_state`` hand their caller-supplied step arrays
(``observations``, ``cumulants``, ``next_observations``) straight to
``jax.lax.scan`` with no bound on the leading (step) axis and no check that
the three arrays share a length. A hostile or mistaken caller supplying a huge
array forces JAX to trace/compile a scan of that length, hanging the process
well before any step executes -- the same hang class already fixed for other
scan-driven array loops elsewhere in this repository (``core/sarsa.py``,
``core/partner_policy_fusion.py``, ``core/learners.py``, ``utils/nexting.py``).

All three functions are exported public API (``alberta_framework.__init__``).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework import (
    DemonType,
    GVFSpec,
    HordeLearner,
    MixedHorde,
    ObGDBounding,
    create_horde_spec,
    run_horde_learning_loop,
    run_horde_learning_loop_final_state,
    run_mixed_horde_learning_loop,
)
from alberta_framework.core.horde import (
    _HORDE_SEQUENCE_MAX_STEPS,
    MixedHordeState,
    _require_horde_matching_length,
    _require_horde_sequence_length,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPState


def _make_all_gamma0_spec(n: int) -> list[GVFSpec]:
    return [
        GVFSpec(
            name=f"d{i}", demon_type=DemonType.PREDICTION, gamma=0.0, lamda=0.0, cumulant_index=i
        )
        for i in range(n)
    ]


def _shared_horde(
    n_demons: int = 2, feature_dim: int = 5
) -> tuple[HordeLearner, MultiHeadMLPState]:
    horde = HordeLearner(
        horde_spec=create_horde_spec(_make_all_gamma0_spec(n_demons)),
        hidden_sizes=(4,),
        sparsity=0.0,
        bounder=ObGDBounding(kappa=2.0),
    )
    return horde, horde.init(feature_dim, jr.key(0))


def _mixed_horde(n_demons: int = 2, feature_dim: int = 5) -> tuple[MixedHorde, MixedHordeState]:
    mixed = MixedHorde(
        horde_spec=create_horde_spec(_make_all_gamma0_spec(n_demons)),
        hidden_sizes=(4,),
        sparsity=0.0,
        bounder=ObGDBounding(kappa=2.0),
    )
    return mixed, mixed.init(feature_dim, jr.key(0))


def _arrays(num_steps: int, feature_dim: int = 5, n_demons: int = 2) -> tuple[Array, Array, Array]:
    key = jr.key(42)
    k1, k2 = jr.split(key)
    observations = jr.normal(k1, (num_steps, feature_dim))
    cumulants = jr.normal(k2, (num_steps, n_demons))
    next_observations = jnp.zeros((num_steps, feature_dim))
    return observations, cumulants, next_observations


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={first.shape[0]}")

    monkeypatch.setattr("alberta_framework.core.horde.jax.lax.scan", spy)
    return seen


# =============================================================================
# _require_horde_sequence_length / _require_horde_matching_length unit tests
# =============================================================================


class TestRequireHordeSequenceLength:
    def test_rejects_non_array(self) -> None:
        with pytest.raises(TypeError, match="must be a JAX array"):
            _require_horde_sequence_length("observations", [1.0, 2.0, 3.0])

    def test_rejects_scalar(self) -> None:
        with pytest.raises(ValueError, match="leading step axis"):
            _require_horde_sequence_length("observations", jnp.asarray(1.0))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            _require_horde_sequence_length("observations", jnp.zeros((0, 5)))

    def test_rejects_oversized(self) -> None:
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            _require_horde_sequence_length(
                "observations", jnp.zeros((_HORDE_SEQUENCE_MAX_STEPS + 1, 5))
            )

    def test_accepts_last_fit_length(self) -> None:
        length = _require_horde_sequence_length(
            "observations", jnp.zeros((_HORDE_SEQUENCE_MAX_STEPS, 5))
        )
        assert length == _HORDE_SEQUENCE_MAX_STEPS

    def test_rejects_origin_hang_class_sized(self) -> None:
        # Mirrors the hang class this guard closes: a caller-supplied huge
        # leading axis must be rejected long before it reaches lax.scan.
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            _require_horde_sequence_length("observations", jnp.zeros((200_000, 5)))


class TestRequireHordeMatchingLength:
    def test_rejects_non_array(self) -> None:
        with pytest.raises(TypeError, match="must be a JAX array"):
            _require_horde_matching_length("cumulants", [1.0, 2.0], expected=2)

    def test_rejects_mismatched_length(self) -> None:
        with pytest.raises(ValueError, match="must share the same leading length"):
            _require_horde_matching_length("cumulants", jnp.zeros((3, 2)), expected=5)

    def test_accepts_matching_length(self) -> None:
        _require_horde_matching_length("cumulants", jnp.zeros((5, 2)), expected=5)


# =============================================================================
# run_horde_learning_loop
# =============================================================================


class TestRunHordeLearningLoopSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        horde, state = _shared_horde()
        observations, cumulants, next_observations = _arrays(_HORDE_SEQUENCE_MAX_STEPS + 1)
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            run_horde_learning_loop(horde, state, observations, cumulants, next_observations)
        assert seen == []

    def test_origin_hang_class_sized_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        horde, state = _shared_horde()
        observations, cumulants, next_observations = _arrays(200_000)
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            run_horde_learning_loop(horde, state, observations, cumulants, next_observations)
        assert seen == []

    def test_mismatched_length_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        horde, state = _shared_horde()
        observations, cumulants, _ = _arrays(10)
        _, _, next_observations = _arrays(9)
        with pytest.raises(ValueError, match="must share the same leading length"):
            run_horde_learning_loop(horde, state, observations, cumulants, next_observations)
        assert seen == []

    def test_last_fit_length_still_runs(self) -> None:
        horde, state = _shared_horde()
        observations, cumulants, next_observations = _arrays(16)
        result = run_horde_learning_loop(horde, state, observations, cumulants, next_observations)
        assert result.td_errors.shape[0] == 16


# =============================================================================
# run_mixed_horde_learning_loop
# =============================================================================


class TestRunMixedHordeLearningLoopSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        mixed, state = _mixed_horde()
        observations, cumulants, next_observations = _arrays(_HORDE_SEQUENCE_MAX_STEPS + 1)
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            run_mixed_horde_learning_loop(
                mixed, state, observations, cumulants, next_observations
            )
        assert seen == []

    def test_last_fit_length_still_runs(self) -> None:
        mixed, state = _mixed_horde()
        observations, cumulants, next_observations = _arrays(16)
        result = run_mixed_horde_learning_loop(
            mixed, state, observations, cumulants, next_observations
        )
        assert result.td_errors.shape[0] == 16


# =============================================================================
# run_horde_learning_loop_final_state
# =============================================================================


class TestRunHordeLearningLoopFinalStateSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        horde, state = _shared_horde()
        observations, cumulants, next_observations = _arrays(_HORDE_SEQUENCE_MAX_STEPS + 1)
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            run_horde_learning_loop_final_state(
                horde, state, observations, cumulants, next_observations
            )
        assert seen == []

    def test_last_fit_length_still_runs(self) -> None:
        horde, state = _shared_horde()
        observations, cumulants, next_observations = _arrays(16)
        final_state = run_horde_learning_loop_final_state(
            horde, state, observations, cumulants, next_observations
        )
        assert final_state is not None
