"""Scan sequence-length ceiling for the off-policy Horde learning loop (hang guard).

``run_off_policy_horde_learning_loop`` hands its caller-supplied step arrays
(``observations``, ``cumulants``, ``next_observations``, ``rhos``, and
optionally ``discounts``) straight to ``jax.lax.scan`` with no bound on the
leading (step) axis and no check that the arrays share a length. A hostile or
mistaken caller supplying a huge array forces JAX to trace/compile a scan of
that length, hanging the process well before any step executes -- the same
hang class already fixed for other scan-driven array loops elsewhere in this
repository (``core/horde.py``, ``core/sarsa.py``,
``core/partner_policy_fusion.py``, ``core/learners.py``, ``utils/nexting.py``).

``run_off_policy_horde_learning_loop`` and
``run_off_policy_horde_learning_loop_batched`` are exported public API
(``alberta_framework.__init__``); the batched variant calls the guarded
function internally and inherits the guard transitively.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.off_policy_horde import (
    _OFF_POLICY_HORDE_SEQUENCE_MAX_STEPS,
    OffPolicyHordeLearner,
    _require_off_policy_horde_matching_length,
    _require_off_policy_horde_sequence_length,
    run_off_policy_horde_learning_loop,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, create_horde_spec


def _spec(
    gammas: tuple[float, ...] = (0.0, 0.5), lamdas: tuple[float, ...] = (0.0, 0.4)
) -> HordeSpec:
    demons = tuple(
        GVFSpec(
            name=f"demon_{i}",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=lamdas[i],
            cumulant_index=i,
        )  # type: ignore[call-arg]
        for i, gamma in enumerate(gammas)
    )
    return create_horde_spec(demons)


def _learner_and_state(
    n_demons: int = 2, feature_dim: int = 3
) -> tuple[OffPolicyHordeLearner, MultiHeadMLPState]:
    learner = OffPolicyHordeLearner(
        _spec(),
        hidden_sizes=(5,),
        optimizer=LMS(step_size=0.01),
        ratio_clip=2.0,
    )
    state = learner.init(feature_dim, jr.key(4))
    return learner, state


def _arrays(
    num_steps: int, feature_dim: int = 3, n_demons: int = 2
) -> tuple[Array, Array, Array, Array]:
    key = jr.key(42)
    k1, k2, k3 = jr.split(key, 3)
    observations = jr.normal(k1, (num_steps, feature_dim))
    cumulants = jr.normal(k2, (num_steps, n_demons))
    next_observations = jr.normal(k3, (num_steps, feature_dim))
    rhos = jnp.ones((num_steps, n_demons), dtype=jnp.float32)
    return observations, cumulants, next_observations, rhos


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={first.shape[0]}")

    monkeypatch.setattr("alberta_framework.core.off_policy_horde.jax.lax.scan", spy)
    return seen


# =============================================================================
# _require_off_policy_horde_sequence_length / _require_off_policy_horde_matching_length
# =============================================================================


class TestRequireOffPolicyHordeSequenceLength:
    def test_rejects_non_array(self) -> None:
        with pytest.raises(TypeError, match="must be a JAX array"):
            _require_off_policy_horde_sequence_length("observations", [1.0, 2.0, 3.0])

    def test_rejects_scalar(self) -> None:
        with pytest.raises(ValueError, match="leading step axis"):
            _require_off_policy_horde_sequence_length("observations", jnp.asarray(1.0))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            _require_off_policy_horde_sequence_length("observations", jnp.zeros((0, 3)))

    def test_rejects_oversized(self) -> None:
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            _require_off_policy_horde_sequence_length(
                "observations",
                jnp.zeros((_OFF_POLICY_HORDE_SEQUENCE_MAX_STEPS + 1, 3)),
            )

    def test_accepts_last_fit_length(self) -> None:
        length = _require_off_policy_horde_sequence_length(
            "observations", jnp.zeros((_OFF_POLICY_HORDE_SEQUENCE_MAX_STEPS, 3))
        )
        assert length == _OFF_POLICY_HORDE_SEQUENCE_MAX_STEPS

    def test_rejects_origin_hang_class_sized(self) -> None:
        # Mirrors the hang class this guard closes: a caller-supplied huge
        # leading axis must be rejected long before it reaches lax.scan.
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            _require_off_policy_horde_sequence_length("observations", jnp.zeros((200_000, 3)))


class TestRequireOffPolicyHordeMatchingLength:
    def test_rejects_non_array(self) -> None:
        with pytest.raises(TypeError, match="must be a JAX array"):
            _require_off_policy_horde_matching_length("cumulants", [1.0, 2.0], expected=2)

    def test_rejects_mismatched_length(self) -> None:
        with pytest.raises(ValueError, match="must share the same leading length"):
            _require_off_policy_horde_matching_length("cumulants", jnp.zeros((3, 2)), expected=5)

    def test_accepts_matching_length(self) -> None:
        _require_off_policy_horde_matching_length("cumulants", jnp.zeros((5, 2)), expected=5)


# =============================================================================
# run_off_policy_horde_learning_loop
# =============================================================================


class TestRunOffPolicyHordeLearningLoopSequenceLengthGuard:
    def test_oversized_rejected_before_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy_scan(monkeypatch)
        learner, state = _learner_and_state()
        observations, cumulants, next_observations, rhos = _arrays(
            _OFF_POLICY_HORDE_SEQUENCE_MAX_STEPS + 1
        )
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            run_off_policy_horde_learning_loop(
                learner, state, observations, cumulants, next_observations, rhos
            )
        assert seen == []

    def test_origin_hang_class_sized_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner, state = _learner_and_state()
        observations, cumulants, next_observations, rhos = _arrays(200_000)
        with pytest.raises(ValueError, match=r"length must be an integer in \[1,"):
            run_off_policy_horde_learning_loop(
                learner, state, observations, cumulants, next_observations, rhos
            )
        assert seen == []

    def test_mismatched_cumulants_length_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner, state = _learner_and_state()
        observations, _, next_observations, rhos = _arrays(10)
        _, cumulants, _, _ = _arrays(9)
        with pytest.raises(ValueError, match="must share the same leading length"):
            run_off_policy_horde_learning_loop(
                learner, state, observations, cumulants, next_observations, rhos
            )
        assert seen == []

    def test_mismatched_rhos_length_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner, state = _learner_and_state()
        observations, cumulants, next_observations, _ = _arrays(10)
        _, _, _, rhos = _arrays(9)
        with pytest.raises(ValueError, match="must share the same leading length"):
            run_off_policy_horde_learning_loop(
                learner, state, observations, cumulants, next_observations, rhos
            )
        assert seen == []

    def test_mismatched_discounts_length_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner, state = _learner_and_state()
        observations, cumulants, next_observations, rhos = _arrays(10)
        discounts = jnp.zeros((9, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="must share the same leading length"):
            run_off_policy_horde_learning_loop(
                learner,
                state,
                observations,
                cumulants,
                next_observations,
                rhos,
                discounts,
            )
        assert seen == []

    def test_last_fit_length_still_runs(self) -> None:
        learner, state = _learner_and_state()
        observations, cumulants, next_observations, rhos = _arrays(16)
        result = run_off_policy_horde_learning_loop(
            learner, state, observations, cumulants, next_observations, rhos
        )
        assert result.td_errors.shape[0] == 16

    def test_explicit_discounts_still_runs(self) -> None:
        learner, state = _learner_and_state()
        observations, cumulants, next_observations, rhos = _arrays(8)
        discounts = jnp.full((8, 2), 0.5, dtype=jnp.float32)
        result = run_off_policy_horde_learning_loop(
            learner,
            state,
            observations,
            cumulants,
            next_observations,
            rhos,
            discounts,
        )
        assert result.td_errors.shape[0] == 8
