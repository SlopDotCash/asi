"""Tests for the Pavlovian / classical-conditioning stream.

Covers:
- Trial dynamics (CS-then-US pairing at the configured delay).
- Phase progression (extinction, blocking).
- Statistical properties (partial reinforcement rate, distractor
  independence).
- JIT compatibility under ``jax.lax.scan``.
- Determinism with the same key.
"""

from __future__ import annotations

from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    PavlovianPhase,
    PavlovianState,
    acquisition_scenario,
    blocking_scenario,
    extinction_scenario,
    partial_reinforcement_scenario,
    reacquisition_scenario,
)

# ``np.longdouble`` is only wider than float64 on platforms whose C long double
# carries extra mantissa bits (x86-64 80-bit extended, or IEEE quad). On
# aarch64 macOS it *is* float64, so no longdouble value can distinguish a
# single narrowing from a double-rounded one, and the smallest longdouble
# subnormal survives ``float()`` unchanged -- the two properties below become
# untestable rather than false. The Fraction-parametrized cases keep the
# double-rounding semantics covered on every platform.
LONGDOUBLE_IS_EXTENDED = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
requires_extended_longdouble = pytest.mark.skipif(
    not LONGDOUBLE_IS_EXTENDED,
    reason="requires np.longdouble wider than float64; on this platform they are identical",
)

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(
    stream: ClassicalConditioningStream,
    state: PavlovianState,
    n_steps: int,
) -> tuple[PavlovianState, jnp.ndarray, jnp.ndarray]:
    """Run a stream for ``n_steps`` and return (final_state, obs, targets)."""
    obs_list = []
    tgt_list = []
    for i in range(n_steps):
        ts, state = stream.step(state, jnp.array(i))
        obs_list.append(ts.observation)
        tgt_list.append(ts.target)
    return state, jnp.stack(obs_list), jnp.stack(tgt_list)


# ---------------------------------------------------------------------------
# Trial dynamics
# ---------------------------------------------------------------------------


def test_acquisition_us_follows_cs():
    """Every US must be preceded by a CS exactly ``cs_us_delay`` steps prior.

    In a noiseless ACQUISITION scenario with contingency = 1.0 the count
    of (CS_onset_at_t, US_at_t+delay) pairs should equal the count of US
    events.
    """
    cs_us_delay = 5
    stream = acquisition_scenario(
        n_steps=2000,
        cs_us_delay=cs_us_delay,
        cs_duration=1,
        iti_min=5,
        iti_max=10,
        noise_std=0.0,
        distractor_prob=0.0,
    )
    state = stream.init(jr.key(0))
    _, obs, tgt = _collect(stream, state, 1000)

    cs_indicator = (obs[:, 0] > 0.5).astype(jnp.float32)
    us_indicator = (tgt[:, 0] > 0.5).astype(jnp.float32)

    n_us = int(jnp.sum(us_indicator))
    assert n_us > 0, "expected at least some US events"

    # For every step where US fires, the CS must have been on at t - delay.
    us_indices = jnp.where(us_indicator > 0.5)[0]
    # Drop early indices where t - delay < 0.
    us_indices = us_indices[us_indices >= cs_us_delay]

    cs_at_minus_delay = cs_indicator[us_indices - cs_us_delay]
    n_cs_then_us = int(jnp.sum(cs_at_minus_delay > 0.5))
    assert n_cs_then_us == int(us_indices.shape[0])


def test_extinction_no_us_after_extinction_phase():
    """No US fires once we are unambiguously inside the extinction phase.

    Any trial in flight at the phase boundary is cancelled by the stream,
    so US events should be exactly zero throughout the extinction phase.
    """
    n_acq = 500
    n_ext = 500
    stream = extinction_scenario(
        n_acquisition=n_acq,
        n_extinction=n_ext,
        cs_us_delay=5,
        cs_duration=1,
        iti_min=5,
        iti_max=10,
        noise_std=0.0,
        distractor_prob=0.0,
    )
    state = stream.init(jr.key(1))
    final_state, obs, tgt = _collect(stream, state, n_acq + n_ext)

    us_indicator = tgt[:, 0] > 0.5
    us_in_acq = int(jnp.sum(us_indicator[:n_acq]))
    us_in_ext = int(jnp.sum(us_indicator[n_acq:]))

    assert us_in_acq > 0, "acquisition phase must produce US events"
    assert us_in_ext == 0, (
        f"extinction phase produced {us_in_ext} US events (expected 0)"
    )

    # Also check we did at least see CS firings in extinction (so the
    # stream is genuinely running, not stuck).
    cs_in_ext = int(jnp.sum(obs[n_acq:, 0] > 0.5))
    assert cs_in_ext > 0


def test_partial_reinforcement_rate():
    """Empirical P(US | trial) is within +/- 0.05 of the configured rate.

    A "trial" is defined here as the rising edge of the CS indicator.
    Each trial may or may not be followed by a US ``cs_us_delay`` steps
    later, and the configured contingency controls the rate.
    """
    p = 0.5
    cs_us_delay = 5
    n_steps = 5000
    stream = partial_reinforcement_scenario(
        p=p,
        n_steps=n_steps,
        cs_us_delay=cs_us_delay,
        cs_duration=1,
        iti_min=5,
        iti_max=15,
        noise_std=0.0,
        distractor_prob=0.0,
    )
    state = stream.init(jr.key(2))
    _, obs, tgt = _collect(stream, state, n_steps)

    cs_ind = (obs[:, 0] > 0.5).astype(jnp.int32)
    # CS rising edges
    cs_diff = jnp.concatenate([jnp.array([0]), jnp.diff(cs_ind)])
    cs_onsets = jnp.where(cs_diff > 0)[0]
    # Drop trials whose US would land past the trace window.
    valid_onsets = cs_onsets[cs_onsets + cs_us_delay < n_steps]
    n_trials = int(valid_onsets.shape[0])
    assert n_trials > 100, f"expected many trials, got {n_trials}"

    us_indicator = tgt[:, 0] > 0.5
    us_after = us_indicator[valid_onsets + cs_us_delay]
    empirical_p = float(jnp.mean(us_after.astype(jnp.float32)))

    assert abs(empirical_p - p) < 0.05, (
        f"empirical P(US|CS)={empirical_p:.3f} not within 0.05 of {p}"
    )


def test_blocking_compound_phase_only_compound_cs():
    """In the compound phase only the (CS_0, CS_1) pair fires together.

    During ``compound_cs0_cs1``, every step where CS_0 = 1 must also
    have CS_1 = 1 and vice versa. There must be no "lone" CS events.
    """
    n_pre = 200
    n_compound = 600
    stream = blocking_scenario(
        n_pretrain=n_pre,
        n_compound=n_compound,
        cs_us_delay=5,
        cs_duration=1,
        iti_min=5,
        iti_max=10,
        noise_std=0.0,
        distractor_prob=0.0,
    )
    state = stream.init(jr.key(3))
    _, obs, _ = _collect(stream, state, n_pre + n_compound)

    cs0 = obs[n_pre:, 0] > 0.5
    cs1 = obs[n_pre:, 1] > 0.5

    # In compound phase: CS_0 == CS_1 at every step.
    chex.assert_trees_all_close(
        cs0.astype(jnp.float32),
        cs1.astype(jnp.float32),
    )
    # And there should be at least some CS firings.
    assert int(jnp.sum(cs0)) > 0


def test_pretrain_phase_only_cs0_fires():
    """During the blocking pretrain phase CS_1 should never fire."""
    stream = blocking_scenario(
        n_pretrain=500,
        n_compound=100,
        cs_us_delay=5,
        cs_duration=1,
        iti_min=5,
        iti_max=10,
        noise_std=0.0,
        distractor_prob=0.0,
    )
    state = stream.init(jr.key(4))
    _, obs, _ = _collect(stream, state, 500)

    cs1 = obs[:, 1] > 0.5
    assert int(jnp.sum(cs1)) == 0


def test_distractors_dont_predict_us():
    """A distractor should be uncorrelated with the US (correlation ~ 0)."""
    n_steps = 5000
    stream = acquisition_scenario(
        n_steps=n_steps,
        n_distractors=3,
        cs_us_delay=5,
        cs_duration=1,
        iti_min=5,
        iti_max=15,
        noise_std=0.0,
        distractor_prob=0.1,
    )
    state = stream.init(jr.key(5))
    _, obs, tgt = _collect(stream, state, n_steps)

    us = tgt[:, 0]
    # Distractors live at indices [n_cs : n_cs + n_distractors].
    n_cs = stream.n_cs
    for d_idx in range(stream.n_distractors):
        distractor = obs[:, n_cs + d_idx]
        # Pearson correlation; skip if a column is constant.
        std_d = float(jnp.std(distractor))
        std_u = float(jnp.std(us))
        if std_d < 1e-6 or std_u < 1e-6:
            continue
        corr = float(jnp.corrcoef(distractor, us)[0, 1])
        assert abs(corr) < 0.1, (
            f"distractor {d_idx} correlated with US: corr={corr:.3f}"
        )


# ---------------------------------------------------------------------------
# Static structure
# ---------------------------------------------------------------------------


def test_observation_shape():
    """``feature_dim`` matches ``n_cs + n_distractors`` and target is shape (1,)."""
    stream = acquisition_scenario(n_steps=200, n_distractors=4)
    assert stream.feature_dim == stream.n_cs + stream.n_distractors == 1 + 4

    state = stream.init(jr.key(6))
    ts, _ = stream.step(state, jnp.array(0))
    chex.assert_shape(ts.observation, (5,))
    chex.assert_shape(ts.target, (1,))
    chex.assert_tree_all_finite(ts.observation)
    chex.assert_tree_all_finite(ts.target)


def test_blocking_scenario_has_two_cs():
    """Blocking scenario must expose two CS features."""
    stream = blocking_scenario(n_pretrain=10, n_compound=10)
    assert stream.n_cs == 2
    assert stream.feature_dim == 2


def test_phases_in_order_and_named():
    """Reacquisition scenario phases are in the declared order and named."""
    stream = reacquisition_scenario(
        n_acquisition=100, n_extinction=100, n_reacquisition=100
    )
    assert tuple(p.name for p in stream.phases) == (
        "acquisition",
        "extinction",
        "reacquisition",
    )
    assert tuple(p.cs_us_contingency for p in stream.phases) == (1.0, 0.0, 1.0)


def test_extinction_phase_has_no_us_during_partial_reinforcement():
    """A direct contingency=0 phase emits zero US events even mid-stream."""
    stream = ClassicalConditioningStream(
        phases=(
            PavlovianPhase(
                name="acq", n_steps=300, cs_us_contingency=1.0, cs_active=(0,)
            ),
            PavlovianPhase(
                name="ext", n_steps=300, cs_us_contingency=0.0, cs_active=(0,)
            ),
        ),
        n_cs=1,
        cs_us_delay=5,
        iti_min=5,
        iti_max=10,
        noise_std=0.0,
        distractor_prob=0.0,
    )
    state = stream.init(jr.key(7))
    _, _, tgt = _collect(stream, state, 600)
    us = tgt[:, 0] > 0.5
    n_us_ext = int(jnp.sum(us[300:]))
    assert n_us_ext == 0


# ---------------------------------------------------------------------------
# JIT / scan
# ---------------------------------------------------------------------------


def test_jit_compatibility():
    """The stream's ``step`` must run under ``jax.lax.scan`` and ``jax.jit``."""
    stream = acquisition_scenario(
        n_steps=500, cs_us_delay=5, iti_min=3, iti_max=8, noise_std=0.05
    )

    def run(state, indices):
        def step_fn(carry, idx):
            ts, new_state = stream.step(carry, idx)
            return new_state, (ts.observation, ts.target)

        final_state, (obs, tgt) = jax.lax.scan(step_fn, state, indices)
        return final_state, obs, tgt

    jit_run = jax.jit(run)
    state = stream.init(jr.key(8))
    final_state, obs, tgt = jit_run(state, jnp.arange(200))

    chex.assert_shape(obs, (200, stream.feature_dim))
    chex.assert_shape(tgt, (200, 1))
    chex.assert_tree_all_finite(obs)
    chex.assert_tree_all_finite(tgt)
    # State remains a valid pytree of the right type.
    assert isinstance(final_state, PavlovianState)


def test_deterministic():
    """Identical keys must yield identical trajectories."""
    stream = acquisition_scenario(
        n_steps=500, cs_us_delay=5, iti_min=3, iti_max=12, noise_std=0.05
    )
    key = jr.key(9)

    state_a = stream.init(key)
    _, obs_a, tgt_a = _collect(stream, state_a, 200)

    state_b = stream.init(key)
    _, obs_b, tgt_b = _collect(stream, state_b, 200)

    chex.assert_trees_all_close(obs_a, obs_b)
    chex.assert_trees_all_close(tgt_a, tgt_b)


def test_different_keys_differ():
    """Different keys must yield different trajectories."""
    stream = acquisition_scenario(
        n_steps=500, cs_us_delay=5, iti_min=3, iti_max=12, noise_std=0.05
    )

    state_a = stream.init(jr.key(11))
    _, obs_a, _ = _collect(stream, state_a, 200)
    state_b = stream.init(jr.key(12))
    _, obs_b, _ = _collect(stream, state_b, 200)

    diff = float(jnp.mean(jnp.abs(obs_a - obs_b)))
    assert diff > 1e-6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_construct_rejects_empty_phases():
    """Constructor raises on empty phases."""
    try:
        ClassicalConditioningStream(phases=(), n_cs=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty phases")


def test_construct_rejects_bad_cs_index():
    """Constructor raises when a phase references an out-of-range CS index."""
    bad_phase = PavlovianPhase(
        name="bad", n_steps=10, cs_us_contingency=1.0, cs_active=(5,)
    )
    try:
        ClassicalConditioningStream(phases=(bad_phase,), n_cs=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid CS index")


@pytest.mark.parametrize("contingency", [-0.1, 1.1, float("nan")])
def test_construct_rejects_invalid_phase_contingency(contingency: float):
    """Every phase contingency must be a finite probability."""
    with pytest.raises(ValueError, match="cs_us_contingency"):
        PavlovianPhase(
            name="bad",
            n_steps=10,
            cs_us_contingency=contingency,
            cs_active=(0,),
        )


def test_partial_reinforcement_rejects_invalid_p():
    """``p`` outside [0, 1] is rejected."""
    for bad_p in (-0.1, 1.1):
        try:
            partial_reinforcement_scenario(p=bad_p)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for p={bad_p}")


def _valid_phase(**overrides: object) -> PavlovianPhase:
    payload = {
        "name": "acq",
        "n_steps": 10,
        "cs_us_contingency": 1.0,
        "cs_active": (0,),
    }
    payload.update(overrides)
    return PavlovianPhase(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        -1.0,
        1e100,
        10**400,
    ],
)
def test_construct_rejects_illegal_noise_std(value: object) -> None:
    """Noise must remain non-negative and finite in float32 execution."""
    with pytest.raises(ValueError, match="noise_std"):
        ClassicalConditioningStream(phases=(_valid_phase(),), noise_std=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        float(np.finfo(np.float32).max),
        float(np.nextafter(np.float32(0.0), np.float32(1.0))),
    ],
)
def test_construct_accepts_float32_noise_std_boundaries(value: float) -> None:
    """Finite float32 endpoints survive constructor canonicalization."""
    stream = ClassicalConditioningStream(phases=(_valid_phase(),), noise_std=value)
    assert stream._noise_std == value  # noqa: SLF001 - normalization is under test


def test_construct_canonicalizes_underflowing_noise_std_to_zero() -> None:
    """A positive host float below float32 range has exact zero-noise semantics."""
    stream = ClassicalConditioningStream(phases=(_valid_phase(),), noise_std=1e-50)
    assert stream._noise_std == 0.0  # noqa: SLF001 - normalization is under test


def test_construct_canonicalizes_noise_std_to_float32() -> None:
    """Stored noise matches the scalar used by the float32 trajectory."""
    stream = ClassicalConditioningStream(phases=(_valid_phase(),), noise_std=0.1)
    assert stream._noise_std == float(np.float32(0.1))  # noqa: SLF001


@requires_extended_longdouble
def test_construct_narrows_original_noise_real_once() -> None:
    midpoint_plus = (
        np.longdouble(1.0)
        + np.longdouble(2.0) ** -24
        + np.longdouble(2.0) ** -60
    )
    assert np.float32(midpoint_plus) != np.float32(float(midpoint_plus))

    stream = ClassicalConditioningStream(
        phases=(_valid_phase(),),
        noise_std=midpoint_plus,
    )
    assert stream._noise_std == float(np.float32(midpoint_plus))  # noqa: SLF001


@requires_extended_longdouble
def test_construct_rejects_negative_real_that_rounds_to_zero() -> None:
    below_zero = -np.nextafter(np.longdouble(0.0), np.longdouble(1.0))
    assert float(below_zero) == 0.0
    with pytest.raises(ValueError, match="noise_std"):
        ClassicalConditioningStream(phases=(_valid_phase(),), noise_std=below_zero)


@pytest.mark.parametrize(
    ("noise_std", "expected"),
    [
        (
            Fraction(1, 1) + Fraction(1, 2**24) - Fraction(1, 2**60),
            1.0,
        ),
        (Fraction(1, 1) + Fraction(1, 2**24), 1.0),
        (
            Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60),
            float(np.nextafter(np.float32(1.0), np.float32(2.0))),
        ),
    ],
    ids=("below", "tie-to-even", "above"),
)
def test_construct_rounds_fraction_noise_midpoints_once(
    noise_std: Fraction,
    expected: float,
) -> None:
    stream = ClassicalConditioningStream(
        phases=(_valid_phase(),),
        noise_std=noise_std,
    )
    assert stream._noise_std == expected  # noqa: SLF001


def test_construct_applies_exact_float32_overflow_midpoint() -> None:
    float32_max = (2**24 - 1) * 2**104
    overflow_midpoint = float32_max + 2**103

    stream = ClassicalConditioningStream(
        phases=(_valid_phase(),),
        noise_std=Fraction(overflow_midpoint - 1),
    )
    assert stream._noise_std == float(np.finfo(np.float32).max)  # noqa: SLF001
    with pytest.raises(ValueError, match="noise_std"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            noise_std=Fraction(overflow_midpoint),
        )


def test_construct_applies_exact_subnormal_midpoint_and_signed_zero() -> None:
    subnormal_midpoint = Fraction(1, 2**150)
    tie = ClassicalConditioningStream(
        phases=(_valid_phase(),),
        noise_std=subnormal_midpoint,
    )
    above = ClassicalConditioningStream(
        phases=(_valid_phase(),),
        noise_std=subnormal_midpoint + Fraction(1, 2**200),
    )
    negative_zero = ClassicalConditioningStream(
        phases=(_valid_phase(),),
        noise_std=-0.0,
    )

    assert tie._noise_std == 0.0  # noqa: SLF001
    assert above._noise_std == float(  # noqa: SLF001
        np.nextafter(np.float32(0.0), np.float32(1.0))
    )
    assert np.signbit(negative_zero._noise_std)  # noqa: SLF001


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_construct_rejects_illegal_distractor_prob(value: object) -> None:
    """Distractor probability must be a finite real in ``[0, 1]``."""
    with pytest.raises(ValueError, match="distractor_prob"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            distractor_prob=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, False])
def test_construct_rejects_bool_phase_contingency(value: bool) -> None:
    """A bool is not a scientific contingency, even though ``True == 1``."""
    with pytest.raises(ValueError, match="cs_us_contingency"):
        ClassicalConditioningStream(phases=(_valid_phase(cs_us_contingency=value),))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_partial_reinforcement_rejects_illegal_p(value: object) -> None:
    """Scenario ``p`` must be a finite real in ``[0, 1]``, not a bool or NaN."""
    with pytest.raises(ValueError, match=r"\bp\b"):
        partial_reinforcement_scenario(p=value)  # type: ignore[arg-type]


class _SpoofedReal:
    """Mimics ``float`` via ``__class__`` to defeat ``isinstance`` checks."""

    def __init__(self, value: float = 0.5) -> None:
        self._value = value

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return self._value

    def __lt__(self, other: object) -> bool:
        return self._value < other  # type: ignore[operator]

    def __le__(self, other: object) -> bool:
        return self._value <= other  # type: ignore[operator]


class _RaisingSpoofedReal:
    """A ``__class__`` spoof whose numeric hooks raise when actually used."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        raise RuntimeError("untrusted __float__ hook executed")

    def __lt__(self, other: object) -> bool:
        raise RuntimeError("untrusted __lt__ hook executed")

    def __le__(self, other: object) -> bool:
        raise RuntimeError("untrusted __le__ hook executed")

    def __repr__(self) -> str:
        raise RuntimeError("untrusted __repr__ hook executed")

    def __str__(self) -> str:
        raise RuntimeError("untrusted __str__ hook executed")


def test_construct_rejects_class_spoofed_noise_std() -> None:
    """A non-real whose ``__class__`` reports ``float`` must still be rejected."""
    with pytest.raises(ValueError, match="noise_std"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            noise_std=_SpoofedReal(0.1),  # type: ignore[arg-type]
        )


def test_construct_raising_class_spoofed_noise_std_stays_a_value_error() -> None:
    """A spoof with raising numeric hooks must not leak its raw exception."""
    with pytest.raises(ValueError, match="noise_std"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            noise_std=_RaisingSpoofedReal(),  # type: ignore[arg-type]
        )


def test_construct_rejects_class_spoofed_distractor_prob() -> None:
    """A non-real whose ``__class__`` reports ``float`` must still be rejected."""
    with pytest.raises(ValueError, match="distractor_prob"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            distractor_prob=_SpoofedReal(0.1),  # type: ignore[arg-type]
        )


def test_construct_raising_class_spoofed_distractor_prob_stays_a_value_error() -> None:
    """A spoof with raising numeric hooks must not leak its raw exception."""
    with pytest.raises(ValueError, match="distractor_prob"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            distractor_prob=_RaisingSpoofedReal(),  # type: ignore[arg-type]
        )


def test_construct_rejects_class_spoofed_phase_contingency() -> None:
    """A non-real whose ``__class__`` reports ``float`` must still be rejected."""
    with pytest.raises(ValueError, match="cs_us_contingency"):
        ClassicalConditioningStream(
            phases=(_valid_phase(cs_us_contingency=_SpoofedReal(0.1)),)
        )


def test_construct_raising_class_spoofed_phase_contingency_stays_a_value_error() -> None:
    """A spoof with raising numeric/repr hooks must not leak its raw exception."""
    with pytest.raises(ValueError, match="cs_us_contingency"):
        ClassicalConditioningStream(
            phases=(_valid_phase(cs_us_contingency=_RaisingSpoofedReal()),)
        )


def test_construct_accepts_zero_noise_and_zero_distractor_prob() -> None:
    stream = ClassicalConditioningStream(
        phases=(_valid_phase(),),
        noise_std=0.0,
        distractor_prob=0.0,
    )
    assert stream.feature_dim == 1


@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, 1.5, np.int64(5), -1, float("nan"), "5", None],
)
def test_iti_min_rejects_bool_nonintegral_and_out_of_domain(value: object) -> None:
    """``iti_min`` is a built-in int in ``[0, iti_max]``."""
    with pytest.raises(ValueError, match="iti_min"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            iti_min=value,  # type: ignore[arg-type]
            iti_max=20,
        )


@pytest.mark.parametrize("value", [True, False, 20.0, np.int64(20), -1])
def test_iti_max_rejects_bool_nonintegral_and_out_of_domain(value: object) -> None:
    with pytest.raises(ValueError, match="iti_max"):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            iti_min=0,
            iti_max=value,  # type: ignore[arg-type]
        )


def test_iti_min_accepts_zero_and_equal_max_endpoints() -> None:
    zero = ClassicalConditioningStream(phases=(_valid_phase(),), iti_min=0, iti_max=20)
    equal = ClassicalConditioningStream(phases=(_valid_phase(),), iti_min=0, iti_max=0)
    matched = ClassicalConditioningStream(phases=(_valid_phase(),), iti_min=5, iti_max=5)
    assert zero.feature_dim == 1
    assert equal.feature_dim == 1
    assert matched.feature_dim == 1


def test_iti_min_rejects_greater_than_iti_max() -> None:
    with pytest.raises(ValueError, match="iti_min <= iti_max"):
        ClassicalConditioningStream(phases=(_valid_phase(),), iti_min=6, iti_max=5)


@pytest.mark.parametrize(
    ("field", "minimum"),
    [
        ("n_cs", 1),
        ("n_distractors", 0),
        ("cs_us_delay", 1),
        ("cs_duration", 1),
    ],
)
@pytest.mark.parametrize("value", [True, False, 1.0, np.int64(1)])
def test_pavlovian_integer_fields_reject_bool_and_nonintegral(
    field: str, minimum: int, value: object
) -> None:
    del minimum
    with pytest.raises(ValueError, match=field):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            **{field: value},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, False, 1.0, np.int64(10), 0])
def test_phase_n_steps_requires_positive_builtin_int(value: object) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        ClassicalConditioningStream(phases=(_valid_phase(n_steps=value),))


@pytest.mark.parametrize("value", [True, False, 1.0, np.int64(-1)])
def test_phase_compound_index_requires_builtin_int(value: object) -> None:
    with pytest.raises(ValueError, match="compound_index"):
        ClassicalConditioningStream(
            phases=(_valid_phase(compound_index=value),),
            n_cs=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_distractors", 2**31),
        ("cs_us_delay", 2**31),
        ("cs_duration", 2**31),
        ("iti_max", 2**31 - 1),
    ],
)
def test_schedule_fields_reject_values_outside_jax_int32(
    field: str,
    value: int,
) -> None:
    """Accepted configuration must remain representable in JAX state."""
    kwargs: dict[str, object] = {field: value}
    with pytest.raises(ValueError, match=field):
        ClassicalConditioningStream(
            phases=(_valid_phase(),),
            **kwargs,  # type: ignore[arg-type]
        )


def test_phase_n_steps_rejects_values_outside_jax_int32() -> None:
    """Phase validation must run before materializing the int32 phase array."""
    with pytest.raises(ValueError, match="n_steps"):
        ClassicalConditioningStream(phases=(_valid_phase(n_steps=2**31),))


def test_schedule_fields_accept_jax_int32_upper_endpoints() -> None:
    stream = ClassicalConditioningStream(
        phases=(_valid_phase(n_steps=2**31 - 1),),
        cs_us_delay=2**31 - 1,
        cs_duration=2**31 - 1,
        iti_min=2**31 - 2,
        iti_max=2**31 - 2,
    )
    state = stream.init(jr.key(99))
    stream.step(state, jnp.array(0))
    assert int(state.iti_steps_remaining) == 2**31 - 2


def test_reacquisition_runs_three_phases():
    """Reacquisition scenario has three distinct contingency periods."""
    n_acq = 200
    n_ext = 200
    n_re = 200
    stream = reacquisition_scenario(
        n_acquisition=n_acq,
        n_extinction=n_ext,
        n_reacquisition=n_re,
        cs_us_delay=5,
        cs_duration=1,
        iti_min=5,
        iti_max=10,
        noise_std=0.0,
        distractor_prob=0.0,
    )
    state = stream.init(jr.key(13))
    _, _, tgt = _collect(stream, state, n_acq + n_ext + n_re)
    us = tgt[:, 0] > 0.5
    n_us_acq = int(jnp.sum(us[:n_acq]))
    n_us_ext = int(jnp.sum(us[n_acq : n_acq + n_ext]))
    n_us_re = int(jnp.sum(us[n_acq + n_ext :]))
    assert n_us_acq > 0
    assert n_us_ext == 0
    assert n_us_re > 0
