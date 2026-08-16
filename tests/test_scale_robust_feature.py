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

import pytest

from alberta_framework._seed_validation import JAX_KEY_SEED_MAX
from alberta_framework.evaluation.scale_robust_feature import (
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    run_scale_robust_feature_evaluation,
)


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
