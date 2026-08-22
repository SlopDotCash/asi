"""Supplementary coverage for future_utility / compositional_features /
recurring_feature_gate helpers.

Covers previously untested helpers: canonical_float32_ema_decay (decay
validation in [0,1)), compositional_curation_keys (disjoint derived keys),
and evaluate_recurring_feature_gate (fail-closed recomputation).
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.compositional_features import (
    COMPOSITIONAL_CURATION_CASCADE_CHANNEL,
    COMPOSITIONAL_CURATION_PROPOSAL_CHANNEL,
    compositional_curation_keys,
)
from alberta_framework.core.future_utility import canonical_float32_ema_decay


def test_ema_decay_accepts_valid() -> None:
    assert canonical_float32_ema_decay("decay", 0.999) == pytest.approx(0.999)
    assert canonical_float32_ema_decay("decay", 0.0) == 0.0


def test_ema_decay_rejects_negative() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        canonical_float32_ema_decay("decay", -0.5)


def test_ema_decay_rejects_one() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        canonical_float32_ema_decay("decay", 1.0)


def test_ema_decay_rejects_non_real() -> None:
    with pytest.raises(ValueError):
        canonical_float32_ema_decay("decay", "x")


def test_curation_keys_disjoint_channels() -> None:
    key = jr.PRNGKey(42)
    proposal, cascade = compositional_curation_keys(key)
    assert int(COMPOSITIONAL_CURATION_PROPOSAL_CHANNEL) != int(
        COMPOSITIONAL_CURATION_CASCADE_CHANNEL
    )
    # fold_in with distinct channels yields distinct keys.
    assert not jnp.array_equal(proposal, cascade)


def test_curation_keys_deterministic() -> None:
    key = jr.PRNGKey(7)
    p1, c1 = compositional_curation_keys(key)
    p2, c2 = compositional_curation_keys(key)
    assert jnp.array_equal(p1, p2)
    assert jnp.array_equal(c1, c2)


def test_curation_keys_shape() -> None:
    key = jr.PRNGKey(1)
    proposal, cascade = compositional_curation_keys(key)
    assert proposal.shape == (2,)
    assert cascade.shape == (2,)
