"""Reject oversized policy-archive latents before per-element walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.policy_archive import (
    _MAX_LATENT_WIDTH,
    PolicyEntry,
)


def test_policy_archive_latent_width_cap_constant() -> None:
    assert _MAX_LATENT_WIDTH == 4096


def test_policy_archive_accepts_max_latent_width() -> None:
    latent = tuple(0.0 for _ in range(_MAX_LATENT_WIDTH))
    entry = PolicyEntry(identity="a", policy_bytes=b"abcd", latent=latent, score=1.0)
    assert len(entry.latent) == _MAX_LATENT_WIDTH


def test_policy_archive_rejects_oversized_latent_before_value_walk() -> None:
    latent = tuple(0.0 for _ in range(_MAX_LATENT_WIDTH + 1))
    with pytest.raises(ValueError, match="latent length"):
        PolicyEntry(identity="a", policy_bytes=b"abcd", latent=latent, score=1.0)
