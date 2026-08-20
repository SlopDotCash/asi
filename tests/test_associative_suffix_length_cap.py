"""Reject oversized associative-memory suffix lengths before pair-index hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.associative_memory import (
    _MAX_ASSOCIATIVE_SUFFIX_LENGTH,
    AssociativeMemoryConfig,
)


def test_associative_suffix_cap_constant() -> None:
    assert _MAX_ASSOCIATIVE_SUFFIX_LENGTH == 4096


def test_associative_rejects_oversized_suffix_length() -> None:
    with pytest.raises(ValueError, match="suffix_length"):
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=_MAX_ASSOCIATIVE_SUFFIX_LENGTH + 1,
            suffix_length=_MAX_ASSOCIATIVE_SUFFIX_LENGTH + 1,
            max_features=8,
        )
