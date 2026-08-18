"""Hostile input and boundary validation for RTU PPO RNG isolation dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_rtu_ppo_rng_isolation import (
    IsolatedRTUPPOSource,
    RTUPPORngIsolationError,
    SourceReplacement,
)


def test_source_replacement_valid_construction() -> None:
    rep = SourceReplacement(
        replacement_id="rep_1",
        before=b"before",
        after=b"after",
    )
    assert rep.replacement_id == "rep_1"


def test_source_replacement_rejects_empty_fields() -> None:
    with pytest.raises(RTUPPORngIsolationError, match="replacement_id must be a non-empty string"):
        SourceReplacement(
            replacement_id="",
            before=b"before",
            after=b"after",
        )

    with pytest.raises(RTUPPORngIsolationError, match="before must be non-empty bytes"):
        SourceReplacement(
            replacement_id="rep_1",
            before=b"",
            after=b"after",
        )

    with pytest.raises(RTUPPORngIsolationError, match="after must be non-empty bytes"):
        SourceReplacement(
            replacement_id="rep_1",
            before=b"before",
            after=b"",
        )


def test_isolated_rtu_ppo_source_rejects_invalid_inputs() -> None:
    with pytest.raises(RTUPPORngIsolationError, match="source must be non-empty bytes"):
        IsolatedRTUPPOSource(
            source=b"",
            upstream_source_sha256="a" * 64,
            source_sha256="b" * 64,
            patch=b"patch",
            patch_sha256="c" * 64,
            descriptor={},
            descriptor_sha256="d" * 64,
        )

    with pytest.raises(
        RTUPPORngIsolationError, match="upstream_source_sha256 must be a 64-character lowercase"
    ):
        IsolatedRTUPPOSource(
            source=b"source",
            upstream_source_sha256="invalid",
            source_sha256="b" * 64,
            patch=b"patch",
            patch_sha256="c" * 64,
            descriptor={},
            descriptor_sha256="d" * 64,
        )
