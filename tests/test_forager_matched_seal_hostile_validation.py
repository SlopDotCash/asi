"""Hostile input and boundary validation for Forager matched seal bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matched_seal import (
    ContentVerifiedSealBundle,
    ForagerMatchedSealError,
)


def test_content_verified_seal_bundle_rejects_invalid_root() -> None:
    with pytest.raises(ForagerMatchedSealError, match="output_root must be a Path"):
        ContentVerifiedSealBundle(
            output_root="invalid/path",  # type: ignore[arg-type]
            manifest={},
            open_protocol=None,  # type: ignore[arg-type]
            open_score_evidence=None,  # type: ignore[arg-type]
            open_verification_request=None,  # type: ignore[arg-type]
            recorded_bindings_cache={},
            selection_result=None,  # type: ignore[arg-type]
            selection_report={},
            sealed_protocol=None,  # type: ignore[arg-type]
            sealed_transition={},
            sealed_transition_sha256="a" * 64,
        )


def test_content_verified_seal_bundle_rejects_invalid_protocol() -> None:
    with pytest.raises(
        ForagerMatchedSealError, match="open_protocol must be a ForagerMatchedProtocol"
    ):
        ContentVerifiedSealBundle(
            output_root=Path("output/seal"),
            manifest={},
            open_protocol=None,  # type: ignore[arg-type]
            open_score_evidence=None,  # type: ignore[arg-type]
            open_verification_request=None,  # type: ignore[arg-type]
            recorded_bindings_cache={},
            selection_result=None,  # type: ignore[arg-type]
            selection_report={},
            sealed_protocol=None,  # type: ignore[arg-type]
            sealed_transition={},
            sealed_transition_sha256="a" * 64,
        )


def test_open_directory_validation() -> None:
    from alberta_framework.benchmarks.forager_matched_seal import _OpenDirectory

    d = _OpenDirectory(path=Path("/tmp"), descriptor=3, inode_identity=(1, 2, 3))
    assert d.descriptor == 3

    with pytest.raises(ForagerMatchedSealError, match="path must be a Path"):
        _OpenDirectory(path="/tmp", descriptor=3, inode_identity=(1, 2, 3))  # type: ignore[arg-type]

    with pytest.raises(ForagerMatchedSealError, match="descriptor must be a non-negative int"):
        _OpenDirectory(path=Path("/tmp"), descriptor=-1, inode_identity=(1, 2, 3))

    with pytest.raises(ForagerMatchedSealError, match="inode_identity must be a 3-element tuple"):
        _OpenDirectory(path=Path("/tmp"), descriptor=3, inode_identity=(1, 2))  # type: ignore[arg-type]
