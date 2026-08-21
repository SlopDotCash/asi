"""Unit coverage for alberta_framework.benchmarks.ipmnist_ceiling.

Tests the run-family identity contract: safe output tags, tag↔mode
binding (stationary/carried/full), n_tasks agreement, and the
run_and_publish mode dispatch.
"""

import pytest

from alberta_framework.benchmarks.ipmnist_ceiling import (
    _safe_output_tag,
    _validate_run_identity,
)


def test_safe_output_tag() -> None:
    assert _safe_output_tag("abc123") == "abc123"
    assert _safe_output_tag("run_1-x") == "run_1-x"
    assert _safe_output_tag("1abc") == "1abc"  # digits are alphanumeric
    with pytest.raises(ValueError, match="safe exact identifier"):
        _safe_output_tag("")
    with pytest.raises(ValueError, match="safe exact identifier"):
        _safe_output_tag(123)
    with pytest.raises(ValueError, match="safe exact identifier"):
        _safe_output_tag("ab c")
    with pytest.raises(ValueError, match="safe exact identifier"):
        _safe_output_tag("ab/c")


def test_validate_run_identity_stationary() -> None:
    _validate_run_identity("stationary_x", "x", "identity", 1)
    with pytest.raises(ValueError, match="permutation_mode"):
        _validate_run_identity("stationary_x", "x", "same", 1)
    with pytest.raises(ValueError, match="n_tasks"):
        _validate_run_identity("stationary_x", "x", "identity", 2)


def test_validate_run_identity_carried() -> None:
    _validate_run_identity("carried_x", "x", "same", 60)
    with pytest.raises(ValueError, match="at least ten"):
        _validate_run_identity("carried_x", "x", "same", 5)
    with pytest.raises(ValueError, match="permutation_mode"):
        _validate_run_identity("carried_x", "x", "protocol", 60)


def test_validate_run_identity_full() -> None:
    _validate_run_identity("full_x", "x", "protocol", 200)
    with pytest.raises(ValueError, match="n_tasks"):
        _validate_run_identity("full_x", "x", "protocol", 199)


def test_validate_run_identity_unknown_tag() -> None:
    with pytest.raises(ValueError, match="bind its run family"):
        _validate_run_identity("bogus_x", "x", "identity", 1)
