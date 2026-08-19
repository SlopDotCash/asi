"""Hostile string gate for forager campaign destination_name before in."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_campaign import (
    ForagerMatchedCampaignError,
    _link_anonymous_no_replace,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __contains__(self, item: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile contains must not run")


def test_destination_rejects_hostile_before_in(tmp_path) -> None:
    hostile = _HostileStr("myfile")
    _HostileStr.calls = 0
    with pytest.raises(ForagerMatchedCampaignError, match="anonymous publication name is unsafe"):
        _link_anonymous_no_replace(0, tmp_path, hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_destination_rejects_non_string(tmp_path) -> None:
    with pytest.raises(ForagerMatchedCampaignError, match="anonymous publication name is unsafe"):
        _link_anonymous_no_replace(0, tmp_path, 123)  # type: ignore[arg-type]


def test_destination_rejects_dot_benign(tmp_path) -> None:
    with pytest.raises(ForagerMatchedCampaignError, match="anonymous publication name is unsafe"):
        _link_anonymous_no_replace(0, tmp_path, ".")


def test_destination_rejects_slash(tmp_path) -> None:
    with pytest.raises(ForagerMatchedCampaignError, match="anonymous publication name is unsafe"):
        _link_anonymous_no_replace(0, tmp_path, "a/b")


def test_benign_valid_passes_gate(tmp_path) -> None:
    # valid name passes the name gate; later linkat error is allowed
    try:
        _link_anonymous_no_replace(0, tmp_path, "valid_name")
    except ForagerMatchedCampaignError as exc:
        assert "anonymous publication name is unsafe" not in str(exc)
    except Exception:
        pass
