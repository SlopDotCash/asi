"""Unnamed-file publication retains no-replace semantics without capabilities."""
from __future__ import annotations

import ctypes
import errno
from types import SimpleNamespace

import pytest

from alberta_framework.benchmarks import reference_life_scorecard as scorecard


@pytest.mark.parametrize("first_error", [errno.EPERM, errno.ENOENT])
def test_link_fallback_uses_same_descriptor_and_no_replace(monkeypatch, first_error):
    calls = []

    def linkat(*args):
        calls.append(args)
        ctypes.set_errno(first_error)
        return -1 if len(calls) == 1 else 0

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(linkat=linkat))
    scorecard._link_unnamed_file(17, 23, "result.json")
    assert calls == [
        (17, b"", 23, b"result.json", 0x1000),
        (-100, b"/proc/self/fd/17", 23, b"result.json", 0x400),
    ]


@pytest.mark.parametrize(
    "errors,expected_calls",
    [([errno.EEXIST], 1), ([errno.EPERM, errno.EEXIST], 2)],
)
def test_existing_destination_is_never_retried_or_replaced(monkeypatch, errors, expected_calls):
    calls = []

    def linkat(*args):
        calls.append(args)
        ctypes.set_errno(errors[len(calls) - 1])
        return -1

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(linkat=linkat))
    with pytest.raises(FileExistsError):
        scorecard._link_unnamed_file(17, 23, "result.json")
    assert len(calls) == expected_calls


@pytest.mark.parametrize("errors", [[errno.EIO], [errno.EPERM, errno.ENOENT]])
def test_publication_errors_remain_visible(monkeypatch, errors):
    calls = []

    def linkat(*args):
        calls.append(args)
        ctypes.set_errno(errors[len(calls) - 1])
        return -1

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(linkat=linkat))
    with pytest.raises(OSError) as caught:
        scorecard._link_unnamed_file(17, 23, "result.json")
    assert caught.value.errno == errors[-1]
    assert len(calls) == len(errors)
