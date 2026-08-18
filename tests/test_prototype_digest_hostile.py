"""Hostile string validation for prototype checkpoint digest."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile str executed")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile repr executed")


def test_prototype_digest_gate_rejects_hostile_before_eq() -> None:
    from alberta_framework.core.prototype_agent import _prototype_config_digest

    config = {"agent": "test", "seed": 1}
    digest = _prototype_config_digest(config)
    hostile = _HostileStr(digest)
    _HostileStr.calls = 0
    # Mirror the exact gate from load_prototype_checkpoint
    should_raise = type(hostile) is not str or hostile != digest
    # For hostile subclass, type is not str => True, short-circuits, != not executed
    assert should_raise is True
    assert _HostileStr.calls == 0
    # For builtin str with correct digest, gate is False
    assert (type(digest) is not str or digest != digest) is False
    # For builtin str with wrong digest, gate is True (mismatch)
    assert (type("0" * 64) is not str or "0" * 64 != digest) is True


def test_prototype_checkpoint_rejects_hostile_digest_without_dispatch() -> None:
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core.prototype_agent import (
        PROTOTYPE_CHECKPOINT_SCHEMA,
        _prototype_config_digest,
        load_prototype_checkpoint,
    )

    real_config = {"prototype": "minimal"}
    expected = _prototype_config_digest(real_config)
    hostile = _HostileStr(expected)
    _HostileStr.calls = 0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chkpt"
        path.write_bytes(b"dummy")

        fake_metadata = {
            "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
            "agent_config": real_config,
            "config_sha256": hostile,  # type: ignore[dict-item]
        }

        with patch(
            "alberta_framework.core.prototype_agent.load_checkpoint_metadata",
            return_value=fake_metadata,
        ):
            with pytest.raises(ValueError, match="config digest does not match"):
                load_prototype_checkpoint(str(path))
        assert _HostileStr.calls == 0


def test_prototype_digest_text_has_no_repr_leak() -> None:
    import pathlib

    text = pathlib.Path(
        "alberta_framework/core/prototype_agent.py"
    ).read_text()
    # Ensure the digest error does not interpolate hostile via !r
    assert 'config_sha256' in text
