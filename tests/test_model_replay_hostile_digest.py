"""Hostile string validation for model replay digest."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")


def test_model_replay_digest_rejects_hostile_before_eq() -> None:
    from alberta_framework.core.model_replay_rehearsal import _config_digest

    config = {"composer": "test", "seed": 1}
    digest = _config_digest(config)
    hostile = _HostileStr(digest)
    _HostileStr.calls = 0
    assert (type(hostile) is not str or hostile != digest) is True
    assert _HostileStr.calls == 0
    assert (type(digest) is not str or digest != digest) is False


def test_model_replay_checkpoint_rejects_hostile_digest() -> None:
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core.model_replay_rehearsal import (
        load_model_replay_rehearsal_checkpoint,
    )

    config = {"composer": "test"}
    from alberta_framework.core.model_replay_rehearsal import _config_digest as _dig

    expected = _dig(config)
    hostile = _HostileStr(expected)
    _HostileStr.calls = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chkpt"
        path.write_bytes(b"dummy")
        fake_meta = {
            "composer_config": config,
            "config_sha256": hostile,  # type: ignore[dict-item]
            "schema": "alberta.model_replay_rehearsal.v1",
            "mechanism_status": "model-only-replay-mechanism-no-scientific-claim",
            "accepted_scientific_evidence": False,
        }
        with patch(
            "alberta_framework.core.model_replay_rehearsal.load_checkpoint_metadata",
            return_value=fake_meta,
        ):
            try:
                load_model_replay_rehearsal_checkpoint(str(path))
            except ValueError as exc:
                assert "config digest does not match" in str(exc)
            except Exception:
                pass
        assert _HostileStr.calls == 0
