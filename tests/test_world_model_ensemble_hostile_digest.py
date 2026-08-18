"""Hostile string validation for world-model ensemble digest."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")


def test_ensemble_digest_rejects_hostile_before_eq() -> None:
    from alberta_framework.core.world_model_ensemble import _ensemble_config_digest

    config = {"ensemble": "test"}
    digest = _ensemble_config_digest(config)
    hostile = _HostileStr(digest)
    _HostileStr.calls = 0
    assert (type(hostile) is not str or hostile != digest) is True
    assert _HostileStr.calls == 0


def test_ensemble_checkpoint_rejects_hostile_digest() -> None:
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core.world_model_ensemble import (
        WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
        _ensemble_config_digest,
        load_world_model_ensemble_checkpoint,
    )

    config = {"ensemble": "test"}
    expected = _ensemble_config_digest(config)
    hostile = _HostileStr(expected)
    _HostileStr.calls = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chkpt"
        path.write_bytes(b"dummy")
        fake_meta = {
            "schema": WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
            "ensemble_config": config,
            "config_sha256": hostile,  # type: ignore[dict-item]
        }
        with patch(
            "alberta_framework.core.world_model_ensemble.load_checkpoint_metadata",
            return_value=fake_meta,
        ):
            with pytest.raises(ValueError, match="config digest does not match"):
                load_world_model_ensemble_checkpoint(str(path))
        assert _HostileStr.calls == 0
