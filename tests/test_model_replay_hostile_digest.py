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

    def __ne__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile ne executed")


class _HostileDict(dict[str, object]):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile dict eq executed")

    def __ne__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile dict ne executed")


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
            with pytest.raises(ValueError, match="config digest does not match"):
                load_model_replay_rehearsal_checkpoint(str(path))
        assert _HostileStr.calls == 0


@pytest.mark.parametrize("field", ["schema", "mechanism_status"])
def test_model_replay_rejects_hostile_identity_metadata(field: str, tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core import model_replay_rehearsal as replay

    metadata: dict[str, object] = {
        "schema": replay.MODEL_REPLAY_REHEARSAL_SCHEMA,
        "mechanism_status": replay.MECHANISM_STATUS,
        "accepted_scientific_evidence": False,
    }
    metadata[field] = _HostileStr(str(metadata[field]))
    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    _HostileStr.calls = 0
    with patch.object(replay, "load_checkpoint_metadata", return_value=metadata):
        with pytest.raises(ValueError):
            replay.load_model_replay_rehearsal_checkpoint(path)
    assert _HostileStr.calls == 0


def test_model_replay_rejects_hostile_resource_metadata(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import Mock, patch

    from alberta_framework.core import model_replay_rehearsal as replay

    config: dict[str, object] = {"composer": "test"}
    composer = Mock()
    composer.to_config.return_value = config
    composer.init.return_value = object()
    composer.resource_budget.return_value.to_config.return_value = {"bytes": 1}
    metadata = {
        "schema": replay.MODEL_REPLAY_REHEARSAL_SCHEMA,
        "mechanism_status": replay.MECHANISM_STATUS,
        "accepted_scientific_evidence": False,
        "composer_config": config,
        "config_sha256": replay._config_digest(config),
        "resource_budget": _HostileDict({"bytes": 1}),
    }
    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    _HostileDict.calls = 0
    with (
        patch.object(replay, "load_checkpoint_metadata", return_value=metadata),
        patch.object(replay.ModelReplayRehearsal, "from_config", return_value=composer),
    ):
        with pytest.raises(ValueError, match="resource budget"):
            replay.load_model_replay_rehearsal_checkpoint(path)
    assert _HostileDict.calls == 0


def test_model_replay_rejects_hostile_config_mapping(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core import model_replay_rehearsal as replay

    metadata = {
        "schema": replay.MODEL_REPLAY_REHEARSAL_SCHEMA,
        "mechanism_status": replay.MECHANISM_STATUS,
        "accepted_scientific_evidence": False,
        "composer_config": _HostileDict({"composer": "test"}),
    }
    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    _HostileDict.calls = 0
    with patch.object(replay, "load_checkpoint_metadata", return_value=metadata):
        with pytest.raises(ValueError, match="missing composer_config"):
            replay.load_model_replay_rehearsal_checkpoint(path)
    assert _HostileDict.calls == 0


def test_model_replay_rejects_undeclared_checkpoint_metadata(tmp_path: object) -> None:
    """A development checkpoint cannot carry an unvalidated promotion claim."""
    from pathlib import Path
    from unittest.mock import Mock, patch

    from alberta_framework.core import model_replay_rehearsal as replay

    config: dict[str, object] = {"composer": "test"}
    budget = {"bytes": 1}
    state = object()
    composer = Mock()
    composer.to_config.return_value = config
    composer.init.return_value = state
    composer.resource_budget.return_value.to_config.return_value = budget
    metadata = {
        "schema": replay.MODEL_REPLAY_REHEARSAL_SCHEMA,
        "mechanism_status": replay.MECHANISM_STATUS,
        "accepted_scientific_evidence": False,
        "composer_config": config,
        "config_sha256": replay._config_digest(config),
        "resource_budget": budget,
        "scientific_promotion_allowed": True,
    }
    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    with (
        patch.object(replay, "load_checkpoint_metadata", return_value=metadata),
        patch.object(replay.ModelReplayRehearsal, "from_config", return_value=composer),
        patch.object(replay, "load_checkpoint", return_value=(state, metadata)),
    ):
        with pytest.raises(ValueError, match="metadata fields"):
            replay.load_model_replay_rehearsal_checkpoint(path)
