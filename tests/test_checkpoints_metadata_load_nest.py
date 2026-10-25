"""Reject deep on-disk checkpoint metadata before Orbax json.loads.

Origin ``load_checkpoint_metadata`` delegates to Orbax ``JsonRestore``, which
``json.loads`` the path-fed ``metadata/metadata`` file with no nesting
preflight. A 16_000-deep object nest RecursionError's the C decoder on
origin/main. Overlay fail-closes at the shared 32-deep JSON ceiling before
loads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from alberta_framework import LinearLearner
from alberta_framework.core.checkpoints import (
    _JSON_MAX_DEPTH,
    _on_disk_metadata_path,
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

pytestmark = pytest.mark.unit


def _nested_json_bytes(depth: int) -> bytes:
    return ("{\"k\":" * depth + "true" + "}" * depth).encode("ascii")


def _write_hostile_metadata(path: Path, depth: int) -> None:
    meta_path = _on_disk_metadata_path(path)
    meta_path.write_bytes(_nested_json_bytes(depth))


def _write_raw_metadata(path: Path, payload: bytes) -> None:
    _on_disk_metadata_path(path).write_bytes(payload)


def test_frozen_checkpoint_json_nest_bound() -> None:
    assert _JSON_MAX_DEPTH == 32


def test_last_fit_checkpoint_still_roundtrips(tmp_path: Path) -> None:
    learner = LinearLearner()
    state = learner.init(3)
    path = tmp_path / "honest"
    save_checkpoint(state, path, metadata={"epoch": 1})
    loaded = load_checkpoint_metadata(path)
    assert loaded["epoch"] == 1
    restored, meta = load_checkpoint(state, path)
    assert meta["epoch"] == 1
    assert restored.weights.shape == state.weights.shape


def test_last_fit_on_disk_nest_still_loads(tmp_path: Path) -> None:
    learner = LinearLearner()
    state = learner.init(3)
    path = tmp_path / "last-fit"
    save_checkpoint(state, path, metadata={"epoch": 1})
    _write_hostile_metadata(path, _JSON_MAX_DEPTH)
    loaded = load_checkpoint_metadata(path)
    assert loaded["k"] is True or isinstance(loaded, dict)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"epoch":1,"epoch":2,"_format_version":2}',
        b'{"learner_config":{"step_size":0.1,"step_size":0.2},"_format_version":2}',
    ],
    ids=["top-level", "nested"],
)
@pytest.mark.parametrize("metadata_only", [True, False], ids=["metadata-only", "full"])
def test_on_disk_metadata_rejects_duplicate_object_keys(
    tmp_path: Path,
    payload: bytes,
    *,
    metadata_only: bool,
) -> None:
    learner = LinearLearner()
    state = learner.init(3)
    path = tmp_path / "duplicate"
    save_checkpoint(state, path, metadata={"epoch": 1})
    _write_raw_metadata(path, payload)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        if metadata_only:
            load_checkpoint_metadata(path)
        else:
            load_checkpoint(state, path)


def test_origin_recursion_class_rejects_before_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    learner = LinearLearner()
    state = learner.init(3)
    path = tmp_path / "hostile"
    save_checkpoint(state, path, metadata={"epoch": 1})
    _write_hostile_metadata(path, 16_000)

    def fail_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("json.loads ran before the on-disk metadata nest gate")

    monkeypatch.setattr(json, "loads", fail_loads)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting limit"):
        load_checkpoint_metadata(path)
    # Anti-runaway guard only; the ordering above is pinned structurally.
    # The 5 s budget absorbs the shared-CI-runner contention class observed
    # in run 36962287499 (0.60 s against the runtime-profile nest gate).
    assert time.perf_counter() - started < 5.0


def test_symlinked_metadata_deep_nest_rejects_before_loads(tmp_path: Path) -> None:
    """Symlinked metadata/metadata must not bypass the on-disk nest preflight."""
    learner = LinearLearner()
    state = learner.init(3)
    ckpt = tmp_path / "linked"
    save_checkpoint(state, ckpt, metadata={"epoch": 1})

    deep = tmp_path / "deep.json"
    deep.write_bytes(_nested_json_bytes(16_000))

    meta_path = _on_disk_metadata_path(ckpt)
    meta_path.unlink()
    meta_path.symlink_to(deep)
    assert meta_path.is_symlink()

    with pytest.raises(ValueError, match="nesting limit"):
        load_checkpoint_metadata(ckpt)


def test_symlinked_metadata_deep_nest_rejects_load_checkpoint(tmp_path: Path) -> None:
    """Symlinked metadata bypass must not leak into load_checkpoint either."""
    learner = LinearLearner()
    state = learner.init(3)
    ckpt = tmp_path / "linked-full"
    save_checkpoint(state, ckpt, metadata={"epoch": 1})

    deep = tmp_path / "deep.json"
    deep.write_bytes(_nested_json_bytes(16_000))

    meta_path = _on_disk_metadata_path(ckpt)
    meta_path.unlink()
    meta_path.symlink_to(deep)
    assert meta_path.is_symlink()

    with pytest.raises(ValueError, match="nesting limit"):
        load_checkpoint(state, ckpt)
