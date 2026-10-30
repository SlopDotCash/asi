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


@pytest.mark.parametrize("metadata_only", [False, True])
def test_symlinked_metadata_is_not_a_self_contained_checkpoint(
    tmp_path: Path,
    metadata_only: bool,
) -> None:
    learner = LinearLearner()
    state = learner.init(3)
    checkpoint = tmp_path / "linked-external-metadata"
    save_checkpoint(state, checkpoint, metadata={"epoch": 1})

    metadata_path = _on_disk_metadata_path(checkpoint)
    external_metadata = tmp_path / "external.json"
    external_metadata.write_bytes(metadata_path.read_bytes())
    metadata_path.unlink()
    metadata_path.symlink_to(external_metadata)

    with pytest.raises(ValueError, match="regular file"):
        if metadata_only:
            load_checkpoint_metadata(checkpoint)
        else:
            load_checkpoint(state, checkpoint)


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
    """Symlinked metadata must be rejected before any JSON parser runs."""
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

    with pytest.raises(ValueError, match="regular file"):
        load_checkpoint_metadata(ckpt)


def test_symlinked_metadata_deep_nest_rejects_load_checkpoint(tmp_path: Path) -> None:
    """Full restore must reject symlinked metadata before parsing it."""
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

    with pytest.raises(ValueError, match="regular file"):
        load_checkpoint(state, ckpt)
