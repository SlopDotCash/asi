"""Checkpoint state payloads must remain self-contained regular files."""

from __future__ import annotations

import shutil
from pathlib import Path

import jax.numpy as jnp
import pytest

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

pytestmark = pytest.mark.unit


def test_load_rejects_symlinked_state_payload(tmp_path: Path) -> None:
    state = {"weights": jnp.arange(4, dtype=jnp.float32)}
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(state, checkpoint)

    state_blob = next((checkpoint / "state" / "d").iterdir())
    external_blob = tmp_path / "external-state-blob"
    external_blob.write_bytes(state_blob.read_bytes())
    state_blob.unlink()
    state_blob.symlink_to(external_blob)

    with pytest.raises(ValueError, match="regular file"):
        load_checkpoint(
            {"weights": jnp.zeros(4, dtype=jnp.float32)},
            checkpoint,
        )


def test_load_rejects_symlinked_metadata_directory(tmp_path: Path) -> None:
    state = {"weights": jnp.arange(4, dtype=jnp.float32)}
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(state, checkpoint, metadata={"epoch": 1})

    external_meta_dir = tmp_path / "external-meta-dir"
    external_meta_dir.mkdir()
    (external_meta_dir / "metadata").write_text(
        '{"_format_version": 2, "epoch": 99}', encoding="utf-8"
    )
    shutil.rmtree(checkpoint / "metadata")
    (checkpoint / "metadata").symlink_to(external_meta_dir)

    with pytest.raises(ValueError, match="regular directory"):
        load_checkpoint_metadata(checkpoint)

    with pytest.raises(ValueError, match="regular directory"):
        load_checkpoint(
            {"weights": jnp.zeros(4, dtype=jnp.float32)},
            checkpoint,
        )


def test_load_rejects_symlinked_metadata_file(tmp_path: Path) -> None:
    state = {"weights": jnp.arange(4, dtype=jnp.float32)}
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(state, checkpoint, metadata={"epoch": 1})

    meta_file = checkpoint / "metadata" / "metadata"
    external_meta = tmp_path / "external-metadata.json"
    external_meta.write_text('{"_format_version": 2, "epoch": 77}', encoding="utf-8")
    meta_file.unlink()
    meta_file.symlink_to(external_meta)

    with pytest.raises(ValueError, match="regular file"):
        load_checkpoint_metadata(checkpoint)

    with pytest.raises(ValueError, match="regular file"):
        load_checkpoint(
            {"weights": jnp.zeros(4, dtype=jnp.float32)},
            checkpoint,
        )


def test_load_rejects_non_directory_metadata_entry(tmp_path: Path) -> None:
    state = {"weights": jnp.arange(4, dtype=jnp.float32)}
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(state, checkpoint, metadata={"epoch": 1})

    shutil.rmtree(checkpoint / "metadata")
    (checkpoint / "metadata").write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="regular directory"):
        load_checkpoint_metadata(checkpoint)

    with pytest.raises(ValueError, match="regular directory"):
        load_checkpoint(
            {"weights": jnp.zeros(4, dtype=jnp.float32)},
            checkpoint,
        )


def test_load_round_trips_a_self_contained_checkpoint(tmp_path: Path) -> None:
    state = {"weights": jnp.arange(4, dtype=jnp.float32)}
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(state, checkpoint)

    restored, _ = load_checkpoint({"weights": jnp.zeros(4, dtype=jnp.float32)}, checkpoint)

    assert restored["weights"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_load_reports_missing_state_as_file_not_found(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint({"weights": jnp.arange(4, dtype=jnp.float32)}, checkpoint)
    shutil.rmtree(checkpoint / "state")

    with pytest.raises(FileNotFoundError, match="checkpoint state not found"):
        load_checkpoint({"weights": jnp.zeros(4, dtype=jnp.float32)}, checkpoint)
