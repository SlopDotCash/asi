"""Checkpoint state payloads must remain self-contained regular files."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint

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
