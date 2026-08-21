"""Fail-closed NPZ preflight for maintained IPMNIST campaign ceiling runs."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_campaign_tools import build_ceiling_summary

pytestmark = pytest.mark.unit


def _valid_metadata(*, n_tasks: int = 1) -> str:
    payload: dict[str, object] = {
        "schema": "asi.ipmnist_ceiling.run.v2",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "tag": "stationary_sigma0_ndecay099",
        "spec_name": "sigma0_ndecay099",
        "seed": 0,
        "perm_mode": "identity",
        "n_tasks": n_tasks,
        "task_length": 5000,
        "per_task_accuracy": [1.0] * n_tasks,
        "mean_accuracy": 1.0,
        "wall_clock_seconds": 1.0,
        "provenance": {"schema": "asi.ipmnist.ceiling_run_provenance.v1"},
    }
    return json.dumps(payload)


def _npy_header_bytes(shape: tuple[int, ...], dtype: object) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array_header_2_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return buffer.getvalue()


def test_ceiling_rejects_oversize_npy_header_before_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    ceiling = tmp_path / "ceiling"
    ceiling.mkdir()
    path = ceiling / "stationary_sigma0_ndecay099_seed0.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.npy", _npy_header_bytes((), np.dtype("<U1")))
        archive.writestr(
            "per_step.npy", _npy_header_bytes((80_000, 5000), np.uint8)
        )

    def _forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.load must not run after an oversize npy header")

    monkeypatch.setattr(np, "load", _forbidden_load)
    with pytest.raises(ValueError, match="unbounded"):
        build_ceiling_summary(ceiling, tmp_path / "confirm")


def test_ceiling_rejects_compressed_oversize_members_before_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ceiling = tmp_path / "ceiling"
    ceiling.mkdir()
    path = ceiling / "stationary_sigma0_ndecay099_seed0.npz"
    per_step = np.zeros((80_000, 5000), dtype=np.uint8)
    np.savez_compressed(
        path,
        metadata=np.asarray(_valid_metadata()),
        per_step=per_step,
    )
    del per_step

    def _forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.load must not run after an oversize compressed NPZ")

    monkeypatch.setattr(np, "load", _forbidden_load)
    with pytest.raises(ValueError, match="unbounded"):
        build_ceiling_summary(ceiling, tmp_path / "confirm")


def test_ceiling_accepts_bounded_compressed_npz(tmp_path: Path) -> None:
    ceiling = tmp_path / "ceiling"
    ceiling.mkdir()
    per_step = np.ones((1, 5000), dtype=np.uint8)
    np.savez_compressed(
        ceiling / "stationary_sigma0_ndecay099_seed0.npz",
        metadata=np.asarray(_valid_metadata()),
        per_step=per_step,
    )
    report = build_ceiling_summary(ceiling, tmp_path / "confirm")
    assert report["stationary_sigma0_ndecay099"]["avg_online_mean"] == 1.0
