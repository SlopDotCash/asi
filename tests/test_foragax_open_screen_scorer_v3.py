"""Unit coverage for alberta_framework.benchmarks._foragax_open_screen_scorer_v3.

Tests the frozen EMA scorer (shared with v2 semantics), canonical
relative-path validation, NPY header contract checks, and the NPZ zip
inventory validation (member safety, sizes, duplicates).
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks._foragax_open_screen_scorer_v3 import (
    _relative,
    _validate_reward_header,
    _validate_zip,
    score_rewards,
)


def _make_npz(path: Path, rewards: np.ndarray) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        buffer = io.BytesIO()
        np.save(buffer, rewards)
        zf.writestr("rewards.npy", buffer.getvalue())


def test_score_rewards_basic() -> None:
    rewards = np.ones(100, dtype=np.float32)
    result = score_rewards(rewards, horizon=100)
    assert result["reward_sum_float64"] == 100.0
    assert result["reward_shape"] == [100]
    assert len(result["reward_trace_sha256"]) == 64


def test_score_rewards_rejects_bad() -> None:
    with pytest.raises(ValueError, match="exact shape"):
        score_rewards(np.ones(99), horizon=100)
    with pytest.raises(ValueError, match="finite"):
        score_rewards(np.array([1.0, np.nan]), horizon=2)


def test_relative() -> None:
    assert _relative("results/x").as_posix() == "results/x"
    for bad in ["/abs", "a/../b", "./a", "a//b"]:
        with pytest.raises(ValueError, match="canonical and relative"):
            _relative(bad)


def test_validate_reward_header_ok(tmp_path: Path) -> None:
    buf = io.BytesIO()
    np.save(buf, np.ones(50, dtype=np.float32))
    buf.seek(0)
    with zipfile.ZipFile(tmp_path / "x.zip", "w") as zf:
        zf.writestr("rewards.npy", buf.getvalue())
    with zipfile.ZipFile(tmp_path / "x.zip") as zf:
        info = zf.infolist()[0]
        _validate_reward_header(zf, info, horizon=50, path=tmp_path / "x.zip")  # no raise


def test_validate_reward_header_rejects_wrong_shape(tmp_path: Path) -> None:
    buf = io.BytesIO()
    np.save(buf, np.ones(60, dtype=np.float32))  # wrong length
    buf.seek(0)
    with zipfile.ZipFile(tmp_path / "x.zip", "w") as zf:
        zf.writestr("rewards.npy", buf.getvalue())
    with zipfile.ZipFile(tmp_path / "x.zip") as zf:
        info = zf.infolist()[0]
        with pytest.raises(ValueError, match="contract drift"):
            _validate_reward_header(zf, info, horizon=50, path=tmp_path / "x.zip")


def test_validate_zip_ok(tmp_path: Path) -> None:
    path = tmp_path / "seed0.npz"
    _make_npz(path, np.ones(50, dtype=np.float32))
    with path.open("rb") as stream:
        _validate_zip(stream, path, horizon=50)  # no raise


def test_validate_zip_rejects_traversal(tmp_path: Path) -> None:
    path = tmp_path / "evil.npz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("../evil.npy", b"x" * 8)
    with path.open("rb") as stream:
        with pytest.raises(ValueError, match="invalid NPZ"):
            _validate_zip(stream, path, horizon=50)


def test_validate_zip_rejects_duplicate_members(tmp_path: Path) -> None:
    path = tmp_path / "dup.npz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("rewards.npy", b"x" * 8)
        zf.writestr("rewards.npy", b"y" * 8)  # duplicate
    with path.open("rb") as stream:
        with pytest.raises(ValueError, match="invalid NPZ"):
            _validate_zip(stream, path, horizon=50)
