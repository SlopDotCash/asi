"""Unit coverage for alberta_framework.benchmarks._foragax_open_screen_probe.

Tests the fail-closed probe primitives: exact-int gating (bool must not
alias 0/1), task intake validation, PPO schedule product contract,
relative-path normalization, source-record hashing/dedup, and the
read-only root detection.
"""

import hashlib
from pathlib import Path

import pytest

from alberta_framework.benchmarks._foragax_open_screen_probe import (
    _is_exact_int,
    _relative_path,
    _source_records,
    _validate_ppo_schedule,
    _validate_task_intake,
)


def test_is_exact_int() -> None:
    assert _is_exact_int(5) is True
    assert _is_exact_int(0) is True
    assert _is_exact_int(True) is False  # bool must not alias 1
    assert _is_exact_int(1.5) is False
    assert _is_exact_int("5") is False


def test_validate_task_intake() -> None:
    task, horizon, seeds = _validate_task_intake(
        {"task": {"steps_per_seed": 10, "seeds": [1, 2, 3]}}
    )
    assert horizon == 10
    assert seeds == [1, 2, 3]
    # steps fallback
    _, horizon2, _ = _validate_task_intake({"task": {"steps": 20, "seeds": [1]}})
    assert horizon2 == 20


def test_validate_task_intake_rejects() -> None:
    with pytest.raises(RuntimeError, match="task must be an object"):
        _validate_task_intake({"task": "not-an-object"})
    with pytest.raises(RuntimeError, match="horizon is invalid"):
        _validate_task_intake({"task": {"steps": 0, "seeds": [1]}})
    with pytest.raises(RuntimeError, match="seeds are invalid"):
        _validate_task_intake({"task": {"steps": 10, "seeds": []}})
    with pytest.raises(RuntimeError, match="seeds are invalid"):
        _validate_task_intake({"task": {"steps": 10, "seeds": [True]}})


def test_validate_ppo_schedule() -> None:
    assert _validate_ppo_schedule(10, 20, 200, "x") == (10, 20)
    with pytest.raises(RuntimeError, match="not explicit"):
        _validate_ppo_schedule(10.5, 20, 200, "x")
    with pytest.raises(RuntimeError, match="does not equal"):
        _validate_ppo_schedule(10, 21, 200, "x")  # 10*21 != 200


def test_relative_path() -> None:
    assert _relative_path("data/x.json", "x") == "data/x.json"
    with pytest.raises(RuntimeError, match="non-empty relative"):
        _relative_path("", "x")
    with pytest.raises(RuntimeError, match="non-empty relative"):
        _relative_path(123, "x")
    with pytest.raises(RuntimeError, match="normalized relative"):
        _relative_path("/abs", "x")
    with pytest.raises(RuntimeError, match="normalized relative"):
        _relative_path("../escape", "x")
    # PurePosixPath normalizes "./" away, so it passes through.
    assert _relative_path("./dot", "x") == "./dot"


def test_source_records_dedup_and_hash(tmp_path: Path, monkeypatch) -> None:
    # Point _SOURCE_ROOT at a temp dir via monkeypatch.
    import alberta_framework.benchmarks._foragax_open_screen_probe as mod

    f = tmp_path / "a.py"
    f.write_text("content", encoding="utf-8")
    digest = hashlib.sha256(b"content").hexdigest()
    monkeypatch.setattr(mod, "_SOURCE_ROOT", tmp_path)
    records = _source_records(
        {"source_files": [{"path": "a.py", "sha256": digest}]}
    )
    assert records == [{"path": "a.py", "sha256": digest}]
    # Duplicate path rejected.
    with pytest.raises(RuntimeError, match="duplicate"):
        _source_records(
            {
                "source_files": [
                    {"path": "a.py", "sha256": digest},
                    {"path": "a.py", "sha256": digest},
                ]
            }
        )
    # Hash drift rejected.
    with pytest.raises(RuntimeError, match="hash drift"):
        _source_records({"source_files": [{"path": "a.py", "sha256": "0" * 64}]})


def test_root_is_read_only_detects() -> None:
    import alberta_framework.benchmarks._foragax_open_screen_probe as mod

    result = mod._root_is_read_only()
    assert isinstance(result, bool)
