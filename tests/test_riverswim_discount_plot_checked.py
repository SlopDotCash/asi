"""Regressions for the post-collection plotting entry point."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "outputs/reference_life_control_calibration/discounted-sarsa-horizon-2026-10-15"
HERE = ROOT / "plot-audit-2026-11-01"
spec = importlib.util.spec_from_file_location("plot_checked", HERE / "plot_checked.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_indices_include_last_point_once() -> None:
    for length in (1, 100, 101, 201, 20000):
        indices = plot.sample_indices(length)
        assert indices[-1] == length - 1
        assert len(indices) == len(set(indices))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "wrong"),
        ("arm", "gamma_099"),
        ("seed", 101002),
        ("accepted_events", 1),
        ("config", {}),
        (
            "config",
            {
                "gamma": 0.9900000095367432,
                "lamda": 0.800000011920929,
                "step_size": 0.014000000432133675,
            },
        ),
        ("rewards", [0.0]),
        ("rewards", [2.0] * 20000),
        ("rewards", [float("nan")] * 20000),
        ("positions", []),
    ],
)
def test_rejects_wrong_record_metadata_and_lengths(
    tmp_path: Path, field: str, value: object
) -> None:
    for seed in plot.SEEDS:
        for arm in plot.ARMS:
            filename = f"{arm}-seed{seed}.json"
            shutil.copyfile(ROOT / filename, tmp_path / filename)
    target = tmp_path / "gamma_090-seed101001.json"
    record = json.loads(target.read_text(encoding="utf-8"))
    record[field] = value
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        plot.load_records(tmp_path)


def test_refuses_frozen_paths() -> None:
    with pytest.raises(ValueError):
        plot.output_paths(ROOT / "nested", ROOT)


def test_render_creates_nested_output_and_refuses_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output = tmp_path / "new" / "nested"
    command = [
        sys.executable,
        str(HERE / "plot_checked.py"),
        "--records",
        str(ROOT),
        "--out",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    assert first.returncode == 0, first.stderr
    assert (output / "reward-prefixes.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in (output / "reward-prefixes.svg").read_text(encoding="utf-8")
    second = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    assert second.returncode != 0
    assert "must not already exist" in second.stderr
