"""Hostile validation for the rule-discovery summary builder (#2134).

The maintained summary must reject arm substitution (a payload whose
config_name belongs to a different arm than the expected filename) and
stage substitution (60-task screen shards presented as 200-task
confirmation shards) before aggregation or provenance publication.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.rule_discovery_summary import (
    CHAMPION,
    SCREEN_ARMS,
    _arm,
)


def _write_shard(
    directory: Path,
    name: str,
    seed: int,
    *,
    n_tasks: int = 60,
    config_name: str | None = None,
) -> Path:
    """Write a minimal valid shard payload for a given arm/seed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_seed{seed}.json"
    payload = {
        "config_name": config_name if config_name is not None else name,
        "seed": seed,
        "per_task_accuracy": [0.8] * n_tasks,
        "config": {
            "hidden1": 300,
            "hidden2": 150,
            "input_dim": 784,
            "n_classes": 10,
            "n_tasks": n_tasks,
            "task_length": 5000,
        },
    }
    path.write_text(json.dumps(payload))
    return path


class TestArmSubstitution:
    """A shard whose config_name differs from the expected arm must fail."""

    def test_config_name_mismatch_rejected(self, tmp_path: Path) -> None:
        # Filename says disc_r1, payload claims sigma0_shiftnorm_d099.
        _write_shard(
            tmp_path,
            "disc_r1",
            0,
            n_tasks=60,
            config_name="sigma0_shiftnorm_d099",
        )
        with pytest.raises(
            ValueError,
            match=r"config_name 'sigma0_shiftnorm_d099' does not match expected arm 'disc_r1'",
        ):
            _arm(tmp_path, "disc_r1", [0], expected_tasks=60)

    def test_matching_config_name_accepted(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, "disc_r1", 0, n_tasks=60, config_name="disc_r1")
        result = _arm(tmp_path, "disc_r1", [0], expected_tasks=60)
        assert result["per_seed"] == pytest.approx([0.8])


class TestStageSubstitution:
    """Screen (60-task) and confirmation (200-task) shards must not mix."""

    def test_screen_shard_rejected_at_confirmation(self, tmp_path: Path) -> None:
        # A 60-task shard presented where 200 tasks are required.
        _write_shard(tmp_path, CHAMPION, 0, n_tasks=60, config_name=CHAMPION)
        with pytest.raises(
            ValueError,
            match=r"has 60 tasks, expected 200 for this stage",
        ):
            _arm(tmp_path, CHAMPION, [0], expected_tasks=200)

    def test_confirmation_shard_rejected_at_screen(self, tmp_path: Path) -> None:
        # A 200-task shard presented where 60 tasks are required.
        _write_shard(tmp_path, "disc_r1", 0, n_tasks=200, config_name="disc_r1")
        with pytest.raises(
            ValueError,
            match=r"has 200 tasks, expected 60 for this stage",
        ):
            _arm(tmp_path, "disc_r1", [0], expected_tasks=60)

    def test_correct_stage_accepted(self, tmp_path: Path) -> None:
        _write_shard(tmp_path, CHAMPION, 0, n_tasks=200, config_name=CHAMPION)
        result = _arm(tmp_path, CHAMPION, [0], expected_tasks=200)
        assert result["per_seed"] == pytest.approx([0.8])
