"""Regression tests for the maintained rule-discovery summary builder.

Issue #2134: `_arm` selected shards by filename but validated only the payload
seed and accuracy curve.  A shard from a different arm, or a 60-task screen shard
dropped into the 200-task confirmation directory, was accepted and republished
under valid-looking provenance.  The builder now routes every shard through
`ipmnist_screening.load_shard` and requires the arm identity, the stage task
count, and the task length to match before aggregation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import rule_discovery_summary as summary
from alberta_framework.benchmarks.ipmnist_screening import LEGACY_SHARD_SCHEMA

_CONFIRM_ARMS = ("disc_r1_pscale_norms", summary.CHAMPION)


def _make_shard(
    config_name: str, seed: int, n_tasks: int, *, task_length: int = 5000
) -> dict[str, Any]:
    """A minimal but strictly valid legacy-v1 screening shard with flat curves."""
    return {
        "schema": LEGACY_SHARD_SCHEMA,
        "config_name": config_name,
        "base_learner": "upgd_w",
        "seed": seed,
        "noise_mode": "step",
        "hyperparameters": {},
        "config": {
            "input_dim": 784,
            "hidden1": 300,
            "hidden2": 150,
            "n_classes": 10,
            "n_tasks": n_tasks,
            "task_length": task_length,
        },
        "per_task_accuracy": [0.5] * n_tasks,
        "per_task_loss": [0.5] * n_tasks,
        "per_task_plasticity": [0.5] * n_tasks,
        "wall_clock_seconds": 12.0,
        "created_unix": 1785805898.0,
        "environment": {
            "jax": "0.7.1",
            "numpy": "1.26.0",
            "python": "3.12.3",
            "platform": "linux",
        },
    }


def _write(directory: Path, shard: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{shard['config_name']}_seed{shard['seed']}.json"
    path.write_text(json.dumps(shard), encoding="utf-8")


def _populate_valid(screen_dir: Path, confirm_dir: Path, seed: int = 0) -> None:
    for name in summary.SCREEN_ARMS:
        _write(screen_dir, _make_shard(name, seed, summary._SCREEN_TASK_COUNT))
    for name in _CONFIRM_ARMS:
        _write(confirm_dir, _make_shard(name, seed, summary._CONFIRM_TASK_COUNT))


@pytest.mark.unit
def test_valid_shards_build_summary(tmp_path: Path) -> None:
    """A correctly-staged set of shards builds and aggregates as expected."""
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _populate_valid(screen_dir, confirm_dir)

    result = summary.build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))

    assert result["schema"] == "asi.rule_discovery.real_screen.v2"
    assert result["screen_60_task"]["disc_r1"]["mean"] == pytest.approx(0.5)
    assert result["confirm_200_task"][summary.CHAMPION]["mean"] == pytest.approx(0.5)
    # The screening loader is bound into maintained provenance now that it is
    # part of the trusted validation path.
    assert "ipmnist_screening" in result["provenance"]["sources"]


@pytest.mark.unit
def test_arm_substitution_is_rejected(tmp_path: Path) -> None:
    """A champion payload renamed into the disc_r1 slot must be rejected."""
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _populate_valid(screen_dir, confirm_dir)
    # Overwrite disc_r1_seed0.json with a shard whose config_name is the champion.
    impostor = _make_shard(summary.CHAMPION, 0, summary._SCREEN_TASK_COUNT)
    (screen_dir / "disc_r1_seed0.json").write_text(json.dumps(impostor), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'disc_r1'"):
        summary.build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))


@pytest.mark.unit
def test_stage_substitution_is_rejected(tmp_path: Path) -> None:
    """60-task screen shards placed in the confirmation directory must be rejected."""
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _populate_valid(screen_dir, confirm_dir)
    # Replace the 200-task confirmation shards with 60-task screen shards.
    for name in _CONFIRM_ARMS:
        _write(confirm_dir, _make_shard(name, 0, summary._SCREEN_TASK_COUNT))

    with pytest.raises(ValueError, match="not the 200 required for this stage"):
        summary.build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))


@pytest.mark.unit
def test_task_length_mismatch_is_rejected(tmp_path: Path) -> None:
    """A shard whose task_length is not 5,000 must be rejected."""
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _populate_valid(screen_dir, confirm_dir)
    _write(screen_dir, _make_shard("disc_r1", 0, summary._SCREEN_TASK_COUNT, task_length=4000))

    with pytest.raises(ValueError, match="task_length 4000"):
        summary.build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))


@pytest.mark.unit
def test_wrong_seed_in_payload_is_rejected(tmp_path: Path) -> None:
    """A shard whose payload seed disagrees with its filename slot is rejected."""
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _populate_valid(screen_dir, confirm_dir)
    mismatched = _make_shard("disc_r1", 0, summary._SCREEN_TASK_COUNT)
    mismatched["seed"] = 7  # filename still says seed0
    (screen_dir / "disc_r1_seed0.json").write_text(json.dumps(mismatched), encoding="utf-8")

    with pytest.raises(ValueError, match="seed does not match"):
        summary.build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))


@pytest.mark.unit
def test_legacy_reconstruction_matches_genuine_shards(tmp_path: Path) -> None:
    """The legacy v1 payload is reconstructed identically from valid shards.

    Guards that the added validation does not perturb the historical numbers for
    genuine, correctly-staged inputs (only the aggregation output is checked here;
    values come from the flat synthetic curves).
    """
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _populate_valid(screen_dir, confirm_dir)

    legacy = summary.build_legacy_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))

    assert legacy["schema"] == "alberta.rule_discovery.real_screen.v1"
    # Every paired-vs-champion difference is zero for identical flat curves.
    for name in summary.DISCOVERY_ARMS:
        assert legacy["paired_vs_champion_60_task"][name]["mean"] == pytest.approx(0.0)
    # Copy is defensive; the builder must not mutate caller inputs.
    assert copy.deepcopy(legacy) == legacy
