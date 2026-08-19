"""Hostile int gate for upgd_ipmnist hyperparameters and wall_clock before float."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __float__")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __eq__")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __float__ float")


def _write_payload(payload: dict) -> Path:
    # Write via json but hostile will be serialized as int via json dumps calling int.__repr__?
    # Instead we monkey-patch _strict_json_object to return hostile directly
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(json.dumps(payload).encode())
    tmp.close()
    return Path(tmp.name)


def test_hyperparameter_rejects_hostile_before_float(monkeypatch) -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0

    def fake_strict(path: Path):
        return {
            "schema": "test",
            "schema_version": 2,
            "evidence_policy": "nonpromoting",
            "deviations": [],
            "learner": "upgd",
            "hyperparameters": {"lr": hostile},
            "config": {
                "n_tasks": 1,
                "task_length": 1,
                "input_dim": 1,
                "hidden1": 1,
                "hidden2": 1,
                "n_classes": 2,
            },
            "wall_clock_seconds": 1.0,
            "seed_count": 1,
            "seed_ids": [0],
            "per_task_accuracy": [[0.5]],
            "per_task_loss": [[0.1]],
            "per_task_plasticity": [[0.5]],
            "matches_selected_publication_configuration": False,
            "selected_publication_configuration_match_scope": "network_task_shape_and_horizon_only",
        }

    import alberta_framework.benchmarks.upgd_ipmnist as mod

    monkeypatch.setattr(mod, "_strict_json_object", fake_strict)
    with pytest.raises(ValueError):
        mod._validated_partial_payload(Path("dummy.json"), schema="test", seed_field="seed_ids")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_wall_clock_rejects_hostile_before_float(monkeypatch) -> None:
    hostile = _HostileFloat(1.0)
    _HostileFloat.calls = 0

    def fake_strict(path: Path):
        return {
            "schema": "test",
            "schema_version": 2,
            "evidence_policy": "nonpromoting",
            "deviations": [],
            "learner": "upgd",
            "hyperparameters": {"lr": 0.01},
            "config": {
                "n_tasks": 1,
                "task_length": 1,
                "input_dim": 1,
                "hidden1": 1,
                "hidden2": 1,
                "n_classes": 2,
            },
            "wall_clock_seconds": hostile,
            "seed_count": 1,
            "seed_ids": [0],
            "per_task_accuracy": [[0.5]],
            "per_task_loss": [[0.1]],
            "per_task_plasticity": [[0.5]],
            "matches_selected_publication_configuration": False,
            "selected_publication_configuration_match_scope": "network_task_shape_and_horizon_only",
        }

    import alberta_framework.benchmarks.upgd_ipmnist as mod

    monkeypatch.setattr(mod, "_strict_json_object", fake_strict)
    with pytest.raises(ValueError):
        mod._validated_partial_payload(Path("dummy.json"), schema="test", seed_field="seed_ids")  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
