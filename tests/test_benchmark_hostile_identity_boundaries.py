"""Hostile identity regressions for development benchmark host boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks import _foragax_open_screen_probe as probe
from alberta_framework.benchmarks import foragax_open_screen as screen
from alberta_framework.benchmarks import forager_results as results
from alberta_framework.benchmarks import ipmnist_screening as ipmnist
from alberta_framework.benchmarks import official_foragax
from alberta_framework.benchmarks import reference_life_scorecard as scorecard
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile string truth hook executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile string comparison hook executed")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("hostile string repr hook executed")

    def strip(self, _chars: str | None = None) -> str:
        type(self).calls += 1
        raise AssertionError("hostile string strip hook executed")

    __hash__ = str.__hash__


class _HostileDict(dict[str, object]):
    def items(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile mapping iteration hook executed")


class _HostileList(list[object]):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile sequence iteration hook executed")


class _HostileValidatorDict(dict[str, object]):
    calls = 0

    def get(self, key: str, default: object = None) -> object:
        del key, default
        type(self).calls += 1
        raise AssertionError("hostile mapping get hook executed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile mapping iteration hook executed")


def test_open_screen_helpers_reject_hostile_identities_before_hooks() -> None:
    hostile = _HostileString("results")
    _HostileString.calls = 0
    with pytest.raises(screen.ScreenError, match="relative path"):
        screen._normalized_relative_path(hostile, "result_root")
    with pytest.raises(screen.ScreenError, match="object"):
        screen._require_dict(_HostileDict(), "payload")
    with pytest.raises(screen.ScreenError, match="array"):
        screen._require_list(_HostileList(), "records")
    assert _HostileString.calls == 0


def test_open_screen_probe_rejects_hostile_scalar_identities_before_hooks() -> None:
    hostile = _HostileString("configs/agent.json")
    _HostileString.calls = 0
    with pytest.raises(RuntimeError, match="relative path"):
        probe._relative_path(hostile, "configuration")
    assert probe._is_exact_int(True) is False
    assert probe._is_exact_int(1) is True
    assert probe._is_exact_int(type("IntSubclass", (int,), {})(1)) is False
    assert _HostileString.calls == 0


def test_scorecard_canonical_json_rejects_hostile_identities_before_hooks() -> None:
    hostile = _HostileString("value")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="canonical JSON"):
        scorecard.canonical_json_bytes({"field": hostile})
    with pytest.raises(ValueError, match="canonical JSON"):
        scorecard.canonical_json_bytes(_HostileDict({"field": "value"}))
    assert scorecard._is_sha256(_HostileString("a" * 64)) is False
    assert _HostileString.calls == 0


@pytest.mark.parametrize(
    "validator",
    (scorecard.validate_scorecard_run_record, scorecard.validate_scorecard_artifact),
)
def test_scorecard_public_validators_admit_exact_json_before_mapping_hooks(
    validator: object,
) -> None:
    hostile = _HostileValidatorDict()
    _HostileValidatorDict.calls = 0

    with pytest.raises(ValueError, match="exact JSON object"):
        validator(hostile)  # type: ignore[operator]

    assert _HostileValidatorDict.calls == 0


def test_forager_result_specs_reject_hostile_strings_before_hooks(tmp_path: Path) -> None:
    hostile = _HostileString("DQN-15")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="expected_config_agent"):
        results.LegacyFOVSQLiteRunSpec(
            agent="DQN",
            path=tmp_path / "result.sqlite",
            config_path=tmp_path / "config.json",
            run_index=0,
            stored_seed=0,
            expected_config_agent=hostile,
            expected_aperture_size=15,
        )
    with pytest.raises(ValueError, match="agent must be a non-empty string"):
        results.LegacyFOVSQLiteRunSpec(
            agent=hostile,
            path=tmp_path / "result.sqlite",
            config_path=tmp_path / "config.json",
            run_index=0,
            stored_seed=0,
            expected_config_agent="DQN-15",
            expected_aperture_size=15,
        )
    with pytest.raises(ValueError, match="agent must be a non-empty string"):
        results.OfficialForagaxRunSpec(agent=hostile, seed=0, path=tmp_path / "run.npz")
    with pytest.raises(ValueError, match="full lowercase SHA-256"):
        results._validate_sha256(_HostileString("a" * 64), name="digest")
    with pytest.raises(ValueError, match="40-character Git SHA"):
        results._validate_git_sha(_HostileString("a" * 40), name="commit")
    assert _HostileString.calls == 0


def test_ipmnist_hostile_strings_fail_before_comparison_or_repr() -> None:
    hostile = _HostileString("upgd_w")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="must be a non-empty string"):
        ipmnist._required_nonempty_string(hostile, context="identity")
    with pytest.raises(ValueError, match="noise_mode must be"):
        ipmnist._validated_screening_noise_mode(
            hostile,
            ipmnist.screening_spec("upgd_w_control"),
        )

    config = IPMNISTConfig(n_tasks=1, task_length=1)
    curve = np.zeros((1,), dtype=np.float64)
    result = ipmnist.ScreeningRunResult(
        config_name="upgd_w_control",
        base_learner="upgd_w",
        hyperparameters={},
        seed=0,
        config=config,
        per_task_accuracy=curve,
        per_task_loss=curve,
        per_task_plasticity=curve,
        wall_clock_seconds=0.0,
    )
    object.__setattr__(result, "base_learner", hostile)
    with pytest.raises(ValueError, match="base_learner must be a non-empty string"):
        ipmnist.shard_payload(
            result,
            source_provenance={},
            dataset_provenance={},
            environment={},
        )
    assert _HostileString.calls == 0


def test_official_foragax_request_rejects_hostile_repository_before_hooks(
    tmp_path: Path,
) -> None:
    hostile = _HostileString("https://github.com/Normal-Computing/foragax-agents")
    _HostileString.calls = 0
    with pytest.raises(
        official_foragax.OfficialForagaxValidationError,
        match="expected_repository must be a string",
    ):
        official_foragax.OfficialForagaxRunRequest(
            repository=tmp_path,
            execution_commit="a" * 40,
            config_path=Path("config.json"),
            interpreter=Path("python"),
            output_dir=tmp_path / "output",
            index=0,
            expected_repository=hostile,
        )
    assert _HostileString.calls == 0
