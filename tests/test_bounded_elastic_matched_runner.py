from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.bounded_elastic_matched_runner as runner
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult, screening_spec
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    registered_bounded_elastic_hyperparameters,
)
from tests._forager_matched_platform import requires_o_tmpfile

SMALL = IPMNISTConfig(n_tasks=1, task_length=5000, input_dim=2, hidden1=4, hidden2=2, n_classes=2)


def _run_for_test(
    data_x: object, data_y: object, *, config: IPMNISTConfig
) -> dict[str, object]:
    return runner._run_bounded_elastic_matched_authorized(
        data_x,
        data_y,
        config=config,
        seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )


@requires_o_tmpfile
def test_plan_is_prospective_and_public_execution_is_hard_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = runner.frozen_plan()
    assert plan["seeds"] == [51_562_001, 51_562_002, 51_562_003, 51_562_004, 51_562_005]
    assert plan["reviewed_execution_transition"] is False
    assert plan["execution_authorized"] is False
    assert plan["reservation_precedes_execution_and_publication"] is True
    assert plan["pre_dispatch_failure_receipts_retained"] is False
    assert plan["post_dispatch_failure_tombstone_retained"] is True
    assert plan["post_dispatch_retry_prevention"] is True
    assert plan["seed_policy"] == {
        "campaign_roster_status": "reserved_unconsumed",
        "test_only_seeds": [201, 202, 203, 204, 205],
        "campaign_and_test_rosters_disjoint": True,
    }
    assert set(runner.CAMPAIGN_SEEDS).isdisjoint(runner.TEST_ONLY_SEEDS)
    monkeypatch.setattr(runner, "_REVIEWED_EXECUTION_TRANSITION", True)
    monkeypatch.setattr(runner, "_EXECUTION_AUTHORIZED", True)
    assert runner.frozen_plan() == plan
    monkeypatch.setattr(runner, "_REVIEWED_EXECUTION_TRANSITION", False)
    monkeypatch.setattr(runner, "_EXECUTION_AUTHORIZED", False)
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("runner dispatched before authorization")

    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    with pytest.raises(RuntimeError, match="standalone.*disabled"):
        runner.run_bounded_elastic_matched(*_data(), config=SMALL)
    with pytest.raises(RuntimeError, match="not authorized"):
        runner._run_bounded_elastic_matched_authorized(
            *_data(),
            config=SMALL,
            seeds=runner.CAMPAIGN_SEEDS,
            _capability=runner._EXECUTION_CAPABILITY,
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        runner._validate_bounded_elastic_matched_authorized(
            {},
            *_data(),
            config=SMALL,
            seeds=runner.CAMPAIGN_SEEDS,
            _capability=runner._EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_direct_internal_reexecution_cannot_bypass_campaign_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("campaign dispatch bypassed authorization")

    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    with pytest.raises(RuntimeError, match="not authorized"):
        runner._validate_bounded_elastic_matched(
            {},
            *_data(),
            config=SMALL,
            seeds=runner.CAMPAIGN_SEEDS,
            reexecute=True,
        )
    assert calls == 0


def _data() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(10_000, dtype=np.float32).reshape(5000, 2) / 10_000.0
    y = np.arange(5000, dtype=np.int32) % 2
    return x, y


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(runner._canonical(unsigned)).hexdigest()


def _fake_run(
    data_x: np.ndarray,
    data_y: np.ndarray,
    spec: object,
    seed: int,
    config: IPMNISTConfig,
) -> ScreeningRunResult:
    del data_x, data_y
    resolved = cast(Any, spec)
    value = 0.5 + float(
        runner.TEST_ONLY_SEEDS.index(seed) + tuple(runner.ARMS).index(resolved.name)
    ) / 100.0
    return ScreeningRunResult(
        config_name=resolved.name,
        base_learner="upgd_w",
        hyperparameters=registered_bounded_elastic_hyperparameters(resolved.name),
        seed=seed,
        config=config,
        per_task_accuracy=np.asarray([value], dtype=np.float64),
        per_task_loss=np.asarray([1.0 - value], dtype=np.float64),
        per_task_plasticity=np.asarray([value], dtype=np.float64),
        wall_clock_seconds=1.0,
    )


@requires_o_tmpfile
def test_campaign_runs_all_four_arms_across_frozen_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))
    runner._validate_bounded_elastic_matched_authorized(
        result, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )
    assert [(row["seed"], row["arm"]) for row in result["rows"]] == [
        (seed, arm) for seed in runner.TEST_ONLY_SEEDS for arm in runner.ARMS
    ]
    assert all(row["result"]["outcome_retained"] is True for row in result["rows"])
    assert result["aggregate"]["outcome"] == "rejected"
    comparisons = result["aggregate"]["primary_comparisons"]
    assert [comparison["candidate"] for comparison in comparisons] == [
        "bounded_growth",
        "bounded_elastic",
    ]
    assert all(comparison["baseline"] == "bounded_fixed_cbp" for comparison in comparisons)
    for seed in runner.TEST_ONLY_SEEDS:
        rows = [row for row in result["rows"] if row["seed"] == seed]
        assert len({row["execution_identity"]["schedule_sha256"] for row in rows}) == 1
        assert len({row["execution_identity"]["initial_parameters_sha256"] for row in rows}) == 1


@requires_o_tmpfile
def test_campaign_rejects_source_drift_during_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_source = runner._source_identity()
    calls = 0

    def drift_after_first_run(*args: object, **kwargs: object) -> ScreeningRunResult:
        nonlocal calls
        calls += 1
        drifted = dict(original_source)
        drifted["alberta_framework/benchmarks/ipmnist_screening.py"] = "0" * 64
        monkeypatch.setattr(runner, "_source_identity", lambda: drifted)
        return _fake_run(*args, **kwargs)

    monkeypatch.setattr(runner, "run_screening_config", drift_after_first_run)
    with pytest.raises(RuntimeError, match="changed during matched execution"):
        _run_for_test(*_data(), config=SMALL)
    assert calls == 1


@requires_o_tmpfile
def test_campaign_and_reexecution_reject_dataset_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _data()
    calls = 0

    def mutate_once(
        data_x: np.ndarray,
        data_y: np.ndarray,
        spec: object,
        seed: int,
        config: IPMNISTConfig,
    ) -> ScreeningRunResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            data_x[0, 0] += np.float32(0.25)
        return _fake_run(data_x, data_y, spec, seed, config)

    monkeypatch.setattr(runner, "run_screening_config", mutate_once)
    with pytest.raises(RuntimeError, match="dataset changed during matched execution"):
        _run_for_test(x, y, config=SMALL)

    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    clean_x, clean_y = _data()
    result = _run_for_test(clean_x, clean_y, config=SMALL)
    calls = 0
    monkeypatch.setattr(runner, "run_screening_config", mutate_once)
    with pytest.raises(RuntimeError, match="dataset changed during strict reexecution"):
        runner._validate_bounded_elastic_matched_authorized(
            result,
            clean_x,
            clean_y,
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


@requires_o_tmpfile
def test_validator_rejects_identity_resource_and_roster_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))

    forged = copy.deepcopy(result)
    forged["rows"][0]["execution_identity"]["schedule_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )

    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["resources"]["data_steps"] = 1
    _resign(forged)
    with pytest.raises(ValueError, match="step/query"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )

    missing = copy.deepcopy(result)
    missing["rows"].pop()
    _resign(missing)
    with pytest.raises(ValueError, match="roster"):
        runner._validate_bounded_elastic_matched_authorized(
            missing, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


@requires_o_tmpfile
def test_validator_reexecutes_and_rejects_self_consistent_metric_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))
    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["metrics"]["mean_online_accuracy"] = 0.99
    forged["aggregate"] = runner._aggregate(forged["rows"])
    _resign(forged)

    with pytest.raises(ValueError, match="reexecution"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


@requires_o_tmpfile
def test_campaign_rejects_unregistered_outcome_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))
    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["outcome"] = "supported"
    _resign(forged)

    with pytest.raises(ValueError, match="inconclusive"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


@requires_o_tmpfile
def test_writer_is_create_only_and_retains_negative_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    destination = tmp_path / "bounded-elastic.json"
    runner._write_bounded_elastic_matched_authorized(
        destination, result, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )
    retained = json.loads(destination.read_bytes())
    runner._validate_bounded_elastic_matched_authorized(
        retained, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )
    with pytest.raises(FileExistsError):
        runner._write_bounded_elastic_matched_authorized(
            destination, result, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert not destination.with_name(f".{destination.name}.reservation").exists()


@requires_o_tmpfile
def test_writer_strictly_rereads_before_link_and_retains_failed_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    destination = tmp_path / "bounded-elastic.json"
    marker = destination.with_name(f".{destination.name}.reservation")

    def fail_strict_stage(*_args: object, **_kwargs: object) -> tuple[int, int]:
        raise ValueError("strict stage failed")

    monkeypatch.setattr(runner, "_strict_reread_prepared_output", fail_strict_stage)

    with pytest.raises(ValueError, match="strict stage failed"):
        runner._write_bounded_elastic_matched_authorized(
            destination,
            result,
            *_data(),
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert not destination.exists()
    assert marker.read_bytes() == b"asi-bounded-elastic-consumed-without-result-v1\n"


@requires_o_tmpfile
def test_writer_retains_reservation_after_reexecution_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    destination = tmp_path / "bounded-elastic.json"
    marker = destination.with_name(f".{destination.name}.reservation")

    def fail_dispatch(*_args: object, **_kwargs: object) -> ScreeningRunResult:
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(runner, "run_screening_config", fail_dispatch)

    with pytest.raises(RuntimeError, match="dispatch failed"):
        runner._write_bounded_elastic_matched_authorized(
            destination,
            result,
            *_data(),
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert not destination.exists()
    assert marker.read_bytes() == b"asi-bounded-elastic-consumed-without-result-v1\n"


@requires_o_tmpfile
@pytest.mark.parametrize("replace_linked_inode", [False, True])
def test_post_link_failure_rolls_back_only_the_exact_published_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replace_linked_inode: bool,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    destination = tmp_path / "bounded-elastic.json"
    marker = destination.with_name(f".{destination.name}.reservation")
    original_fsync = runner.os.fsync
    directory_fsyncs = 0

    def fail_after_link(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(runner.os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                if replace_linked_inode:
                    destination.unlink()
                    destination.write_bytes(b"foreign replacement")
                raise OSError("post-link directory fsync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", fail_after_link)
    with pytest.raises(OSError, match="post-link"):
        runner._write_bounded_elastic_matched_authorized(
            destination,
            result,
            *_data(),
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    if replace_linked_inode:
        assert destination.read_bytes() == b"foreign replacement"
    else:
        assert not destination.exists()
    assert marker.read_bytes() == b"asi-bounded-elastic-consumed-without-result-v1\n"


def test_link_success_followed_by_exception_rolls_back_exact_inode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    destination = tmp_path / "bounded-elastic.json"
    original_link = runner._link_unnamed_file

    def link_then_interrupt(file_fd: int, directory_fd: int, name: str) -> None:
        original_link(file_fd, directory_fd, name)
        raise KeyboardInterrupt("interrupted after link")

    monkeypatch.setattr(runner, "_link_unnamed_file", link_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="interrupted after link"):
        runner._write_bounded_elastic_matched_authorized(
            destination,
            result,
            *_data(),
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert not destination.exists()
    marker = destination.with_name(f".{destination.name}.reservation")
    assert marker.read_bytes() == b"asi-bounded-elastic-consumed-without-result-v1\n"


def test_writer_rejects_replaced_visible_reservation_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    destination = tmp_path / "bounded-elastic.json"
    original_validate = runner._validate_bounded_elastic_matched

    def replace_marker(*args: object, **kwargs: object) -> None:
        original_validate(*args, **kwargs)
        marker = destination.with_name(f".{destination.name}.reservation")
        marker.unlink()
        marker.write_text("replacement", encoding="ascii")

    monkeypatch.setattr(runner, "_validate_bounded_elastic_matched", replace_marker)
    with pytest.raises(ValueError, match="owned visible regular marker"):
        runner._write_bounded_elastic_matched_authorized(
            destination,
            result,
            *_data(),
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert not destination.exists()
    marker = destination.with_name(f".{destination.name}.reservation")
    assert marker.read_text(encoding="ascii") == "replacement"
    marker.unlink()


def test_writer_parent_swap_does_not_publish_through_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    requested = tmp_path / "requested"
    destination = requested / "bounded-elastic.json"
    retired = tmp_path / "retired"
    competitor = b"replacement-directory"
    original_link = runner._link_unnamed_file

    def swap_parent(file_fd: int, directory_fd: int, name: str) -> None:
        requested.rename(retired)
        requested.mkdir()
        destination.write_bytes(competitor)
        original_link(file_fd, directory_fd, name)

    monkeypatch.setattr(runner, "_link_unnamed_file", swap_parent)
    with pytest.raises(RuntimeError, match="parent changed"):
        runner._write_bounded_elastic_matched_authorized(
            destination,
            result,
            *_data(),
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert destination.read_bytes() == competitor
    assert not (retired / destination.name).exists()


def test_public_writer_is_closed_before_creating_output(tmp_path: Path) -> None:
    destination = tmp_path / "never" / "report.json"
    with pytest.raises(RuntimeError, match="standalone.*disabled"):
        runner.write_bounded_elastic_matched(
            destination, {}, *_data(), config=SMALL
        )
    with pytest.raises(RuntimeError, match="reservation-first transaction"):
        runner._write_bounded_elastic_matched_authorized(
            destination,
            {},
            *_data(),
            config=SMALL,
            seeds=runner.CAMPAIGN_SEEDS,
            _capability=runner._EXECUTION_CAPABILITY,
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        runner._reserve_output(
            destination,
            _capability=runner._EXECUTION_CAPABILITY,
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        runner._publish_bounded_elastic_matched_reserved(
            (-1, "unused", "unused", -1, -1, -1),
            destination,
            {},
            *_data(),
            config=SMALL,
            seeds=runner.CAMPAIGN_SEEDS,
            _capability=runner._EXECUTION_CAPABILITY,
        )
    assert not destination.parent.exists()


def test_standalone_paths_remain_disabled_after_flag_transition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "never" / "report.json"
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("standalone path dispatched a runner")

    monkeypatch.setattr(runner, "_REVIEWED_EXECUTION_TRANSITION", True)
    monkeypatch.setattr(runner, "_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    with pytest.raises(RuntimeError, match="standalone.*disabled"):
        runner.run_bounded_elastic_matched(*_data(), config=SMALL)
    with pytest.raises(RuntimeError, match="standalone.*disabled"):
        runner.validate_bounded_elastic_matched({}, *_data(), config=SMALL)
    with pytest.raises(RuntimeError, match="standalone.*disabled"):
        runner.write_bounded_elastic_matched(destination, {}, *_data(), config=SMALL)
    assert calls == 0
    assert not destination.parent.exists()


def test_transaction_reserves_before_dataset_load_or_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "report.json"
    marker = destination.with_name(f".{destination.name}.reservation")
    marker.write_bytes(b"prior reservation")
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("consumer work preceded reservation")

    monkeypatch.setattr(runner, "load_mnist_train", forbidden)
    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    with pytest.raises(FileExistsError):
        runner._run_and_publish_bounded_elastic_matched_authorized(
            tmp_path,
            destination,
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0
    assert marker.read_bytes() == b"prior reservation"


def test_transaction_retains_owned_tombstone_after_first_dispatch_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "report.json"
    x, y = _data()
    calls = 0

    def fail_after_dispatch(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("injected consumer failure")

    monkeypatch.setattr(runner, "load_mnist_train", lambda _home: (x, y))
    monkeypatch.setattr(runner, "run_screening_config", fail_after_dispatch)
    with pytest.raises(RuntimeError, match="injected consumer failure"):
        runner._run_and_publish_bounded_elastic_matched_authorized(
            tmp_path,
            destination,
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    marker = destination.with_name(f".{destination.name}.reservation")
    assert marker.read_bytes() == b"asi-bounded-elastic-consumed-without-result-v1\n"
    assert not destination.exists()
    with pytest.raises(FileExistsError):
        runner._run_and_publish_bounded_elastic_matched_authorized(
            tmp_path,
            destination,
            config=SMALL,
            seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 1


def test_transaction_publishes_only_after_strict_reexecution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "report.json"
    x, y = _data()
    calls = 0

    def counted_run(*args: object, **kwargs: object) -> ScreeningRunResult:
        nonlocal calls
        calls += 1
        return _fake_run(*args, **kwargs)

    monkeypatch.setattr(runner, "load_mnist_train", lambda _home: (x, y))
    monkeypatch.setattr(runner, "run_screening_config", counted_run)
    report = runner._run_and_publish_bounded_elastic_matched_authorized(
        tmp_path,
        destination,
        config=SMALL,
        seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )
    assert json.loads(destination.read_bytes()) == report
    assert calls == 2 * len(runner.TEST_ONLY_SEEDS) * len(runner.ARMS)
    assert not destination.with_name(f".{destination.name}.reservation").exists()


def test_preflight_rejects_unbounded_or_wrong_dataset_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("screening must not execute")

    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    x, y = _data()
    with pytest.raises(ValueError, match="float32"):
        runner._run_bounded_elastic_matched_authorized(
            x.astype(np.float64), y, config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    huge = IPMNISTConfig(
        n_tasks=1,
        task_length=5000,
        input_dim=8_000,
        hidden1=10_000,
        hidden2=1,
        n_classes=2,
    )
    with pytest.raises(ValueError, match="persistent-memory"):
        runner._run_bounded_elastic_matched_authorized(
            x, y, config=huge, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_static_numeric_preflight_precedes_schedule_and_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _data()
    monkeypatch.setattr(runner, "_MAX_COMBINED_NUMERIC_BYTES", 1_000)
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("allocation or execution preceded aggregate preflight")

    monkeypatch.setattr(runner, "build_schedule", forbidden)
    monkeypatch.setattr(runner, "init_mlp_params", forbidden)
    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="static 256 MiB numeric accounting"):
        _run_for_test(x, y, config=SMALL)
    assert calls == 0


def test_resource_envelope_uses_task_length_and_rejects_unaccounted_dataset_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = runner._numeric_resource_envelope(config=SMALL, dataset_rows=5000)
    assert envelope["schedule_bytes"] == SMALL.n_tasks * (
        SMALL.task_length + SMALL.input_dim
    ) * 4
    assert "combined_numeric_bytes" not in envelope
    plan = runner.frozen_plan()
    assert "backend copies" in cast(str, plan["numeric_resource_scope"])
    transaction = cast(dict[str, int], plan["transaction_resource_accounting"])
    assert transaction == {
        "campaign_rows": 20,
        "initial_runner_dispatches": 20,
        "strict_reexecution_dispatches": 20,
        "total_runner_dispatches": 40,
        "total_observations": 1_600_000,
        "total_optimizer_updates": 1_600_000,
        "total_data_steps": 1_600_000,
        "total_environment_steps": 0,
        "total_model_queries": 3_200_000,
    }

    x, y = _data()
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("noncontiguous dataset reached execution")

    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="C-contiguous"):
        _run_for_test(x[:, ::-1], y, config=SMALL)
    assert calls == 0


def test_json_preflight_rejects_nested_hostile_type_without_dispatching_hooks() -> None:
    calls = 0

    class HostileMeta(type):
        def __hash__(cls) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile type was hashed")

        def __eq__(cls, other: object) -> bool:
            del other
            nonlocal calls
            calls += 1
            raise AssertionError("hostile type was compared")

    class Hostile(metaclass=HostileMeta):
        pass

    with pytest.raises(ValueError, match="exact JSON values"):
        runner._json_preflight({"outer": [{"inner": Hostile()}]})
    assert calls == 0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"field": 1 << 80}, "out-of-range integer"),
        ({"x" * 257: 1}, "oversized field name"),
        ({"field": "\ud800"}, "invalid Unicode"),
    ],
)
def test_json_preflight_rejects_hostile_scalar_bounds(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        runner._json_preflight(value)


def test_seed_roster_validation_does_not_dispatch_hostile_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class HostileInt(int):
        def __eq__(self, other: object) -> bool:
            del other
            nonlocal calls
            calls += 1
            raise AssertionError("hostile seed equality dispatched")

    hostile = (HostileInt(201), *runner.TEST_ONLY_SEEDS[1:])
    with pytest.raises(RuntimeError, match="seed roster"):
        runner._run_bounded_elastic_matched_authorized(
            *_data(),
            config=SMALL,
            seeds=cast(Any, hostile),
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_output_path_bounds_fail_before_directory_creation(tmp_path: Path) -> None:
    destination = tmp_path / "never" / ("x" * 241) / "report.json"
    with pytest.raises(ValueError, match="oversized component"):
        runner._reserve_output(
            destination, _capability=runner._TEST_EXECUTION_CAPABILITY
        )
    assert not (tmp_path / "never").exists()


def test_registered_controls_are_exact() -> None:
    for arm in runner.ARMS:
        assert screening_spec(arm).hyperparameters == registered_bounded_elastic_hyperparameters(
            arm
        )


def test_source_identity_uses_exact_audited_sources_and_dependency_inputs() -> None:
    assert set(runner._source_identity()) == {
        "alberta_framework/_seed_validation.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/evaluation/bounded_elastic_ipmnist_nonpromoting.py",
        "alberta_framework/evaluation/bounded_elastic_matched_runner.py",
        "pyproject.toml",
        "uv.lock",
    }


@pytest.mark.parametrize("dependency_input", ["pyproject.toml", "uv.lock"])
def test_source_identity_detects_dependency_input_mutation(
    monkeypatch: pytest.MonkeyPatch, dependency_input: str
) -> None:
    original = runner._source_identity()
    read_bytes = Path.read_bytes

    def mutated(path: Path) -> bytes:
        payload = read_bytes(path)
        if path.name == dependency_input:
            return payload + b"\nprospective-mutation"
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutated)
    changed = runner._source_identity()
    assert changed[dependency_input] != original[dependency_input]
    assert all(
        changed[path] == digest
        for path, digest in original.items()
        if path != dependency_input
    )
