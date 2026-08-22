from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import alberta_framework.evaluation.nap_matched_campaign as campaign
from alberta_framework.benchmarks.nap_ipmnist import CAMPAIGN_RESERVED_SEEDS
from alberta_framework.benchmarks.plasticity_diagnostics import INPUT_DIM


def _data() -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(12 * INPUT_DIM, dtype=np.float32).reshape(12, INPUT_DIM)
    return (values % 256) / 255.0, np.arange(12, dtype=np.int32) % 10


def test_plan_freezes_fresh_canonical_nonpromoting_campaign() -> None:
    plan = campaign.frozen_plan()
    assert plan["seeds"] == list(CAMPAIGN_RESERVED_SEEDS)
    assert plan["profile_id"] == "bounded-development"
    assert plan["dataset"]["sha256"] == (
        "234322a369029211eb4555087fc5448c972215e4a50dc4e4d8a21b5a3f8d4d9a"
    )
    assert plan["primary_paired_question"]["candidate"] == "nap"
    assert plan["primary_paired_question"]["control"] == "nap_mechanism_off"
    assert plan["execution"]["reviewed_transition"] is False
    assert plan["execution"]["authorized"] is False
    assert plan["policy"]["scientific_promotion_allowed"] is False
    assert plan["output_path"] == "outputs/nap_matched/v1/report.json"


def test_public_execution_fails_before_reservation_or_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("unauthorized campaign reached a consumer")

    monkeypatch.setattr(campaign, "_reserve", forbidden)
    monkeypatch.setattr(campaign, "_load_canonical_dataset", forbidden)
    monkeypatch.setattr(campaign, "_run_comparator_for_seeds", forbidden)
    with pytest.raises(PermissionError, match="not authorized"):
        campaign.run_and_publish(tmp_path)
    assert calls == 0


def test_test_roster_executes_and_independently_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    real = campaign._run_comparator_for_seeds
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(campaign, "_run_comparator_for_seeds", counted)
    report = campaign._run_for_test(*_data())
    assert calls == len(campaign.TEST_ONLY_SEEDS)
    campaign.validate_report(
        report, *_data(), seeds=campaign.TEST_ONLY_SEEDS, profile_id="contract-smoke",
        reexecute=True
    )
    assert calls == 2 * len(campaign.TEST_ONLY_SEEDS)


def test_validator_rejects_metric_resource_and_policy_forgery() -> None:
    report = campaign._run_for_test(*_data())
    for mutate in (
        lambda value: value["rows"][0]["result"]["arms"][0]["task_accuracy"].__setitem__(0, 2.0),
        lambda value: value["rows"][0]["result"]["arms"][0]["receipt"].__setitem__("data_steps", 1),
        lambda value: value["policy"].__setitem__("scientific_promotion_allowed", True),
    ):
        forged = copy.deepcopy(report)
        mutate(forged)
        campaign._sign(forged)
        with pytest.raises(ValueError):
            campaign.validate_report(
                forged, *_data(), seeds=campaign.TEST_ONLY_SEEDS,
                profile_id="contract-smoke", reexecute=False
            )


def test_replay_ignores_timing_only() -> None:
    report = campaign._run_for_test(*_data())
    for row in report["rows"]:
        for arm in row["result"]["arms"]:
            arm["receipt"]["elapsed_ns"] += 1
    campaign._sign(report)
    campaign.validate_report(
        report, *_data(), seeds=campaign.TEST_ONLY_SEEDS,
        profile_id="contract-smoke", reexecute=True
    )


def test_short_transaction_writes_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    output = bytearray()

    def short_write(fd: int, payload: object) -> int:
        del fd
        chunk = bytes(payload)[:2]
        output.extend(chunk)
        return len(chunk)

    monkeypatch.setattr(campaign.os, "write", short_write)
    campaign._write_all(9, b"abcdefg")
    assert output == b"abcdefg"


def test_registered_output_parent_replacement_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = tmp_path / "registered"
    parent.mkdir()
    destination = parent / "report.json"
    monkeypatch.setattr(campaign, "OUTPUT_PATH", destination)
    reservation = campaign._reserve(destination)
    moved = tmp_path / "moved"
    parent.rename(moved)
    parent.mkdir()
    try:
        with pytest.raises(RuntimeError, match="parent identity"):
            campaign._assert_visible_parent(reservation)
    finally:
        campaign._finish(reservation, consumed=False)
