"""Prospective end-to-end Intentional Updates TD/control contract."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Never

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks import intentional_updates_control as lane
from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    intentional_updates_development_record,
    screening_spec,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _test_only_seed_roster() -> Iterator[None]:
    patch = pytest.MonkeyPatch()
    patch.setattr(lane, "SEEDS", lane.TEST_ONLY_SEEDS)
    yield
    patch.undo()


def _control(
    arm: str, *, seed: int, horizon: int = 512, phase_length: int = 64
) -> dict[str, object]:
    return lane._run_control_shard_authorized(
        arm,
        seed=seed,
        horizon=horizon,
        phase_length=phase_length,
        _capability=lane._TEST_EXECUTION_CAPABILITY,
    )


def test_plan_is_fresh_prospective_and_permanently_nonpromoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = lane.frozen_plan()
    assert plan["seeds"] == [41_562_001, 41_562_002, 41_562_003, 41_562_004]
    assert lane.QUARANTINED_SEEDS == (31_561_001, 31_561_002, 31_561_003, 31_561_004)
    assert set(lane.CAMPAIGN_SEEDS).isdisjoint(lane.TEST_ONLY_SEEDS)
    assert plan["execution_authorized"] is False
    assert plan["reviewed_execution_transition"] is False
    assert plan["scientific_promotion_allowed"] is False
    assert plan["negative_outcomes_retained"] is True
    assert plan["confidence_critical"] == 5.391949071934058
    assert plan["confidence_critical"].hex() == "0x1.5915b18f69e09p+2"
    assert plan["protocol_families"] == ["supervised_ipmnist", "td_control"]
    assert plan["dataset"]["x"]["sha256"] == (
        "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
    )
    assert plan["dataset"]["y"]["sha256"] == (
        "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
    )
    monkeypatch.setattr(lane, "_REVIEWED_EXECUTION_TRANSITION", True)
    monkeypatch.setattr(lane, "_EXECUTION_AUTHORIZED", True)
    assert lane.frozen_plan() == plan


def test_catalog_cli_is_read_only_and_execution_stays_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert lane.main(["--catalog"]) == 0
    assert json.loads(capsys.readouterr().out) == lane.frozen_plan()


@pytest.mark.parametrize("dependency_input", ["pyproject.toml", "uv.lock"])
def test_source_identity_binds_exact_dependency_inputs(
    monkeypatch: pytest.MonkeyPatch, dependency_input: str,
) -> None:
    plan_files = lane.frozen_plan()["source_identity_policy"]["hashed_files"]
    assert "pyproject.toml" in plan_files
    assert "uv.lock" in plan_files
    original = lane._source_identity()
    read_bytes = Path.read_bytes

    def mutated(path: Path) -> bytes:
        payload = read_bytes(path)
        return payload + b"\nmutation" if path.name == dependency_input else payload

    monkeypatch.setattr(Path, "read_bytes", mutated)
    changed = lane._source_identity()
    assert changed[dependency_input] != original[dependency_input]


def test_runtime_identity_binds_dependencies_jax_devices_config_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lane._runtime_identity()
    assert set(original["packages"]) == {
        "chex", "jax", "jaxlib", "jaxtyping", "numpy", "orbax-checkpoint",
        "scikit-learn", "scipy",
    }
    assert original["jax"]["devices"]
    assert "jax_default_prng_impl" in original["jax"]["config"]
    assert "XLA_FLAGS" in original["process_environment"]
    monkeypatch.setenv("JAX_RANDOM_SEED_OFFSET", "provenance-mutation")
    changed = lane._runtime_identity()
    assert changed != original
    assert changed["process_environment"]["JAX_RANDOM_SEED_OFFSET"] == (
        "provenance-mutation"
    )


def test_current_runtime_rejects_oversized_environment_before_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XLA_FLAGS", "x" * 16_385)
    with pytest.raises(ValueError, match="bounded exact JSON string"):
        lane._current_runtime()


def test_runtime_identity_bounds_device_inventory_before_attribute_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class HostileDevice:
        def __getattribute__(self, name: str) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError(f"device attribute accessed before bound: {name}")

    oversized = [HostileDevice() for _ in range(lane._MAX_RUNTIME_DEVICES + 1)]
    monkeypatch.setattr(lane.jax, "devices", lambda: oversized)
    with pytest.raises(RuntimeError, match="device inventory is out of bounds"):
        lane._runtime_identity()
    assert calls == 0


@pytest.mark.parametrize(
    ("fixed", "off"),
    [
        ("fixed_td0", "intentional_td0_off"),
        ("fixed_trace", "intentional_trace_off"),
        ("fixed_q_lambda", "intentional_q_lambda_off"),
    ],
)
def test_mechanism_off_reduces_bit_exactly_to_fixed_consumer(
    fixed: str, off: str,
) -> None:
    expected = _control(fixed, seed=lane.SEEDS[0], horizon=48, phase_length=12)
    actual = _control(off, seed=lane.SEEDS[0], horizon=48, phase_length=12)
    assert actual["arm"] == off
    assert actual["execution_arm"] == fixed
    for key in ("trajectory", "final_state", "metrics"):
        assert actual[key] == expected[key]
    expected_resources = dict(expected["resources"])
    actual_resources = dict(actual["resources"])
    expected_resources.pop("timing_telemetry_ns")
    actual_resources.pop("timing_telemetry_ns")
    assert actual_resources == expected_resources


@pytest.mark.parametrize("arm", lane.CONTROL_ARMS)
def test_each_control_arm_runs_end_to_end_with_exact_resources(arm: str) -> None:
    record = _control(arm, seed=lane.SEEDS[1], horizon=48, phase_length=12)
    assert lane.validate_control_shard(record) == record
    assert len(record["trajectory"]["rewards"]) == 48
    assert record["resources"]["environment_steps"] == 48
    assert record["resources"]["observations"] == 48
    assert record["resources"]["updates"] == 48
    assert record["resources"]["model_queries"] == 96
    assert record["resources"]["timing_is_selection_metric"] is False
    assert type(record["resources"]["timing_telemetry_ns"]) is int
    assert record["policy"]["scientific_promotion_allowed"] is False


@pytest.mark.parametrize(
    ("arm", "expected_bytes"),
    [
        ("fixed_td0", 24),
        ("intentional_td0", 44),
        ("fixed_trace", 24),
        ("intentional_trace", 44),
        ("fixed_q_lambda", 40),
        ("intentional_q_lambda", 68),
    ],
)
def test_control_persistent_numeric_bytes_are_exact(
    arm: str, expected_bytes: int
) -> None:
    record = _control(arm, seed=lane.SEEDS[0], horizon=8, phase_length=2)
    assert record["resources"]["persistent_numeric_bytes"] == expected_bytes


def test_prediction_and_control_information_and_rng_are_explicit() -> None:
    prediction = _control(
        "intentional_trace", seed=lane.SEEDS[2], horizon=16, phase_length=4
    )
    control = _control(
        "intentional_q_lambda", seed=lane.SEEDS[2], horizon=16, phase_length=4
    )
    assert prediction["resources"]["action_queries"] == 0
    assert prediction["resources"]["rng_fold_ins"] == 17
    assert prediction["resources"]["rng_integer_draws"] == 17
    assert control["resources"]["action_queries"] == 0
    assert control["resources"]["rng_fold_ins"] == 17
    assert control["resources"]["rng_splits"] == 0
    assert control["resources"]["rng_uniform_draws"] == 0
    assert control["resources"]["rng_integer_draws"] == 17
    assert control["identity"]["behavior_rng_impl"] == "threefry2x32"
    assert prediction["final_state"]["behavior_rng_root"]
    assert prediction["information"]["boundary_information"] == []
    assert prediction["information"]["task_information"] == []


@pytest.mark.parametrize(
    ("fixed", "candidate"),
    [
        ("fixed_td0", "intentional_td0"),
        ("fixed_trace", "intentional_trace"),
        ("fixed_q_lambda", "intentional_q_lambda"),
    ],
)
def test_pairs_share_one_exogenous_behavior_trajectory(
    fixed: str, candidate: str
) -> None:
    control = _control(fixed, seed=lane.SEEDS[0], horizon=64, phase_length=8)
    intentional = _control(candidate, seed=lane.SEEDS[0], horizon=64, phase_length=8)
    for field in ("states", "actions", "rewards"):
        assert control["trajectory"][field] == intentional["trajectory"][field]


def test_test_seed_roster_produces_distinct_behavior_schedules() -> None:
    schedules = {
        tuple(_control("fixed_td0", seed=seed, horizon=64)["trajectory"]["actions"])
        for seed in lane.SEEDS
    }
    assert len(schedules) == len(lane.SEEDS)


def test_behavior_trajectory_is_invariant_to_jax_x64_mode() -> None:
    expected = _control("fixed_td0", seed=lane.SEEDS[0], horizon=64, phase_length=8)
    with jax.enable_x64():
        actual = _control("fixed_td0", seed=lane.SEEDS[0], horizon=64, phase_length=8)
    assert actual["trajectory"] == expected["trajectory"]


def test_validator_rejects_nested_subclasses_without_hooks() -> None:
    record = _control(
        "intentional_trace", seed=lane.SEEDS[3], horizon=16, phase_length=4
    )

    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("must not iterate")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("must not compare")

    hostile = copy.deepcopy(record)
    hostile["trajectory"] = HostileDict(hostile["trajectory"])
    with pytest.raises(ValueError, match="exact JSON"):
        lane.validate_control_shard(hostile)
    assert HostileDict.calls == 0


def test_validator_rejects_resource_result_identity_and_policy_forgery() -> None:
    record = _control("intentional_td0", seed=lane.SEEDS[0], horizon=16, phase_length=4)
    for path, replacement in (
        (("resources", "updates"), 15),
        (("trajectory", "rewards"), [99.0] * 16),
        (("identity", "source_sha256"), {"forged": "0" * 64}),
        (("policy", "scientific_promotion_allowed"), True),
    ):
        hostile = copy.deepcopy(record)
        hostile[path[0]][path[1]] = replacement
        with pytest.raises(ValueError):
            lane.validate_control_shard(hostile)


def test_campaign_execution_is_closed_before_independent_review() -> None:
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.run_control_shard("fixed_td0", seed=lane.SEEDS[0])
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.run_campaign(Path("unused.npz"), Path("unused.json"))


def test_runtime_flag_alone_cannot_bypass_reviewed_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("reservation occurred without reviewed transition")

    monkeypatch.setattr(lane, "_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(lane, "_reserve", forbidden)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.run_campaign(Path("unused.npz"), lane.OUTPUT_PATH)
    assert calls == 0


def test_execution_cli_fails_before_dataset_or_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"dataset": 0, "consumer": 0}

    def forbidden_dataset(*args: object, **kwargs: object) -> object:
        calls["dataset"] += 1
        raise AssertionError("dataset load occurred before authorization")

    def forbidden_consumer(*args: object, **kwargs: object) -> object:
        calls["consumer"] += 1
        raise AssertionError("consumer ran before authorization")

    monkeypatch.setattr(lane, "_load_dataset", forbidden_dataset)
    monkeypatch.setattr(lane, "_run", forbidden_consumer)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.main(["--dataset", "unused.npz"])
    assert calls == {"dataset": 0, "consumer": 0}


def test_validator_bounds_config_before_reexecution(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _control("fixed_td0", seed=lane.SEEDS[0], horizon=16, phase_length=4)
    hostile = copy.deepcopy(record)
    hostile["config"]["horizon"] = 1 << 40
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("validator reexecuted before exact bounds")

    monkeypatch.setattr(lane, "_run", forbidden)
    with pytest.raises(ValueError, match="horizon"):
        lane.validate_control_shard(hostile)
    assert calls == 0


def _synthetic_supervised_records() -> list[dict[str, object]]:
    records = []
    for seed in lane.SEEDS:
        for arm, offset in (
            ("intentional_updates_off", 0.0),
            ("intentional_updates_ipmnist", 0.01),
        ):
            spec = screening_spec(arm)
            result = ScreeningRunResult(
                config_name=arm,
                base_learner=spec.base_learner,
                hyperparameters=spec.hyperparameters,
                seed=seed,
                config=lane.SUPERVISED_CONFIG,
                per_task_accuracy=np.full(8, 0.5 + offset, dtype=np.float64),
                per_task_loss=np.full(8, 0.7 - offset, dtype=np.float64),
                per_task_plasticity=np.full(8, 0.4 + offset, dtype=np.float64),
                wall_clock_seconds=0.125,
            )
            records.append(intentional_updates_development_record(result))
    return records


@pytest.fixture(scope="module")
def complete_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    supervised = _synthetic_supervised_records()
    control = [
        _control(arm, seed=seed)
        for seed in lane.SEEDS
        for arm in lane.CONTROL_ARMS
    ]
    return supervised, control


def test_report_recomputes_all_four_bonferroni_paired_questions(
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    assert set(report["paired_comparisons"]) == {
        "supervised_ipmnist", "td0", "trace", "q_lambda"
    }
    assert all(
        item["outcome"] in {"supported", "rejected", "inconclusive"}
        for item in report["paired_comparisons"].values()
    )
    expected_q_deltas = []
    for run in report["runs"]:
        controls = {record["arm"]: record for record in run["control"]}
        fixed = controls["fixed_q_lambda"]["metrics"]
        intentional = controls["intentional_q_lambda"]["metrics"]
        assert fixed["mean_reward"] == intentional["mean_reward"]
        expected_q_deltas.append(
            fixed["mean_squared_td_error"] - intentional["mean_squared_td_error"]
        )
    assert report["paired_comparisons"]["q_lambda"]["deltas"] == expected_q_deltas
    assert lane.validate_report(report, require_current_source=True) == report


def test_report_rejects_runtime_identity_drift(
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    runtime = {"backend": "cpu", "dependency_lock": "a" * 64}
    monkeypatch.setattr(lane, "_current_runtime", lambda: runtime)
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    monkeypatch.setattr(
        lane,
        "_current_runtime",
        lambda: {"backend": "cpu", "dependency_lock": "b" * 64},
    )
    with pytest.raises(ValueError, match="runtime environment"):
        lane.validate_report(report, require_current_source=True)


def test_report_rejects_missing_shard_arithmetic_and_promotion(
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    missing = copy.deepcopy(report)
    missing["runs"][0]["control"].pop()
    with pytest.raises(ValueError, match="complete"):
        lane.validate_report(missing, require_current_source=True)
    forged = copy.deepcopy(report)
    forged["paired_comparisons"]["td0"]["mean_delta"] = 9.0
    with pytest.raises(ValueError, match="paired arithmetic"):
        lane.validate_report(forged, require_current_source=True)
    promoting = copy.deepcopy(report)
    promoting["policy"]["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        lane.validate_report(promoting, require_current_source=True)


def test_report_publication_is_no_replace_and_rejects_symlink_parent(
    tmp_path: Path,
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    destination = tmp_path / "new" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    assert lane._publish_report_authorized(
        destination, report, _capability=lane._TEST_EXECUTION_CAPABILITY
    ) == destination
    with pytest.raises(FileExistsError):
        lane._publish_report_authorized(
            destination, report, _capability=lane._TEST_EXECUTION_CAPABILITY
        )

    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    linked_destination = linked / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", linked_destination)
    with pytest.raises(OSError):
        lane._publish_report_authorized(
            linked_destination, report, _capability=lane._TEST_EXECUTION_CAPABILITY
        )
    assert not (target / "report.json").exists()


def test_public_publisher_is_closed_without_creating_output(
    tmp_path: Path,
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    destination = tmp_path / "never" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.publish_report(destination, report)
    assert not destination.parent.exists()


def test_reservation_is_exclusive_and_parent_swap_stays_descriptor_pinned(
    tmp_path: Path,
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    requested = tmp_path / "requested"
    destination = requested / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    reservation = lane._reserve(destination)
    try:
        with pytest.raises(FileExistsError):
            lane._reserve(destination)
        moved = tmp_path / "moved"
        requested.rename(moved)
        requested.mkdir()
        lane._publish_reserved(reservation, report)
    finally:
        lane._release(reservation)
    assert (moved / "report.json").is_file()
    assert not destination.exists()


def test_post_link_validation_failure_removes_only_the_published_inode(
    tmp_path: Path,
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    destination = tmp_path / "failed" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    reservation = lane._reserve(destination)
    original = lane.validate_report
    calls = 0

    def fail_after_publish(value: object, *, require_current_source: bool = True) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected post-link validation failure")
        return original(value, require_current_source=require_current_source)

    monkeypatch.setattr(lane, "validate_report", fail_after_publish)
    try:
        with pytest.raises(ValueError, match="injected post-link"):
            lane._publish_reserved(reservation, report)
    finally:
        lane._release(reservation)
    assert not destination.exists()


def test_extra_hard_link_is_rejected_and_rolled_back(
    tmp_path: Path,
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    destination = tmp_path / "links" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    reservation = lane._reserve(destination)
    original_unlink = lane.os.unlink
    skipped = False

    def leave_first_temporary_link(
        path: str, *, dir_fd: int | None = None
    ) -> None:
        nonlocal skipped
        if not skipped and path.startswith(".report.json.") and path.endswith(".tmp"):
            skipped = True
            return
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(lane.os, "unlink", leave_first_temporary_link)
    try:
        with pytest.raises(ValueError, match="unique regular-file link"):
            lane._publish_reserved(reservation, report)
    finally:
        lane._release(reservation)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))


def test_strict_reread_converts_deep_json_recursion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "deep" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    reservation = lane._reserve(destination)
    raw = ("[" * 10_000 + "0" + "]" * 10_000).encode("ascii")
    destination.write_bytes(raw)
    metadata = destination.stat()
    try:
        with pytest.raises(ValueError, match="not strict JSON"):
            lane._strict_reread(
                reservation,
                raw,
                expected_identity=(metadata.st_dev, metadata.st_ino),
            )
    finally:
        destination.unlink()
        lane._release(reservation)


def test_campaign_failure_after_dispatch_retains_consumed_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "consumed" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane, "CAMPAIGN_SEEDS", lane.TEST_ONLY_SEEDS)
    monkeypatch.setattr(lane, "SEEDS", lane.TEST_ONLY_SEEDS)
    monkeypatch.setattr(lane, "_REVIEWED_EXECUTION_TRANSITION", True)
    monkeypatch.setattr(lane, "_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    monkeypatch.setattr(
        lane,
        "_load_dataset",
        lambda _path: (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        lane,
        "_screening_dataset_provenance",
        lambda _inputs, _labels: lane.frozen_plan()["dataset"],
    )

    def fail_first_consumer(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("injected consumer failure")

    monkeypatch.setattr(lane, "run_screening_config", fail_first_consumer)
    with pytest.raises(RuntimeError, match="injected consumer failure"):
        lane.run_campaign(Path("unused.npz"), destination)
    marker = destination.parent / ".report.json.reservation"
    assert marker.read_text(encoding="ascii") == (
        "reserved:report.json; retained as consumed-without-result after dispatch\n"
    )
    assert not destination.exists()
    with pytest.raises(FileExistsError):
        lane.run_campaign(Path("unused.npz"), destination)


@pytest.mark.parametrize(
    ("horizon", "phase_length"),
    [(0, 1), (10_001, 1), (8, 0), (8, 9), (True, 1)],
)
def test_control_bounds_fail_before_execution(horizon: int, phase_length: int) -> None:
    with pytest.raises(ValueError):
        lane._run_control_shard_authorized(
            "fixed_td0",
            seed=lane.SEEDS[0],
            horizon=horizon,
            phase_length=phase_length,
            _capability=lane._TEST_EXECUTION_CAPABILITY,
        )
