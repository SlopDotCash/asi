"""Executable acceptance tests for the recurring multi-agent benchmark."""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from alberta_framework.evaluation.continual_multiagent import (
    AcceptanceThresholds,
    ContinualMultiAgentConfig,
    aggregate_evidence,
    evaluate_acceptance,
    paired_bootstrap_mean_interval,
    run_continual_multiagent_benchmark,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    artifact_json,
    build_evidence_artifact,
    load_evidence_artifact,
    scientific_content_sha256,
    validate_evidence_artifact,
    write_evidence_artifact,
)
from alberta_framework.evaluation.continual_multiagent_cli import (
    main as evidence_cli_main,
)

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def benchmark_report():
    """Run the promoted paired thirty-seed experiment once per test module."""

    return run_continual_multiagent_benchmark()


@pytest.fixture(scope="module")
def smoke_report():
    """Run the explicitly underpowered smoke schedule once per test module."""

    return run_continual_multiagent_benchmark(seeds=(0, 1, 2))


def test_uninterrupted_seeded_benchmark_records_required_evidence(
    benchmark_report,
) -> None:
    report = benchmark_report
    assert len(report.condition_results) == 90
    assert report.aggregate.seeds == tuple(range(30, 60))
    assert report.aggregate.joint_adaptive_phase_rewards.shape == (3,)
    assert report.aggregate.joint_adaptive_performance_matrix.shape == (3, 2)
    assert report.aggregate.action_scalars_per_step == 2
    assert report.aggregate.state_scalars == 10
    assert report.aggregate.state_bytes == 48
    assert report.aggregate.budgets_identical
    assert report.aggregate.all_values_finite

    for result in report.condition_results:
        assert result.online_rewards.shape == (3 * report.config.phase_steps,)
        assert result.phase_mean_rewards.shape == (3,)
        assert result.performance_matrix.shape == (3, 2)
        assert result.recovery_lengths.shape == (2,)
        assert np.all(np.isfinite(result.online_rewards))
        assert np.all((0.0 <= result.online_rewards) & (result.online_rewards <= 1.0))
        assert result.timing.wall_seconds > 0.0
        assert result.timing.mean_update_latency_ms >= 0.0


def test_default_acceptance_passes_only_from_numeric_multi_seed_evidence(
    benchmark_report,
) -> None:
    report = benchmark_report
    assert report.acceptance.passed, report.acceptance.failures
    assert not report.acceptance.failures
    assert len(report.aggregate.seeds) >= report.thresholds.minimum_seed_count
    assert report.aggregate.reward_uplift_over_frozen >= 0.15
    assert report.aggregate.partner_uplift >= 0.20
    assert report.aggregate.reward_uplift_interval.lower >= 0.15
    assert report.aggregate.partner_uplift_interval.lower >= 0.20
    assert report.aggregate.recurrent_a_probe_reward >= 0.90
    assert report.aggregate.mean_interference_forgetting <= 0.01
    assert (
        report.aggregate.recurrence_recovery_fraction
        >= report.thresholds.minimum_recurrence_recovery_fraction
    )


def test_three_seed_smoke_cannot_be_promoted_as_scientific_evidence(
    smoke_report,
) -> None:
    assert not smoke_report.acceptance.passed
    seed_check = next(
        check
        for check in smoke_report.acceptance.checks
        if check.name == "seed_count"
    )
    assert not seed_check.passed
    assert seed_check.actual == 3.0
    assert seed_check.threshold == 30.0
    schedule_check = next(
        check
        for check in smoke_report.acceptance.checks
        if check.name == "evidence_seed_schedule"
    )
    assert not schedule_check.passed


def test_seeded_learning_evidence_is_exactly_reproducible(
    benchmark_report,
) -> None:
    evidence_seed = benchmark_report.aggregate.seeds[0]
    repeated = run_continual_multiagent_benchmark(seeds=(evidence_seed,))
    original = tuple(
        result
        for result in benchmark_report.condition_results
        if result.seed == evidence_seed
    )

    assert len(original) == len(repeated.condition_results) == 3
    for first, second in zip(original, repeated.condition_results, strict=True):
        assert first.condition == second.condition
        np.testing.assert_array_equal(first.online_rewards, second.online_rewards)
        np.testing.assert_array_equal(
            first.performance_matrix,
            second.performance_matrix,
        )
        np.testing.assert_array_equal(
            first.phase_mean_rewards,
            second.phase_mean_rewards,
        )
    repeated_aggregate = aggregate_evidence(
        benchmark_report.condition_results,
        confidence_level=benchmark_report.config.confidence_level,
        bootstrap_resamples=benchmark_report.config.bootstrap_resamples,
        bootstrap_seed=benchmark_report.config.bootstrap_seed,
    )
    assert (
        benchmark_report.aggregate.reward_uplift_interval
        == repeated_aggregate.reward_uplift_interval
    )
    assert (
        benchmark_report.aggregate.partner_uplift_interval
        == repeated_aggregate.partner_uplift_interval
    )


def test_acceptance_fails_closed_and_returns_all_failure_evidence(
    benchmark_report,
) -> None:
    impossible = AcceptanceThresholds(
        minimum_reward_uplift_over_frozen=1.0,
        minimum_partner_uplift=1.0,
        minimum_recurrent_a_probe_reward=1.1,
        maximum_mean_forgetting=-1.0,
        maximum_interference_forgetting=-1.0,
        minimum_recurrence_recovery_fraction=1.1,
        maximum_mean_recurrence_recovery_steps=-1.0,
        maximum_mean_stability_gap=-1.0,
        maximum_update_latency_ms=-1.0,
    )
    decision = evaluate_acceptance(benchmark_report.aggregate, impossible)

    assert not decision.passed
    assert len(decision.failures) == 9
    names = {failure.name for failure in decision.failures}
    assert "reward_uplift_over_frozen" in names
    assert "partner_uplift" in names
    assert "mean_forgetting" in names
    assert all(failure.detail for failure in decision.failures)
    assert all(np.isfinite(failure.actual) for failure in decision.failures)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_steps": 1},
        {"learning_rate": 0.0},
        {"exploration_rate": 1.1},
        {"probe_horizon": 65},
        {"probe_tail_steps": 0},
        {"recovery_window": 65},
        {"bootstrap_resamples": 999},
        {"confidence_level": 1.0},
        {"bootstrap_seed": -1},
    ],
)
def test_invalid_scientific_configuration_is_rejected(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        ContinualMultiAgentConfig(**kwargs)


def test_seed_validation_rejects_non_reproducible_schedules() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_continual_multiagent_benchmark(seeds=())
    with pytest.raises(ValueError, match="unique"):
        run_continual_multiagent_benchmark(seeds=(1, 1))
    with pytest.raises(ValueError, match="lie in"):
        run_continual_multiagent_benchmark(seeds=(-1,))


def test_paired_bootstrap_resamples_within_seed_differences() -> None:
    """The interval must never independently shuffle the two conditions."""

    differences = np.asarray((0.9, -0.2, 0.4, 0.1), dtype=np.float64)
    confidence_level = 0.80
    resamples = 2_000
    seed = 91
    interval = paired_bootstrap_mean_interval(
        differences,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        differences.size,
        size=(resamples, differences.size),
        dtype=np.int64,
    )
    manual_means = np.mean(differences[indices], axis=1)
    expected = np.quantile(manual_means, (0.10, 0.90))

    assert interval.method == "paired-percentile-bootstrap"
    assert interval.estimate == pytest.approx(float(np.mean(differences)))
    assert interval.lower == pytest.approx(float(expected[0]))
    assert interval.upper == pytest.approx(float(expected[1]))


def test_aggregate_intervals_use_matched_seed_differences(
    benchmark_report,
) -> None:
    prequential = {
        (result.seed, result.condition): result.summary.prequential_performance
        for result in benchmark_report.condition_results
    }
    reward_differences = np.asarray(
        [
            prequential[(seed, "joint_adaptive")]
            - prequential[(seed, "frozen")]
            for seed in benchmark_report.aggregate.seeds
        ],
        dtype=np.float64,
    )
    coadaptation_differences = np.asarray(
        [
            prequential[(seed, "joint_adaptive")]
            - prequential[(seed, "learner_only")]
            for seed in benchmark_report.aggregate.seeds
        ],
        dtype=np.float64,
    )

    assert benchmark_report.aggregate.reward_uplift_interval.estimate == pytest.approx(
        float(np.mean(reward_differences))
    )
    assert benchmark_report.aggregate.partner_uplift_interval.estimate == pytest.approx(
        float(np.mean(coadaptation_differences))
    )


def test_public_aggregation_rejects_duplicate_seed_weighting(
    benchmark_report,
) -> None:
    evidence_seed = benchmark_report.aggregate.seeds[0]
    one_seed = tuple(
        result
        for result in benchmark_report.condition_results
        if result.seed == evidence_seed
    )
    with pytest.raises(ValueError, match="unique"):
        aggregate_evidence(one_seed + one_seed)


def test_artifact_has_deterministic_scientific_content_and_digest(
    benchmark_report,
) -> None:
    first = build_evidence_artifact(benchmark_report)
    second = build_evidence_artifact(benchmark_report)

    assert first["schema_version"] == SCHEMA_VERSION
    assert first["content"] == second["content"]
    assert first["content_digest"] == second["content_digest"]
    digest = first["content_digest"]
    assert isinstance(digest, dict)
    assert digest["sha256"] == scientific_content_sha256(first["content"])
    assert len(digest["sha256"]) == 64

    # Host timing is intentionally inspectable but outside the declared
    # scientific-content digest scope.
    changed_diagnostics = copy.deepcopy(first)
    operational = changed_diagnostics["operational_diagnostics"]
    assert isinstance(operational, dict)
    operational["maximum_update_latency_ms"] = 123_456.0
    assert (
        changed_diagnostics["content_digest"]["sha256"]
        == first["content_digest"]["sha256"]
    )
    validation = validate_evidence_artifact(changed_diagnostics)
    assert not validation.valid
    assert not validation.accepted


def test_artifact_is_strict_json_with_narrow_claim_and_seed_evidence(
    benchmark_report,
    tmp_path,
) -> None:
    path = tmp_path / "continual_multiagent.json"
    artifact = build_evidence_artifact(benchmark_report)
    path.write_text(
        json.dumps(artifact, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    loaded = load_evidence_artifact(path)
    validation = validate_evidence_artifact(loaded)

    assert validation.valid
    assert validation.accepted
    content = loaded["content"]
    assert content["protocol"]["protocol_version"] == PROTOCOL_VERSION
    assert len(content["seed_summaries"]) == 30
    assert (
        "coadaptation_uplift_over_learner_only"
        in content["aggregate"]
    )
    excluded = content["protocol"]["excluded_claims"]
    assert "general feature discovery" in excluded
    assert "intelligence amplification or recommendation intervention" in excluded
    assert content["protocol"]["seed_roles"]["promoted_held_out_evidence"] == list(
        range(30, 60)
    )
    recovery_interval = content["aggregate"][
        "recurrence_recovery_fraction_interval"
    ]
    assert recovery_interval["method"] == "wilson-score"
    assert recovery_interval["sample_size"] == 30
    assert recovery_interval["lower"] < content["aggregate"][
        "recurrence_recovery_fraction"
    ]


def test_artifact_loader_rejects_duplicate_object_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"first","schema_version":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_evidence_artifact(path)


def test_artifact_digest_tampering_fails_closed(benchmark_report) -> None:
    artifact = build_evidence_artifact(benchmark_report)
    tampered = copy.deepcopy(artifact)
    content = tampered["content"]
    assert isinstance(content, dict)
    aggregate = content["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["recurrent_a_probe_reward"] = 0.0

    validation = validate_evidence_artifact(tampered)
    assert not validation.valid
    assert not validation.accepted
    assert "content_digest.sha256 does not match content" in validation.errors


def test_rehashed_source_provenance_tampering_fails_current_source_binding(
    benchmark_report,
) -> None:
    fabricated = copy.deepcopy(build_evidence_artifact(benchmark_report))
    content = fabricated["content"]
    assert isinstance(content, dict)
    provenance = content["source_provenance"]
    assert isinstance(provenance, dict)
    source_hashes = provenance["source_sha256"]
    assert isinstance(source_hashes, dict)
    first_source = next(iter(source_hashes))
    source_hashes[first_source] = "0" * 64
    digest = fabricated["content_digest"]
    assert isinstance(digest, dict)
    digest["sha256"] = scientific_content_sha256(content)

    validation = validate_evidence_artifact(fabricated)

    assert not validation.valid
    assert not validation.accepted
    assert any("current pinned source hashes" in error for error in validation.errors)


def test_rehashed_unknown_scientific_field_fails_schema(
    benchmark_report,
) -> None:
    fabricated = copy.deepcopy(build_evidence_artifact(benchmark_report))
    content = fabricated["content"]
    assert isinstance(content, dict)
    content["unsupported_claim"] = True
    digest = fabricated["content_digest"]
    assert isinstance(digest, dict)
    digest["sha256"] = scientific_content_sha256(content)

    validation = validate_evidence_artifact(fabricated)

    assert not validation.valid
    assert not validation.accepted
    assert "content keys do not match the v1 schema" in validation.errors


def test_rehashed_incomplete_seed_payload_still_fails_closed(
    benchmark_report,
) -> None:
    fabricated = copy.deepcopy(build_evidence_artifact(benchmark_report))
    content = fabricated["content"]
    assert isinstance(content, dict)
    summaries = content["seed_summaries"]
    assert isinstance(summaries, list)
    content["seed_summaries"] = summaries[:1]
    digest = fabricated["content_digest"]
    assert isinstance(digest, dict)
    digest["sha256"] = scientific_content_sha256(content)

    validation = validate_evidence_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any(
        "must contain exactly unique held-out seeds 30-59" in error
        for error in validation.errors
    )
    assert any(
        "aggregate.seeds must match" in error for error in validation.errors
    )


def test_rehashed_interval_with_wrong_sample_size_fails_closed(
    benchmark_report,
) -> None:
    fabricated = copy.deepcopy(build_evidence_artifact(benchmark_report))
    content = fabricated["content"]
    assert isinstance(content, dict)
    aggregate = content["aggregate"]
    assert isinstance(aggregate, dict)
    interval = aggregate["reward_uplift_over_frozen_paired_interval"]
    assert isinstance(interval, dict)
    interval["sample_size"] = 1
    digest = fabricated["content_digest"]
    assert isinstance(digest, dict)
    digest["sha256"] = scientific_content_sha256(content)

    validation = validate_evidence_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("sample_size must be 30" in error for error in validation.errors)


def test_writer_refuses_to_overwrite_an_existing_artifact(
    benchmark_report,
    tmp_path,
) -> None:
    path = tmp_path / "existing.json"
    path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_evidence_artifact(path, benchmark_report)

    assert path.read_text(encoding="utf-8") == "sentinel\n"


def test_writer_allows_only_one_simultaneous_creator(
    benchmark_report,
    tmp_path,
) -> None:
    path = tmp_path / "raced.json"
    barrier = threading.Barrier(2)

    def attempt() -> dict[str, object]:
        barrier.wait()
        return write_evidence_artifact(path, benchmark_report)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt) for _ in range(2)]

    artifacts: list[dict[str, object]] = []
    collisions = 0
    for future in futures:
        try:
            artifacts.append(future.result())
        except FileExistsError:
            collisions += 1

    assert len(artifacts) == 1
    assert collisions == 1
    assert path.read_text(encoding="utf-8") == artifact_json(artifacts[0])


def test_cli_writes_and_verifies_accepted_artifact_without_rerun(
    benchmark_report,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "accepted.json"
    status = evidence_cli_main(
        ["--output", str(path)],
        report=benchmark_report,
    )
    emitted = json.loads(capsys.readouterr().out)

    assert status == 0
    assert emitted["accepted"] is True
    assert emitted["seed_count"] == 30
    assert path.exists()
    assert validate_evidence_artifact(load_evidence_artifact(path)).accepted

    verify_status = evidence_cli_main(["--verify", str(path)])
    verified = json.loads(capsys.readouterr().out)
    assert verify_status == 0
    assert verified["valid"] is True
    assert verified["accepted"] is True


def test_cli_fails_for_underpowered_smoke_without_rerun(
    smoke_report,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "smoke.json"
    status = evidence_cli_main(
        [
            "--seed-start",
            "0",
            "--seed-count",
            "3",
            "--output",
            str(path),
        ],
        report=smoke_report,
    )
    emitted = json.loads(capsys.readouterr().out)

    assert status == 1
    assert emitted["accepted"] is False
    assert emitted["valid"] is False
    artifact = load_evidence_artifact(path)
    assert not validate_evidence_artifact(artifact).valid
    content = artifact["content"]
    assert content["acceptance"]["passed"] is False
    seed_check = next(
        check
        for check in content["acceptance"]["checks"]
        if check["name"] == "seed_count"
    )
    assert seed_check["actual"] == 3.0
    assert seed_check["passed"] is False


def test_cli_rechecks_impossible_thresholds_without_rerun(
    benchmark_report,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "impossible.json"
    status = evidence_cli_main(
        [
            "--output",
            str(path),
            "--minimum-reward-uplift-over-frozen",
            "1.0",
        ],
        report=benchmark_report,
    )
    emitted = json.loads(capsys.readouterr().out)

    assert status == 1
    assert emitted["accepted"] is False
    artifact = load_evidence_artifact(path)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid
    assert not validation.accepted
    content = artifact["content"]
    failed = {
        check["name"]
        for check in content["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "reward_uplift_over_frozen" in failed


def test_cli_rejects_weaker_than_canonical_thresholds_without_rerun(
    benchmark_report,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "weakened.json"
    status = evidence_cli_main(
        [
            "--output",
            str(path),
            "--minimum-reward-uplift-over-frozen",
            "0.0",
        ],
        report=benchmark_report,
    )
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["accepted"] is False
    assert emitted["valid"] is False
    assert not path.exists()
