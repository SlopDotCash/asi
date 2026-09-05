from __future__ import annotations

import dataclasses
import json

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks.native_supervised_suite import (
    ARM_IDS,
    BENCHMARK_IDS,
    FROZEN_SEEDS,
    build_task_stream,
    catalog_payload,
    main,
    run_native_suite,
    validate_result,
)

pytestmark = pytest.mark.integration


def _fixture(n_classes: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(n_classes, dtype=np.int32), 3)
    values = np.arange(labels.size * int(np.prod(shape)), dtype=np.float32)
    images = (values.reshape((labels.size, *shape)) % 17) / 17.0
    return images, labels


@pytest.mark.parametrize("benchmark_id", BENCHMARK_IDS)
def test_all_catalog_lanes_construct_and_run_end_to_end(benchmark_id: str) -> None:
    n_classes = 100 if benchmark_id == "split_cifar100" else 10
    shape = (4, 4, 3) if n_classes == 100 else (4, 4)
    images, labels = _fixture(n_classes, shape)
    result = run_native_suite(
        benchmark_id, images, labels, seed=FROZEN_SEEDS[0], examples_per_task=2,
        replay_capacity=3,
    )
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    assert result.development_only and not result.scientific_promotion_allowed
    assert result.negative_results_must_be_retained and not result.task_information_used_by_learner
    assert len(result.dataset_sha256) == 64
    assert len(result.schedule_sha256) == 64
    assert len(result.source_sha256) == 64
    assert len(result.runtime_identity) == 4
    for arm in result.arms:
        assert arm.receipt.data_steps == len(arm.task_accuracies) * 2
        assert arm.receipt.data_bytes_read > 0
        assert arm.receipt.persistent_bytes > 0


def test_stream_is_deterministic_and_task_information_is_not_in_examples() -> None:
    images, labels = _fixture(10, (4, 4))
    first = build_task_stream(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[1], examples_per_task=2
    )
    second = build_task_stream(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[1], examples_per_task=2
    )
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.inputs, right.inputs)
        np.testing.assert_array_equal(left.labels, right.labels)
        assert left.inputs.shape[1] == 16


def test_result_binds_dataset_schedule_source_and_runtime_identities() -> None:
    images, labels = _fixture(10, (4, 4))
    first = run_native_suite(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[0], examples_per_task=2
    )
    replay = run_native_suite(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[0], examples_per_task=2
    )
    other_seed = run_native_suite(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[1], examples_per_task=2
    )
    changed_images = images.copy()
    changed_images[0, 0, 0] += np.float32(0.25)
    other_data = run_native_suite(
        "split_mnist", changed_images, labels, seed=FROZEN_SEEDS[0], examples_per_task=2
    )
    assert first.dataset_sha256 == replay.dataset_sha256 == other_seed.dataset_sha256
    assert first.schedule_sha256 == replay.schedule_sha256
    assert first.schedule_sha256 != other_seed.schedule_sha256
    assert first.dataset_sha256 != other_data.dataset_sha256
    with pytest.raises(ValueError, match="source identity"):
        validate_result(dataclasses.replace(first, source_sha256="0" * 64))
    with pytest.raises(ValueError, match="runtime identity"):
        validate_result(
            dataclasses.replace(first, runtime_identity=("forged", *first.runtime_identity[1:]))
        )


def test_replay_and_mechanism_off_have_exact_causal_receipts() -> None:
    images, labels = _fixture(10, (4, 4))
    result = run_native_suite(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[2], examples_per_task=2,
        replay_capacity=3,
    )
    online, replay, centroid, frozen = result.arms
    assert centroid.receipt.parameter_updates == 10
    assert online.receipt.parameter_updates == 19
    assert replay.receipt.replay_inserts == 10
    assert replay.receipt.replay_samples == 9
    assert replay.receipt.parameter_updates == 19
    assert online.receipt.model_queries == replay.receipt.model_queries
    assert frozen.receipt.parameter_updates == 0
    assert frozen.receipt.replay_samples == 0


def test_jit_and_eager_sgd_paths_match_end_to_end_except_timing() -> None:
    images, labels = _fixture(10, (4, 4))
    compiled = run_native_suite(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[3], examples_per_task=1,
        replay_capacity=2,
    )
    with jax.disable_jit():
        eager = run_native_suite(
            "split_mnist", images, labels, seed=FROZEN_SEEDS[3], examples_per_task=1,
            replay_capacity=2,
        )
    for left, right in zip(compiled.arms, eager.arms, strict=True):
        assert left.online_accuracy == right.online_accuracy
        assert left.task_accuracies == right.task_accuracies
        assert dataclasses.replace(left.receipt, elapsed_ns=0) == dataclasses.replace(
            right.receipt, elapsed_ns=0
        )


def test_validator_rejects_promotion_and_counter_forgery() -> None:
    images, labels = _fixture(10, (4, 4))
    result = run_native_suite(
        "rotated_mnist", images, labels, seed=FROZEN_SEEDS[0], examples_per_task=1
    )
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_result(dataclasses.replace(result, scientific_promotion_allowed=True))
    forged_receipt = dataclasses.replace(result.arms[0].receipt, data_steps=1)
    forged = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=forged_receipt), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="data receipt"):
        validate_result(forged)
    with pytest.raises(ValueError, match="exact SuiteResult"):
        validate_result(dataclasses.asdict(result))


def test_validator_rejects_forged_headline_accuracy() -> None:
    """The headline must reconstruct from the retained per-task accuracies."""
    images, labels = _fixture(10, (4, 4))
    result = run_native_suite(
        "split_mnist", images, labels, seed=FROZEN_SEEDS[0], examples_per_task=2
    )
    original = result.arms[0]
    forged_accuracy = 0.0 if original.online_accuracy != 0.0 else 1.0
    forged = dataclasses.replace(
        result,
        arms=(
            dataclasses.replace(original, online_accuracy=forged_accuracy),
            *result.arms[1:],
        ),
    )

    with pytest.raises(ValueError, match="accuracy record mismatch"):
        validate_result(forged)


@pytest.mark.parametrize(
    ("seed", "count"), [(True, 1), (FROZEN_SEEDS[0], True), (0, 1), (FROZEN_SEEDS[0], 65)]
)
def test_hostile_axes_are_rejected(seed: object, count: object) -> None:
    images, labels = _fixture(10, (4, 4))
    with pytest.raises(ValueError):
        build_task_stream("split_mnist", images, labels, seed=seed, examples_per_task=count)


def test_catalog_cli_is_metadata_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--catalog"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == catalog_payload()
    assert payload["avalanche_revision"].endswith("eb075be393e1f458b2c352514ff6c17b5a2c0f4e")
