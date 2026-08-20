"""End-to-end and hostile contracts for the nonpromoting AdamO diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Never

import numpy as np
import pytest

import alberta_framework.benchmarks.adamo_diagnostic as diagnostic
from alberta_framework.benchmarks.adamo_diagnostic import (
    ARMS,
    FROZEN_DEVELOPMENT_SEEDS,
    FROZEN_MATCHED_DEVELOPMENT_SEEDS,
    SCHEMA,
    main,
    run_adamo_diagnostic,
    validate_adamo_diagnostic,
)
from alberta_framework.benchmarks.ipmnist_screening import run_screening_config, screening_spec
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_data() -> tuple[np.ndarray, np.ndarray]:
    inputs = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4)
    labels = np.arange(8, dtype=np.int32) % 2
    return inputs, labels


@pytest.fixture(scope="module")
def receipt(tiny_data: tuple[np.ndarray, np.ndarray]) -> dict[str, object]:
    inputs, labels = tiny_data
    return run_adamo_diagnostic(
        inputs, labels, profile="contract-smoke", seed=FROZEN_DEVELOPMENT_SEEDS[0]
    )


def test_end_to_end_runner_binds_diagnostics_and_exact_resources(
    receipt: dict[str, object],
) -> None:
    assert validate_adamo_diagnostic(receipt) == receipt
    assert receipt["schema"] == SCHEMA
    arms = receipt["arms"]
    assert isinstance(arms, list)
    assert [arm["arm"] for arm in arms] == list(ARMS)
    for arm in arms:
        assert len(arm["post_task_diagnostics"]) == 2
        assert arm["resources"]["observations"] == 8
        assert arm["resources"]["model_queries"] == 18
        assert arm["resources"]["jacobian_reverse_rows"] == 4


def test_inert_reduction_includes_curves_params_state_and_jacobian(
    receipt: dict[str, object],
) -> None:
    control, inert = receipt["arms"][:2]
    for key in (
        "per_task_accuracy",
        "per_task_loss",
        "per_task_plasticity",
        "post_task_diagnostics",
    ):
        assert inert[key] == control[key]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(scientific_promotion_allowed=True), "permanent protocol"),
        (lambda value: value["protocol"].update(learner_boundary_information=["task"]),
         "protocol identity"),
        (lambda value: value["arms"][0]["resources"].update(model_queries=17),
         "accounting mismatch"),
        (lambda value: value["arms"][0]["post_task_diagnostics"][0].update(
            jacobian_mean_singular_value=float("nan")), "finite"),
        (lambda value: value["arms"][1]["post_task_diagnostics"][0].update(
            parameter_sha256="0" * 64), "does not reduce"),
        (lambda value: value.update(runtime={
            "python": "forged", "jax": "forged", "numpy": "forged", "backend": "forged"
        }), "current runtime"),
    ],
)
def test_hostile_receipts_fail_closed(
    receipt: dict[str, object], mutation: object, message: str,
) -> None:
    hostile = copy.deepcopy(receipt)
    mutation(hostile)
    with pytest.raises(ValueError, match=message):
        validate_adamo_diagnostic(hostile)


def test_hostile_scalar_alias_and_unfrozen_seed_are_rejected(
    receipt: dict[str, object], tiny_data: tuple[np.ndarray, np.ndarray],
) -> None:
    hostile = copy.deepcopy(receipt)
    hostile["arms"][0]["resources"]["updates"] = True
    with pytest.raises(ValueError, match="accounting mismatch"):
        validate_adamo_diagnostic(hostile)
    with pytest.raises(ValueError, match="frozen"):
        run_adamo_diagnostic(*tiny_data, profile="contract-smoke", seed=9)


def test_contract_and_matched_seed_schedules_are_disjoint(
    tiny_data: tuple[np.ndarray, np.ndarray],
) -> None:
    assert set(FROZEN_DEVELOPMENT_SEEDS).isdisjoint(FROZEN_MATCHED_DEVELOPMENT_SEEDS)
    with pytest.raises(ValueError, match="frozen"):
        run_adamo_diagnostic(
            *tiny_data,
            profile="contract-smoke",
            seed=FROZEN_MATCHED_DEVELOPMENT_SEEDS[0],
        )
    with pytest.raises(ValueError, match="selected frozen"):
        run_adamo_diagnostic(
            *tiny_data,
            profile="bounded-development",
            seed=FROZEN_MATCHED_DEVELOPMENT_SEEDS[0],
        )


def test_public_function_cannot_consume_reserved_matched_seed(
    tiny_data: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_run(*args: object, **kwargs: object) -> Never:
        nonlocal calls
        calls += 1
        raise AssertionError("reserved matched execution must not start")

    monkeypatch.setattr(diagnostic, "run_screening_config", forbidden_run)
    with pytest.raises(ValueError, match="selected frozen"):
        run_adamo_diagnostic(
            *tiny_data,
            profile="bounded-development",
            seed=FROZEN_MATCHED_DEVELOPMENT_SEEDS[0],
        )
    assert calls == 0


def test_public_bounded_profile_retains_consumed_schedule_compatibility(
    tiny_data: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_schedule(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"profile": kwargs["profile"], "seed": kwargs["seed"]}

    monkeypatch.setattr(diagnostic, "_run_adamo_diagnostic_schedule", capture_schedule)
    result = run_adamo_diagnostic(
        *tiny_data,
        profile="bounded-development",
        seed=FROZEN_DEVELOPMENT_SEEDS[0],
    )
    assert result == {"profile": "bounded-development", "seed": FROZEN_DEVELOPMENT_SEEDS[0]}
    assert captured["seed_schedule"] == FROZEN_DEVELOPMENT_SEEDS


def test_public_cli_cannot_consume_reserved_matched_seed_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_load(path: Path) -> Never:
        nonlocal calls
        calls += 1
        raise AssertionError("reserved matched CLI must not load data")

    monkeypatch.setattr(diagnostic, "_load_dataset", forbidden_load)
    with pytest.raises(SystemExit):
        main(
            [
                "--dataset", str(tmp_path / "unused.npz"),
                "--profile", "bounded-development",
                "--seed", str(FROZEN_MATCHED_DEVELOPMENT_SEEDS[0]),
            ]
        )
    assert calls == 0


def test_private_matched_executor_requires_exact_capability_before_run(
    tiny_data: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_run(*args: object, **kwargs: object) -> Never:
        nonlocal calls
        calls += 1
        raise AssertionError("unauthorized private execution must not start")

    monkeypatch.setattr(diagnostic, "run_screening_config", forbidden_run)
    with pytest.raises(RuntimeError, match="capability"):
        diagnostic._run_matched_adamo_diagnostic(
            *tiny_data,
            profile="bounded-development",
            seed=FROZEN_MATCHED_DEVELOPMENT_SEEDS[0],
            capability=object(),
        )
    assert calls == 0


def test_retained_receipt_can_be_validated_offline(
    receipt: dict[str, object],
) -> None:
    retained = copy.deepcopy(receipt)
    retained["runtime"]["backend"] = "gpu"
    retained["source"]["adapter_sha256"] = "0" * 64
    assert validate_adamo_diagnostic(retained, require_current_identity=False) == retained
    with pytest.raises(ValueError, match="current runtime"):
        validate_adamo_diagnostic(retained)


@pytest.mark.parametrize(
    "path",
    [
        ("frozen_development_seeds",),
        ("config",),
        ("dataset",),
        ("protocol",),
        ("runtime",),
        ("source",),
        ("arms",),
        ("arms", 0),
        ("arms", 0, "hyperparameters"),
        ("arms", 0, "post_task_diagnostics"),
        ("arms", 0, "post_task_diagnostics", 0),
        ("arms", 0, "resources"),
    ],
)
def test_nested_container_subclasses_are_rejected_without_invoking_hooks(
    receipt: dict[str, object], path: tuple[object, ...],
) -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("must not iterate hostile dict")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("must not compare hostile dict")

    class HostileList(list[object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("must not iterate hostile list")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("must not compare hostile list")

    hostile = copy.deepcopy(receipt)
    parent: object = hostile
    for component in path[:-1]:
        parent = parent[component]  # type: ignore[index]
    leaf = parent[path[-1]]  # type: ignore[index]
    replacement = HostileDict(leaf) if type(leaf) is dict else HostileList(leaf)
    parent[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(ValueError, match="exact JSON"):
        validate_adamo_diagnostic(hostile)
    assert HostileDict.calls == 0
    assert HostileList.calls == 0


def test_frozen_seed_schedule_rejects_boolean_member_alias(
    receipt: dict[str, object],
) -> None:
    hostile = copy.deepcopy(receipt)
    hostile["frozen_development_seeds"][0] = True
    with pytest.raises(ValueError, match="frozen seed schedule"):
        validate_adamo_diagnostic(hostile)


def test_runner_observer_is_a_downstream_exact_function_only(
    tiny_data: tuple[np.ndarray, np.ndarray],
) -> None:
    class CallableObject:
        def __call__(self, *_: object) -> None:
            return None

    config = IPMNISTConfig(
        n_tasks=2, task_length=4, input_dim=4, hidden1=3, hidden2=2, n_classes=2
    )
    with pytest.raises(TypeError, match="exact Python function"):
        run_screening_config(
            *tiny_data,
            screening_spec("adamw_control"),
            FROZEN_DEVELOPMENT_SEEDS[0],
            config,
            _task_observer=CallableObject(),  # type: ignore[arg-type]
        )


def test_catalog_cli_is_read_only_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--catalog"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["arms"] == list(ARMS)
    assert set(catalog["profiles"]) == {"contract-smoke", "bounded-development"}
    assert catalog["negative_outcomes_retained"] is True
    assert catalog["scientific_promotion_allowed"] is False


def test_npz_cli_runs_the_current_runner_end_to_end(
    tmp_path: Path, tiny_data: tuple[np.ndarray, np.ndarray],
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "tiny.npz"
    np.savez(dataset, inputs=tiny_data[0], labels=tiny_data[1])
    assert main(["--dataset", str(dataset), "--profile", "contract-smoke"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert validate_adamo_diagnostic(payload) == payload
    assert payload["negative_outcomes_retained"] is True
