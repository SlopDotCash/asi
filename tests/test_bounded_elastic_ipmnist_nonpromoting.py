"""Unit coverage for alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting.

Tests the registered-arm hyperparameter surface, the resource-expectation
math (growth/pruning branches, byte budgets), and the exact validation
helpers.
"""

import pytest

from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    _int,
    _parameter_count,
    _strings,
    bounded_elastic_resource_expectations,
    registered_bounded_elastic_hyperparameters,
)


def test_registered_hyperparameters_valid() -> None:
    hp = registered_bounded_elastic_hyperparameters("bounded_elastic")
    assert isinstance(hp, dict)
    assert hp["growth_enabled"] == 1.0
    assert hp["pruning_enabled"] == 1.0
    # Mutating the returned copy must not affect the registry.
    hp["step_size"] = 999.0
    again = registered_bounded_elastic_hyperparameters("bounded_elastic")
    assert again["step_size"] != 999.0


def test_registered_hyperparameters_rejects_bad() -> None:
    with pytest.raises(ValueError, match="arm must be one of"):
        registered_bounded_elastic_hyperparameters("bogus")
    with pytest.raises(ValueError, match="arm must be one of"):
        registered_bounded_elastic_hyperparameters(123)


def test_parameter_count_formula() -> None:
    # input_dim*active1 + active1 + active1*hidden2 + hidden2 + hidden2*classes + classes
    assert _parameter_count(2, 3, 4, 5, active1=3) == (
        2 * 3 + 3 + 3 * 4 + 4 + 4 * 5 + 5
    )


def test_resource_expectations_bounded_elastic() -> None:
    res = bounded_elastic_resource_expectations(
        arm="bounded_elastic", n_tasks=10, input_dim=16, hidden1=32, hidden2=16, n_classes=10
    )
    assert res["units_grown"] == 10
    assert res["units_pruned"] == 10
    assert res["structure_events"] == 10
    assert res["peak_active_hidden1_units"] == 16  # initial_active = 32*0.5
    assert res["persistent_bytes"] > 0


def test_resource_expectations_bounded_growth() -> None:
    res = bounded_elastic_resource_expectations(
        arm="bounded_growth", n_tasks=10, input_dim=16, hidden1=32, hidden2=16, n_classes=10
    )
    # initial_active=16, peak = min(32, 16+10) = 26
    assert res["peak_active_hidden1_units"] == 26
    assert res["units_grown"] == 10
    assert res["units_pruned"] == 0


def test_resource_expectations_structure_off() -> None:
    res = bounded_elastic_resource_expectations(
        arm="bounded_structure_off", n_tasks=10, input_dim=16, hidden1=32, hidden2=16, n_classes=10
    )
    assert res["units_grown"] == 0
    assert res["units_pruned"] == 0
    assert res["structure_events"] == 0


def test_resource_expectations_rejects_bad_arm() -> None:
    with pytest.raises(ValueError, match="arm must be one of"):
        bounded_elastic_resource_expectations(
            arm="bogus", n_tasks=1, input_dim=1, hidden1=1, hidden2=1, n_classes=1
        )


def test_int_validation() -> None:
    assert _int(5, context="x", positive=True) == 5
    with pytest.raises(ValueError, match="positive"):
        _int(0, context="x", positive=True)
    with pytest.raises(ValueError, match="nonnegative"):
        _int(-1, context="x")


def test_strings_validation() -> None:
    assert _strings(["a", "b"], context="x") == ("a", "b")
    with pytest.raises(ValueError, match="duplicates"):
        _strings(["a", "a"], context="x")
    with pytest.raises(ValueError, match="non-empty"):
        _strings([""], context="x")
