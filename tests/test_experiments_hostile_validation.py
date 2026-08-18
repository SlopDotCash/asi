"""Hostile validation for experiments facade."""

import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.utils.experiments import (
    ExperimentConfig,
    SingleRunResult,
    _require_coordinate_hash,
    _require_finite_metric_array,
    _require_hyperparameter_coordinate,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


def test_finite_metric_hostile_without_repr() -> None:
    import numpy as np

    evil = _EvilStr("metric")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_finite_metric_array(np.array([1.0]), evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)


def test_finite_metric_string_subclass_rejected() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="must be an exact string"):
        _require_finite_metric_array(
            np.array([1.0]), _StringSubclass("metric")  # type: ignore[arg-type]
        )


def test_finite_metric_sanitized() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="contains non-finite samples") as exc:
        _require_finite_metric_array(np.array([float("inf")]), "my_metric")
    assert "!r" not in str(exc.value)
    assert "my_metric" in str(exc.value)


def test_hyperparam_coordinate_hostile_name() -> None:
    evil = _EvilStr("bad")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_hyperparameter_coordinate(1.0, name=evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_hyperparam_string_subclass_rejected() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_hyperparameter_coordinate(
            1.0, name=_StringSubclass("bad")  # type: ignore[arg-type]
        )


def test_coordinate_hash_hostile_name() -> None:
    evil = _EvilStr("bad")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_coordinate_hash(1.0, name=evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_coordinate_hash_sanitized() -> None:
    class BadHash:
        def __hash__(self) -> int:
            raise RuntimeError("bad hash")

    with pytest.raises(ValueError, match="cannot be hashed") as exc:
        _require_coordinate_hash(BadHash(), name="good_name")
    assert "!r" not in str(exc.value)
    assert "good_name" in str(exc.value)


def test_multi_seed_rejects_hostile_config_name_before_hash_or_factories() -> None:
    evil = _EvilStr("candidate")
    _EvilStr.calls = 0

    def fail_factory():
        raise AssertionError("factory must not run")

    with pytest.raises(ValueError, match="must be an exact string"):
        ExperimentConfig(evil, fail_factory, fail_factory, 1)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def _legal_config(**overrides: object) -> ExperimentConfig:
    def _learner() -> LinearLearner:
        return LinearLearner()

    payload: dict[str, object] = {
        "name": "fixture",
        "learner_factory": _learner,
        "stream_factory": _learner,
        "num_steps": 2,
    }
    payload.update(overrides)
    return ExperimentConfig(**payload)  # type: ignore[arg-type]


def _legal_run(**overrides: object) -> SingleRunResult:
    payload: dict[str, object] = {
        "config_name": "fixture",
        "seed": 0,
        "metrics_history": [{"squared_error": 0.1}],
        "final_state": LinearLearner().init(2),
    }
    payload.update(overrides)
    return SingleRunResult(**payload)  # type: ignore[arg-type]


def test_experiment_records_accept_canonical_identities() -> None:
    config = _legal_config()
    run = _legal_run()
    assert config.num_steps == 2
    assert run.seed == 0
    assert run.config_name == "fixture"


def test_experiment_records_reject_leftover_integer_identities() -> None:
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        _legal_config(num_steps=True)
    with pytest.raises(ValueError, match="seed"):
        _legal_run(seed=True)


def test_experiment_records_reject_leftover_name_and_host_identities() -> None:
    with pytest.raises(ValueError, match="name must be an exact string"):
        _legal_config(name=True)
    with pytest.raises(ValueError, match="config_name must be an exact string"):
        _legal_run(config_name=True)
    with pytest.raises(TypeError, match="final_state must be an exact LearnerState"):
        _legal_run(final_state=None)
    with pytest.raises(ValueError, match="learner_factory must be callable"):
        _legal_config(learner_factory=None)
    with pytest.raises(ValueError, match="metrics_history must be an exact list"):
        _legal_run(metrics_history={"squared_error": 0.1})
