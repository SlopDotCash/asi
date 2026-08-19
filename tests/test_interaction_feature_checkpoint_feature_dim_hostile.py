"""Hostile ``feature_dim`` identity containment at the checkpoint boundary.

``save_interaction_feature_checkpoint`` and ``load_interaction_feature_checkpoint``
gated ``feature_dim`` with ``isinstance(feature_dim, bool) or not
isinstance(feature_dim, int) or feature_dim < 1``. ``isinstance`` accepts
subclasses, so a hostile ``int`` subclass overriding ``__lt__``/``__eq__``/
``__index__`` passed the gate and its overridden dunder then ran during the
"trusted" ``feature_dim < 1`` comparison — before the value's type was
confirmed safe. This is the same spoofable-identity shape already closed at
other numeric gates across this codebase (e.g. ``_require_int32`` itself,
used everywhere else in this module).

Both sites now route through the already-hardened ``_require_int32`` helper,
which rejects on an exact-type membership check (``type(value) not in
_ACTUAL_INT_TYPES``) before ever calling ``operator.index`` or comparing the
value, so a hostile subclass's dunders never run.
"""

from __future__ import annotations

from pathlib import Path

import jax.random as jr
import pytest

import alberta_framework.core.interaction_features as interaction_features
from alberta_framework.core.interaction_features import (
    INTERACTION_FEATURE_CHECKPOINT_SCHEMA,
    FixedBudgetInteractionLearner,
    load_interaction_feature_checkpoint,
    save_interaction_feature_checkpoint,
)

pytestmark = pytest.mark.unit


class _HostileInt(int):
    """An ``int`` subclass whose comparison/index dunders must never run."""

    calls = 0

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile __lt__ ran")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile __eq__ ran")

    def __index__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile __index__ ran")


def _learner() -> FixedBudgetInteractionLearner:
    return FixedBudgetInteractionLearner(
        n_features=1,
        n_tasks=1,
        candidate_count=0,
        replacement_interval=0,
        min_feature_age=0,
    )


def test_save_checkpoint_rejects_hostile_feature_dim_before_comparison(
    tmp_path: Path,
) -> None:
    learner = _learner()
    state = learner.init(2, jr.key(0))
    hostile = _HostileInt(2)
    _HostileInt.calls = 0

    with pytest.raises(ValueError, match="feature_dim"):
        save_interaction_feature_checkpoint(
            learner,
            state,
            tmp_path / "hostile_ckpt",
            feature_dim=hostile,  # type: ignore[arg-type]
        )
    assert _HostileInt.calls == 0
    assert not (tmp_path / "hostile_ckpt").exists()


def test_save_checkpoint_rejects_bool_and_nonpositive_feature_dim(
    tmp_path: Path,
) -> None:
    learner = _learner()
    state = learner.init(2, jr.key(0))

    with pytest.raises(ValueError, match="feature_dim"):
        save_interaction_feature_checkpoint(
            learner,
            state,
            tmp_path / "bool_ckpt",
            feature_dim=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="feature_dim"):
        save_interaction_feature_checkpoint(
            learner, state, tmp_path / "zero_ckpt", feature_dim=0
        )


def test_save_checkpoint_accepts_genuine_positive_feature_dim(
    tmp_path: Path,
) -> None:
    learner = _learner()
    state = learner.init(2, jr.key(0))

    save_interaction_feature_checkpoint(
        learner, state, tmp_path / "good_ckpt", feature_dim=2
    )
    loaded_learner, loaded_state = load_interaction_feature_checkpoint(tmp_path / "good_ckpt")
    assert loaded_learner.to_config() == learner.to_config()


def test_load_checkpoint_rejects_hostile_feature_dim_before_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = _learner()
    state = learner.init(2, jr.key(0))
    hostile = _HostileInt(2)
    _HostileInt.calls = 0
    fake_metadata = {
        "schema": INTERACTION_FEATURE_CHECKPOINT_SCHEMA,
        "learner_config": learner.to_config(),
        "feature_dim": hostile,
        "memory_accounting": learner.memory_accounting(state),
    }
    monkeypatch.setattr(
        interaction_features, "load_checkpoint_metadata", lambda path: fake_metadata
    )

    with pytest.raises(ValueError, match="feature_dim"):
        load_interaction_feature_checkpoint(tmp_path / "does-not-exist")
    assert _HostileInt.calls == 0


def test_load_checkpoint_rejects_bool_and_nonpositive_feature_dim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = _learner()
    state = learner.init(2, jr.key(0))

    for bad_feature_dim in (True, 0, -1):
        fake_metadata = {
            "schema": INTERACTION_FEATURE_CHECKPOINT_SCHEMA,
            "learner_config": learner.to_config(),
            "feature_dim": bad_feature_dim,
            "memory_accounting": learner.memory_accounting(state),
        }
        monkeypatch.setattr(
            interaction_features, "load_checkpoint_metadata", lambda path: fake_metadata
        )
        with pytest.raises(ValueError, match="feature_dim"):
            load_interaction_feature_checkpoint(tmp_path / "does-not-exist")
