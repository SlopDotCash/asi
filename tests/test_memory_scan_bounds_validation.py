# mypy: disable-error-code="call-arg"
"""Tests for scan sequence bounds and array dimension validation across memory modules.

Covers:
- AssociativeMemoryLearner.run_associative_memory_arrays
- PrototypeMemoryLearner.run_prototype_memory_arrays
- UPGDMemoryLearner.run_upgd_memory_arrays
- WorkingMemoryFeaturizer.transform_working_memory_arrays
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    AssociativeMemoryState,
    run_associative_memory_arrays,
)
from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    run_prototype_memory_arrays,
)
from alberta_framework.core.upgd_memory import (
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    UPGDMemoryState,
    run_upgd_memory_arrays,
)
from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    transform_working_memory_arrays,
)


class _HostileUntrusted:
    """Object lacking trusted array attributes."""

    pass


class TestAssociativeMemoryScanBounds:
    def _make_learner_and_state(self) -> tuple[AssociativeMemoryLearner, AssociativeMemoryState]:
        cfg = AssociativeMemoryConfig(
            vocab_size=4,
            block_size=8,
            suffix_length=2,
            max_features=16,
        )
        learner = AssociativeMemoryLearner(cfg)
        state = learner.init()
        return learner, state

    def test_run_associative_memory_arrays_rejects_non_learner(self) -> None:
        learner, state = self._make_learner_and_state()
        contexts = jnp.zeros((10, 8), dtype=jnp.int32)
        labels = jnp.zeros((10,), dtype=jnp.int32)
        with pytest.raises(TypeError, match="learner must be an actual AssociativeMemoryLearner"):
            run_associative_memory_arrays("invalid", state, contexts, labels)  # type: ignore[arg-type]

    def test_run_associative_memory_arrays_rejects_non_state(self) -> None:
        learner, _ = self._make_learner_and_state()
        contexts = jnp.zeros((10, 8), dtype=jnp.int32)
        labels = jnp.zeros((10,), dtype=jnp.int32)
        with pytest.raises(TypeError, match="state must be an actual AssociativeMemoryState"):
            run_associative_memory_arrays(learner, "invalid", contexts, labels)  # type: ignore[arg-type]

    def test_run_associative_memory_arrays_rejects_untrusted_arrays(self) -> None:
        learner, state = self._make_learner_and_state()
        contexts = jnp.zeros((10, 8), dtype=jnp.int32)
        labels = jnp.zeros((10,), dtype=jnp.int32)
        with pytest.raises(TypeError, match="contexts must be a trusted array"):
            run_associative_memory_arrays(learner, state, _HostileUntrusted(), labels)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="labels must be a trusted array"):
            run_associative_memory_arrays(learner, state, contexts, _HostileUntrusted())  # type: ignore[arg-type]

    def test_run_associative_memory_arrays_rejects_invalid_ranks(self) -> None:
        learner, state = self._make_learner_and_state()
        with pytest.raises(ValueError, match="contexts must be 2-dimensional"):
            run_associative_memory_arrays(
                learner,
                state,
                jnp.zeros((10, 8, 1), dtype=jnp.int32),
                jnp.zeros((10,), dtype=jnp.int32),
            )
        with pytest.raises(ValueError, match="labels must be 1-dimensional"):
            run_associative_memory_arrays(
                learner,
                state,
                jnp.zeros((10, 8), dtype=jnp.int32),
                jnp.zeros((10, 1), dtype=jnp.int32),
            )

    def test_run_associative_memory_arrays_rejects_empty_steps(self) -> None:
        learner, state = self._make_learner_and_state()
        with pytest.raises(ValueError, match="associative memory step count must be between 1"):
            run_associative_memory_arrays(
                learner,
                state,
                jnp.zeros((0, 8), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
            )

    def test_run_associative_memory_arrays_rejects_dimension_mismatches(self) -> None:
        learner, state = self._make_learner_and_state()
        with pytest.raises(ValueError, match="contexts block_size .* must match config block_size"):
            run_associative_memory_arrays(
                learner,
                state,
                jnp.zeros((10, 4), dtype=jnp.int32),
                jnp.zeros((10,), dtype=jnp.int32),
            )
        with pytest.raises(ValueError, match="labels step count .* must match contexts step count"):
            run_associative_memory_arrays(
                learner,
                state,
                jnp.zeros((10, 8), dtype=jnp.int32),
                jnp.zeros((5,), dtype=jnp.int32),
            )

    def test_run_associative_memory_arrays_happy_path(self) -> None:
        learner, state = self._make_learner_and_state()
        contexts = jnp.zeros((10, 8), dtype=jnp.int32)
        labels = jnp.zeros((10,), dtype=jnp.int32)
        result = run_associative_memory_arrays(learner, state, contexts, labels)
        assert result.predictions.shape == (10, 4)
        assert result.metrics.shape == (10, 8)
        assert result.updates_applied.shape == (10,)


class TestPrototypeMemoryScanBounds:
    def _make_learner(self) -> PrototypeMemoryLearner:
        cfg = PrototypeMemoryConfig(
            feature_dim=4,
            n_classes=3,
            slots_per_class=2,
        )
        return PrototypeMemoryLearner(cfg)

    def test_run_prototype_memory_arrays_rejects_non_learner(self) -> None:
        obs = jnp.zeros((10, 4), dtype=jnp.float32)
        tgt = jnp.zeros((10, 3), dtype=jnp.float32)
        with pytest.raises(TypeError, match="learner must be an actual PrototypeMemoryLearner"):
            run_prototype_memory_arrays("invalid", obs, tgt)  # type: ignore[arg-type]

    def test_run_prototype_memory_arrays_rejects_non_state(self) -> None:
        learner = self._make_learner()
        obs = jnp.zeros((10, 4), dtype=jnp.float32)
        tgt = jnp.zeros((10, 3), dtype=jnp.float32)
        with pytest.raises(TypeError, match="state must be an actual PrototypeMemoryState"):
            run_prototype_memory_arrays(learner, obs, tgt, state="invalid")  # type: ignore[arg-type]

    def test_run_prototype_memory_arrays_rejects_untrusted_arrays(self) -> None:
        learner = self._make_learner()
        obs = jnp.zeros((10, 4), dtype=jnp.float32)
        tgt = jnp.zeros((10, 3), dtype=jnp.float32)
        with pytest.raises(TypeError, match="observations must be a trusted array"):
            run_prototype_memory_arrays(learner, _HostileUntrusted(), tgt)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="targets must be a trusted array"):
            run_prototype_memory_arrays(learner, obs, _HostileUntrusted())  # type: ignore[arg-type]

    def test_run_prototype_memory_arrays_rejects_invalid_ranks(self) -> None:
        learner = self._make_learner()
        with pytest.raises(ValueError, match="observations must be 2-dimensional"):
            run_prototype_memory_arrays(
                learner,
                jnp.zeros((10,), dtype=jnp.float32),
                jnp.zeros((10, 3), dtype=jnp.float32),
            )
        with pytest.raises(ValueError, match="targets must be 2-dimensional"):
            run_prototype_memory_arrays(
                learner,
                jnp.zeros((10, 4), dtype=jnp.float32),
                jnp.zeros((10, 3, 1), dtype=jnp.float32),
            )

    def test_run_prototype_memory_arrays_rejects_empty_steps(self) -> None:
        learner = self._make_learner()
        with pytest.raises(ValueError, match="prototype memory step count must be between 1"):
            run_prototype_memory_arrays(
                learner,
                jnp.zeros((0, 4), dtype=jnp.float32),
                jnp.zeros((0, 3), dtype=jnp.float32),
            )

    def test_run_prototype_memory_arrays_rejects_dimension_mismatches(self) -> None:
        learner = self._make_learner()
        with pytest.raises(ValueError, match="observations feature_dim .* must match config"):
            run_prototype_memory_arrays(
                learner,
                jnp.zeros((10, 8), dtype=jnp.float32),
                jnp.zeros((10, 3), dtype=jnp.float32),
            )
        with pytest.raises(ValueError, match="same step count"):
            run_prototype_memory_arrays(
                learner,
                jnp.zeros((10, 4), dtype=jnp.float32),
                jnp.zeros((5, 3), dtype=jnp.float32),
            )
        with pytest.raises(ValueError, match="targets n_classes .* must match config"):
            run_prototype_memory_arrays(
                learner,
                jnp.zeros((10, 4), dtype=jnp.float32),
                jnp.zeros((10, 2), dtype=jnp.float32),
            )

    def test_run_prototype_memory_arrays_happy_path(self) -> None:
        learner = self._make_learner()
        obs = jnp.zeros((10, 4), dtype=jnp.float32)
        tgt = jnp.zeros((10, 3), dtype=jnp.float32)
        result = run_prototype_memory_arrays(learner, obs, tgt)
        assert result.predictions.shape == (10, 3)
        assert result.metrics.shape == (10, 6)
        assert result.updates_applied.shape == (10,)


class TestUPGDMemoryScanBounds:
    def _make_learner_and_state(self) -> tuple[UPGDMemoryLearner, UPGDMemoryState]:
        cfg = UPGDMemoryConfig(
            feature_dim=4,
            n_heads=2,
            hidden_sizes=(8,),
            slots_per_class=2,
        )
        learner = UPGDMemoryLearner(cfg)
        state = learner.init(key=jr.key(0))
        return learner, state

    def test_run_upgd_memory_arrays_rejects_non_learner(self) -> None:
        learner, state = self._make_learner_and_state()
        obs = jnp.zeros((10, 4), dtype=jnp.float32)
        tgt = jnp.zeros((10, 2), dtype=jnp.float32)
        with pytest.raises(TypeError, match="learner must be an actual UPGDMemoryLearner"):
            run_upgd_memory_arrays("invalid", state, obs, tgt)  # type: ignore[arg-type]

    def test_run_upgd_memory_arrays_rejects_non_state(self) -> None:
        learner, _ = self._make_learner_and_state()
        obs = jnp.zeros((10, 4), dtype=jnp.float32)
        tgt = jnp.zeros((10, 2), dtype=jnp.float32)
        with pytest.raises(TypeError, match="state must be an actual UPGDMemoryState"):
            run_upgd_memory_arrays(learner, "invalid", obs, tgt)  # type: ignore[arg-type]

    def test_run_upgd_memory_arrays_rejects_empty_steps(self) -> None:
        learner, state = self._make_learner_and_state()
        with pytest.raises(ValueError, match="steps must be positive"):
            run_upgd_memory_arrays(
                learner,
                state,
                jnp.zeros((0, 4), dtype=jnp.float32),
                jnp.zeros((0, 2), dtype=jnp.float32),
            )

    def test_run_upgd_memory_arrays_rejects_dimension_mismatches(self) -> None:
        learner, state = self._make_learner_and_state()
        with pytest.raises(ValueError, match="observations must have shape"):
            run_upgd_memory_arrays(
                learner,
                state,
                jnp.zeros((10, 6), dtype=jnp.float32),
                jnp.zeros((10, 2), dtype=jnp.float32),
            )
        with pytest.raises(ValueError, match="targets must have shape"):
            run_upgd_memory_arrays(
                learner,
                state,
                jnp.zeros((10, 4), dtype=jnp.float32),
                jnp.zeros((5, 2), dtype=jnp.float32),
            )
        with pytest.raises(ValueError, match="targets must have shape"):
            run_upgd_memory_arrays(
                learner,
                state,
                jnp.zeros((10, 4), dtype=jnp.float32),
                jnp.zeros((10, 3), dtype=jnp.float32),
            )

    def test_run_upgd_memory_arrays_happy_path(self) -> None:
        learner, state = self._make_learner_and_state()
        obs = jnp.zeros((10, 4), dtype=jnp.float32)
        tgt = jnp.zeros((10, 2), dtype=jnp.float32)
        result = run_upgd_memory_arrays(learner, state, obs, tgt)
        assert result.predictions.shape == (10, 2)
        assert result.metrics.shape == (10, 10)
        assert result.updates_applied.shape == (10,)


class TestWorkingMemoryScanBounds:
    def _make_featurizer(self) -> WorkingMemoryFeaturizer:
        cfg = WorkingMemoryConfig(
            observation_dim=3,
            action_dim=2,
            reward_dim=1,
            observation_decay_rates=(0.5,),
            action_decay_rates=(0.5,),
            reward_decay_rates=(0.5,),
        )
        return WorkingMemoryFeaturizer(cfg)

    def test_transform_working_memory_arrays_rejects_non_featurizer(self) -> None:
        obs = jnp.zeros((10, 3), dtype=jnp.float32)
        act = jnp.zeros((10, 2), dtype=jnp.float32)
        rew = jnp.zeros((10, 1), dtype=jnp.float32)
        with pytest.raises(TypeError, match="featurizer must be an actual WorkingMemoryFeaturizer"):
            transform_working_memory_arrays("invalid", obs, act, rew)  # type: ignore[arg-type]

    def test_transform_working_memory_arrays_rejects_non_state(self) -> None:
        featurizer = self._make_featurizer()
        obs = jnp.zeros((10, 3), dtype=jnp.float32)
        act = jnp.zeros((10, 2), dtype=jnp.float32)
        rew = jnp.zeros((10, 1), dtype=jnp.float32)
        with pytest.raises(TypeError, match="state must be an actual WorkingMemoryState"):
            transform_working_memory_arrays(featurizer, obs, act, rew, state="invalid")  # type: ignore[arg-type]

    def test_transform_working_memory_arrays_rejects_untrusted_arrays(self) -> None:
        featurizer = self._make_featurizer()
        obs = jnp.zeros((10, 3), dtype=jnp.float32)
        act = jnp.zeros((10, 2), dtype=jnp.float32)
        rew = jnp.zeros((10, 1), dtype=jnp.float32)
        with pytest.raises(TypeError, match="observations must be a trusted array"):
            transform_working_memory_arrays(featurizer, _HostileUntrusted(), act, rew)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="actions must be a trusted array"):
            transform_working_memory_arrays(featurizer, obs, _HostileUntrusted(), rew)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="rewards must be a trusted array"):
            transform_working_memory_arrays(featurizer, obs, act, _HostileUntrusted())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="external_gates must be a trusted array"):
            transform_working_memory_arrays(
                featurizer,
                obs,
                act,
                rew,
                external_gates=_HostileUntrusted(),  # type: ignore[arg-type]
            )

    def test_transform_working_memory_arrays_rejects_invalid_ranks(self) -> None:
        featurizer = self._make_featurizer()
        obs = jnp.zeros((10, 3), dtype=jnp.float32)
        act = jnp.zeros((10, 2), dtype=jnp.float32)
        rew = jnp.zeros((10, 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations must be 2-dimensional"):
            transform_working_memory_arrays(
                featurizer, jnp.zeros((10,), dtype=jnp.float32), act, rew
            )
        with pytest.raises(ValueError, match="actions must be 2-dimensional"):
            transform_working_memory_arrays(
                featurizer, obs, jnp.zeros((10,), dtype=jnp.float32), rew
            )
        with pytest.raises(ValueError, match="rewards must be 2-dimensional"):
            transform_working_memory_arrays(
                featurizer, obs, act, jnp.zeros((10,), dtype=jnp.float32)
            )
        with pytest.raises(ValueError, match="external_gates must be 1-dimensional"):
            transform_working_memory_arrays(
                featurizer,
                obs,
                act,
                rew,
                external_gates=jnp.zeros((10, 1), dtype=jnp.float32),
            )

    def test_transform_working_memory_arrays_rejects_empty_steps(self) -> None:
        featurizer = self._make_featurizer()
        with pytest.raises(ValueError, match="transform step count must be between 1"):
            transform_working_memory_arrays(
                featurizer,
                jnp.zeros((0, 3), dtype=jnp.float32),
                jnp.zeros((0, 2), dtype=jnp.float32),
                jnp.zeros((0, 1), dtype=jnp.float32),
            )

    def test_transform_working_memory_arrays_rejects_dimension_mismatches(self) -> None:
        featurizer = self._make_featurizer()
        obs = jnp.zeros((10, 3), dtype=jnp.float32)
        act = jnp.zeros((10, 2), dtype=jnp.float32)
        rew = jnp.zeros((10, 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations have an invalid shape"):
            transform_working_memory_arrays(
                featurizer,
                jnp.zeros((10, 4), dtype=jnp.float32),
                act,
                rew,
            )
        with pytest.raises(ValueError, match="actions have an invalid shape"):
            transform_working_memory_arrays(
                featurizer,
                obs,
                jnp.zeros((5, 2), dtype=jnp.float32),
                rew,
            )
        with pytest.raises(ValueError, match="rewards have an invalid shape"):
            transform_working_memory_arrays(
                featurizer,
                obs,
                act,
                jnp.zeros((5, 1), dtype=jnp.float32),
            )
        with pytest.raises(ValueError, match="external_gates have an invalid shape"):
            transform_working_memory_arrays(
                featurizer,
                obs,
                act,
                rew,
                external_gates=jnp.zeros((5,), dtype=jnp.float32),
            )

    def test_transform_working_memory_arrays_happy_path(self) -> None:
        featurizer = self._make_featurizer()
        obs = jnp.zeros((10, 3), dtype=jnp.float32)
        act = jnp.zeros((10, 2), dtype=jnp.float32)
        rew = jnp.zeros((10, 1), dtype=jnp.float32)
        gates = jnp.ones((10,), dtype=jnp.float32)
        result = transform_working_memory_arrays(
            featurizer, obs, act, rew, external_gates=gates
        )
        assert result.features.shape == (10, featurizer.config.feature_dim())
        assert result.updates_applied.shape == (10,)
