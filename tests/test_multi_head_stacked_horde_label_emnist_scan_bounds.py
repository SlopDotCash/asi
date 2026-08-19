"""Tests for scan sequence bounds and array dimension validation.

Covers MultiHeadLearner, StackedHorde, and Label-permuted EMNIST.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.upgd_label_emnist import (
    LabelEMNISTConfig,
    LabelEMNISTRunResult,
    build_schedule,
    run_label_emnist,
)
from alberta_framework.core.multi_head_learner import (
    BatchedMultiHeadResult,
    MultiHeadLearningResult,
    MultiHeadMLPLearner,
    run_multi_head_learning_loop,
    run_multi_head_learning_loop_batched,
)
from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    StackedHordeState,
    StackedLinearHorde,
    run_stacked_horde_scan,
)


class _HostileArray:
    """Object mimicking array interface with hostile behavior."""

    @property
    def shape(self) -> tuple[int, ...]:
        return (10, 4)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> object:
        return jnp.float32


class TestMultiHeadLearnerScanBounds:
    def test_run_multi_head_learning_loop_rejects_non_learner(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        state = learner.init(feature_dim=3, key=jr.key(0))
        obs = jnp.ones((10, 3), dtype=jnp.float32)
        tgt = jnp.ones((10, 2), dtype=jnp.float32)

        with pytest.raises(TypeError, match="learner must be an actual MultiHeadMLPLearner"):
            run_multi_head_learning_loop(
                "not_a_learner",  # type: ignore[arg-type]
                state,
                obs,
                tgt,
            )

    def test_run_multi_head_learning_loop_rejects_non_state(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        obs = jnp.ones((10, 3), dtype=jnp.float32)
        tgt = jnp.ones((10, 2), dtype=jnp.float32)

        with pytest.raises(TypeError, match="state must be an actual MultiHeadMLPState"):
            run_multi_head_learning_loop(
                learner,
                "not_a_state",  # type: ignore[arg-type]
                obs,
                tgt,
            )

    def test_run_multi_head_learning_loop_rejects_untrusted_arrays(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        state = learner.init(feature_dim=4, key=jr.key(0))
        tgt = jnp.ones((10, 2), dtype=jnp.float32)
        obs = jnp.ones((10, 4), dtype=jnp.float32)

        with pytest.raises(TypeError, match="observations must be a trusted array"):
            run_multi_head_learning_loop(learner, state, _HostileArray(), tgt)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="targets must be a trusted array"):
            run_multi_head_learning_loop(learner, state, obs, _HostileArray())  # type: ignore[arg-type]

    def test_run_multi_head_learning_loop_rejects_invalid_ranks(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        state = learner.init(feature_dim=3, key=jr.key(0))

        with pytest.raises(ValueError, match="observations must be 2-dimensional"):
            run_multi_head_learning_loop(
                learner,
                state,
                jnp.ones((10,), dtype=jnp.float32),
                jnp.ones((10, 2), dtype=jnp.float32),
            )

        with pytest.raises(ValueError, match="targets must be 2-dimensional"):
            run_multi_head_learning_loop(
                learner,
                state,
                jnp.ones((10, 3), dtype=jnp.float32),
                jnp.ones((10,), dtype=jnp.float32),
            )

    def test_run_multi_head_learning_loop_rejects_dimension_mismatches(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        state = learner.init(feature_dim=3, key=jr.key(0))

        with pytest.raises(ValueError, match="targets step count .* must match observations"):
            run_multi_head_learning_loop(
                learner,
                state,
                jnp.ones((10, 3), dtype=jnp.float32),
                jnp.ones((5, 2), dtype=jnp.float32),
            )

        with pytest.raises(ValueError, match="targets head count .* must match learner.n_heads"):
            run_multi_head_learning_loop(
                learner,
                state,
                jnp.ones((10, 3), dtype=jnp.float32),
                jnp.ones((10, 4), dtype=jnp.float32),
            )

    def test_run_multi_head_learning_loop_rejects_empty_steps(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        state = learner.init(feature_dim=3, key=jr.key(0))

        with pytest.raises(
            ValueError, match="observations must contain between 1 and signed-int32 steps"
        ):
            run_multi_head_learning_loop(
                learner,
                state,
                jnp.ones((0, 3), dtype=jnp.float32),
                jnp.ones((0, 2), dtype=jnp.float32),
            )

    def test_run_multi_head_learning_loop_batched_validations(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        obs = jnp.ones((10, 3), dtype=jnp.float32)
        tgt = jnp.ones((10, 2), dtype=jnp.float32)
        keys = jr.split(jr.key(0), 4)

        with pytest.raises(TypeError, match="learner must be an actual MultiHeadMLPLearner"):
            run_multi_head_learning_loop_batched(
                "not_a_learner",  # type: ignore[arg-type]
                obs,
                tgt,
                keys,
            )

        with pytest.raises(TypeError, match="keys must be a trusted array"):
            run_multi_head_learning_loop_batched(
                learner, obs, tgt, _HostileArray()  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError, match="keys must be rank-1 or rank-2 array"):
            run_multi_head_learning_loop_batched(
                learner,
                obs,
                tgt,
                jnp.ones((2, 2, 2), dtype=jnp.uint32),
            )

        with pytest.raises(ValueError, match="keys must contain between 1 and signed-int32 seeds"):
            run_multi_head_learning_loop_batched(
                learner,
                obs,
                tgt,
                jnp.ones((0, 2), dtype=jnp.uint32),
            )

    def test_run_multi_head_learning_loop_success(self) -> None:
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), sparsity=0.0)
        state = learner.init(feature_dim=3, key=jr.key(0))
        obs = jnp.ones((5, 3), dtype=jnp.float32)
        tgt = jnp.ones((5, 2), dtype=jnp.float32)

        res = run_multi_head_learning_loop(learner, state, obs, tgt)
        assert isinstance(res, MultiHeadLearningResult)
        assert res.per_head_metrics.shape == (5, 2, 3)

        keys = jr.split(jr.key(0), 3)
        batched_res = run_multi_head_learning_loop_batched(learner, obs, tgt, keys)
        assert isinstance(batched_res, BatchedMultiHeadResult)
        assert batched_res.per_head_metrics.shape == (3, 5, 2, 3)


class TestStackedHordeScanBounds:
    def test_run_stacked_horde_scan_rejects_non_components(self) -> None:
        cfg = StackedHordeConfig(
            n_demons=2,
            feature_dim=3,
            gammas=(0.9, 0.9),
            lamdas=(0.5, 0.5),
            cumulant_indices=(0, 1),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        features = jnp.ones((5, 3), dtype=jnp.float32)
        sources = jnp.ones((5, 2), dtype=jnp.float32)

        with pytest.raises(TypeError, match="horde must be an actual StackedLinearHorde"):
            run_stacked_horde_scan("not_a_horde", state, features, sources)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="state must be an actual StackedHordeState"):
            run_stacked_horde_scan(horde, "not_a_state", features, sources)  # type: ignore[arg-type]

    def test_run_stacked_horde_scan_rejects_untrusted_arrays(self) -> None:
        cfg = StackedHordeConfig(
            n_demons=2,
            feature_dim=4,
            gammas=(0.9, 0.9),
            lamdas=(0.5, 0.5),
            cumulant_indices=(0, 1),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        features = jnp.ones((5, 4), dtype=jnp.float32)
        sources = jnp.ones((5, 2), dtype=jnp.float32)

        with pytest.raises(TypeError, match="features must be a trusted array"):
            run_stacked_horde_scan(horde, state, _HostileArray(), sources)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="cumulant_sources must be a trusted array"):
            run_stacked_horde_scan(horde, state, features, _HostileArray())  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="rhos must be a trusted array"):
            run_stacked_horde_scan(
                horde,
                state,
                features,
                sources,
                rhos=_HostileArray(),  # type: ignore[arg-type]
            )

    def test_run_stacked_horde_scan_rejects_insufficient_steps(self) -> None:
        cfg = StackedHordeConfig(
            n_demons=2,
            feature_dim=3,
            gammas=(0.9, 0.9),
            lamdas=(0.5, 0.5),
            cumulant_indices=(0, 1),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        with pytest.raises(
            ValueError, match="features must contain between 2 and signed-int32 steps"
        ):
            run_stacked_horde_scan(
                horde,
                state,
                jnp.ones((1, 3), dtype=jnp.float32),
                jnp.ones((1, 2), dtype=jnp.float32),
            )

        with pytest.raises(
            ValueError, match="features must contain between 2 and signed-int32 steps"
        ):
            run_stacked_horde_scan(
                horde,
                state,
                jnp.ones((0, 3), dtype=jnp.float32),
                jnp.ones((0, 2), dtype=jnp.float32),
            )

    def test_run_stacked_horde_scan_rejects_mismatched_rows(self) -> None:
        cfg = StackedHordeConfig(
            n_demons=2,
            feature_dim=3,
            gammas=(0.9, 0.9),
            lamdas=(0.5, 0.5),
            cumulant_indices=(0, 1),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        with pytest.raises(
            ValueError, match="features and cumulant_sources must have the same number of rows"
        ):
            run_stacked_horde_scan(
                horde,
                state,
                jnp.ones((5, 3), dtype=jnp.float32),
                jnp.ones((3, 2), dtype=jnp.float32),
            )

    def test_run_stacked_horde_scan_rejects_invalid_rhos_shape(self) -> None:
        cfg = StackedHordeConfig(
            n_demons=2,
            feature_dim=3,
            gammas=(0.9, 0.9),
            lamdas=(0.5, 0.5),
            cumulant_indices=(0, 1),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        with pytest.raises(ValueError, match=r"rhos must have shape \(num_steps,\)"):
            run_stacked_horde_scan(
                horde,
                state,
                jnp.ones((5, 3), dtype=jnp.float32),
                jnp.ones((5, 2), dtype=jnp.float32),
                rhos=jnp.ones((3,), dtype=jnp.float32),
            )

    def test_run_stacked_horde_scan_success(self) -> None:
        cfg = StackedHordeConfig(
            n_demons=2,
            feature_dim=3,
            gammas=(0.9, 0.9),
            lamdas=(0.5, 0.5),
            cumulant_indices=(0, 1),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        features = jnp.ones((5, 3), dtype=jnp.float32)
        sources = jnp.ones((5, 2), dtype=jnp.float32)

        final_state, td_errors = run_stacked_horde_scan(horde, state, features, sources)
        assert isinstance(final_state, StackedHordeState)
        assert td_errors.shape == (4, 2)


class TestLabelEMNISTScanBounds:
    def test_build_schedule_rejects_non_config(self) -> None:
        with pytest.raises(TypeError, match="config must be an exact LabelEMNISTConfig"):
            build_schedule(jr.key(0), "not_a_config", 100)  # type: ignore[arg-type]

    def test_build_schedule_rejects_non_int_n_train(self) -> None:
        config = LabelEMNISTConfig(n_tasks=2, task_length=4, n_classes=3)
        with pytest.raises(TypeError, match="n_train must be an exact built-in int"):
            build_schedule(jr.key(0), config, "100")  # type: ignore[arg-type]

    def test_build_schedule_rejects_invalid_n_train(self) -> None:
        config = LabelEMNISTConfig(n_tasks=2, task_length=4, n_classes=3)
        with pytest.raises(ValueError, match="n_train must be positive and at most signed-int32"):
            build_schedule(jr.key(0), config, 0)

        with pytest.raises(ValueError, match="n_train=2 is smaller than task_length=4"):
            build_schedule(jr.key(0), config, 2)

    def test_run_label_emnist_rejects_invalid_config_type(self) -> None:
        data_x = np.ones((50, 6), dtype=np.float32)
        data_y = np.zeros((50,), dtype=np.int32)
        with pytest.raises(TypeError, match="config must be an exact LabelEMNISTConfig or None"):
            run_label_emnist(
                data_x,
                data_y,
                "upgd_w",
                [0],
                config="not_a_config",  # type: ignore[arg-type]
            )

    def test_run_label_emnist_rejects_invalid_return_per_step(self) -> None:
        data_x = np.ones((50, 6), dtype=np.float32)
        data_y = np.zeros((50,), dtype=np.int32)
        with pytest.raises(TypeError, match="return_per_step must be a boolean"):
            run_label_emnist(
                data_x,
                data_y,
                "upgd_w",
                [0],
                return_per_step="yes",  # type: ignore[arg-type]
            )

    def test_run_label_emnist_valid_execution(self) -> None:
        config = LabelEMNISTConfig(
            n_tasks=2,
            task_length=4,
            input_dim=6,
            hidden1=4,
            hidden2=4,
            n_classes=3,
        )
        rng = np.random.default_rng(0)
        data_x = rng.standard_normal((20, 6)).astype(np.float32)
        data_y = rng.integers(0, 3, size=20).astype(np.int32)

        res = run_label_emnist(
            data_x,
            data_y,
            "upgd_w",
            [0],
            config=config,
            return_per_step=True,
        )
        assert isinstance(res, LabelEMNISTRunResult)
        assert res.per_task_accuracy.shape == (1, 2)
