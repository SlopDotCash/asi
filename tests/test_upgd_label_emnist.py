"""Tests for the Label-permuted EMNIST replication lane.

Covers schedule exactness (task boundaries, cumulative label-permutation
composition, without-replacement sampling), plan/shard/merge accounting, and a
tiny synthetic smoke run. Benchmark executions never happen inside pytest.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.upgd_label_emnist as upgd_label_emnist
from alberta_framework.benchmarks.upgd_label_emnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    SGD_EMA_NORM_HYPERPARAMETERS,
    UPGD_EMA_NORM_HYPERPARAMETERS,
    UPGD_EMA_NORM_SIGMA0_HYPERPARAMETERS,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    LabelEMNISTConfig,
    LabelEMNISTRunResult,
    build_artifact,
    build_comparison,
    build_plan_payload,
    build_schedule,
    load_plan,
    merge_partials,
    partial_payload,
    resolve_hyperparameters,
    run_label_emnist,
    summarize_result,
    task_index_for_step,
)

TINY = LabelEMNISTConfig(
    n_tasks=3, task_length=8, input_dim=6, hidden1=8, hidden2=4, n_classes=5
)

DATASET_META = {
    "source": "synthetic:test",
    "train_rows_used": 50,
    "x_sha256": "0" * 64,
    "y_sha256": "1" * 64,
}


def _tiny_data(n_train: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_train, TINY.input_dim)).astype(np.float32)
    y = rng.integers(0, TINY.n_classes, size=n_train).astype(np.int32)
    return x, y


class TestConfig:
    def test_default_config_matches_selected_publication_shape(self):
        config = LabelEMNISTConfig()
        assert config.n_tasks == 400
        assert config.task_length == 2500
        assert config.n_steps == 1_000_000
        assert (config.input_dim, config.hidden1, config.hidden2) == (784, 300, 150)
        assert config.n_classes == 47
        assert config.matches_selected_publication_configuration

    def test_shrunk_config_does_not_match_selected_publication_shape(self):
        assert not TINY.matches_selected_publication_configuration

    def test_published_hyperparameters(self):
        assert UPGD_W_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 0.01,
            "utility_decay": 0.9,
            "noise_std": 0.001,
            "weight_decay": 0.0,
        }
        assert ADAMW_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 1e-4,
            "beta1": 0.0,
            "beta2": 0.9999,
            "eps": 1e-8,
            "weight_decay": 0.1,
        }

    def test_resolve_hyperparameters_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown hyperparameters"):
            resolve_hyperparameters("upgd_w", {"sigma": 0.1})
        with pytest.raises(ValueError, match="unknown learner"):
            resolve_hyperparameters("sgd")

    @pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.1"])
    def test_resolve_hyperparameters_rejects_nonfinite_or_non_json_numbers(
        self, value: object
    ) -> None:
        with pytest.raises(ValueError, match="hyperparameter 'step_size'"):
            resolve_hyperparameters("upgd_w", {"step_size": value})  # type: ignore[dict-item]

    def test_resolve_hyperparameters_rejects_class_spoofed_number(self) -> None:
        class SpoofedNumber:
            @property
            def __class__(self) -> type[float]:
                return float

            def __float__(self) -> float:
                return 0.1

        with pytest.raises(ValueError, match="hyperparameter 'step_size'"):
            resolve_hyperparameters(  # type: ignore[dict-item]
                "upgd_w", {"step_size": SpoofedNumber()}
            )

    @pytest.mark.parametrize("value", [1e100, 10**400, 1e-50])
    def test_resolve_hyperparameters_rejects_float32_unsafe_values(
        self, value: int | float
    ) -> None:
        with pytest.raises(ValueError, match="hyperparameter 'step_size'"):
            resolve_hyperparameters("upgd_w", {"step_size": value})

    def test_resolve_hyperparameters_rejects_hostile_numeric_subclass(self) -> None:
        class HostileFloat(float):
            def as_integer_ratio(self) -> tuple[int, int]:
                raise RuntimeError("must not run")

        with pytest.raises(ValueError, match="hyperparameter 'step_size'"):
            resolve_hyperparameters(  # type: ignore[dict-item]
                "upgd_w", {"step_size": HostileFloat(0.1)}
            )

    @pytest.mark.parametrize(
        ("learner", "name", "value"),
        [
            ("upgd_w", "step_size", 0.0),
            ("upgd_w", "utility_decay", 1.0),
            ("upgd_w", "noise_std", -0.1),
            ("adamw", "beta1", -0.1),
            ("adamw", "beta2", 1.0),
            ("adamw", "eps", 0.0),
            ("adamw", "weight_decay", -0.1),
            ("upgd_ema_norm", "norm_decay", 1.0),
            ("upgd_ema_norm", "norm_epsilon", 0.0),
            ("sgd_ema_norm", "step_size", 0.0),
            ("sgd_ema_norm", "weight_decay", -0.1),
        ],
    )
    def test_resolve_hyperparameters_enforces_field_domains(
        self, learner: str, name: str, value: float
    ) -> None:
        with pytest.raises(ValueError, match=f"hyperparameter {name!r}"):
            resolve_hyperparameters(learner, {name: value})

    def test_resolve_hyperparameters_accepts_endpoints_and_canonicalizes_ints(self) -> None:
        resolved = resolve_hyperparameters(
            "upgd_ema_norm",
            {
                "utility_decay": 0,
                "noise_std": 0,
                "weight_decay": 1,
                "norm_decay": 0,
            },
        )
        assert all(type(resolved[name]) is float for name in resolved)
        assert resolved["utility_decay"] == 0.0
        assert resolved["norm_decay"] == 0.0
        assert resolved["weight_decay"] == 1.0

    def test_normalized_arm_hyperparameters(self):
        """EMA-norm transfer arms: published EMNIST UPGD-W values + the exact
        screening-lane normalizer settings (norm_decay=0.999, eps=1e-8)."""
        assert UPGD_EMA_NORM_HYPERPARAMETERS == {
            **UPGD_W_PROTOCOL_HYPERPARAMETERS,
            "norm_decay": 0.999,
            "norm_epsilon": 1e-8,
        }
        assert UPGD_EMA_NORM_SIGMA0_HYPERPARAMETERS == {
            **UPGD_EMA_NORM_HYPERPARAMETERS,
            "noise_std": 0.0,
        }
        # Bare-conditioning control: weight decay matched to the published
        # EMNIST UPGD-W decay (0.0 here, unlike the IPMNIST lane's 0.01).
        assert SGD_EMA_NORM_HYPERPARAMETERS == {
            "step_size": 0.01,
            "weight_decay": 0.0,
            "norm_decay": 0.999,
            "norm_epsilon": 1e-8,
        }
        for learner, expected in (
            ("upgd_ema_norm", UPGD_EMA_NORM_HYPERPARAMETERS),
            ("upgd_ema_norm_sigma0", UPGD_EMA_NORM_SIGMA0_HYPERPARAMETERS),
            ("sgd_ema_norm", SGD_EMA_NORM_HYPERPARAMETERS),
        ):
            assert resolve_hyperparameters(learner) == expected
        with pytest.raises(ValueError, match="unknown hyperparameters"):
            resolve_hyperparameters("sgd_ema_norm", {"noise_std": 0.001})


class TestScheduleExactness:
    def test_task_index_changes_exactly_at_multiples_of_task_length(self):
        length = TINY.task_length
        for task in range(TINY.n_tasks):
            assert task_index_for_step(task * length, length) == task
            assert task_index_for_step((task + 1) * length - 1, length) == task

    def test_label_permutations_are_valid_permutations(self):
        schedule = build_schedule(jr.key(0), TINY, n_train=50)
        assert schedule.label_permutations.shape == (TINY.n_tasks, TINY.n_classes)
        expected = np.arange(TINY.n_classes)
        for task in range(TINY.n_tasks):
            row = np.sort(np.asarray(schedule.label_permutations[task]))
            np.testing.assert_array_equal(row, expected)

    def test_label_permutations_compose_cumulatively(self):
        """Row t must equal fresh_t[row_{t-1}] with row_{-1} = identity.

        This pins the upstream ``randperm(47)[targets]`` cumulative mutation
        (the first task itself is permuted), independently recomputing the
        per-task fresh permutations from the documented key derivation.
        """
        config = LabelEMNISTConfig(
            n_tasks=5, task_length=4, input_dim=6, hidden1=8, hidden2=4, n_classes=47
        )
        key = jr.key(7)
        schedule = build_schedule(key, config, n_train=50)
        key_perm, _ = jr.split(key)
        previous = np.arange(config.n_classes)
        for task in range(config.n_tasks):
            fresh = np.asarray(jr.permutation(jr.fold_in(key_perm, task), config.n_classes))
            expected = fresh[previous]
            np.testing.assert_array_equal(
                np.asarray(schedule.label_permutations[task]), expected
            )
            previous = expected

    def test_example_indices_sample_without_replacement(self):
        n_train = 11
        schedule = build_schedule(jr.key(3), TINY, n_train=n_train)
        assert schedule.example_indices.shape == (TINY.n_tasks, TINY.task_length)
        indices = np.asarray(schedule.example_indices)
        assert indices.min() >= 0 and indices.max() < n_train
        for task in range(TINY.n_tasks):
            assert len(set(indices[task].tolist())) == TINY.task_length

    def test_schedule_is_deterministic_per_key(self):
        first = build_schedule(jr.key(9), TINY, n_train=20)
        second = build_schedule(jr.key(9), TINY, n_train=20)
        np.testing.assert_array_equal(
            np.asarray(first.label_permutations), np.asarray(second.label_permutations)
        )
        np.testing.assert_array_equal(
            np.asarray(first.example_indices), np.asarray(second.example_indices)
        )
        third = build_schedule(jr.key(10), TINY, n_train=20)
        assert not np.array_equal(
            np.asarray(first.example_indices), np.asarray(third.example_indices)
        )

    def test_schedule_rejects_dataset_smaller_than_task(self):
        with pytest.raises(ValueError, match="without replacement"):
            build_schedule(jr.key(0), TINY, n_train=TINY.task_length - 1)


class TestSeedBoundary:
    @pytest.mark.parametrize(
        "seeds",
        [
            (),
            (0, 0),
            (True,),
            (np.int64(0),),
            (0.0,),
            (-1,),
            (2**32,),
            (0, 2**32),
        ],
    )
    def test_run_rejects_noncanonical_seed_identities_before_setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seeds: tuple[object, ...],
    ) -> None:
        def unexpected_setup(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid seeds reached learner setup")

        monkeypatch.setattr(upgd_label_emnist, "resolve_hyperparameters", unexpected_setup)
        with pytest.raises(ValueError, match="seeds"):
            run_label_emnist(
                np.empty((1, 1), dtype=np.float32),
                np.empty((1,), dtype=np.int32),
                "adamw",
                seeds=seeds,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("seeds", [(True,), (np.int64(0),), (-1,), (2**32,)])
    def test_plan_rejects_noncanonical_seed_identities(
        self, seeds: tuple[object, ...]
    ) -> None:
        with pytest.raises(ValueError, match="seed IDs"):
            build_plan_payload(
                TINY,
                seeds,  # type: ignore[arg-type]
                DATASET_META,
            )

    def test_plan_cli_rejects_aliased_seed_before_loading_emnist(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected_load(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid CLI seeds reached dataset loading")

        monkeypatch.setattr(
            upgd_label_emnist, "load_emnist_balanced_train", unexpected_load
        )
        with pytest.raises(ValueError, match=r"seed IDs\[1\].*uint32"):
            upgd_label_emnist.main(
                [
                    "plan",
                    "--plan-out",
                    str(tmp_path / "must-not-exist.json"),
                    "--seed-list",
                    f"0,{2**32}",
                ]
            )


class TestEMNISTArrayCache:
    @staticmethod
    def _write_cache(tmp_path, x: np.ndarray, y: np.ndarray, meta_text: str) -> None:
        x_path, y_path, meta_path = upgd_label_emnist._npy_cache_paths(tmp_path)
        np.save(x_path, x)
        np.save(y_path, y)
        meta_path.write_text(meta_text, encoding="utf-8")

    @staticmethod
    def _metadata(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
        return {
            "source": "synthetic:test-cache",
            "details": {"parser": "fixture"},
            "x_sha256": upgd_label_emnist.materialized_array_sha256(x),
            "y_sha256": upgd_label_emnist.materialized_array_sha256(y),
        }

    def test_clean_cache_metadata_remains_compatible(self, tmp_path) -> None:
        x = np.asarray([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
        y = np.asarray([1, 0], dtype=np.int32)
        metadata = self._metadata(x, y)
        self._write_cache(tmp_path, x, y, json.dumps(metadata))

        loaded_x, loaded_y, loaded_metadata = (
            upgd_label_emnist.load_emnist_balanced_train(tmp_path)
        )

        np.testing.assert_array_equal(loaded_x, x)
        np.testing.assert_array_equal(loaded_y, y)
        assert loaded_metadata == metadata

    def test_cache_metadata_rejects_duplicate_top_level_key(self, tmp_path) -> None:
        x = np.asarray([[0.0]], dtype=np.float32)
        y = np.asarray([0], dtype=np.int32)
        metadata = self._metadata(x, y)
        meta_text = json.dumps(metadata).replace(
            '"source": "synthetic:test-cache"',
            '"source": "first", "source": "second"',
        )
        self._write_cache(tmp_path, x, y, meta_text)

        with pytest.raises(ValueError, match="duplicate JSON key: 'source'"):
            upgd_label_emnist.load_emnist_balanced_train(tmp_path)

    def test_cache_metadata_rejects_duplicate_nested_key(self, tmp_path) -> None:
        x = np.asarray([[0.0]], dtype=np.float32)
        y = np.asarray([0], dtype=np.int32)
        metadata = self._metadata(x, y)
        meta_text = json.dumps(metadata).replace(
            '"details": {"parser": "fixture"}',
            '"details": {"parser": "first", "parser": "second"}',
        )
        self._write_cache(tmp_path, x, y, meta_text)

        with pytest.raises(ValueError, match="duplicate JSON key: 'parser'"):
            upgd_label_emnist.load_emnist_balanced_train(tmp_path)

    def test_cache_metadata_still_enforces_array_digests(self, tmp_path) -> None:
        x = np.asarray([[0.0]], dtype=np.float32)
        y = np.asarray([0], dtype=np.int32)
        metadata = self._metadata(x, y)
        metadata["x_sha256"] = "0" * 64
        self._write_cache(tmp_path, x, y, json.dumps(metadata))

        with pytest.raises(RuntimeError, match="does not match its pinned digests"):
            upgd_label_emnist.load_emnist_balanced_train(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda x, y: (x, y + 100), "must be smaller than"),
        (lambda x, y: (x, y.astype(np.float32)), "integer class labels"),
        (lambda x, y: (x.at[0, 0].set(np.nan), y), "finite"),
    ],
)
def test_run_label_emnist_rejects_out_of_domain_inputs(
    monkeypatch: pytest.MonkeyPatch, mutate, message: str
) -> None:
    def unexpected_setup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("out-of-domain data reached learner setup")

    x, y = _tiny_data()
    x, y = mutate(jnp.asarray(x), jnp.asarray(y))
    monkeypatch.setattr(upgd_label_emnist, "resolve_hyperparameters", unexpected_setup)
    with pytest.raises(ValueError, match=message):
        run_label_emnist(x, y, "adamw", seeds=[0], config=TINY)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda x, y: (np.full(x.shape, np.timedelta64("NaT", "s")), y),
            "real numeric",
        ),
        (
            lambda x, y: (
                x[: TINY.task_length - 1],
                y[: TINY.task_length - 1],
            ),
            "task_length",
        ),
    ],
)
def test_run_label_emnist_rejects_boundary_gaps_before_setup(
    monkeypatch: pytest.MonkeyPatch, mutate, message: str
) -> None:
    """Issue #527: timedelta inputs and short datasets must fail before setup."""

    def unexpected_setup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("out-of-domain data reached learner setup")

    x, y = mutate(*_tiny_data())
    monkeypatch.setattr(upgd_label_emnist, "resolve_hyperparameters", unexpected_setup)
    with pytest.raises(ValueError, match=message):
        run_label_emnist(x, y, "adamw", seeds=[0], config=TINY)


@pytest.fixture(scope="class")
def debug_run():
    """Run the shared tiny diagnostic once for this test module."""
    x, y = _tiny_data()
    return run_label_emnist(
        x, y, "upgd_w", seeds=[0, 1], config=TINY, return_per_step=True
    )


class TestTinySmokeRun:

    @pytest.mark.parametrize("progress_every", [0, -1, True, np.int64(1)])
    def test_rejects_invalid_progress_interval_before_learner_setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        progress_every: object,
    ) -> None:
        def unexpected_setup(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid progress interval reached learner setup")

        monkeypatch.setattr(upgd_label_emnist, "resolve_hyperparameters", unexpected_setup)
        with pytest.raises(ValueError, match="progress_every must be a positive integer"):
            run_label_emnist(
                np.empty((TINY.task_length, TINY.input_dim), dtype=np.float32),
                np.zeros(TINY.task_length, dtype=np.int32),
                "adamw",
                seeds=[0],
                config=TINY,
                progress_every=progress_every,  # type: ignore[arg-type]
            )

    def test_shapes_and_bounds(self, debug_run):
        assert debug_run.per_task_accuracy.shape == (2, TINY.n_tasks)
        assert debug_run.per_step_accuracy.shape == (2, TINY.n_tasks, TINY.task_length)
        assert np.all(debug_run.per_task_accuracy >= 0.0)
        assert np.all(debug_run.per_task_accuracy <= 1.0)
        assert np.all(np.isin(debug_run.per_step_accuracy, [0.0, 1.0]))
        assert np.all(debug_run.per_task_loss > 0.0)
        assert np.all(debug_run.per_task_plasticity >= 0.0)
        assert np.all(debug_run.per_task_plasticity <= 1.0)

    def test_per_task_accuracy_is_mean_of_per_step(self, debug_run):
        np.testing.assert_allclose(
            debug_run.per_task_accuracy,
            debug_run.per_step_accuracy.mean(axis=2),
            atol=1e-6,
        )

    def test_average_online_accuracy_is_mean_over_tasks(self, debug_run):
        np.testing.assert_allclose(
            debug_run.average_online_accuracy,
            debug_run.per_task_accuracy.mean(axis=1),
            atol=1e-12,
        )

    def test_first_step_accuracy_recomputed_externally(self, debug_run):
        """The first prediction must be the initial net on the permuted label."""
        from alberta_framework.benchmarks.upgd_ipmnist import mlp_logits

        x, y = _tiny_data()
        for seed_row in range(2):
            params = {
                name: jnp.asarray(value[seed_row])
                for name, value in debug_run.initial_params.items()
            }
            example = int(debug_run.example_indices[seed_row, 0, 0])
            permuted_label = int(
                debug_run.label_permutations[seed_row, 0, int(y[example])]
            )
            logits = mlp_logits(params, jnp.asarray(x[example]))
            expected = float(int(np.argmax(np.asarray(logits))) == permuted_label)
            assert debug_run.per_step_accuracy[seed_row, 0, 0] == expected

    def test_adamw_runs_and_is_deterministic(self):
        x, y = _tiny_data()
        first = run_label_emnist(x, y, "adamw", seeds=[5], config=TINY)
        second = run_label_emnist(x, y, "adamw", seeds=[5], config=TINY)
        np.testing.assert_array_equal(first.per_task_accuracy, second.per_task_accuracy)

    def test_upgd_w_tiny_trajectory_pinned_across_registry_refactor(self):
        """Tiny-run trajectories captured BEFORE the full-step registry refactor.

        The refactor (grads-interface learners wrapped into the screening-style
        full-step API) must be behavior-preserving for the v1 arms: same RNG
        stream, same metric definitions, same values.
        """
        x, y = _tiny_data()
        result = run_label_emnist(x, y, "upgd_w", seeds=[0, 1], config=TINY)
        np.testing.assert_array_equal(
            result.per_task_accuracy, [[0.0, 0.25, 0.125], [0.0, 0.125, 0.125]]
        )
        np.testing.assert_allclose(
            result.per_task_loss,
            [
                [1.648679256439209, 1.5686269998550415, 1.6457500457763672],
                [1.6169909238815308, 1.736443042755127, 1.7514290809631348],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.per_task_plasticity,
            [
                [0.004521459806710482, 0.004360879771411419, 0.004870759788900614],
                [0.0034332401119172573, 0.004289050120860338, 0.005140929948538542],
            ],
            atol=1e-6,
        )

    def test_adamw_tiny_trajectory_pinned_across_registry_refactor(self):
        x, y = _tiny_data()
        result = run_label_emnist(x, y, "adamw", seeds=[0, 1], config=TINY)
        np.testing.assert_array_equal(
            result.per_task_accuracy, [[0.0, 0.25, 0.125], [0.0, 0.125, 0.125]]
        )
        np.testing.assert_allclose(
            result.per_task_loss,
            [
                [1.6493254899978638, 1.5711777210235596, 1.6426843404769897],
                [1.6191235780715942, 1.7410117387771606, 1.7633652687072754],
            ],
            atol=1e-6,
        )


class TestNormalizedTransferArms:
    """EMA input-conditioning transfer arms (screening-lane factories)."""

    def test_upgd_ema_norm_runs_and_normalizer_is_engaged(self):
        x, y = _tiny_data()
        norm = run_label_emnist(x, y, "upgd_ema_norm", seeds=[0, 1], config=TINY)
        raw = run_label_emnist(x, y, "upgd_w", seeds=[0, 1], config=TINY)
        assert norm.per_task_accuracy.shape == (2, TINY.n_tasks)
        assert np.all(norm.per_task_accuracy >= 0.0)
        assert np.all(norm.per_task_accuracy <= 1.0)
        assert np.all(norm.per_task_loss > 0.0)
        assert np.all(norm.per_task_plasticity >= 0.0)
        assert np.all(norm.per_task_plasticity <= 1.0)
        # The normalizer must actually change the trajectory vs raw UPGD-W.
        assert not np.array_equal(norm.per_task_loss, raw.per_task_loss)

    def test_sigma0_arm_equals_ema_norm_with_zero_noise_override(self):
        x, y = _tiny_data()
        sigma0 = run_label_emnist(x, y, "upgd_ema_norm_sigma0", seeds=[3], config=TINY)
        override = run_label_emnist(
            x, y, "upgd_ema_norm", seeds=[3], config=TINY,
            hyperparameters={"noise_std": 0.0},
        )
        np.testing.assert_array_equal(
            sigma0.per_task_accuracy, override.per_task_accuracy
        )
        np.testing.assert_array_equal(sigma0.per_task_loss, override.per_task_loss)

    def test_sgd_ema_norm_runs_and_is_deterministic(self):
        x, y = _tiny_data()
        first = run_label_emnist(x, y, "sgd_ema_norm", seeds=[5], config=TINY)
        second = run_label_emnist(x, y, "sgd_ema_norm", seeds=[5], config=TINY)
        np.testing.assert_array_equal(first.per_task_accuracy, second.per_task_accuracy)
        np.testing.assert_array_equal(first.per_task_loss, second.per_task_loss)

    def test_plan_binds_normalized_arm_hyperparameters(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        learners = ("upgd_ema_norm", "upgd_ema_norm_sigma0", "sgd_ema_norm")
        payload = build_plan_payload(TINY, [0, 1], DATASET_META, learners=learners)
        assert payload["plan"]["hyperparameters"] == {
            "upgd_ema_norm": UPGD_EMA_NORM_HYPERPARAMETERS,
            "upgd_ema_norm_sigma0": UPGD_EMA_NORM_SIGMA0_HYPERPARAMETERS,
            "sgd_ema_norm": SGD_EMA_NORM_HYPERPARAMETERS,
        }
        path = tmp_path / "plan.json"
        atomic_write_new_json(path, payload)
        loaded = load_plan(path)
        assert loaded["plan"]["learner_ids"] == list(learners)
        assert loaded["plan"]["planned_shard_count"] == 6


class TestPlanShardMergeAccounting:
    def _plan(self):
        return build_plan_payload(TINY, [0, 1], DATASET_META)

    def _result(self, learner: str, seed: int):
        x, y = _tiny_data()
        return run_label_emnist(x, y, learner, seeds=[seed], config=TINY)

    def test_plan_roundtrip_and_validation(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        payload = self._plan()
        path = tmp_path / "plan.json"
        atomic_write_new_json(path, payload)
        loaded = load_plan(path)
        assert loaded["plan"]["planned_shard_count"] == 4
        assert loaded["plan_sha256"] == payload["plan_sha256"]

    def test_load_plan_rejects_truncated_or_extra_config_keys(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json
        from alberta_framework.benchmarks.upgd_label_emnist import canonical_json_sha256

        protocol_fields = ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes")
        for dropped in protocol_fields:
            payload = self._plan()
            body = dict(payload["plan"])
            config = dict(body["config"])
            del config[dropped]
            body["config"] = config
            payload["plan"] = body
            payload["plan_sha256"] = canonical_json_sha256(body)
            path = tmp_path / f"plan_missing_{dropped}.json"
            atomic_write_new_json(path, payload)
            with pytest.raises(ValueError, match="config"):
                load_plan(path)

        payload = self._plan()
        body = dict(payload["plan"])
        config = dict(body["config"])
        config["extra_field"] = 1
        body["config"] = config
        payload["plan"] = body
        payload["plan_sha256"] = canonical_json_sha256(body)
        path = tmp_path / "plan_extra.json"
        atomic_write_new_json(path, payload)
        with pytest.raises(ValueError, match="config"):
            load_plan(path)

    def test_plan_rejects_bad_seed_lists(self):
        with pytest.raises(ValueError, match="unique"):
            build_plan_payload(TINY, [1, 1], DATASET_META)
        with pytest.raises(ValueError, match="sorted"):
            build_plan_payload(TINY, [2, 1], DATASET_META)
        with pytest.raises(ValueError, match="at least one seed"):
            build_plan_payload(TINY, [], DATASET_META)

    def test_merge_full_coverage_and_accounting(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        plan = self._plan()
        paths = []
        expected: dict[tuple[str, int], float] = {}
        for learner in ("upgd_w", "adamw"):
            for seed in (0, 1):
                result = self._result(learner, seed)
                expected[(learner, seed)] = float(result.average_online_accuracy[0])
                path = tmp_path / f"{learner}_seed{seed}.json"
                atomic_write_new_json(path, partial_payload(result, plan["plan_sha256"]))
                paths.append(path)
        results, coverage = merge_partials(plan, paths)
        assert coverage["complete"] and coverage["merged_shard_count"] == 4
        for learner in ("upgd_w", "adamw"):
            assert results[learner].seeds == (0, 1)
            for row, seed in enumerate(results[learner].seeds):
                assert (
                    abs(
                        float(results[learner].average_online_accuracy[row])
                        - expected[(learner, seed)]
                    )
                    < 1e-5
                )
        artifact = build_artifact(plan, results, coverage, partial_paths=paths)
        assert artifact["coverage"]["complete"]
        assert len(artifact["partial_manifest"]) == 4
        assert set(artifact["learners"]) == {"upgd_w", "adamw"}
        summary = artifact["learners"]["upgd_w"]
        assert summary["n_seeds"] == 2
        assert (
            abs(
                summary["average_online_accuracy_mean"]
                - np.mean([expected[("upgd_w", 0)], expected[("upgd_w", 1)]])
            )
            < 1e-5
        )

    def test_merge_rejects_duplicates_missing_and_foreign_shards(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        plan = self._plan()
        result = self._result("adamw", 0)
        good = tmp_path / "adamw_seed0.json"
        atomic_write_new_json(good, partial_payload(result, plan["plan_sha256"]))
        duplicate = tmp_path / "adamw_seed0_copy.json"
        atomic_write_new_json(duplicate, partial_payload(result, plan["plan_sha256"]))
        with pytest.raises(ValueError, match="duplicate shard"):
            merge_partials(plan, [good, duplicate], allow_incomplete=True)
        with pytest.raises(ValueError, match="missing planned shards"):
            merge_partials(plan, [good])
        _, coverage = merge_partials(plan, [good], allow_incomplete=True)
        assert not coverage["complete"]
        assert ["upgd_w", 0] in coverage["missing_pairs"]
        assert len(coverage["missing_pairs"]) == 3
        foreign = tmp_path / "foreign.json"
        atomic_write_new_json(foreign, partial_payload(result, "f" * 64))
        with pytest.raises(ValueError, match="different plan"):
            merge_partials(plan, [good, foreign], allow_incomplete=True)

    def test_partial_rejects_multi_seed_and_unplanned_identity(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        plan = self._plan()
        x, y = _tiny_data()
        multi = run_label_emnist(x, y, "adamw", seeds=[0, 1], config=TINY)
        with pytest.raises(ValueError, match="exactly one seed"):
            partial_payload(multi, plan["plan_sha256"])
        unplanned = self._result("adamw", 7)
        path = tmp_path / "unplanned.json"
        atomic_write_new_json(path, partial_payload(unplanned, plan["plan_sha256"]))
        with pytest.raises(ValueError, match="not planned"):
            merge_partials(plan, [path], allow_incomplete=True)

    def _synthetic_result(self, accuracies: np.ndarray) -> LabelEMNISTRunResult:
        n_seeds = int(accuracies.shape[0])
        return LabelEMNISTRunResult(
            learner="adamw",
            hyperparameters=dict(ADAMW_PROTOCOL_HYPERPARAMETERS),
            seeds=tuple(range(n_seeds)),
            config=TINY,
            per_task_accuracy=np.broadcast_to(
                accuracies[:, None], (n_seeds, TINY.n_tasks)
            ).copy(),
            per_task_loss=np.zeros((n_seeds, TINY.n_tasks)),
            per_task_plasticity=np.zeros((n_seeds, TINY.n_tasks)),
            average_online_accuracy=np.asarray(accuracies, dtype=np.float64),
            wall_clock_seconds=0.0,
        )

    def test_summarize_result_refuses_vacuous_single_seed_stderr(self):
        with pytest.raises(ValueError, match="fewer than two observations"):
            summarize_result(self._synthetic_result(np.asarray([0.5])))
        with pytest.raises(ValueError, match="seeds must be non-empty"):
            self._synthetic_result(np.asarray([], dtype=np.float64))

    def test_summarize_result_two_seed_stderr_is_sample_se(self):
        values = np.asarray([0.25, 0.75], dtype=np.float64)
        summary = summarize_result(self._synthetic_result(values))
        expected = float(values.std(ddof=1) / np.sqrt(values.shape[0]))
        assert summary["n_seeds"] == 2
        assert summary["average_online_accuracy_stderr"] == expected

    def test_summary_and_comparison_flag_logic(self):
        x, y = _tiny_data()
        upgd = run_label_emnist(x, y, "upgd_w", seeds=[0, 1], config=TINY)
        adam = run_label_emnist(x, y, "adamw", seeds=[0, 1], config=TINY)
        summaries = {"upgd_w": summarize_result(upgd), "adamw": summarize_result(adam)}
        comparison = build_comparison(summaries)
        assert set(comparison["learners"]) == {"upgd_w", "adamw"}
        assert "upgd_w_beats_adamw" in comparison
        assert "upgd_w_rises" in comparison
        for entry in comparison["learners"].values():
            assert entry["reproduction_gap_flagged"] is (
                abs(entry["gap"]) > comparison["gap_threshold"]
            )


@pytest.mark.unit
class TestPlanShardFloatAliasIntake:
    """Issue #525: int/bool aliases of float hyperparameters must be rejected.

    Python's ``0 == 0.0`` and ``True == 1.0`` make ``float(v)`` coercion and
    dict ``==`` alias-tolerant, so an aliased arm could pass the plan and shard
    gates while the sibling ``ipmnist_screening`` lane stays strict.
    """

    def _write_plan(self, tmp_path, mutate=None):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        payload = build_plan_payload(TINY, [0, 1], DATASET_META)
        if mutate is not None:
            mutate(payload["plan"])
            payload["plan_sha256"] = upgd_label_emnist.canonical_json_sha256(
                payload["plan"]
            )
        path = tmp_path / "plan.json"
        atomic_write_new_json(path, payload)
        return path

    def test_plan_rejects_int_alias_hyperparameter(self, tmp_path):
        def mutate(body):
            assert body["hyperparameters"]["adamw"]["beta1"] == 0.0
            body["hyperparameters"]["adamw"]["beta1"] = 0

        path = self._write_plan(tmp_path, mutate)
        with pytest.raises(ValueError, match="finite floats"):
            load_plan(path)

    def test_plan_rejects_bool_alias_hyperparameter(self, tmp_path):
        def mutate(body):
            assert body["hyperparameters"]["upgd_w"]["weight_decay"] == 0.0
            body["hyperparameters"]["upgd_w"]["weight_decay"] = False

        path = self._write_plan(tmp_path, mutate)
        with pytest.raises(ValueError, match="finite floats"):
            load_plan(path)

    def test_plan_rejects_non_numeric_hyperparameter(self, tmp_path):
        def mutate(body):
            body["hyperparameters"]["upgd_w"]["lr"] = "0.01"

        path = self._write_plan(tmp_path, mutate)
        with pytest.raises(ValueError, match="finite floats"):
            load_plan(path)

    def test_plan_rejects_incomplete_hyperparameter_arm(self, tmp_path):
        def mutate(body):
            del body["hyperparameters"]["upgd_w"]["noise_std"]

        path = self._write_plan(tmp_path, mutate)
        with pytest.raises(ValueError, match="complete"):
            load_plan(path)

    def test_plan_still_accepts_exact_float_hyperparameters(self, tmp_path):
        path = self._write_plan(tmp_path)
        loaded = load_plan(path)
        for learner, hp in loaded["plan"]["hyperparameters"].items():
            assert all(type(v) is float for v in hp.values()), learner

    def test_shard_rejects_int_alias_hyperparameter(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        plan = build_plan_payload(TINY, [0, 1], DATASET_META)
        x, y = _tiny_data()
        result = run_label_emnist(x, y, "adamw", seeds=[0], config=TINY)
        payload = partial_payload(result, plan["plan_sha256"])
        assert payload["hyperparameters"]["beta1"] == 0.0
        payload["hyperparameters"] = dict(payload["hyperparameters"], beta1=0)
        path = tmp_path / "aliased_int.json"
        atomic_write_new_json(path, payload)
        with pytest.raises(ValueError, match="finite floats"):
            merge_partials(plan, [path], allow_incomplete=True)

    def test_shard_rejects_bool_alias_hyperparameter(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        plan = build_plan_payload(TINY, [0, 1], DATASET_META)
        x, y = _tiny_data()
        result = run_label_emnist(x, y, "upgd_w", seeds=[0], config=TINY)
        payload = partial_payload(result, plan["plan_sha256"])
        assert payload["hyperparameters"]["weight_decay"] == 0.0
        payload["hyperparameters"] = dict(payload["hyperparameters"], weight_decay=False)
        path = tmp_path / "aliased_bool.json"
        atomic_write_new_json(path, payload)
        with pytest.raises(ValueError, match="finite floats"):
            merge_partials(plan, [path], allow_incomplete=True)

    def test_shard_rejects_unequal_float_hyperparameter(self, tmp_path):
        from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

        plan = build_plan_payload(TINY, [0, 1], DATASET_META)
        x, y = _tiny_data()
        result = run_label_emnist(x, y, "adamw", seeds=[0], config=TINY)
        payload = partial_payload(result, plan["plan_sha256"])
        payload["hyperparameters"] = dict(payload["hyperparameters"], lr=2e-4)
        path = tmp_path / "unregistered.json"
        atomic_write_new_json(path, payload)
        with pytest.raises(ValueError, match="differ from the plan"):
            merge_partials(plan, [path], allow_incomplete=True)


def _legal_label_emnist_run_result(**overrides: object) -> LabelEMNISTRunResult:
    payload: dict[str, object] = {
        "learner": "adamw",
        "hyperparameters": dict(ADAMW_PROTOCOL_HYPERPARAMETERS),
        "seeds": (0,),
        "config": TINY,
        "per_task_accuracy": np.zeros((1, TINY.n_tasks)),
        "per_task_loss": np.zeros((1, TINY.n_tasks)),
        "per_task_plasticity": np.zeros((1, TINY.n_tasks)),
        "average_online_accuracy": np.zeros((1,)),
        "wall_clock_seconds": 1.0,
    }
    payload.update(overrides)
    return LabelEMNISTRunResult(**payload)  # type: ignore[arg-type]


def test_label_emnist_run_result_rejects_leftover_identities() -> None:
    """Public Label-EMNIST result records must not keep leftover bool/NaN identities."""

    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_label_emnist_run_result(wall_clock_seconds=True)
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_label_emnist_run_result(wall_clock_seconds=float("nan"))
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_label_emnist_run_result(wall_clock_seconds=float("inf"))
    with pytest.raises(ValueError, match="learner"):
        _legal_label_emnist_run_result(learner=True)
    with pytest.raises(ValueError, match="seeds"):
        _legal_label_emnist_run_result(seeds=(True,))

    legal = _legal_label_emnist_run_result()
    dumped = json.dumps(
        {
            "learner": legal.learner,
            "seeds": list(legal.seeds),
            "wall_clock_seconds": legal.wall_clock_seconds,
        },
        allow_nan=False,
    )
    assert '"wall_clock_seconds": 1.0' in dumped
    assert '"wall_clock_seconds": true' not in dumped
    assert '"learner": true' not in dumped
    assert '"seeds": [true]' not in dumped


def test_label_emnist_result_rejects_hostile_scalar_and_seed_containers() -> None:
    class HostileFloat(float):
        def __float__(self) -> float:
            raise AssertionError("hostile float conversion must not run")

    class HostileSeeds:
        def __iter__(self):
            raise AssertionError("hostile seed iteration must not run")

    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_label_emnist_run_result(wall_clock_seconds=HostileFloat(1.0))
    with pytest.raises(ValueError, match="exact tuple"):
        _legal_label_emnist_run_result(seeds=HostileSeeds())
    with pytest.raises(ValueError, match="non-empty"):
        _legal_label_emnist_run_result(seeds=())
    with pytest.raises(ValueError, match="unique"):
        _legal_label_emnist_run_result(seeds=(0, 0))
