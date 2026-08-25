"""Tests for the UPGD Input-permuted MNIST replication lane.

CI-cheap: everything runs on tiny synthetic data. Real-MNIST benchmark runs
happen only through ``python -m alberta_framework.benchmarks.upgd_ipmnist``.
"""

import hashlib
import json
import stat
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.upgd_ipmnist as upgd_ipmnist
from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    ARTIFACT_SCHEMA,
    PAPER_REFERENCE,
    PARTIAL_SCHEMA,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    IPMNISTRunResult,
    LeanUPGDState,
    LearnerUpdateResult,
    _make_adamw_learner,
    _split_flat_noise,
    build_artifact,
    build_comparison,
    build_schedule,
    canonical_upgd_w,
    lean_upgd_w_update,
    merge_partial_results,
    partial_payload,
    resolve_hyperparameters,
    run_ipmnist,
    summarize_result,
    task_index_for_step,
)

TINY = IPMNISTConfig(n_tasks=2, task_length=200, input_dim=16, hidden1=32, hidden2=16)
N_TRAIN = 300


class TestAtomicWriteNew:
    def test_publishes_read_only_destination_without_temporary_link(self, tmp_path):
        destination = tmp_path / "result.json"

        published = upgd_ipmnist.atomic_write_new(destination, b'{"ok":true}\n')

        assert published == destination
        assert destination.read_bytes() == b'{"ok":true}\n'
        assert destination.stat().st_mode & stat.S_IWUSR == 0
        assert list(tmp_path.glob(".result.json.*.tmp")) == []

    def test_windows_readonly_cleanup_restores_public_destination(
        self, tmp_path, monkeypatch
    ):
        temporary = tmp_path / ".result.json.test.tmp"
        destination = tmp_path / "result.json"
        temporary.write_bytes(b'{"ok":true}\n')
        upgd_ipmnist.os.link(temporary, destination)
        temporary.chmod(0o444)
        original_unlink = Path.unlink
        attempts = 0

        def deny_first_unlink(path, *args, **kwargs):
            nonlocal attempts
            if path == temporary and attempts == 0:
                attempts += 1
                raise PermissionError("simulated Windows read-only hard link")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(upgd_ipmnist, "_WINDOWS_READONLY_UNLINK", True)
        monkeypatch.setattr(Path, "unlink", deny_first_unlink)

        upgd_ipmnist._unlink_published_temporary(
            temporary,
            destination,
            published=True,
        )

        assert attempts == 1
        assert not temporary.exists()
        assert destination.read_bytes() == b'{"ok":true}\n'
        assert destination.stat().st_mode & stat.S_IWUSR == 0

    def test_concurrent_destination_is_never_changed(self, tmp_path, monkeypatch):
        destination = tmp_path / "result.json"
        foreign = b'{"owner":"other"}\n'

        def lose_link_race(_source, target):
            Path(target).write_bytes(foreign)
            raise FileExistsError("simulated destination race")

        monkeypatch.setattr(upgd_ipmnist.os, "link", lose_link_race)

        with pytest.raises(FileExistsError, match="refusing to overwrite immutable output"):
            upgd_ipmnist.atomic_write_new(destination, b'{"owner":"ours"}\n')

        assert destination.read_bytes() == foreign
        assert destination.stat().st_mode & stat.S_IWUSR != 0
        assert list(tmp_path.glob(".result.json.*.tmp")) == []


class _HostileString(str):
    """A string facade that must not cross external shard identity gates."""

    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("untrusted __bool__ hook executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("untrusted __eq__ hook executed")

    __hash__ = str.__hash__


def _synthetic_dataset(
    seed: int, n_train: int, input_dim: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Clusterable synthetic stream: 10 gaussian prototypes plus noise."""
    key_centers, key_labels, key_noise = jr.split(jr.key(seed), 3)
    centers = jr.normal(key_centers, (n_classes, input_dim), dtype=jnp.float32)
    y = jr.randint(key_labels, (n_train,), 0, n_classes)
    x = centers[y] + 0.3 * jr.normal(key_noise, (n_train, input_dim), dtype=jnp.float32)
    return np.asarray(x), np.asarray(y.astype(jnp.int32))


class TestProtocolConstants:
    def test_default_config_matches_selected_publication_shape(self):
        config = IPMNISTConfig()
        assert config.n_tasks == 200
        assert config.task_length == 5000
        assert config.n_steps == 1_000_000
        assert config.input_dim == 784
        assert (config.hidden1, config.hidden2) == (300, 150)
        assert config.n_classes == 10
        assert config.matches_selected_publication_configuration

    def test_shrunk_config_does_not_match_selected_publication_shape(self):
        assert not TINY.matches_selected_publication_configuration

    @pytest.mark.parametrize(
        "integer_type",
        sorted(
            {
                np.dtype(code).type
                for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")
            },
            key=lambda value: value.__name__,
        ),
    )
    def test_config_accepts_every_numpy_integer_dtype_family(self, integer_type):
        config = IPMNISTConfig(
            n_tasks=integer_type(2),
            task_length=integer_type(3),
            input_dim=integer_type(4),
            hidden1=integer_type(5),
            hidden2=integer_type(6),
            n_classes=integer_type(2),
        )
        assert all(type(value) is int for value in config.to_config().values())

    @pytest.mark.parametrize(
        "value",
        [True, np.bool_(True), 1.0, "1", 0, -1, 2**31],
    )
    def test_config_rejects_non_integer_and_out_of_range_dimensions(self, value):
        with pytest.raises(ValueError, match="n_tasks"):
            IPMNISTConfig(n_tasks=value)

    def test_config_rejects_spoofs_and_subclasses_without_calling_hooks(self):
        class HostileIndex:
            def __index__(self):
                raise AssertionError("must not call hostile __index__")

            def __repr__(self):
                raise AssertionError("must not call hostile __repr__")

        class HostileInt(int):
            def __index__(self):
                raise AssertionError("must not call subclass __index__")

            def __repr__(self):
                raise AssertionError("must not call subclass __repr__")

        class HostileNumpyInt(np.int64):
            def __index__(self):
                raise AssertionError("must not call NumPy subclass __index__")

            def __repr__(self):
                raise AssertionError("must not call NumPy subclass __repr__")

        for value in (HostileIndex(), HostileInt(2), HostileNumpyInt(2)):
            with pytest.raises(ValueError, match="n_tasks"):
                IPMNISTConfig(n_tasks=value)

    def test_config_derived_horizon_boundaries_are_allocation_free(self):
        boundary = IPMNISTConfig(n_tasks=1, task_length=2**31 - 1, input_dim=1)
        assert boundary.n_steps == 2**31 - 1
        with pytest.raises(ValueError, match="run horizon"):
            IPMNISTConfig(n_tasks=2, task_length=2**30, input_dim=1)

    def test_config_permutation_schedule_boundaries_are_allocation_free(self):
        boundary = IPMNISTConfig(n_tasks=2**31 - 1, task_length=1, input_dim=1)
        assert boundary.n_tasks * boundary.input_dim == 2**31 - 1
        with pytest.raises(ValueError, match="permutation schedule"):
            IPMNISTConfig(n_tasks=2, task_length=1, input_dim=2**30)

    @pytest.mark.parametrize(
        ("field_values"),
        [
            {"input_dim": 46_341, "hidden1": 46_341, "hidden2": 1, "n_classes": 1},
            {"input_dim": 1, "hidden1": 46_341, "hidden2": 46_341, "n_classes": 1},
            {"input_dim": 1, "hidden1": 1, "hidden2": 46_341, "n_classes": 46_341},
        ],
    )
    def test_config_rejects_each_oversized_parameter_matrix(self, field_values):
        with pytest.raises(ValueError, match="parameter allocation"):
            IPMNISTConfig(**field_values)

    def test_config_total_parameter_allocation_boundary_is_exact_and_allocation_free(self):
        boundary = IPMNISTConfig(
            n_tasks=1,
            task_length=1,
            input_dim=(2**31 - 1) - 5,
            hidden1=1,
            hidden2=1,
            n_classes=1,
        )
        assert (
            boundary.input_dim * boundary.hidden1
            + boundary.hidden1 * boundary.hidden2
            + boundary.hidden2 * boundary.n_classes
            + boundary.hidden1
            + boundary.hidden2
            + boundary.n_classes
            == 2**31 - 1
        )
        assert boundary.parameter_count == 2**31 - 1
        with pytest.raises(ValueError, match="parameter allocation"):
            IPMNISTConfig(
                n_tasks=1,
                task_length=1,
                input_dim=(2**31 - 1) - 4,
                hidden1=1,
                hidden2=1,
                n_classes=1,
            )

    def test_config_roundtrip_canonicalizes_numpy_integer_scalars(self):
        config = IPMNISTConfig(
            n_tasks=np.int16(2),
            task_length=np.uint16(3),
            input_dim=np.int32(4),
            hidden1=np.uint32(5),
            hidden2=np.int64(6),
            n_classes=np.uint64(2),
        )
        assert IPMNISTConfig(**config.to_config()) == config
        assert all(type(value) is int for value in config.to_config().values())

    def test_published_hyperparameters(self):
        assert UPGD_W_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 0.01,
            "utility_decay": 0.9999,
            "noise_std": 0.1,
            "weight_decay": 0.01,
        }
        assert ADAMW_PROTOCOL_HYPERPARAMETERS == {
            "step_size": 1e-4,
            "beta1": 0.0,
            "beta2": 0.99,
            "eps": 1e-8,
            "weight_decay": 0.0,
        }

    def test_resolve_hyperparameters_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown hyperparameters"):
            resolve_hyperparameters("upgd_w", {"lr": 0.1})
        with pytest.raises(ValueError, match="unknown learner"):
            resolve_hyperparameters("sgd", None)

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
            ("upgd_w", "utility_decay", -0.1),
            ("upgd_w", "utility_decay", 1.0),
            ("upgd_w", "noise_std", -0.1),
            ("upgd_w", "weight_decay", -0.1),
            ("adamw", "step_size", 0.0),
            ("adamw", "beta1", -0.1),
            ("adamw", "beta1", 1.0),
            ("adamw", "beta2", 1.0),
            ("adamw", "eps", 0.0),
            ("adamw", "weight_decay", -0.1),
        ],
    )
    def test_resolve_hyperparameters_enforces_field_domains(
        self, learner: str, name: str, value: float
    ) -> None:
        with pytest.raises(ValueError, match=f"hyperparameter {name!r}"):
            resolve_hyperparameters(learner, {name: value})

    def test_resolve_hyperparameters_accepts_endpoints_and_canonicalizes_ints(self) -> None:
        resolved = resolve_hyperparameters(
            "upgd_w",
            {"utility_decay": 0, "noise_std": 0, "weight_decay": 1},
        )
        assert resolved["utility_decay"] == 0.0
        assert resolved["noise_std"] == 0.0
        assert resolved["weight_decay"] == 1.0
        assert all(type(resolved[name]) is float for name in resolved)


class TestSchedule:
    def test_task_index_changes_exactly_at_multiples_of_task_length(self):
        length = TINY.task_length
        steps = np.arange(3 * length)
        tasks = np.asarray(task_index_for_step(steps, length))
        # Exact boundary steps: last step of task t and first step of task t+1.
        assert tasks[0] == 0
        assert tasks[length - 1] == 0
        assert tasks[length] == 1
        assert tasks[2 * length - 1] == 1
        assert tasks[2 * length] == 2
        # Each task owns exactly task_length consecutive steps.
        np.testing.assert_array_equal(np.bincount(tasks), [length, length, length])

    def test_permutations_are_valid_and_task_indexed(self):
        schedule = build_schedule(jr.key(0), TINY, N_TRAIN)
        perms = np.asarray(schedule.permutations)
        assert perms.shape == (TINY.n_tasks, TINY.input_dim)
        for row in perms:
            np.testing.assert_array_equal(np.sort(row), np.arange(TINY.input_dim))
        # The first task is itself permuted-in-general (schedule rows differ).
        assert not np.array_equal(perms[0], perms[1])

    def test_example_indices_sample_without_replacement(self):
        schedule = build_schedule(jr.key(1), TINY, N_TRAIN)
        examples = np.asarray(schedule.example_indices)
        assert examples.shape == (TINY.n_tasks, TINY.task_length)
        assert examples.min() >= 0
        assert examples.max() < N_TRAIN
        for row in examples:
            assert len(np.unique(row)) == TINY.task_length

    def test_schedule_is_deterministic_per_key(self):
        first = build_schedule(jr.key(7), TINY, N_TRAIN)
        second = build_schedule(jr.key(7), TINY, N_TRAIN)
        np.testing.assert_array_equal(
            np.asarray(first.permutations), np.asarray(second.permutations)
        )
        np.testing.assert_array_equal(
            np.asarray(first.example_indices), np.asarray(second.example_indices)
        )

    def test_schedule_rejects_dataset_smaller_than_task(self):
        with pytest.raises(ValueError, match="without replacement"):
            build_schedule(jr.key(0), TINY, TINY.task_length - 1)


class TestInputDomain:
    """Labels index the softmax: JAX clamps out-of-range gathers, so the runner must refuse them."""

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda x, y: (x, y + 100), "must be smaller than"),
            (lambda x, y: (x, y - 1), "non-negative"),
            (lambda x, y: (x, y.astype(np.float32) + 0.9), "integer class labels"),
            (lambda x, y: (x.at[0, 0].set(np.inf), y), "finite"),
            (lambda x, y: (x.at[3, 1].set(np.nan), y), "finite"),
            (
                lambda x, y: (x.astype(jnp.complex64) + jnp.complex64(1j), y),
                "real numeric",
            ),
        ],
    )
    def test_run_rejects_out_of_domain_inputs_before_setup(
        self, monkeypatch: pytest.MonkeyPatch, mutate, message: str
    ) -> None:
        def unexpected_setup(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("out-of-domain data reached learner setup")

        x, y = _synthetic_dataset(0, N_TRAIN, TINY.input_dim, TINY.n_classes)
        x, y = mutate(jnp.asarray(x), jnp.asarray(y))
        monkeypatch.setattr(upgd_ipmnist, "resolve_hyperparameters", unexpected_setup)
        with pytest.raises(ValueError, match=message):
            run_ipmnist(x, y, "adamw", seeds=[0], config=TINY)

    @pytest.mark.parametrize("progress_every", [0, -1, True, 1.5])
    def test_run_rejects_invalid_progress_interval_before_setup(
        self, monkeypatch: pytest.MonkeyPatch, progress_every: object
    ) -> None:
        def unexpected_setup(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid progress interval reached learner setup")

        x, y = _synthetic_dataset(0, N_TRAIN, TINY.input_dim, TINY.n_classes)
        monkeypatch.setattr(upgd_ipmnist, "resolve_hyperparameters", unexpected_setup)
        with pytest.raises(ValueError, match="progress_every"):
            run_ipmnist(
                x,
                y,
                "adamw",
                seeds=[0],
                config=TINY,
                progress_every=progress_every,  # type: ignore[arg-type]
            )


@pytest.mark.unit
class TestInputDomainBoundary:
    """Issue #527: the shared validator is the single gate ahead of learner setup."""

    @staticmethod
    def _unexpected_setup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("out-of-domain data reached learner setup")

    def test_run_rejects_timedelta_inputs_before_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x = np.full((N_TRAIN, TINY.input_dim), np.timedelta64("NaT", "s"))
        y = np.zeros(N_TRAIN, dtype=np.int32)
        monkeypatch.setattr(upgd_ipmnist, "resolve_hyperparameters", self._unexpected_setup)
        with pytest.raises(ValueError, match="real numeric"):
            run_ipmnist(x, y, "adamw", seeds=[0], config=TINY)

    def test_run_rejects_short_dataset_before_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, y = _synthetic_dataset(0, TINY.task_length - 1, TINY.input_dim, TINY.n_classes)
        monkeypatch.setattr(upgd_ipmnist, "resolve_hyperparameters", self._unexpected_setup)
        with pytest.raises(ValueError, match="task_length"):
            run_ipmnist(x, y, "adamw", seeds=[0], config=TINY)

    @pytest.mark.parametrize("kind", ["timedelta64[s]", "datetime64[s]", "bool"])
    def test_validator_rejects_non_real_dtype_kinds(self, kind: str) -> None:
        x = np.zeros((4, TINY.input_dim), dtype=kind)
        y = np.zeros(4, dtype=np.int32)
        with pytest.raises(ValueError, match="real numeric"):
            upgd_ipmnist.validated_ipmnist_data(
                x, y, input_dim=TINY.input_dim, n_classes=TINY.n_classes
            )

    def test_validator_enforces_min_length(self) -> None:
        x, y = _synthetic_dataset(0, 5, TINY.input_dim, TINY.n_classes)
        with pytest.raises(ValueError, match="task_length"):
            upgd_ipmnist.validated_ipmnist_data(
                x, y, input_dim=TINY.input_dim, n_classes=TINY.n_classes, min_length=6
            )
        upgd_ipmnist.validated_ipmnist_data(
            x, y, input_dim=TINY.input_dim, n_classes=TINY.n_classes, min_length=5
        )


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

        monkeypatch.setattr(upgd_ipmnist, "resolve_hyperparameters", unexpected_setup)
        with pytest.raises(ValueError, match="seeds"):
            run_ipmnist(
                np.empty((1, 1), dtype=np.float32),
                np.empty((1,), dtype=np.int32),
                "adamw",
                seeds=seeds,  # type: ignore[arg-type]
            )

    def test_run_preserves_full_uint32_seed_identities(self) -> None:
        config = IPMNISTConfig(
            n_tasks=1,
            task_length=1,
            input_dim=2,
            hidden1=2,
            hidden2=2,
            n_classes=2,
        )
        data_x = np.asarray([[0.25, -0.5]], dtype=np.float32)
        data_y = np.asarray([1], dtype=np.int32)
        result = run_ipmnist(
            data_x,
            data_y,
            "adamw",
            seeds=(2**32 - 1, 0),
            config=config,
            return_per_step=True,
        )

        assert result.seeds == (2**32 - 1, 0)
        assert result.initial_params is not None
        assert not np.array_equal(result.initial_params["w1"][0], result.initial_params["w1"][1])

    def test_cli_rejects_aliased_seed_before_loading_mnist(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected_load(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid CLI seeds reached dataset loading")

        monkeypatch.setattr(upgd_ipmnist, "load_mnist_train", unexpected_load)
        with pytest.raises(ValueError, match=r"seeds\[1\].*uint32"):
            upgd_ipmnist.main_v2_compat(
                [
                    "--learners",
                    "adamw",
                    "--seed-list",
                    f"0,{2**32}",
                    "--partial-out",
                    str(tmp_path / "must-not-exist.json"),
                ]
            )


@pytest.fixture(scope="module")
def debug_run():
    data_x, data_y = _synthetic_dataset(0, N_TRAIN, TINY.input_dim, TINY.n_classes)
    return run_ipmnist(
        data_x,
        data_y,
        "upgd_w",
        seeds=(0, 1),
        config=TINY,
        return_per_step=True,
    ), (data_x, data_y)


class TestAccuracyAccounting:

    def test_per_task_accuracy_is_mean_of_per_step(self, debug_run):
        result, _ = debug_run
        assert result.per_step_accuracy.shape == (2, TINY.n_tasks, TINY.task_length)
        np.testing.assert_allclose(
            result.per_task_accuracy,
            result.per_step_accuracy.mean(axis=2),
            atol=1e-6,
        )

    def test_per_step_accuracy_is_zero_or_one(self, debug_run):
        result, _ = debug_run
        assert set(np.unique(result.per_step_accuracy)) <= {0.0, 1.0}

    def test_average_online_accuracy_is_mean_over_tasks(self, debug_run):
        result, _ = debug_run
        np.testing.assert_allclose(
            result.average_online_accuracy,
            result.per_task_accuracy.mean(axis=1),
            atol=1e-7,
        )

    def test_first_step_accuracy_recomputed_externally(self, debug_run):
        """Step 0's accuracy must equal an out-of-band numpy forward pass."""
        result, (data_x, data_y) = debug_run
        for seed_index in range(2):
            permutation = result.permutations[seed_index, 0]
            example = int(result.example_indices[seed_index, 0, 0])
            x = data_x[example][permutation]
            hidden = np.maximum(
                x @ result.initial_params["w1"][seed_index]
                + result.initial_params["b1"][seed_index],
                0.0,
            )
            hidden = np.maximum(
                hidden @ result.initial_params["w2"][seed_index]
                + result.initial_params["b2"][seed_index],
                0.0,
            )
            logits = (
                hidden @ result.initial_params["w3"][seed_index]
                + result.initial_params["b3"][seed_index]
            )
            expected = float(np.argmax(logits) == data_y[example])
            assert result.per_step_accuracy[seed_index, 0, 0] == pytest.approx(expected)

    def test_losses_and_plasticity_within_bounds(self, debug_run):
        result, _ = debug_run
        assert np.all(np.isfinite(result.per_task_loss))
        assert np.all(result.per_task_loss > 0.0)
        assert np.all(result.per_task_plasticity >= 0.0)
        assert np.all(result.per_task_plasticity <= 1.0)


class TestSmoke:
    """Tiny-scale smoke: 2 tasks x 200 steps, both learners, 2 seeds."""

    @pytest.mark.parametrize("learner", ["upgd_w", "adamw"])
    def test_published_hyperparameters_run_within_bounds(self, learner):
        """Published hyperparameters are tuned for 5,000-step MNIST tasks;
        at 200 tiny synthetic steps we assert sanity, not learning."""
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(data_x, data_y, learner, seeds=(0, 1), config=TINY)
        assert result.per_task_accuracy.shape == (2, TINY.n_tasks)
        assert np.all(np.isfinite(result.per_task_accuracy))
        assert np.all(result.per_task_accuracy >= 0.0)
        assert np.all(result.per_task_accuracy <= 1.0)
        assert np.all(np.isfinite(result.per_task_loss))

    @pytest.mark.parametrize(
        ("learner", "overrides", "threshold"),
        [
            # Calibrated on seeds 0-5: upgd_w reaches >=0.31 mean, adamw
            # >=0.85; thresholds sit at 2x chance (0.1) and below half the
            # measured means, respectively.
            ("upgd_w", {"step_size": 0.1, "noise_std": 0.01}, 0.2),
            ("adamw", {"step_size": 0.01}, 0.4),
        ],
    )
    def test_learner_learns_above_chance_at_smoke_scale(self, learner, overrides, threshold):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(
            data_x, data_y, learner, seeds=(0, 1), config=TINY, hyperparameters=overrides
        )
        assert result.per_task_accuracy.mean() > threshold

    def test_runs_are_deterministic_per_seed(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        first = run_ipmnist(data_x, data_y, "upgd_w", seeds=(5,), config=TINY)
        second = run_ipmnist(data_x, data_y, "upgd_w", seeds=(5,), config=TINY)
        np.testing.assert_array_equal(first.per_task_accuracy, second.per_task_accuracy)

    def test_hyperparameter_override_is_recorded(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(
            data_x,
            data_y,
            "adamw",
            seeds=(0,),
            config=TINY,
            hyperparameters={"weight_decay": 0.01},
        )
        assert result.hyperparameters["weight_decay"] == pytest.approx(0.01)


class TestLeanUPGDParity:
    """The lean benchmark step must match CanonicalUPGD exactly.

    ``CanonicalUPGD(profile="official_experiment_global", mode="protecting")``
    is the audited equation source; the lean step exists purely for CPU scan
    speed. Both consume the same supplied noise, so agreement is equation
    exactness, not luck.
    """

    def test_multi_step_parity_with_canonical(self):
        hp = dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        shapes = {"w1": (7, 5), "b1": (5,), "w2": (5, 3), "b2": (3,)}
        key = jr.key(42)
        key, params_key = jr.split(key)
        params = {
            name: 0.3 * jr.normal(jr.fold_in(params_key, i), shape, jnp.float32)
            for i, (name, shape) in enumerate(sorted(shapes.items()))
        }
        canonical = canonical_upgd_w(hp)
        canonical_params = dict(params)
        canonical_state = canonical.init(canonical_params)
        lean_params = dict(params)
        lean_state = LeanUPGDState(
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )
        for step in range(5):
            key, grads_key, noise_key = jr.split(key, 3)
            grads = {
                name: jr.normal(jr.fold_in(grads_key, i), shape, jnp.float32)
                for i, (name, shape) in enumerate(sorted(shapes.items()))
            }
            noise = {
                name: hp["noise_std"]
                * jr.normal(jr.fold_in(noise_key, i), shape, jnp.float32)
                for i, (name, shape) in enumerate(sorted(shapes.items()))
            }
            update = canonical.update(
                canonical_state, canonical_params, grads, jr.fold_in(key, step), noise=noise
            )
            canonical_params, canonical_state = update.params, update.state
            lean_params, lean_state = lean_upgd_w_update(
                lean_params, lean_state, grads, noise, hp
            )
            for name in shapes:
                np.testing.assert_allclose(
                    np.asarray(lean_params[name]),
                    np.asarray(canonical_params[name]),
                    atol=1e-6,
                    err_msg=f"step {step} param {name}",
                )
                np.testing.assert_allclose(
                    np.asarray(lean_state.utility[name]),
                    np.asarray(canonical_state.utility_ema[name]),
                    atol=1e-6,
                    err_msg=f"step {step} utility {name}",
                )


class TestAdamWTransaction:
    """AdamW's parameter leaves form one checked learner transaction."""

    @staticmethod
    def _params() -> dict[str, jnp.ndarray]:
        return {
            "bias": jnp.asarray([0.2, -0.1], dtype=jnp.float32),
            "weight": jnp.asarray([[0.4, -0.3], [0.1, 0.5]], dtype=jnp.float32),
        }

    @staticmethod
    def _assert_tree_equal(actual, expected) -> None:
        import jax

        actual_leaves, actual_structure = jax.tree.flatten(actual)
        expected_leaves, expected_structure = jax.tree.flatten(expected)
        assert actual_structure == expected_structure
        for actual_leaf, expected_leaf in zip(
            actual_leaves, expected_leaves, strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf), np.asarray(expected_leaf)
            )

    def test_nonfinite_leaf_rejects_entire_params_and_optimizer_state(self):
        import jax

        hp = dict(ADAMW_PROTOCOL_HYPERPARAMETERS)
        init_fn, step_fn = _make_adamw_learner(hp)
        params = self._params()
        state = init_fn(params)
        grads = {
            "bias": jnp.asarray([jnp.inf, 0.25], dtype=jnp.float32),
            "weight": jnp.asarray([[0.3, -0.2], [0.4, 0.1]], dtype=jnp.float32),
        }

        compiled_step = jax.jit(step_fn)
        result = compiled_step(params, state, grads, jr.key(0))

        assert isinstance(result, LearnerUpdateResult)
        assert not bool(result.update_applied)
        self._assert_tree_equal(result.params, params)
        self._assert_tree_equal(result.state, state)
        # The established two-value caller surface remains available.
        unpacked_params, unpacked_state = result
        self._assert_tree_equal(unpacked_params, params)
        self._assert_tree_equal(unpacked_state, state)

        recovered = compiled_step(
            result.params,
            result.state,
            {
                "bias": jnp.asarray([0.1, 0.25], dtype=jnp.float32),
                "weight": grads["weight"],
            },
            jr.key(1),
        )
        assert bool(recovered.update_applied)
        assert not np.array_equal(
            np.asarray(recovered.params["weight"]), np.asarray(params["weight"])
        )

    def test_finite_post_apply_overflow_rejects_entire_transaction(self):
        import jax

        hp = dict(ADAMW_PROTOCOL_HYPERPARAMETERS)
        hp["step_size"] = 3e38
        hp["weight_decay"] = 0.0
        init_fn, step_fn = _make_adamw_learner(hp)
        params = {
            "bias": jnp.asarray([-3e38], dtype=jnp.float32),
            "weight": self._params()["weight"],
        }
        state = init_fn(params)
        grads = {
            "bias": jnp.asarray([1.0], dtype=jnp.float32),
            "weight": jnp.zeros_like(params["weight"]),
        }

        compiled_step = jax.jit(step_fn)
        result = compiled_step(params, state, grads, jr.key(2))

        assert not bool(result.update_applied)
        self._assert_tree_equal(result.params, params)
        self._assert_tree_equal(result.state, state)

        recovered = compiled_step(
            result.params,
            result.state,
            {name: jnp.zeros_like(value) for name, value in grads.items()},
            jr.key(3),
        )
        assert bool(recovered.update_applied)
        self._assert_tree_equal(recovered.params, params)

    def test_finite_step_matches_leafwise_adamw_and_jit(self):
        import jax

        from alberta_framework.core.baseline_optimizers import Adam

        hp = dict(ADAMW_PROTOCOL_HYPERPARAMETERS)
        hp["weight_decay"] = 0.02
        init_fn, step_fn = _make_adamw_learner(hp)
        params = self._params()
        state = init_fn(params)
        grads = {
            "bias": jnp.asarray([-0.3, 0.25], dtype=jnp.float32),
            "weight": jnp.asarray([[0.3, -0.2], [0.4, 0.1]], dtype=jnp.float32),
        }
        optimizer = Adam(
            step_size=hp["step_size"],
            beta1=hp["beta1"],
            beta2=hp["beta2"],
            eps=hp["eps"],
            weight_decay=hp["weight_decay"],
        )
        expected_params: dict[str, jnp.ndarray] = {}
        expected_state = {}
        for name, value in params.items():
            step, leaf_state = optimizer.update_from_gradient(
                state[name], grads[name], error=None, param=value
            )
            expected_params[name] = value - step
            expected_state[name] = leaf_state

        eager = step_fn(params, state, grads, jr.key(1))
        compiled = jax.jit(step_fn)(params, state, grads, jr.key(1))

        assert bool(eager.update_applied)
        assert bool(compiled.update_applied)
        self._assert_tree_equal(eager.params, expected_params)
        self._assert_tree_equal(eager.state, expected_state)
        self._assert_tree_equal(compiled, eager)


class TestSplitFlatNoise:
    def test_slices_in_sorted_name_order_with_exact_values(self):
        shapes = {"b1": (5,), "b2": (3,), "w1": (7, 5), "w2": (5, 3)}
        total = sum(int(np.prod(shape)) for shape in shapes.values())
        flat = jr.normal(jr.key(0), (total,), jnp.float32)
        split = _split_flat_noise(flat, shapes)
        assert set(split) == set(shapes)
        offset = 0
        for name in sorted(shapes):
            count = int(np.prod(shapes[name]))
            assert split[name].shape == shapes[name]
            np.testing.assert_array_equal(
                np.asarray(split[name]).ravel(),
                np.asarray(flat[offset:offset + count]),
            )
            offset += count
        assert offset == total


class TestRunnerTrajectoryExactness:
    """run_ipmnist's optimized inner loop vs a plain dict-based reference loop.

    The reference reproduces the exact published RNG stream contract: one
    ``jr.split`` per step of the per-seed noise-key chain, one flat
    ``N(0, sigma^2)`` draw sliced per sorted parameter name.
    """

    SMALL = IPMNISTConfig(n_tasks=2, task_length=50, input_dim=16, hidden1=32, hidden2=16)

    @pytest.mark.parametrize("learner", ["upgd_w", "adamw"])
    def test_matches_reference_loop(self, learner):
        import jax

        from alberta_framework.benchmarks.upgd_ipmnist import (
            _LEARNER_FACTORIES,
            cross_entropy_loss,
            init_mlp_params,
        )

        config = self.SMALL
        data_x, data_y = _synthetic_dataset(11, N_TRAIN, config.input_dim, config.n_classes)
        result = run_ipmnist(
            data_x, data_y, learner, seeds=(0,), config=config, return_per_step=True
        )

        hp = resolve_hyperparameters(learner)
        init_fn, step_fn = _LEARNER_FACTORIES[learner](hp)
        root = jr.key(jnp.uint32(0))
        key_init, key_schedule, key_noise = jr.split(root, 3)
        params = init_mlp_params(key_init, config)
        schedule = build_schedule(key_schedule, config, N_TRAIN)
        state = init_fn(params)
        key = key_noise
        xs = jnp.asarray(data_x, jnp.float32)
        ys = jnp.asarray(data_y, jnp.int32)
        accuracies = np.zeros((config.n_tasks, config.task_length))
        for task in range(config.n_tasks):
            permutation = schedule.permutations[task]
            for i in range(config.task_length):
                example = schedule.example_indices[task, i]
                x = xs[example][permutation]
                y = ys[example]
                (_, logits), grads = jax.value_and_grad(
                    cross_entropy_loss, has_aux=True
                )(params, x, y)
                accuracies[task, i] = float(jnp.argmax(logits) == y)
                key, step_key = jr.split(key)
                params, state = step_fn(params, state, grads, step_key)
        np.testing.assert_array_equal(result.per_step_accuracy[0], accuracies)


class TestNoisePoolMode:
    def test_noise_mode_recorded_and_defaults_to_step(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        exact = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        assert exact.noise_mode == "step"
        pooled = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        assert pooled.noise_mode == "pool"

    def test_pool_mode_is_deterministic_and_bounded(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        first = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(2,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        second = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(2,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        np.testing.assert_array_equal(first.per_task_accuracy, second.per_task_accuracy)
        assert np.all(np.isfinite(first.per_task_loss))
        assert np.all(first.per_task_accuracy >= 0.0)
        assert np.all(first.per_task_accuracy <= 1.0)

    def test_pool_mode_changes_upgd_but_not_adamw(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        upgd_exact = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        upgd_pool = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        assert not np.array_equal(upgd_exact.per_task_loss, upgd_pool.per_task_loss)

        adamw_exact = run_ipmnist(data_x, data_y, "adamw", seeds=(0,), config=TINY)
        adamw_pool = run_ipmnist(
            data_x, data_y, "adamw", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        np.testing.assert_array_equal(
            adamw_exact.per_task_accuracy, adamw_pool.per_task_accuracy
        )
        np.testing.assert_array_equal(adamw_exact.per_task_loss, adamw_pool.per_task_loss)

    def test_pool_mode_rejects_invalid_pool_size(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        with pytest.raises(ValueError, match="noise_pool_steps"):
            run_ipmnist(
                data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
                noise_mode="pool", noise_pool_steps=1,
            )

    def test_pool_mode_rejects_noninteger_pool_size_before_allocation(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        with pytest.raises(ValueError, match="noise_pool_steps"):
            run_ipmnist(
                data_x,
                data_y,
                "upgd_w",
                seeds=(0,),
                config=TINY,
                noise_mode="pool",
                noise_pool_steps=2.5,  # type: ignore[arg-type]
            )

    def test_pool_mode_rejects_derived_length_overflow_before_allocation(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        with pytest.raises(ValueError, match=r"noise_pool_steps \* parameter_count"):
            run_ipmnist(
                data_x,
                data_y,
                "upgd_w",
                seeds=(0,),
                config=TINY,
                noise_mode="pool",
                noise_pool_steps=2**31 - 1,
            )

    def test_pool_mode_shard_serialization_fails_closed(self):
        data_x, data_y = _synthetic_dataset(3, N_TRAIN, TINY.input_dim, TINY.n_classes)
        pooled = run_ipmnist(
            data_x, data_y, "upgd_w", seeds=(0,), config=TINY,
            noise_mode="pool", noise_pool_steps=4,
        )
        with pytest.raises(ValueError, match="noise_mode"):
            partial_payload(pooled)


class TestPartialMerge:
    def test_partial_loader_rejects_non_path_and_oversized_shards(
        self, tmp_path, monkeypatch
    ) -> None:
        with pytest.raises(ValueError, match="path must be a Path"):
            upgd_ipmnist._strict_json_object(str(tmp_path / "partial.json"))  # type: ignore[arg-type]

        path = tmp_path / "oversized.json"
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            upgd_ipmnist,
            "_MAX_PARTIAL_JSON_BYTES",
            1,
        )
        with pytest.raises(ValueError, match="no larger than"):
            upgd_ipmnist._strict_json_object(path)

    @staticmethod
    def _minimal_v2_payload() -> dict[str, object]:
        payload = {field: None for field in upgd_ipmnist._V2_PARTIAL_FIELDS}
        payload.update(
            {
                "schema": PARTIAL_SCHEMA,
                "schema_version": 2,
                "evidence_policy": upgd_ipmnist.NONPROMOTING_POLICY,
                "deviations": list(upgd_ipmnist.PROTOCOL_DEVIATIONS),
                "learner": "adamw",
                "hyperparameters": {"step_size": 0.001},
            }
        )
        return payload

    def test_v2_partial_manifest_rejects_hostile_learner_before_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "hostile.json"
        path.write_bytes(b"{}")
        payload = {"schema": PARTIAL_SCHEMA, "learner": _HostileString("adamw"), "seed_id": 0}
        _HostileString.calls = 0
        monkeypatch.setattr(
            upgd_ipmnist, "_decode_strict_json_object", lambda _raw, *, path: payload
        )

        with pytest.raises(ValueError, match="identity"):
            upgd_ipmnist._v2_partial_manifest([path])

        assert _HostileString.calls == 0

    @pytest.mark.parametrize("location", ["learner", "hyperparameter_name"])
    def test_partial_validation_rejects_hostile_names_before_hooks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        location: str,
    ) -> None:
        payload = self._minimal_v2_payload()
        hostile = _HostileString("adamw" if location == "learner" else "step_size")
        if location == "learner":
            payload["learner"] = hostile
        else:
            payload["hyperparameters"] = {hostile: 0.001}
        _HostileString.calls = 0
        monkeypatch.setattr(upgd_ipmnist, "_strict_json_object", lambda _path: payload)

        with pytest.raises(ValueError, match="learner|hyperparameters"):
            upgd_ipmnist._validated_partial_payload(
                tmp_path / "hostile.json", schema=PARTIAL_SCHEMA, seed_field="seed_id"
            )

        assert _HostileString.calls == 0

    def test_partial_validation_rejects_hostile_hyperparameter_value_before_float_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hostile int/float subclass hyperparameter value must be rejected by an
        exact-type check before its ``__float__`` hook ever runs, matching the
        already-hardened ``IPMNISTRunResult.__post_init__`` convention in this module.
        """

        class HostileFloat(float):
            def __float__(self) -> float:
                raise AssertionError("hostile float conversion must not run")

        payload = self._minimal_v2_payload()
        payload["hyperparameters"] = {"step_size": HostileFloat(0.001)}
        monkeypatch.setattr(upgd_ipmnist, "_strict_json_object", lambda _path: payload)

        with pytest.raises(ValueError, match="finite named numbers"):
            upgd_ipmnist._validated_partial_payload(
                tmp_path / "hostile.json", schema=PARTIAL_SCHEMA, seed_field="seed_id"
            )

    def test_partial_validation_rejects_hostile_wall_clock_before_float_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hostile int/float subclass ``wall_clock_seconds`` must be rejected before
        its ``__float__``/``__add__`` hooks run; otherwise it survives validation and
        ``_merge_partial_results`` later calls the unguarded builtin ``sum()`` over it.

        Uses a fully valid shard payload (from a real run) so the function actually
        reaches the ``wall_clock_seconds`` gate instead of failing earlier on the
        deliberately-incomplete minimal fixture's hyperparameters.
        """

        class HostileFloat(float):
            def __float__(self) -> float:
                raise AssertionError("hostile float conversion must not run")

            def __add__(self, other: object) -> float:
                raise AssertionError("hostile addition must not run")

            __radd__ = __add__

        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        payload = partial_payload(shard)
        payload["wall_clock_seconds"] = HostileFloat(1.0)
        monkeypatch.setattr(upgd_ipmnist, "_strict_json_object", lambda _path: payload)

        with pytest.raises(ValueError, match="wall_clock_seconds"):
            upgd_ipmnist._validated_partial_payload(
                tmp_path / "hostile.json", schema=PARTIAL_SCHEMA, seed_field="seed_id"
            )

    def test_partial_validation_normalizes_integer_wall_clock_seconds(
        self, tmp_path: Path
    ) -> None:
        """An integer ``wall_clock_seconds`` is a legitimate finite real value (the
        v1 legacy schema and hand-built shards may supply one) and must be accepted
        and coerced to a builtin ``float`` via ``_require_finite_real``, matching the
        already-hardened convention used for the live-run result dataclass.
        """
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        payload = partial_payload(shard)
        payload["wall_clock_seconds"] = 3
        path = tmp_path / "integer-wall-clock.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        validated = upgd_ipmnist._validated_partial_payload(
            path, schema=PARTIAL_SCHEMA, seed_field="seed_id"
        )

        assert validated["wall_clock_seconds"] == 3.0
        assert type(validated["wall_clock_seconds"]) is float

    def test_v2_partial_manifest_binds_identity_and_digest_to_one_read(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "partial.json"
        original = json.dumps(
            {"schema": PARTIAL_SCHEMA, "learner": "adamw", "seed_id": 7}
        ).encode("utf-8")
        replacement = json.dumps(
            {"schema": PARTIAL_SCHEMA, "learner": "upgd_w", "seed_id": 11}
        )
        path.write_bytes(original)

        original_read_bytes = Path.read_bytes

        def replace_after_read(target: Path) -> bytes:
            snapshot = original_read_bytes(target)
            target.write_text(replacement, encoding="utf-8")
            return snapshot

        monkeypatch.setattr(Path, "read_bytes", replace_after_read)

        manifest = upgd_ipmnist._v2_partial_manifest([path])

        assert manifest == [
            {
                "learner": "adamw",
                "seed_id": 7,
                "path": path.as_posix(),
                "size_bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
        ]

    def test_partial_roundtrip_and_merge(self, tmp_path):
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard_a = run_ipmnist(data_x, data_y, "upgd_w", seeds=(1,), config=TINY)
        shard_b = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(partial_payload(shard_a)))
        path_b.write_text(json.dumps(partial_payload(shard_b)))
        merged = merge_partial_results([path_a, path_b])
        result = merged["upgd_w"]
        assert result.seeds == (0, 1)  # sorted by seed, not file order
        assert result.per_task_accuracy.shape == (2, TINY.n_tasks)
        np.testing.assert_allclose(
            result.per_task_accuracy[1], shard_a.per_task_accuracy[0], atol=1e-9
        )
        np.testing.assert_allclose(
            result.per_task_accuracy[0], shard_b.per_task_accuracy[0], atol=1e-9
        )

    def test_merge_rejects_duplicate_seeds_and_config_mismatch(self, tmp_path):
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0,), config=TINY)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(partial_payload(shard)))
        path_b.write_text(json.dumps(partial_payload(shard)))
        with pytest.raises(ValueError, match="duplicate seeds"):
            merge_partial_results([path_a, path_b])

        other_config = IPMNISTConfig(
            n_tasks=2, task_length=200, input_dim=16, hidden1=32, hidden2=8
        )
        other = run_ipmnist(data_x, data_y, "upgd_w", seeds=(1,), config=other_config)
        path_b.write_text(json.dumps(partial_payload(other)))
        with pytest.raises(ValueError, match="disagree on config"):
            merge_partial_results([path_a, path_b])

    def test_merge_rejects_seed_identity_outside_jax_key_domain(self, tmp_path) -> None:
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard = run_ipmnist(data_x, data_y, "adamw", seeds=(0,), config=TINY)
        payload = partial_payload(shard)
        payload["seed_id"] = 2**32
        path = tmp_path / "aliased-seed.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="uint32"):
            merge_partial_results([path])

    def test_merge_rejects_hyperparameters_outside_learner_domain(self, tmp_path) -> None:
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard = run_ipmnist(data_x, data_y, "adamw", seeds=(0,), config=TINY)
        payload = partial_payload(shard)
        payload["hyperparameters"]["step_size"] = -1.0
        path = tmp_path / "negative-step-size.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="invalid hyperparameters"):
            merge_partial_results([path])

    def test_merge_rejects_integer_alias_of_float_hyperparameter(self, tmp_path) -> None:
        data_x, data_y = _synthetic_dataset(6, N_TRAIN, TINY.input_dim, TINY.n_classes)
        shard = run_ipmnist(data_x, data_y, "adamw", seeds=(0,), config=TINY)
        payload = partial_payload(shard)
        payload["hyperparameters"]["beta1"] = 0
        path = tmp_path / "integer-beta1.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="complete learner configuration"):
            merge_partial_results([path])

    def test_v2_partial_is_single_seed_and_recursively_omits_legacy_marker(
        self, tmp_path, debug_run
    ):
        multi_seed, _ = debug_run
        with pytest.raises(ValueError, match="exactly one seed"):
            partial_payload(multi_seed)

        data_x, data_y = _synthetic_dataset(9, N_TRAIN, TINY.input_dim, TINY.n_classes)
        one_seed = run_ipmnist(data_x, data_y, "upgd_w", seeds=(3,), config=TINY)
        payload = partial_payload(one_seed)
        assert payload["schema"] == PARTIAL_SCHEMA
        assert payload["seed_id"] == 3
        assert payload["seed_count"] == 1
        assert "is_protocol_exact" not in json.dumps(payload)

        path = tmp_path / "v2.json"
        payload["is_protocol_exact"] = True
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="fields do not match"):
            merge_partial_results([path])

    def test_v2_artifact_records_scoped_match_and_exact_seed_facts(self, tmp_path):
        data_x, data_y = _synthetic_dataset(10, N_TRAIN, TINY.input_dim, TINY.n_classes)
        run = run_ipmnist(data_x, data_y, "adamw", seeds=(7,), config=TINY)
        artifact = build_artifact({"adamw": run}, TINY, tmp_path / "cache")

        assert artifact["schema"] == ARTIFACT_SCHEMA
        assert artifact["schema_version"] == 2
        assert artifact["protocol"]["matches_selected_publication_configuration"] is False
        assert artifact["protocol"]["selected_publication_configuration_match_scope"] == (
            "network_task_shape_and_horizon_only"
        )
        assert artifact["study_design"]["exact_seed_ids_by_learner"] == {"adamw": [7]}
        assert artifact["study_design"]["exact_seed_count_by_learner"] == {"adamw": 1}
        assert artifact["study_design"]["published_seed_count"] == 20
        assert artifact["evidence_policy"]["scientific_promotion_allowed"] is False
        assert "is_protocol_exact" not in json.dumps(artifact)


class TestSummariesAndComparison:
    def test_summary_and_comparison_flag_logic(self):
        data_x, data_y = _synthetic_dataset(4, N_TRAIN, TINY.input_dim, TINY.n_classes)
        result = run_ipmnist(data_x, data_y, "upgd_w", seeds=(0, 1), config=TINY)
        summary = summarize_result(result)
        assert summary["n_seeds"] == 2
        assert len(summary["per_task_accuracy_mean"]) == TINY.n_tasks
        assert summary["average_online_accuracy_mean"] == pytest.approx(
            float(result.average_online_accuracy.mean())
        )

        # Synthetic comparison: a 0.03 gap must be flagged, 0.01 must not.
        flagged = build_comparison(
            {"upgd_w": {**summary, "average_online_accuracy_mean": 0.75}}
        )
        assert flagged["learners"]["upgd_w"]["reproduction_gap_flagged"]
        clean = build_comparison(
            {"upgd_w": {**summary, "average_online_accuracy_mean": 0.79}}
        )
        assert not clean["learners"]["upgd_w"]["reproduction_gap_flagged"]
        assert clean["reference"] is PAPER_REFERENCE


def _legal_ipmnist_run_result(**overrides: object) -> IPMNISTRunResult:
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
    return IPMNISTRunResult(**payload)  # type: ignore[arg-type]


def test_ipmnist_run_result_rejects_leftover_identities() -> None:
    """Public IPMNIST result records must not keep leftover bool/NaN identities."""

    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_ipmnist_run_result(wall_clock_seconds=True)
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_ipmnist_run_result(wall_clock_seconds=float("nan"))
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_ipmnist_run_result(wall_clock_seconds=float("inf"))
    with pytest.raises(ValueError, match="learner"):
        _legal_ipmnist_run_result(learner=True)
    with pytest.raises(ValueError, match="seeds"):
        _legal_ipmnist_run_result(seeds=(True,))
    with pytest.raises(ValueError, match="noise_mode"):
        _legal_ipmnist_run_result(noise_mode=True)

    legal = _legal_ipmnist_run_result()
    dumped = json.dumps(
        {
            "learner": legal.learner,
            "seeds": list(legal.seeds),
            "wall_clock_seconds": legal.wall_clock_seconds,
            "noise_mode": legal.noise_mode,
        },
        allow_nan=False,
    )
    assert '"wall_clock_seconds": 1.0' in dumped
    assert '"wall_clock_seconds": true' not in dumped
    assert '"learner": true' not in dumped
    assert '"seeds": [true]' not in dumped
    assert '"noise_mode": true' not in dumped


def test_ipmnist_run_result_rejects_hostile_scalar_and_seed_containers() -> None:
    class HostileFloat(float):
        def __float__(self) -> float:
            raise AssertionError("hostile float conversion must not run")

    class HostileSeeds:
        def __iter__(self):
            raise AssertionError("hostile seed iteration must not run")

    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _legal_ipmnist_run_result(wall_clock_seconds=HostileFloat(1.0))
    with pytest.raises(ValueError, match="exact tuple"):
        _legal_ipmnist_run_result(seeds=HostileSeeds())
    with pytest.raises(ValueError, match="non-empty"):
        _legal_ipmnist_run_result(seeds=())
    with pytest.raises(ValueError, match="unique"):
        _legal_ipmnist_run_result(seeds=(0, 0))
