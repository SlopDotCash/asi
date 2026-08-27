"""Fail-closed source/data provenance pins for new IPMNIST screening shards."""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import alberta_framework.benchmarks.ipmnist_screening as screening
from alberta_framework.benchmarks.ipmnist_screening import (
    IPMNISTConfig,
    ScreeningRunResult,
)

SMALL = IPMNISTConfig(
    n_tasks=3, task_length=30, input_dim=784, hidden1=8, hidden2=6, n_classes=10
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    (repo / "alberta_framework/benchmarks").mkdir(parents=True)
    (repo / "alberta_framework/__init__.py").write_text("\n", encoding="utf-8")
    (repo / "alberta_framework/benchmarks/ipmnist_screening.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def _source_binding(*, source_sha256: str = "3" * 64) -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.source_provenance.v1",
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "git_object_format": "sha1",
        "relevant_source_scope": "tracked:alberta_framework/**,pyproject.toml,uv.lock",
        "relevant_source_file_count": 3,
        "relevant_source_sha256": source_sha256,
        "uv_lock_sha256": "4" * 64,
        "worktree_clean": True,
    }


def _runtime_binding(*, machine: str = "test-machine") -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.runtime.v1",
        "python": {"implementation": "CPython", "version": "3.12.12"},
        "platform": {"system": "TestOS", "release": "1", "machine": machine},
        "packages": {
            "chex": "0.1.91",
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "numpy": "2.5.1",
            "scikit-learn": "1.7.1",
        },
        "jax": {
            "backend": "cpu",
            "devices": [
                {"id": 0, "platform": "cpu", "device_kind": "test-cpu", "process_index": 0}
            ],
            "config": {
                "jax_enable_x64": False,
                "jax_default_matmul_precision": None,
                "jax_disable_jit": False,
                "jax_numpy_dtype_promotion": "standard",
                "jax_numpy_rank_promotion": "allow",
                "jax_random_seed_offset": 0,
                "jax_threefry_partitionable": True,
                "jax_default_prng_impl": "threefry2x32",
            },
        },
        "process_environment": {
            "CUDA_VISIBLE_DEVICES": None,
            "JAX_DEFAULT_MATMUL_PRECISION": None,
            "JAX_ENABLE_X64": None,
            "JAX_PLATFORM_NAME": None,
            "JAX_PLATFORMS": None,
            "OMP_NUM_THREADS": "1",
            "XLA_FLAGS": None,
        },
    }


def _dataset_binding(*, x_sha256: str = "5" * 64) -> dict[str, object]:
    return {
        "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
        "source": {
            "provider": "openml",
            "name": "mnist_784",
            "version": 1,
            "row_start": 0,
            "row_stop_exclusive": 60000,
        },
        "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
        "x": {"dtype": "<f4", "shape": [60000, 784], "sha256": x_sha256},
        "y": {"dtype": "<i4", "shape": [60000], "sha256": "6" * 64},
    }


def _result(seed: int = 0) -> ScreeningRunResult:
    spec = screening.screening_spec("upgd_w_control")
    return ScreeningRunResult(
        config_name=spec.name,
        base_learner=spec.base_learner,
        hyperparameters=dict(spec.hyperparameters),
        seed=seed,
        config=SMALL,
        per_task_accuracy=np.full(SMALL.n_tasks, 0.5),
        per_task_loss=np.full(SMALL.n_tasks, 0.1),
        per_task_plasticity=np.full(SMALL.n_tasks, 0.5),
        wall_clock_seconds=1.0,
    )


def _payload(seed: int = 0) -> dict[str, object]:
    return screening.shard_payload(
        _result(seed),
        source_provenance=_source_binding(),
        dataset_provenance=_dataset_binding(),
        environment=_runtime_binding(),
    )


def _legacy_payload(seed: int = 0) -> dict[str, object]:
    legacy = _payload(seed)
    legacy["schema"] = "alberta.ipmnist_screening.shard.v1"
    legacy.pop("source_provenance")
    legacy.pop("dataset_provenance")
    legacy["environment"] = {
        "jax": "0.11.0",
        "numpy": "2.5.1",
        "python": "3.12.12",
        "platform": "legacy-platform",
    }
    return legacy


def test_clean_source_provenance_binds_head_tree_lock_and_actual_bytes(tmp_path: Path) -> None:
    repo = _source_repo(tmp_path)

    provenance = screening._screening_source_provenance(repo)

    assert provenance["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert provenance["git_tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert provenance["uv_lock_sha256"] == hashlib.sha256(
        (repo / "uv.lock").read_bytes()
    ).hexdigest()
    assert provenance["worktree_clean"] is True
    assert provenance["relevant_source_file_count"] == 4
    assert len(str(provenance["relevant_source_sha256"])) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        "tracked",
        "tracked-nonsource",
        "untracked-package",
        "untracked-python",
        "untracked-root-bytecode",
    ],
)
def test_source_provenance_rejects_dirty_or_untracked_package_source(
    tmp_path: Path, mutation: str
) -> None:
    repo = _source_repo(tmp_path)
    if mutation == "tracked":
        (repo / "alberta_framework/benchmarks/ipmnist_screening.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
    elif mutation == "tracked-nonsource":
        (repo / "README.md").write_text("changed\n", encoding="utf-8")
    elif mutation == "untracked-package":
        (repo / "alberta_framework/untracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif mutation == "untracked-python":
        (repo / "local_override.py").write_text("VALUE = 2\n", encoding="utf-8")
    else:
        source = repo / "bytecode_source.py"
        source.write_text("VALUE = 7\n", encoding="utf-8")
        py_compile.compile(
            str(source),
            cfile=str(repo / "shadowmodule.pyc"),
            doraise=True,
        )
        source.unlink()

    with pytest.raises(RuntimeError, match="source worktree is not clean"):
        screening._screening_source_provenance(repo)


def test_source_provenance_allows_append_only_output_files(tmp_path: Path) -> None:
    repo = _source_repo(tmp_path)
    expected = screening._screening_source_provenance(repo)
    output = repo / "outputs/ipmnist_screening/replication_r1/shards/one.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")

    assert screening._screening_source_provenance(repo) == expected


def test_source_digest_changes_with_clean_committed_source_bytes(tmp_path: Path) -> None:
    repo = _source_repo(tmp_path)
    first = screening._screening_source_provenance(repo)
    (repo / "alberta_framework/benchmarks/ipmnist_screening.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    _git(repo, "add", "alberta_framework/benchmarks/ipmnist_screening.py")
    _git(repo, "commit", "-qm", "change source")

    second = screening._screening_source_provenance(repo)

    assert second["git_commit"] != first["git_commit"]
    assert second["git_tree"] != first["git_tree"]
    assert second["relevant_source_sha256"] != first["relevant_source_sha256"]
    assert second["uv_lock_sha256"] == first["uv_lock_sha256"]


def test_source_provenance_rejects_assume_unchanged_hidden_edit(tmp_path: Path) -> None:
    repo = _source_repo(tmp_path)
    source = repo / "alberta_framework/benchmarks/ipmnist_screening.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "update-index", "--assume-unchanged", source.relative_to(repo).as_posix())
    assert _git(repo, "status", "--short") == ""

    with pytest.raises(RuntimeError, match="index flags|differs from HEAD"):
        screening._screening_source_provenance(repo)


@pytest.mark.parametrize(
    ("relative", "exclude"),
    [
        ("sitecustomize.py", "sitecustomize.py"),
        ("alberta_framework/sourceless.pyc", "alberta_framework/*.pyc"),
    ],
)
def test_source_provenance_rejects_ignored_importable_source(
    tmp_path: Path, relative: str, exclude: str
) -> None:
    repo = _source_repo(tmp_path)
    with (repo / ".git/info/exclude").open("a", encoding="utf-8") as stream:
        stream.write(f"{exclude}\n")
    source = repo / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"hidden source\n")
    assert _git(repo, "status", "--short") == ""

    with pytest.raises(RuntimeError, match="source worktree is not clean"):
        screening._screening_source_provenance(repo)


def test_source_provenance_fails_when_git_or_lock_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Git repository"):
        screening._screening_source_provenance(tmp_path)

    repo = _source_repo(tmp_path)
    (repo / "uv.lock").unlink()
    with pytest.raises(RuntimeError, match="source worktree is not clean|uv.lock"):
        screening._screening_source_provenance(repo)


def test_dataset_provenance_hashes_materialized_x_and_y_with_dtype_and_shape() -> None:
    x = np.arange(24, dtype=np.float64).reshape(3, 8)[:, ::2]
    y = np.asarray([2, 1, 0], dtype=np.int64)

    first = screening._materialized_dataset_provenance(x, y)
    second = screening._materialized_dataset_provenance(
        np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
    )
    changed = screening._materialized_dataset_provenance(
        np.asarray(x[::-1], dtype=np.float32), np.asarray(y, dtype=np.int32)
    )

    assert first == second
    assert first["x"]["dtype"] == "<f4"
    assert first["x"]["shape"] == [3, 4]
    assert first["y"]["dtype"] == "<i4"
    assert first["y"]["shape"] == [3]
    assert changed["x"]["sha256"] != first["x"]["sha256"]


def test_streamed_array_hash_matches_the_prior_tobytes_contract() -> None:
    arrays = {
        "features": np.arange(24, dtype=">f4").reshape(4, 6)[:, ::2],
        "labels": np.asarray([3, 2, 1, 0], dtype=">i4"),
    }
    domain = "alberta.test.array-bundle.v1"
    expected = hashlib.sha256()
    expected.update(domain.encode("ascii") + b"\0")
    for name in sorted(arrays):
        encoded_name = name.encode("ascii")
        array = screening._canonical_hash_array(arrays[name])
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        payload = array.tobytes(order="C")
        expected.update(len(encoded_name).to_bytes(4, "little"))
        expected.update(encoded_name)
        expected.update(len(header).to_bytes(8, "little"))
        expected.update(header)
        expected.update(len(payload).to_bytes(8, "little"))
        expected.update(payload)

    assert screening._array_bundle_sha256(domain, arrays) == expected.hexdigest()


def test_direct_shard_payload_refuses_unbound_v2() -> None:
    with pytest.raises(TypeError, match="source_provenance"):
        screening.shard_payload(_result())

    inconsistent = replace(
        _result(),
        config=IPMNISTConfig(
            n_tasks=SMALL.n_tasks,
            task_length=SMALL.task_length,
            input_dim=12,
            hidden1=SMALL.hidden1,
            hidden2=SMALL.hidden2,
            n_classes=5,
        ),
    )
    with pytest.raises(ValueError, match="dataset width"):
        screening.shard_payload(
            inconsistent,
            source_provenance=_source_binding(),
            dataset_provenance=_dataset_binding(),
            environment=_runtime_binding(),
        )


@pytest.mark.parametrize("field", ["base_learner", "hyperparameters"])
def test_v2_writer_and_loader_bind_registered_mechanism(
    tmp_path: Path, field: str
) -> None:
    result = _result()
    if field == "base_learner":
        # Construction now rejects unsupported learner identities before the
        # writer boundary.  Simulate a hostile adopted instance so this test
        # continues to prove that the writer independently binds the registry.
        changed_result = replace(result)
        object.__setattr__(changed_result, "base_learner", "forged")
        replacement: object = "forged"
    else:
        changed_result = replace(
            result, hyperparameters={**result.hyperparameters, "step_size": 999.0}
        )
        replacement = changed_result.hyperparameters
    with pytest.raises(ValueError, match="registered arm"):
        screening.shard_payload(
            changed_result,
            source_provenance=_source_binding(),
            dataset_provenance=_dataset_binding(),
            environment=_runtime_binding(),
        )

    payload = _payload()
    payload[field] = replacement
    path = tmp_path / f"forged-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="registered arm"):
        screening.load_shard(path)


@pytest.mark.parametrize(
    ("field", "alias"),
    [("noise_std", False), ("gate_beta", 1)],
)
def test_v2_writer_and_loader_reject_registered_float_type_aliases(
    tmp_path: Path, field: str, alias: object
) -> None:
    spec = screening.screening_spec("sigma0_shiftnorm_d099")
    aliased_hyperparameters = {**spec.hyperparameters, field: alias}
    result = replace(
        _result(),
        config_name=spec.name,
        base_learner=spec.base_learner,
        hyperparameters=dict(spec.hyperparameters),
    )
    # Bypass the now-stricter constructor to retain an independent writer
    # boundary test for JSON bool/int aliases of registered float fields.
    object.__setattr__(result, "hyperparameters", aliased_hyperparameters)
    with pytest.raises(ValueError, match="registered arm"):
        screening.shard_payload(
            result,
            source_provenance=_source_binding(),
            dataset_provenance=_dataset_binding(),
            environment=_runtime_binding(),
        )

    payload = _payload()
    payload["config_name"] = spec.name
    payload["base_learner"] = spec.base_learner
    payload["hyperparameters"] = aliased_hyperparameters
    path = tmp_path / f"aliased-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="registered arm"):
        screening.load_shard(path)


@pytest.mark.parametrize(
    ("field", "alias"),
    [("development_only", 1), ("scientific_promotion_allowed", 0)],
)
def test_v2_loader_rejects_boolean_policy_number_aliases(
    tmp_path: Path, field: str, alias: object
) -> None:
    payload = _payload()
    payload["evidence_policy"] = {
        **screening.NONPROMOTING_POLICY,
        field: alias,
    }
    path = tmp_path / f"aliased-policy-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen nonpromoting policy"):
        screening.load_shard(path)


@pytest.mark.parametrize("value", [True, "0.5"])
def test_v2_curves_reject_boolean_and_string_numbers(tmp_path: Path, value: object) -> None:
    payload = _payload()
    payload["per_task_accuracy"] = [value, 0.5, 0.5]
    path = tmp_path / "invalid-curve.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="list of finite JSON numbers"):
        screening.load_shard(path)

    bool_result = replace(_result())
    object.__setattr__(
        bool_result,
        "per_task_accuracy",
        np.asarray([True, False, True], dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="per_task_accuracy"):
        screening.shard_payload(
            bool_result,
            source_provenance=_source_binding(),
            dataset_provenance=_dataset_binding(),
            environment=_runtime_binding(),
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("per_task_accuracy", -0.01),
        ("per_task_accuracy", 1.01),
        ("per_task_loss", -0.01),
        ("per_task_plasticity", -0.01),
        ("per_task_plasticity", 1.01),
    ],
)
def test_v2_writer_and_loader_reject_out_of_domain_curves(
    tmp_path: Path, field: str, invalid: float
) -> None:
    curve = np.full(SMALL.n_tasks, 0.5, dtype=np.float64)
    curve[1] = invalid
    result = replace(_result())
    object.__setattr__(result, field, curve)
    with pytest.raises(ValueError, match=field):
        screening.shard_payload(
            result,
            source_provenance=_source_binding(),
            dataset_provenance=_dataset_binding(),
            environment=_runtime_binding(),
        )

    payload = _payload()
    payload[field] = curve.tolist()
    path = tmp_path / f"invalid-domain-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        screening.load_shard(path)


@pytest.mark.parametrize("failure", ["curve", "wall-clock"])
def test_v2_writer_and_atomic_publish_reject_nonfinite_json(
    tmp_path: Path, failure: str
) -> None:
    result = _result()
    if failure == "curve":
        object.__setattr__(
            result,
            "per_task_loss",
            np.asarray([0.1, np.nan, 0.1], dtype=np.float64),
        )
    else:
        object.__setattr__(result, "wall_clock_seconds", float("inf"))
    with pytest.raises(ValueError, match="finite"):
        screening.shard_payload(
            result,
            source_provenance=_source_binding(),
            dataset_provenance=_dataset_binding(),
            environment=_runtime_binding(),
        )

    output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="JSON compliant|Out of range float"):
        screening._atomic_write_json(output, {"value": float("nan")})
    assert not output.exists()


def test_v2_loader_is_strict_and_legacy_v1_remains_readable(tmp_path: Path) -> None:
    v2_path = tmp_path / "v2.json"
    payload = _payload()
    v2_path.write_text(json.dumps(payload), encoding="utf-8")
    assert screening.load_shard(v2_path)["schema"].endswith(".v2")

    changed = dict(payload)
    changed["unexpected"] = True
    v2_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected field"):
        screening.load_shard(v2_path)

    changed = json.loads(json.dumps(payload))
    changed["dataset_provenance"]["x"]["shape"] = [64, 12]
    changed["dataset_provenance"]["y"]["shape"] = [64]
    v2_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical x"):
        screening.load_shard(v2_path)

    for field, value, message in (
        ("input_dim", 12, "dataset width"),
        ("n_classes", 5, "n_classes must be 10"),
    ):
        changed = json.loads(json.dumps(payload))
        changed["config"][field] = value
        v2_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            screening.load_shard(v2_path)

    changed = json.loads(json.dumps(payload))
    changed["dataset_provenance"]["source"]["version"] = True
    v2_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="source selection"):
        screening.load_shard(v2_path)

    legacy = _legacy_payload()
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert screening.load_shard(legacy_path)["schema"].endswith(".v1")


@pytest.mark.parametrize(
    "mutation",
    ["negative-id", "negative-process", "duplicate-device"],
)
def test_v2_writer_and_loader_reject_invalid_runtime_device_topology(
    tmp_path: Path, mutation: str
) -> None:
    environment = _runtime_binding()
    jax_binding = environment["jax"]
    assert isinstance(jax_binding, dict)
    devices = jax_binding["devices"]
    assert isinstance(devices, list)
    device = devices[0]
    assert isinstance(device, dict)
    if mutation == "negative-id":
        device["id"] = -1
    elif mutation == "negative-process":
        device["process_index"] = -1
    elif mutation == "duplicate-device":
        devices.append(dict(device))

    with pytest.raises(ValueError, match="runtime environment"):
        screening.shard_payload(
            _result(),
            source_provenance=_source_binding(),
            dataset_provenance=_dataset_binding(),
            environment=environment,
        )

    payload = _payload()
    payload["environment"] = environment
    path = tmp_path / f"invalid-runtime-{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime environment"):
        screening.load_shard(path)


def test_homogeneous_legacy_merge_emits_legacy_summary(tmp_path: Path) -> None:
    paths = [tmp_path / f"legacy-seed{seed}.json" for seed in (0, 1)]
    for seed, path in enumerate(paths):
        path.write_text(json.dumps(_legacy_payload(seed)), encoding="utf-8")

    summary = screening.merge_shards(
        paths, control_name="upgd_w_control", slope_window=2
    )

    assert summary["schema"].endswith(".v1")
    assert "source_provenance" not in summary
    assert "dataset_provenance" not in summary


def test_v2_merge_carries_identical_bindings_and_rejects_all_drift(tmp_path: Path) -> None:
    paths = []
    for seed in (0, 1):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(_payload(seed)), encoding="utf-8")
        paths.append(path)

    summary = screening.merge_shards(paths, control_name="upgd_w_control", slope_window=2)
    assert summary["schema"].endswith(".v2")
    assert summary["source_provenance"] == _source_binding()
    assert summary["dataset_provenance"] == _dataset_binding()
    assert summary["environment"] == _runtime_binding()
    assert [(entry["config_name"], entry["seed"]) for entry in summary["shard_manifest"]] == [
        ("upgd_w_control", 0),
        ("upgd_w_control", 1),
    ]
    for entry in summary["shard_manifest"]:
        raw = Path(entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()

    for field, replacement, message in (
        ("source_provenance", _source_binding(source_sha256="9" * 64), "source provenance"),
        ("dataset_provenance", _dataset_binding(x_sha256="9" * 64), "dataset"),
        ("environment", _runtime_binding(machine="other"), "runtime environment"),
    ):
        changed = _payload(1)
        changed[field] = replacement
        paths[1].write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            screening.merge_shards(paths, control_name="upgd_w_control", slope_window=2)

    paths[0].write_text(paths[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        screening._require_embedded_artifact_manifest_unchanged(
            summary["shard_manifest"], context="test shard inputs"
        )


def test_merge_rejects_mixed_v1_v2_shards(tmp_path: Path) -> None:
    current = _payload(0)
    legacy = _legacy_payload(1)
    paths = [tmp_path / "current.json", tmp_path / "legacy.json"]
    for path, payload in zip(paths, (current, legacy), strict=True):
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple shard schemas"):
        screening.merge_shards(paths, control_name="upgd_w_control", slope_window=2)


@pytest.mark.parametrize("slope_window", [True, 0, 1])
def test_merge_rejects_invalid_slope_window(
    tmp_path: Path, slope_window: object
) -> None:
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="slope_window"):
        screening.merge_shards(
            [path],
            control_name="upgd_w_control",
            slope_window=slope_window,  # type: ignore[arg-type]
        )


def test_merge_rejects_slope_window_larger_than_task_trace(tmp_path: Path) -> None:
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="slope_window cannot exceed n_tasks"):
        screening.merge_shards(
            [path],
            control_name="upgd_w_control",
            slope_window=SMALL.n_tasks + 1,
        )


def test_proxy_validation_refuses_mixed_or_mismatched_v2_provenance(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_payload(0)), encoding="utf-8")
    changed = _payload(1)
    changed["dataset_provenance"] = _dataset_binding(x_sha256="9" * 64)
    second.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple dataset bindings"):
        screening.validate_proxy([first, second], tmp_path)

    second.write_text(json.dumps(_legacy_payload(1)), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple shard schemas"):
        screening.validate_proxy([first, second], tmp_path)


@pytest.mark.parametrize(
    "atol",
    [True, 0, -1e-8, 1.000001e-6, float("nan"), float("inf")],
)
def test_proxy_validation_rejects_unsafe_tolerance(
    tmp_path: Path, atol: object
) -> None:
    with pytest.raises(ValueError, match="atol"):
        screening.validate_proxy([], tmp_path, atol=atol)  # type: ignore[arg-type]


@pytest.mark.parametrize("drift", ["source", "environment", "dataset"])
def test_run_cli_binds_before_execution_and_refuses_prepublication_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    events: list[str] = []
    sources = iter([_source_binding(), _source_binding(source_sha256="9" * 64)])
    environments = iter([_runtime_binding(), _runtime_binding(machine="other")])
    datasets = iter([_dataset_binding(), _dataset_binding(x_sha256="9" * 64)])

    def source() -> dict[str, object]:
        events.append("source")
        return next(sources) if drift == "source" else _source_binding()

    def environment() -> dict[str, object]:
        events.append("environment")
        return next(environments) if drift == "environment" else _runtime_binding()

    def dataset(*_args: object) -> dict[str, object]:
        events.append("dataset")
        return next(datasets) if drift == "dataset" else _dataset_binding()

    monkeypatch.setattr(screening, "_screening_source_provenance", source)
    monkeypatch.setattr(screening, "_screening_runtime_environment", environment)
    monkeypatch.setattr(screening, "_screening_dataset_provenance", dataset)
    monkeypatch.setattr(
        screening,
        "load_mnist_train",
        lambda _path: events.append("load") or (np.zeros((1, 12)), np.zeros(1)),
    )
    monkeypatch.setattr(
        screening,
        "run_screening_config",
        lambda *_args, **_kwargs: events.append("run") or _result(),
    )
    monkeypatch.setattr(
        screening,
        "_atomic_write_json",
        lambda *_args, **_kwargs: events.append("publish"),
    )

    with pytest.raises(RuntimeError, match="changed during execution"):
        screening.main(
            [
                "run",
                "--config-name",
                "upgd_w_control",
                "--seed",
                "0",
                "--out",
                str(tmp_path / "shard.json"),
            ]
        )

    assert events[:3] == ["source", "environment", "load"]
    assert events[-1] != "publish"


@pytest.mark.parametrize("drift", ["source", "environment"])
def test_v2_derivation_requires_current_shard_binding_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    summary = screening.merge_shards(
        [path], control_name="upgd_w_control", slope_window=2
    )
    source = _source_binding()
    environment = _runtime_binding()
    bindings = (source, environment)
    monkeypatch.setattr(
        screening,
        "_screening_source_provenance",
        lambda: (
            _source_binding(source_sha256="9" * 64)
            if drift == "source"
            else source
        ),
    )
    monkeypatch.setattr(
        screening,
        "_screening_runtime_environment",
        lambda: (
            _runtime_binding(machine="different")
            if drift == "environment"
            else environment
        ),
    )

    with pytest.raises(RuntimeError, match="changed during derivation"):
        screening._require_v2_derivation_context(summary, bindings)
