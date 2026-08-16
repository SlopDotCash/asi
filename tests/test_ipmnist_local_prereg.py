"""Fail-closed contracts for the two local IPMNIST preregistrations."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import runpy
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = runpy.run_path(str(_ROOT / ".github" / "scripts" / "ipmnist_local_prereg.py"))
_PROTOCOLS = cast(dict[str, Any], _DRIVER["LOCAL_PROTOCOLS"])
_amendment_line = cast(Any, _DRIVER["amendment_line"])
_authorization_line = cast(Any, _DRIVER["authorization_line"])
_build_parser = cast(Any, _DRIVER["build_parser"])
_canonical_sha256 = cast(Any, _DRIVER["_canonical_sha256"])
_claim_namespace = cast(Any, _DRIVER["_claim_namespace"])
_classify_issue184 = cast(Any, _DRIVER["classify_issue184"])
_claim_local_launch = cast(Any, _DRIVER["claim_local_launch"])
_execution_attempt_payload = cast(Any, _DRIVER["_execution_attempt_payload"])
_l2init_gate_passes = cast(Any, _DRIVER["l2init_gate_passes"])
_parse_cpuset = cast(Any, _DRIVER["_parse_cpuset"])
_receipt_sha256 = cast(Any, _DRIVER["_receipt_sha256"])
_record_cache_receipt = cast(Any, _DRIVER["record_cache_receipt"])
_run_local_shard = cast(Any, _DRIVER["run_local_shard"])
_strict_json = cast(Any, _DRIVER["_strict_json"])
_strict_json_bytes = cast(Any, _DRIVER["_strict_json_bytes"])
_validate_and_publish_result = cast(Any, _DRIVER["validate_and_publish_result"])
_validate_result_bundle = cast(Any, _DRIVER["validate_result_bundle"])
_validate_runner_receipt = cast(Any, _DRIVER["_validate_runner_receipt"])
_verify_owner_records = cast(Any, _DRIVER["verify_owner_records"])
_write_json_exclusive = cast(Any, _DRIVER["_write_json_exclusive"])


def _binding() -> dict[str, str]:
    return {
        "source": "1" * 40,
        "tree": "2" * 40,
        "uv_lock_sha256": "3" * 64,
        "benchmark_blob_sha1": "4" * 40,
        "ref_name": "ipmnist-prereg-example",
        "runner_receipt_sha256": "5" * 64,
        "data_home_sha256": "6" * 64,
    }


def test_local_protocols_pin_every_stage_and_arm() -> None:
    issue184 = _PROTOCOLS["issue184"]
    assert issue184.issue == 184
    assert issue184.namespace == "rls_preset_ablation_r1"
    assert issue184.control == "rls_head_resid_l1_preset005"
    assert issue184.candidate == "rls_head_resid_l1_noreset"
    assert [stage.key for stage in issue184.stages] == ["screen_60"]
    assert issue184.stages[0].seeds == (0, 1, 2)
    assert issue184.stages[0].n_tasks == 60

    issue14 = _PROTOCOLS["issue14-v2"]
    assert issue14.issue == 14
    assert issue14.namespace == "rls_l2init_v2"
    assert issue14.control == "rls_head_resid_l1_preset005"
    assert issue14.candidate == "rls_head_resid_l1_preset005_l2init"
    assert [(stage.key, stage.seeds, stage.n_tasks) for stage in issue14.stages] == [
        ("screen_60", (20, 21, 22), 60),
        ("confirm_200_tuning", (20, 21, 22), 200),
        ("confirm_200_evaluation", tuple(range(23, 40)), 200),
    ]
    assert issue14.max_shards == 46


def test_owner_lines_bind_source_runner_data_home_and_frozen_plan() -> None:
    protocol = _PROTOCOLS["issue184"]
    amendment = _amendment_line(protocol, **_binding())
    amendment_sha256 = _canonical_sha256(amendment)
    authorization = _authorization_line(
        protocol,
        amendment_comment_id=123,
        amendment_sha256=amendment_sha256,
        **_binding(),
    )

    assert amendment == (
        "ASI_LOCAL_PREREG_AMENDMENT_V1 issue=184 protocol=issue184 "
        f"source={'1' * 40} tree={'2' * 40} uv_lock_sha256={'3' * 64} "
        f"benchmark_blob_sha1={'4' * 40} ref=ipmnist-prereg-example "
        f"runner_receipt_sha256={'5' * 64} data_home_sha256={'6' * 64} "
        "namespace=outputs/ipmnist_screening/rls_preset_ablation_r1 "
        f"plan_sha256={_canonical_sha256(protocol)} compute=uncompensated"
    )
    assert authorization.endswith(
        "amendment_comment_id=123 "
        f"amendment_sha256={amendment_sha256} protocol_approval=approved "
        "seed_budget=approved compute=authorized-uncompensated"
    )


def _owner_comment(
    comment_id: int, body: str, timestamp: str, *, issue: int = 184
) -> dict[str, Any]:
    return {
        "id": comment_id,
        "body": body,
        "user": {"id": 18_633_264, "login": "lalalune"},
        "author_association": "MEMBER",
        "created_at": timestamp,
        "updated_at": timestamp,
        "html_url": (
            f"https://github.com/elizaOS/asi/issues/{issue}"
            f"#issuecomment-{comment_id}"
        ),
    }


def _comments() -> tuple[list[dict[str, Any]], dt.datetime]:
    protocol = _PROTOCOLS["issue184"]
    amendment = _amendment_line(protocol, **_binding())
    amendment_comment = _owner_comment(123, amendment, "2026-08-16T09:00:00Z")
    authorization = _authorization_line(
        protocol,
        amendment_comment_id=123,
        amendment_sha256=_canonical_sha256(amendment),
        **_binding(),
    )
    return [
        amendment_comment,
        _owner_comment(456, authorization, "2026-08-16T09:01:00Z"),
    ], dt.datetime(2026, 8, 16, 9, 2, tzinfo=dt.UTC)


def test_owner_records_require_exact_unedited_strictly_ordered_comments() -> None:
    comments, launch_time = _comments()
    receipt = _verify_owner_records(
        _PROTOCOLS["issue184"],
        comments=comments,
        launch_time=launch_time,
        **_binding(),
    )
    assert receipt["amendment_comment_id"] == 123
    assert receipt["authorization_comment_id"] == 456
    assert receipt["amendment_created_at"] < receipt["authorization_created_at"]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "edited",
        "wrong-owner",
        "wrong-url",
        "equal-time",
        "after-launch",
        "bool-id",
    ],
)
def test_owner_records_reject_ambiguous_or_malformed_comments(mutation: str) -> None:
    comments, launch_time = _comments()
    if mutation == "duplicate":
        comments.append({**comments[0], "id": 124})
    elif mutation == "edited":
        comments[1]["updated_at"] = "2026-08-16T09:01:01Z"
    elif mutation == "wrong-owner":
        comments[0]["user"] = {"id": 1, "login": "lalalune"}
    elif mutation == "wrong-url":
        comments[0]["html_url"] = "https://example.invalid/comments/123"
    elif mutation == "equal-time":
        comments[1]["created_at"] = comments[1]["updated_at"] = "2026-08-16T09:00:00Z"
    elif mutation == "after-launch":
        comments[1]["created_at"] = comments[1]["updated_at"] = "2026-08-16T09:02:00Z"
    else:
        comments[0]["id"] = True

    with pytest.raises(RuntimeError):
        _verify_owner_records(
            _PROTOCOLS["issue184"],
            comments=comments,
            launch_time=launch_time,
            **_binding(),
        )


def test_owner_records_reject_near_match_instead_of_parsing_it_permissively() -> None:
    comments, launch_time = _comments()
    comments[0]["body"] += " extra=field"
    with pytest.raises(RuntimeError, match="amendment"):
        _verify_owner_records(
            _PROTOCOLS["issue184"],
            comments=comments,
            launch_time=launch_time,
            **_binding(),
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", (0,)),
        ("0-3", (0, 1, 2, 3)),
        ("0-2,5,7-8", (0, 1, 2, 5, 7, 8)),
    ],
)
def test_cpuset_parser_accepts_only_canonical_sets(value: str, expected: tuple[int, ...]) -> None:
    assert _parse_cpuset(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "01", "2-1", "1,1", "0-2,2", "2,1", "0, 1", "-1", "true"],
)
def test_cpuset_parser_rejects_ambiguous_or_noncanonical_sets(value: str) -> None:
    with pytest.raises(ValueError, match="cpuset"):
        _parse_cpuset(value)


@pytest.mark.parametrize(
    ("mean_diff", "diffs", "expected"),
    [
        (0.002000001, (0.001, 0.002, 0.003), "no_reset_win"),
        (0.002, (0.001, 0.002, 0.003), "inconclusive"),
        (-0.002000001, (-0.001, -0.002, -0.003), "reset_load_bearing"),
        (-0.002, (-0.001, -0.002, -0.003), "inconclusive"),
        (0.001, (0.0015, -0.0015, 0.0), "practical_equivalence"),
        (-0.001, (0.0015, -0.0015, 0.0), "practical_equivalence"),
    ],
)
def test_issue184_outcome_boundaries_are_frozen(
    mean_diff: float, diffs: tuple[float, ...], expected: str
) -> None:
    assert _classify_issue184(mean_diff, diffs) == expected


@pytest.mark.parametrize(
    ("mean_diff", "diffs", "expected"),
    [
        (0.002000001, (0.001, 0.002, 0.003), True),
        (0.002, (0.001, 0.002, 0.003), False),
        (0.003, (0.001, 0.0, 0.003), False),
    ],
)
def test_issue14_gate_is_strict(
    mean_diff: float, diffs: tuple[float, ...], expected: bool
) -> None:
    assert _l2init_gate_passes(mean_diff, diffs) is expected


def test_outcome_functions_reject_type_coercion_and_nonfinite_values() -> None:
    for value in (True, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError), match="finite|float"):
            _l2init_gate_passes(value, (0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="coverage"):
        _classify_issue184(0.0, (0.0, 0.0))


@pytest.mark.parametrize("occupied_kind", ["file", "directory", "symlink", "dangling"])
def test_namespace_claim_refuses_every_occupied_inode(
    tmp_path: Path, occupied_kind: str
) -> None:
    target = tmp_path / "namespace"
    if occupied_kind == "file":
        target.write_text("occupied", encoding="utf-8")
    elif occupied_kind == "directory":
        target.mkdir()
    elif occupied_kind == "symlink":
        destination = tmp_path / "destination"
        destination.mkdir()
        target.symlink_to(destination, target_is_directory=True)
    else:
        target.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(FileExistsError, match="occupied"):
        _claim_namespace(target)


def test_namespace_claim_is_atomic_and_a_second_launch_stays_rejected(tmp_path: Path) -> None:
    target = tmp_path / "namespace"
    _claim_namespace(target)
    assert target.is_dir() and not target.is_symlink()
    with pytest.raises(FileExistsError, match="occupied"):
        _claim_namespace(target)


def test_strict_json_and_exclusive_writer_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _strict_json(duplicate)

    overflow = tmp_path / "overflow.json"
    overflow.write_text('{"value": 1e999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        _strict_json(overflow)

    receipt = tmp_path / "receipt.json"
    _write_json_exclusive(receipt, {"value": 1})
    with pytest.raises(FileExistsError):
        _write_json_exclusive(receipt, {"value": 2})
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"value": 1}
    assert not os.path.islink(receipt)


def _screening_environment() -> dict[str, Any]:
    return {
        "schema": "alberta.ipmnist_screening.runtime.v1",
        "python": {"implementation": "CPython", "version": "3.12.3"},
        "platform": {"system": "Linux", "release": "test", "machine": "x86_64"},
        "packages": {
            "chex": "0.1.92",
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "numpy": "2.5.1",
            "scikit-learn": "1.9.0",
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
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_DEFAULT_MATMUL_PRECISION": None,
            "JAX_ENABLE_X64": "false",
            "JAX_PLATFORM_NAME": "cpu",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        },
    }


def _runner_receipt() -> dict[str, Any]:
    screening = _screening_environment()
    return {
        "schema": "asi.ipmnist_local_prereg.runner.v1",
        "runner": "local-beast-linux-x86_64-cpu",
        "platform": screening["platform"],
        "cpu": {
            "model": "Intel Test CPU",
            "requested_cpuset": "0-2",
            "effective_cpuset": [0, 1, 2],
        },
        "python_optimization_level": 0,
        "python": screening["python"],
        "packages": screening["packages"],
        "jax": screening["jax"],
        "process_environment": {
            **screening["process_environment"],
            "PYTHONHASHSEED": "0",
            "PYTHONOPTIMIZE": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        "screening_environment": screening,
        "cache_contract": {
            "relative_path": (
                "openml/openml.org/data/v1/download/52667/mnist_784.arff.gz"
            ),
            "size_bytes": 15_469_256,
            "sha256": "fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78",
        },
    }


def _repository_identity() -> dict[str, Any]:
    return {
        "schema": "asi.ipmnist_local_prereg.source.v1",
        "source": "1" * 40,
        "tree": "2" * 40,
        "uv_lock_sha256": "3" * 64,
        "benchmark_blob_sha1": "4" * 40,
        "source_provenance": {
            "schema": "alberta.ipmnist_screening.source_provenance.v1",
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "git_object_format": "sha1",
            "relevant_source_scope": "tracked:alberta_framework/**,pyproject.toml,uv.lock",
            "relevant_source_file_count": 3,
            "relevant_source_sha256": "7" * 64,
            "uv_lock_sha256": "3" * 64,
            "worktree_clean": True,
        },
    }


def test_runner_receipt_binds_linux_cpu_cpuset_packages_jax_and_environment() -> None:
    receipt = _runner_receipt()
    assert _validate_runner_receipt(receipt, expected_cpuset="0-2") == receipt

    for mutation in (
        "system",
        "cpuset",
        "package",
        "jax",
        "jax-type",
        "optimization",
        "environment",
    ):
        changed = json.loads(json.dumps(receipt))
        if mutation == "system":
            changed["platform"]["system"] = "Darwin"
        elif mutation == "cpuset":
            changed["cpu"]["effective_cpuset"] = [0, 1]
        elif mutation == "package":
            changed["packages"]["jax"] = "0.11.1"
        elif mutation == "jax":
            changed["jax"]["config"]["jax_disable_jit"] = True
        elif mutation == "jax-type":
            changed["jax"]["config"]["jax_disable_jit"] = 0
        elif mutation == "optimization":
            changed["python_optimization_level"] = False
        else:
            changed["process_environment"]["OMP_NUM_THREADS"] = 1
        with pytest.raises(ValueError):
            _validate_runner_receipt(changed, expected_cpuset="0-2")


def _launch_comments(
    protocol: Any, runner: dict[str, Any], data_home: Path
) -> tuple[list[dict[str, Any]], dt.datetime]:
    binding = {
        "source": "1" * 40,
        "tree": "2" * 40,
        "uv_lock_sha256": "3" * 64,
        "benchmark_blob_sha1": "4" * 40,
        "ref_name": "ipmnist-prereg-example",
        "runner_receipt_sha256": _receipt_sha256(runner),
        "data_home_sha256": _canonical_sha256(str(data_home.absolute())),
    }
    amendment = _amendment_line(protocol, **binding)
    authorization = _authorization_line(
        protocol,
        amendment_comment_id=123,
        amendment_sha256=_canonical_sha256(amendment),
        **binding,
    )
    return [
        _owner_comment(
            123, amendment, "2026-08-16T09:00:00Z", issue=protocol.issue
        ),
        _owner_comment(
            456, authorization, "2026-08-16T09:01:00Z", issue=protocol.issue
        ),
    ], dt.datetime(2026, 8, 16, 9, 2, tzinfo=dt.UTC)


def test_launch_atomically_claims_namespace_and_writes_bound_receipts(tmp_path: Path) -> None:
    (tmp_path / "outputs/ipmnist_screening").mkdir(parents=True)
    runner = _runner_receipt()
    data_home = tmp_path / "openml-cache"
    comments, launch_time = _launch_comments(_PROTOCOLS["issue184"], runner, data_home)
    namespace = _claim_local_launch(
        protocol_key="issue184",
        root=tmp_path,
        repository="elizaOS/asi",
        ref_name="ipmnist-prereg-example",
        cpuset="0-2",
        data_home=data_home,
        token="token",
        launch_time=launch_time,
        runner_receipt=runner,
        repository_identity=_repository_identity(),
        comments=comments,
        tag_payload={
            "ref": "refs/tags/ipmnist-prereg-example",
            "object": {"type": "commit", "sha": "1" * 40},
        },
    )
    assert namespace == tmp_path / "outputs/ipmnist_screening/rls_preset_ablation_r1"
    launch = _strict_json(namespace / "launch.v1.json")
    assert launch["authorization"]["authorization_comment_id"] == 456
    assert launch["runner_receipt_sha256"] == _receipt_sha256(runner)
    assert _strict_json(namespace / "runner.v1.json") == runner

    with pytest.raises(FileExistsError, match="occupied"):
        _claim_local_launch(
            protocol_key="issue184",
            root=tmp_path,
            repository="elizaOS/asi",
            ref_name="ipmnist-prereg-example",
            cpuset="0-2",
            data_home=data_home,
            token="token",
            launch_time=launch_time,
            runner_receipt=runner,
            repository_identity=_repository_identity(),
            comments=comments,
            tag_payload={
                "ref": "refs/tags/ipmnist-prereg-example",
                "object": {"type": "commit", "sha": "1" * 40},
            },
        )


def test_launch_rejects_existing_cache_or_noncanonical_remote_tag(tmp_path: Path) -> None:
    (tmp_path / "outputs/ipmnist_screening").mkdir(parents=True)
    runner = _runner_receipt()
    data_home = tmp_path / "openml-cache"
    cache = data_home / runner["cache_contract"]["relative_path"]
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"already materialized")
    comments, launch_time = _launch_comments(_PROTOCOLS["issue184"], runner, data_home)
    kwargs = {
        "protocol_key": "issue184",
        "root": tmp_path,
        "repository": "elizaOS/asi",
        "ref_name": "ipmnist-prereg-example",
        "cpuset": "0-2",
        "data_home": data_home,
        "token": "token",
        "launch_time": launch_time,
        "runner_receipt": runner,
        "repository_identity": _repository_identity(),
        "comments": comments,
        "tag_payload": {
            "ref": "refs/tags/ipmnist-prereg-example",
            "object": {"type": "commit", "sha": "1" * 40},
        },
    }
    with pytest.raises(FileExistsError, match="cache"):
        _claim_local_launch(**kwargs)
    assert not os.path.lexists(
        tmp_path / "outputs/ipmnist_screening/rls_preset_ablation_r1"
    )

    cache.unlink()
    kwargs["tag_payload"] = {
        "ref": "refs/tags/ipmnist-prereg-example",
        "object": {"type": "tag", "sha": "1" * 40},
    }
    with pytest.raises(RuntimeError, match="lightweight"):
        _claim_local_launch(**kwargs)

    kwargs["tag_payload"] = {
        "ref": "refs/tags/ipmnist-prereg-example",
        "object": {"type": "commit", "sha": "1" * 40},
    }
    kwargs["repository"] = "lalalune/alberta"
    with pytest.raises(RuntimeError, match="repository"):
        _claim_local_launch(**kwargs)


def test_cache_receipt_hashes_exact_bytes_and_refuses_overwrite(tmp_path: Path) -> None:
    namespace = tmp_path / "outputs/ipmnist_screening/rls_preset_ablation_r1"
    namespace.mkdir(parents=True)
    data_home = tmp_path / "openml-cache"
    relative = Path("openml/cache.bin")
    cache = data_home / relative
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"canonical-cache")
    launch = {
        "schema": "asi.ipmnist_local_prereg.launch.v1",
        "protocol_key": "issue184",
        "data_home": str(data_home.absolute()),
        "data_home_sha256": _canonical_sha256(str(data_home.absolute())),
        "cache_contract": {
            "relative_path": relative.as_posix(),
            "size_bytes": len(b"canonical-cache"),
            "sha256": _canonical_sha256(b"canonical-cache"),
        },
    }
    _write_json_exclusive(namespace / "launch.v1.json", launch)
    receipt = _record_cache_receipt(
        protocol_key="issue184",
        root=tmp_path,
        data_home=data_home,
        expected_relative_path=relative,
        expected_size=len(b"canonical-cache"),
        expected_sha256=_canonical_sha256(b"canonical-cache"),
    )
    assert receipt["size_bytes"] == len(b"canonical-cache")
    assert receipt["sha256"] == _canonical_sha256(b"canonical-cache")
    with pytest.raises(FileExistsError):
        _record_cache_receipt(
            protocol_key="issue184",
            root=tmp_path,
            data_home=data_home,
            expected_relative_path=relative,
            expected_size=len(b"canonical-cache"),
            expected_sha256=_canonical_sha256(b"canonical-cache"),
        )


def _dataset_provenance() -> dict[str, Any]:
    return {
        "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
        "source": {
            "provider": "openml",
            "name": "mnist_784",
            "version": 1,
            "row_start": 0,
            "row_stop_exclusive": 60_000,
        },
        "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
        "x": {"dtype": "<f4", "shape": [60_000, 784], "sha256": "8" * 64},
        "y": {"dtype": "<i4", "shape": [60_000], "sha256": "9" * 64},
    }


def _write_protocol_receipts(root: Path, protocol_key: str) -> Path:
    protocol = _PROTOCOLS[protocol_key]
    namespace = root / "outputs/ipmnist_screening" / cast(str, protocol.namespace)
    namespace.mkdir(parents=True)
    runner = _runner_receipt()
    _write_json_exclusive(namespace / "runner.v1.json", runner)
    data_home = root / "openml-cache"
    data_home_text = str(data_home.absolute())
    binding = {
        "source": "1" * 40,
        "tree": "2" * 40,
        "uv_lock_sha256": "3" * 64,
        "benchmark_blob_sha1": "4" * 40,
        "ref_name": "ipmnist-prereg-example",
        "runner_receipt_sha256": _receipt_sha256(runner),
        "data_home_sha256": _canonical_sha256(data_home_text),
    }
    amendment = _amendment_line(protocol, **binding)
    authorization_line = _authorization_line(
        protocol,
        amendment_comment_id=123,
        amendment_sha256=_canonical_sha256(amendment),
        **binding,
    )
    authorization = {
        "amendment_comment_id": 123,
        "amendment_comment_url": (
            f"https://github.com/elizaOS/asi/issues/{protocol.issue}#issuecomment-123"
        ),
        "amendment_created_at": "2026-08-16T09:00:00Z",
        "amendment_updated_at": "2026-08-16T09:00:00Z",
        "amendment_line": amendment,
        "amendment_sha256": _canonical_sha256(amendment),
        "authorization_comment_id": 456,
        "authorization_comment_url": (
            f"https://github.com/elizaOS/asi/issues/{protocol.issue}#issuecomment-456"
        ),
        "authorization_created_at": "2026-08-16T09:01:00Z",
        "authorization_updated_at": "2026-08-16T09:01:00Z",
        "authorization_line": authorization_line,
        "authorization_sha256": _canonical_sha256(authorization_line),
    }
    launch = {
        "schema": "asi.ipmnist_local_prereg.launch.v1",
        "protocol_key": protocol.key,
        "protocol": asdict(protocol),
        "plan_sha256": _canonical_sha256(protocol),
        "repository": _repository_identity(),
        "ref_name": "ipmnist-prereg-example",
        "data_home": data_home_text,
        "data_home_sha256": _canonical_sha256(data_home_text),
        "cache_contract": runner["cache_contract"],
        "runner_receipt": "runner.v1.json",
        "runner_receipt_sha256": _receipt_sha256(runner),
        "authorization": authorization,
        "launch_created_at": "2026-08-16T09:02:00Z",
        "dataset_accessed": False,
        "rerun_allowed": False,
    }
    _write_json_exclusive(namespace / "launch.v1.json", launch)
    launch_raw = (namespace / "launch.v1.json").read_bytes()
    cache = {
        "schema": "asi.ipmnist_local_prereg.cache.v1",
        "protocol_key": protocol.key,
        "data_home": data_home_text,
        "data_home_sha256": _canonical_sha256(data_home_text),
        "relative_path": runner["cache_contract"]["relative_path"],
        "size_bytes": runner["cache_contract"]["size_bytes"],
        "sha256": runner["cache_contract"]["sha256"],
        "launch_receipt_sha256": _canonical_sha256(launch_raw),
        "checked_at": "2026-08-16T09:03:00Z",
    }
    _write_json_exclusive(namespace / "cache.v1.json", cache)
    return namespace


def _benchmark_argv(
    *,
    root: Path,
    protocol: Any,
    stage: Any,
    config_name: str,
    seed: int,
    data_home: str,
) -> list[str]:
    output = (
        root
        / "outputs/ipmnist_screening"
        / cast(str, protocol.namespace)
        / cast(str, stage.key)
        / "shards"
        / f"{config_name}_seed{seed}.json"
    ).relative_to(root)
    return [
        "run",
        "--config-name",
        config_name,
        "--seed",
        str(seed),
        "--n-tasks",
        str(stage.n_tasks),
        "--task-length",
        "5000",
        "--data-home",
        data_home,
        "--out",
        output.as_posix(),
        "--progress-every",
        "10",
        "--noise-mode",
        "step",
    ]


def _write_execution_bindings(
    root: Path,
    protocol: Any,
    stage: Any,
    paths: list[Path],
    *,
    summary_created_unix: float,
) -> None:
    namespace = root / "outputs/ipmnist_screening" / cast(str, protocol.namespace)
    runner, runner_raw = _strict_json_bytes(namespace / "runner.v1.json")
    launch, launch_raw = _strict_json_bytes(namespace / "launch.v1.json")
    cache, cache_raw = _strict_json_bytes(namespace / "cache.v1.json")
    bindings = namespace / cast(str, stage.key) / "bindings"
    bindings.mkdir()
    cache_unix = dt.datetime.fromisoformat(
        cast(str, cache["checked_at"]).replace("Z", "+00:00")
    ).timestamp()
    for shard_path in paths:
        shard, shard_raw = _strict_json_bytes(shard_path)
        created_unix = cast(float, shard["created_unix"])
        started_unix = (cache_unix + created_unix) / 2.0
        finished_unix = (created_unix + summary_created_unix) / 2.0
        bound_unix = (finished_unix + summary_created_unix) / 2.0
        config_name = cast(str, shard["config_name"])
        seed = cast(int, shard["seed"])
        attempt = _execution_attempt_payload(
            root=root,
            protocol=protocol,
            stage=stage,
            config_name=config_name,
            seed=seed,
            shard_path=shard_path,
            runner=runner,
            runner_raw=runner_raw,
            launch=launch,
            launch_raw=launch_raw,
            cache=cache,
            cache_raw=cache_raw,
            started_unix=started_unix,
        )
        attempt_path = bindings / f"{config_name}_seed{seed}.attempt.v1.json"
        _write_json_exclusive(attempt_path, attempt)
        attempt_raw = attempt_path.read_bytes()
        receipt = {
            "schema": "asi.ipmnist_local_prereg.shard_execution.v1",
            "protocol_key": protocol.key,
            "stage_key": stage.key,
            "config_name": config_name,
            "seed": seed,
            "repository": launch["repository"],
            "runner_context": runner,
            "runner_receipt_sha256": hashlib.sha256(runner_raw).hexdigest(),
            "launch_receipt_sha256": hashlib.sha256(launch_raw).hexdigest(),
            "cache_context": cache,
            "cache_receipt_sha256": hashlib.sha256(cache_raw).hexdigest(),
            "attempt_receipt": {
                "path": attempt_path.relative_to(root).as_posix(),
                "size_bytes": len(attempt_raw),
                "sha256": hashlib.sha256(attempt_raw).hexdigest(),
            },
            "data_home": launch["data_home"],
            "data_home_sha256": launch["data_home_sha256"],
            "cache_contract": launch["cache_contract"],
            "shard": {
                "path": shard_path.relative_to(root).as_posix(),
                "size_bytes": len(shard_raw),
                "sha256": hashlib.sha256(shard_raw).hexdigest(),
                "created_unix": created_unix,
            },
            "started_unix": started_unix,
            "finished_unix": finished_unix,
            "bound_unix": bound_unix,
            "benchmark_argv": _benchmark_argv(
                root=root,
                protocol=protocol,
                stage=stage,
                config_name=config_name,
                seed=seed,
                data_home=cast(str, launch["data_home"]),
            ),
            "rerun_allowed": False,
        }
        _write_json_exclusive(
            bindings / f"{config_name}_seed{seed}.execution.v1.json", receipt
        )


def _write_stage(
    root: Path,
    protocol_key: str,
    stage_key: str,
    *,
    differences: tuple[float, ...],
    dataset_provenance: dict[str, Any] | None = None,
    created_unix: float | None = None,
) -> Path:
    from alberta_framework.benchmarks.ipmnist_screening import (
        ScreeningRunResult,
        merge_shards,
        screening_spec,
        shard_payload,
    )
    from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

    protocol = _PROTOCOLS[protocol_key]
    stage = next(value for value in protocol.stages if value.key == stage_key)
    assert len(differences) == len(stage.seeds)
    dataset = _dataset_provenance() if dataset_provenance is None else dataset_provenance
    namespace = root / "outputs/ipmnist_screening" / cast(str, protocol.namespace)
    stage_name = cast(str, stage.key)
    shards_dir = namespace / stage_name / "shards"
    shards_dir.mkdir(parents=True)
    config = IPMNISTConfig(
        n_tasks=stage.n_tasks,
        task_length=5_000,
        input_dim=784,
        hidden1=300,
        hidden2=150,
        n_classes=10,
    )
    paths: list[Path] = []
    for seed, difference in zip(stage.seeds, differences, strict=True):
        for arm, accuracy in (
            (protocol.control, 0.5),
            (protocol.candidate, 0.5 + difference),
        ):
            spec = screening_spec(arm)
            result = ScreeningRunResult(
                config_name=arm,
                base_learner=spec.base_learner,
                hyperparameters=dict(spec.hyperparameters),
                seed=seed,
                config=config,
                per_task_accuracy=np.full(stage.n_tasks, accuracy, dtype=np.float64),
                per_task_loss=np.full(stage.n_tasks, 0.5, dtype=np.float64),
                per_task_plasticity=np.full(stage.n_tasks, 0.5, dtype=np.float64),
                wall_clock_seconds=1.0,
            )
            path = shards_dir / f"{arm}_seed{seed}.json"
            payload = shard_payload(
                result,
                source_provenance=_repository_identity()["source_provenance"],
                dataset_provenance=dataset,
                environment=_screening_environment(),
            )
            if created_unix is not None:
                payload["created_unix"] = created_unix
            path.write_text(
                json.dumps(payload, allow_nan=False, sort_keys=True), encoding="utf-8"
            )
            paths.append(path)
    prior = Path.cwd()
    os.chdir(root)
    try:
        summary = merge_shards(
            [path.relative_to(root) for path in paths],
            control_name=protocol.control,
            slope_window=15,
        )
    finally:
        os.chdir(prior)
    summary_path = namespace / stage_name / "summary.json"
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    _write_execution_bindings(
        root,
        protocol,
        stage,
        paths,
        summary_created_unix=cast(float, summary["created_unix"]),
    )
    return summary_path


def test_run_shard_wrapper_binds_one_exact_in_process_execution(tmp_path: Path) -> None:
    from alberta_framework.benchmarks.ipmnist_screening import (
        ScreeningRunResult,
        screening_spec,
        shard_payload,
    )
    from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

    namespace = _write_protocol_receipts(tmp_path, "issue184")
    protocol = _PROTOCOLS["issue184"]
    stage = protocol.stages[0]
    arm = cast(str, protocol.control)
    seed = 0
    calls: list[list[str]] = []

    def fake_benchmark(argv: Any) -> int:
        arguments = list(argv)
        calls.append(arguments)
        assert arguments == _benchmark_argv(
            root=tmp_path,
            protocol=protocol,
            stage=stage,
            config_name=arm,
            seed=seed,
            data_home=str((tmp_path / "openml-cache").absolute()),
        )
        config = IPMNISTConfig(n_tasks=60, task_length=5_000)
        spec = screening_spec(arm)
        result = ScreeningRunResult(
            config_name=arm,
            base_learner=spec.base_learner,
            hyperparameters=dict(spec.hyperparameters),
            seed=seed,
            config=config,
            per_task_accuracy=np.full(60, 0.5, dtype=np.float64),
            per_task_loss=np.full(60, 0.5, dtype=np.float64),
            per_task_plasticity=np.full(60, 0.5, dtype=np.float64),
            wall_clock_seconds=1.0,
        )
        payload = shard_payload(
            result,
            source_provenance=_repository_identity()["source_provenance"],
            dataset_provenance=_dataset_provenance(),
            environment=_screening_environment(),
        )
        payload["created_unix"] = 1_786_880_002.0
        output = tmp_path / arguments[arguments.index("--out") + 1]
        output.write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True), encoding="utf-8"
        )
        return 0

    times = iter((1_786_880_000.0, 1_786_880_004.0, 1_786_880_005.0))
    receipt = _run_local_shard(
        protocol_key="issue184",
        root=tmp_path,
        stage_key="screen_60",
        config_name=arm,
        seed=seed,
        repository_identity=_repository_identity(),
        runner_receipt=_runner_receipt(),
        verify_cache_file=False,
        benchmark_main=fake_benchmark,
        clock=lambda: next(times),
    )

    assert len(calls) == 1
    assert receipt["runner_context"] == _runner_receipt()
    assert receipt["cache_context"] == _strict_json(namespace / "cache.v1.json")
    assert receipt["data_home"] == str((tmp_path / "openml-cache").absolute())
    assert receipt["shard"]["sha256"] == hashlib.sha256(
        (namespace / f"screen_60/shards/{arm}_seed0.json").read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError, match="already consumed"):
        _run_local_shard(
            protocol_key="issue184",
            root=tmp_path,
            stage_key="screen_60",
            config_name=arm,
            seed=seed,
            repository_identity=_repository_identity(),
            runner_receipt=_runner_receipt(),
            verify_cache_file=False,
            benchmark_main=fake_benchmark,
        )
    assert len(calls) == 1


def test_failed_shard_attempt_is_consumed_before_the_benchmark_can_be_retried(
    tmp_path: Path,
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    protocol = _PROTOCOLS["issue184"]
    arm = cast(str, protocol.control)
    calls: list[list[str]] = []

    def failing_benchmark(argv: Any) -> int:
        calls.append(list(argv))
        return 1

    kwargs = {
        "protocol_key": "issue184",
        "root": tmp_path,
        "stage_key": "screen_60",
        "config_name": arm,
        "seed": 0,
        "repository_identity": _repository_identity(),
        "runner_receipt": _runner_receipt(),
        "verify_cache_file": False,
        "benchmark_main": failing_benchmark,
        "clock": lambda: 1_786_880_000.0,
    }
    with pytest.raises(RuntimeError, match="exit code 1"):
        _run_local_shard(**kwargs)
    attempt = namespace / f"screen_60/bindings/{arm}_seed0.attempt.v1.json"
    assert attempt.is_file()
    assert not (namespace / f"screen_60/shards/{arm}_seed0.json").exists()
    assert not (
        namespace / f"screen_60/bindings/{arm}_seed0.execution.v1.json"
    ).exists()

    with pytest.raises(FileExistsError, match="already consumed"):
        _run_local_shard(**kwargs)
    assert len(calls) == 1


def test_concurrent_shard_attempts_claim_once_before_entering_the_benchmark(
    tmp_path: Path,
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    protocol = _PROTOCOLS["issue184"]
    arm = cast(str, protocol.control)
    entered = threading.Event()
    release = threading.Event()
    calls: list[list[str]] = []
    errors: list[BaseException] = []

    def blocked_benchmark(argv: Any) -> int:
        calls.append(list(argv))
        entered.set()
        assert release.wait(timeout=5.0)
        return 1

    kwargs = {
        "protocol_key": "issue184",
        "root": tmp_path,
        "stage_key": "screen_60",
        "config_name": arm,
        "seed": 0,
        "repository_identity": _repository_identity(),
        "runner_receipt": _runner_receipt(),
        "verify_cache_file": False,
        "benchmark_main": blocked_benchmark,
        "clock": lambda: 1_786_880_000.0,
    }

    def invoke() -> None:
        try:
            _run_local_shard(**kwargs)
        except BaseException as exc:
            errors.append(exc)

    original_cwd = Path.cwd()
    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    try:
        first.start()
        assert entered.wait(timeout=5.0)
        second.start()
        second.join(timeout=1.0)
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
    finally:
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        os.chdir(original_cwd)

    assert not first.is_alive() and not second.is_alive()
    assert len(calls) == 1
    assert sum(isinstance(exc, FileExistsError) for exc in errors) == 1
    assert sum(isinstance(exc, RuntimeError) for exc in errors) == 1
    assert (
        namespace / f"screen_60/bindings/{arm}_seed0.attempt.v1.json"
    ).is_file()


def _write_combined_summary(root: Path) -> Path:
    from alberta_framework.benchmarks.ipmnist_screening import merge_shards

    protocol = _PROTOCOLS["issue14-v2"]
    namespace = root / "outputs/ipmnist_screening" / cast(str, protocol.namespace)
    combined = namespace / "confirm_200_all"
    combined.mkdir()
    paths = sorted((namespace / "confirm_200_tuning/shards").iterdir()) + sorted(
        (namespace / "confirm_200_evaluation/shards").iterdir()
    )
    prior = Path.cwd()
    os.chdir(root)
    try:
        summary = merge_shards(
            [path.relative_to(root) for path in paths],
            control_name=protocol.control,
            slope_window=15,
        )
    finally:
        os.chdir(prior)
    path = combined / "summary.json"
    path.write_text(json.dumps(summary, allow_nan=False, sort_keys=True), encoding="utf-8")
    return path


def _validate(root: Path, protocol_key: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _validate_result_bundle(
            protocol_key=protocol_key,
            root=root,
            repository_identity=_repository_identity(),
            runner_receipt=_runner_receipt(),
            verify_cache_file=False,
        ),
    )


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        (0.003, "no_reset_win"),
        (-0.003, "reset_load_bearing"),
        (0.0005, "practical_equivalence"),
        (0.0017, "inconclusive"),
    ],
)
def test_issue184_bundle_reconstructs_exact_summary_and_outcome(
    tmp_path: Path, difference: float, expected: str
) -> None:
    _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(
        tmp_path,
        "issue184",
        "screen_60",
        differences=(difference,) * 3,
    )
    result = _validate(tmp_path, "issue184")
    assert result["outcome"] == expected
    assert result["n_shards"] == 6


@pytest.mark.parametrize("tamper", ["metric", "manifest-path", "manifest-hash"])
def test_result_validation_rejects_summary_or_manifest_tamper(
    tmp_path: Path, tamper: str
) -> None:
    _write_protocol_receipts(tmp_path, "issue184")
    summary_path = _write_stage(
        tmp_path, "issue184", "screen_60", differences=(0.003,) * 3
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tamper == "metric":
        candidate = next(
            row
            for row in summary["results"]
            if row["config_name"] == "rls_head_resid_l1_noreset"
        )
        candidate["paired_vs_control"]["mean_diff"] = 0.5
    elif tamper == "manifest-path":
        summary["shard_manifest"][0]["path"] = "outputs/forged.json"
    else:
        summary["shard_manifest"][0]["sha256"] = "f" * 64
    summary_path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="reconstruction|manifest"):
        _validate(tmp_path, "issue184")


@pytest.mark.parametrize("coverage", ["missing", "extra", "filename-swap"])
def test_result_validation_rejects_nonexact_shard_coverage(
    tmp_path: Path, coverage: str
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    shards = namespace / "screen_60/shards"
    if coverage == "missing":
        next(iter(shards.iterdir())).unlink()
    elif coverage == "extra":
        (shards / "extra.json").write_text("{}", encoding="utf-8")
    else:
        first = shards / "rls_head_resid_l1_preset005_seed0.json"
        second = shards / "rls_head_resid_l1_noreset_seed1.json"
        first_raw, second_raw = first.read_bytes(), second.read_bytes()
        first.write_bytes(second_raw)
        second.write_bytes(first_raw)
    with pytest.raises(ValueError, match="coverage|filename/payload"):
        _validate(tmp_path, "issue184")


def test_result_validation_rejects_shard_source_runtime_or_dataset_drift(
    tmp_path: Path,
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    shard_path = namespace / "screen_60/shards/rls_head_resid_l1_noreset_seed0.json"
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["source_provenance"]["git_commit"] = "a" * 40
    shard_path.write_text(json.dumps(shard, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        _validate(tmp_path, "issue184")


def test_result_validation_binds_exact_runner_receipt_bytes(tmp_path: Path) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    runner_path = namespace / "runner.v1.json"
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner_path.write_text(json.dumps(runner, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="runner receipt SHA-256"):
        _validate(tmp_path, "issue184")


def test_issue14_stage1_rejection_is_terminal_and_forbids_later_stage(tmp_path: Path) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue14-v2")
    _write_stage(tmp_path, "issue14-v2", "screen_60", differences=(-0.001,) * 3)
    assert _validate(tmp_path, "issue14-v2")["outcome"] == "stage1_rejected"

    (namespace / "confirm_200_tuning").mkdir()
    with pytest.raises(ValueError, match="unexpected|terminal"):
        _validate(tmp_path, "issue14-v2")


def test_issue14_passed_stage_requires_the_next_exact_stage(tmp_path: Path) -> None:
    _write_protocol_receipts(tmp_path, "issue14-v2")
    _write_stage(tmp_path, "issue14-v2", "screen_60", differences=(0.003,) * 3)
    with pytest.raises(ValueError, match="confirm_200_tuning"):
        _validate(tmp_path, "issue14-v2")


def test_issue14_stage2_rejection_is_terminal(tmp_path: Path) -> None:
    _write_protocol_receipts(tmp_path, "issue14-v2")
    _write_stage(tmp_path, "issue14-v2", "screen_60", differences=(0.003,) * 3)
    _write_stage(
        tmp_path, "issue14-v2", "confirm_200_tuning", differences=(0.001,) * 3
    )
    result = _validate(tmp_path, "issue14-v2")
    assert result["outcome"] == "stage2_rejected"
    assert result["n_shards"] == 12


def test_issue14_stage2_must_be_created_only_after_stage1_gate(tmp_path: Path) -> None:
    _write_protocol_receipts(tmp_path, "issue14-v2")
    _write_stage(
        tmp_path, "issue14-v2", "confirm_200_tuning", differences=(0.001,) * 3
    )
    _write_stage(tmp_path, "issue14-v2", "screen_60", differences=(0.003,) * 3)
    with pytest.raises(ValueError, match="created after the prior gate"):
        _validate(tmp_path, "issue14-v2")


def test_issue14_stages_must_share_one_exact_dataset(tmp_path: Path) -> None:
    _write_protocol_receipts(tmp_path, "issue14-v2")
    _write_stage(tmp_path, "issue14-v2", "screen_60", differences=(0.003,) * 3)
    changed_dataset = _dataset_provenance()
    changed_dataset["x"]["sha256"] = "a" * 64
    _write_stage(
        tmp_path,
        "issue14-v2",
        "confirm_200_tuning",
        differences=(0.001,) * 3,
        dataset_provenance=changed_dataset,
    )
    with pytest.raises(ValueError, match="dataset"):
        _validate(tmp_path, "issue14-v2")


@pytest.mark.parametrize(
    ("evaluation_difference", "expected"),
    [(0.003, "win"), (0.001, "no_win")],
)
def test_issue14_full_bundle_reports_evaluation_and_combined_outcome(
    tmp_path: Path, evaluation_difference: float, expected: str
) -> None:
    _write_protocol_receipts(tmp_path, "issue14-v2")
    _write_stage(tmp_path, "issue14-v2", "screen_60", differences=(0.003,) * 3)
    _write_stage(
        tmp_path, "issue14-v2", "confirm_200_tuning", differences=(0.003,) * 3
    )
    _write_stage(
        tmp_path,
        "issue14-v2",
        "confirm_200_evaluation",
        differences=(evaluation_difference,) * 17,
    )
    _write_combined_summary(tmp_path)
    result = _validate(tmp_path, "issue14-v2")
    assert result["outcome"] == expected
    assert result["n_shards"] == 46
    assert result["evaluation"]["n_seeds"] == 17
    assert result["combined"]["n_seeds"] == 20


def test_result_publication_is_one_shot_after_success(tmp_path: Path) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    result = _validate_and_publish_result(
        protocol_key="issue184",
        root=tmp_path,
        repository_identity=_repository_identity(),
        runner_receipt=_runner_receipt(),
        verify_cache_file=False,
    )
    assert result["outcome"] == "no_reset_win"
    assert (namespace / "result-claim.v1.json").is_file()
    assert (namespace / "result.v1.json").is_file()
    assert _validate(tmp_path, "issue184") == result
    with pytest.raises(FileExistsError):
        _validate_and_publish_result(
            protocol_key="issue184",
            root=tmp_path,
            repository_identity=_repository_identity(),
            runner_receipt=_runner_receipt(),
            verify_cache_file=False,
        )


def test_failed_result_validation_consumes_the_only_attempt(tmp_path: Path) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue14-v2")
    _write_stage(tmp_path, "issue14-v2", "screen_60", differences=(0.003,) * 3)
    kwargs = {
        "protocol_key": "issue14-v2",
        "root": tmp_path,
        "repository_identity": _repository_identity(),
        "runner_receipt": _runner_receipt(),
        "verify_cache_file": False,
    }
    with pytest.raises(ValueError, match="confirm_200_tuning"):
        _validate_and_publish_result(**kwargs)
    assert (namespace / "result-claim.v1.json").is_file()
    assert not (namespace / "result.v1.json").exists()
    with pytest.raises(FileExistsError):
        _validate_and_publish_result(**kwargs)


def test_result_validation_rejects_shards_created_before_cache_receipt(
    tmp_path: Path,
) -> None:
    from alberta_framework.benchmarks.ipmnist_screening import merge_shards

    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    paths = sorted((namespace / "screen_60/shards").iterdir())
    for path in paths:
        shard = json.loads(path.read_text(encoding="utf-8"))
        shard["created_unix"] = 1.0
        path.write_text(json.dumps(shard, allow_nan=False, sort_keys=True), encoding="utf-8")
    prior = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary = merge_shards(
            [path.relative_to(tmp_path) for path in paths],
            control_name="rls_head_resid_l1_preset005",
            slope_window=15,
        )
    finally:
        os.chdir(prior)
    (namespace / "screen_60/summary.json").write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="after the cache receipt"):
        _validate(tmp_path, "issue184")


def _first_execution_binding(namespace: Path) -> Path:
    return (
        namespace
        / "screen_60/bindings/rls_head_resid_l1_noreset_seed0.execution.v1.json"
    )


def _first_execution_attempt(namespace: Path) -> Path:
    return (
        namespace
        / "screen_60/bindings/rls_head_resid_l1_noreset_seed0.attempt.v1.json"
    )


@pytest.mark.parametrize("receipt", ["attempt", "execution"])
def test_every_shard_requires_one_attempt_and_execution_binding(
    tmp_path: Path, receipt: str
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    path = (
        _first_execution_attempt(namespace)
        if receipt == "attempt"
        else _first_execution_binding(namespace)
    )
    path.unlink()
    with pytest.raises(ValueError, match="execution binding"):
        _validate(tmp_path, "issue184")


@pytest.mark.parametrize(
    "tamper",
    [
        "cpu-model",
        "cpuset",
        "pythonhashseed",
        "pythonoptimize",
        "xla-preallocate",
        "cache",
        "data-home",
        "runner-digest",
        "launch-digest",
        "cache-digest",
        "attempt-digest",
        "shard-digest",
    ],
)
def test_execution_binding_rejects_resigned_runner_cache_or_shard_drift(
    tmp_path: Path, tamper: str
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    binding_path = _first_execution_binding(namespace)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if tamper == "cpu-model":
        binding["runner_context"]["cpu"]["model"] = "Different CPU"
    elif tamper == "cpuset":
        binding["runner_context"]["cpu"]["effective_cpuset"] = [0, 1]
    elif tamper == "pythonhashseed":
        binding["runner_context"]["process_environment"]["PYTHONHASHSEED"] = "1"
    elif tamper == "pythonoptimize":
        binding["runner_context"]["process_environment"]["PYTHONOPTIMIZE"] = "1"
    elif tamper == "xla-preallocate":
        binding["runner_context"]["process_environment"][
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ] = "true"
    elif tamper == "cache":
        binding["cache_context"]["sha256"] = "a" * 64
    elif tamper == "data-home":
        binding["data_home"] = "/tmp/different-cache"
    elif tamper == "runner-digest":
        binding["runner_receipt_sha256"] = "a" * 64
    elif tamper == "launch-digest":
        binding["launch_receipt_sha256"] = "a" * 64
    elif tamper == "cache-digest":
        binding["cache_receipt_sha256"] = "a" * 64
    elif tamper == "attempt-digest":
        binding["attempt_receipt"]["sha256"] = "a" * 64
    else:
        binding["shard"]["sha256"] = "a" * 64
    binding_path.write_text(
        json.dumps(binding, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="execution binding"):
        _validate(tmp_path, "issue184")


@pytest.mark.parametrize(
    "created_unix",
    [
        dt.datetime(2026, 8, 16, 9, 1, tzinfo=dt.UTC).timestamp(),
        dt.datetime(2026, 8, 16, 9, 2, tzinfo=dt.UTC).timestamp(),
        dt.datetime(2026, 8, 16, 9, 2, 30, tzinfo=dt.UTC).timestamp(),
        dt.datetime(2026, 8, 16, 9, 3, tzinfo=dt.UTC).timestamp(),
    ],
)
def test_every_shard_must_be_strictly_after_launch_and_cache(
    tmp_path: Path, created_unix: float
) -> None:
    _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(
        tmp_path,
        "issue184",
        "screen_60",
        differences=(0.003,) * 3,
        created_unix=created_unix,
    )
    with pytest.raises(ValueError, match="after.*launch.*cache|after the cache receipt"):
        _validate(tmp_path, "issue184")


@pytest.mark.parametrize("offset", [0.0, -1.0])
def test_every_shard_must_be_strictly_before_its_summary(
    tmp_path: Path, offset: float
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    summary_path = _write_stage(
        tmp_path, "issue184", "screen_60", differences=(0.003,) * 3
    )
    shard_created = max(
        cast(float, _strict_json(path)["created_unix"])
        for path in (namespace / "screen_60/shards").iterdir()
    )
    summary = _strict_json(summary_path)
    summary["created_unix"] = shard_created + offset
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="summary.*strictly after.*shard"):
        _validate(tmp_path, "issue184")


def _publication_kwargs(root: Path, protocol_key: str = "issue184") -> dict[str, Any]:
    return {
        "protocol_key": protocol_key,
        "root": root,
        "repository_identity": _repository_identity(),
        "runner_receipt": _runner_receipt(),
        "verify_cache_file": False,
    }


def test_result_preflight_never_claims_through_a_root_symlink(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    namespace = _write_protocol_receipts(actual, "issue184")
    _write_stage(actual, "issue184", "screen_60", differences=(0.003,) * 3)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="root.*symlink"):
        _validate_and_publish_result(**_publication_kwargs(alias))
    assert not (namespace / "result-claim.v1.json").exists()


def test_result_preflight_never_claims_through_a_namespace_symlink(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external_namespace = _write_protocol_receipts(external, "issue184")
    _write_stage(external, "issue184", "screen_60", differences=(0.003,) * 3)
    safe = tmp_path / "safe"
    output_root = safe / "outputs/ipmnist_screening"
    output_root.mkdir(parents=True)
    (output_root / "rls_preset_ablation_r1").symlink_to(
        external_namespace, target_is_directory=True
    )

    with pytest.raises(ValueError, match="namespace.*symlink"):
        _validate_and_publish_result(**_publication_kwargs(safe))
    assert not (external_namespace / "result-claim.v1.json").exists()


def test_result_preflight_rejects_shard_symlink_before_claim(tmp_path: Path) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    shard = namespace / "screen_60/shards/rls_head_resid_l1_noreset_seed0.json"
    external = tmp_path / "external-shard.json"
    external.write_bytes(shard.read_bytes())
    shard.unlink()
    shard.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        _validate_and_publish_result(**_publication_kwargs(tmp_path))
    assert not (namespace / "result-claim.v1.json").exists()


@pytest.mark.parametrize(
    "claim",
    [
        {},
        {
            "schema": "asi.ipmnist_local_prereg.result_claim.v1",
            "protocol_key": "issue184",
            "claimed_at": "1970-01-01T00:00:01Z",
            "rerun_allowed": False,
        },
    ],
)
def test_result_validation_rejects_malformed_or_premature_claim(
    tmp_path: Path, claim: dict[str, Any]
) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    (namespace / "result-claim.v1.json").write_text(
        json.dumps(claim, allow_nan=False, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="result claim"):
        _validate(tmp_path, "issue184")


def test_published_result_tamper_is_rejected_on_strict_reload(tmp_path: Path) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    result = _validate_and_publish_result(
        protocol_key="issue184",
        root=tmp_path,
        repository_identity=_repository_identity(),
        runner_receipt=_runner_receipt(),
        verify_cache_file=False,
    )
    stored = json.loads((namespace / "result.v1.json").read_text(encoding="utf-8"))
    assert stored == result
    stored["outcome"] = "inconclusive"
    (namespace / "result.v1.json").write_text(
        json.dumps(stored, allow_nan=False, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="published result"):
        _validate(tmp_path, "issue184")


def test_result_validation_rejects_shard_swap_during_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import alberta_framework.benchmarks.ipmnist_screening as screening

    namespace = _write_protocol_receipts(tmp_path, "issue184")
    summary_path = _write_stage(
        tmp_path, "issue184", "screen_60", differences=(0.003,) * 3
    )
    target = sorted((namespace / "screen_60/shards").iterdir())[0]
    replacement = json.loads(target.read_text(encoding="utf-8"))
    replacement["per_task_accuracy"] = [0.9] * 60
    replacement_raw = json.dumps(
        replacement, allow_nan=False, sort_keys=True
    ).encode()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    relative = target.relative_to(tmp_path).as_posix()
    entry = next(item for item in summary["shard_manifest"] if item["path"] == relative)
    entry["size_bytes"] = len(replacement_raw)
    entry["sha256"] = hashlib.sha256(replacement_raw).hexdigest()
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    original_merge = screening.merge_shards

    def swap_after_merge(*args: Any, **kwargs: Any) -> dict[str, Any]:
        merged = original_merge(*args, **kwargs)
        target.write_bytes(replacement_raw)
        return merged

    monkeypatch.setattr(screening, "merge_shards", swap_after_merge)
    with pytest.raises(RuntimeError, match="changed while"):
        _validate(tmp_path, "issue184")


def test_result_validation_rejects_summary_swap_during_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alberta_framework.benchmarks.ipmnist_screening as screening

    _write_protocol_receipts(tmp_path, "issue184")
    summary_path = _write_stage(
        tmp_path, "issue184", "screen_60", differences=(0.003,) * 3
    )
    original_merge = screening.merge_shards

    def swap_after_merge(*args: Any, **kwargs: Any) -> dict[str, Any]:
        merged = original_merge(*args, **kwargs)
        summary_path.write_text("{}", encoding="utf-8")
        return merged

    monkeypatch.setattr(screening, "merge_shards", swap_after_merge)
    with pytest.raises(RuntimeError, match="changed while"):
        _validate(tmp_path, "issue184")


def test_result_validation_rejects_empty_directory_added_during_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alberta_framework.benchmarks.ipmnist_screening as screening

    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    original_merge = screening.merge_shards

    def add_directory_after_merge(*args: Any, **kwargs: Any) -> dict[str, Any]:
        merged = original_merge(*args, **kwargs)
        (namespace / "screen_60/shards/unexpected").mkdir()
        return merged

    monkeypatch.setattr(screening, "merge_shards", add_directory_after_merge)
    with pytest.raises(RuntimeError, match="namespace changed while"):
        _validate(tmp_path, "issue184")


def test_result_validation_rejects_external_hard_link_alias(tmp_path: Path) -> None:
    namespace = _write_protocol_receipts(tmp_path, "issue184")
    _write_stage(tmp_path, "issue184", "screen_60", differences=(0.003,) * 3)
    target = sorted((namespace / "screen_60/shards").iterdir())[0]
    os.link(target, tmp_path / "external-shard-alias.json")

    with pytest.raises(ValueError, match="hard-link alias"):
        _validate(tmp_path, "issue184")


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        (
            "amendment",
            [
                "--repository",
                "elizaOS/asi",
                "--ref-name",
                "ipmnist-prereg-example",
                "--cpuset",
                "0-2",
                "--data-home",
                "/tmp/mnist-cache",
            ],
        ),
        (
            "authorization",
            [
                "--repository",
                "elizaOS/asi",
                "--ref-name",
                "ipmnist-prereg-example",
                "--cpuset",
                "0-2",
                "--data-home",
                "/tmp/mnist-cache",
            ],
        ),
        (
            "launch",
            [
                "--repository",
                "elizaOS/asi",
                "--ref-name",
                "ipmnist-prereg-example",
                "--cpuset",
                "0-2",
                "--data-home",
                "/tmp/mnist-cache",
            ],
        ),
        ("record-cache", ["--data-home", "/tmp/mnist-cache"]),
        (
            "run-shard",
            [
                "--stage",
                "screen_60",
                "--config-name",
                "rls_head_resid_l1_preset005",
                "--seed",
                "0",
            ],
        ),
        ("validate", []),
    ],
)
def test_cli_exposes_only_explicit_protocol_commands(
    command: str, extra: list[str]
) -> None:
    args = _build_parser().parse_args(
        [command, "--protocol", "issue184", "--root", "/tmp/repo", *extra]
    )
    assert callable(args.handler)
