"""Fail-closed tests for the standalone matched-v3 local execution bootstrap."""

from __future__ import annotations

import base64
import copy
import gc
import hashlib
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_PATH = (
    _ROOT / "alberta_framework" / "benchmarks" / "forager_matched_v3_local_execution_bootstrap.py"
)
_RUNNER_PATH = _ROOT / "alberta_framework" / "benchmarks" / "forager_matched_v3_local_runner.py"
_SNAPSHOT_PATH = (
    _ROOT / "alberta_framework" / "benchmarks" / "forager_matched_v3_local_source_snapshot.py"
)


def _direct_module(
    *,
    source_path: Path,
    module_name: str,
    injections: dict[str, Any] | None = None,
) -> types.ModuleType:
    source = source_path.read_bytes()
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    if injections is not None:
        module.__dict__.update(injections)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture(scope="module")
def bootstrap() -> Iterator[types.ModuleType]:
    source = _BOOTSTRAP_PATH.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    name = "_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1"
    module = _direct_module(
        source_path=_BOOTSTRAP_PATH,
        module_name=name,
        injections={"_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256": digest},
    )
    # pytest may preload numerical packages.  Tests exercise the isolated module's
    # other gates directly and replace only that ambient-process observation.
    setattr(module, "_ISOLATED_PARENT_BOUNDARY", True)
    setattr(module, "_live_forbidden_modules", lambda: ())
    yield module
    sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def runner_parser() -> Iterator[types.ModuleType]:
    name = "_alberta_forager_matched_v3_bootstrap_test_runner_parser_v1"
    module = _direct_module(source_path=_RUNNER_PATH, module_name=name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def snapshot_module() -> Iterator[types.ModuleType]:
    name = "_alberta_forager_matched_v3_bootstrap_test_snapshot_v1"
    module = _direct_module(source_path=_SNAPSHOT_PATH, module_name=name)
    yield module
    sys.modules.pop(name, None)


@dataclass(frozen=True)
class _Harness:
    repository: Path
    runtime_root: Path
    scratch_root: Path
    manifest: bytes
    manifest_sha256: str
    local_receipt: bytes
    reward_trace: bytes
    python_executable: Path


@pytest.fixture
def harness(
    tmp_path: Path,
    runner_parser: types.ModuleType,
    snapshot_module: types.ModuleType,
) -> _Harness:
    repository = tmp_path / "repository"
    benchmark_directory = repository / "alberta_framework" / "benchmarks"
    benchmark_directory.mkdir(parents=True)
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "bootstrap-fixture"\n',
        encoding="ascii",
    )
    (repository / "uv.lock").write_text("version = 1\n", encoding="ascii")
    (repository / "FORAGER_BENCHMARK.md").write_text(
        "# Fixture Forager protocol\n", encoding="ascii"
    )
    (repository / "alberta_framework" / "__init__.py").write_bytes(b"")
    (benchmark_directory / "__init__.py").write_bytes(b"")
    shutil.copyfile(
        _RUNNER_PATH,
        benchmark_directory / "forager_matched_v3_local_runner.py",
    )
    shutil.copyfile(
        _SNAPSHOT_PATH,
        benchmark_directory / "forager_matched_v3_local_source_snapshot.py",
    )
    measured = snapshot_module.measure_matched_v3_local_source_snapshot(repository_root=repository)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir(mode=0o700)
    scratch_root.chmod(0o700)

    reward_trace = bytes(runner_parser.MATCHED_V3_LOCAL_RUNNER_HORIZON)
    sources = dict(runner_parser._PINNED_SOURCE_SHA256)
    sources["local_runner_observed"] = hashlib.sha256(_RUNNER_PATH.read_bytes()).hexdigest()
    receipt = runner_parser._completion_receipt(
        candidate_id="causal_e025_q050",
        environment_seed=17,
        agent_seed=23,
        implementation_kind="alberta_causal_map",
        underlying_agent_name="fixture_agent",
        trace=reward_trace,
        cumulative_reward=0,
        source_sha256_by_id=sources,
    )
    local_receipt = runner_parser._canonical_json(receipt)
    runner_parser.parse_matched_v3_local_completion_receipt(
        local_receipt,
        reward_trace=reward_trace,
        expected_receipt_sha256=hashlib.sha256(local_receipt).hexdigest(),
    )
    return _Harness(
        repository=repository,
        runtime_root=runtime_root,
        scratch_root=scratch_root,
        manifest=measured.canonical_manifest_bytes,
        manifest_sha256=measured.full_sha256,
        local_receipt=local_receipt,
        reward_trace=reward_trace,
        python_executable=Path(sys.executable).resolve(strict=True),
    )


def _fresh_scratch(harness: _Harness, name: str) -> Path:
    scratch = harness.scratch_root / name
    scratch.mkdir(mode=0o700)
    scratch.chmod(0o700)
    return scratch


def _rewrite_child_record(
    bootstrap: types.ModuleType,
    raw: bytes,
    mutate: Any,
) -> bytes:
    value = bootstrap._strict_json_load(raw)
    mutate(value)
    body = dict(value)
    body.pop("child_record_body_sha256")
    value["child_record_body_sha256"] = hashlib.sha256(bootstrap._canonical_json(body)).hexdigest()
    rewritten = bootstrap._canonical_json(value)
    if type(rewritten) is not bytes:
        raise AssertionError("bootstrap canonical encoder returned a non-bytes value")
    return rewritten


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
    *,
    mode: str = "success",
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if mode == "nonzero":
            return bootstrap._ProcessResult(41, b"", b"fixture failure")
        if mode == "signal":
            return bootstrap._ProcessResult(-9, b"", b"")
        request_metadata = os.fstat(kwargs["pass_fds"][2])
        request_raw = os.pread(
            kwargs["pass_fds"][2],
            request_metadata.st_size,
            0,
        )
        request, request_sha256 = bootstrap._parse_request(
            request_raw,
            maximum_bytes=request_metadata.st_size,
        )
        process_contract = request["process_contract"]
        runtime_paths = tuple(item["path"] for item in request["runtime_import_roots"])
        original_paths = ("/usr/lib/python3.12",)
        final_paths = (
            request["repository_root"]["path"],
            *runtime_paths,
            *original_paths,
        )
        child_process = bootstrap._child_process_record(
            request_contract=process_contract,
            original_stdlib_paths=original_paths,
            final_sys_path=final_paths,
            repository_path=request["repository_root"]["path"],
            runtime_paths=runtime_paths,
            cache_proc_path=process_contract["cache_proc_path"],
        )
        local_receipt = harness.local_receipt
        reward_trace = harness.reward_trace
        if mode == "local_receipt":
            local_receipt += b"x"
        if mode == "reward_trace":
            reward_trace = b"\x01" + reward_trace[1:]
        child_record = bootstrap._build_child_record(
            bootstrap_source_sha256=process_contract["bootstrap_source_sha256"],
            request_sha256=request_sha256,
            manifest_full_sha256=request["source_snapshot"]["full_sha256"],
            manifest_tree_sha256=request["source_snapshot"]["tree_sha256"],
            pre_full_sha256=request["source_snapshot"]["full_sha256"],
            pre_tree_sha256=request["source_snapshot"]["tree_sha256"],
            post_full_sha256=request["source_snapshot"]["full_sha256"],
            post_tree_sha256=request["source_snapshot"]["tree_sha256"],
            candidate_id=request["cell"]["candidate_id"],
            environment_seed=request["cell"]["environment_seed"],
            agent_seed=request["cell"]["agent_seed"],
            process_contract=child_process,
            local_receipt=local_receipt,
            reward_trace=reward_trace,
        )
        if mode == "snapshot":
            child_record = _rewrite_child_record(
                bootstrap,
                child_record,
                lambda value: value["source_snapshot"].__setitem__("post_tree_sha256", "e" * 64),
            )
        if mode == "sys_path":
            child_record = _rewrite_child_record(
                bootstrap,
                child_record,
                lambda value: value["process_contract"].__setitem__(
                    "final_sys_path_entry_count",
                    value["process_contract"]["final_sys_path_entry_count"] + 1,
                ),
            )
        if mode == "environment":
            child_record = _rewrite_child_record(
                bootstrap,
                child_record,
                lambda value: value["process_contract"].__setitem__("environment_sha256", "c" * 64),
            )
        frame = bootstrap._result_frame(child_record, local_receipt, reward_trace)
        if mode == "extra_frame":
            frame += b"x"
        elif mode == "truncated_frame":
            frame = frame[:-1]
        if mode == "oversized_result":
            os.ftruncate(kwargs["result_fd"], kwargs["maximum_result_bytes"] + 1)
        else:
            bootstrap._write_all(
                kwargs["result_fd"],
                frame,
                maximum_bytes=kwargs["maximum_result_bytes"],
            )
        if mode in {"source_transport", "request_transport"}:
            inherited_index = 1 if mode == "source_transport" else 2
            transport_path = os.readlink(f"/proc/self/fd/{kwargs['pass_fds'][inherited_index]}")
            writer = os.open(transport_path, os.O_WRONLY)
            try:
                os.pwrite(writer, b"x", 0)
            finally:
                os.close(writer)
        if mode == "repository_transport":
            runner_path = (
                Path(request["repository_root"]["path"])
                / "alberta_framework"
                / "benchmarks"
                / "forager_matched_v3_local_runner.py"
            )
            runner_path.write_bytes(runner_path.read_bytes() + b"\n")
        return bootstrap._ProcessResult(0, b"child stdout\n", b"child stderr\n")

    monkeypatch.setattr(bootstrap, "_run_bounded_child", fake_run)
    return calls


def _execute(
    bootstrap: types.ModuleType,
    harness: _Harness,
    *,
    scratch_name: str,
    capability: object | None = None,
    manifest: bytes | None = None,
    manifest_sha256: str | None = None,
    explicit_execution_opt_in: Any = True,
) -> object:
    exact_capability = capability
    if exact_capability is None:
        exact_capability = bootstrap.issue_matched_v3_local_bootstrap_execution_capability(
            explicit_execution_opt_in=True
        )
    return bootstrap.execute_matched_v3_local_bootstrap_cell(
        python_executable=harness.python_executable,
        repository_root=harness.repository,
        runtime_import_roots=(harness.runtime_root,),
        scratch_parent=_fresh_scratch(harness, scratch_name),
        expected_source_snapshot_bytes=(harness.manifest if manifest is None else manifest),
        expected_source_snapshot_sha256=(
            harness.manifest_sha256 if manifest_sha256 is None else manifest_sha256
        ),
        candidate_id="causal_e025_q050",
        environment_seed=17,
        agent_seed=23,
        maximum_request_bytes=2 * 1024 * 1024,
        maximum_bootstrap_source_bytes=1024 * 1024,
        maximum_result_bytes=2 * 1024 * 1024,
        maximum_stdout_bytes=1024 * 1024,
        maximum_stderr_bytes=1024 * 1024,
        timeout_seconds=10,
        explicit_execution_opt_in=explicit_execution_opt_in,
        execution_capability=exact_capability,
    )


def test_descriptor_is_exact_and_non_authorizing(bootstrap: types.ModuleType) -> None:
    raw = bootstrap.canonical_matched_v3_local_execution_bootstrap_descriptor_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "6e62e2c6f2e1d157bee74c0866c96eededda21c5d77073d2adc05cc40dc72733"
    )
    descriptor = bootstrap.parse_matched_v3_local_execution_bootstrap_descriptor(raw)
    assert descriptor["process"]["fresh_processes_per_cell"] == 1
    assert descriptor["process"]["full_process_group_killed_on_all_exits"] is True
    assert descriptor["process"]["direct_child_waited_and_reaped"] is True
    assert descriptor["process"]["descendant_reaping_attested"] is False
    assert descriptor["process"]["environment"]["fixed"]["JAX_PLATFORMS"] == "cpu"
    assert descriptor["process"]["environment"]["fixed"]["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert descriptor["caller_inputs"]["default_paths"] is False
    assert descriptor["pinned_components"]["local_source_snapshot"] == {
        "descriptor_sha256": ("5ba69445a00dfc0bc36a4d05dafcc534b291430d491c3f71560570d7eb862899"),
        "source_sha256": "cfb4c9df2b0d767a40aeeba4bd044ba50c2e595054db768966105a0df9233cbb",
    }
    assert not any(descriptor["claims"].values())
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
        bootstrap.parse_matched_v3_local_execution_bootstrap_descriptor(raw + b" ")


def test_public_execution_surface_has_no_defaults(bootstrap: types.ModuleType) -> None:
    signature = inspect.signature(bootstrap.execute_matched_v3_local_bootstrap_cell)
    assert set(signature.parameters) == {
        "python_executable",
        "repository_root",
        "runtime_import_roots",
        "scratch_parent",
        "expected_source_snapshot_bytes",
        "expected_source_snapshot_sha256",
        "candidate_id",
        "environment_seed",
        "agent_seed",
        "maximum_request_bytes",
        "maximum_bootstrap_source_bytes",
        "maximum_result_bytes",
        "maximum_stdout_bytes",
        "maximum_stderr_bytes",
        "timeout_seconds",
        "explicit_execution_opt_in",
        "execution_capability",
    }
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )


def test_exact_type_and_alias_validators_fail_closed(
    bootstrap: types.ModuleType,
    tmp_path: Path,
) -> None:
    class StringAlias(str):
        pass

    class IntegerAlias(int):
        pass

    for candidate_value in (
        StringAlias("causal_e025_q050"),
        b"causal_e025_q050",
        "bad id",
    ):
        with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
            bootstrap._validate_candidate_id(candidate_value)
    for seed_value in (True, IntegerAlias(1), 1.0, -1, 2**31):
        with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
            bootstrap._require_uint31(seed_value, "seed")
    for ceiling_value in (True, IntegerAlias(1), 1.0, 0):
        with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
            bootstrap._require_positive_ceiling(ceiling_value, "ceiling")
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
        bootstrap._validate_runtime_root_tuple([tmp_path])
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
        bootstrap._exact_absolute_path(str(tmp_path), "path")
    shared: list[Any] = []
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="alias"):
        bootstrap._canonical_json({"left": shared, "right": shared})


def test_fake_process_success_and_two_stage_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    calls = _install_fake_process(monkeypatch, bootstrap, harness)
    outcome = _execute(bootstrap, harness, scratch_name="success")
    assert len(calls) == 1
    call = calls[0]
    assert call["argv"][1:4] == ("-I", "-S", "-B")
    assert call["environment"] == {
        "HOME": call["environment"]["XDG_CACHE_HOME"],
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "JAX_PLATFORMS": "cpu",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "XDG_CACHE_HOME": call["environment"]["HOME"],
    }
    assert call["environment"]["HOME"].startswith("/proc/self/fd/")
    assert len(set(call["pass_fds"])) == 5
    completion = bootstrap.consume_matched_v3_local_bootstrap_outcome(
        outcome_capability=outcome,
        explicit_content_access_opt_in=True,
    )
    assert completion.reward_trace == harness.reward_trace
    receipt = completion.receipt()
    assert receipt["cell"] == {
        "candidate_id": "causal_e025_q050",
        "environment_seed": 17,
        "agent_seed": 23,
    }
    assert receipt["source_snapshot"]["continuous_immutability_attested"] is False
    assert not any(receipt["claims"].values())
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
        bootstrap.consume_matched_v3_local_bootstrap_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=True,
        )


@pytest.mark.parametrize(
    "mode,match",
    [
        ("nonzero", "exited nonzero"),
        ("signal", "died from signal"),
        ("snapshot", "source linkage"),
        ("sys_path", "sys.path linkage"),
        ("environment", "environment_sha256"),
        ("extra_frame", "truncated or extra"),
        ("truncated_frame", "truncated or extra"),
        ("oversized_result", "bounded regular file"),
        ("local_receipt", "completion replay"),
        ("reward_trace", "completion replay"),
        ("source_transport", "immutable source or request"),
        ("request_transport", "immutable source or request"),
        ("repository_transport", "full local source snapshot verification failed"),
    ],
)
def test_fake_process_failures_are_closed(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
    mode: str,
    match: str,
) -> None:
    calls = _install_fake_process(monkeypatch, bootstrap, harness, mode=mode)
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match=match):
        _execute(bootstrap, harness, scratch_name=f"failure-{mode}")
    assert len(calls) == 1


def test_manifest_rejection_precedes_process_and_spends_capability(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    calls = _install_fake_process(monkeypatch, bootstrap, harness)
    capability = bootstrap.issue_matched_v3_local_bootstrap_execution_capability(
        explicit_execution_opt_in=True
    )
    corrupted = harness.manifest[:-2] + b"x\n"
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
        _execute(
            bootstrap,
            harness,
            scratch_name="bad-manifest",
            capability=capability,
            manifest=corrupted,
            manifest_sha256=hashlib.sha256(corrupted).hexdigest(),
        )
    assert calls == []
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="consumed"):
        _execute(
            bootstrap,
            harness,
            scratch_name="spent-capability",
            capability=capability,
        )


def test_capabilities_are_opaque_nonserializable_and_weak(
    bootstrap: types.ModuleType,
) -> None:
    capability = bootstrap.issue_matched_v3_local_bootstrap_execution_capability(
        explicit_execution_opt_in=True
    )
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(capability)
    before = len(bootstrap._EXECUTION_CAPABILITIES)
    del capability
    gc.collect()
    assert len(bootstrap._EXECUTION_CAPABILITIES) < before
    for value in (False, 1, None):
        with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
            bootstrap.issue_matched_v3_local_bootstrap_execution_capability(
                explicit_execution_opt_in=value
            )


def test_pid_binding_spends_execution_capability(bootstrap: types.ModuleType) -> None:
    capability = bootstrap.issue_matched_v3_local_bootstrap_execution_capability(
        explicit_execution_opt_in=True
    )
    bootstrap._EXECUTION_CAPABILITIES[capability].pid += 1
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="PID"):
        bootstrap._consume_execution_capability(capability)
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="consumed"):
        bootstrap._consume_execution_capability(capability)


def test_execution_opt_in_type_rejection_does_not_spend_capability(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    calls = _install_fake_process(monkeypatch, bootstrap, harness)
    capability = bootstrap.issue_matched_v3_local_bootstrap_execution_capability(
        explicit_execution_opt_in=True
    )
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="opt-in"):
        _execute(
            bootstrap,
            harness,
            scratch_name="wrong-execution-opt-in",
            capability=capability,
            explicit_execution_opt_in=1,
        )
    assert calls == []
    outcome = _execute(
        bootstrap,
        harness,
        scratch_name="right-execution-opt-in",
        capability=capability,
    )
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="opt-in"):
        bootstrap.consume_matched_v3_local_bootstrap_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=1,
        )
    completion = bootstrap.consume_matched_v3_local_bootstrap_outcome(
        outcome_capability=outcome,
        explicit_content_access_opt_in=True,
    )
    assert completion.candidate_id == "causal_e025_q050"


def test_outcome_is_opaque_pid_bound_and_permanently_spent(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    _install_fake_process(monkeypatch, bootstrap, harness)
    outcome = _execute(bootstrap, harness, scratch_name="outcome-pid")
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(outcome)
    bootstrap._OUTCOME_CAPABILITIES[outcome].pid += 1
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="PID"):
        bootstrap.consume_matched_v3_local_bootstrap_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=True,
        )
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="consumed"):
        bootstrap.consume_matched_v3_local_bootstrap_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=True,
        )


def test_outcome_source_drift_is_permanently_consuming(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    _install_fake_process(monkeypatch, bootstrap, harness)
    outcome = _execute(bootstrap, harness, scratch_name="source-drift")
    original = bootstrap._current_bootstrap_source_sha256
    monkeypatch.setattr(bootstrap, "_current_bootstrap_source_sha256", lambda _maximum: "d" * 64)
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="stale"):
        bootstrap.consume_matched_v3_local_bootstrap_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=True,
        )
    monkeypatch.setattr(bootstrap, "_current_bootstrap_source_sha256", original)
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="consumed"):
        bootstrap.consume_matched_v3_local_bootstrap_outcome(
            outcome_capability=outcome,
            explicit_content_access_opt_in=True,
        )


def test_receipt_or_completion_cannot_recreate_outcome_authority(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    _install_fake_process(monkeypatch, bootstrap, harness)
    outcome = _execute(bootstrap, harness, scratch_name="receipt-no-authority")
    completion = bootstrap.consume_matched_v3_local_bootstrap_outcome(
        outcome_capability=outcome,
        explicit_content_access_opt_in=True,
    )
    detached = bootstrap.MatchedV3LocalBootstrapCompletion(
        candidate_id=completion.candidate_id,
        environment_seed=completion.environment_seed,
        agent_seed=completion.agent_seed,
        canonical_receipt_bytes=completion.canonical_receipt_bytes,
        receipt_sha256=completion.receipt_sha256,
        canonical_child_record_bytes=completion.canonical_child_record_bytes,
        local_completion_receipt_bytes=completion.local_completion_receipt_bytes,
        reward_trace=completion.reward_trace,
        stdout=completion.stdout,
        stderr=completion.stderr,
    )
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="authentic"):
        bootstrap.consume_matched_v3_local_bootstrap_outcome(
            outcome_capability=detached,
            explicit_content_access_opt_in=True,
        )


def test_receipt_replay_rejects_coherently_rehashed_cpu_claim_drift(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    _install_fake_process(monkeypatch, bootstrap, harness)
    outcome = _execute(bootstrap, harness, scratch_name="coherent-receipt-drift")
    completion = bootstrap.consume_matched_v3_local_bootstrap_outcome(
        outcome_capability=outcome,
        explicit_content_access_opt_in=True,
    )
    mutated_child = _rewrite_child_record(
        bootstrap,
        completion.canonical_child_record_bytes,
        lambda value: value["process_contract"].__setitem__("jax_default_backend", "gpu"),
    )
    frame = bootstrap._result_frame(
        mutated_child,
        completion.local_completion_receipt_bytes,
        completion.reward_trace,
    )
    receipt = bootstrap._strict_json_load(completion.canonical_receipt_bytes)
    receipt["result"]["frame_size_bytes"] = len(frame)
    receipt["result"]["frame_sha256"] = hashlib.sha256(frame).hexdigest()
    receipt["result"]["child_record_size_bytes"] = len(mutated_child)
    receipt["result"]["child_record_sha256"] = hashlib.sha256(mutated_child).hexdigest()
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_body_sha256")
    receipt["receipt_body_sha256"] = hashlib.sha256(
        bootstrap._canonical_json(receipt_body)
    ).hexdigest()
    mutated_receipt = bootstrap._canonical_json(receipt)
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="CPU"):
        bootstrap.parse_matched_v3_local_bootstrap_receipt(
            mutated_receipt,
            expected_receipt_sha256=hashlib.sha256(mutated_receipt).hexdigest(),
            canonical_child_record_bytes=mutated_child,
            local_completion_receipt_bytes=completion.local_completion_receipt_bytes,
            reward_trace=completion.reward_trace,
            stdout=completion.stdout,
            stderr=completion.stderr,
        )


def test_strict_frame_rejects_length_and_magic_mutations(bootstrap: types.ModuleType) -> None:
    frame = bootstrap._result_frame(b"{}\n", b"{}\n", b"\x00")
    for mutation in (
        b"X" + frame[1:],
        frame[:-1],
        frame + b"x",
        frame[:16] + (2**63).to_bytes(8, "big") + frame[24:],
    ):
        with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError):
            bootstrap._parse_result_frame(
                mutation,
                maximum_result_bytes=1024,
            )


def test_descriptor_only_clean_import_observes_no_external_state() -> None:
    source = _BOOTSTRAP_PATH.read_bytes()
    probe = r"""
import base64, builtins, hashlib, hmac, json, os, re, selectors, signal
import stat, struct, subprocess, sys, threading, time, types, weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, cast
source = base64.b64decode(sys.stdin.buffer.read(), validate=True)
events = []
def blocked(*args, **kwargs):
    events.append("external")
    raise AssertionError("descriptor import observed external state")
builtins.open = blocked
for name in ("open", "stat", "lstat", "scandir", "getcwd", "getpid"):
    setattr(os, name, blocked)
subprocess.Popen = blocked
name = "_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1"
module = types.ModuleType(name)
module.__file__ = "/nonexistent/caller-carried-bootstrap.py"
module.__package__ = ""
module.__dict__["_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256"] = (
    hashlib.sha256(source).hexdigest()
)
sys.modules[name] = module
exec(compile(source, module.__file__, "exec"), module.__dict__)
forbidden = ("alberta_framework", "chex", "foragax", "jax", "jaxlib", "numpy", "scipy")
loaded = [key for key in sys.modules if any(key == p or key.startswith(p + ".") for p in forbidden)]
assert events == []
assert loaded == []
assert len(module._EXECUTION_CAPABILITIES) == 0
assert len(module._OUTCOME_CAPABILITIES) == 0
print(json.dumps({"descriptor": module.LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256, "ok": True}))
"""
    completed = subprocess.run(
        [str(Path(sys.executable).resolve()), "-I", "-S", "-B", "-c", probe],
        input=base64.b64encode(source),
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    payload = json.loads(completed.stdout)
    assert payload == {
        "descriptor": "6e62e2c6f2e1d157bee74c0866c96eededda21c5d77073d2adc05cc40dc72733",
        "ok": True,
    }


def test_exact_jax_environment_selects_cpu_without_cache_writes(tmp_path: Path) -> None:
    runtime_root = _ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
    assert runtime_root.is_dir()
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    cache_fd = os.open(cache, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    cache_proc_path = f"/proc/self/fd/{cache_fd}"
    probe = r"""
import json, os, sys
sys.dont_write_bytecode = True
sys.pycache_prefix = sys.argv[2]
sys.path.insert(0, sys.argv[1])
import jax
devices = jax.devices()
assert jax.default_backend() == "cpu"
assert devices and all(device.platform == "cpu" for device in devices)
assert jax.config.jax_platforms == "cpu"
assert jax.config.jax_enable_compilation_cache is False
assert os.environ["HOME"] == sys.argv[2]
assert os.environ["XDG_CACHE_HOME"] == sys.argv[2]
print(json.dumps({"backend": jax.default_backend(), "device_count": len(devices)}))
"""
    try:
        completed = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "-I",
                "-S",
                "-B",
                "-c",
                probe,
                str(runtime_root),
                cache_proc_path,
            ],
            env={
                "HOME": cache_proc_path,
                "JAX_ENABLE_COMPILATION_CACHE": "false",
                "JAX_PLATFORMS": "cpu",
                "LANG": "C",
                "LC_ALL": "C",
                "TZ": "UTC",
                "XDG_CACHE_HOME": cache_proc_path,
            },
            pass_fds=(cache_fd,),
            capture_output=True,
            check=False,
            timeout=30,
        )
    finally:
        os.close(cache_fd)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert json.loads(completed.stdout)["backend"] == "cpu"
    assert list(cache.iterdir()) == []


def test_real_clean_child_reaches_only_dependency_transition(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: types.ModuleType,
    harness: _Harness,
) -> None:
    real_run = bootstrap._run_bounded_child
    observed: list[Any] = []

    def capture(**kwargs: Any) -> Any:
        result = real_run(**kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(bootstrap, "_run_bounded_child", capture)
    with pytest.raises(bootstrap.ForagerMatchedV3LocalExecutionBootstrapError, match="nonzero"):
        _execute(bootstrap, harness, scratch_name="real-clean-probe")
    assert len(observed) == 1
    assert observed[0].returncode > 0
    decoded = observed[0].stderr.decode("utf-8", "replace")
    assert "local runner execution failed inside the verified child boundary" in decoded
    assert any(marker in decoded for marker in ("No module named", "source", "dependency"))


def test_bounded_process_runner_enforces_timeout_and_output(
    bootstrap: types.ModuleType,
    tmp_path: Path,
) -> None:
    executable = Path(sys.executable).resolve(strict=True)
    executable_fd = os.open(executable, os.O_RDONLY)
    result_path = tmp_path / "result.bin"
    result_fd = os.open(result_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        common = {
            "executable_proc_path": f"/proc/self/fd/{executable_fd}",
            "cwd": str(tmp_path),
            "environment": {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            "pass_fds": (executable_fd, result_fd),
            "result_fd": result_fd,
            "maximum_result_bytes": 1024,
            "maximum_stdout_bytes": 1024,
            "maximum_stderr_bytes": 1024,
            "timeout_seconds": 1,
        }
        with pytest.raises(
            bootstrap.ForagerMatchedV3LocalExecutionBootstrapError,
            match="wall-time",
        ):
            bootstrap._run_bounded_child(
                argv=(
                    str(executable),
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    "import time; time.sleep(5)",
                ),
                **common,
            )
        with pytest.raises(
            bootstrap.ForagerMatchedV3LocalExecutionBootstrapError,
            match="stdout",
        ):
            bootstrap._run_bounded_child(
                argv=(
                    str(executable),
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    "print('x' * 5000)",
                ),
                **common,
            )
        marker = tmp_path / "grandchild.pid"
        group_probe = (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-I','-S','-B','-c',"
            "'import time;time.sleep(60)']);"
            "open(sys.argv[1],'w').write(str(p.pid));"
            "time.sleep(60)"
        )
        with pytest.raises(
            bootstrap.ForagerMatchedV3LocalExecutionBootstrapError,
            match="wall-time",
        ):
            bootstrap._run_bounded_child(
                argv=(
                    str(executable),
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    group_probe,
                    str(marker),
                ),
                **common,
            )
        descendant_pid = int(marker.read_text(encoding="ascii"))
        descendant_stat = Path(f"/proc/{descendant_pid}/stat")
        state: str | None = None
        for _attempt in range(100):
            try:
                state = descendant_stat.read_text(encoding="ascii").rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                state = None
                break
            if state in {"X", "Z"}:
                break
            time.sleep(0.01)
        assert state is None or state in {"X", "Z"}
    finally:
        os.close(result_fd)
        os.close(executable_fd)
