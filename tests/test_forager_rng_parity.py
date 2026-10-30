from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks import forager_rng_parity as parity


def test_key_frame_rejects_integer_subclasses_before_comparison_hooks() -> None:
    class HostileInt(int):
        def __lt__(self, other: object) -> bool:
            raise AssertionError("hostile comparison must not run")

        def __gt__(self, other: object) -> bool:
            raise AssertionError("hostile comparison must not run")

    with pytest.raises(parity.ForagerRngParityError, match=r"input_key\[0\]"):
        parity.KeyFrame(
            input_key=(HostileInt(0), 0),
            next_key=(0, 0),
            environment_key=(0, 0),
        )


def _runtime_identity() -> parity.VerifiedRuntimeIdentity:
    return parity.VerifiedRuntimeIdentity(
        required_oci_image_id=parity.REQUIRED_OCI_IMAGE_ID,
        build_attestation_sha256=parity.REQUIRED_BUILD_ATTESTATION_SHA256,
        source_repository=parity.REQUIRED_SOURCE_REPOSITORY,
        source_commit=parity.REQUIRED_SOURCE_COMMIT,
        source_tree_git_sha1=parity.REQUIRED_SOURCE_TREE_GIT_SHA1,
        source_archive_sha256=parity.REQUIRED_SOURCE_ARCHIVE_SHA256,
        source_archive_inventory_sha256=(parity.REQUIRED_SOURCE_ARCHIVE_INVENTORY_SHA256),
        dependency_lock_sha256=parity.REQUIRED_DEPENDENCY_LOCK_SHA256,
        wrapper_source_path=parity.REQUIRED_WRAPPER_PATH.as_posix(),
        wrapper_source_sha256=parity.REQUIRED_WRAPPER_SHA256,
        source_mount_mode=parity.REQUIRED_SOURCE_MOUNT_MODE,
        foragax_distribution=parity.REQUIRED_FORAGAX_DISTRIBUTION,
        foragax_version=parity.REQUIRED_FORAGAX_VERSION,
        foragax_wheel_sha256=parity.REQUIRED_FORAGAX_WHEEL_SHA256,
        foragax_install_tree_hash_scheme=(parity.REQUIRED_FORAGAX_INSTALL_TREE_HASH_SCHEME),
        foragax_install_tree_sha256=parity.REQUIRED_FORAGAX_INSTALL_TREE_SHA256,
        python_version=parity.REQUIRED_PYTHON_VERSION,
        python_executable_sha256=parity.REQUIRED_PYTHON_EXECUTABLE_SHA256,
        jax_version=parity.REQUIRED_JAX_VERSION,
        jaxlib_version=parity.REQUIRED_JAXLIB_VERSION,
        backend=parity.REQUIRED_BACKEND,
        cpu_device_count=1,
        prng_impl=parity.REQUIRED_PRNG_IMPL,
        threefry_partitionable=True,
        jax_enable_x64=False,
        probe_module_sha256="a" * 64,
    )


def _trace(config: parity.FixedActionProbeConfig) -> parity.RawEnvironmentTrace:
    reset_keys, transition_keys = parity.expected_key_schedule(config)
    reset = parity.RawResetRecord(
        keys=reset_keys,
        observation=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        state={
            "position": np.asarray([1, 2], dtype=np.int32),
            "occupancy": np.asarray([[0, 1], [2, 0]], dtype=np.uint8),
        },
    )
    transitions = tuple(
        parity.RawTransitionRecord(
            index=index,
            action=action,
            keys=transition_keys[index],
            observation=np.full((2, 2, 3), index + action, dtype=np.float32),
            reward=np.asarray(index - action, dtype=np.float32),
            done=np.asarray(False, dtype=np.bool_),
            info={
                "discount": np.asarray(1.0, dtype=np.float32),
                "object_id": np.asarray(index, dtype=np.int32),
            },
            state={
                "position": np.asarray([index + 1, action], dtype=np.int32),
                "occupancy": np.asarray([[0, 1], [2, index]], dtype=np.int32),
            },
        )
        for index, action in enumerate(config.actions)
    )
    return parity.RawEnvironmentTrace(reset=reset, transitions=transitions)


def _result() -> parity.ParityProbeResult:
    config = parity.FixedActionProbeConfig(seed=2_300_001, actions=(0, 3, 1, 2))
    trace = _trace(config)
    return parity.compare_fixed_action_traces(
        config,
        trace,
        trace,
        _runtime_identity(),
    )


def _rehash_result(payload: dict[str, Any]) -> None:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    payload["payload_sha256"] = hashlib.sha256(parity.canonical_json_bytes(unsigned)).hexdigest()


def _rehash_trace(payload: dict[str, Any]) -> None:
    trace = payload["matched_trace"]
    unsigned_trace = dict(trace)
    unsigned_trace.pop("trace_sha256", None)
    trace_sha256 = hashlib.sha256(parity.canonical_json_bytes(unsigned_trace)).hexdigest()
    trace["trace_sha256"] = trace_sha256
    payload["wrapper_trace_sha256"] = trace_sha256
    payload["direct_trace_sha256"] = trace_sha256
    _rehash_result(payload)


def _collector_result(
    collector: parity.CollectorKind,
    config: parity.FixedActionProbeConfig,
    trace: parity.RawEnvironmentTrace,
) -> parity.ParityCollectorResult:
    digest = parity.digest_environment_trace(config, trace, runner_label=collector)
    draft = parity.ParityCollectorResult(
        collector=collector,
        runtime=_runtime_identity(),
        config=config,
        trace=digest,
        payload_sha256="",
    )
    return replace(
        draft,
        payload_sha256=hashlib.sha256(
            parity.canonical_json_bytes(draft.unsigned_dict())
        ).hexdigest(),
    )


def test_matching_traces_produce_canonical_hash_only_result() -> None:
    result = _result()
    payload = result.to_dict()

    assert result.wrapper_trace_sha256 == result.direct_trace_sha256
    assert result.wrapper_trace_sha256 == result.matched_trace.trace_sha256
    assert (
        result.payload_sha256
        == hashlib.sha256(parity.canonical_json_bytes(result.unsigned_dict())).hexdigest()
    )

    for invalid_digest in (None, False, 0):
        with pytest.raises(parity.ForagerRngParityError, match="payload_sha256 must be a string"):
            replace(result, payload_sha256=invalid_digest)  # type: ignore[arg-type]

    assert payload["status"] == parity.MATCH_STATUS
    assert payload["evidence_boundary"] == parity.CONTENT_IDENTITY_BOUNDARY
    assert payload["promotion_authorized"] is False
    assert "timestamp" not in result.canonical_bytes.decode("utf-8")

    transition = payload["matched_trace"]["transitions"][0]
    assert set(transition["reward"]) == {
        "leaf_count",
        "structure_sha256",
        "content_sha256",
    }
    assert "values" not in transition["reward"]
    assert parity.validate_parity_result(result.canonical_bytes) == result


def test_collector_result_rejects_falsey_nonstring_payload_digest() -> None:
    config = parity.FixedActionProbeConfig(seed=2_300_001, actions=(0, 3, 1, 2))
    result = _collector_result("wrapper", config, _trace(config))
    for invalid_digest in (None, False, 0):
        with pytest.raises(parity.ForagerRngParityError, match="payload_sha256 must be a string"):
            replace(result, payload_sha256=invalid_digest)  # type: ignore[arg-type]


def test_live_orchestration_uses_distinct_collectors_and_detects_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parity.FixedActionProbeConfig(seed=73, actions=(0, 1, 2))
    wrapper_trace = _trace(config)
    direct_trace = _trace(config)
    calls: list[str] = []

    monkeypatch.setattr(parity, "collect_verified_runtime_identity", _runtime_identity)

    def wrapper_runner(actual: parity.FixedActionProbeConfig) -> parity.RawEnvironmentTrace:
        assert actual == config
        calls.append("wrapper")
        return wrapper_trace

    def direct_runner(actual: parity.FixedActionProbeConfig) -> parity.RawEnvironmentTrace:
        assert actual == config
        calls.append("direct")
        return direct_trace

    monkeypatch.setattr(parity, "_run_wrapper_trace", wrapper_runner)
    monkeypatch.setattr(parity, "_run_direct_trace", direct_runner)
    result = parity.run_live_parity_probe(config)
    assert calls == ["wrapper", "direct"]
    assert wrapper_trace is not direct_trace
    assert result.wrapper_trace_sha256 == result.direct_trace_sha256

    first = direct_trace.transitions[0]
    changed_direct = replace(
        direct_trace,
        transitions=(
            replace(first, reward=np.asarray(123.0, dtype=np.float32)),
            *direct_trace.transitions[1:],
        ),
    )
    monkeypatch.setattr(parity, "_run_direct_trace", lambda _config: changed_direct)
    with pytest.raises(parity.ForagerRngParityMismatchError, match="reward"):
        parity.run_live_parity_probe(config)


def test_isolated_collectors_round_trip_and_combine() -> None:
    config = parity.FixedActionProbeConfig(seed=79, actions=(3, 2, 1, 0))
    wrapper = _collector_result("wrapper", config, _trace(config))
    direct = _collector_result("direct", config, _trace(config))

    assert parity.validate_collector_result(wrapper.canonical_bytes) == wrapper
    assert parity.validate_collector_result(direct.canonical_bytes) == direct
    combined = parity.compare_collector_results(wrapper, direct)
    assert combined.wrapper_trace_sha256 == wrapper.trace.trace_sha256
    assert combined.direct_trace_sha256 == direct.trace.trace_sha256

    with pytest.raises(parity.ForagerRngParityError, match="payload_sha256"):
        parity.compare_collector_results(
            wrapper,
            replace(direct, payload_sha256="0" * 64),
        )

    changed_trace = _trace(config)
    changed_first = replace(
        changed_trace.transitions[0],
        state={"changed": np.asarray(1, dtype=np.int32)},
    )
    changed_direct = _collector_result(
        "direct",
        config,
        replace(
            changed_trace,
            transitions=(changed_first, *changed_trace.transitions[1:]),
        ),
    )
    with pytest.raises(parity.ForagerRngParityMismatchError, match="state"):
        parity.compare_collector_results(wrapper, changed_direct)


def test_live_collector_dispatches_only_the_requested_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parity.FixedActionProbeConfig(seed=83, actions=(0, 3))
    trace = _trace(config)
    calls: list[str] = []
    monkeypatch.setattr(parity, "collect_verified_runtime_identity", _runtime_identity)

    def wrapper_runner(_config: parity.FixedActionProbeConfig) -> parity.RawEnvironmentTrace:
        calls.append("wrapper")
        return trace

    def direct_runner(_config: parity.FixedActionProbeConfig) -> parity.RawEnvironmentTrace:
        calls.append("direct")
        return trace

    monkeypatch.setattr(parity, "_run_wrapper_trace", wrapper_runner)
    monkeypatch.setattr(parity, "_run_direct_trace", direct_runner)

    wrapper = parity.run_live_collector(config, "wrapper")
    assert calls == ["wrapper"]
    direct = parity.run_live_collector(config, "direct")
    assert calls == ["wrapper", "direct"]
    assert parity.compare_collector_results(wrapper, direct).matched_trace == wrapper.trace


@pytest.mark.parametrize(
    ("seed", "actions"),
    [
        (True, (0,)),
        (-1, (0,)),
        (2**31, (0,)),
        (0, ()),
        (0, (True,)),
        (0, (-1,)),
        (0, (4,)),
        (0, [0]),
    ],
)
def test_probe_config_rejects_lossy_or_out_of_range_values(
    seed: Any,
    actions: Any,
) -> None:
    with pytest.raises(parity.ForagerRngParityError):
        parity.FixedActionProbeConfig(seed=seed, actions=actions)


def test_key_schedule_binds_reset_and_every_transition() -> None:
    config = parity.FixedActionProbeConfig(seed=17, actions=(0, 1, 2, 3))
    reset, transitions = parity.expected_key_schedule(config)

    assert reset.input_key == (0, 17)
    assert len(transitions) == len(config.actions)
    assert transitions[0].input_key == reset.next_key
    assert all(
        transitions[index].next_key == transitions[index + 1].input_key
        for index in range(len(transitions) - 1)
    )
    assert len({reset.environment_key, *(item.environment_key for item in transitions)}) == 5


@pytest.mark.parametrize(
    ("field", "error_path"),
    [
        ("reset_observation", "reset.observation"),
        ("reset_state", "reset.state"),
        ("observation", "transitions[0].observation"),
        ("reward", "transitions[0].reward"),
        ("done", "transitions[0].done"),
        ("info", "transitions[0].info"),
        ("state", "transitions[0].state"),
    ],
)
def test_mismatch_localization_is_component_specific(field: str, error_path: str) -> None:
    config = parity.FixedActionProbeConfig(seed=31, actions=(0, 2))
    wrapper = _trace(config)
    direct = _trace(config)
    if field == "reset_observation":
        direct = replace(
            direct,
            reset=replace(direct.reset, observation=np.asarray([999], dtype=np.int32)),
        )
    elif field == "reset_state":
        direct = replace(
            direct,
            reset=replace(direct.reset, state=np.asarray([999], dtype=np.int32)),
        )
    else:
        first = direct.transitions[0]
        replacement: Any
        if field == "done":
            replacement = np.asarray(True, dtype=np.bool_)
        elif field == "info":
            replacement = {"changed": np.asarray(1, dtype=np.int32)}
        else:
            replacement = np.asarray([999], dtype=np.float32)
        direct = replace(
            direct,
            transitions=(replace(first, **{field: replacement}), *direct.transitions[1:]),
        )

    with pytest.raises(parity.ForagerRngParityMismatchError) as caught:
        parity.compare_fixed_action_traces(
            config,
            wrapper,
            direct,
            _runtime_identity(),
        )
    assert error_path in str(caught.value)


def test_trace_rejects_wrong_key_action_index_and_length_before_comparison() -> None:
    config = parity.FixedActionProbeConfig(seed=41, actions=(0, 1))
    trace = _trace(config)
    wrong_words = replace(trace.reset.keys, environment_key=(0, 0))
    with pytest.raises(parity.ForagerRngParityError, match="split-chain"):
        parity.digest_environment_trace(
            config,
            replace(trace, reset=replace(trace.reset, keys=wrong_words)),
            runner_label="test",
        )

    with pytest.raises(parity.ForagerRngParityError, match="action"):
        parity.digest_environment_trace(
            config,
            replace(
                trace,
                transitions=(replace(trace.transitions[0], action=3), trace.transitions[1]),
            ),
            runner_label="test",
        )

    with pytest.raises(parity.ForagerRngParityError, match="count"):
        parity.digest_environment_trace(
            config,
            replace(trace, transitions=trace.transitions[:1]),
            runner_label="test",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_oci_image_id", "sha256:" + "0" * 64),
        ("wrapper_source_sha256", "0" * 64),
        ("foragax_install_tree_sha256", "0" * 64),
        ("backend", "gpu"),
        ("prng_impl", "rbg"),
        ("threefry_partitionable", False),
        ("jax_enable_x64", True),
    ],
)
def test_runtime_identity_tampering_fails_closed(field: str, value: Any) -> None:
    config = parity.FixedActionProbeConfig(seed=5, actions=(0,))
    trace = _trace(config)
    with pytest.raises(parity.ForagerRngParityError, match="runtime identity"):
        parity.compare_fixed_action_traces(
            config,
            trace,
            trace,
            replace(_runtime_identity(), **{field: value}),
        )


def test_pytree_fingerprint_distinguishes_scalar_from_length_one_vector() -> None:
    scalar = np.asarray(1.5, dtype=np.float32)
    vector = np.asarray([1.5], dtype=np.float32)
    assert parity.fingerprint_pytree(scalar, label="value") != parity.fingerprint_pytree(
        vector, label="value"
    )


def test_pytree_fingerprint_is_deterministic_and_rejects_unsafe_leaves() -> None:
    first = {
        "b": np.asarray([1, 2], dtype=np.int32),
        "a": np.asarray(0.5, dtype=np.float32),
    }
    second = {
        "a": np.asarray(0.5, dtype=np.float32),
        "b": np.asarray([1, 2], dtype=np.int32),
    }
    assert parity.fingerprint_pytree(first, label="first") == parity.fingerprint_pytree(
        second, label="second"
    )

    for invalid in (
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([object()], dtype=object),
        np.asarray([1 + 2j], dtype=np.complex64),
    ):
        with pytest.raises(parity.ForagerRngParityError):
            parity.fingerprint_pytree(invalid, label="invalid")


def test_runtime_verifier_rejects_nonisolated_python() -> None:
    assert sys.flags.isolated == 0
    with pytest.raises(parity.ForagerRngParityError, match="-I -B"):
        parity._validate_isolated_runtime()


def test_pinned_upstream_tree_fits_the_fail_closed_verification_budget() -> None:
    # Git tree a5ad878a... has 12,512 entries, 10,944 blobs, 305,558,868
    # aggregate blob bytes, and one largest blob of 10,258,251 bytes.
    assert parity.MAX_TREE_ENTRIES >= 12_512
    assert parity.MAX_TREE_FILES >= 10_944
    assert parity.MAX_TREE_BYTES >= 305_558_868
    assert parity.MAX_TREE_FILE_BYTES >= 10_258_251
    assert parity.REQUIRED_SOURCE_DIRECTORY_MODES == frozenset({0o555, 0o755})


def test_no_follow_tree_walk_rejects_links_and_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"ab")
    root_mode = stat.S_IMODE(tmp_path.stat().st_mode)
    file_mode = stat.S_IMODE(payload.stat().st_mode)
    directories, files = parity._verified_tree_entries(
        tmp_path,
        label="synthetic tree",
        require_read_only=False,
        capture_contents=True,
        directory_modes={root_mode},
        file_modes={file_mode},
    )
    assert directories == ()
    assert files[0].contents == b"ab"

    symlink = tmp_path / "linked.bin"
    symlink.symlink_to(payload)
    with pytest.raises(parity.ForagerRngParityError, match="symbolic link"):
        parity._verified_tree_entries(
            tmp_path,
            label="synthetic tree",
            require_read_only=False,
            capture_contents=False,
            directory_modes={root_mode},
            file_modes={file_mode},
        )
    symlink.unlink()

    hardlink = tmp_path / "hard.bin"
    os.link(payload, hardlink)
    with pytest.raises(parity.ForagerRngParityError, match="single-link"):
        parity._verified_tree_entries(
            tmp_path,
            label="synthetic tree",
            require_read_only=False,
            capture_contents=False,
            directory_modes={root_mode},
            file_modes={file_mode},
        )
    hardlink.unlink()

    monkeypatch.setattr(parity, "MAX_TREE_BYTES", 1)
    with pytest.raises(parity.ForagerRngParityError, match="aggregate byte"):
        parity._verified_tree_entries(
            tmp_path,
            label="synthetic tree",
            require_read_only=False,
            capture_contents=False,
            directory_modes={root_mode},
            file_modes={file_mode},
        )


def test_result_self_hash_detects_plain_tampering_and_external_hash_binds_identity() -> None:
    result = _result()
    payload = result.to_dict()
    payload["matched_trace"]["transitions"][0]["reward"]["content_sha256"] = "b" * 64
    with pytest.raises(parity.ForagerRngParityError, match="payload_sha256"):
        parity.validate_parity_result(payload)

    rehashed = result.to_dict()
    rehashed["matched_trace"]["transitions"][0]["reward"]["content_sha256"] = "b" * 64
    _rehash_trace(rehashed)
    reparsed = parity.validate_parity_result(rehashed)
    assert reparsed.payload_sha256 != result.payload_sha256
    assert reparsed.to_dict()["promotion_authorized"] is False
    assert reparsed.to_dict()["evidence_boundary"] == parity.CONTENT_IDENTITY_BOUNDARY
    with pytest.raises(parity.ForagerRngParityError, match="externally expected"):
        parity.validate_parity_result(
            rehashed,
            expected_payload_sha256=result.payload_sha256,
        )


def test_rehashed_fixed_runtime_or_key_drift_is_still_rejected() -> None:
    payload = _result().to_dict()
    payload["runtime"]["required_oci_image_id"] = "sha256:" + "0" * 64
    _rehash_result(payload)
    with pytest.raises(parity.ForagerRngParityError, match="runtime identity"):
        parity.validate_parity_result(payload)

    payload = _result().to_dict()
    payload["matched_trace"]["reset"]["keys"]["environment_key"] = [0, 0]
    _rehash_trace(payload)
    with pytest.raises(parity.ForagerRngParityError, match="split-chain"):
        parity.validate_parity_result(payload)


def test_rehashed_json_type_confusion_is_rejected() -> None:
    payload = _result().to_dict()
    payload["task"]["reward_delay"] = False
    _rehash_result(payload)
    with pytest.raises(parity.ForagerRngParityError, match="task differs"):
        parity.validate_parity_result(payload)

    payload = _result().to_dict()
    payload["probe"]["action_count"] = True
    _rehash_result(payload)
    with pytest.raises(parity.ForagerRngParityError, match="must be an integer"):
        parity.validate_parity_result(payload)

    payload = _result().to_dict()
    payload["runtime"]["threefry_partitionable"] = 1
    _rehash_result(payload)
    with pytest.raises(parity.ForagerRngParityError, match="must be boolean"):
        parity.validate_parity_result(payload)


def test_strict_result_decoder_rejects_duplicate_and_nonfinite_json() -> None:
    with pytest.raises(parity.ForagerRngParityError, match="duplicate JSON"):
        parity.validate_parity_result('{"schema_version":"a","schema_version":"b"}')
    with pytest.raises(parity.ForagerRngParityError, match="non-finite"):
        parity.decode_strict_json('{"value":1e10000}')


def test_task_and_rng_descriptors_are_exact_and_self_hashed() -> None:
    task = parity.task_descriptor()
    task_unsigned = dict(task)
    task_sha256 = task_unsigned.pop("task_sha256")
    assert task_sha256 == hashlib.sha256(parity.canonical_json_bytes(task_unsigned)).hexdigest()
    assert task["environment_id"] == "ForagaxTwoBiomeLarge-v1"
    assert task["aperture_size"] == 9
    assert task["observation_type"] == "color"

    contract = parity.rng_contract_descriptor()
    contract_unsigned = dict(contract)
    contract_sha256 = contract_unsigned.pop("rng_contract_sha256")
    assert (
        contract_sha256
        == hashlib.sha256(parity.canonical_json_bytes(contract_unsigned)).hexdigest()
    )
    assert contract["prng_impl"] == "threefry2x32"
    assert contract["jax_threefry_partitionable"] is True
    assert contract["backend"] == "cpu"


def test_result_round_trip_is_canonical_and_order_independent() -> None:
    payload = _result().to_dict()
    shuffled = json.loads(json.dumps(payload, sort_keys=False))
    parsed = parity.validate_parity_result(shuffled)
    assert parsed.canonical_bytes == parity.canonical_json_bytes(payload)
    assert parsed.payload_sha256 == payload["payload_sha256"]


def test_key_frame_rejects_leftover_bool_identities() -> None:
    """Public key-split records must not keep leftover bool-as-uint32 identities."""

    with pytest.raises(parity.ForagerRngParityError, match="input_key"):
        parity.KeyFrame((True, False), (0, 1), (2, 3))
    with pytest.raises(parity.ForagerRngParityError, match="next_key"):
        parity.KeyFrame((0, 1), (True, 0), (2, 3))
    frame = parity.KeyFrame((0, 1), (2, 3), (4, 5))
    dumped = json.dumps(frame.to_dict(), allow_nan=False)
    assert '"input_key": [0, 1]' in dumped
    assert "true" not in dumped


def test_tree_digest_rejects_leftover_identities() -> None:
    digest_a = hashlib.sha256(b"structure").hexdigest()
    digest_b = hashlib.sha256(b"content").hexdigest()
    with pytest.raises(parity.ForagerRngParityError, match="leaf_count"):
        parity.TreeDigest(True, digest_a, digest_b)
    with pytest.raises(parity.ForagerRngParityError, match="structure_sha256"):
        parity.TreeDigest(1, "not-a-digest", digest_b)
    with pytest.raises(parity.ForagerRngParityError, match="content_sha256"):
        parity.TreeDigest(1, digest_a, float("nan"))  # type: ignore[arg-type]
    digest = parity.TreeDigest(3, digest_a, digest_b)
    dumped = json.dumps(digest.to_dict(), allow_nan=False)
    assert '"leaf_count": 3' in dumped
    assert '"leaf_count": true' not in dumped
