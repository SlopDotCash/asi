"""Exact-resume contracts for the development-only reference life."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import struct
import threading
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

import alberta_framework.reference_life_checkpoint as checkpoint_module
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.reference_life import (
    LifePhase,
    ReferenceLifeRunner,
    ReferenceLifeState,
    ReferenceLifeStep,
    _switching_metric_schedule,
    build_prototype_switching_life,
)
from alberta_framework.reference_life_checkpoint import (
    load_reference_life_checkpoint,
    save_reference_life_checkpoint,
)
from alberta_framework.streams.closed_loop import SwitchingTwoStateConfig
from tests._forager_matched_platform import HAS_RENAMEAT2, requires_renameat2

pytestmark = [pytest.mark.integration, pytest.mark.slow]

LIFECYCLE_ID = "prototype.0000001100000012"


def test_switching_metric_schedule_is_constant_time_at_counter_horizon() -> None:
    maximum_events = 2**31 - 4
    assert _switching_metric_schedule(maximum_events, 2) == (
        (1_073_741_822, 1_073_741_822),
        1,
        1_073_741_821,
        2,
    )
    with pytest.raises(ValueError, match="requires nonnegative"):
        _switching_metric_schedule(-1, 2)
    with pytest.raises(ValueError, match="requires nonnegative"):
        _switching_metric_schedule(1, 0)


def test_checkpoint_validator_rejects_generation_and_zero_event_metric_forgery() -> None:
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)

    impossible_generation = dataclasses.replace(
        state,
        commit_generation=state.accepted_events + state.checkpoint_generation - 1,
    )
    with pytest.raises(ValueError, match="generation history"):
        runner.validate_checkpoint_state(impossible_generation)

    metrics = state.metrics
    nonzero_empty_phase = dataclasses.replace(
        metrics,
        reward_sum=metrics.reward_sum + 1.0,
        regret_sum=metrics.regret_sum - 1.0,
        phase_reward_sums=(metrics.phase_reward_sums[0], 1.0),
        phase_regret_sums=(metrics.phase_regret_sums[0], -1.0),
    )
    negative_zero_empty_phase = dataclasses.replace(
        metrics,
        phase_reward_sums=(metrics.phase_reward_sums[0], -0.0),
    )
    completed_empty_phase = dataclasses.replace(
        metrics,
        first_completed_segment_reward=(metrics.first_completed_segment_reward[0], 0.0),
        latest_completed_segment_reward=(metrics.latest_completed_segment_reward[0], 0.0),
    )
    for forged_metrics in (
        nonzero_empty_phase,
        negative_zero_empty_phase,
        completed_empty_phase,
    ):
        forged = dataclasses.replace(state, metrics=forged_metrics)
        with pytest.raises(ValueError, match="zero-event phase"):
            runner.validate_checkpoint_state(forged)


def _agent_config() -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=2,
                n_primitive_actions=2,
                base_step_size=0.05,
                epsilon_base=0.25,
            )
        )
    )


def _runner(*, horizon: int = 6) -> ReferenceLifeRunner:
    return build_prototype_switching_life(
        agent_config=_agent_config(),
        environment_config=SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        lifecycle_id=LIFECYCLE_ID,
        seed=29,
        max_accepted_events=horizon,
    )


def _advance(
    runner: ReferenceLifeRunner,
    state: ReferenceLifeState,
    count: int,
) -> tuple[ReferenceLifeState, tuple[ReferenceLifeStep, ...]]:
    steps: list[ReferenceLifeStep] = []
    for _ in range(count):
        step = runner.step(state)
        assert step.accepted, step.rejection_reason
        assert step.event is not None
        steps.append(step)
        state = step.state
    return state, tuple(steps)


def _assert_semantic_state_exact(
    left: ReferenceLifeState,
    right: ReferenceLifeState,
) -> None:
    _assert_typed_exact(left, right)


def _assert_typed_exact(left: object, right: object, *, path: str = "state") -> None:
    if isinstance(left, jax.Array) or isinstance(right, jax.Array):
        assert isinstance(left, jax.Array) and isinstance(right, jax.Array), path
        assert left.shape == right.shape and left.dtype == right.dtype, path
        if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            left.dtype, jax.dtypes.prng_key
        ):
            assert str(jax.random.key_impl(left)) == str(jax.random.key_impl(right)), path
            left = jax.random.key_data(left)
            right = jax.random.key_data(right)
        assert np.asarray(left).tobytes(order="C") == np.asarray(right).tobytes(order="C"), path
        return
    if dataclasses.is_dataclass(left) or dataclasses.is_dataclass(right):
        assert type(left) is type(right) and dataclasses.is_dataclass(left), path
        for field in dataclasses.fields(left):
            if field.name == "_owner_token":
                continue
            _assert_typed_exact(
                getattr(left, field.name),
                getattr(right, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(left, float) or isinstance(right, float):
        assert type(left) is type(right), path
        assert struct.pack(">d", left) == struct.pack(">d", right), path
        return
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        assert isinstance(left, np.generic) and isinstance(right, np.generic), path
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes(), path
        return
    if isinstance(left, tuple) or isinstance(right, tuple):
        assert isinstance(left, tuple) and isinstance(right, tuple), path
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_typed_exact(left_item, right_item, path=f"{path}[{index}]")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert type(left) is type(right) and left.keys() == right.keys(), path
        for key in left:
            _assert_typed_exact(left[key], right[key], path=f"{path}.{key}")
        return
    assert type(left) is type(right) and left == right, path


@pytest.fixture(scope="module")
def saved_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path]:
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)
    assert state.phase is LifePhase.QUIESCENT
    barrier_state, checkpoint = save_reference_life_checkpoint(
        runner,
        state,
        tmp_path_factory.mktemp("reference-life-checkpoints"),
    )
    return runner, state, barrier_state, checkpoint


@requires_renameat2
def test_quiescent_checkpoint_restores_exact_continuation(
    saved_case: tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path],
) -> None:
    runner, state, barrier_state, checkpoint = saved_case
    assert barrier_state.commit_generation == state.commit_generation + 1
    assert barrier_state.checkpoint_generation == state.checkpoint_generation + 1

    uninterrupted_state, uninterrupted_steps = _advance(runner, barrier_state, 4)
    restored_runner, restored_state = load_reference_life_checkpoint(checkpoint)
    assert restored_runner is not runner
    assert restored_runner.agent_adapter is not runner.agent_adapter
    assert restored_state.agent_state._owner_token is not barrier_state.agent_state._owner_token
    restored_runner.agent_adapter.current_decision(restored_state.agent_state)
    with pytest.raises(ValueError, match="owner is another"):
        runner.agent_adapter.current_decision(restored_state.agent_state)
    _assert_semantic_state_exact(restored_state, barrier_state)
    restored_final, restored_steps = _advance(restored_runner, restored_state, 4)

    _assert_typed_exact(restored_runner.config, runner.config, path="config")
    _assert_typed_exact(restored_steps, uninterrupted_steps, path="steps")
    assert uninterrupted_state.phase is LifePhase.COMPLETED
    _assert_semantic_state_exact(restored_final, uninterrupted_state)
    assert restored_final.transcript_sha256 == uninterrupted_state.transcript_sha256
    assert restored_final.environment_rng_cursor == uninterrupted_state.environment_rng_cursor
    assert restored_final.checkpoint_generation == 1


def _write_canonical_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )


def _rehash_bundle(path: Path) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for relative, child in manifest["children"]["files"].items():
        payload = (path / relative).read_bytes()
        child["sha256"] = hashlib.sha256(payload).hexdigest()
        child["size"] = len(payload)
    payload = {key: value for key, value in manifest.items() if key != "bundle_id"}
    bundle_id = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    ).hexdigest()
    manifest["bundle_id"] = bundle_id
    _write_canonical_json(manifest_path, manifest)
    (path / "COMMITTED").write_text(f"{bundle_id}\n", encoding="ascii")


@requires_renameat2
def test_restore_rejects_incomplete_tampered_unknown_and_symlink_bundles(
    saved_case: tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path],
    tmp_path: Path,
) -> None:
    _, _, _, checkpoint = saved_case

    incomplete = tmp_path / "incomplete"
    shutil.copytree(checkpoint, incomplete)
    (incomplete / "COMMITTED").unlink()
    with pytest.raises(ValueError, match="root entries"):
        load_reference_life_checkpoint(incomplete)

    tampered = tmp_path / "tampered"
    shutil.copytree(checkpoint, tampered)
    state_payload = json.loads((tampered / "life_state.json").read_text(encoding="ascii"))
    state_payload["runner"]["transcript_sha256"] = "0" * 64
    _write_canonical_json(tampered / "life_state.json", state_payload)
    with pytest.raises(ValueError, match="content hash"):
        load_reference_life_checkpoint(tampered)

    semantic = tmp_path / "semantic"
    shutil.copytree(checkpoint, semantic)
    semantic_state = json.loads((semantic / "life_state.json").read_text(encoding="ascii"))
    semantic_state["metrics"]["phase_event_counts"] = [1, 1]
    semantic_state["metrics"]["current_phase"] = 1
    semantic_state["metrics"]["current_segment_events"] = 1
    _write_canonical_json(semantic / "life_state.json", semantic_state)
    _rehash_bundle(semantic)
    with pytest.raises(ValueError, match="environment schedule|phase schedule"):
        load_reference_life_checkpoint(semantic)

    impossible_generation = tmp_path / "impossible-generation"
    shutil.copytree(checkpoint, impossible_generation)
    generation_state = json.loads(
        (impossible_generation / "life_state.json").read_text(encoding="ascii")
    )
    generation_state["runner"]["commit_generation"] = 2
    _write_canonical_json(impossible_generation / "life_state.json", generation_state)
    _rehash_bundle(impossible_generation)
    with pytest.raises(ValueError, match="generation history"):
        load_reference_life_checkpoint(impossible_generation)

    coerced_environment = tmp_path / "coerced-environment"
    shutil.copytree(checkpoint, coerced_environment)
    coerced_state = json.loads(
        (coerced_environment / "life_state.json").read_text(encoding="ascii")
    )
    encoded_step = coerced_state["environment"]["step_count"]
    step_value = int.from_bytes(
        bytes.fromhex(encoded_step["payload_hex"]),
        byteorder="little",
        signed=True,
    )
    encoded_step["dtype"] = "int64"
    encoded_step["payload_hex"] = step_value.to_bytes(8, "little", signed=True).hex()
    _write_canonical_json(coerced_environment / "life_state.json", coerced_state)
    _rehash_bundle(coerced_environment)
    with pytest.raises(ValueError, match="cannot preserve its encoded shape and dtype"):
        load_reference_life_checkpoint(coerced_environment)

    unknown = tmp_path / "unknown"
    shutil.copytree(checkpoint, unknown)
    manifest = json.loads((unknown / "manifest.json").read_text(encoding="ascii"))
    manifest["schema"] = "asi.reference_life_checkpoint.unknown"
    _write_canonical_json(unknown / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest.schema is unsupported"):
        load_reference_life_checkpoint(unknown)

    prototype_unknown = tmp_path / "prototype-unknown"
    shutil.copytree(checkpoint, prototype_unknown)
    metadata_path = prototype_unknown / "prototype" / "metadata" / "metadata"
    prototype_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prototype_metadata["schema"] = "alberta.prototype_agent.unknown"
    metadata_path.write_text(json.dumps(prototype_metadata), encoding="utf-8")
    _rehash_bundle(prototype_unknown)
    with pytest.raises(ValueError, match="nested Prototype checkpoint is not exact"):
        load_reference_life_checkpoint(prototype_unknown)

    symlinked = tmp_path / "symlinked"
    shutil.copytree(checkpoint, symlinked)
    (symlinked / "life_state.json").unlink()
    (symlinked / "life_state.json").symlink_to(checkpoint / "life_state.json")
    with pytest.raises(ValueError, match="symlink"):
        load_reference_life_checkpoint(symlinked)


@requires_renameat2
def test_restore_rejects_rehashed_duplicate_prototype_metadata_key(
    saved_case: tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path],
    tmp_path: Path,
) -> None:
    _, _, _, checkpoint = saved_case
    duplicate = tmp_path / "duplicate-prototype-metadata"
    shutil.copytree(checkpoint, duplicate)
    metadata_path = duplicate / "prototype" / "metadata" / "metadata"
    raw = metadata_path.read_text(encoding="utf-8")
    assert raw.startswith("{")
    metadata_path.write_text(
        '{"schema":"alberta.prototype_agent.v3",' + raw[1:],
        encoding="utf-8",
    )
    _rehash_bundle(duplicate)

    with pytest.raises(ValueError, match="duplicate key 'schema'"):
        load_reference_life_checkpoint(duplicate)


@requires_renameat2
def test_publish_failure_and_existing_generation_leave_runner_unchanged(
    saved_case: tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, existing_checkpoint = saved_case
    duplicate_runner = _runner()
    duplicate_state, _ = _advance(duplicate_runner, duplicate_runner.init(), 2)
    with pytest.raises(FileExistsError, match="already exists"):
        save_reference_life_checkpoint(
            duplicate_runner,
            duplicate_state,
            existing_checkpoint.parent,
        )
    assert duplicate_runner.current_state is duplicate_state

    failed_runner = _runner()
    failed_state, _ = _advance(failed_runner, failed_runner.init(), 2)

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("injected Prototype child failure")

    monkeypatch.setattr(checkpoint_module, "save_prototype_checkpoint", fail_save)
    parent = tmp_path / "failed-publish"
    with pytest.raises(OSError, match="injected Prototype child failure"):
        save_reference_life_checkpoint(failed_runner, failed_state, parent)
    assert failed_runner.current_state is failed_state
    assert not tuple(parent.glob("generation-*"))
    assert not tuple(parent.glob(".*.staging-*"))


@requires_renameat2
def test_post_commit_lock_cleanup_failure_returns_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)
    real_flock = fcntl.flock

    def fail_cleanup(descriptor: int, operation: int) -> None:
        if operation == fcntl.LOCK_UN:
            os.close(descriptor)
            raise OSError("injected unlock and close failure")
        real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", fail_cleanup)
    barrier, checkpoint = save_reference_life_checkpoint(
        runner,
        state,
        tmp_path / "cleanup-failure",
    )
    assert runner.current_state is barrier
    assert checkpoint.is_dir()
    restored_runner, restored_state = load_reference_life_checkpoint(checkpoint)
    _assert_semantic_state_exact(restored_state, barrier)
    assert restored_runner.current_state is restored_state


def _bundle_bytes(path: Path) -> dict[str, bytes | None]:
    return {
        entry.relative_to(path).as_posix(): None if entry.is_dir() else entry.read_bytes()
        for entry in sorted(path.rglob("*"))
    }


@requires_renameat2
@pytest.mark.parametrize("stale", [False, True], ids=["current", "stale"])
def test_save_to_existing_generation_is_rejected_without_mutation(
    saved_case: tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path],
    tmp_path: Path,
    stale: bool,
) -> None:
    _, _, barrier_state, checkpoint = saved_case
    generation = tmp_path / "existing-generation"
    shutil.copytree(checkpoint, generation)
    before = _bundle_bytes(generation)

    runner = _runner()
    initial = runner.init()
    if stale:
        current = runner.step(initial).state
        state = initial
        expected = "stale"
    else:
        state, _ = _advance(runner, initial, 2)
        current = state
        expected = "bundle root"
    with pytest.raises(ValueError, match=expected):
        save_reference_life_checkpoint(runner, state, generation)

    assert runner.current_state is current
    assert _bundle_bytes(generation) == before
    restored_runner, restored_state = load_reference_life_checkpoint(generation)
    _assert_semantic_state_exact(restored_state, barrier_state)
    assert restored_runner.current_state is restored_state


@requires_renameat2
def test_save_inside_existing_generation_is_rejected_without_mutation(
    saved_case: tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path],
    tmp_path: Path,
) -> None:
    _, _, barrier_state, checkpoint = saved_case
    generation = tmp_path / "existing-generation"
    shutil.copytree(checkpoint, generation)
    before = _bundle_bytes(generation)
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)

    with pytest.raises(ValueError, match="inside an existing bundle root"):
        save_reference_life_checkpoint(runner, state, generation / "prototype")

    assert runner.current_state is state
    assert _bundle_bytes(generation) == before
    restored_runner, restored_state = load_reference_life_checkpoint(generation)
    _assert_semantic_state_exact(restored_state, barrier_state)
    assert restored_runner.current_state is restored_state


@requires_renameat2
def test_save_through_ancestor_symlink_into_generation_is_rejected(
    saved_case: tuple[ReferenceLifeRunner, ReferenceLifeState, ReferenceLifeState, Path],
    tmp_path: Path,
) -> None:
    _, _, barrier_state, checkpoint = saved_case
    generation = tmp_path / "existing-generation"
    shutil.copytree(checkpoint, generation)
    before = _bundle_bytes(generation)
    alias = tmp_path / "generation-alias"
    alias.symlink_to(generation, target_is_directory=True)
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)

    with pytest.raises(ValueError, match="inside an existing bundle root"):
        save_reference_life_checkpoint(runner, state, alias / "prototype")

    assert runner.current_state is state
    assert _bundle_bytes(generation) == before
    restored_runner, restored_state = load_reference_life_checkpoint(generation)
    _assert_semantic_state_exact(restored_state, barrier_state)
    assert restored_runner.current_state is restored_state


@pytest.mark.skipif(HAS_RENAMEAT2, reason="Linux publishes the generation instead of refusing")
def test_save_fails_closed_without_renameat2_and_publishes_nothing(tmp_path: Path) -> None:
    """Off Linux the publisher refuses before any generation becomes visible."""

    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)
    assert state.phase is LifePhase.QUIESCENT
    with pytest.raises(OSError, match="atomic no-replace rename requires Linux renameat2"):
        save_reference_life_checkpoint(runner, state, tmp_path)

    published = [entry for entry in tmp_path.iterdir() if not entry.name.startswith(".")]
    assert published == []


@requires_renameat2
def test_checkpoint_filesystem_publication_stays_inside_runner_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)
    parent = (tmp_path / "barrier-store").absolute()
    first_marker = parent / "manifest.json"
    entered_filesystem = threading.Event()
    release_filesystem = threading.Event()
    step_started = threading.Event()
    step_finished = threading.Event()
    gate_lock = threading.Lock()
    gate_used = False
    real_lexists = os.path.lexists

    def gated_lexists(path: os.PathLike[str] | str) -> bool:
        nonlocal gate_used
        should_gate = False
        with gate_lock:
            if not gate_used and Path(path) == first_marker:
                gate_used = True
                should_gate = True
        if should_gate:
            entered_filesystem.set()
            assert release_filesystem.wait(timeout=10.0)
        return real_lexists(path)

    monkeypatch.setattr(os.path, "lexists", gated_lexists)
    save_results: list[tuple[ReferenceLifeState, Path]] = []
    save_errors: list[BaseException] = []
    step_errors: list[BaseException] = []

    def save() -> None:
        try:
            save_results.append(save_reference_life_checkpoint(runner, state, parent))
        except BaseException as exc:
            save_errors.append(exc)

    def race_step() -> None:
        step_started.set()
        try:
            runner.step(state)
        except BaseException as exc:
            step_errors.append(exc)
        finally:
            step_finished.set()

    save_thread = threading.Thread(target=save)
    save_thread.start()
    assert entered_filesystem.wait(timeout=10.0)
    step_thread = threading.Thread(target=race_step)
    step_thread.start()
    assert step_started.wait(timeout=10.0)
    assert not step_finished.wait(timeout=0.1)
    release_filesystem.set()
    save_thread.join(timeout=30.0)
    step_thread.join(timeout=30.0)

    assert not save_thread.is_alive() and not step_thread.is_alive()
    assert not save_errors
    assert len(save_results) == 1
    barrier, checkpoint = save_results[0]
    assert runner.current_state is barrier
    assert len(step_errors) == 1
    assert isinstance(step_errors[0], ValueError)
    assert "stale" in str(step_errors[0])
    restored_runner, restored_state = load_reference_life_checkpoint(checkpoint)
    _assert_semantic_state_exact(restored_state, barrier)
    assert restored_runner.current_state is restored_state
