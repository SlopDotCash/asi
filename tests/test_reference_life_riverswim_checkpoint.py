"""Exact-resume gate for the stochastic RiverSwim reference life."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import struct
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
    SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID,
    SWITCHING_ENVIRONMENT_STATE_SCHEMA,
    ReferenceLifeState,
    RiverSwimReferenceEnvironment,
    build_prototype_riverswim_life,
)
from alberta_framework.reference_life_checkpoint import (
    load_reference_life_checkpoint,
    save_reference_life_checkpoint,
)
from alberta_framework.streams.closed_loop import RiverSwimConfig, RiverSwimMDP

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_LIFECYCLE_ID = "prototype.0000003100000032"


def test_runtime_identity_binds_jax_random_seed_offset() -> None:
    seed_offset = int(jax.config.jax_random_seed_offset)
    baseline = checkpoint_module._runtime_identity()
    jax.config.update(  # type: ignore[no-untyped-call]
        "jax_random_seed_offset", seed_offset + 1
    )
    try:
        changed = checkpoint_module._runtime_identity()
    finally:
        jax.config.update(  # type: ignore[no-untyped-call]
            "jax_random_seed_offset", seed_offset
        )

    assert baseline != changed
    assert baseline["jax_random_seed_offset"] == seed_offset
    assert changed["jax_random_seed_offset"] == seed_offset + 1


def _runner() -> Any:
    agent_config = PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=3,
                n_primitive_actions=2,
                base_step_size=0.05,
                epsilon_base=1.0,
            )
        )
    )
    return build_prototype_riverswim_life(
        agent_config=agent_config,
        environment_config=RiverSwimConfig(  # type: ignore[call-arg]
            n_states=3,
            p_right_up=0.5,
            p_right_down=0.25,
            reward_left=0.01,
            reward_right=1.0,
            initial_state=0,
        ),
        lifecycle_id=_LIFECYCLE_ID,
        seed=53,
        max_accepted_events=6,
    )


def _advance(runner: Any, state: Any, count: int) -> tuple[Any, tuple[Any, ...]]:
    steps = []
    for _ in range(count):
        step = runner.step(state)
        assert step.accepted, step.rejection_reason
        assert step.event is not None
        steps.append(step)
        state = step.state
    return state, tuple(steps)


def test_riverswim_checkpoint_validator_rejects_sub_tolerance_metric_forgery() -> None:
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)
    delta = 5.0e-13
    forged_metrics = dataclasses.replace(
        state.metrics,
        reward_sum=state.metrics.reward_sum + delta,
        regret_sum=state.metrics.regret_sum - delta,
    )
    forged: ReferenceLifeState = dataclasses.replace(state, metrics=forged_metrics)

    with pytest.raises(ValueError, match="stationary|phase totals|regret algebra"):
        runner.validate_checkpoint_state(forged)

    forged_oracle: ReferenceLifeState = dataclasses.replace(
        state,
        metrics=dataclasses.replace(
            state.metrics,
            oracle_reward_sum=state.metrics.oracle_reward_sum + delta,
        ),
    )
    with pytest.raises(ValueError, match="stationary|schedule|regret algebra"):
        runner.validate_checkpoint_state(forged_oracle)


def _assert_typed_exact(left: object, right: object, *, path: str = "value") -> None:
    if isinstance(left, jax.Array) or isinstance(right, jax.Array):
        assert isinstance(left, jax.Array) and isinstance(right, jax.Array), path
        assert left.shape == right.shape and left.dtype == right.dtype, path
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):  # type: ignore[attr-defined]
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


def test_riverswim_quiescent_checkpoint_restores_exact_stochastic_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    state, _ = _advance(runner, runner.init(), 2)
    barrier, checkpoint = save_reference_life_checkpoint(runner, state, tmp_path)
    original_final, original_steps = _advance(runner, barrier, 4)

    seed_offset = int(jax.config.jax_random_seed_offset)
    jax.config.update(  # type: ignore[no-untyped-call]
        "jax_random_seed_offset", seed_offset + 1
    )
    try:
        with pytest.raises(ValueError, match="runtime_identity"):
            load_reference_life_checkpoint(checkpoint)
    finally:
        jax.config.update(  # type: ignore[no-untyped-call]
            "jax_random_seed_offset", seed_offset
        )

    restored_runner, restored = load_reference_life_checkpoint(checkpoint)
    restored_environment = restored_runner.environment_adapter
    original_environment = runner.environment_adapter
    assert isinstance(restored_environment, RiverSwimReferenceEnvironment)
    assert isinstance(original_environment, RiverSwimReferenceEnvironment)
    assert restored_environment is not original_environment
    assert restored_environment._environment is not original_environment._environment
    with pytest.raises(AttributeError, match="immutable"):
        setattr(restored_environment, "_environment", RiverSwimMDP())
    _assert_typed_exact(restored, barrier, path="barrier")
    restored_final, restored_steps = _advance(restored_runner, restored, 4)

    _assert_typed_exact(restored_runner.config, runner.config, path="config")
    _assert_typed_exact(restored_steps, original_steps, path="steps")
    _assert_typed_exact(restored_final, original_final, path="final")
    assert any(
        int(step.event.command.effective_action.to_python()) == 1
        for step in original_steps
        if step.event is not None
    ), "post-checkpoint branch must exercise RiverSwim's stochastic RIGHT action"

    for name, implementation_id, state_schema in (
        (
            "unknown-environment",
            "asi.unknown_environment.preview1",
            "asi.unknown_environment_state.preview1",
        ),
        (
            "crossed-environment",
            SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID,
            None,
        ),
    ):
        forged = tmp_path / name
        shutil.copytree(checkpoint, forged)
        manifest = json.loads((forged / "manifest.json").read_text(encoding="ascii"))
        manifest["life_config"]["environment"]["implementation_id"] = implementation_id
        if state_schema is not None:
            manifest["life_config"]["environment"]["state_schema"] = state_schema
        _write_canonical_json(forged / "manifest.json", manifest)
        _rehash_bundle(forged)
        with pytest.raises(ValueError, match="unsupported or crossed"):
            load_reference_life_checkpoint(forged)

    crossed_state = tmp_path / "crossed-environment-state"
    shutil.copytree(checkpoint, crossed_state)
    state_payload = json.loads(
        (crossed_state / "life_state.json").read_text(encoding="ascii")
    )
    state_payload["environment"]["schema"] = SWITCHING_ENVIRONMENT_STATE_SCHEMA
    _write_canonical_json(crossed_state / "life_state.json", state_payload)
    _rehash_bundle(crossed_state)
    with pytest.raises(ValueError, match="discriminator differs from the live manifest"):
        load_reference_life_checkpoint(crossed_state)

    out_of_range_state = tmp_path / "out-of-range-riverswim-state"
    shutil.copytree(checkpoint, out_of_range_state)
    state_payload = json.loads(
        (out_of_range_state / "life_state.json").read_text(encoding="ascii")
    )
    state_payload["environment"]["state_index"]["payload_hex"] = (
        (99).to_bytes(4, byteorder="little", signed=True).hex()
    )
    _write_canonical_json(out_of_range_state / "life_state.json", state_payload)
    _rehash_bundle(out_of_range_state)
    with pytest.raises(ValueError, match="outside the chain"):
        load_reference_life_checkpoint(out_of_range_state)

    wrong_metrics = tmp_path / "wrong-riverswim-metrics-mode"
    shutil.copytree(checkpoint, wrong_metrics)
    manifest = json.loads((wrong_metrics / "manifest.json").read_text(encoding="ascii"))
    manifest["life_config"]["metrics"]["config"]["mode"] = "switching_two_phase"
    _write_canonical_json(wrong_metrics / "manifest.json", manifest)
    _rehash_bundle(wrong_metrics)
    with pytest.raises(ValueError, match="metrics mode"):
        load_reference_life_checkpoint(wrong_metrics)

    oversized = tmp_path / "oversized-riverswim"
    shutil.copytree(checkpoint, oversized)
    manifest = json.loads((oversized / "manifest.json").read_text(encoding="ascii"))
    manifest["life_config"]["environment"]["config"]["n_states"] = 13
    _write_canonical_json(oversized / "manifest.json", manifest)
    _rehash_bundle(oversized)
    oracle_called = False

    def forbidden_oracle(self: RiverSwimMDP) -> float:
        del self
        nonlocal oracle_called
        oracle_called = True
        raise AssertionError("oversized restore must reject before oracle enumeration")

    monkeypatch.setattr(RiverSwimMDP, "optimal_average_reward", forbidden_oracle)
    with pytest.raises(ValueError, match="exceeds the RiverSwim reference bound"):
        load_reference_life_checkpoint(oversized)
    assert oracle_called is False
