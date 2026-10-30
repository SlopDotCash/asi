"""Independent checks for recorded linear-SARSA diagnostic snapshots."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _checker() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "check_riverswim_sarsa.py"
    spec = importlib.util.spec_from_file_location("check_riverswim_sarsa", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=[0.0, 0.8])
def source_record(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Use the real public update, with a fixed, legal trajectory and actions."""
    import jax.numpy as jnp
    import jax.random as jr

    from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig
    from alberta_framework.reference_life_controls import DiscountedSARSAReferenceConfig
    from alberta_framework.streams.closed_loop import RiverSwimConfig

    # Chex generates these constructors/replace at runtime; match the existing
    # repository's narrow annotations for that static-typing limitation.
    env = RiverSwimConfig(n_states=3)  # type: ignore[call-arg]
    config = DiscountedSARSAReferenceConfig.for_riverswim(env, lamda=request.param)
    agent = SARSAAgent(
        SARSAConfig(n_actions=2, gamma=config.gamma),  # type: ignore[call-arg]
        hidden_sizes=(),
        step_size=config.step_size,
        sparsity=0.0,
        use_layer_norm=False,
        lamda=config.lamda,
    )
    state = agent.init(3, jr.key(42))
    actions = [0, 1, 1, 1, 0]
    positions = [0, 1, 2, 2, 1]
    rewards = [env.reward_left, 0.0, 0.0, env.reward_right, 0.0]
    state = state.replace(  # type: ignore[attr-defined]
        last_action=jnp.int32(actions[0]), last_observation=jnp.eye(3)[0]
    )

    def snapshot(index: int) -> dict[str, Any]:
        inner = state.learner_state
        weights = np.stack([np.asarray(w)[0] for w in inner.head_params.weights], axis=1)
        biases = np.array([np.asarray(b)[0] for b in inner.head_params.biases])
        traces = np.concatenate([np.asarray(a).ravel() for pair in inner.head_traces for a in pair])
        parameters = np.concatenate([weights.ravel(), biases.ravel()]).astype(np.float64)
        return {
            "accepted_events": index,
            "q_by_state_action": (weights + biases).astype(np.float64).tolist(),
            "eligibility_l2": float(np.linalg.norm(traces.astype(np.float64))),
            "eligibility_max_abs": float(np.max(np.abs(traces))),
            "parameter_l2": float(np.linalg.norm(parameters)),
        }

    snapshots = [snapshot(0)]
    for i, (reward, position) in enumerate(zip(rewards, positions, strict=True)):
        next_action = actions[i + 1] if i + 1 < len(actions) else 1
        result = agent.update(
            state,
            jnp.float32(reward),
            jnp.eye(3)[position],
            jnp.bool_(False),
            jnp.int32(next_action),
        )
        assert int(result.action) == next_action
        state = result.state
        snapshots.append(snapshot(i + 1))
    return {
        "schema": "asi.riverswim_sarsa_step_size_diagnostic.shard.v1",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reference_dev_population_allowed": False,
        "config": config.to_config(),
        "environment": {
            "n_states": 3,
            "initial_state": 0,
            "reward_left": env.reward_left,
            "reward_right": env.reward_right,
            "p_right_up": env.p_right_up,
            "p_right_down": env.p_right_down,
        },
        "accepted_events": len(actions),
        "actions": actions,
        "positions": positions,
        "rewards": rewards,
        "snapshots": snapshots,
    }


@pytest.fixture
def record(source_record: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(source_record)


def test_real_update_diagnostics_match_independent_recurrence(record: dict[str, Any]) -> None:
    original = copy.deepcopy(record)
    result = _checker().check_record(record)
    assert record == original
    assert result["snapshots_checked"] == len(record["snapshots"])
    assert result["consistent_final_next_actions"] == [1]
    assert result["policy_verified"] is False
    assert result["scientific_promotion_allowed"] is False


@pytest.mark.parametrize("name", ["lambda0", "lambda08", "scaled", "gamma099"])
def test_real_record_prefixes_match_without_network_or_learner_execution(name: str) -> None:
    path = Path(__file__).parent / "fixtures" / "riverswim_sarsa" / f"{name}.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    result = _checker().check_record(fixture["record"])
    assert result["events_checked"] == fixture["origin"]["prefix_events"] == 100
    assert result["snapshots_checked"] == 2
    assert result["source_provenance_verified"] is False


def test_finite_forged_q_snapshot_is_rejected(record: dict[str, Any]) -> None:
    record["snapshots"][2]["q_by_state_action"][0][0] += 0.25
    # This still satisfies the existing finite-shape-only snapshot checks.
    assert np.isfinite(record["snapshots"][2]["q_by_state_action"]).all()
    with pytest.raises(ValueError, match="snapshot 2.*q_by_state_action"):
        _checker().check_record(record)


@pytest.mark.parametrize("field", ["eligibility_l2", "eligibility_max_abs", "parameter_l2"])
def test_finite_forged_norm_is_rejected(record: dict[str, Any], field: str) -> None:
    record["snapshots"][2][field] += 0.25
    with pytest.raises(ValueError, match=field):
        _checker().check_record(record)


def test_final_snapshot_must_match_one_complete_bootstrap_branch(record: dict[str, Any]) -> None:
    record["snapshots"][-1]["q_by_state_action"][0][0] += 0.25
    with pytest.raises(ValueError, match="final snapshot"):
        _checker().check_record(record)


def test_impossible_transition_is_rejected(record: dict[str, Any]) -> None:
    record["positions"][0] = 2
    with pytest.raises(ValueError, match="transition"):
        _checker().check_record(record)


def test_reward_uses_previous_state_and_executed_action(record: dict[str, Any]) -> None:
    record["rewards"][0] = 0.0
    with pytest.raises(ValueError, match="reward"):
        _checker().check_record(record)


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_sizes": [4]},
        {"use_layer_norm": True},
        {"algorithm": "differential_sarsa"},
        {"normalizer": {}},
        {"n_actions": True},
    ],
)
def test_unsupported_config_fails_closed(record: dict[str, Any], change: dict[str, Any]) -> None:
    record["config"].update(change)
    with pytest.raises(ValueError, match="config"):
        _checker().check_record(record)


def test_initial_trace_and_parameter_norm_are_checked(record: dict[str, Any]) -> None:
    record["snapshots"][0]["eligibility_l2"] = 1.0
    with pytest.raises(ValueError, match="snapshot 0"):
        _checker().check_record(record)


def test_mixed_boolean_q_values_are_not_silently_coerced(record: dict[str, Any]) -> None:
    record["snapshots"][1]["q_by_state_action"][0][0] = True
    with pytest.raises(ValueError, match="without booleans"):
        _checker().check_record(record)


def test_duplicate_snapshot_is_rejected(record: dict[str, Any]) -> None:
    record["snapshots"].insert(2, copy.deepcopy(record["snapshots"][1]))
    with pytest.raises(ValueError, match="snapshot indices"):
        _checker().check_record(record)


def test_ambiguous_last_bootstrap_is_reported_instead_of_invented(record: dict[str, Any]) -> None:
    record["config"]["step_size"] = 0.0
    record.update(
        accepted_events=1,
        actions=[0],
        positions=[0],
        rewards=[record["environment"]["reward_left"]],
    )
    zero = [[0.0, 0.0] for _ in range(record["environment"]["n_states"])]
    record["snapshots"] = [
        {
            "accepted_events": 0,
            "q_by_state_action": zero,
            "eligibility_l2": 0.0,
            "eligibility_max_abs": 0.0,
            "parameter_l2": 0.0,
        },
        {
            "accepted_events": 1,
            "q_by_state_action": zero,
            "eligibility_l2": 2**0.5,
            "eligibility_max_abs": 1.0,
            "parameter_l2": 0.0,
        },
    ]
    assert _checker().check_record(record)["consistent_final_next_actions"] == [0, 1]


def test_cli_reads_record_without_mutation(
    record: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    original = path.read_bytes()
    assert _checker().main([str(path)]) == 0
    assert path.read_bytes() == original
    result = json.loads(capsys.readouterr().out)
    assert result["events_checked"] == 5
    assert result["source_provenance_verified"] is False


@pytest.mark.parametrize("raw", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_cli_rejects_ambiguous_or_nonfinite_json(
    raw: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "record.json"
    path.write_text(raw, encoding="utf-8")
    assert _checker().main([str(path)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "failed" in output.err


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted_events", True),
        ("accepted_events", 20001),
        ("schema", []),
        ("scientific_promotion_allowed", True),
    ],
)
def test_invalid_record_boundaries_fail_closed(
    record: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    record[field] = value
    with pytest.raises(ValueError):
        _checker().check_record(record)
