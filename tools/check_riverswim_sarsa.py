"""Check recorded linear SARSA diagnostics without rerunning an environment.

This is an additional, nonpromoting consistency check for the trace records
proposed in ASI PRs #3220 and #3225. It does not replace their validators.
It conditions on the recorded actions and initial Q values, with zero initial
biases/traces. It verifies neither initialization provenance nor policy/RNG.
The last bootstrap action is missing from these schemas: both possibilities
are checked and the compatible actions are reported, never silently inferred.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

MAX_EVENTS = 20_000
MAX_INPUT_BYTES = 32 * 1024 * 1024
# This is a diagnostic float32 replay tolerance, not a scientific gate. NumPy
# and XLA can fuse multiply-adds differently. It is reported with every result
# and must not be tuned to certify a particular artifact or learning claim.
ABSOLUTE_TOLERANCE = 1e-4
SUPPORTED_SCHEMAS = {
    "asi.riverswim_sarsa_step_size_diagnostic.shard.v1",
    "asi.riverswim_sarsa_discount_horizon.shard.v1",
}
CONFIG_FIELDS = {
    "algorithm",
    "boundary_mode",
    "environment_kind",
    "epsilon_decay_steps",
    "epsilon_end",
    "epsilon_start",
    "gamma",
    "hidden_sizes",
    "initial_key_contract",
    "lamda",
    "leaky_relu_slope",
    "n_actions",
    "observation_dim",
    "rng_implementation",
    "rng_schedule",
    "schema",
    "sparsity",
    "step_size",
    "use_layer_norm",
}
type FloatArray = NDArray[np.float32]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, label: str, *, lower: float, upper: float) -> float:
    _require(type(value) in (int, float), f"{label} must be a real JSON number")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds numeric bounds") from exc
    _require(math.isfinite(converted) and lower <= converted <= upper, f"{label} out of bounds")
    return converted


def _integer(value: Any, label: str, *, lower: int, upper: int) -> int:
    _require(type(value) is int and lower <= value <= upper, f"{label} out of bounds")
    return int(value)


def _array(value: Any, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    _require(type(value) is list, f"{label} must be a JSON array")
    _require(
        all(type(row) is list and all(type(v) in (int, float) for v in row) for row in value),
        f"{label} must contain numeric rows without booleans",
    )
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a rectangular numeric array") from exc
    _require(array.shape == shape and array.dtype.kind in "fiu", f"{label} shape/type")
    result = array.astype(np.float64)
    _require(bool(np.isfinite(result).all()), f"{label} must be finite")
    return result


def _snapshot_matches(
    snapshot: dict[str, Any],
    weights: FloatArray,
    biases: FloatArray,
    traces: FloatArray,
    bias_traces: FloatArray,
) -> None:
    label = f"snapshot {snapshot['accepted_events']}"
    observed = _array(snapshot.get("q_by_state_action"), (weights.shape[1], 2), label)
    expected = (weights.T + biases).astype(np.float64)
    _require(
        bool(np.isfinite(expected).all())
        and bool(np.allclose(observed, expected, rtol=0.0, atol=ABSOLUTE_TOLERANCE)),
        f"{label}: q_by_state_action differs from independent recurrence",
    )
    flat_traces = np.concatenate((traces.ravel(), bias_traces)).astype(np.float64)
    flat_parameters = np.concatenate((weights.ravel(), biases)).astype(np.float64)
    metrics = {
        "eligibility_l2": float(np.linalg.norm(flat_traces)),
        "eligibility_max_abs": float(np.max(np.abs(flat_traces))),
        "parameter_l2": float(np.linalg.norm(flat_parameters)),
    }
    for name, expected_metric in metrics.items():
        actual = _number(snapshot.get(name), f"{label}: {name}", lower=0.0, upper=1e30)
        _require(
            math.isfinite(expected_metric) and abs(actual - expected_metric) <= ABSOLUTE_TOLERANCE,
            f"{label}: {name} differs from independent recurrence",
        )


def _advance(
    weights: FloatArray,
    biases: FloatArray,
    traces: FloatArray,
    bias_traces: FloatArray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action: int,
    alpha: np.float32,
    gamma: np.float32,
    lamda: np.float32,
) -> None:
    # Independent scalar TD and linear-feature equations; do not call the
    # production SARSA/Horde update or its trace helpers.
    previous_q = np.float32(weights[action, state] + biases[action])
    next_q = np.float32(weights[next_action, next_state] + biases[next_action])
    delta = np.float32(np.float32(reward) + gamma * next_q - previous_q)
    if lamda == 0:
        # Production storage freezes inactive head traces at lambda=0. They
        # receive no credit, but remain present in the recorded trace norms.
        traces[action] = 0
        traces[action, state] = 1
        bias_traces[action] = 1
        weights[action] += alpha * delta * traces[action]
        biases[action] += alpha * delta
    else:
        decay = np.float32(gamma * lamda)
        traces *= decay
        bias_traces *= decay
        traces[action, state] += 1
        bias_traces[action] += 1
        weights += alpha * delta * traces
        biases += alpha * delta * bias_traces


def check_record(record: dict[str, Any]) -> dict[str, Any]:
    """Reject inconsistent snapshots; never mutate the input or create output files."""
    _require(type(record) is dict, "record must be an object")
    schema = record.get("schema")
    _require(type(schema) is str and schema in SUPPORTED_SCHEMAS, "unsupported record schema")
    _require(record.get("development_only") is True, "development_only must be true")
    for flag in ("scientific_promotion_allowed", "reference_dev_population_allowed"):
        _require(record.get(flag) is False, f"{flag} must be false")
    config, env = record.get("config"), record.get("environment")
    _require(type(config) is dict and set(config) == CONFIG_FIELDS, "unsupported config fields")
    _require(type(env) is dict, "environment must be an object")
    config = cast(dict[str, Any], config)
    env = cast(dict[str, Any], env)
    required_config = {
        "algorithm": "discounted_sarsa",
        "environment_kind": "riverswim",
        "boundary_mode": "continuing_unit_discount_only",
        "hidden_sizes": [],
        "use_layer_norm": False,
        "schema": "asi.reference_life_control_config.preview1",
        "rng_implementation": "threefry2x32",
        "rng_schedule": "owned_sarsa_key.preview1",
        "initial_key_contract": "scalar_threefry2x32_typed_or_legacy.preview1",
    }
    for name, expected in required_config.items():
        _require(
            type(config[name]) is type(expected) and config[name] == expected,
            f"unsupported config {name}",
        )
    _integer(config["n_actions"], "config n_actions", lower=2, upper=2)
    n_states = _integer(env.get("n_states"), "environment n_states", lower=2, upper=12)
    _integer(config["observation_dim"], "config observation_dim", lower=n_states, upper=n_states)
    state = _integer(env.get("initial_state"), "initial_state", lower=0, upper=n_states - 1)
    _integer(config["epsilon_decay_steps"], "config epsilon_decay_steps", lower=0, upper=2**31 - 1)
    for name in ("epsilon_start", "epsilon_end", "leaky_relu_slope"):
        _number(config[name], f"config {name}", lower=0, upper=1)
    _number(config["sparsity"], "config sparsity", lower=0, upper=0)
    alpha = np.float32(_number(config["step_size"], "config step_size", lower=0, upper=1))
    gamma = np.float32(_number(config["gamma"], "config gamma", lower=0, upper=1))
    lamda = np.float32(_number(config["lamda"], "config lamda", lower=0, upper=1))
    up = _number(env.get("p_right_up"), "p_right_up", lower=0, upper=1)
    down = _number(env.get("p_right_down"), "p_right_down", lower=0, upper=1)
    _require(up > 0 and down > 0 and up + down <= 1, "invalid transition probabilities")
    left_reward = _number(env.get("reward_left"), "reward_left", lower=-1e6, upper=1e6)
    right_reward = _number(env.get("reward_right"), "reward_right", lower=-1e6, upper=1e6)
    count = _integer(record.get("accepted_events"), "accepted_events", lower=1, upper=MAX_EVENTS)
    actions, rewards, positions = (record.get(k) for k in ("actions", "rewards", "positions"))
    for name, values in (("actions", actions), ("rewards", rewards), ("positions", positions)):
        _require(type(values) is list and len(values) == count, f"{name} length/type")
    actions, rewards, positions = (cast(list[Any], a) for a in (actions, rewards, positions))
    snapshots = record.get("snapshots")
    _require(type(snapshots) is list and 2 <= len(snapshots) <= 32, "snapshot count/type")
    snapshots = cast(list[dict[str, Any]], snapshots)
    _require(all(type(s) is dict for s in snapshots), "snapshots must be objects")
    raw_indices = [s.get("accepted_events") for s in snapshots]
    _require(all(type(i) is int and 0 <= i <= count for i in raw_indices), "snapshot indices")
    indices = cast(list[int], raw_indices)
    _require(
        indices == sorted(set(indices)) and indices[0] == 0 and indices[-1] == count,
        "snapshot indices must be unique, ordered and include both endpoints",
    )
    by_index = {s["accepted_events"]: s for s in snapshots}
    initial = _array(snapshots[0].get("q_by_state_action"), (n_states, 2), "initial Q")
    _require(bool((np.abs(initial) <= 1e6).all()), "initial Q exceeds supported range")
    weights = initial.T.astype(np.float32).copy()
    biases = np.zeros(2, dtype=np.float32)
    traces = np.zeros_like(weights)
    bias_traces = np.zeros_like(biases)
    _snapshot_matches(snapshots[0], weights, biases, traces, bias_traces)
    final_actions = []
    with np.errstate(over="raise", invalid="raise"):
        try:
            for index in range(count):
                action = _integer(actions[index], "action", lower=0, upper=1)
                next_state = _integer(positions[index], "position", lower=0, upper=n_states - 1)
                reward = _number(rewards[index], "reward", lower=-1e6, upper=1e6)
                expected_reward = (
                    left_reward
                    if state == 0 and action == 0
                    else right_reward
                    if state == n_states - 1 and action == 1
                    else 0.0
                )
                _require(reward == expected_reward, f"event {index + 1}: reward lattice")
                if action == 0:
                    transition_possible = next_state == max(0, state - 1)
                else:
                    transition_possible = abs(next_state - state) <= 1
                    if next_state == state and up + down == 1 and 0 < state < n_states - 1:
                        transition_possible = False
                _require(transition_possible, f"event {index + 1}: impossible transition")
                if index + 1 == count:
                    for next_action in (0, 1):
                        w, b, z, zb = (a.copy() for a in (weights, biases, traces, bias_traces))
                        _advance(
                            w,
                            b,
                            z,
                            zb,
                            state,
                            action,
                            reward,
                            next_state,
                            next_action,
                            alpha,
                            gamma,
                            lamda,
                        )
                        try:
                            _snapshot_matches(by_index[count], w, b, z, zb)
                        except ValueError:
                            continue
                        final_actions.append(next_action)
                    _require(bool(final_actions), "final snapshot matches neither bootstrap action")
                else:
                    next_action = _integer(actions[index + 1], "action", lower=0, upper=1)
                    _advance(
                        weights,
                        biases,
                        traces,
                        bias_traces,
                        state,
                        action,
                        reward,
                        next_state,
                        next_action,
                        alpha,
                        gamma,
                        lamda,
                    )
                    if index + 1 in by_index:
                        _snapshot_matches(by_index[index + 1], weights, biases, traces, bias_traces)
                state = next_state
        except FloatingPointError as exc:
            raise ValueError("independent recurrence exceeds finite float32 range") from exc
    return {
        "schema": "asi.riverswim_sarsa_diagnostic_consistency.v1",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reference_dev_population_allowed": False,
        "events_checked": count,
        "snapshots_checked": len(snapshots),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "consistent_final_next_actions": final_actions,
        "policy_verified": False,
        "environment_rng_verified": False,
        "source_provenance_verified": False,
        "conditioning": "recorded actions and initial Q; zero initial biases and traces",
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        _require(name not in result, f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="existing JSON record; opened read-only")
    args = parser.parse_args(argv)
    try:
        with args.record.open("rb") as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
        _require(len(raw) <= MAX_INPUT_BYTES, "record exceeds input byte limit")
        record = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        result = check_record(record)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        print(f"diagnostic consistency check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
