"""Nonpromoting linear SARSA consumer of the Intentional TD value optimizer.

Run ``python -m alberta_framework.benchmarks.intentional_td_development --plan``
before executing the literal development comparison. This is separate from
the frozen 144-shard reference-life scorecard and does not populate it.
"""

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.intentional_td import (
    IntentionalTDConfig,
    init_intentional_td,
    intentional_td_update,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

SEEDS = (15610, 15611, 15612, 15613)
STEPS = 10_000
PHASE_LENGTH = 500
EPSILON = 0.1
RIVERSWIM_STATES = 3
ARMS = {
    "intentional": IntentionalTDConfig(eta=0.1, lamda=0.8),
    "intentional_no_trace": IntentionalTDConfig(eta=0.1, lamda=0.0),
    "intentional_no_rms": IntentionalTDConfig(eta=0.1, lamda=0.8, use_rmsprop=False),
    "fixed_trace": IntentionalTDConfig(eta=0.05, lamda=0.8, enabled=False),
    "fixed_step": IntentionalTDConfig(eta=0.05, lamda=0.0, enabled=False),
}
ENVIRONMENTS = ("switching", "riverswim")


def development_plan() -> dict[str, Any]:
    """Literal plan; no workload or seed reservation occurs here."""
    return {
        "schema": "asi.intentional_td.development_plan.v1",
        "issue": "https://github.com/SlopDotCash/asi/issues/1561",
        "paper": "arXiv:2604.19033v1",
        "official_code": "sharifnassab/Intentional_RL@e86e26fd8613ac212e9a52c3fed8a01d0a31f685",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reference_dev": False,
        "seeds": list(SEEDS),
        "steps_per_arm": STEPS,
        "environments": {
            "switching_phase_length": PHASE_LENGTH, "riverswim_states": RIVERSWIM_STATES,
        },
        "arms": {name: dataclasses.asdict(config) for name, config in ARMS.items()},
        "epsilon": EPSILON,
        "initial_parameters": "zero linear action values on one-hot observations; no bias",
        "metric": "mean reward per transition; phase means on SwitchingTwoState",
        "baseline": "fresh fixed_trace and fixed_step arms; no historical matched value exists",
        "selection_rule": "no tuning; report every arm, seed, failure and negative result",
        "decision_rule": (
            "development benefit only if intentional beats both fixed controls by mean reward "
            ">0.02 in both environments and every paired seed delta is positive; otherwise "
            "negative or inconclusive. This never promotes an arm."
        ),
        "deviations": [
            "Linear on-policy SARSA consumer, not deep Q-learning or policy-gradient reproduction",
            "Uses official code's bias-corrected sigma mean, not paper Eq. 12's unnormalized sum",
            "No replay, pretraining, reset at switches, task IDs, or privileged oracle inputs",
            "Shared root keys and policy/execution key schedules; actions/states may diverge",
            "Fixed controls retain unused state arrays as explicit accounting ballast",
            "Wall time includes compilation and is telemetry only, not compute parity",
        ],
    }


def _action(weights: jax.Array, observation: jax.Array, key: jax.Array) -> jax.Array:
    explore_key, random_key, tie_key = jr.split(key, 3)
    values = weights @ observation
    greedy = jr.categorical(tie_key, jnp.where(values == jnp.max(values), 0.0, -jnp.inf))
    random = jr.randint(random_key, (), 0, weights.shape[0])
    return jnp.where(jr.uniform(explore_key) < EPSILON, random, greedy).astype(jnp.int32)


def _numeric_bytes(tree: Any) -> int:
    return sum(np.asarray(value).nbytes for value in jax.tree.leaves(tree))


def run_arm(environment: str, arm: str, seed: int) -> dict[str, Any]:
    """One uninterrupted, fixed-budget development life in an existing MDP."""
    if type(environment) is not str or environment not in ENVIRONMENTS:
        raise ValueError(f"unknown development environment: {environment!r}")
    config = ARMS[arm]
    env: Any = (
        SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=PHASE_LENGTH))  # type: ignore[call-arg]
        if environment == "switching"
        else RiverSwimMDP(RiverSwimConfig(n_states=RIVERSWIM_STATES))  # type: ignore[call-arg]
    )
    key = jr.key(seed, impl="threefry2x32")
    env_state = env.init(jr.fold_in(key, 0))
    observation = env.observe(env_state)
    weights = jnp.zeros((2, observation.size), dtype=jnp.float32)
    state = init_intentional_td(weights)
    action = _action(weights, observation, jr.fold_in(key, 1))
    initial_bytes = _numeric_bytes((weights, state, env_state, observation, action)) + 8
    agent_bytes = _numeric_bytes((weights, state))

    def transition(carry: Any, index: jax.Array) -> tuple[Any, Any]:
        weights, state, env_state, observation, action = carry
        transition_key = jr.fold_in(key, 2 + 2 * index)
        policy_key = jr.fold_in(key, 3 + 2 * index)
        next_observation, reward, next_env = env.step(env_state, action, transition_key)
        next_action = _action(weights, next_observation, policy_key)
        prediction = weights[action] @ observation
        target = reward + config.gamma * (weights[next_action] @ next_observation)
        error = target - prediction
        gradient = jnp.zeros_like(weights).at[action].set(observation)
        updated_weights, updated_state = intentional_td_update(
            weights, state, gradient, error, config
        )
        return (updated_weights, updated_state, next_env, next_observation, next_action), (
            reward,
            error,
        )

    started = time.perf_counter()
    final, (rewards, errors) = jax.jit(
        lambda carry: jax.lax.scan(transition, carry, jnp.arange(STEPS, dtype=jnp.int32))
    )((weights, state, env_state, observation, action))
    jax.block_until_ready(final)  # type: ignore[no-untyped-call]
    elapsed = time.perf_counter() - started
    rewards_host = np.asarray(rewards, dtype=np.float64)
    errors_host = np.asarray(errors, dtype=np.float64)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(final)):
        raise ValueError(f"nonfinite final state: {environment}/{arm}/{seed}")
    if not np.isfinite(errors_host).all() or not np.isfinite(rewards_host).all():
        raise ValueError(f"nonfinite trajectory: {environment}/{arm}/{seed}")
    final_bytes = _numeric_bytes(final) + 8
    if final_bytes != initial_bytes or int(final[1].step) != STEPS:
        raise ValueError("state size or update count drifted")
    return {
        "environment": environment,
        "arm": arm,
        "seed": seed,
        "mean_reward": float(rewards_host.mean()),
        "phase_mean_rewards": rewards_host.reshape(-1, PHASE_LENGTH).mean(axis=1).tolist(),
        "rms_preupdate_td_error": float(np.sqrt(np.mean(errors_host**2))),
        "environment_steps": STEPS,
        "updates": STEPS,
        "prediction_queries": 3 * STEPS + 1,
        "agent_persistent_numeric_bytes": agent_bytes,
        "initial_dynamic_numeric_bytes": initial_bytes,
        "final_dynamic_numeric_bytes": final_bytes,
        "accounting_scope": "dynamic arrays only; excludes static environment tables/executable",
        "retained_trajectory_numeric_bytes": rewards.nbytes + errors.nbytes,
        "final_weights": np.asarray(final[0]).tolist(),
        "reward_sha256": hashlib.sha256(np.asarray(rewards).tobytes()).hexdigest(),
        "wall_seconds_including_compile": elapsed,
    }


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for path in sorted((root / "alberta_framework").rglob("*.py")):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python_sources_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "sources": sources,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "jax_enable_x64": jax.config.jax_enable_x64,
        "backend": jax.default_backend(),
        "machine": platform.machine(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = development_plan()
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None:
        parser.error("--output is required for execution")
    # Claim a NEW output before source inspection or any workload. Ordinary
    # source/run exceptions become explicit failures; output I/O and process
    # interruption are outside this development-only publication guarantee.
    with args.output.open("x", encoding="utf-8") as output:
        identity: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        try:
            try:
                identity = _source_identity()
            except Exception as error:
                failure: dict[str, Any] = {
                    "phase": "source_identity",
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
                failures.append(failure)
                print(json.dumps(failure), flush=True)
            else:
                for environment in ENVIRONMENTS:
                    for seed in SEEDS:
                        for arm in ARMS:
                            try:
                                row = run_arm(environment, arm, seed)
                            except Exception as error:
                                failure = {
                                    "environment": environment,
                                    "arm": arm,
                                    "seed": seed,
                                    "error": str(error),
                                    "error_type": type(error).__name__,
                                }
                                failures.append(failure)
                                print(json.dumps(failure), flush=True)
                            else:
                                rows.append(row)
                                print(json.dumps(row), flush=True)
        finally:
            result = {"plan": plan, "source": identity, "runs": rows, "failures": failures}
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
