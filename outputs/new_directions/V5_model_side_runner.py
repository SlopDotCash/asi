"""V5 runner — model-side identification against reference-side weights.

Pre-registered in elizaOS/asi#1870. Development diagnostic, permanently
nonpromoting. Touches no pinned artifact, edits no registered source, consumes
no promotion seed.

Discharges ledger entry 15: V4's model-side arm was void because both sides
were correlated through one forward pass, collapsing onto position. Here the
reference descriptor comes from the model only and the post descriptor from
post-shift data only.

Per the pre-registration, both sanity controls are computed for every arm
*before* any online checkpoint, and an arm failing either is marked void.

Usage:
    python V5_model_side_runner.py --data-home <openml cache> --out V5_model_side.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.optimize import linear_sum_assignment

from alberta_framework.benchmarks import ipmnist_screening as screening
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    init_mlp_params,
    load_mnist_train,
)

SCHEMA = "alberta.new_directions.v5_model_side.v1"
SEEDS = (0, 1, 2)
BOUNDARIES = (0, 1, 2)
CHECKPOINTS = (50, 200, 500, 2000)
THRESHOLDS = (0.001, 0.01)
SOLVERS = ("hungarian", "greedy")
ARMS = ("F5a_weight_path", "F5b_gradient_affinity", "F5c_v1_data_side")
REFERENCE_ARM = "sigma0_ndecay099"
EMA_DECAY = 0.99
EMA_EPSILON = 1e-8
TASK_LENGTH = 5000
INPUT_DIM = 784
N_CLASSES = 10
ORACLE_GATE = 0.95
NO_SHIFT_GATE = 0.99


# --------------------------------------------------------------- reference


def ema_reference(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Annealed fast-EMA mean/var using the screening module's exact equations."""
    state = screening.EMANormState(
        mean=jnp.zeros(INPUT_DIM, dtype=jnp.float32),
        var=jnp.ones(INPUT_DIM, dtype=jnp.float32),
        count=jnp.array(0.0, dtype=jnp.float32),
    )

    def step(carry: Any, observation: Any) -> tuple[Any, None]:
        _, new_state = screening.ema_normalize(carry, observation, EMA_DECAY, EMA_EPSILON)
        return new_state, None

    final, _ = jax.lax.scan(step, state, jnp.asarray(samples))
    return np.asarray(final.mean), np.asarray(final.var)


def class_conditional_ema(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-class annealed fast-EMA means — V1's data-side reference (F5c)."""
    out = np.zeros((INPUT_DIM, N_CLASSES), dtype=np.float64)
    for c in range(N_CLASSES):
        rows = samples[labels == c]
        if rows.shape[0] == 0:
            continue
        mean, _ = ema_reference(rows)
        out[:, c] = mean
    return out


def _gate_products(params: dict[str, Any], samples: np.ndarray, mean_gate: bool) -> np.ndarray:
    """Return the (hidden1, classes) path matrix used by both model-side arms.

    ``mean_gate`` uses the mean activation pattern (F5a, data-free in the sense
    that only the reference task's gate statistics enter); otherwise the
    per-sample gate products are averaged exactly (F5b), which equals the mean
    input-output Jacobian because ``w1`` does not depend on the sample.
    """
    x = jnp.asarray(samples)
    z1 = x @ params["w1"] + params["b1"]
    a1 = jax.nn.relu(z1)
    z2 = a1 @ params["w2"] + params["b2"]
    g1 = (z1 > 0).astype(jnp.float32)
    g2 = (z2 > 0).astype(jnp.float32)
    if mean_gate:
        gate1 = g1.mean(axis=0)
        gate2 = g2.mean(axis=0)
        path = (gate1[:, None] * params["w2"]) @ (gate2[:, None] * params["w3"])
        return np.asarray(path, dtype=np.float64)

    def one(gate1: Any, gate2: Any) -> Any:
        return (gate1[:, None] * params["w2"]) @ (gate2[:, None] * params["w3"])

    # Accumulate in chunks: the per-sample intermediate is (hidden1, hidden2),
    # so vmapping the full oracle window at once would materialize tens of GB.
    batched = jax.jit(lambda a, b: jax.vmap(one)(a, b).sum(axis=0))
    total = np.zeros((params["w2"].shape[0], params["w3"].shape[1]), dtype=np.float64)
    chunk = 512
    for start in range(0, g1.shape[0], chunk):
        total += np.asarray(
            batched(g1[start : start + chunk], g2[start : start + chunk]),
            dtype=np.float64,
        )
    return total / float(g1.shape[0])


def class_conditional_exact(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Exact per-class means — the oracle control's data-side reference."""
    out = np.zeros((INPUT_DIM, N_CLASSES), dtype=np.float64)
    for c in range(N_CLASSES):
        rows = samples[labels == c]
        if rows.shape[0] == 0:
            continue
        out[:, c] = rows.astype(np.float64).mean(axis=0)
    return out


def reference_descriptors(
    params: dict[str, Any],
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    exact: bool = False,
) -> dict[str, np.ndarray]:
    """Reference-side descriptors, all in (position, class) space.

    ``exact`` selects the control estimator: the sanity controls compare
    exact full-dataset statistics on both sides, so F5c must use exact class
    means there. Its online estimator is the annealed fast-EMA (decay 0.99),
    whose ~100-sample effective window never converges to the full-dataset
    mean and so cannot clear an oracle gate by construction.
    """
    w1 = np.asarray(params["w1"], dtype=np.float64)
    data_side = (
        class_conditional_exact(samples, labels)
        if exact
        else class_conditional_ema(samples, labels)
    )
    return {
        "F5a_weight_path": w1 @ _gate_products(params, samples, mean_gate=True),
        "F5b_gradient_affinity": w1 @ _gate_products(params, samples, mean_gate=False),
        "F5c_v1_data_side": data_side,
    }


# -------------------------------------------------------------- post-shift


def post_descriptor(samples: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Class-conditional batch means over the post-shift window (shared by all arms)."""
    out = np.zeros((INPUT_DIM, N_CLASSES), dtype=np.float64)
    seen = 0
    for c in range(N_CLASSES):
        rows = samples[labels == c]
        if rows.shape[0] == 0:
            continue
        seen += 1
        out[:, c] = rows.astype(np.float64).mean(axis=0)
    return out, seen


# ------------------------------------------------------------------ matching


def _standardize(descriptor: np.ndarray) -> np.ndarray:
    mean = descriptor.mean(axis=0, keepdims=True)
    std = descriptor.std(axis=0, keepdims=True)
    return (descriptor - mean) / (std + 1e-12)


def _cost_matrix(reference: np.ndarray, post: np.ndarray) -> np.ndarray:
    left = _standardize(reference)
    right = _standardize(post)
    left_sq = (left**2).sum(axis=1)[:, None]
    right_sq = (right**2).sum(axis=1)[None, :]
    return np.maximum(left_sq + right_sq - 2.0 * (left @ right.T), 0.0)


def _greedy(cost: np.ndarray) -> np.ndarray:
    n_post = cost.shape[1]
    order = np.argsort(cost, axis=None, kind="stable")
    assignment = np.full(n_post, -1, dtype=np.int64)
    used = np.zeros(cost.shape[0], dtype=bool)
    filled = 0
    for flat in order:
        i, j = divmod(int(flat), n_post)
        if assignment[j] != -1 or used[i]:
            continue
        assignment[j] = i
        used[i] = True
        filled += 1
        if filled == n_post:
            break
    return assignment


def solve(cost: np.ndarray, solver: str) -> np.ndarray:
    if solver == "hungarian":
        rows, cols = linear_sum_assignment(cost)
        assignment = np.empty(cost.shape[1], dtype=np.int64)
        assignment[cols] = rows
        return assignment
    return _greedy(cost)


def score(
    reference: dict[str, np.ndarray],
    post: np.ndarray,
    truth: np.ndarray,
    relevant: dict[float, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    """Accuracy for every arm x solver against one post descriptor."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, descriptor in reference.items():
        out[arm] = {}
        cost = _cost_matrix(descriptor, post)
        for solver in SOLVERS:
            correct = solve(cost, solver) == truth
            entry = {"accuracy_all": float(correct.mean())}
            for threshold, mask in relevant.items():
                selected = mask[truth]
                entry[f"accuracy_relevant_var{threshold}"] = (
                    float(correct[selected].mean()) if selected.any() else 0.0
                )
                entry[f"n_relevant_var{threshold}"] = int(selected.sum())
            out[arm][solver] = entry
    return out


# ------------------------------------------------------------------- driving


def train_reference_network(
    seed: int, boundary: int, x: np.ndarray, y: np.ndarray, schedule: Any
) -> dict[str, Any]:
    """Run the reference arm online through tasks 0..boundary inclusive."""
    spec = screening.screening_spec(REFERENCE_ARM)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    config = IPMNISTConfig(n_tasks=max(BOUNDARIES) + 2, task_length=TASK_LENGTH)
    params = init_mlp_params(jr.key(np.uint32(seed)), config)
    state = init_fn(params)
    hyperparameters = spec.hyperparameters

    def one_step(carry: Any, batch: Any) -> tuple[Any, None]:
        carry_params, carry_state = carry
        sample, label = batch
        new_params, new_state, _ = step_fn(
            carry_params, carry_state, sample, label, hyperparameters
        )
        return (new_params, new_state), None

    for task in range(boundary + 1):
        permutation = np.asarray(schedule.permutations[task])
        indices = np.asarray(schedule.example_indices[task])
        (params, state), _ = jax.lax.scan(
            one_step,
            (params, state),
            (jnp.asarray(x[indices][:, permutation]), jnp.asarray(y[indices])),
        )
    return params


def cell_context(
    seed: int, boundary: int, x: np.ndarray, y: np.ndarray, schedule: Any, params: dict[str, Any]
) -> dict[str, Any]:
    """Everything a cell needs that does not depend on the post-shift window."""
    permutation_pre = np.asarray(schedule.permutations[boundary])
    permutation_post = np.asarray(schedule.permutations[boundary + 1])
    pre_indices = np.asarray(schedule.example_indices[boundary])
    pre_samples = x[pre_indices][:, permutation_pre]
    _, reference_var = ema_reference(pre_samples)
    return {
        "permutation_pre": permutation_pre,
        "permutation_post": permutation_post,
        "pre_samples": pre_samples,
        "pre_labels": y[pre_indices],
        "post_indices": np.asarray(schedule.example_indices[boundary + 1]),
        "reference_var": reference_var,
        "truth": np.argsort(permutation_pre)[permutation_post],
        "relevant": {t: reference_var > t for t in THRESHOLDS},
        "params": params,
    }


def run_controls(context: dict[str, Any], x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Oracle and no-shift controls, on exact full-dataset statistics."""
    params = context["params"]
    reference = reference_descriptors(
        params, x[:, context["permutation_pre"]], y, exact=True
    )
    results: dict[str, Any] = {}
    for name, permutation in (
        ("exact_statistic_oracle", context["permutation_post"]),
        ("no_shift", context["permutation_pre"]),
    ):
        post, _ = post_descriptor(x[:, permutation], y)
        truth = (
            context["truth"]
            if name == "exact_statistic_oracle"
            else np.arange(INPUT_DIM)
        )
        results[name] = score(reference, post, truth, context["relevant"])
    return results


def control_verdict(controls: dict[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {}
    for arm in ARMS:
        oracle = controls["exact_statistic_oracle"][arm]["hungarian"][
            "accuracy_relevant_var0.01"
        ]
        no_shift = controls["no_shift"][arm]["hungarian"]["accuracy_relevant_var0.01"]
        passed = oracle >= ORACLE_GATE and no_shift >= NO_SHIFT_GATE
        verdict[arm] = {
            "oracle": round(oracle, 6),
            "no_shift": round(no_shift, 6),
            "oracle_pass": oracle >= ORACLE_GATE,
            "no_shift_pass": no_shift >= NO_SHIFT_GATE,
            "void": not passed,
        }
    return verdict


def sample_floor(aggregates: list[dict[str, Any]], arm: str, solver: str) -> Any:
    points = sorted(
        (a["n_samples"], a["accuracy_relevant_var0.01_mean"])
        for a in aggregates
        if a["arm"] == arm and a["solver"] == solver
    )
    for (n0, a0), (n1, a1) in zip(points, points[1:]):
        if a0 < 0.90 <= a1:
            return float(n1) if a1 == a0 else float(n0 + (0.90 - a0) * (n1 - n0) / (a1 - a0))
    if points and points[0][1] >= 0.90:
        return float(points[0][0])
    return "> 2000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--boundaries", type=int, nargs="+", default=list(BOUNDARIES))
    args = parser.parse_args()

    started = time.time()
    x, y = load_mnist_train(args.data_home)
    config = IPMNISTConfig(n_tasks=max(args.boundaries) + 2, task_length=TASK_LENGTH)

    contexts: list[tuple[int, int, dict[str, Any]]] = []
    for seed in args.seeds:
        schedule = build_schedule(jr.key(np.uint32(seed)), config, x.shape[0])
        for boundary in args.boundaries:
            params = train_reference_network(seed, boundary, x, y, schedule)
            contexts.append(
                (seed, boundary, cell_context(seed, boundary, x, y, schedule, params))
            )

    # Controls first, for every arm, before any online checkpoint (pre-registered).
    first_seed, first_boundary, first_context = contexts[0]
    controls = run_controls(first_context, x, y)
    verdict = control_verdict(controls)

    cells: list[dict[str, Any]] = []
    for seed, boundary, context in contexts:
        reference = reference_descriptors(
            context["params"], context["pre_samples"], context["pre_labels"]
        )
        for n_samples in CHECKPOINTS:
            take = context["post_indices"][:n_samples]
            post, classes_seen = post_descriptor(
                x[take][:, context["permutation_post"]], y[take]
            )
            scored = score(reference, post, context["truth"], context["relevant"])
            for arm, by_solver in scored.items():
                for solver, entry in by_solver.items():
                    cells.append(
                        {
                            "seed": seed,
                            "boundary_task": boundary,
                            "arm": arm,
                            "solver": solver,
                            "n_samples": int(n_samples),
                            "classes_observed": classes_seen,
                            **entry,
                        }
                    )

    aggregates: list[dict[str, Any]] = []
    for arm in ARMS:
        for solver in SOLVERS:
            for n_samples in CHECKPOINTS:
                group = [
                    c
                    for c in cells
                    if c["arm"] == arm and c["solver"] == solver and c["n_samples"] == n_samples
                ]
                if not group:
                    continue
                entry: dict[str, Any] = {
                    "arm": arm,
                    "solver": solver,
                    "n_samples": n_samples,
                    "n_cells": len(group),
                    "accuracy_all_mean": float(np.mean([g["accuracy_all"] for g in group])),
                }
                for threshold in THRESHOLDS:
                    key = f"accuracy_relevant_var{threshold}"
                    values = [g[key] for g in group]
                    entry[f"{key}_mean"] = float(np.mean(values))
                    entry[f"{key}_min"] = float(np.min(values))
                aggregates.append(entry)

    floors = {
        f"{arm}/{solver}": (
            "void" if verdict[arm]["void"] else sample_floor(aggregates, arm, solver)
        )
        for arm in ARMS
        for solver in SOLVERS
    }
    live = [
        a
        for a in aggregates
        if a["n_samples"] <= 500 and not verdict[a["arm"]]["void"]
    ]
    best = max(live, key=lambda a: a["accuracy_relevant_var0.01_min"], default=None)

    artifact = {
        "schema": SCHEMA,
        "cells": cells,
        "aggregates": aggregates,
        "controls": controls,
        "control_verdict": verdict,
        "control_cell": {"seed": first_seed, "boundary_task": first_boundary},
        "protocol": {
            "data": "IPMNIST protocol MNIST train split (load_mnist_train)",
            "permutations": "build_schedule per-seed protocol permutations",
            "post_descriptor": "class-conditional batch means over the first N post-shift samples",
            "reference_arm": REFERENCE_ARM,
            "reference_variance_for_relevance": (
                "annealed fast-EMA (decay 0.99), identical across arms"
            ),
            "seeds": list(args.seeds),
            "boundaries": list(args.boundaries),
            "sample_checkpoints": list(CHECKPOINTS),
            "relevance_thresholds_on_reference_var": list(THRESHOLDS),
            "arms": list(ARMS),
            "solvers": list(SOLVERS),
        },
        "promotion": {
            "criterion": (
                ">90% of relevant (var > 0.01) pixels correctly assigned within "
                "<=500 samples (min over seed x boundary cells)"
            ),
            "promoted": bool(best and best["accuracy_relevant_var0.01_min"] > 0.90),
            "best_configuration": None
            if best is None
            else {
                "arm": best["arm"],
                "solver": best["solver"],
                "n_samples": best["n_samples"],
                "accuracy_relevant_var0.01_min": best["accuracy_relevant_var0.01_min"],
            },
            "sample_floor_n_star": floors,
        },
        "evidence_policy": (
            "development_screening_diagnostic; permanently nonpromoting. No frozen "
            "protocol, no held-out seed, no registered source edited."
        ),
        "wall_clock_seconds": round(time.time() - started, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} in {artifact['wall_clock_seconds']}s")
    print(f"promoted={artifact['promotion']['promoted']}")
    for arm, v in verdict.items():
        print(f"  {arm:24s} oracle={v['oracle']:.4f} no_shift={v['no_shift']:.4f} void={v['void']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
