"""V4 runner — higher-order permutation fingerprints across a boundary.

Pre-registered in elizaOS/asi#1311. Development diagnostic, permanently
nonpromoting (``development_screening_diagnostic``). Touches no pinned
artifact, edits no registered source, consumes no promotion seed.

Protocol is inherited from V1 (``V1_assignment.md``) except for the deviations
recorded in ``V4_fingerprints.md``: seeds {0,1,2}
x boundaries {0,1,2}, post-shift checkpoints N in {50,200,500,2000}, reference
statistics = annealed fast-EMA (decay 0.99, ``ema_normalize`` equations) over
the 5,000-sample pre-shift task, relevance on reference variance, Hungarian
and global-greedy solvers.

F4 uses ``sigma0_ndecay099`` — the arm whose plain annealed fast-EMA is the
pre-registered reference — resolved in advance on the issue.

Usage:
    python V4_fingerprints_runner.py --data-home <openml cache> \
        --out V4_fingerprints.json
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

SCHEMA = "alberta.new_directions.v4_fingerprints.v1"
SEEDS = (0, 1, 2)
BOUNDARIES = (0, 1, 2)
CHECKPOINTS = (50, 200, 500, 2000)
THRESHOLDS = (0.001, 0.01)
SOLVERS = ("hungarian", "greedy")
FINGERPRINTS = ("F3a_rowsum", "F3b_topk", "F3c_spectral", "F4a_act_corr", "F4b_grad_corr")
TOPK = 16
SPECTRAL_M = 8
REFERENCE_ARM = "sigma0_ndecay099"
EMA_DECAY = 0.99
EMA_EPSILON = 1e-8
TASK_LENGTH = 5000
INPUT_DIM = 784


# ---------------------------------------------------------------- statistics


def ema_reference(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Annealed fast-EMA mean/var, using the screening module's exact equations."""
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


def _safe_corrcoef(samples: np.ndarray) -> np.ndarray:
    """Pixel-pixel correlation with constant columns mapped to zero correlation."""
    centered = samples - samples.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    live = std > 1e-12
    scaled = np.zeros_like(centered)
    scaled[:, live] = centered[:, live] / std[live]
    correlation = (scaled.T @ scaled) / max(samples.shape[0], 1)
    correlation[~live, :] = 0.0
    correlation[:, ~live] = 0.0
    return correlation.astype(np.float64)


# --------------------------------------------------------------- descriptors


def f3_descriptors(samples: np.ndarray) -> dict[str, np.ndarray]:
    """F3a/F3b/F3c: permutation-equivariant reductions of pixel-pixel correlation."""
    correlation = _safe_corrcoef(samples)
    np.fill_diagonal(correlation, 0.0)

    rowsum = correlation.sum(axis=1, keepdims=True)

    magnitude = np.abs(correlation)
    k = min(TOPK, magnitude.shape[1])
    topk = -np.sort(-magnitude, axis=1)[:, :k]

    # Symmetric by construction; eigh is stable and ordered ascending.
    values, vectors = np.linalg.eigh(correlation)
    order = np.argsort(-np.abs(values))[:SPECTRAL_M]
    leading = vectors[:, order]
    # Deterministic, permutation-invariant sign convention: positive column sum.
    sums = leading.sum(axis=0)
    signs = np.where(sums < 0.0, -1.0, 1.0)
    signs = np.where(np.abs(sums) < 1e-12, 1.0, signs)
    spectral = leading * signs
    spectral = spectral * np.sqrt(np.abs(values[order]))[None, :]

    return {
        "F3a_rowsum": rowsum,
        "F3b_topk": topk,
        "F3c_spectral": spectral,
    }


def f4_descriptors(
    samples: np.ndarray, labels: np.ndarray, params: dict[str, Any]
) -> dict[str, np.ndarray]:
    """F4a/F4b: coupling between each input pixel and the first hidden layer."""
    x = jnp.asarray(samples)

    def hidden(batch: Any) -> Any:
        return jax.nn.relu(batch @ params["w1"] + params["b1"])

    activations = np.asarray(hidden(x), dtype=np.float64)

    def _coupling(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_c = left - left.mean(axis=0, keepdims=True)
        right_c = right - right.mean(axis=0, keepdims=True)
        left_s = left_c.std(axis=0)
        right_s = right_c.std(axis=0)
        left_live = left_s > 1e-12
        right_live = right_s > 1e-12
        left_z = np.zeros_like(left_c)
        right_z = np.zeros_like(right_c)
        left_z[:, left_live] = left_c[:, left_live] / left_s[left_live]
        right_z[:, right_live] = right_c[:, right_live] / right_s[right_live]
        return (left_z.T @ right_z) / max(left.shape[0], 1)

    act_corr = _coupling(samples.astype(np.float64), activations)

    # F4b: batch correlation of |dL/dx_i| with each hidden unit, using the
    # protocol's own softmax cross-entropy on the post-prediction label.  The
    # preregistration called this an EMA; that implementation deviation is
    # recorded explicitly and the arm is void under its oracle gate.
    def per_example_input_grad(sample: Any, label: Any) -> Any:
        def loss(inp: Any) -> Any:
            logits, _, _, _, _ = screening._forward_with_activations(params, inp)  # noqa: SLF001
            return -jax.nn.log_softmax(logits)[label]

        return jnp.abs(jax.grad(loss)(sample))

    grads = np.asarray(
        jax.vmap(per_example_input_grad)(x, jnp.asarray(labels)), dtype=np.float64
    )
    grad_corr = _coupling(grads, activations)

    return {"F4a_act_corr": act_corr, "F4b_grad_corr": grad_corr}


def effective_rank(samples: np.ndarray, params: dict[str, Any]) -> float:
    """Participation-ratio effective rank of the first-layer activation covariance."""
    activations = np.asarray(
        jax.nn.relu(jnp.asarray(samples) @ params["w1"] + params["b1"]), dtype=np.float64
    )
    centered = activations - activations.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(samples.shape[0], 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = eigenvalues.sum()
    if total <= 0.0:
        return 0.0
    return float((total**2) / float((eigenvalues**2).sum()))


# ------------------------------------------------------------------ matching


def _standardize(descriptor: np.ndarray) -> np.ndarray:
    """Z-score each descriptor dimension across pixels (permutation-invariant)."""
    mean = descriptor.mean(axis=0, keepdims=True)
    std = descriptor.std(axis=0, keepdims=True)
    return (descriptor - mean) / (std + 1e-12)


def _cost_matrix(reference: np.ndarray, post: np.ndarray) -> np.ndarray:
    left = _standardize(reference)
    right = _standardize(post)
    left_sq = (left**2).sum(axis=1)[:, None]
    right_sq = (right**2).sum(axis=1)[None, :]
    return np.maximum(left_sq + right_sq - 2.0 * (left @ right.T), 0.0)


def _greedy_assignment(cost: np.ndarray) -> np.ndarray:
    """Global greedy: repeatedly take the cheapest remaining (post, reference) pair."""
    n_post = cost.shape[1]
    order = np.argsort(cost, axis=None, kind="stable")
    assignment = np.full(n_post, -1, dtype=np.int64)
    used_reference = np.zeros(cost.shape[0], dtype=bool)
    filled = 0
    for flat in order:
        reference_index, post_index = divmod(int(flat), n_post)
        if assignment[post_index] != -1 or used_reference[reference_index]:
            continue
        assignment[post_index] = reference_index
        used_reference[reference_index] = True
        filled += 1
        if filled == n_post:
            break
    return assignment


def solve(cost: np.ndarray, solver: str) -> np.ndarray:
    """Return ``assignment[post_position] = reference_position``."""
    if solver == "hungarian":
        rows, cols = linear_sum_assignment(cost)
        assignment = np.empty(cost.shape[1], dtype=np.int64)
        assignment[cols] = rows
        return assignment
    return _greedy_assignment(cost)


# ------------------------------------------------------------------- driving


def train_reference_network(
    seed: int, boundary: int, x: np.ndarray, y: np.ndarray, schedule: Any
) -> dict[str, Any]:
    """Run ``sigma0_ndecay099`` online through tasks 0..boundary inclusive."""
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
        samples = jnp.asarray(x[indices][:, permutation])
        labels = jnp.asarray(y[indices])
        (params, state), _ = jax.lax.scan(one_step, (params, state), (samples, labels))
    return params


def run_cell(
    seed: int,
    boundary: int,
    x: np.ndarray,
    y: np.ndarray,
    schedule: Any,
    params: dict[str, Any],
    oracle: bool = False,
    no_shift: bool = False,
) -> list[dict[str, Any]]:
    """Measure every fingerprint x checkpoint x solver for one seed x boundary cell."""
    permutation_pre = np.asarray(schedule.permutations[boundary])
    permutation_post = (
        permutation_pre if no_shift else np.asarray(schedule.permutations[boundary + 1])
    )
    pre_indices = np.asarray(schedule.example_indices[boundary])
    post_indices = np.asarray(schedule.example_indices[boundary + 1])

    pre_samples = x[pre_indices][:, permutation_pre]
    reference_mean, reference_var = ema_reference(pre_samples)

    inverse_pre = np.argsort(permutation_pre)
    truth = inverse_pre[permutation_post]

    relevant = {
        threshold: reference_var > threshold for threshold in THRESHOLDS
    }

    # Both controls are pipeline-correctness checks, so both use exact
    # full-dataset statistics: running them at the online checkpoints would
    # conflate estimator noise with an implementation fault.
    exact = oracle or no_shift
    if exact:
        reference_source = x[:, permutation_pre]
        reference_labels = y
        checkpoints: tuple[int, ...] = (x.shape[0],)
    else:
        reference_source = pre_samples
        reference_labels = y[pre_indices]
        checkpoints = CHECKPOINTS

    reference_descriptors = {
        **f3_descriptors(reference_source),
        **f4_descriptors(reference_source, reference_labels, params),
    }
    rank = effective_rank(reference_source, params)

    rows: list[dict[str, Any]] = []
    for n_samples in checkpoints:
        if exact:
            post_samples = x[:, permutation_post]
            post_labels = y
        else:
            take = post_indices[:n_samples]
            post_samples = x[take][:, permutation_post]
            post_labels = y[take]

        post_descriptors = {
            **f3_descriptors(post_samples),
            **f4_descriptors(post_samples, post_labels, params),
        }

        for fingerprint in FINGERPRINTS:
            cost = _cost_matrix(reference_descriptors[fingerprint], post_descriptors[fingerprint])
            for solver in SOLVERS:
                assignment = solve(cost, solver)
                correct = assignment == truth
                row: dict[str, Any] = {
                    "seed": seed,
                    "boundary_task": boundary,
                    "fingerprint": fingerprint,
                    "solver": solver,
                    "n_samples": int(n_samples),
                    "accuracy_all": float(correct.mean()),
                    "activation_effective_rank": rank,
                }
                for threshold, mask in relevant.items():
                    selected = mask[truth]
                    row[f"accuracy_relevant_var{threshold}"] = (
                        float(correct[selected].mean()) if selected.any() else 0.0
                    )
                    row[f"n_relevant_var{threshold}"] = int(selected.sum())
                rows.append(row)
    return rows


# ----------------------------------------------------------------- aggregate


def aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fingerprint in FINGERPRINTS:
        for solver in SOLVERS:
            for n_samples in CHECKPOINTS:
                group = [
                    c
                    for c in cells
                    if c["fingerprint"] == fingerprint
                    and c["solver"] == solver
                    and c["n_samples"] == n_samples
                ]
                if not group:
                    continue
                entry: dict[str, Any] = {
                    "fingerprint": fingerprint,
                    "solver": solver,
                    "n_samples": n_samples,
                    "n_cells": len(group),
                    "accuracy_all_mean": float(np.mean([c["accuracy_all"] for c in group])),
                }
                for threshold in THRESHOLDS:
                    key = f"accuracy_relevant_var{threshold}"
                    values = [c[key] for c in group]
                    entry[f"{key}_mean"] = float(np.mean(values))
                    entry[f"{key}_min"] = float(np.min(values))
                out.append(entry)
    return out


def sample_floor(aggregates: list[dict[str, Any]], fingerprint: str, solver: str) -> Any:
    """Smallest N where mean relevant(var>0.01) accuracy crosses 0.90, interpolated."""
    points = sorted(
        [
            (a["n_samples"], a["accuracy_relevant_var0.01_mean"])
            for a in aggregates
            if a["fingerprint"] == fingerprint and a["solver"] == solver
        ]
    )
    for (n0, a0), (n1, a1) in zip(points, points[1:]):
        if a0 < 0.90 <= a1:
            if a1 == a0:
                return float(n1)
            return float(n0 + (0.90 - a0) * (n1 - n0) / (a1 - a0))
    if points and points[0][1] >= 0.90:
        return float(points[0][0])
    return "> 2000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--boundaries", type=int, nargs="+", default=list(BOUNDARIES))
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()

    started = time.time()
    x, y = load_mnist_train(args.data_home)
    config = IPMNISTConfig(n_tasks=max(args.boundaries) + 2, task_length=TASK_LENGTH)

    controls: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []

    for seed in args.seeds:
        schedule = build_schedule(jr.key(np.uint32(seed)), config, x.shape[0])
        for boundary in args.boundaries:
            params = train_reference_network(seed, boundary, x, y, schedule)

            if not args.skip_controls and seed == args.seeds[0] and boundary == args.boundaries[0]:
                controls["exact_statistic_oracle"] = run_cell(
                    seed, boundary, x, y, schedule, params, oracle=True
                )
                controls["no_shift"] = run_cell(
                    seed, boundary, x, y, schedule, params, no_shift=True
                )
            cells.extend(run_cell(seed, boundary, x, y, schedule, params))

    # Pre-registered control gates decide which fingerprints carry a result at
    # all: a family that misses the oracle bar is mis-implemented, not
    # uninformative, so its online numbers are recorded but marked void.
    verdict: dict[str, Any] = {}
    for fingerprint in FINGERPRINTS:
        oracle_rows = [
            r
            for r in controls.get("exact_statistic_oracle", [])
            if r["fingerprint"] == fingerprint and r["solver"] == "hungarian"
        ]
        shift_rows = [
            r
            for r in controls.get("no_shift", [])
            if r["fingerprint"] == fingerprint and r["solver"] == "hungarian"
        ]
        oracle_pass = bool(oracle_rows) and all(
            r["accuracy_relevant_var0.01"] >= 0.95 for r in oracle_rows
        )
        shift_pass = bool(shift_rows) and all(
            r["accuracy_relevant_var0.01"] >= 0.99 for r in shift_rows
        )
        verdict[fingerprint] = {
            "oracle_pass": oracle_pass,
            "no_shift_pass": shift_pass,
            "void": not (oracle_pass and shift_pass),
            "reason": None
            if (oracle_pass and shift_pass)
            else "failed pre-registered sanity control; mis-implemented, not uninformative",
        }

    aggregates = aggregate(cells)
    # A fingerprint that failed a preregistered correctness control has no
    # licensed online result.  Preserve its raw diagnostic rows, but do not
    # turn them into the protocol's secondary N* measurement.
    floors = {
        f"{fingerprint}/{solver}": (
            "void"
            if verdict[fingerprint]["void"]
            else sample_floor(aggregates, fingerprint, solver)
        )
        for fingerprint in FINGERPRINTS
        for solver in SOLVERS
    }

    best = max(
        (a for a in aggregates if a["n_samples"] <= 500 and not verdict[a["fingerprint"]]["void"]),
        key=lambda a: a["accuracy_relevant_var0.01_min"],
        default=None,
    )
    promoted = bool(best and best["accuracy_relevant_var0.01_min"] > 0.90)

    artifact = {
        "schema": SCHEMA,
        "cells": cells,
        "aggregates": aggregates,
        "controls": controls,
        "control_verdict": verdict,
        "deviations": [
            (
                "controls were computed after the first online cell in the shipped "
                "execution rather than before all online rows"
            ),
            (
                "the control implementation was changed from online-checkpoint to exact "
                "full-dataset statistics after the first runner draft"
            ),
            (
                "F3 reference correlation descriptors use batch statistics over the "
                "5000 pre-shift examples; the inherited EMA statistics supply the "
                "relevance mask, not the correlation descriptor"
            ),
            (
                "F4b uses batch gradient-activation correlation rather than the "
                "preregistered EMA and is void because its oracle gate failed"
            ),
        ],
        "protocol": {
            "data": "IPMNIST protocol MNIST train split (load_mnist_train)",
            "permutations": "build_schedule per-seed protocol permutations",
            "reference_relevance_statistics": (
                "annealed fast-EMA mean/variance (decay 0.99) over 5000 samples"
            ),
            "fingerprint_reference_statistics": (
                "batch descriptors over the 5000 pre-shift examples"
            ),
            "post_shift": "fresh statistics from stream start",
            "reference_arm": REFERENCE_ARM,
            "seeds": list(args.seeds),
            "boundaries": list(args.boundaries),
            "sample_checkpoints": list(CHECKPOINTS),
            "relevance_thresholds_on_reference_var": list(THRESHOLDS),
            "fingerprints": list(FINGERPRINTS),
            "solvers": list(SOLVERS),
            "topk": TOPK,
            "spectral_m": SPECTRAL_M,
        },
        "promotion": {
            "criterion": (
                ">90% of relevant (var > 0.01) pixels correctly assigned within "
                "<=500 samples (min over seed x boundary cells)"
            ),
            "promoted": promoted,
            "best_configuration": None
            if best is None
            else {
                "fingerprint": best["fingerprint"],
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
    print(f"wrote {args.out} in {artifact['wall_clock_seconds']}s; promoted={promoted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
