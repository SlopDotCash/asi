"""V8 — partial identification DELIVERED LATE, as a real identifier would.

Development diagnostic, permanently nonpromoting. Preregistration:
``V7_partial_remap.md``.

V7 granted identification instantly at step 0 and measured +0.056 at p=0.79,
refuting its own prediction that partial identification is worthless. This
runner removes the free lunch: each task runs the TRUE permutation for the
first ``onset`` steps -- the learner pays the ordinary transient exactly as it
normally would -- and only then switches to the partially-corrected layout.

Onsets are paired with the accuracy V1 actually measured at that sample budget
(N=200 -> 0.619, N=500 -> 0.785, N=2000 -> 0.840), so each arm answers: what
would an identifier that is honest about its own sample cost deliver?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_WT = os.environ.get("EXPECT_WT")
if _WT:
    sys.path.insert(0, _WT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import alberta_framework as _af  # noqa: E402

if _WT:
    assert _af.__file__.startswith(_WT), (_af.__file__, _WT)

from alberta_framework.benchmarks.ipmnist_screening import (  # noqa: E402
    SCREENING_REGISTRY,
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)

RELEVANT_VAR_THRESHOLD = 0.01
LEARNER = "rls_head_resid_l1_preset005"


def partial_permutation(
    pi_0: np.ndarray, relevant: np.ndarray, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Layout with a fraction ``p`` of relevant pixels restored to task 0.

    ``x[j] = base[perm[j]]``, so a perfect oracle is ``perm = pi_0``. A random
    fraction ``p`` of the *relevant* positions take their task-0 base pixel;
    the remaining relevant positions receive a random assignment of the
    relevant base pixels left over, so the result is always a valid
    permutation.

    Every position not chosen -- remaining relevant positions AND the
    background positions -- receives a random assignment from the base pixels
    left over. This matters: at ``p = 0`` the result is then a uniform random
    permutation over all ``input_dim`` positions, which is exactly the
    distribution the true protocol draws, so the ``p000`` arm is a valid null
    for the control.

    An earlier revision pinned background positions to ``pi_0`` so that
    ``p = 1`` would equal ``pi_0`` bitwise. That made ``p000`` a
    background-pinned stream rather than the protocol's own distribution, and
    the preregistered null check caught it (``p000`` came in 0.0142 BELOW
    control). The ``p = 1`` anchor is therefore checked empirically against the
    carried-oracle regime instead of by assertion: background pixels are
    near-constant, so scrambling them among themselves is functionally inert.
    """
    input_dim = pi_0.shape[0]
    relevant_positions = np.flatnonzero(relevant[pi_0])
    n_correct = int(round(p * relevant_positions.size))
    chosen = rng.choice(relevant_positions, size=n_correct, replace=False)
    perm = np.full(input_dim, -1, dtype=np.int64)
    perm[chosen] = pi_0[chosen]
    remaining_positions = np.flatnonzero(perm < 0)
    remaining_pixels = np.setdiff1d(
        np.arange(input_dim, dtype=np.int64), pi_0[chosen], assume_unique=False
    )
    perm[remaining_positions] = rng.permutation(remaining_pixels)
    assert np.array_equal(np.sort(perm), np.arange(input_dim)), "not a permutation"
    correct = int((perm[relevant_positions] == pi_0[relevant_positions]).sum())
    assert correct >= n_correct, (correct, n_correct)
    return perm


def run_arm(
    data_x: jnp.ndarray,
    data_y: jnp.ndarray,
    config: IPMNISTConfig,
    seed: int,
    permutations: np.ndarray,
    example_indices: jnp.ndarray,
    early_permutations: np.ndarray | None = None,
    onset: int = 0,
) -> dict[str, object]:
    """``run_screening_config``'s loop, optionally switching layout mid-task.

    With ``onset > 0`` the first ``onset`` steps of each task use
    ``early_permutations`` (the true schedule) and the remainder uses
    ``permutations`` (the oracle's partially-corrected layout). The switch is
    a build-time split into two scans, so neither segment carries a per-step
    branch.
    """
    spec = SCREENING_REGISTRY[LEARNER]
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    root = jr.key(jnp.uint32(seed), impl="threefry2x32")
    key_init, _key_schedule, key_noise = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    state = init_fn(params)

    def run_task(params, state, key, permutation, examples):
        def one_step(carry, example):
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            key, step_key = jr.split(key)
            new_params, new_state, metrics = step_fn(
                step_params, step_state, x, y, step_key
            )
            return (new_params, new_state, key), metrics

        (params, state, key), (acc, loss, plast) = jax.lax.scan(
            one_step, (params, state, key), examples
        )
        return params, state, key, acc, loss, plast

    def run_task_split(params, state, key, perm_early, perm_late, examples):
        params, state, key, acc_a, loss_a, plast_a = run_task(
            params, state, key, perm_early, examples[:onset]
        )
        params, state, key, acc_b, loss_b, plast_b = run_task(
            params, state, key, perm_late, examples[onset:]
        )
        return (
            params, state, key,
            jnp.concatenate([acc_a, acc_b]),
            jnp.concatenate([loss_a, loss_b]),
            jnp.concatenate([plast_a, plast_b]),
        )

    run_task_jit = jax.jit(run_task)
    run_split_jit = jax.jit(run_task_split)
    perms = jnp.asarray(permutations, dtype=jnp.int32)
    perms_early = (
        jnp.asarray(early_permutations, dtype=jnp.int32)
        if early_permutations is not None
        else None
    )
    per_task, per_step_sum = [], np.zeros(config.task_length, dtype=np.float64)
    plast_all = []
    started = time.monotonic()
    for task in range(config.n_tasks):
        if onset > 0:
            assert perms_early is not None
            params, state, key_noise, acc, _loss, plast = run_split_jit(
                params, state, key_noise, perms_early[task], perms[task],
                example_indices[task],
            )
        else:
            params, state, key_noise, acc, _loss, plast = run_task_jit(
                params, state, key_noise, perms[task], example_indices[task]
            )
        acc_np = np.asarray(acc, dtype=np.float64)
        per_task.append(float(acc_np.mean()))
        per_step_sum += acc_np
        plast_all.append(float(np.asarray(plast, dtype=np.float64).mean()))
    return {
        "average_online_accuracy": float(np.mean(per_task)),
        "per_task_accuracy": per_task,
        "mean_within_task_curve": (per_step_sum / config.n_tasks).tolist(),
        "average_plasticity": float(np.mean(plast_all)),
        "wall_clock_seconds": time.monotonic() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--task-length", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--onsets", type=str, nargs="+",
        default=["200:0.619", "500:0.785", "2000:0.840"],
        help="onset_step:identification_accuracy pairs from V1's measurements",
    )
    ap.add_argument("--data-home", type=Path, default=None)
    args = ap.parse_args()

    config = IPMNISTConfig(n_tasks=args.n_tasks, task_length=args.task_length)
    data_home = args.data_home or default_openml_data_home()
    raw_x, raw_y = load_mnist_train(data_home)
    data_x = jnp.asarray(raw_x, dtype=jnp.float32)
    data_y = jnp.asarray(raw_y, dtype=jnp.int32)
    n_train = int(data_x.shape[0])

    variance = np.asarray(jnp.var(data_x, axis=0), dtype=np.float64)
    relevant = variance > RELEVANT_VAR_THRESHOLD
    print(f"relevant pixels: {int(relevant.sum())}/{relevant.size}", flush=True)

    results: list[dict[str, object]] = []
    for seed in args.seeds:
        root = jr.key(jnp.uint32(seed), impl="threefry2x32")
        _key_init, key_schedule, _key_noise = jr.split(root, 3)
        schedule = build_schedule(key_schedule, config, n_train)
        true_perms = np.asarray(schedule.permutations, dtype=np.int64)
        pi_0 = true_perms[0]

        specs: list[tuple[str, np.ndarray, int]] = [
            ("control", true_perms, 0)
        ]
        for pair in args.onsets:
            onset_str, acc_str = pair.split(":")
            onset, p = int(onset_str), float(acc_str)
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(round(p * 1000)), onset])
            )
            perms = true_perms.copy()
            for t in range(1, config.n_tasks):
                perms[t] = partial_permutation(pi_0, relevant, p, rng)
            specs.append((f"N{onset}_p{int(round(p * 100)):03d}", perms, onset))

        for name, perms, onset in specs:
            out = run_arm(
                data_x, data_y, config, seed, perms, schedule.example_indices,
                early_permutations=true_perms, onset=onset,
            )
            out.update({"arm": name, "seed": seed, "onset": onset})
            results.append(out)
            print(
                f"seed {seed} {name:8s} avg={out['average_online_accuracy']:.6f} "
                f"plast={out['average_plasticity']:.5f} "
                f"({out['wall_clock_seconds']:.0f}s)",
                flush=True,
            )

    payload = {
        "schema": "alberta.new_directions.V8_delayed_remap.v1",
        "evidence_class": "development_diagnostic",
        "scientific_promotion_allowed": False,
        "learner": LEARNER,
        "relevant_var_threshold": RELEVANT_VAR_THRESHOLD,
        "n_relevant_pixels": int(relevant.sum()),
        "protocol": {"n_tasks": args.n_tasks, "task_length": args.task_length},
        "seeds": args.seeds,
        "onsets": args.onsets,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
