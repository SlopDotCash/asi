"""Maintained, nonpromoting IPMNIST campaign analysis tools.

Historical programs stored below ``outputs/`` are immutable run records.  This
module owns the live analysis contracts and takes every input path explicitly,
so reproduction never depends on a contributor's home directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.ipmnist_provenance import analysis_provenance

DEFAULT_BASE = "upgd_ema_norm_sigma0"
DEFAULT_ARMS = (
    "sigma0_hidden_norm",
    "sigma0_localgate",
    "sigma0_ndecay099",
    "sigma0_ndecay09999",
    "sigma0_eps1e6",
    "sigma0_eps1e4",
    "sigma0_gate_beta05",
    "sigma0_gate_beta2",
)
DEFAULT_THRESHOLD = 0.002
CONFIRM_ALIGNMENT_ATOL = 1e-7
LATE_LO, LATE_HI = 4000, 5000
BUCKETS = (
    (0, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3500),
    (3500, 5000),
)
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _safe_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe exact identifier")
    return value


def _finite_scalar(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite built-in number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite built-in number")
    return number


def _frontier_arms(value: object, *, base: str) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError("arms must be a non-empty exact list or tuple")
    raw = cast(list[object] | tuple[object, ...], value)
    if len(raw) == 0:
        raise ValueError("arms must be a non-empty exact list or tuple")
    arms = tuple(
        _safe_identifier(raw[index], name=f"arms[{index}]")
        for index in range(len(raw))
    )
    if len(set(arms)) != len(arms) or base in arms:
        raise ValueError("arms must be unique and distinct from base")
    return arms


def across_seed_spread(values: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Return sample standard deviation, or zero below two observations."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size < 2:
        return 0.0
    return float(array.std(ddof=1))


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is forbidden")
        result[key] = value
    return result


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    if type(payload) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def seed_means(root: Path, config_name: str) -> dict[int, float]:
    """Read mean per-task accuracy by seed for one campaign arm."""
    config_name = _safe_identifier(config_name, name="config_name")
    result: dict[int, float] = {}
    for path in sorted(root.glob(f"{config_name}_seed*.json")):
        payload = _json_object(path)
        raw_seed = payload.get("seed")
        accuracies = payload.get("per_task_accuracy")
        if type(accuracies) is not list or not accuracies:
            raise ValueError(f"{path} lacks a valid seed or per_task_accuracy")
        seed = require_jax_seed(raw_seed, name=f"{path} seed")
        if path.name != f"{config_name}_seed{seed}.json":
            raise ValueError(f"{path} filename disagrees with payload seed")
        values = np.asarray(accuracies, dtype=np.float64)
        if (
            values.ndim != 1
            or not bool(np.all(np.isfinite(values)))
            or not bool(np.all((0.0 <= values) & (values <= 1.0)))
        ):
            raise ValueError(f"{path} has invalid per_task_accuracy")
        if seed in result:
            raise ValueError(f"duplicate seed {seed} for {config_name}")
        result[seed] = statistics.mean(float(value) for value in values)
    return result


def build_frontier(
    screen_dir: Path,
    confirm_dir: Path,
    *,
    base: str = DEFAULT_BASE,
    arms: Sequence[str] = DEFAULT_ARMS,
    threshold: float = DEFAULT_THRESHOLD,
    created_unix: float | None = None,
) -> dict[str, Any]:
    """Build a paired-seed development frontier without mixing seed sets."""
    if screen_dir.resolve() == confirm_dir.resolve():
        raise ValueError("screen_dir and confirm_dir must resolve to distinct paths")
    base = _safe_identifier(base, name="base")
    checked_arms = _frontier_arms(arms, base=base)
    threshold = _finite_scalar(threshold, name="threshold")
    if created_unix is not None:
        created_unix = _finite_scalar(created_unix, name="created_unix")
        if created_unix < 0:
            raise ValueError("created_unix must be nonnegative")
    screen_base = seed_means(screen_dir, base)
    confirm_base = seed_means(confirm_dir, base)
    if not screen_base:
        raise ValueError(f"screen base {base!r} has no seeds")
    rows: list[dict[str, Any]] = []
    for arm in checked_arms:
        screen = seed_means(screen_dir, arm)
        confirm = seed_means(confirm_dir, arm)
        if screen.keys() != screen_base.keys():
            raise ValueError(
                f"screen seed sets differ for {arm!r} and base {base!r}: "
                f"{sorted(screen)} != {sorted(screen_base)}"
            )
        paired_seeds = sorted(screen)
        row: dict[str, Any] = {
            "config_name": arm,
            "n_screen_seeds": len(paired_seeds),
            "screen_mean": statistics.mean(screen[seed] for seed in paired_seeds),
            "screen_paired_delta_vs_base": statistics.mean(
                screen[seed] - screen_base[seed] for seed in paired_seeds
            ),
            "screen_per_seed_delta": [
                round(screen[seed] - screen_base[seed], 6) for seed in paired_seeds
            ],
        }
        row["confirmation_candidate"] = row["screen_paired_delta_vs_base"] > threshold
        if confirm:
            if confirm.keys() != confirm_base.keys():
                raise ValueError(
                    f"confirm seed sets differ for {arm!r} and base {base!r}: "
                    f"{sorted(confirm)} != {sorted(confirm_base)}"
                )
            confirm_seeds = sorted(confirm)
            row["confirm_mean"] = statistics.mean(
                confirm[seed] for seed in confirm_seeds
            )
            row["n_confirm_seeds"] = len(confirm_seeds)
            row["confirm_paired_delta_vs_base"] = statistics.mean(
                confirm[seed] - confirm_base[seed] for seed in confirm_seeds
            )
        rows.append(row)
    rows.sort(key=lambda row: -float(row.get("screen_mean", 0.0)))
    input_paths = [
        path
        for directory in (screen_dir, confirm_dir)
        for name in (base, *checked_arms)
        for path in directory.glob(f"{name}_seed*.json")
    ]
    return {
        "schema": "asi.ipmnist.frontier.v2",
        "base": base,
        "base_screen_mean": statistics.mean(screen_base.values()) if screen_base else None,
        "base_confirm_mean": (
            statistics.mean(confirm_base.values()) if confirm_base else None
        ),
        "confirmation_threshold": threshold,
        "created_unix": time.time() if created_unix is None else created_unix,
        "evidence_policy": {
            "development_only": True,
            "evidence_class": "development_screening_diagnostic",
            "scientific_promotion_allowed": False,
        },
        "provenance": analysis_provenance(
            command="frontier",
            input_paths=input_paths,
            sources={"campaign_tools": Path(__file__)},
            repository_root=Path(__file__).resolve().parents[2],
        ),
        "results": rows,
    }


def _ceiling_runs(
    ceiling_dir: Path,
    prefix: str,
    *,
    input_paths: list[Path] | None = None,
) -> dict[int, dict[str, Any]]:
    prefix = _safe_identifier(prefix, name="prefix")
    runs: dict[int, dict[str, Any]] = {}
    for path in sorted(ceiling_dir.glob(f"{prefix}_seed*.json")):
        payload = _json_object(path)
        raw_seed = payload.get("seed")
        tag = payload.get("tag")
        seed = require_jax_seed(raw_seed, name=f"{path} seed")
        tag = _safe_identifier(tag, name=f"{path} tag")
        if tag != prefix:
            raise ValueError(f"{path} tag disagrees with requested prefix")
        if seed in runs:
            raise ValueError(f"duplicate ceiling seed {seed} for {prefix}")
        if path.name != f"{prefix}_seed{seed}.json":
            raise ValueError(f"{path} filename disagrees with payload seed")
        per_step_path = ceiling_dir / f"{tag}_seed{seed}_per_step.npy"
        payload["per_step"] = _validated_ceiling_run(
            payload, np.load(per_step_path, allow_pickle=False), path=path
        )
        if input_paths is not None:
            input_paths.extend((path, per_step_path))
        runs[seed] = payload
    for path in sorted(ceiling_dir.glob(f"{prefix}_seed*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            payload = json.loads(str(archive["metadata"].item()))
            per_step = np.asarray(archive["per_step"])
        if type(payload) is not dict:
            raise ValueError(f"{path} metadata must be a JSON object")
        if payload.get("schema") != "asi.ipmnist_ceiling.run.v2":
            raise ValueError(f"{path} has an unsupported maintained run schema")
        if (
            payload.get("evidence_class")
            != "development_screening_diagnostic"
            or payload.get("development_only") is not True
            or payload.get("scientific_promotion_allowed") is not False
        ):
            raise ValueError(f"{path} violates the permanently nonpromoting policy")
        wall_seconds = _finite_scalar(
            payload.get("wall_clock_seconds"), name=f"{path} wall_clock_seconds"
        )
        if wall_seconds < 0:
            raise ValueError(f"{path} wall_clock_seconds must be nonnegative")
        provenance = payload.get("provenance")
        if (
            type(provenance) is not dict
            or provenance.get("schema") != "asi.ipmnist.ceiling_run_provenance.v1"
        ):
            raise ValueError(f"{path} lacks maintained run provenance")
        seed = require_jax_seed(payload.get("seed"), name=f"{path} seed")
        if seed in runs:
            raise ValueError(f"duplicate ceiling seed {seed} for {prefix}")
        if path.name != f"{prefix}_seed{seed}.npz":
            raise ValueError(f"{path} filename disagrees with payload seed")
        tag = _safe_identifier(payload.get("tag"), name=f"{path} tag")
        if tag != prefix:
            raise ValueError(f"{path} tag disagrees with requested prefix")
        payload["per_step"] = _validated_ceiling_run(payload, per_step, path=path)
        if input_paths is not None:
            input_paths.append(path)
        runs[seed] = payload
    return runs


def _validated_ceiling_run(
    payload: dict[str, Any], per_step: object, *, path: Path
) -> np.ndarray[Any, Any]:
    n_tasks = payload.get("n_tasks")
    task_length = payload.get("task_length")
    if type(n_tasks) is not int or n_tasks <= 0:
        raise ValueError(f"{path} has invalid n_tasks")
    if type(task_length) is not int or task_length != 5_000:
        raise ValueError(f"{path} task_length must equal 5000")
    tag = _safe_identifier(payload.get("tag"), name=f"{path} tag")
    spec_name = _safe_identifier(payload.get("spec_name"), name=f"{path} spec_name")
    permutation_mode = payload.get("perm_mode")
    if type(permutation_mode) is not str:
        raise ValueError(f"{path} perm_mode must be an exact string")
    if tag == f"stationary_{spec_name}":
        expected_mode, expected_tasks = "identity", 1
    elif tag == f"carried_{spec_name}":
        expected_mode, expected_tasks = "same", None
    elif tag == f"full_{spec_name}":
        expected_mode, expected_tasks = "protocol", 200
    else:
        raise ValueError(f"{path} tag does not bind its spec_name")
    if permutation_mode != expected_mode:
        raise ValueError(f"{path} perm_mode disagrees with tag family")
    if expected_tasks is not None and n_tasks != expected_tasks:
        raise ValueError(f"{path} n_tasks disagrees with tag family")
    if expected_tasks is None and n_tasks < 10:
        raise ValueError(f"{path} carried runs require at least ten tasks")
    array = np.asarray(per_step)
    if array.shape != (n_tasks, task_length):
        raise ValueError(f"{path} per_step shape disagrees with metadata")
    if array.dtype != np.dtype(np.uint8) or not bool(np.all((array == 0) | (array == 1))):
        raise ValueError(f"{path} per_step must be a uint8 binary matrix")
    raw_per_task = payload.get("per_task_accuracy")
    if type(raw_per_task) is not list:
        raise ValueError(f"{path} per_task_accuracy must be an exact list")
    per_task = _finite_vector(raw_per_task, name="per_task_accuracy")
    if per_task.shape != (n_tasks,) or not bool(np.all((0.0 <= per_task) & (per_task <= 1.0))):
        raise ValueError(f"{path} per_task_accuracy has invalid shape or domain")
    observed_means = array.astype(np.float64).mean(axis=1)
    if not bool(
        np.allclose(
            per_task,
            observed_means,
            rtol=0.0,
            atol=CONFIRM_ALIGNMENT_ATOL,
            equal_nan=False,
        )
    ):
        raise ValueError(f"{path} per_task_accuracy disagrees with per_step means")
    mean_accuracy = _finite_scalar(payload.get("mean_accuracy"), name="mean_accuracy")
    if not 0.0 <= mean_accuracy <= 1.0 or not math.isclose(
        mean_accuracy,
        float(per_task.mean()),
        rel_tol=0.0,
        abs_tol=CONFIRM_ALIGNMENT_ATOL,
    ):
        raise ValueError(f"{path} mean_accuracy disagrees with per_task_accuracy")
    return array


def _finite_vector(value: object, *, name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a finite vector")
    return array


def validate_confirm_alignment(
    runs: Sequence[dict[str, Any]],
    confirm_dir: Path,
    *,
    input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Validate rounded ceiling JSON against confirmation shards.

    Both files store decimal JSON derived from independently rounded floating
    computations.  The fixed absolute tolerance admits the observed encoding
    delta while rejecting a score-relevant disagreement; relative tolerance is
    deliberately zero so the gate does not widen with accuracy magnitude.
    """
    per_seed: dict[str, Any] = {}
    maximum = 0.0
    for run in runs:
        seed = cast(int, run["seed"])
        reference_path = confirm_dir / f"sigma0_ndecay099_seed{seed}.json"
        reference = _json_object(reference_path)
        reference_seed = require_jax_seed(
            reference.get("seed"), name=f"{reference_path} seed"
        )
        if reference_seed != seed:
            raise ValueError(f"{reference_path} seed does not match ceiling seed {seed}")
        if input_paths is not None:
            input_paths.append(reference_path)
        observed = _finite_vector(run["per_task_accuracy"], name="per_task_accuracy")
        expected = _finite_vector(reference["per_task_accuracy"], name="reference")
        if observed.shape != expected.shape:
            raise ValueError(f"ceiling and confirm shapes differ for seed {seed}")
        max_delta = float(np.max(np.abs(observed - expected), initial=0.0))
        if not bool(
            np.allclose(
                observed,
                expected,
                rtol=0.0,
                atol=CONFIRM_ALIGNMENT_ATOL,
                equal_nan=False,
            )
        ):
            raise ValueError(
                f"ceiling and confirm per-task values differ for seed {seed}: "
                f"max_abs_delta={max_delta:.17g} exceeds atol={CONFIRM_ALIGNMENT_ATOL}"
            )
        maximum = max(maximum, max_delta)
        per_seed[str(seed)] = {
            "n_values": int(observed.size),
            "max_abs_delta": max_delta,
        }
    return {
        "absolute_tolerance": CONFIRM_ALIGNMENT_ATOL,
        "relative_tolerance": 0.0,
        "max_abs_delta": maximum,
        "per_seed": per_seed,
    }


def build_ceiling_summary(ceiling_dir: Path, confirm_dir: Path) -> dict[str, Any]:
    """Recompute the maintained ceiling/error-budget summary from raw runs."""
    input_paths: list[Path] = []
    report: dict[str, Any] = {
        "schema": "asi.ipmnist.ceiling_summary.v2",
        "evidence_policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
        }
    }
    for spec in ("sigma0_ndecay099", "adamw_control", "sgd_ema_norm"):
        runs = _ceiling_runs(
            ceiling_dir, f"stationary_{spec}", input_paths=input_paths
        )
        if not runs:
            continue
        means = [float(run["mean_accuracy"]) for run in runs.values()]
        curves = np.stack(
            [np.asarray(run["per_step"])[0].astype(np.float64) for run in runs.values()]
        )
        mean_curve = curves.mean(axis=0)
        report[f"stationary_{spec}"] = {
            "avg_online_mean": float(np.mean(means)),
            "avg_online_per_seed": means,
            "sample_standard_deviation": across_seed_spread(means),
            "late_window": float(mean_curve[LATE_LO:LATE_HI].mean()),
            "buckets": {
                f"{start}-{end}": round(float(mean_curve[start:end].mean()), 5)
                for start, end in BUCKETS
            },
        }

    for spec in ("sigma0_ndecay099", "adamw_control"):
        runs = _ceiling_runs(ceiling_dir, f"carried_{spec}", input_paths=input_paths)
        if not runs:
            continue
        per_task = np.stack(
            [
                _finite_vector(run["per_task_accuracy"], name="per_task_accuracy")
                for run in runs.values()
            ]
        )
        late_tasks = per_task[:, -10:].mean(axis=1)
        last_task_late = np.stack(
            [
                np.asarray(run["per_step"], dtype=np.float64)[-10:, LATE_LO:LATE_HI].mean()
                for run in runs.values()
            ]
        )
        report[f"carried_{spec}"] = {
            "per_task_mean_curve": [round(float(value), 5) for value in per_task.mean(axis=0)],
            "late_task_avg_online": float(late_tasks.mean()),
            "late_task_sample_standard_deviation": across_seed_spread(late_tasks),
            "late_task_late_window": float(last_task_late.mean()),
        }

    batch_paths = sorted(ceiling_dir.glob("batch_reference_seed*.json"))
    batch = []
    batch_seeds: set[int] = set()
    for path in batch_paths:
        run = _json_object(path)
        seed = require_jax_seed(run.get("seed"), name=f"{path} seed")
        if seed in batch_seeds:
            raise ValueError(f"{path} has a duplicate seed")
        if path.name != f"batch_reference_seed{seed}.json":
            raise ValueError(f"{path} filename disagrees with payload seed")
        batch_seeds.add(seed)
        best_score = _finite_scalar(
            run.get("test_accuracy_best"), name="test_accuracy_best"
        )
        if not 0.0 <= best_score <= 1.0:
            raise ValueError(f"{path} test_accuracy_best is outside [0, 1]")
        batch.append(run)
    input_paths.extend(batch_paths)
    if batch:
        best_scores = [float(run["test_accuracy_best"]) for run in batch]
        report["batch_reference"] = {
            "test_best_mean": float(np.mean(best_scores)),
            "test_best_sample_standard_deviation": across_seed_spread(best_scores),
            "per_seed": best_scores,
        }

    full = _ceiling_runs(
        ceiling_dir, "full_sigma0_ndecay099", input_paths=input_paths
    )
    if full:
        ordered = [full[seed] for seed in sorted(full)]
        alignment = validate_confirm_alignment(
            ordered, confirm_dir, input_paths=input_paths
        )
        array = np.stack(
            [np.asarray(run["per_step"], dtype=np.float64) for run in ordered]
        )
        overall = float(array.mean())
        plateau_by_task = array[:, :, LATE_LO:LATE_HI].mean(axis=2)
        asymptotic_error = float((1.0 - plateau_by_task).mean())
        total_error = 1.0 - overall
        curve = array.mean(axis=(0, 1))
        bucket_rows = []
        for start, end in BUCKETS:
            accuracy = float(curve[start:end].mean())
            excess = float(
                (plateau_by_task[:, :, None] - array[:, :, start:end]).mean()
            )
            contribution = excess * (end - start) / array.shape[2]
            bucket_rows.append(
                [
                    f"{start}-{end}",
                    round(accuracy, 5),
                    round(excess, 5),
                    round(contribution, 5),
                ]
            )
        task_plateau = plateau_by_task.mean(axis=0)
        task_index = np.arange(task_plateau.size)
        late_mask = task_index >= 20
        slope = float(np.polyfit(task_index[late_mask], task_plateau[late_mask], 1)[0])
        report["error_budget"] = {
            "overall": overall,
            "total_error": total_error,
            "asymptotic_error": asymptotic_error,
            "transient_excess": total_error - asymptotic_error,
            "plateau_mean": float(plateau_by_task.mean()),
            "first500": float(array[:, :, :500].mean()),
            "buckets": bucket_rows,
            "plateau_tasks_20_60": float(task_plateau[20:60].mean()),
            "plateau_tasks_160_200": float(task_plateau[160:200].mean()),
            "drift_slope_per_task": slope,
            "confirm_alignment": alignment,
        }
    report["provenance"] = analysis_provenance(
        command="ceiling",
        input_paths=input_paths,
        sources={"campaign_tools": Path(__file__)},
        repository_root=Path(__file__).resolve().parents[2],
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    frontier = subparsers.add_parser("frontier", help="build a paired-seed frontier")
    frontier.add_argument("--screen-dir", type=Path, required=True)
    frontier.add_argument("--confirm-dir", type=Path, required=True)
    frontier.add_argument("--base", default=DEFAULT_BASE)
    frontier.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    frontier.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ceiling = subparsers.add_parser("ceiling", help="recompute the ceiling summary")
    ceiling.add_argument("--ceiling-dir", type=Path, required=True)
    ceiling.add_argument("--confirm-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a maintained analyzer and emit JSON to stdout only."""
    args = _parser().parse_args(argv)
    if args.command == "frontier":
        result = build_frontier(
            args.screen_dir,
            args.confirm_dir,
            base=args.base,
            arms=args.arms,
            threshold=args.threshold,
        )
    else:
        result = build_ceiling_summary(args.ceiling_dir, args.confirm_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
