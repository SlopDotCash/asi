#!/usr/bin/env python3
"""In-container, no-reward preflight for frozen Foragax development screens.

This file is mounted read-only into the pinned development image.  It executes
each frozen entrypoint only through argparse's ``--help`` path, before experiment
or environment construction, and never executes an environment transition.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, TypeGuard, cast

_SOURCE_ROOT = Path("/opt/foragax-agents")
_PROTOCOL_ROOT = Path("/protocol")
_BASE_PROTOCOL_ROOT = Path("/protocol-input")
_PREDECESSOR_PROTOCOL_ROOT = Path("/predecessor-protocol")
_SCORER_PATH = Path("/harness/scorer.py")
_REFERENCE_SCORER_PATH = Path("/harness/reference_scorer.py")
_SUPPORTED_SCHEMAS = {
    "alberta.forager_fov_baseline_screening.v1",
    "alberta.forager_fov_stateful_baseline_screening.v1",
}
_CPU_SCHEMAS = {
    "alberta.forager_fov_baseline_screening_cpu.v3": (
        "alberta.forager_fov_baseline_screening.v1"
    ),
    "alberta.forager_fov_stateful_baseline_screening_cpu.v3": (
        "alberta.forager_fov_stateful_baseline_screening.v1"
    ),
}
_PREDECESSOR_SCHEMAS = {
    "alberta.forager_fov_baseline_screening_cpu.v3": (
        "alberta.forager_fov_baseline_screening_cpu.v2"
    ),
    "alberta.forager_fov_stateful_baseline_screening_cpu.v3": (
        "alberta.forager_fov_stateful_baseline_screening_cpu.v2"
    ),
}
_V3_PREFLIGHT_CONTRACT = {
    "mode": "exact_entrypoint_help_before_experiment_construction",
    "arguments": ["--help"],
    "entrypoints_from_frozen_configurations": True,
    "environment_or_transition_construction_allowed": False,
    "required_cache_directories": {
        "MPLCONFIGDIR": "/tmp/alberta-matplotlib-cache",
        "NUMBA_CACHE_DIR": "/tmp/alberta-numba-cache",
    },
}
_FROZEN_PROTOCOL_SHA256 = {
    "alberta.forager_fov_baseline_screening.v1": (
        "b94c906f9d3c1f2abf226d7049bd9305700d7f5a81012b07b36cc10d458d3174"
    ),
    "alberta.forager_fov_stateful_baseline_screening.v1": (
        "df80dfafbd3bfc8eca7f1e6eccadc0ba93da4a3ec8f12ebe2befd430a21915e8"
    ),
    "alberta.forager_fov_baseline_screening_cpu.v2": (
        "d384a44dcf8161e8d7c521ea3fda7720118cff93047bbb1261d84f158388e606"
    ),
    "alberta.forager_fov_stateful_baseline_screening_cpu.v2": (
        "83bb51ae792e90d27fb75f0215dc5c07a6a064468305f141508fe3ca57b13731"
    ),
    "alberta.forager_fov_baseline_screening_cpu.v3": (
        "e5a0f0fbe3fc9cd7245abe01a6a177eea030b7b533e6d85992b88b1b91c11dd0"
    ),
    "alberta.forager_fov_stateful_baseline_screening_cpu.v3": (
        "a7cbca5735341ad580f09d705116f7131633b9cbe2494ef3bdc2d5ab6073c34d"
    ),
}


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_exact_int(value: Any) -> TypeGuard[int]:
    """True only for exact ints; bool is an int subclass and must not alias 0/1."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_task_intake(protocol: dict[str, Any]) -> tuple[dict[str, Any], int, list[int]]:
    task = protocol.get("task")
    if not isinstance(task, dict):
        _fail("protocol task must be an object")
    horizon = task.get("steps_per_seed", task.get("steps"))
    seeds = task.get("seeds")
    if not _is_exact_int(horizon) or horizon <= 0:
        _fail("protocol horizon is invalid")
    if not isinstance(seeds, list) or not seeds or not all(_is_exact_int(v) for v in seeds):
        _fail("protocol seeds are invalid")
    return task, horizon, cast(list[int], seeds)


def _validate_ppo_schedule(
    rollout: Any,
    updates: Any,
    horizon: int,
    relative: str,
) -> tuple[int, int]:
    if not _is_exact_int(rollout) or not _is_exact_int(updates):
        _fail(f"PPO schedule is not explicit: {relative}")
    if rollout * updates != horizon:
        _fail(f"PPO schedule does not equal the frozen horizon: {relative}")
    return rollout, updates


def _relative_path(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        _fail(f"{label} must be a normalized relative path")
    return value


def _source_records(protocol: dict[str, Any]) -> list[dict[str, str]]:
    raw = protocol.get("source_files")
    if raw is None:
        implementation = protocol.get("implementation")
        if not isinstance(implementation, dict):
            _fail("protocol has no implementation object")
        raw = implementation.get("source_files")
    if not isinstance(raw, list) or not raw:
        _fail("protocol source file list must be non-empty")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _fail(f"source_files[{index}] must be an object")
        relative = _relative_path(item.get("path"), f"source_files[{index}].path")
        expected = item.get("sha256")
        if type(expected) is not str or len(expected) != 64:
            _fail(f"source_files[{index}].sha256 is invalid")
        if relative in seen:
            _fail(f"duplicate source path: {relative}")
        seen.add(relative)
        unresolved_source = _SOURCE_ROOT / relative
        if unresolved_source.is_symlink():
            _fail(f"source is not a regular non-symlink file: {relative}")
        source = unresolved_source.resolve(strict=True)
        if not source.is_file():
            _fail(f"source is not a regular non-symlink file: {relative}")
        actual = _sha256(source)
        if actual != expected:
            _fail(f"source hash drift for {relative}: expected {expected}, got {actual}")
        records.append({"path": relative, "sha256": actual})
    return records


def _root_is_read_only() -> bool:
    with Path("/proc/self/mountinfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            left = line.rstrip("\n").split(" - ", maxsplit=1)[0].split()
            if len(left) >= 6 and left[4] == "/":
                return "ro" in left[5].split(",")
    return False


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        _fail(f"configuration must be an object: {path}")
    return cast(dict[str, Any], value)


def _canonical_sqlite(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        schema = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        columns = connection.execute('PRAGMA table_info("_metadata_")').fetchall()
        names = [str(row[1]) for row in columns]
        quoted = ", ".join('"' + name.replace('"', '""') + '"' for name in names)
        rows = connection.execute(
            f'SELECT {quoted} FROM "_metadata_" ORDER BY "id", "seed"'
        ).fetchall()

        def tagged(value: Any) -> dict[str, Any]:
            if value is None:
                return {"type": "null", "value": None}
            if type(value) is int:
                return {"type": "integer", "value": value}
            if type(value) is float:
                return {"type": "real", "value": value.hex()}
            if type(value) is str:
                return {"type": "text", "value": value}
            if type(value) is bytes:
                return {"type": "blob", "value": value.hex()}
            _fail("metadata database contains an unsupported SQLite value")

        return {
            "schema": [list(row) for row in schema],
            "columns": [list(row) for row in columns],
            "rows": [[tagged(value) for value in row] for row in rows],
        }
    finally:
        connection.close()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load scorer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_scorer_equivalence(
    schema: str,
    scoring: dict[str, Any],
) -> dict[str, Any]:
    if schema == "alberta.forager_fov_baseline_screening_cpu.v3":
        if "reference_implementation_sha256" in scoring:
            _fail("baseline CPU protocol unexpectedly declares a reference scorer")
        return {"status": "not_applicable", "reason": "base protocol has no bound scorer"}

    expected_reference = scoring.get("reference_implementation_sha256")
    if (
        type(expected_reference) is not str
        or _sha256(_REFERENCE_SCORER_PATH) != expected_reference
    ):
        _fail("reference scorer identity drift")
    import numpy as np

    candidate = _load_module(_SCORER_PATH, "frozen_cpu_scorer")
    reference = _load_module(_REFERENCE_SCORER_PATH, "frozen_v1_reference_scorer")
    horizon = int(reference.HORIZON)
    indices = np.arange(horizon)
    cases = [
        np.ones((horizon,), dtype=np.float32),
        np.where(indices % 7 == 0, 1.5, -0.25).astype(np.float32),
    ]
    exact_fields = (
        "reward_dtype",
        "reward_shape",
        "reward_trace_sha256",
        "ema_sample_count",
        "ema_tail_start_index",
        "ema_tail_sample_count",
    )
    float_fields = (
        "reward_sum_float64",
        "fov_last_10pct_ema_auc",
        "final_unadjusted_ema",
    )
    reports: list[dict[str, Any]] = []
    for index, rewards in enumerate(cases):
        archive = Path("/tmp") / f"scorer-equivalence-{index}.npz"
        archive.unlink(missing_ok=True)
        np.savez_compressed(archive, rewards=rewards)
        expected = reference.score(archive)
        actual = candidate.score_rewards(rewards, horizon)
        if any(actual[field] != expected[field] for field in exact_fields) or any(
            float(actual[field]).hex() != float(expected[field]).hex()
            for field in float_fields
        ):
            _fail(f"pinned-image scorer is not bitwise equivalent for case {index}")
        reports.append(
            {
                "case": index,
                "reward_trace_sha256": actual["reward_trace_sha256"],
                "score_float64_hex": float(actual["fov_last_10pct_ema_auc"]).hex(),
                "final_ema_float64_hex": float(actual["final_unadjusted_ema"]).hex(),
            }
        )
        archive.unlink()
    return {
        "status": "passed",
        "reference_scorer_sha256": expected_reference,
        "cases": reports,
    }


def _validate_configurations(
    protocol: dict[str, Any],
    schema: str,
) -> list[dict[str, Any]]:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))
    from experiment import ExperimentModel  # type: ignore[import-not-found]
    from ml_instrumentation.metadata import attach_metadata
    from PyExpUtils.results.tools import getParamsAsDict

    task, horizon, typed_seeds = _validate_task_intake(protocol)

    raw_configs = protocol.get("configurations")
    if not isinstance(raw_configs, list) or not raw_configs:
        _fail("protocol configurations must be a non-empty list")
    checked: list[dict[str, Any]] = []
    for index, item in enumerate(raw_configs):
        if not isinstance(item, dict):
            _fail(f"configurations[{index}] must be an object")
        relative = _relative_path(item.get("path"), f"configurations[{index}].path")
        expected_hash = item.get("sha256")
        unresolved_config_path = _BASE_PROTOCOL_ROOT / relative
        if unresolved_config_path.is_symlink():
            _fail(f"configuration is not a regular non-symlink file: {relative}")
        config_path = unresolved_config_path.resolve(strict=True)
        if not config_path.is_file():
            _fail(f"configuration is not a regular non-symlink file: {relative}")
        actual_hash = _sha256(config_path)
        if actual_hash != expected_hash:
            _fail(f"configuration hash drift for {relative}")
        config = _load_config(config_path)
        experiment = ExperimentModel.load(str(config_path))
        if experiment.numPermutations() != 1:
            _fail(f"configuration must resolve exactly one permutation: {relative}")
        stored = [experiment.getRun(seed) for seed in typed_seeds]
        if stored != typed_seeds:
            _fail(f"run indices do not resolve to the frozen stored seeds: {relative}")
        hypers = experiment.get_hypers(typed_seeds[0])
        if any(experiment.get_hypers(seed) != hypers for seed in typed_seeds[1:]):
            _fail(f"frozen seeds resolve different hyperparameters: {relative}")
        if config.get("total_steps") != horizon or experiment.total_steps != horizon:
            _fail(f"configuration horizon drift: {relative}")
        if config.get("problem") != "Foragax":
            _fail(f"configuration problem drift: {relative}")
        environment = hypers.get("environment")
        expected_environment = {
            "env_id": task.get("env_id"),
            "aperture_size": task.get("aperture_size"),
        }
        if schema == "alberta.forager_fov_stateful_baseline_screening.v1":
            expected_environment["observation_type"] = task.get("observation_type")
        if environment != expected_environment:
            _fail(f"configuration environment drift: {relative}")
        nested_experiment = hypers.get("experiment", {})
        if not isinstance(nested_experiment, dict) or nested_experiment.get("seed_offset", 0) != 0:
            _fail(f"configuration seed offset drift: {relative}")
        agent = config.get("agent")
        if type(agent) is not str or not agent:
            _fail(f"configuration agent is invalid: {relative}")
        if agent.startswith("PPO"):
            rollout, updates = _validate_ppo_schedule(
                hypers.get("rollout_steps"),
                hypers.get("num_updates"),
                horizon,
                relative,
            )
            entrypoint = "src/rtu_ppo.py"
            effective = [
                stored_seed + int(experiment.get_hypers(seed).get("seed_offset", 0))
                for seed, stored_seed in zip(typed_seeds, stored, strict=True)
            ]
        else:
            rollout = None
            updates = None
            entrypoint = "src/continuing_main.py"
            effective = [
                stored_seed
                + int(
                    experiment.get_hypers(seed)
                    .get("experiment", {})
                    .get("seed_offset", 0)
                )
                for seed, stored_seed in zip(typed_seeds, stored, strict=True)
            ]
        roots: set[str] = set()
        for seed in typed_seeds:
            context = experiment.buildSaveContext(seed, base="/run-output/results")
            database = PurePosixPath(context.resolve("results.db"))
            data = PurePosixPath(context.resolve(f"data/{seed}.npz"))
            expected_prefix = PurePosixPath("/run-output")
            if (
                not database.is_relative_to(expected_prefix)
                or not data.is_relative_to(expected_prefix)
                or database.name != "results.db"
                or data.parent.name != "data"
                or data.name != f"{seed}.npz"
                or database.parent != data.parent.parent
            ):
                _fail(f"configuration result path contract drift: {relative}")
            roots.add(database.parent.relative_to(expected_prefix).as_posix())
        if len(roots) != 1:
            _fail(f"configuration derives multiple result roots: {relative}")

        metadata_path = Path("/tmp") / f"expected-metadata-{index}.db"
        metadata_path.unlink(missing_ok=True)
        for seed in typed_seeds:
            metadata = getParamsAsDict(experiment, seed)
            metadata |= {"seed": experiment.getRun(seed)}
            attach_metadata(str(metadata_path), seed, metadata)
        metadata_contract = _canonical_sqlite(metadata_path)
        metadata_path.unlink()
        checked.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "agent": agent,
                "entrypoint": entrypoint,
                "num_permutations": experiment.numPermutations(),
                "stored_seeds": stored,
                "effective_seeds": effective,
                "rollout_steps": rollout,
                "num_updates": updates,
                "result_root": next(iter(roots)),
                "metadata_contract": metadata_contract,
            }
        )
    return checked


def _validate_executable_imports(
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Exercise exact entrypoint imports through argparse without a transition."""

    entrypoints = list(
        dict.fromkeys(cast(str, configuration["entrypoint"]) for configuration in configurations)
    )
    # Stderr signatures of compilation/cache fallback under the read-only
    # root: the first two are Numba's warning pair when it cannot persist a
    # JIT cache ("Cannot cache function ...: no locator available"), the
    # third is Matplotlib falling back to an ad-hoc temporary config dir, and
    # the rest are cache-directory/JIT failures.  Any of them means the
    # frozen MPLCONFIGDIR/NUMBA_CACHE_DIR wiring is broken even though
    # ``--help`` still exits 0, so the probe fails closed on their presence.
    forbidden = (
        "cannot cache function",
        "no locator available",
        "matplotlib created a temporary cache directory",
        "mkdir -p failed for path",
        "could not jit compile",
    )
    reports: list[dict[str, Any]] = []
    for entrypoint in entrypoints:
        executable = _SOURCE_ROOT / entrypoint
        command = [sys.executable, "-I", str(executable), "--help"]
        try:
            completed = subprocess.run(
                command,
                cwd=_SOURCE_ROOT,
                check=False,
                capture_output=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as error:
            _fail(f"zero-transition executable preflight timed out: {entrypoint}: {error}")
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        help_marker = "usage:" in stdout and "--help" in stdout
        forbidden_absent = not any(pattern in stderr.lower() for pattern in forbidden)
        if completed.returncode != 0 or not help_marker or not forbidden_absent:
            detail = stderr[-4_096:].strip()
            _fail(
                "zero-transition executable/import preflight failed for "
                f"{entrypoint}: returncode={completed.returncode}; stderr={detail}"
            )
        reports.append(
            {
                "path": entrypoint,
                "argv": command,
                "returncode": completed.returncode,
                "help_marker_present": help_marker,
                "forbidden_cache_diagnostics_absent": forbidden_absent,
            }
        )

    cache_directories: list[dict[str, Any]] = []
    for variable, expected in (
        ("MPLCONFIGDIR", "/tmp/alberta-matplotlib-cache"),
        ("NUMBA_CACHE_DIR", "/tmp/alberta-numba-cache"),
    ):
        if os.environ.get(variable) != expected:
            _fail(f"{variable} does not equal its frozen task-specific cache path")
        path = Path(expected)
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
            _fail(f"{variable} is not an exact non-symlink directory")
        metadata = path.stat()
        writable = os.access(path, os.W_OK | os.X_OK)
        # 65532:65532 is the uid:gid the screen launcher pins via
        # ``docker run --user`` (the conventional distroless "nonroot"
        # identity); ownership is re-checked from inside the container.
        if metadata.st_uid != 65532 or metadata.st_gid != 65532 or not writable:
            _fail(f"{variable} is not owned by and writable to the frozen container user")
        cache_directories.append(
            {
                "environment_variable": variable,
                "path": expected,
                "directory": True,
                "owner_uid": metadata.st_uid,
                "owner_gid": metadata.st_gid,
                "writable": writable,
            }
        )
    return {
        "status": "passed",
        "transition_operations_invoked": False,
        "entrypoints": reports,
        "cache_directories": cache_directories,
    }


def main() -> None:
    protocol_path = _PROTOCOL_ROOT / "PROTOCOL.json"
    protocol = _load_config(protocol_path)
    schema = protocol.get("schema_version")
    if type(schema) is not str or schema not in _CPU_SCHEMAS:
        _fail("unsupported frozen screening protocol schema")
    protocol_sha256 = _sha256(protocol_path)
    if protocol_sha256 != _FROZEN_PROTOCOL_SHA256[schema]:
        _fail("protocol bytes do not match the registered frozen SHA-256")
    if protocol.get("preflight") != _V3_PREFLIGHT_CONTRACT:
        _fail("CPU v3 zero-transition preflight contract drift")

    base_path = _BASE_PROTOCOL_ROOT / "PROTOCOL.json"
    base_protocol = _load_config(base_path)
    base_schema = base_protocol.get("schema_version")
    base_record = protocol.get("base_protocol")
    if (
        not isinstance(base_record, dict)
        or base_schema != _CPU_SCHEMAS[schema]
        or base_record.get("schema_version") != base_schema
        or base_record.get("sha256") != _sha256(base_path)
        or _sha256(base_path) != _FROZEN_PROTOCOL_SHA256[base_schema]
    ):
        _fail("CPU protocol base identity drift")
    predecessor_path = _PREDECESSOR_PROTOCOL_ROOT / "PROTOCOL.json"
    predecessor_protocol = _load_config(predecessor_path)
    predecessor_schema = predecessor_protocol.get("schema_version")
    predecessor_record = protocol.get("supersedes_protocol")
    if (
        not isinstance(predecessor_record, dict)
        or predecessor_schema != _PREDECESSOR_SCHEMAS[schema]
        or predecessor_record.get("schema_version") != predecessor_schema
        or predecessor_record.get("sha256") != _sha256(predecessor_path)
        or _sha256(predecessor_path) != _FROZEN_PROTOCOL_SHA256[cast(str, predecessor_schema)]
    ):
        _fail("CPU v3 predecessor identity drift")
    scoring = protocol.get("scoring")
    if (
        not isinstance(scoring, dict)
        or scoring.get("implementation_sha256") != _sha256(_SCORER_PATH)
    ):
        _fail("CPU protocol scorer identity drift")
    scorer_equivalence = _validate_scorer_equivalence(schema, scoring)

    version = importlib.metadata.version("continual-foragax")
    task = base_protocol.get("task")
    if not isinstance(task, dict):
        _fail("protocol task must be an object")
    expected_version = task.get("foragax_version")
    if version != expected_version:
        _fail(f"continual-foragax version drift: expected {expected_version}, got {version}")

    import jax

    backend = jax.default_backend()
    devices = [f"{device.platform}:{device.id}" for device in jax.devices()]
    configurations = _validate_configurations(base_protocol, base_schema)
    executable_import_preflight = _validate_executable_imports(configurations)
    payload: dict[str, Any] = {
        "schema_version": "alberta.foragax_open_development_preflight.v2",
        "status": "passed",
        "protocol_schema": schema,
        "protocol_sha256": protocol_sha256,
        "base_protocol_schema": base_schema,
        "base_protocol_sha256": _sha256(base_path),
        "predecessor_protocol_sha256": _sha256(predecessor_path),
        "scorer_sha256": _sha256(_SCORER_PATH),
        "scorer_equivalence": scorer_equivalence,
        "executable_import_preflight": executable_import_preflight,
        "source_root": str(_SOURCE_ROOT),
        "source_files": _source_records(base_protocol),
        "configurations": configurations,
        "runtime": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "nonroot": os.getuid() != 0,
            "root_filesystem_read_only": _root_is_read_only(),
            "network_interfaces": sorted(path.name for path in Path("/sys/class/net").iterdir()),
            "nvidia_device_glob": sorted(glob.glob("/dev/nvidia*")),
            "continual_foragax_version": version,
            "jax_default_backend": backend,
            "jax_devices": devices,
            "jax_platform_name": os.environ.get("JAX_PLATFORM_NAME"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "mplconfigdir": os.environ.get("MPLCONFIGDIR"),
            "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        },
    }
    runtime = cast(dict[str, Any], payload["runtime"])
    # The frozen sandbox contract runs the container as 65532:65532 — the
    # conventional distroless "nonroot" uid:gid — pinned by the launcher's
    # ``--user`` flag; verifying it in-process catches a rebuilt image or a
    # launcher that silently dropped the flag and ran as root.
    if not runtime["nonroot"] or runtime["uid"] != 65532 or runtime["gid"] != 65532:
        _fail("container is not running as the exact frozen nonroot identity")
    if not runtime["root_filesystem_read_only"]:
        _fail("container root filesystem is not read-only")
    if runtime["network_interfaces"] != ["lo"]:
        _fail("network-none sandbox exposed a non-loopback interface")
    if runtime["nvidia_device_glob"]:
        _fail("CPU sandbox exposed NVIDIA device nodes")
    if backend != "cpu" or not devices or any(not item.startswith("cpu:") for item in devices):
        _fail("JAX did not resolve an all-CPU device set")
    if runtime["jax_platform_name"] != "cpu" or runtime["jax_platforms"] != "cpu":
        _fail("JAX CPU environment is not exact")
    if runtime["nvidia_visible_devices"] != "void" or runtime["cuda_visible_devices"] != "":
        _fail("GPU visibility environment is not disabled")
    if runtime["pythonhashseed"] != "0":
        _fail("PYTHONHASHSEED is not frozen to zero")
    if runtime["mplconfigdir"] != "/tmp/alberta-matplotlib-cache":
        _fail("MPLCONFIGDIR is not frozen to the task-specific writable cache")
    if runtime["numba_cache_dir"] != "/tmp/alberta-numba-cache":
        _fail("NUMBA_CACHE_DIR is not frozen to the task-specific writable cache")
    print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
