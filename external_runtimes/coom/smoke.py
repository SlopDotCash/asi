"""Bounded real-engine COOM CO8 qualification smoke.

This script runs only inside the isolated image described by the sibling
Dockerfile. It is deliberately fixed-action and mechanism-off: its output is a
runtime qualification receipt, not a performance result.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, cast

import numpy as np

SCHEMA = "asi.coom_external_runtime_qualification.development.v1"
SOURCE_COMMIT = "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e"
SOURCE_TREE = "6e935b4ad6f3e52280de871e56937071aa5cd13f"
SOURCE_ARCHIVE_SHA256 = "a4736e9916468482d75831d53a12a8601c4da91cd40b9b24d313522034a15661"
SOURCE_LICENSE_SHA256 = "47c8691ec5399bc8c58bcfaf0ba43b4ff48e6917c894c03748e3e0d14345d649"
SOURCE_ASSET_MANIFEST_SHA256 = "deaa00979139cf80055f9d04d65800abc78c4feb11e061274e1a4486f9fa6cab"
PATCH_SHA256 = "25bc846908e573ff1c7d02909a9bb895570e0beafe1a928dbd0d5fd4b63835a7"
PATCHED_REWARD_WRAPPER_SHA256 = (
    "0ab457a6bc95dc2551b2c81608d1619549e56ced47e2c85949c39b87b8b5a8cf"
)
EXPECTED_TRACE_SHA256 = "c74968494ccebaaeac4bc1e0c0f1db7546ac5091b831c05a4c0c727266da696f"
EXPECTED_DISTRIBUTIONS = (
    ("Farama-Notifications", "0.0.6"),
    ("cloudpickle", "3.1.2"),
    ("gymnasium", "0.28.1"),
    ("jax-jumpy", "1.0.0"),
    ("numpy", "1.26.4"),
    ("opencv-python-headless", "4.11.0.86"),
    ("pip", "25.0.1"),
    ("pygame-ce", "2.5.8"),
    ("scipy", "1.11.4"),
    ("typing_extensions", "4.16.0"),
    ("vizdoom", "1.3.0"),
)
TASK_NAMES = (
    "pitfall-default",
    "arms_dealer-default",
    "hide_and_seek-default",
    "floor_is_lava-default",
    "chainsaw-default",
    "raise_the_roof-default",
    "run_and_gun-default",
    "health_gathering-default",
)
SEED = 1_582_000
STEPS_PER_TASK = 2
_QUALIFICATION_ROOT = Path("/opt/qualification")
_MAX_MANIFEST_BYTES = 8192
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_PROC_STATUS_BYTES = 64 * 1024


def _array_sha256(value: np.ndarray) -> str:
    if type(value) is not np.ndarray:
        raise ValueError("observation must be an exact NumPy array before hashing")
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _asset_manifest() -> tuple[int, int, str]:
    root = Path("/opt/coom")
    paths = sorted((*root.rglob("*.cfg"), *root.rglob("*.wad")))
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        raw = path.read_bytes()
        total += len(raw)
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return len(paths), total, digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_identity() -> dict[str, object]:
    """Validate the exact non-root runtime before importing or starting COOM."""

    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        raise ValueError("COOM qualification requires Linux process identities")
    uid = os.getuid()
    gid = os.getgid()
    if uid != 65532 or gid != 65532:
        raise ValueError("COOM qualification requires exact UID/GID 65532")

    status_raw = Path("/proc/self/status").read_bytes()
    if len(status_raw) > _MAX_PROC_STATUS_BYTES:
        raise ValueError("process status exceeds its byte limit")
    fields: dict[bytes, bytes] = {}
    for line in status_raw.splitlines():
        if b":" not in line:
            continue
        key, value = line.split(b":", 1)
        if key in (b"CapEff", b"NoNewPrivs"):
            if key in fields:
                raise ValueError("process status contains duplicate security fields")
            fields[key] = value.strip()
    if fields != {b"CapEff": b"0000000000000000", b"NoNewPrivs": b"1"}:
        raise ValueError("COOM qualification requires no capabilities and NoNewPrivs")

    distributions: list[tuple[str, str]] = []
    for index, distribution in enumerate(importlib.metadata.distributions()):
        if index >= len(EXPECTED_DISTRIBUTIONS):
            raise ValueError("installed distributions exceed the exact runtime roster")
        name = distribution.metadata["Name"]
        version = distribution.version
        distributions.append(
            (
                _exact_str(name, name="installed distribution name"),
                _exact_str(version, name="installed distribution version"),
            )
        )
    installed = tuple(sorted(distributions))
    if len(installed) != len(set(installed)) or installed != EXPECTED_DISTRIBUTIONS:
        raise ValueError("installed distributions differ from the exact runtime roster")

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "uid": uid,
        "gid": gid,
        "effective_capabilities_hex": fields[b"CapEff"].decode("ascii"),
        "no_new_privileges": True,
        "installed_distributions": [list(item) for item in installed],
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "gymnasium": importlib.metadata.version("gymnasium"),
        "vizdoom": importlib.metadata.version("vizdoom"),
        "opencv_python_headless": importlib.metadata.version("opencv-python-headless"),
    }


def _git_object(kind: bytes, payload: bytes) -> bytes:
    header = kind + b" " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload, usedforsecurity=False).digest()


def _source_tree_sha1(root: Path) -> str:
    """Recompute the upstream Git tree, reversing only the reviewed import patch."""

    patched_path = Path("COOM/wrappers/reward.py")

    def tree(directory: Path) -> bytes:
        entries: list[tuple[bytes, bytes]] = []
        for path in directory.iterdir():
            relative = path.relative_to(root)
            name = path.name.encode("utf-8")
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                mode = b"40000"
                identity = tree(path)
                sort_key = name + b"/"
            elif stat.S_ISLNK(metadata.st_mode):
                mode = b"120000"
                identity = _git_object(b"blob", os.readlink(path).encode("utf-8"))
                sort_key = name
            elif stat.S_ISREG(metadata.st_mode):
                mode = b"100755" if metadata.st_mode & stat.S_IXUSR else b"100644"
                raw = path.read_bytes()
                if relative == patched_path:
                    current = b"from gymnasium import RewardWrapper"
                    original = b"from gym import RewardWrapper"
                    if raw.count(current) != 1 or original in raw:
                        raise ValueError("reviewed COOM import patch cannot be reversed exactly")
                    raw = raw.replace(current, original)
                identity = _git_object(b"blob", raw)
                sort_key = name
            else:
                raise ValueError("COOM source contains an unsupported filesystem entry")
            entries.append((sort_key, mode + b" " + name + b"\0" + identity))
        payload = b"".join(value for _, value in sorted(entries))
        return _git_object(b"tree", payload)

    return tree(root).hex()


def _exact_keys(
    value: object, expected: set[str] | frozenset[str], *, name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} fields differ from the qualification schema")
    keys = tuple(value.keys())
    if any(type(key) is not str for key in keys):
        raise ValueError(f"{name} keys must be exact strings")
    if frozenset(keys) != frozenset(expected):
        raise ValueError(f"{name} fields differ from the qualification schema")
    return value


def _json_pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if type(key) is not str or key in result:
            raise ValueError("JSON contains duplicate or non-string keys")
        result[key] = value
    return result


def _reject_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON token {token}")


def _exact_str(value: object, *, name: str, maximum: int = 256) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        raise ValueError(f"{name} must be a bounded exact string")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _exact_int_list(value: object, expected: list[int], *, name: str) -> list[int]:
    if (
        type(value) is not list
        or len(value) != len(expected)
        or any(type(item) is not int for item in value)
        or value != expected
    ):
        raise ValueError(f"{name} differs from the exact integer list")
    return value


def _trusted_observation(value: object, *, name: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact NumPy array")
    if value.dtype != np.dtype(np.float64) or value.shape != (84, 84, 3):
        raise ValueError(f"{name} shape or dtype differs from the runtime contract")
    if value.nbytes != 84 * 84 * 3 * np.dtype(np.float64).itemsize:
        raise ValueError(f"{name} byte count differs from its shape and dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _trusted_reward(value: object) -> float:
    # COOM's pitfall wrapper returns one exact np.float64 scalar on the fixed
    # trace; admit only that concrete provider type and Python float. Avoid
    # isinstance/coercion of arbitrary objects, which could dispatch hooks.
    actual_type = type(value)
    if actual_type is not float and actual_type is not np.float64:
        raise ValueError("COOM reward must be an exact float scalar")
    reward = float(cast(float | np.float64, value))
    if not math.isfinite(reward):
        raise ValueError("COOM reward must be finite")
    return reward


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _load_qualification_manifest() -> dict[str, object]:
    raw = (_QUALIFICATION_ROOT / "qualification-manifest.json").read_bytes()
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("qualification manifest exceeds its byte limit")

    value = json.loads(
        raw,
        object_pairs_hook=_json_pairs,
        parse_constant=_reject_constant,
    )
    manifest = _exact_keys(
        value,
        {
            "schema",
            "base_image_digest",
            "dockerfile_sha256",
            "requirements_lock_sha256",
            "smoke_sha256",
            "patch_sha256",
        },
        name="qualification manifest",
    )
    if _exact_str(manifest["schema"], name="manifest schema") != (
        "asi.coom_external_runtime.inputs.v1"
    ):
        raise ValueError("qualification manifest schema differs")
    if _exact_str(manifest["base_image_digest"], name="base image digest") != (
        "python:3.12.12-slim-bookworm@sha256:"
        "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
    ):
        raise ValueError("base image digest differs from the qualification contract")
    files = {
        "dockerfile_sha256": "Dockerfile.source",
        "requirements_lock_sha256": "requirements.lock",
        "smoke_sha256": "smoke.py",
        "patch_sha256": "coom-gymnasium.patch",
    }
    for field, relative in files.items():
        expected = _sha256(manifest[field], name=field)
        if _file_sha256(_QUALIFICATION_ROOT / relative) != expected:
            raise ValueError(f"{relative} differs from the qualification manifest")
    return manifest


def load_receipt(path: Path) -> dict[str, object]:
    """Strictly load one bounded regular receipt without following a final symlink."""

    if type(path) is not type(Path()):
        raise ValueError("receipt path must be an exact concrete Path")
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("strict receipt loading requires O_NOFOLLOW")
    descriptor = os.open(
        Path(path),
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("receipt input must be a regular file")
        if before.st_nlink != 1:
            raise ValueError("receipt input must have exactly one filesystem link")
        if before.st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("receipt input exceeds its byte limit")
        chunks: list[bytes] = []
        remaining = _MAX_RECEIPT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        stable = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
        if (
            len(raw) != before.st_size
            or len(raw) > _MAX_RECEIPT_BYTES
            or after.st_nlink != 1
            or any(getattr(before, field) != getattr(after, field) for field in stable)
        ):
            raise ValueError("receipt input changed while being read")
    finally:
        os.close(descriptor)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("receipt input must be UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("receipt input must be bounded valid JSON") from exc
    if type(value) is not dict:
        raise ValueError("receipt input must be an exact JSON object")
    return value


def _open_output_parent(path: Path) -> tuple[Path, int]:
    if type(path) is not type(Path()):
        raise ValueError("output path must be an exact concrete Path")
    destination = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(os.path.sep, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in destination.parent.parts[1:]:
            if component in ("", ".", ".."):
                raise ValueError("output path contains an unsafe directory component")
            next_descriptor = os.open(
                component,
                os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return destination, descriptor
    except Exception:
        os.close(descriptor)
        raise


def preflight_new_output(path: Path) -> None:
    destination, parent_fd = _open_output_parent(path)
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"refusing to overwrite immutable receipt: {destination}")
    finally:
        os.close(parent_fd)


def _link_unnamed_file(file_fd: int, parent_fd: int, name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    encoded_name = os.fsencode(name)
    if linkat(file_fd, b"", parent_fd, encoded_name, 0x1000) == 0:
        return
    direct_error = ctypes.get_errno()
    if direct_error == errno.EEXIST:
        raise FileExistsError(direct_error, os.strerror(direct_error), name)

    # AT_EMPTY_PATH requires CAP_DAC_READ_SEARCH for an unprivileged process,
    # which deliberately conflicts with this runtime's --cap-drop ALL gate.
    # Linux documents /proc/self/fd plus AT_SYMLINK_FOLLOW as the capability-free
    # way to publish the same O_TMPFILE inode.
    proc_fd_path = os.fsencode(f"/proc/self/fd/{file_fd}")
    if linkat(-100, proc_fd_path, parent_fd, encoded_name, 0x400) == 0:
        return
    fallback_error = ctypes.get_errno()
    if fallback_error == errno.EEXIST:
        raise FileExistsError(fallback_error, os.strerror(fallback_error), name)
    raise OSError(fallback_error, os.strerror(fallback_error), name)


def _sync_filesystem(file_fd: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    syncfs = libc.syncfs
    syncfs.argtypes = (ctypes.c_int,)
    syncfs.restype = ctypes.c_int
    if syncfs(file_fd) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def write_new_receipt(path: Path, receipt: dict[str, object]) -> Path:
    """Atomically publish one validated receipt without replacing existing bytes."""

    validate_receipt(receipt)
    destination, parent_fd = _open_output_parent(path)
    encoded = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    if len(encoded) > _MAX_RECEIPT_BYTES:
        os.close(parent_fd)
        raise ValueError("receipt output exceeds its byte limit")
    file_fd: int | None = None
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite immutable receipt: {destination}")
        if not hasattr(os, "O_TMPFILE"):
            raise OSError("immutable receipt publication requires Linux O_TMPFILE")
        file_fd = os.open(
            ".",
            os.O_WRONLY | os.O_CLOEXEC | os.O_TMPFILE,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            progress = os.write(file_fd, view[written:])
            if progress <= 0:
                raise OSError("receipt output write made no progress")
            written += progress
        os.fsync(file_fd)
        os.fchmod(file_fd, 0o444)
        try:
            _link_unnamed_file(file_fd, parent_fd, destination.name)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable receipt: {destination}"
            ) from exc
        _sync_filesystem(file_fd)
        return destination
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _trace() -> tuple[list[dict[str, object]], int, int, int]:
    from COOM.env.builder import (  # type: ignore[import-not-found]
        build_multi_discrete_actions,
        make_sequence,
    )
    from COOM.utils.config import Sequence  # type: ignore[import-not-found]

    start = time.perf_counter_ns()
    records: list[dict[str, object]] = []
    task_resets = 0
    environment_steps = 0
    environments_value = make_sequence(
        Sequence.CO8,
        doom_kwargs={
            "seed": SEED,
            "render": False,
            "test_only": False,
            "resolution": "160X120",
            "frame_skip": 4,
            "action_space_fn": build_multi_discrete_actions,
            "num_tasks": 8,
        },
        wrapper_config={
            "augment": False,
            "resize": True,
            "frame_height": 84,
            "frame_width": 84,
            "rescale": True,
            "normalize_observation": False,
            "frame_stack": False,
            "lstm": False,
            "record": False,
            "sparse_rewards": False,
        },
    )
    if type(environments_value) is not list or len(environments_value) != len(TASK_NAMES):
        raise ValueError("COOM provider must return exactly eight environments in a list")
    environments = environments_value
    try:
        for task_index, (environment, expected_name) in enumerate(
            zip(environments, TASK_NAMES, strict=True)
        ):
            observed_name = environment.unwrapped.name
            if _exact_str(observed_name, name="environment name") != expected_name:
                raise ValueError("COOM environment order or name differs")
            reset_result = environment.reset()
            if type(reset_result) is not tuple or len(reset_result) != 2:
                raise ValueError("COOM reset must return an exact two-tuple")
            observation, reset_info = reset_result
            _exact_keys(reset_info, frozenset(), name="reset info")
            reset = _trusted_observation(observation, name="reset observation")
            task_resets += 1
            steps: list[dict[str, object]] = []
            for _ in range(STEPS_PER_TASK):
                step_result = environment.step(0)
                if type(step_result) is not tuple or len(step_result) != 5:
                    raise ValueError("COOM step must return an exact five-tuple")
                observation, reward, terminated, truncated, info = step_result
                _exact_keys(info, frozenset(), name="step info")
                value = _trusted_observation(observation, name="step observation")
                trusted_reward = _trusted_reward(reward)
                if type(terminated) is not bool or type(truncated) is not bool:
                    raise ValueError("COOM termination flags must be exact bools")
                steps.append(
                    {
                        "action": 0,
                        "observation_sha256": _array_sha256(value),
                        "observation_shape": list(value.shape),
                        "observation_dtype": value.dtype.str,
                        "reward": trusted_reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": {},
                    }
                )
                environment_steps += 1
            records.append(
                {
                    "task_index": task_index,
                    "name": observed_name,
                    "reset_info": {},
                    "reset_observation_sha256": _array_sha256(reset),
                    "reset_observation_shape": list(reset.shape),
                    "reset_observation_dtype": reset.dtype.str,
                    "steps": steps,
                }
            )
    finally:
        for environment in environments:
            environment.close()
    return records, time.perf_counter_ns() - start, task_resets, environment_steps


def validate_receipt(receipt: object) -> None:
    root = _exact_keys(
        receipt,
        {
            "schema",
            "qualification_inputs",
            "source",
            "runtime",
            "trace",
            "trace_sha256",
            "resource_receipt",
            "claims",
        },
        name="receipt",
    )
    if _exact_str(root["schema"], name="receipt schema") != SCHEMA:
        raise ValueError("receipt schema differs")
    manifest = _load_qualification_manifest()
    supplied_manifest = _exact_keys(
        root["qualification_inputs"],
        frozenset(manifest),
        name="receipt qualification inputs",
    )
    for field, expected in manifest.items():
        if _exact_str(supplied_manifest[field], name=f"receipt {field}") != expected:
            raise ValueError("receipt qualification inputs differ from verified local bytes")
    source = _exact_keys(
        root["source"],
        {
            "repository",
            "commit",
            "git_tree",
            "archive_sha256",
            "license",
            "license_sha256",
            "asset_count",
            "asset_bytes",
            "asset_manifest_sha256",
            "qualification_patch_sha256",
            "qualification_patch_scope",
            "patched_reward_wrapper_sha256",
        },
        name="source",
    )
    expected_source = {
        "repository": "https://github.com/TTomilin/COOM.git",
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_TREE,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "license": "MIT",
        "license_sha256": SOURCE_LICENSE_SHA256,
        "asset_count": 33,
        "asset_bytes": 4_153_440,
        "asset_manifest_sha256": SOURCE_ASSET_MANIFEST_SHA256,
        "qualification_patch_sha256": PATCH_SHA256,
        "qualification_patch_scope": "gym RewardWrapper import only",
        "patched_reward_wrapper_sha256": PATCHED_REWARD_WRAPPER_SHA256,
    }
    for name in (
        "repository",
        "commit",
        "git_tree",
        "license",
        "qualification_patch_scope",
    ):
        _exact_str(source[name], name=f"source {name}", maximum=512)
    for name in (
        "archive_sha256",
        "license_sha256",
        "asset_manifest_sha256",
        "qualification_patch_sha256",
        "patched_reward_wrapper_sha256",
    ):
        _sha256(source[name], name=f"source {name}")
    _exact_int(source["asset_count"], name="asset count", minimum=1)
    _exact_int(source["asset_bytes"], name="asset bytes", minimum=1)
    if source != expected_source:
        raise ValueError("source identity differs from the exact qualification inputs")
    runtime = _exact_keys(
        root["runtime"],
        {
            "python",
            "python_implementation",
            "platform",
            "numpy",
            "scipy",
            "gymnasium",
            "vizdoom",
            "opencv_python_headless",
            "uid",
            "gid",
            "effective_capabilities_hex",
            "no_new_privileges",
            "installed_distributions",
        },
        name="runtime",
    )
    expected_versions = {
        "python": "3.12.12",
        "python_implementation": "CPython",
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "gymnasium": "0.28.1",
        "vizdoom": "1.3.0",
        "opencv_python_headless": "4.11.0.86",
    }
    for name in runtime:
        if name not in {"uid", "gid", "no_new_privileges", "installed_distributions"}:
            _exact_str(runtime[name], name=f"runtime {name}", maximum=256)
    if any(runtime[name] != value for name, value in expected_versions.items()):
        raise ValueError("runtime versions differ from the hash-locked qualification")
    if (
        _exact_int(runtime["uid"], name="runtime uid") != 65532
        or _exact_int(runtime["gid"], name="runtime gid") != 65532
        or _exact_str(
            runtime["effective_capabilities_hex"], name="effective capabilities"
        )
        != "0000000000000000"
        or _exact_bool(runtime["no_new_privileges"], name="no_new_privileges") is not True
    ):
        raise ValueError("runtime process security identity differs")
    installed_value = runtime["installed_distributions"]
    if type(installed_value) is not list or len(installed_value) != len(
        EXPECTED_DISTRIBUTIONS
    ):
        raise ValueError("installed distributions differ from the exact runtime roster")
    installed: list[tuple[str, str]] = []
    for item in installed_value:
        if type(item) is not list or len(item) != 2:
            raise ValueError("installed distribution entries must be exact pairs")
        installed.append(
            (
                _exact_str(item[0], name="installed distribution name"),
                _exact_str(item[1], name="installed distribution version"),
            )
        )
    if tuple(installed) != EXPECTED_DISTRIBUTIONS:
        raise ValueError("installed distributions differ from the exact runtime roster")
    trace = _exact_keys(
        root["trace"],
        {"seed", "sequence", "steps_per_task", "fixed_action", "frame_skip", "resize", "records"},
        name="trace",
    )
    for name, expected in (
        ("seed", SEED),
        ("steps_per_task", 2),
        ("fixed_action", 0),
        ("frame_skip", 4),
    ):
        if _exact_int(trace[name], name=f"trace {name}") != expected:
            raise ValueError("trace protocol differs from the frozen qualification smoke")
    if _exact_str(trace["sequence"], name="trace sequence") != "CO8":
        raise ValueError("trace protocol differs from the frozen qualification smoke")
    _exact_int_list(trace["resize"], [84, 84], name="trace resize")
    records = trace["records"]
    if type(records) is not list or len(records) != 8:
        raise ValueError("trace must contain exactly eight ordered task records")
    for task_index, (record_value, task_name) in enumerate(zip(records, TASK_NAMES, strict=True)):
        record = _exact_keys(
            record_value,
            {
                "task_index",
                "name",
                "reset_info",
                "reset_observation_sha256",
                "reset_observation_shape",
                "reset_observation_dtype",
                "steps",
            },
            name=f"task {task_index}",
        )
        if _exact_int(record["task_index"], name="task index") != task_index:
            raise ValueError("task record identity or reset payload differs")
        if _exact_str(record["name"], name="task name") != task_name:
            raise ValueError("task record identity or reset payload differs")
        _exact_keys(record["reset_info"], frozenset(), name="receipt reset info")
        _exact_int_list(
            record["reset_observation_shape"], [84, 84, 3], name="reset shape"
        )
        if _exact_str(record["reset_observation_dtype"], name="reset dtype") != "<f8":
            raise ValueError("task reset dtype differs")
        _sha256(record["reset_observation_sha256"], name="reset observation hash")
        steps = record["steps"]
        if type(steps) is not list or len(steps) != 2:
            raise ValueError("each task must contain exactly two step records")
        for step_value in steps:
            step = _exact_keys(
                step_value,
                {
                    "action",
                    "info",
                    "observation_dtype",
                    "observation_sha256",
                    "observation_shape",
                    "reward",
                    "terminated",
                    "truncated",
                },
                name="step",
            )
            if _exact_int(step["action"], name="step action") != 0:
                raise ValueError("step payload differs from the exact bounded contract")
            _exact_keys(step["info"], frozenset(), name="receipt step info")
            if _exact_str(step["observation_dtype"], name="step dtype") != "<f8":
                raise ValueError("step dtype differs")
            _exact_int_list(step["observation_shape"], [84, 84, 3], name="step shape")
            if type(step["reward"]) is not float or not math.isfinite(step["reward"]):
                raise ValueError("step reward must be an exact finite float")
            _exact_bool(step["terminated"], name="step terminated")
            _exact_bool(step["truncated"], name="step truncated")
            _sha256(step["observation_sha256"], name="step observation hash")
    trace_bytes = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed_trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
    if (
        _sha256(root["trace_sha256"], name="trace hash") != observed_trace_sha256
        or observed_trace_sha256 != EXPECTED_TRACE_SHA256
    ):
        raise ValueError("trace differs from the independently repeated deterministic golden")
    resources = _exact_keys(
        root["resource_receipt"],
        {
            "task_resets",
            "environment_steps",
            "environment_step_queries",
            "policy_queries",
            "learner_updates",
            "model_queries",
            "elapsed_ns_telemetry_only",
        },
        name="resource receipt",
    )
    derived_resets = len(records)
    derived_steps = sum(len(cast(list[object], record["steps"])) for record in records)
    expected_resources = {
        "task_resets": derived_resets,
        "environment_steps": derived_steps,
        "environment_step_queries": derived_steps,
        "policy_queries": 0,
        "learner_updates": 0,
        "model_queries": 0,
    }
    if any(
        type(resources.get(name)) is not int or resources[name] != value
        for name, value in expected_resources.items()
    ):
        raise ValueError("resource receipt differs from exact fixed-action work")
    elapsed = resources["elapsed_ns_telemetry_only"]
    if type(elapsed) is not int or elapsed < 0:
        raise ValueError("timing telemetry must be a nonnegative exact integer")
    claims = _exact_keys(
        root["claims"],
        {
            "external_runtime_executed",
            "execution_attested",
            "mechanism_off",
            "performance_metrics_computed",
            "paper_parity_claimed",
            "scientific_promotion_allowed",
            "negative_outcome_retained",
        },
        name="claims",
    )
    for name in claims:
        _exact_bool(claims[name], name=f"claim {name}")
    if claims != {
        "external_runtime_executed": True,
        "execution_attested": False,
        "mechanism_off": True,
        "performance_metrics_computed": False,
        "paper_parity_claimed": False,
        "scientific_promotion_allowed": False,
        "negative_outcome_retained": False,
    }:
        raise ValueError("receipt claims exceed the bounded unattested qualification")


def build_receipt() -> dict[str, object]:
    qualification_inputs = _load_qualification_manifest()
    runtime = _runtime_identity()
    root = Path("/opt/coom")
    if _source_tree_sha1(root) != SOURCE_TREE:
        raise SystemExit("COOM source archive does not reconstruct the pinned Git tree")
    if _file_sha256(root / "LICENSE.txt") != SOURCE_LICENSE_SHA256:
        raise SystemExit("COOM license bytes differ from the audited source pin")
    if (
        _file_sha256(root / "COOM/wrappers/reward.py")
        != PATCHED_REWARD_WRAPPER_SHA256
    ):
        raise SystemExit("COOM qualification patch result differs from the reviewed bytes")
    asset_count, asset_bytes, asset_sha256 = _asset_manifest()
    if (asset_count, asset_bytes, asset_sha256) != (
        33,
        4_153_440,
        SOURCE_ASSET_MANIFEST_SHA256,
    ):
        raise SystemExit("COOM WAD/config asset manifest differs from the audited source pin")
    records, elapsed_ns, task_resets, environment_steps = _trace()
    trace = {
        "seed": SEED,
        "sequence": "CO8",
        "steps_per_task": STEPS_PER_TASK,
        "fixed_action": 0,
        "frame_skip": 4,
        "resize": [84, 84],
        "records": records,
    }
    trace_bytes = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "qualification_inputs": qualification_inputs,
        "source": {
            "repository": "https://github.com/TTomilin/COOM.git",
            "commit": SOURCE_COMMIT,
            "git_tree": SOURCE_TREE,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "license": "MIT",
            "license_sha256": SOURCE_LICENSE_SHA256,
            "asset_count": asset_count,
            "asset_bytes": asset_bytes,
            "asset_manifest_sha256": asset_sha256,
            "qualification_patch_sha256": PATCH_SHA256,
            "patched_reward_wrapper_sha256": PATCHED_REWARD_WRAPPER_SHA256,
            "qualification_patch_scope": "gym RewardWrapper import only",
        },
        "runtime": runtime,
        "trace": trace,
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "resource_receipt": {
            "task_resets": task_resets,
            "environment_steps": environment_steps,
            "environment_step_queries": environment_steps,
            "policy_queries": 0,
            "learner_updates": 0,
            "model_queries": 0,
            "elapsed_ns_telemetry_only": elapsed_ns,
        },
        "claims": {
            "external_runtime_executed": True,
            "execution_attested": False,
            "mechanism_off": True,
            "performance_metrics_computed": False,
            "paper_parity_claimed": False,
            "scientific_promotion_allowed": False,
            "negative_outcome_retained": False,
        },
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt_file(path: Path) -> dict[str, object]:
    receipt = load_receipt(path)
    validate_receipt(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="atomically retain a new receipt")
    mode.add_argument(
        "--validate-receipt",
        "--validate",
        dest="validate_receipt",
        type=Path,
        help="strictly validate a retained receipt",
    )
    args = parser.parse_args(argv)
    if args.validate_receipt is not None:
        receipt = validate_receipt_file(args.validate_receipt)
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema": receipt["schema"],
                    "trace_sha256": receipt["trace_sha256"],
                    "scientific_promotion_allowed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.output is not None:
        preflight_new_output(args.output)
    receipt = build_receipt()
    if args.output is None:
        json.dump(receipt, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
    else:
        write_new_receipt(args.output, receipt)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
