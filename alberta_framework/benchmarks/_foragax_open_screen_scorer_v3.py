#!/usr/bin/env python3
"""Pinned-image scorer for CPU-v3 Foragax open-development screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

EMA_DECAY = 0.999
EMA_INITIAL_VALUE = 0.0
SUBSAMPLE_EVERY = 100
TAIL_FRACTION = 0.1
MAX_NPZ_BYTES = 64 * 1024**2
MAX_ZIP_MEMBERS = 512
MAX_UNCOMPRESSED_BYTES = 64 * 1024**2
MAX_REWARD_MEMBER_BYTES = 4 * 1024**2


def _require_host_int(name: str, value: object, *, minimum: int) -> int:
    """Reject bool/float aliases that ordered comparisons treat as legal counts."""
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_seed_list(seeds: object) -> list[int]:
    if type(seeds) is not list:
        raise ValueError("seeds must be a list of built-in integers")
    return [
        _require_host_int(f"seeds[{index}]", seed, minimum=0)
        for index, seed in enumerate(seeds)
    ]


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _validate_reward_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    horizon: int,
    path: Path,
) -> None:
    horizon = _require_host_int("horizon", horizon, minimum=1)
    with archive.open(info, "r") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(
                member,
                max_header_size=10_000,
            )
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(
                member,
                max_header_size=10_000,
            )
        else:
            raise ValueError(f"rewards.npy uses an unsupported NPY version: {path}")
        expected_data_bytes = horizon * dtype.itemsize
        if (
            shape != (horizon,)
            or dtype.kind not in {"i", "u", "f"}
            or dtype.hasobject
            or expected_data_bytes < 0
            or expected_data_bytes > MAX_REWARD_MEMBER_BYTES
            or info.file_size - member.tell() != expected_data_bytes
        ):
            raise ValueError(f"rewards.npy header/size contract drift: {path}")


def _validate_zip(stream: BinaryIO, path: Path, horizon: int) -> None:
    horizon = _require_host_int("horizon", horizon, minimum=1)
    stream.seek(0)
    try:
        with zipfile.ZipFile(stream, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                not infos
                or len(infos) > MAX_ZIP_MEMBERS
                or len(names) != len(set(names))
                or names.count("rewards.npy") != 1
            ):
                raise ValueError(f"invalid NPZ member inventory: {path}")
            total = 0
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    info.is_dir()
                    or member.is_absolute()
                    or member.name != info.filename
                    or member.suffix != ".npy"
                    or "." in member.parts
                    or ".." in member.parts
                    or info.flag_bits & 0x1
                    or info.file_size < 0
                    or info.file_size > MAX_UNCOMPRESSED_BYTES
                    or (
                        info.filename == "rewards.npy"
                        and info.file_size > MAX_REWARD_MEMBER_BYTES
                    )
                    or (info.compress_size == 0 and info.file_size != 0)
                ):
                    raise ValueError(f"unsafe NPZ member: {path}:{info.filename}")
                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError(f"NPZ expands beyond its bound: {path}")
                if info.filename == "rewards.npy":
                    _validate_reward_header(archive, info, horizon, path)
            if archive.testzip() is not None:
                raise ValueError(f"NPZ CRC validation failed: {path}")
    except NotImplementedError as error:
        raise ValueError(f"NPZ uses unsupported ZIP compression: {path}") from error


def _load_reward_archive(
    path: Path,
    horizon: int,
) -> tuple[np.ndarray, str, int]:
    horizon = _require_host_int("horizon", horizon, minimum=1)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_NPZ_BYTES
        ):
            raise ValueError(f"unsafe reward archive: {path}")
        _validate_zip(stream, path, horizon)
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            if archive.files.count("rewards") != 1:
                raise ValueError(f"NPZ rewards inventory is invalid: {path}")
            rewards = np.asarray(archive["rewards"])
        scored_input = np.array(rewards, copy=True)
        if scored_input.shape != (horizon,):
            raise ValueError(f"raw reward array must have exact shape ({horizon},)")
        digest = _sha256_stream(stream)
        after = os.fstat(stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"reward archive changed during scoring: {path}")
    return scored_input, digest, after.st_size


def score_rewards(rewards: np.ndarray, horizon: int) -> dict[str, Any]:
    """Compute the frozen scorer with bit-identical arithmetic."""
    horizon = _require_host_int("horizon", horizon, minimum=1)
    if rewards.shape != (horizon,):
        raise ValueError(f"raw reward array must have exact shape ({horizon},)")
    if rewards.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("raw rewards must have a numeric dtype")
    if not bool(np.all(np.isfinite(rewards))):
        raise ValueError("raw rewards must all be finite")
    rewards64 = rewards.astype(np.float64, copy=False)
    ema = EMA_INITIAL_VALUE
    samples: list[float] = []
    for index, reward in enumerate(rewards64):
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * float(reward)
        if index % SUBSAMPLE_EVERY == 0:
            samples.append(ema)
    expected = (horizon + SUBSAMPLE_EVERY - 1) // SUBSAMPLE_EVERY
    if len(samples) != expected:
        raise RuntimeError("internal EMA sample count mismatch")
    tail_start = int((1.0 - TAIL_FRACTION) * len(samples))
    tail = np.asarray(samples[tail_start:], dtype=np.float64)
    if tail.size != expected - tail_start or tail.size == 0:
        raise RuntimeError("internal EMA tail boundary mismatch")
    contiguous = np.ascontiguousarray(rewards)
    with np.errstate(over="ignore", invalid="ignore"):
        reward_sum = float(np.sum(rewards64, dtype=np.float64))
        score = float(np.mean(tail, dtype=np.float64))
    if not all(math.isfinite(value) for value in (reward_sum, score, ema)):
        raise ValueError("raw rewards produce a non-finite aggregate or EMA")
    return {
        "reward_dtype": rewards.dtype.str,
        "reward_shape": [horizon],
        "reward_trace_sha256": hashlib.sha256(
            contiguous.tobytes(order="C")
        ).hexdigest(),
        "reward_sum_float64": reward_sum,
        "fov_last_10pct_ema_auc": score,
        "final_unadjusted_ema": float(ema),
        "ema_sample_count": expected,
        "ema_tail_start_index": tail_start,
        "ema_tail_sample_count": int(tail.size),
    }


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or "." in path.parts
        or ".." in path.parts
    ):
        raise ValueError("result root is not canonical and relative")
    return path


def score_archives(
    payload_root: Path,
    result_root: str,
    seeds: list[int],
    horizon: int,
) -> dict[str, Any]:
    horizon = _require_host_int("horizon", horizon, minimum=1)
    seeds = _require_seed_list(seeds)
    root = payload_root.resolve(strict=True)
    relative_root = _relative(result_root)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        relative = relative_root / "data" / f"{seed}.npz"
        path = root / relative
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"reward archive path contains a symlink: {path}")
        rewards, npz_sha256, npz_size = _load_reward_archive(path, horizon)
        scored = score_rewards(rewards, horizon)
        records.append(
            {
                "seed": seed,
                "archive_path": (PurePosixPath("payload") / relative).as_posix(),
                "npz_sha256": npz_sha256,
                "npz_size_bytes": npz_size,
                **scored,
            }
        )
    return {
        "schema_version": "alberta.foragax_open_screen_scoring.v2",
        "horizon": horizon,
        "seeds": seeds,
        "result_root": relative_root.as_posix(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    args = parser.parse_args()
    if args.horizon <= 0 or any(seed < 0 for seed in args.seed):
        raise SystemExit("horizon and seeds must be non-negative integers")
    try:
        payload = score_archives(
            args.payload_root,
            args.result_root,
            args.seed,
            args.horizon,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
