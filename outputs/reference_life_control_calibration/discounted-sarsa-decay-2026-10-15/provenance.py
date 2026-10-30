"""Fail-closed provenance checks for the RiverSwim decay diagnostic."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def require_record_source_commit(actual: object, expected: str, label: str) -> None:
    """Reject a raw record whose source commit differs from the plan."""

    if actual != expected:
        raise ValueError(
            f"{label} source commit {actual} does not match pinned plan commit {expected}"
        )


def require_record_digest(actual: object, expected: str, label: str, field: str) -> None:
    """Reject a raw record whose recorded digest differs from the retained bytes."""

    if actual != expected:
        raise ValueError(
            f"{label} {field} {actual} does not match retained bytes {expected}"
        )


def require_record_sources(
    record: dict[str, object], label: str, digest_by_path: dict[str, str]
) -> None:
    """Reject a raw record whose source digests differ from the pinned commit blobs."""

    recorded = record.get("source_sha256")
    if recorded == digest_by_path:
        return
    recorded_map = recorded if isinstance(recorded, dict) else {}
    differing = sorted(
        path
        for path in set(recorded_map) | set(digest_by_path)
        if recorded_map.get(path) != digest_by_path.get(path)
    )
    preview = ", ".join(differing[:3]) or "no shared source paths"
    raise ValueError(f"{label} source_sha256 differs from commit blobs at {preview}")


def source_blob_digests(repo_root: Path, commit: str, paths: list[str]) -> dict[str, str]:
    """Return the SHA-256 of every tracked path's committed blob content."""

    request = "".join(f"{commit}:{path}\n" for path in paths).encode("utf-8")
    try:
        completed = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=repo_root,
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot read commit {commit} in {repo_root}: {error}") from error
    data = completed.stdout
    position = 0
    digests: dict[str, str] = {}
    for path in paths:
        header_end = data.find(b"\n", position)
        if header_end < 0:
            raise ValueError(f"source path {path} is missing from commit {commit}")
        header = data[position:header_end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"source path {path} is missing from commit {commit}")
        size = int(header[2])
        start = header_end + 1
        digests[path] = hashlib.sha256(data[start : start + size]).hexdigest()
        position = start + size + 1
    return digests
