"""Canonical provenance helpers for maintained IPMNIST diagnostics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    """Hash one regular input file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Return stable byte identity and size for one consumed file."""
    resolved = path.resolve(strict=True)
    name = str(resolved)
    if relative_to is not None:
        try:
            name = str(resolved.relative_to(relative_to.resolve(strict=True)))
        except ValueError:
            pass
    metadata = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"provenance input is not a regular file: {resolved}")
    return {"path": name, "bytes": metadata.st_size, "sha256": sha256_file(resolved)}


def input_file_identities(
    paths: Iterable[Path], *, relative_to: Path | None = None
) -> list[dict[str, Any]]:
    """Return sorted, duplicate-free identities for every consumed input."""
    resolved = sorted({path.resolve(strict=True) for path in paths})
    return [file_identity(path, relative_to=relative_to) for path in resolved]


def array_identity(array: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Bind shape, dtype, and canonical C-order bytes of one array."""
    if not isinstance(array, np.ndarray):
        raise ValueError("array identity requires a NumPy array")
    # Object cells are process-local pointers, and a void dtype string omits its fields.
    if array.dtype.kind in "OV":
        raise ValueError(f"array identity cannot bind dtype {array.dtype}")
    # ``ascontiguousarray`` promotes 0-d arrays, so the shape comes from the input.
    contiguous = np.ascontiguousarray(array)
    payload = memoryview(contiguous).cast("B") if contiguous.size else b""
    return {
        "shape": list(array.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def dependency_versions(distributions: Sequence[str]) -> dict[str, str | None]:
    """Record installed versions, explicitly naming absent optional dependencies."""
    result: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def runtime_identity(*, dependencies: Sequence[str]) -> dict[str, Any]:
    """Describe the interpreter, platform, and relevant installed distributions."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependency_versions(dependencies),
    }


def source_identities(
    sources: Mapping[str, Path], *, repository_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Bind every maintained source file that defines an execution path."""
    return {
        name: file_identity(path, relative_to=repository_root)
        for name, path in sorted(sources.items())
    }


def repository_specification_identities(repository_root: Path) -> list[dict[str, Any]]:
    """Bind the project metadata and resolved lock used by the checkout."""
    candidates = (repository_root / "pyproject.toml", repository_root / "uv.lock")
    return input_file_identities(
        (path for path in candidates if path.is_file()), relative_to=repository_root
    )


def analysis_provenance(
    *,
    command: str,
    input_paths: Iterable[Path],
    sources: Mapping[str, Path],
    repository_root: Path | None = None,
    dependencies: Sequence[str] = ("numpy",),
) -> dict[str, Any]:
    """Build complete provenance for one analysis execution and its input bytes."""
    provenance_source = Path(__file__)
    bound_sources = dict(sources)
    bound_sources["ipmnist_provenance"] = provenance_source
    return {
        "schema": "asi.ipmnist.analysis_provenance.v1",
        "command": command,
        "runtime": runtime_identity(dependencies=dependencies),
        "sources": source_identities(bound_sources, repository_root=repository_root),
        "environment_specifications": (
            repository_specification_identities(repository_root)
            if repository_root is not None
            else []
        ),
        "inputs": input_file_identities(input_paths, relative_to=repository_root),
    }
