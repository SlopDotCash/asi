"""Dependency-light parser for the official Foragax OCI console command.

The installed console-script wrapper imports this module to render help before
loading the source-attested implementation.  Keep the parser in one place so
that packaging-safe help and real command execution cannot drift apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser(
    *,
    base_image_default: str,
    source_commit_default: str,
    uv_binary_sha256_default: str,
) -> argparse.ArgumentParser:
    """Build the shared OCI command parser.

    The defaults remain owned by the attested implementation. Help rendering
    does not display or consume them, allowing the lazy wrapper to use inert
    placeholders rather than import that implementation from a hard-linked
    wheel.
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("--source-archive", type=Path, required=True)
    prepare.add_argument("--source-archive-sha256", required=True)
    prepare.add_argument("--dependency-lock", type=Path, required=True)
    prepare.add_argument("--dependency-lock-sha256", required=True)
    prepare.add_argument(
        "--source-commit",
        default=source_commit_default,
    )
    prepare.add_argument("--source-tree-git-sha1", required=True)
    prepare.add_argument("--base-image", default=base_image_default)
    prepare.add_argument("--uv-binary", type=Path, required=True)
    prepare.add_argument(
        "--uv-binary-sha256",
        default=uv_binary_sha256_default,
    )
    prepare.add_argument("--uv-cache-archive", type=Path, required=True)
    prepare.add_argument("--uv-cache-archive-sha256", required=True)
    prepare.add_argument("--debian-bundle", type=Path, required=True)
    prepare.add_argument("--debian-manifest", type=Path, required=True)
    prepare.add_argument("--output-context", type=Path, required=True)
    cache = subparsers.add_parser("archive-cache", allow_abbrev=False)
    cache.add_argument("--cache-root", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--context", type=Path, required=True)
    build.add_argument("--tag", required=True)
    build.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    inspect = subparsers.add_parser("inspect", allow_abbrev=False)
    inspect.add_argument("--context", type=Path, required=True)
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    probe = subparsers.add_parser("cpu-probe", allow_abbrev=False)
    probe.add_argument("--image-id", required=True)
    probe.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    launch = subparsers.add_parser("emit-launch", allow_abbrev=False)
    launch.add_argument("--image-id", required=True)
    launch.add_argument("--entrypoint", required=True)
    launch.add_argument("--config", required=True)
    launch.add_argument("--index", required=True)
    launch.add_argument("--max-steps", type=int)
    launch.add_argument("--gpu", action="store_true")
    launch.add_argument("--driver-host-path")
    launch.add_argument("--driver-container-path")
    launch.add_argument("--device-path", action="append")
    launch.add_argument("--device-index", action="append", type=int)
    launch.add_argument("--cuda-wheel-library-path", action="append")
    launch.add_argument("--driver-user-library-path", action="append")
    launch.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    qualify = subparsers.add_parser("qualify", allow_abbrev=False)
    qualify.add_argument("--first", type=Path, required=True)
    qualify.add_argument("--second", type=Path, required=True)
    qualify.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    qualify.add_argument("--image-id", required=True)
    qualify.add_argument("--runtime-profile-id", required=True)
    qualify.add_argument("--effective-seed", type=int, required=True)
    qualify.add_argument("--steps", type=int, required=True)
    qualify.add_argument("--config-sha256", required=True)
    qualify.add_argument("--source-archive-sha256", required=True)
    qualify.add_argument("--workload-identity", type=Path, required=True)
    qualify.add_argument("--environment-profile-sha256", required=True)
    qualify.add_argument("--output", type=Path, required=True)
    return parser


__all__ = ["build_parser"]
