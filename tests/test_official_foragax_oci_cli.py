"""Unit coverage for alberta_framework.benchmarks._official_foragax_oci_cli.

Tests the shared OCI command parser: subcommand routing, required
arguments, defaults, and abbreviation rejection.
"""

import pytest

from alberta_framework.benchmarks._official_foragax_oci_cli import build_parser


def _parser():
    return build_parser(
        base_image_default="ghcr.io/example/base:latest",
        source_commit_default="abc123",
        uv_binary_sha256_default="0" * 64,
    )


def test_prepare_command_parses() -> None:
    args = _parser().parse_args(
        [
            "prepare",
            "--source-archive",
            "src.tar",
            "--source-archive-sha256",
            "a" * 64,
            "--dependency-lock",
            "uv.lock",
            "--dependency-lock-sha256",
            "b" * 64,
            "--source-tree-git-sha1",
            "c" * 40,
            "--uv-binary",
            "/usr/bin/uv",
            "--uv-cache-archive",
            "cache.tar",
            "--uv-cache-archive-sha256",
            "d" * 64,
            "--debian-bundle",
            "deb.tar",
            "--debian-manifest",
            "manifest.txt",
            "--output-context",
            "out/",
        ]
    )
    assert args.command == "prepare"
    assert args.base_image == "ghcr.io/example/base:latest"  # default applied
    assert args.source_commit == "abc123"


def test_prepare_requires_mandatory() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["prepare"])  # missing required args


def test_no_abbreviation() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["prep"])  # 'prep' is an abbreviation of prepare


def test_unknown_subcommand() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["bogus"])
