from __future__ import annotations

import dataclasses
import hashlib
import os
import pickle
import secrets
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as materialization,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_seed_transport as seed_transport,
)


def _git(root: Path, *arguments: str) -> bytes:
    git_executable = "/usr/bin/git"
    if not Path(git_executable).is_file():
        git_executable = "/bin/git"
    completed = subprocess.run(
        [git_executable, "-c", "core.hooksPath=/dev/null", *arguments],
        cwd=root,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": os.fspath(root / ".hermetic-home-does-not-exist"),
            "LC_ALL": "C",
            "PATH": os.defpath,
            "XDG_CONFIG_HOME": os.fspath(root / ".hermetic-xdg-does-not-exist"),
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=10.0,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed.stdout


def _write(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def _repository(
    root: Path,
    *,
    include_symlink: bool = False,
    include_submodule: bool = False,
) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test User")
    _write(root / "README.txt", b"miniature external source\n")
    _write(root / "bin" / "tool", b"#!/bin/sh\nexit 0\n", 0o755)
    _write(root / "src" / "main.py", b"VALUE = 1\n")
    if include_symlink:
        (root / "linked").symlink_to("README.txt")
    _git(root, "add", "--all")
    if include_submodule:
        fake_commit = "1" * 40
        _git(
            root,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{fake_commit},vendor/nested",
        )
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity(
    root: Path,
    *,
    excluded_gitlinks: tuple[materialization.GitlinkPin, ...] = (),
    portable_path_aliases: tuple[materialization.PortablePathAliasPin, ...] = (),
) -> materialization.ExternalCheckoutIdentity:
    original = (root / "src" / "main.py").read_bytes()
    derived = original.replace(b"1", b"2")
    return materialization.ExternalCheckoutIdentity(
        repository_id="miniature_external",
        canonical_url="https://example.invalid/miniature-external",
        commit_git_sha1=_git(root, "rev-parse", "HEAD").strip().decode("ascii"),
        tree_git_sha1=_git(root, "rev-parse", "HEAD^{tree}").strip().decode("ascii"),
        archive_sha256=_sha256(b"synthetic archive identity"),
        archive_size_bytes=123,
        transport_schema_version="test.miniature_transport.v1",
        transport_descriptor_sha256=_sha256(b"miniature transport descriptor"),
        source_transforms=(
            materialization.SourceTransformPin(
                path="src/main.py",
                upstream_size_bytes=len(original),
                upstream_sha256=_sha256(original),
                derived_size_bytes=len(derived),
                derived_sha256=_sha256(derived),
            ),
        ),
        excluded_gitlinks=excluded_gitlinks,
        portable_path_aliases=portable_path_aliases,
    )


def _derive(
    identity: materialization.ExternalCheckoutIdentity,
) -> materialization._DeriveSources:
    def derive(sources: dict[str, bytes]) -> materialization._DerivedSourceSet:
        assert set(sources) == {"src/main.py"}
        return materialization._DerivedSourceSet(
            sources={"src/main.py": sources["src/main.py"].replace(b"1", b"2")},
            transport_schema_version=identity.transport_schema_version,
            transport_descriptor_sha256=identity.transport_descriptor_sha256,
        )

    return derive


def _materialize(
    root: Path,
    destination: Path,
    identity: materialization.ExternalCheckoutIdentity | None = None,
) -> materialization.ExternalMaterialization:
    identity = _identity(root) if identity is None else identity
    return materialization._materialize_external_checkout_with_identity(
        root,
        destination,
        identity,
        _derive(identity),
    )


_GITLINK_PATH = "vendor"
_GITLINK_COMMIT = "1" * 40
_PORTABLE_ALIAS_PATHS = (
    "aliases/Artifact.bin",
    "aliases/artifact.bin",
)


def _add_gitlink(
    root: Path,
    *,
    path: str = _GITLINK_PATH,
    commit: str = _GITLINK_COMMIT,
    include_gitmodules: bool = True,
    placeholder: str = "absent",
) -> materialization.ExternalCheckoutIdentity:
    if include_gitmodules:
        _write(
            root / ".gitmodules",
            (
                f'[submodule "{path}"]\n\tpath = {path}\n'
                "\turl = https://example.invalid/excluded.git\n"
            ).encode("ascii"),
        )
        _git(root, "add", ".gitmodules")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{commit},{path}")
    _git(root, "commit", "-q", "-m", "add excluded gitlink")

    placeholder_path = root / path
    if placeholder == "empty":
        placeholder_path.mkdir(parents=True)
    elif placeholder == "nonempty":
        placeholder_path.mkdir(parents=True)
        _write(placeholder_path / "payload.py", b"raise RuntimeError('must not execute')\n")
    elif placeholder == "symlink":
        placeholder_path.symlink_to("README.txt")
    elif placeholder == "fifo":
        os.mkfifo(placeholder_path)
    elif placeholder == "hardlink":
        outside = root.parent / f"{root.name}-hardlink-source"
        outside.write_bytes(b"not a directory")
        os.link(outside, placeholder_path)
    elif placeholder != "absent":
        raise AssertionError(f"unknown placeholder fixture: {placeholder}")
    return _identity(
        root,
        excluded_gitlinks=(materialization.GitlinkPin(path=path, commit_git_sha1=commit),),
    )


def _add_portable_alias_pair(root: Path) -> materialization.ExternalCheckoutIdentity:
    raw = b"same bytes, intentionally distinct inodes\n"
    for path in _PORTABLE_ALIAS_PATHS:
        _write(root / path, raw)
    _git(root, "add", "--all")
    _git(root, "commit", "-q", "-m", "add exact portable alias pair")
    blob_git_sha1 = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    return _identity(
        root,
        portable_path_aliases=tuple(
            materialization.PortablePathAliasPin(
                path=path,
                blob_git_sha1=blob_git_sha1,
            )
            for path in _PORTABLE_ALIAS_PATHS
        ),
    )


def _recanonicalize(payload: dict[str, Any]) -> bytes:
    without_digest = dict(payload)
    without_digest.pop("payload_sha256", None)
    payload["payload_sha256"] = _sha256(materialization._canonical_json(without_digest))
    return materialization._canonical_json(payload)


@pytest.mark.unit
def test_generic_materializer_publishes_complete_canonical_tree(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    destination = tmp_path / "derived"

    result = _materialize(root, destination)
    manifest = result.manifest()

    assert result.destination == destination
    assert result.manifest_path == (
        destination / materialization.EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME
    )
    assert result.manifest_path.read_bytes() == result.manifest_bytes
    assert _sha256(result.manifest_bytes) == result.manifest_sha256
    assert (destination / "src" / "main.py").read_bytes() == b"VALUE = 2\n"
    assert (destination / "README.txt").read_bytes() == b"miniature external source\n"
    assert (destination / "bin" / "tool").read_bytes() == b"#!/bin/sh\nexit 0\n"
    assert not (destination / ".git").exists()
    assert stat.S_IMODE((destination / "bin" / "tool").stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "README.txt").stat().st_mode) == 0o644
    assert manifest["source_tree"]["tracked_entry_count"] == 3
    assert manifest["source_tree"]["materialized_regular_file_count"] == 3
    assert manifest["source_tree"]["excluded_gitlink_count"] == 0
    assert manifest["source_tree"]["portable_path_alias_count"] == 0
    assert manifest["source_tree"]["excluded_gitlinks"] == []
    assert manifest["source_tree"]["portable_path_aliases"] == []
    assert manifest["source_tree"]["source_closure_bound"] is True
    assert [item["path"] for item in manifest["source_tree"]["files"]] == [
        "README.txt",
        "bin/tool",
        "src/main.py",
    ]
    assert [item["path"] for item in manifest["source_tree"]["files"] if item["transformed"]] == [
        "src/main.py"
    ]
    assert manifest["checkout_attestation"] == {
        "archive_bytes_verified": False,
        "archive_identity_binding_only": True,
        "clean_worktree_verified": True,
        "commit_verified": True,
        "every_tracked_regular_blob_verified": True,
        "exact_gitlinks_verified": True,
        "gitlink_content_copied": False,
        "gitlink_content_executed": False,
        "gitlink_content_imported": False,
        "gitlink_content_initialized": False,
        "gitlink_placeholders_absent_or_empty_verified": True,
        "portable_path_aliases_have_distinct_inodes": True,
        "portable_path_aliases_verified": True,
        "tree_verified": True,
    }
    assert all(value is False for value in manifest["claims"].values())
    assert (
        materialization.verify_external_materialization_tree(
            destination,
            result.manifest_bytes,
            expected_manifest_sha256=result.manifest_sha256,
        )
        == manifest
    )


@pytest.mark.unit
@pytest.mark.parametrize("placeholder", ["absent", "empty"])
def test_exact_gitlink_is_tree_bound_but_omitted_from_materialization(
    tmp_path: Path,
    placeholder: str,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _add_gitlink(root, placeholder=placeholder)
    destination = tmp_path / "derived"

    result = _materialize(root, destination, identity)
    manifest = result.manifest()

    assert (destination / ".gitmodules").read_bytes() == (root / ".gitmodules").read_bytes()
    assert not (destination / _GITLINK_PATH).exists()
    assert manifest["source_tree"]["tracked_entry_count"] == 5
    assert manifest["source_tree"]["materialized_regular_file_count"] == 4
    assert manifest["source_tree"]["excluded_gitlink_count"] == 1
    assert manifest["source_tree"]["gitlink_content_included"] is False
    assert manifest["source_tree"]["excluded_gitlinks"] == [
        {
            "commit_git_sha1": _GITLINK_COMMIT,
            "content_materialized": False,
            "git_mode": "160000",
            "path": _GITLINK_PATH,
        }
    ]
    raw_tree = _git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    parsed_tree = materialization._parse_tree(raw_tree, identity.excluded_gitlinks)
    assert (_GITLINK_PATH, "160000", _GITLINK_COMMIT) in parsed_tree
    assert materialization._git_tree_sha1(parsed_tree) == identity.tree_git_sha1
    assert (
        materialization.verify_external_materialization_tree(
            destination,
            result.manifest_bytes,
            expected_manifest_sha256=result.manifest_sha256,
        )
        == manifest
    )


@pytest.mark.unit
@pytest.mark.parametrize("placeholder", ["nonempty", "symlink", "fifo", "hardlink"])
def test_gitlink_placeholder_must_be_absent_or_exactly_empty(
    tmp_path: Path,
    placeholder: str,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _add_gitlink(root, placeholder=placeholder)
    destination = tmp_path / "derived"

    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="gitlink placeholder",
    ):
        _materialize(root, destination, identity)
    assert not destination.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "reader",
    ["git_metadata", "checkout_inspection", "source_copy", "materialized_replay"],
)
def test_regular_file_readers_fail_promptly_on_preopen_fifo_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    target = source / "target.bin"
    target.write_bytes(b"data")
    target_stat = target.stat()
    source_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    destination_descriptor = os.open(
        destination,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    original_open = os.open
    swapped = False

    def swap_to_fifo_before_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "target.bin" and not swapped:
            assert dir_fd is not None
            assert flags & os.O_NONBLOCK
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(path, 0o600, dir_fd=dir_fd)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_to_fifo_before_open)
    record = materialization._GitTreeFile(
        path="target.bin",
        git_mode="100644",
        blob_git_sha1=hashlib.sha1(b"blob 4\0data").hexdigest(),
        upstream_size_bytes=4,
        upstream_sha256=_sha256(b"data"),
        source_device=target_stat.st_dev,
        source_inode=target_stat.st_ino,
    )
    started = time.monotonic()
    try:
        with pytest.raises(materialization.ExternalMaterializationError):
            if reader == "git_metadata":
                materialization._read_relative_bytes(
                    source_descriptor,
                    ("target.bin",),
                    maximum_bytes=1024,
                )
            elif reader == "checkout_inspection":
                materialization._read_exact_file(
                    source_descriptor,
                    "target.bin",
                    "100644",
                    record.blob_git_sha1,
                    capture=False,
                )
            elif reader == "source_copy":
                materialization._copy_exact_file_at(
                    source_descriptor,
                    destination_descriptor,
                    record,
                )
            else:
                materialization._hash_materialized_file_at(
                    source_descriptor,
                    "target.bin",
                )
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    assert swapped is True
    assert time.monotonic() - started < 1.0


@pytest.mark.unit
@pytest.mark.parametrize("case", ["missing", "extra", "wrong_commit", "wrong_path"])
def test_gitlink_inventory_must_match_the_exact_identity(tmp_path: Path, case: str) -> None:
    root = _repository(tmp_path / "checkout")
    if case == "missing":
        identity = _identity(
            root,
            excluded_gitlinks=(
                materialization.GitlinkPin(
                    path=_GITLINK_PATH,
                    commit_git_sha1=_GITLINK_COMMIT,
                ),
            ),
        )
        expected_error = "lacks expected gitlinks"
    else:
        exact_identity = _add_gitlink(root)
        if case == "extra":
            identity = dataclasses.replace(exact_identity, excluded_gitlinks=())
            expected_error = "unexpected gitlink"
        elif case == "wrong_commit":
            identity = dataclasses.replace(
                exact_identity,
                excluded_gitlinks=(
                    materialization.GitlinkPin(
                        path=_GITLINK_PATH,
                        commit_git_sha1="2" * 40,
                    ),
                ),
            )
            expected_error = "gitlink commit"
        else:
            identity = dataclasses.replace(
                exact_identity,
                excluded_gitlinks=(
                    materialization.GitlinkPin(
                        path="wrong-path",
                        commit_git_sha1=_GITLINK_COMMIT,
                    ),
                ),
            )
            expected_error = "unexpected gitlink"

    with pytest.raises(materialization.ExternalMaterializationError, match=expected_error):
        _materialize(root, tmp_path / "derived", identity)


@pytest.mark.unit
def test_expected_gitlink_requires_a_tracked_regular_gitmodules_blob(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _add_gitlink(root, include_gitmodules=False)

    with pytest.raises(materialization.ExternalMaterializationError, match=r"\.gitmodules"):
        _materialize(root, tmp_path / "derived", identity)


@pytest.mark.unit
def test_identity_gitlinks_are_exact_sorted_and_nonoverlapping(tmp_path: Path) -> None:
    identity = _identity(_repository(tmp_path / "checkout"))
    unsorted = dataclasses.replace(
        identity,
        excluded_gitlinks=(
            materialization.GitlinkPin(path="z", commit_git_sha1="1" * 40),
            materialization.GitlinkPin(path="a", commit_git_sha1="2" * 40),
        ),
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="path-sorted"):
        materialization._identity_payload(unsorted)

    aliased = dataclasses.replace(
        identity,
        excluded_gitlinks=(
            materialization.GitlinkPin(path="Vendor", commit_git_sha1="1" * 40),
            materialization.GitlinkPin(path="vendor", commit_git_sha1="2" * 40),
        ),
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="aliases"):
        materialization._identity_payload(aliased)

    nested = dataclasses.replace(
        identity,
        excluded_gitlinks=(
            materialization.GitlinkPin(path="vendor", commit_git_sha1="1" * 40),
            materialization.GitlinkPin(path="vendor/nested", commit_git_sha1="2" * 40),
        ),
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="contain"):
        materialization._identity_payload(nested)


@pytest.mark.unit
def test_exact_portable_alias_pair_is_blob_bound_and_materialized_separately(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _add_portable_alias_pair(root)
    result = _materialize(root, tmp_path / "derived", identity)
    manifest = result.manifest()

    source_stats = [(root / path).stat() for path in _PORTABLE_ALIAS_PATHS]
    output_stats = [(result.destination / path).stat() for path in _PORTABLE_ALIAS_PATHS]
    assert len({item.st_ino for item in source_stats}) == 2
    assert len({item.st_ino for item in output_stats}) == 2
    assert all(item.st_nlink == 1 for item in (*source_stats, *output_stats))
    assert {(result.destination / path).read_bytes() for path in _PORTABLE_ALIAS_PATHS} == {
        b"same bytes, intentionally distinct inodes\n"
    }
    assert manifest["source_tree"]["portable_path_alias_count"] == 2
    assert manifest["source_tree"]["portable_path_aliases"] == [
        {
            "materialized_as_distinct_path": True,
            "path": pin.path,
            "upstream_blob_git_sha1": pin.blob_git_sha1,
        }
        for pin in identity.portable_path_aliases
    ]
    assert any("case-sensitive Linux" in limitation for limitation in manifest["limitations"])
    raw_tree = _git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    parsed_tree = materialization._parse_tree(
        raw_tree,
        identity.excluded_gitlinks,
        identity.portable_path_aliases,
    )
    assert (
        materialization._git_tree_sha1(parsed_tree, identity.portable_path_aliases)
        == identity.tree_git_sha1
    )


@pytest.mark.unit
@pytest.mark.parametrize("case", ["missing", "wrong_blob", "extra"])
def test_portable_alias_exception_must_match_exact_paths_and_blobs(
    tmp_path: Path,
    case: str,
) -> None:
    root = _repository(tmp_path / "checkout")
    exact_identity = _add_portable_alias_pair(root)
    if case == "missing":
        changed = dataclasses.replace(exact_identity, portable_path_aliases=())
    elif case == "wrong_blob":
        changed = dataclasses.replace(
            exact_identity,
            portable_path_aliases=(
                exact_identity.portable_path_aliases[0],
                dataclasses.replace(
                    exact_identity.portable_path_aliases[1],
                    blob_git_sha1="0" * 40,
                ),
            ),
        )
    else:
        changed = dataclasses.replace(
            exact_identity,
            portable_path_aliases=(
                materialization.PortablePathAliasPin(
                    path="aliases/ARTIFACT.bin",
                    blob_git_sha1=exact_identity.portable_path_aliases[0].blob_git_sha1,
                ),
                *exact_identity.portable_path_aliases,
            ),
        )

    with pytest.raises(materialization.ExternalMaterializationError, match="portable path aliases"):
        _materialize(root, tmp_path / "derived", changed)


@pytest.mark.unit
@pytest.mark.parametrize(
    "records",
    [
        (
            ("Directory/one.bin", "100644", "1" * 40),
            ("directory/two.bin", "100644", "2" * 40),
        ),
        (
            ("Node", "100644", "1" * 40),
            ("node/child.bin", "100644", "2" * 40),
        ),
    ],
)
def test_portable_alias_exception_rejects_ancestor_and_file_directory_collisions(
    records: tuple[tuple[str, str, str], ...],
) -> None:
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="portable path .*aliases",
    ):
        materialization._git_tree_sha1(records)


@pytest.mark.unit
def test_case_insensitive_destination_alias_collapse_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _add_portable_alias_pair(root)
    original_copy = materialization._copy_exact_file_at

    def collapse_second_alias(
        source_root_descriptor: int,
        destination_root_descriptor: int,
        record: materialization._GitTreeFile,
    ) -> str:
        if record.path != _PORTABLE_ALIAS_PATHS[1]:
            return original_copy(
                source_root_descriptor,
                destination_root_descriptor,
                record,
            )
        first_parts = Path(_PORTABLE_ALIAS_PATHS[0]).parts
        second_parts = Path(_PORTABLE_ALIAS_PATHS[1]).parts
        first_parent = materialization._open_relative_directory(
            destination_root_descriptor,
            first_parts[:-1],
        )
        second_parent = materialization._open_relative_directory(
            destination_root_descriptor,
            second_parts[:-1],
        )
        try:
            os.link(
                first_parts[-1],
                second_parts[-1],
                src_dir_fd=first_parent,
                dst_dir_fd=second_parent,
                follow_symlinks=False,
            )
        finally:
            os.close(first_parent)
            os.close(second_parent)
        return record.upstream_sha256

    monkeypatch.setattr(materialization, "_copy_exact_file_at", collapse_second_alias)
    destination = tmp_path / "derived"
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="hardlinked|distinct inodes",
    ):
        _materialize(root, destination, identity)
    assert not destination.exists()


@pytest.mark.unit
def test_manifest_getter_is_frozen_byte_backed_and_detached(tmp_path: Path) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    first = result.manifest()
    first["claims"]["authority_granted"] = True
    first["source_tree"]["files"][0]["path"] = "changed"
    second = result.manifest()
    assert second["claims"]["authority_granted"] is False
    assert second["source_tree"]["files"][0]["path"] == "README.txt"


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["dirty", "untracked", "ignored"])
def test_materializer_rejects_nonclean_worktrees(tmp_path: Path, kind: str) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    if kind == "dirty":
        (root / "README.txt").write_bytes(b"changed\n")
    elif kind == "untracked":
        (root / "untracked.txt").write_bytes(b"untracked\n")
    else:
        (root / ".git" / "info" / "exclude").write_text("ignored.txt\n")
        (root / "ignored.txt").write_bytes(b"ignored\n")
    destination = tmp_path / "derived"
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="tracked worktree|dirty, untracked",
    ):
        materialization._materialize_external_checkout_with_identity(
            root,
            destination,
            identity,
            _derive(identity),
        )
    assert not destination.exists()


@pytest.mark.unit
def test_materializer_rejects_symlinks_and_unexpected_gitlinks(tmp_path: Path) -> None:
    symlink_root = _repository(tmp_path / "symlink", include_symlink=True)
    symlink_identity = _identity(symlink_root)
    with pytest.raises(materialization.ExternalMaterializationError, match="symlink"):
        materialization._materialize_external_checkout_with_identity(
            symlink_root,
            tmp_path / "symlink-derived",
            symlink_identity,
            _derive(symlink_identity),
        )

    submodule_root = _repository(tmp_path / "submodule", include_submodule=True)
    submodule_identity = _identity(submodule_root)
    with pytest.raises(materialization.ExternalMaterializationError, match="unexpected gitlink"):
        materialization._materialize_external_checkout_with_identity(
            submodule_root,
            tmp_path / "submodule-derived",
            submodule_identity,
            _derive(submodule_identity),
        )


@pytest.mark.unit
def test_path_traversal_and_duplicate_aliases_fail_before_checkout_access(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    pin = identity.source_transforms[0]
    for path in ("../src/main.py", "/src/main.py", "src/../main.py", "src\\main.py"):
        changed = dataclasses.replace(
            identity,
            source_transforms=(dataclasses.replace(pin, path=path),),
        )
        with pytest.raises(materialization.ExternalMaterializationError, match="path"):
            materialization._materialize_external_checkout_with_identity(
                root,
                tmp_path / f"derived-{_sha256(path.encode())[:8]}",
                changed,
                _derive(identity),
            )

    aliased = dataclasses.replace(
        identity,
        source_transforms=(
            dataclasses.replace(pin, path="SRC/main.py"),
            pin,
        ),
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="aliases"):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "aliased",
            aliased,
            _derive(identity),
        )

    raw_tree = b"\0".join(
        (
            b"100644 blob " + b"1" * 40 + b"\tA.txt",
            b"100644 blob " + b"2" * 40 + b"\ta.TXT",
            b"",
        )
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="aliases"):
        materialization._parse_tree(raw_tree)


@pytest.mark.unit
def test_excessive_path_component_depth_uses_materialization_error() -> None:
    path = "/".join("a" for _ in range(materialization._MAX_PATH_COMPONENTS + 1))
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="component-depth limit",
    ):
        materialization._validate_relative_path(path, context="deep fixture path")


@pytest.mark.unit
def test_destination_exists_and_atomic_race_never_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path / "checkout")
    destination = tmp_path / "derived"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(materialization.ExternalMaterializationError, match="already exists"):
        _materialize(root, destination)
    assert sentinel.read_bytes() == b"keep"

    race_destination = tmp_path / "race"
    original_rename = materialization._rename_no_replace_at

    def create_competing_destination(
        parent_descriptor: int,
        source_name: str,
        target_name: str,
    ) -> None:
        os.mkdir(target_name, dir_fd=parent_descriptor)
        descriptor = os.open(
            f"{target_name}/sentinel",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=parent_descriptor,
        )
        os.write(descriptor, b"competitor")
        os.close(descriptor)
        original_rename(parent_descriptor, source_name, target_name)

    monkeypatch.setattr(
        materialization,
        "_rename_no_replace_at",
        create_competing_destination,
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="already exists"):
        _materialize(root, race_destination)
    assert (race_destination / "sentinel").read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".alberta-materialize-*"))


@pytest.mark.unit
def test_destination_cannot_be_nested_within_checkout(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    destination = root / "derived"

    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="cannot be nested within the checkout",
    ):
        _materialize(root, destination)

    assert not destination.exists()
    assert not list(root.glob(".alberta-materialize-*"))


@pytest.mark.unit
def test_original_and_derived_hash_drift_fail_closed(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    pin = identity.source_transforms[0]
    original_drift = dataclasses.replace(
        identity,
        source_transforms=(dataclasses.replace(pin, upstream_sha256="0" * 64),),
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="source SHA-256"):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "original-drift",
            original_drift,
            _derive(original_drift),
        )

    derived_drift = dataclasses.replace(
        identity,
        source_transforms=(dataclasses.replace(pin, derived_sha256="0" * 64),),
    )
    with pytest.raises(materialization.ExternalMaterializationError, match="derived source"):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "derived-drift",
            derived_drift,
            _derive(derived_drift),
        )
    assert not (tmp_path / "original-drift").exists()
    assert not (tmp_path / "derived-drift").exists()


@pytest.mark.unit
def test_failure_cleans_only_unique_temp_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path / "checkout")
    destination = tmp_path / "derived"
    unrelated = tmp_path / ".derived.tmp-unrelated"
    unrelated.mkdir()
    (unrelated / "sentinel").write_bytes(b"keep")
    original_write = materialization._write_exact_bytes_at

    def fail_manifest(root_descriptor: int, path: str, raw: bytes, mode: int) -> None:
        if path == materialization.EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME:
            raise OSError("injected failure")
        original_write(root_descriptor, path, raw, mode)

    monkeypatch.setattr(materialization, "_write_exact_bytes_at", fail_manifest)
    with pytest.raises(materialization.ExternalMaterializationError, match="failed"):
        _materialize(root, destination)
    assert not destination.exists()
    assert (unrelated / "sentinel").read_bytes() == b"keep"
    assert not list(tmp_path.glob(".alberta-materialize-*"))
    assert (unrelated / "sentinel").read_bytes() == b"keep"


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["open", "fstat"])
def test_new_temp_directory_is_removed_if_anchoring_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    token = "ab" * 16
    expected_name = f".alberta-materialize-{token}"
    original_open = os.open
    original_fstat = os.fstat
    temp_descriptors: set[int] = set()

    def controlled_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == expected_name and dir_fd == parent_descriptor and failure == "open":
            raise OSError("injected temp open failure")
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == expected_name and dir_fd == parent_descriptor:
            temp_descriptors.add(descriptor)
        return descriptor

    def controlled_fstat(descriptor: int) -> os.stat_result:
        if failure == "fstat" and descriptor in temp_descriptors:
            raise OSError("injected temp fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(secrets, "token_hex", lambda _: token)
    monkeypatch.setattr(os, "open", controlled_open)
    monkeypatch.setattr(os, "fstat", controlled_fstat)
    try:
        with pytest.raises(
            materialization.ExternalMaterializationError,
            match="could not be anchored",
        ):
            materialization._create_anchored_temp_directory(parent_descriptor)
        assert os.listdir(parent_descriptor) == []
    finally:
        os.close(parent_descriptor)


@pytest.mark.unit
def test_derivation_exception_never_creates_a_destination(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    destination = tmp_path / "derived"

    def fail(_: dict[str, bytes]) -> materialization._DerivedSourceSet:
        raise RuntimeError("injected derivation failure")

    with pytest.raises(materialization.ExternalMaterializationError, match="derivation failed"):
        materialization._materialize_external_checkout_with_identity(
            root,
            destination,
            identity,
            fail,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".alberta-materialize-*"))


@pytest.mark.unit
def test_manifest_parser_denies_authority_even_with_recomputed_digests(
    tmp_path: Path,
) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    payload = result.manifest()
    payload["claims"]["authority_granted"] = True
    changed = _recanonicalize(payload)
    with pytest.raises(materialization.ExternalMaterializationError, match="authority denial"):
        materialization.parse_external_materialization_manifest(
            changed,
            expected_manifest_sha256=_sha256(changed),
        )

    numeric_alias = result.manifest()
    numeric_alias["checkout_attestation"]["commit_verified"] = 1
    numeric_alias["source_tree"]["tracked_entry_count"] = True
    changed_alias = _recanonicalize(numeric_alias)
    with pytest.raises(materialization.ExternalMaterializationError):
        materialization.parse_external_materialization_manifest(
            changed_alias,
            expected_manifest_sha256=_sha256(changed_alias),
        )


@pytest.mark.unit
def test_manifest_parser_rejects_path_aliases_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    payload = result.manifest()
    payload["source_tree"]["files"][1]["path"] = "readme.TXT"
    changed = _recanonicalize(payload)
    with pytest.raises(materialization.ExternalMaterializationError, match="aliases"):
        materialization.parse_external_materialization_manifest(
            changed,
            expected_manifest_sha256=_sha256(changed),
        )
    noncanonical = result.manifest_bytes + b"\n"
    with pytest.raises(materialization.ExternalMaterializationError):
        materialization.parse_external_materialization_manifest(
            noncanonical,
            expected_manifest_sha256=_sha256(noncanonical),
        )
    duplicate_keys = b'{"a":1,"a":2}'
    with pytest.raises(materialization.ExternalMaterializationError, match="duplicate key"):
        materialization.parse_external_materialization_manifest(
            duplicate_keys,
            expected_manifest_sha256=_sha256(duplicate_keys),
        )

    blob_drift = result.manifest()
    blob_drift["source_tree"]["files"][0]["upstream_blob_git_sha1"] = "0" * 40
    changed_blob = _recanonicalize(blob_drift)
    with pytest.raises(materialization.ExternalMaterializationError, match="reconstruct"):
        materialization.parse_external_materialization_manifest(
            changed_blob,
            expected_manifest_sha256=_sha256(changed_blob),
        )


@pytest.mark.unit
def test_verifier_rejects_extra_content_hash_mode_and_symlink(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")

    extra = _materialize(root, tmp_path / "extra")
    (extra.destination / "unexpected").write_bytes(b"extra")
    with pytest.raises(materialization.ExternalMaterializationError, match="contents"):
        materialization.verify_external_materialization_tree(
            extra.destination,
            extra.manifest_bytes,
            expected_manifest_sha256=extra.manifest_sha256,
        )

    drift = _materialize(root, tmp_path / "drift")
    (drift.destination / "README.txt").write_bytes(b"drift")
    with pytest.raises(materialization.ExternalMaterializationError, match="bytes"):
        materialization.verify_external_materialization_tree(
            drift.destination,
            drift.manifest_bytes,
            expected_manifest_sha256=drift.manifest_sha256,
        )

    mode = _materialize(root, tmp_path / "mode")
    (mode.destination / "README.txt").chmod(0o755)
    with pytest.raises(materialization.ExternalMaterializationError, match="mode"):
        materialization.verify_external_materialization_tree(
            mode.destination,
            mode.manifest_bytes,
            expected_manifest_sha256=mode.manifest_sha256,
        )

    linked = _materialize(root, tmp_path / "linked")
    (linked.destination / "README.txt").unlink()
    (linked.destination / "README.txt").symlink_to("src/main.py")
    with pytest.raises(materialization.ExternalMaterializationError, match="not regular"):
        materialization.verify_external_materialization_tree(
            linked.destination,
            linked.manifest_bytes,
            expected_manifest_sha256=linked.manifest_sha256,
        )


@pytest.mark.unit
def test_path_verifier_rechecks_the_named_root_inode_at_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    moved = tmp_path / "verified-then-moved"
    original_verify = materialization._verify_external_materialization_tree_fd

    def swap_name_after_verification(*args: object, **kwargs: object) -> object:
        manifest = original_verify(*args, **kwargs)  # type: ignore[arg-type]
        result.destination.rename(moved)
        result.destination.mkdir()
        return manifest

    monkeypatch.setattr(
        materialization,
        "_verify_external_materialization_tree_fd",
        swap_name_after_verification,
    )
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="materialized root anchor identity changed",
    ):
        materialization.verify_external_materialization_tree(
            result.destination,
            result.manifest_bytes,
            expected_manifest_sha256=result.manifest_sha256,
        )

    assert result.destination.is_dir()
    assert not result.manifest_path.exists()
    assert (moved / materialization.EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME).is_file()


@pytest.mark.unit
def test_retained_capability_is_live_only_inside_its_context(tmp_path: Path) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    manifest = result.manifest()

    with materialization.retain_verified_external_materialization_tree(
        result.destination,
        result.manifest_bytes,
        expected_manifest_sha256=result.manifest_sha256,
    ) as retained:
        assert isinstance(retained, materialization.RetainedExternalMaterializationTree)
        assert retained.closed is False
        proc_fd_path = retained.proc_fd_path
        assert proc_fd_path.startswith("/proc/self/fd/")
        assert Path(proc_fd_path, "README.txt").read_bytes() == b"miniature external source\n"
        assert retained.subprocess_pass_fds == (int(proc_fd_path.rsplit("/", 1)[1]),)
        assert retained.owner_pid == os.getpid()
        assert retained.reverify() == manifest

    assert retained.closed is True
    assert not Path(proc_fd_path).exists()
    with pytest.raises(materialization.ExternalMaterializationError, match="closed"):
        _ = retained.proc_fd_path


@pytest.mark.unit
def test_retained_capability_is_nonserializable_and_pid_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    serialized_manifest = pickle.loads(pickle.dumps(result.manifest()))
    assert serialized_manifest["claims"]["filesystem_capability_granted"] is False
    assert "/proc/self/fd/" not in result.manifest_bytes.decode("ascii")

    with materialization.retain_verified_external_materialization_tree(
        result.destination,
        result.manifest_bytes,
        expected_manifest_sha256=result.manifest_sha256,
    ) as retained:
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(retained)
        actual_pid = os.getpid()
        with monkeypatch.context() as local_patch:
            local_patch.setattr(os, "getpid", lambda: actual_pid + 1)
            with pytest.raises(materialization.ExternalMaterializationError, match="PID change"):
                _ = retained.proc_fd_path
        assert retained.closed is True


@pytest.mark.unit
def test_retained_capability_survives_name_replacement_and_reverify_is_explicit(
    tmp_path: Path,
) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    moved = tmp_path / "retained-inode"

    with materialization.retain_verified_external_materialization_tree(
        result.destination,
        result.manifest_bytes,
        expected_manifest_sha256=result.manifest_sha256,
    ) as retained:
        proc_fd_path = retained.proc_fd_path
        result.destination.rename(moved)
        result.destination.mkdir()
        assert retained.proc_fd_path == proc_fd_path
        assert retained.reverify() == result.manifest()
        assert Path(proc_fd_path, "README.txt").read_bytes() == b"miniature external source\n"

        (moved / "README.txt").write_bytes(b"mutated retained inode\n")
        with pytest.raises(materialization.ExternalMaterializationError, match="bytes"):
            retained.reverify()
        assert retained.closed is True


@pytest.mark.unit
def test_matched_retained_capability_rejects_a_generic_manifest(tmp_path: Path) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    with pytest.raises(materialization.ExternalMaterializationError, match="pinned matched-v3"):
        with materialization.retain_verified_matched_v3_external_materialization_tree(
            result.destination,
            result.manifest_bytes,
            expected_manifest_sha256=result.manifest_sha256,
        ):
            raise AssertionError("generic manifest unexpectedly granted a capability")


@pytest.mark.unit
def test_pinned_identity_is_exact_and_private_construction_mutation_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = materialization.pinned_external_checkout_identity()
    raw = materialization.canonical_pinned_external_checkout_identity_bytes()
    assert materialization.parse_pinned_external_checkout_identity(raw) == expected
    assert _sha256(raw) == materialization.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
    assert materialization.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256 == (
        "74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192"
    )
    assert expected.commit_git_sha1 == seed_transport.UPSTREAM_SOURCE_COMMIT
    assert expected.tree_git_sha1 == seed_transport.UPSTREAM_SOURCE_TREE_GIT_SHA1
    assert expected.archive_sha256 == seed_transport.UPSTREAM_SOURCE_ARCHIVE_SHA256
    assert expected.archive_size_bytes == seed_transport.UPSTREAM_SOURCE_ARCHIVE_SIZE_BYTES
    assert [item.path for item in expected.source_transforms] == list(seed_transport.SOURCE_PATHS)
    assert {item.path: item.upstream_sha256 for item in expected.source_transforms} == dict(
        seed_transport.UPSTREAM_SOURCE_SHA256_BY_PATH
    )
    assert {item.path: item.derived_sha256 for item in expected.source_transforms} == dict(
        seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH
    )
    assert {item.path: item.derived_size_bytes for item in expected.source_transforms} == {
        "src/continuing_main.py": 33_029,
        "src/problems/BaseProblem.py": 1_719,
        "src/problems/Foragax.py": 1_316,
        "src/rtu_ppo.py": 91_286,
    }
    assert expected.excluded_gitlinks == (
        materialization.GitlinkPin(
            path="continual-foragax-loss-of-plasticity",
            commit_git_sha1="8880f3f241ec441e584416b61b0579fca3bc1ef4",
        ),
    )
    assert expected.portable_path_aliases == (
        materialization.PortablePathAliasPin(
            path=(
                "experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/"
                "metrics/NTKRank_LOP_vs_NoLOP.png"
            ),
            blob_git_sha1="566e89612c822a72f39fa84f8f1c4ed65d1c2788",
        ),
        materialization.PortablePathAliasPin(
            path=(
                "experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/"
                "metrics/ntkrank_LOP_vs_NoLOP.png"
            ),
            blob_git_sha1="566e89612c822a72f39fa84f8f1c4ed65d1c2788",
        ),
    )

    monkeypatch.setattr(
        materialization,
        "_PINNED_IDENTITY_CONSTRUCTION",
        dataclasses.replace(expected, commit_git_sha1="0" * 40),
    )
    replayed = materialization.pinned_external_checkout_identity()
    assert replayed == expected

    monkeypatch.setattr(seed_transport, "UPSTREAM_SOURCE_SHA256_BY_PATH", {})
    monkeypatch.setattr(seed_transport, "EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH", {})
    assert materialization.pinned_external_checkout_identity() == expected

    for changed in (raw + b"\n", raw.replace(expected.commit_git_sha1.encode(), b"0" * 40)):
        with pytest.raises(materialization.ExternalMaterializationError):
            materialization.parse_pinned_external_checkout_identity(changed)


@pytest.mark.unit
def test_v2_contract_rejects_exact_legacy_v1_identity_manifest_and_filename(
    tmp_path: Path,
) -> None:
    identity_payload = materialization._identity_payload(
        materialization.pinned_external_checkout_identity()
    )
    identity_payload["schema_version"] = (
        materialization._V1_EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION
    )
    identity_payload.pop("excluded_gitlinks")
    identity_payload.pop("portable_path_aliases")
    legacy_identity_raw = materialization._canonical_json(identity_payload)
    assert _sha256(legacy_identity_raw) == (
        "5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"
    )
    with pytest.raises(materialization.ExternalMaterializationError):
        materialization.parse_pinned_external_checkout_identity(legacy_identity_raw)

    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    legacy_manifest = result.manifest()
    legacy_manifest["schema_version"] = materialization._V1_EXTERNAL_MATERIALIZATION_SCHEMA_VERSION
    legacy_manifest["status"] = "materialized_tracked_source_closure_unqualified"
    legacy_manifest["identity"]["schema_version"] = (
        materialization._V1_EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION
    )
    legacy_manifest["identity"].pop("excluded_gitlinks")
    legacy_manifest["identity"].pop("portable_path_aliases")
    legacy_manifest["identity_sha256"] = _sha256(
        materialization._canonical_json(legacy_manifest["identity"])
    )
    legacy_manifest["checkout_attestation"] = {
        "archive_bytes_verified": False,
        "archive_identity_binding_only": True,
        "clean_worktree_verified": True,
        "commit_verified": True,
        "every_tracked_blob_verified": True,
        "tree_verified": True,
    }
    legacy_source_tree = legacy_manifest["source_tree"]
    legacy_source_tree["scope"] = "complete_materialized_tracked_regular_tree"
    legacy_source_tree["submodules_included"] = False
    legacy_source_tree["tracked_file_count"] = legacy_source_tree.pop(
        "materialized_regular_file_count"
    )
    for key in (
        "excluded_gitlink_count",
        "excluded_gitlinks",
        "gitlink_content_included",
        "portable_path_alias_count",
        "portable_path_aliases",
        "tracked_entry_count",
    ):
        legacy_source_tree.pop(key)
    legacy_manifest["claims"].pop("filesystem_capability_granted")
    legacy_manifest["limitations"] = [
        "The archive identity is bound as provenance; archive bytes were not supplied or verified.",
        "Source closure covers only the complete regular-file tree at the pinned Git commit.",
        "Materialization does not import or execute the derived source tree.",
        "Runtime dependencies, capabilities, RNG traces, and result handling remain unqualified.",
        (
            "Named-path stability assumes no concurrent process with the "
            "materializing effective user ID mutates the checkout, staging "
            "namespace, destination parent, or published path."
        ),
        (
            "Returned destination and manifest paths can later be replaced or mutated by a "
            "process running as the same OS user."
        ),
        (
            "Path checks cover the qualified Linux materialization filesystem and common "
            "cross-platform aliases; they do not claim universal filesystem portability."
        ),
        "A valid manifest grants no execution, ingestion, scientific, or promotion authority.",
    ]
    legacy_raw = _recanonicalize(legacy_manifest)
    with pytest.raises(materialization.ExternalMaterializationError, match="schema version"):
        materialization.parse_external_materialization_manifest(
            legacy_raw,
            expected_manifest_sha256=_sha256(legacy_raw),
        )

    with pytest.raises(materialization.ExternalMaterializationError, match="reserved manifest"):
        materialization._validate_relative_path(
            materialization._V1_EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME,
            context="legacy manifest filename",
        )


@pytest.mark.unit
def test_production_manifest_parser_rejects_generic_identity(tmp_path: Path) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    with pytest.raises(materialization.ExternalMaterializationError, match="pinned matched-v3"):
        materialization.parse_matched_v3_external_materialization_manifest(
            result.manifest_bytes,
            expected_manifest_sha256=result.manifest_sha256,
        )


@pytest.mark.unit
def test_production_parser_rejects_incomplete_pinned_tree_inventory() -> None:
    identity = materialization.pinned_external_checkout_identity()
    files = (
        materialization._GitTreeFile(
            path=".gitmodules",
            git_mode="100644",
            blob_git_sha1="f" * 40,
            upstream_size_bytes=0,
            upstream_sha256=_sha256(b""),
        ),
        *(
            materialization._GitTreeFile(
                path=alias.path,
                git_mode="100644",
                blob_git_sha1=alias.blob_git_sha1,
                upstream_size_bytes=120_871,
                upstream_sha256="0" * 64,
            )
            for alias in identity.portable_path_aliases
        ),
        *(
            materialization._GitTreeFile(
                path=pin.path,
                git_mode="100644",
                blob_git_sha1=f"{index + 1:x}" * 40,
                upstream_size_bytes=pin.upstream_size_bytes,
                upstream_sha256=pin.upstream_sha256,
            )
            for index, pin in enumerate(identity.source_transforms)
        ),
    )
    derived = materialization._DerivedSourceSet(
        sources={},
        transport_schema_version=identity.transport_schema_version,
        transport_descriptor_sha256=identity.transport_descriptor_sha256,
    )
    raw = materialization._canonical_json(materialization._manifest(identity, files, derived))
    with pytest.raises(materialization.ExternalMaterializationError, match="reconstruct"):
        materialization.parse_matched_v3_external_materialization_manifest(
            raw,
            expected_manifest_sha256=_sha256(raw),
        )


@pytest.mark.unit
def test_wrong_commit_tree_and_exact_root_are_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    changed_identities = (
        ("commit", dataclasses.replace(identity, commit_git_sha1="0" * 40)),
        ("tree", dataclasses.replace(identity, tree_git_sha1="0" * 40)),
    )
    for context, changed in changed_identities:
        with pytest.raises(materialization.ExternalMaterializationError, match=context):
            materialization._materialize_external_checkout_with_identity(
                root,
                tmp_path / context,
                changed,
                _derive(changed),
            )
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="Git metadata directory",
    ):
        materialization._materialize_external_checkout_with_identity(
            root / "src",
            tmp_path / "nested",
            identity,
            _derive(identity),
        )


@pytest.mark.unit
def test_forged_untransformed_hashes_cannot_change_git_blob(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    result = _materialize(root, tmp_path / "derived")
    target = result.destination / "README.txt"
    original = target.read_bytes()
    changed = b"X" + original[1:]
    assert len(changed) == len(original)
    target.write_bytes(changed)

    payload = result.manifest()
    record = next(item for item in payload["source_tree"]["files"] if item["path"] == "README.txt")
    record["upstream_sha256"] = _sha256(changed)
    record["materialized_sha256"] = _sha256(changed)
    forged_raw = _recanonicalize(payload)
    result.manifest_path.write_bytes(forged_raw)

    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="untransformed materialized blob",
    ):
        materialization.verify_external_materialization_tree(
            result.destination,
            forged_raw,
            expected_manifest_sha256=_sha256(forged_raw),
        )


@pytest.mark.unit
@pytest.mark.parametrize("feature", ["fsmonitor", "filter", "hook", "alias"])
def test_external_execution_git_features_are_rejected_without_execution(
    tmp_path: Path,
    feature: str,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    sentinel = tmp_path / "executed"
    payload = tmp_path / "payload.sh"
    _write(
        payload,
        f"#!/bin/sh\nprintf executed > {sentinel}\nexit 0\n".encode(),
        0o755,
    )
    if feature == "fsmonitor":
        _git(root, "config", "core.fsmonitor", str(payload))
    elif feature == "filter":
        _git(root, "config", "filter.evil.smudge", str(payload))
    elif feature == "alias":
        _git(root, "config", "alias.evil", f"!{payload}")
    else:
        hook = root / ".git" / "hooks" / "post-checkout"
        _write(hook, payload.read_bytes(), 0o755)

    with pytest.raises(materialization.ExternalMaterializationError):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "derived",
            identity,
            _derive(identity),
        )
    assert not sentinel.exists()
    assert not (tmp_path / "derived").exists()


@pytest.mark.unit
def test_ambient_git_overrides_are_cleared_and_cannot_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    sentinel = tmp_path / "executed"
    payload = tmp_path / "payload.sh"
    _write(
        payload,
        f"#!/bin/sh\nprintf executed > {sentinel}\nexit 0\n".encode(),
        0o755,
    )
    hostile_config = tmp_path / "hostile.gitconfig"
    hostile_config.write_text(f"[core]\n\tfsmonitor = {payload}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile_config))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-worktree"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "wrong-objects"))
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(payload))

    result = materialization._materialize_external_checkout_with_identity(
        root,
        tmp_path / "derived",
        identity,
        _derive(identity),
    )
    assert result.destination.is_dir()
    assert not sentinel.exists()


@pytest.mark.unit
def test_git_inspection_uses_bounded_timeout_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    checkout = materialization._checkout_root(root)
    config_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    environment = materialization._HermeticGitEnvironment(
        {"PATH": os.defpath, "LC_ALL": "C"},
        config_descriptor,
    )
    observed: dict[str, object] = {}

    def bounded_runner(command: object, **kwargs: object) -> tuple[int, bytes, bytes]:
        observed["command"] = command
        observed.update(kwargs)
        return 0, b"bounded\n", b""

    monkeypatch.setattr(materialization, "_run_bounded_process", bounded_runner)
    try:
        assert (
            materialization._git(
                checkout,
                environment,
                ["rev-parse", "HEAD"],
                maximum_stdout_bytes=1234,
            )
            == b"bounded\n"
        )
    finally:
        os.close(config_descriptor)
        materialization._close_checkout_anchor(checkout)

    assert observed["maximum_stdout_bytes"] == 1234
    assert observed["maximum_stderr_bytes"] == materialization._MAX_GIT_STDERR_BYTES
    assert observed["timeout_seconds"] == materialization._GIT_TIMEOUT_SECONDS
    assert observed["pass_fds"] == (
        checkout.root.descriptor,
        checkout.git.descriptor,
        config_descriptor,
    )
    command = observed["command"]
    assert isinstance(command, list)
    assert f"--git-dir=/proc/self/fd/{checkout.git.descriptor}" in command


@pytest.mark.unit
def test_bounded_process_rejects_excess_output(tmp_path: Path) -> None:
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="stdout exceeded its byte limit",
    ):
        materialization._run_bounded_process(
            [sys.executable, "-c", "import os; os.write(1, b'x' * 4096)"],
            cwd=os.fspath(tmp_path),
            environment={"PATH": os.defpath, "LC_ALL": "C"},
            pass_fds=(),
            maximum_stdout_bytes=1024,
            maximum_stderr_bytes=1024,
            timeout_seconds=5.0,
        )


@pytest.mark.unit
def test_bounded_process_enforces_wall_time(tmp_path: Path) -> None:
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="exceeded its time limit",
    ):
        materialization._run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=os.fspath(tmp_path),
            environment={"PATH": os.defpath, "LC_ALL": "C"},
            pass_fds=(),
            maximum_stdout_bytes=1024,
            maximum_stderr_bytes=1024,
            timeout_seconds=0.1,
        )


@pytest.mark.unit
def test_bounded_process_kills_pipe_inheriting_descendant_after_leader_exit(
    tmp_path: Path,
) -> None:
    script = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"
    )
    started = time.monotonic()
    returncode, stdout, stderr = materialization._run_bounded_process(
        [sys.executable, "-c", script],
        cwd=os.fspath(tmp_path),
        environment={"PATH": os.defpath, "LC_ALL": "C"},
        pass_fds=(),
        maximum_stdout_bytes=1024,
        maximum_stderr_bytes=1024,
        timeout_seconds=0.1,
    )
    assert time.monotonic() - started < 1.0
    assert (returncode, stdout, stderr) == (0, b"", b"")


@pytest.mark.unit
def test_bounded_process_does_not_wait_for_setsid_descendant_pipe(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "escaped-child-pid"
    child_script = (
        "import os,sys,time; os.setsid(); "
        "open(sys.argv[1],'w',encoding='ascii').write(str(os.getpid())); time.sleep(60)"
    )
    leader_script = (
        "import os,subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1]]); "
        "exec('while not os.path.exists(sys.argv[1]):\\n time.sleep(0.001)')"
    )
    started = time.monotonic()
    try:
        returncode, stdout, stderr = materialization._run_bounded_process(
            [sys.executable, "-c", leader_script, os.fspath(marker), child_script],
            cwd=os.fspath(tmp_path),
            environment={"PATH": os.defpath, "LC_ALL": "C"},
            pass_fds=(),
            maximum_stdout_bytes=1024,
            maximum_stderr_bytes=1024,
            timeout_seconds=0.2,
        )
        assert time.monotonic() - started < 1.0
        assert (returncode, stdout, stderr) == (0, b"", b"")
    finally:
        if marker.is_file():
            try:
                os.kill(int(marker.read_text(encoding="ascii")), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.unit
def test_invalid_destination_does_not_leak_checkout_descriptors(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    baseline = len(os.listdir("/proc/self/fd"))
    for _ in range(4):
        with pytest.raises(materialization.ExternalMaterializationError):
            materialization._materialize_external_checkout_with_identity(
                root,
                "invalid\0destination",
                identity,
                _derive(identity),
            )
    assert len(os.listdir("/proc/self/fd")) == baseline


@pytest.mark.unit
def test_copy_parent_failure_does_not_leak_source_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    raw = b"source bytes"
    (source / "file").write_bytes(raw)
    source_root = os.open(source, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    record = materialization._GitTreeFile(
        path="file",
        git_mode="100644",
        blob_git_sha1=hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest(),
        upstream_size_bytes=len(raw),
        upstream_sha256=_sha256(raw),
    )

    def fail_parent(root_descriptor: int, path: str) -> tuple[int, str]:
        del root_descriptor, path
        raise OSError("injected destination-parent failure")

    monkeypatch.setattr(materialization, "_output_parent_descriptor", fail_parent)
    baseline = len(os.listdir("/proc/self/fd"))
    try:
        with pytest.raises(OSError, match="destination-parent failure"):
            materialization._copy_exact_file_at(source_root, source_root, record)
        assert len(os.listdir("/proc/self/fd")) == baseline
    finally:
        os.close(source_root)


@pytest.mark.unit
@pytest.mark.parametrize(
    "metadata",
    [
        "replace",
        "alternates",
        "shallow",
        "promisor",
        "partial",
        "modules",
        "commondir",
    ],
)
def test_unsupported_repository_metadata_is_rejected(
    tmp_path: Path,
    metadata: str,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    if metadata == "replace":
        path = root / ".git" / "refs" / "replace" / identity.commit_git_sha1
        _write(path, ("0" * 40 + "\n").encode())
    elif metadata == "alternates":
        _write(root / ".git" / "objects" / "info" / "alternates", b"/tmp\n")
    elif metadata == "shallow":
        _write(root / ".git" / "shallow", b"")
    elif metadata == "promisor":
        _write(root / ".git" / "objects" / "pack" / "fake.promisor", b"")
    elif metadata == "partial":
        _git(root, "config", "remote.origin.promisor", "true")
    elif metadata == "modules":
        (root / ".git" / "modules").mkdir()
    else:
        _write(root / ".git" / "commondir", b".\n")

    expected = "linked-worktree common directory" if metadata == "commondir" else None
    with pytest.raises(materialization.ExternalMaterializationError, match=expected):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "derived",
            identity,
            _derive(identity),
        )


@pytest.mark.unit
def test_direct_git_directory_identity_is_retained_for_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    original_inspect = materialization._inspect_checkout

    def replace_git_name_before_inspection(
        checkout: materialization._CheckoutAnchor,
        expected: materialization.ExternalCheckoutIdentity,
        environment: materialization._HermeticGitEnvironment,
    ) -> object:
        (root / ".git").rename(root / ".git-original")
        (root / ".git").mkdir()
        return original_inspect(checkout, expected, environment)

    monkeypatch.setattr(
        materialization,
        "_inspect_checkout",
        replace_git_name_before_inspection,
    )
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match=r"checkout \.git anchor identity changed",
    ):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "derived",
            identity,
            _derive(identity),
        )

    assert (root / ".git-original" / "HEAD").is_file()
    assert not (tmp_path / "derived").exists()


@pytest.mark.unit
def test_tracked_gitmodules_without_expected_gitlink_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    _write(root / ".gitmodules", b'[submodule "x"]\n\tpath = x\n')
    _git(root, "add", ".gitmodules")
    _git(root, "commit", "-q", "-m", "submodule metadata")
    identity = _identity(root)
    with pytest.raises(materialization.ExternalMaterializationError, match=r"\.gitmodules"):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "derived",
            identity,
            _derive(identity),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "src/name.",
        "src/name ",
        "CON",
        "con.txt",
        "PRN.md",
        "AUX",
        "NUL.json",
        "COM1.py",
        "LPT9.log",
        "src/name:stream",
        "src/a<b",
        "src/a>b",
        "src/a|b",
        "src/a?b",
        "src/a*b",
        "．ｇｉｔ/config",
    ],
)
def test_nonportable_components_are_rejected(tmp_path: Path, path: str) -> None:
    root = _repository(tmp_path / "checkout")
    identity = _identity(root)
    pin = dataclasses.replace(identity.source_transforms[0], path=path)
    changed = dataclasses.replace(identity, source_transforms=(pin,))
    with pytest.raises(materialization.ExternalMaterializationError):
        materialization._materialize_external_checkout_with_identity(
            root,
            tmp_path / "derived",
            changed,
            _derive(identity),
        )


@pytest.mark.unit
def test_source_and_materialized_hardlinks_are_rejected(tmp_path: Path) -> None:
    source_root = _repository(tmp_path / "source-checkout")
    os.link(source_root / "README.txt", tmp_path / "outside-source-link")
    source_identity = _identity(source_root)
    with pytest.raises(materialization.ExternalMaterializationError, match="hardlinked"):
        materialization._materialize_external_checkout_with_identity(
            source_root,
            tmp_path / "source-derived",
            source_identity,
            _derive(source_identity),
        )

    root = _repository(tmp_path / "checkout")
    result = _materialize(root, tmp_path / "derived")
    os.link(result.destination / "README.txt", tmp_path / "outside-derived-link")
    with pytest.raises(materialization.ExternalMaterializationError, match="hardlinked"):
        materialization.verify_external_materialization_tree(
            result.destination,
            result.manifest_bytes,
            expected_manifest_sha256=result.manifest_sha256,
        )


@pytest.mark.unit
def test_materialized_directory_modes_are_normalized_and_verified(tmp_path: Path) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    manifest = result.manifest()
    assert manifest["source_tree"]["normalized_directory_mode"] == "0755"
    assert stat.S_IMODE(result.destination.stat().st_mode) == 0o755
    assert stat.S_IMODE((result.destination / "src").stat().st_mode) == 0o755
    (result.destination / "src").chmod(0o700)
    with pytest.raises(materialization.ExternalMaterializationError, match="directory mode"):
        materialization.verify_external_materialization_tree(
            result.destination,
            result.manifest_bytes,
            expected_manifest_sha256=result.manifest_sha256,
        )


@pytest.mark.unit
def test_symlink_destination_ancestor_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path / "checkout")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(materialization.ExternalMaterializationError, match="ancestor"):
        _materialize(root, linked_parent / "derived")


@pytest.mark.unit
def test_parent_swap_cannot_redirect_publication_or_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    publish_parent = tmp_path / "publish-parent"
    publish_parent.mkdir()
    moved_parent = tmp_path / "publish-parent-moved"
    attacker_parent = tmp_path / "attacker"
    attacker_parent.mkdir()
    sentinel = attacker_parent / "sentinel"
    sentinel.write_bytes(b"keep")
    original = materialization._recheck_directory_anchor
    destination_checks = 0

    def swap_on_final_check(
        anchor: materialization._DirectoryAnchor,
        *,
        context: str,
    ) -> None:
        nonlocal destination_checks
        if context == "destination parent":
            destination_checks += 1
            if destination_checks == 2:
                publish_parent.rename(moved_parent)
                publish_parent.symlink_to(attacker_parent, target_is_directory=True)
        original(anchor, context=context)

    monkeypatch.setattr(
        materialization,
        "_recheck_directory_anchor",
        swap_on_final_check,
    )
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="destination parent anchor",
    ):
        _materialize(root, publish_parent / "derived")
    assert sentinel.read_bytes() == b"keep"
    assert not (attacker_parent / "derived").exists()
    assert not (moved_parent / "derived").exists()
    assert not list(moved_parent.glob(".alberta-materialize-*"))


@pytest.mark.unit
def test_temp_name_swap_never_deletes_or_publishes_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    original = materialization._recheck_temp_name
    replacement_name: str | None = None

    def swap_temp(
        parent_descriptor: int,
        name: str,
        descriptor: int,
        identity: tuple[int, int],
    ) -> None:
        nonlocal replacement_name
        replacement_name = name
        moved = f"{name}.moved"
        os.rename(
            name,
            moved,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.mkdir(name, dir_fd=parent_descriptor)
        sentinel_descriptor = os.open(
            f"{name}/sentinel",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=parent_descriptor,
        )
        os.write(sentinel_descriptor, b"replacement")
        os.close(sentinel_descriptor)
        original(parent_descriptor, name, descriptor, identity)

    monkeypatch.setattr(materialization, "_recheck_temp_name", swap_temp)
    with pytest.raises(materialization.ExternalMaterializationError, match="anchor changed"):
        _materialize(root, tmp_path / "derived")
    assert replacement_name is not None
    assert (tmp_path / replacement_name / "sentinel").read_bytes() == b"replacement"
    assert not (tmp_path / "derived").exists()


@pytest.mark.unit
def test_published_name_must_retain_the_verified_staging_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    destination = tmp_path / "derived"
    moved_name = "derived-moved"
    original_rename = materialization._rename_no_replace_at

    def replace_name_after_rename(parent: int, source: str, target: str) -> None:
        original_rename(parent, source, target)
        os.rename(
            target,
            moved_name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
        )
        os.mkdir(target, dir_fd=parent)
        descriptor = os.open(
            f"{target}/sentinel",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=parent,
        )
        os.write(descriptor, b"replacement")
        os.close(descriptor)

    monkeypatch.setattr(
        materialization,
        "_rename_no_replace_at",
        replace_name_after_rename,
    )
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="published destination anchor changed",
    ):
        _materialize(root, destination)

    assert (destination / "sentinel").read_bytes() == b"replacement"
    assert (
        tmp_path / moved_name / materialization.EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME
    ).is_file()


@pytest.mark.unit
def test_published_inode_is_reverified_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    destination = tmp_path / "derived"
    original_rename = materialization._rename_no_replace_at

    def mutate_after_rename(parent: int, source: str, target: str) -> None:
        original_rename(parent, source, target)
        descriptor = os.open(
            f"{target}/README.txt",
            os.O_WRONLY | os.O_TRUNC,
            dir_fd=parent,
        )
        os.write(descriptor, b"changed after rename\n")
        os.close(descriptor)

    monkeypatch.setattr(materialization, "_rename_no_replace_at", mutate_after_rename)
    with pytest.raises(
        materialization.ExternalMaterializationError,
        match="materialized bytes do not match",
    ):
        _materialize(root, destination)

    assert destination.is_dir()
    assert (destination / "README.txt").read_bytes() == b"changed after rename\n"


@pytest.mark.unit
def test_source_and_staging_verification_bracket_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_inspect = materialization._inspect_checkout
    original_normalize = materialization._normalize_and_fsync_directory_tree
    original_verify = materialization._verify_external_materialization_tree_fd
    original_rename = materialization._rename_no_replace_at

    def inspect(*args: object, **kwargs: object) -> object:
        events.append("source")
        return original_inspect(*args, **kwargs)  # type: ignore[arg-type]

    def normalize(descriptor: int) -> None:
        events.append("directory_fsync")
        original_normalize(descriptor)

    def verify(*args: object, **kwargs: object) -> object:
        events.append("temp_verify")
        return original_verify(*args, **kwargs)  # type: ignore[arg-type]

    def rename(parent: int, source: str, destination: str) -> None:
        events.append("rename")
        original_rename(parent, source, destination)

    monkeypatch.setattr(materialization, "_inspect_checkout", inspect)
    monkeypatch.setattr(materialization, "_normalize_and_fsync_directory_tree", normalize)
    monkeypatch.setattr(materialization, "_verify_external_materialization_tree_fd", verify)
    monkeypatch.setattr(materialization, "_rename_no_replace_at", rename)
    _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    assert events == [
        "source",
        "source",
        "directory_fsync",
        "temp_verify",
        "rename",
        "temp_verify",
    ]


@pytest.mark.unit
def test_parent_fsync_occurs_after_rename_and_failure_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    events: list[str] = []
    original_rename = materialization._rename_no_replace_at
    original_parent_fsync = materialization._fsync_parent_after_publish

    def rename(parent: int, source: str, destination: str) -> None:
        events.append("rename")
        original_rename(parent, source, destination)

    def parent_fsync(parent: int) -> None:
        events.append("parent_fsync")
        original_parent_fsync(parent)

    monkeypatch.setattr(materialization, "_rename_no_replace_at", rename)
    monkeypatch.setattr(materialization, "_fsync_parent_after_publish", parent_fsync)
    _materialize(root, tmp_path / "derived")
    assert events == ["rename", "parent_fsync"]

    def fail_parent_fsync(_: int) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(materialization, "_fsync_parent_after_publish", fail_parent_fsync)
    with pytest.raises(materialization.ExternalMaterializationError, match="publication completed"):
        _materialize(root, tmp_path / "published-but-undurable")
    assert (tmp_path / "published-but-undurable").is_dir()


@pytest.mark.unit
def test_primary_error_is_preserved_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path / "checkout")
    original_write = materialization._write_exact_bytes_at

    def fail_write(root_descriptor: int, path: str, raw: bytes, mode: int) -> None:
        if path == materialization.EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME:
            raise OSError("primary write failure")
        original_write(root_descriptor, path, raw, mode)

    def fail_cleanup(_: int) -> None:
        raise OSError("secondary cleanup failure")

    monkeypatch.setattr(materialization, "_write_exact_bytes_at", fail_write)
    monkeypatch.setattr(materialization, "_safe_remove_open_directory", fail_cleanup)
    with pytest.raises(materialization.ExternalMaterializationError) as captured:
        _materialize(root, tmp_path / "derived")
    assert "primary write failure" in str(captured.value)
    assert any("secondary cleanup failure" in note for note in captured.value.__notes__)


@pytest.mark.unit
def test_limitations_are_frozen_and_recursive_json_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _materialize(_repository(tmp_path / "checkout"), tmp_path / "derived")
    expected = result.manifest()["limitations"]
    monkeypatch.setattr(
        materialization,
        "_MANIFEST_LIMITATIONS_CONSTRUCTION",
        ("mutated emitter state",),
    )
    assert result.manifest()["limitations"] == expected
    assert materialization._frozen_manifest_limitations() == expected
    assert any("later" in limitation and "same OS user" in limitation for limitation in expected)

    recursive = b"[" * 2_000 + b"0" + b"]" * 2_000
    with pytest.raises(materialization.ExternalMaterializationError):
        materialization.parse_external_materialization_manifest(
            recursive,
            expected_manifest_sha256=_sha256(recursive),
        )


@pytest.mark.unit
def test_production_derivation_uses_frozen_descriptor_without_ambient_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {path: f"synthetic:{path}".encode() for path in seed_transport.SOURCE_PATHS}
    expected_sources = {path: f"derived:{path}".encode() for path in seed_transport.SOURCE_PATHS}
    observed: dict[str, object] = {}

    def replay(
        received_sources: dict[str, bytes],
        descriptor_raw: bytes,
    ) -> seed_transport.DerivedExternalSeedTransport:
        observed["sources"] = received_sources
        observed["descriptor_raw"] = descriptor_raw
        return seed_transport.DerivedExternalSeedTransport(
            sources=expected_sources,
            source_sha256_by_path={path: _sha256(raw) for path, raw in expected_sources.items()},
            descriptor={},
            descriptor_sha256=(seed_transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256),
        )

    monkeypatch.setattr(
        seed_transport,
        "replay_matched_v3_external_seed_transport",
        replay,
    )
    derived = materialization._production_derive(sources)

    assert observed == {
        "sources": sources,
        "descriptor_raw": (
            seed_transport.canonical_matched_v3_external_seed_transport_descriptor_bytes()
        ),
    }
    assert derived.transport_schema_version == seed_transport.SCHEMA_VERSION
    assert derived.transport_descriptor_sha256 == (
        seed_transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
    )
    assert dict(derived.sources) == expected_sources
