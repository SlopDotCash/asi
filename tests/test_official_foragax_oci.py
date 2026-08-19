from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks import official_foragax_oci as oci
from alberta_framework.benchmarks._official_foragax_image_helper import (
    harden_tree,
    tree_inventory,
)


def _sha(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_git_archive(
    path: Path,
    *,
    commit: str,
    lock: bytes,
) -> None:
    files = {
        "pyproject.toml": b"[project]\nname='fixture'\n",
        "src/continuing_main.py": b"print('continuing')\n",
        "src/rtu_ppo.py": b"print('ppo')\n",
        "uv.lock": lock,
    }
    with path.open("wb") as handle, tarfile.open(
        fileobj=handle,
        mode="w:",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": commit},
    ) as archive:
        for name in (
            "pyproject.toml",
            "src",
            "src/continuing_main.py",
            "src/rtu_ppo.py",
            "uv.lock",
        ):
            payload = files.get(name)
            info = tarfile.TarInfo(name)
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 1_700_000_000
            if payload is None:
                info.type = tarfile.DIRTYPE
                info.mode = 0o775
                archive.addfile(info)
            else:
                info.type = tarfile.REGTYPE
                info.mode = 0o664
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def _lock_bytes() -> bytes:
    packages = {
        "continual-foragax": "0.55.0",
        "imageio-ffmpeg": "0.6.0",
        "jax": "0.9.0.1",
        "jax-cuda12-pjrt": "0.9.0.1",
        "jax-cuda12-plugin": "0.9.0.1",
        "jaxlib": "0.9.0.1",
    }
    sections = [
        "version = 1\nrevision = 3\n",
        (
            "[[package]]\n"
            'name = "continual-foragax-agents"\n'
            'version = "0.0.0"\n'
            'source = { editable = "." }\n'
        ),
    ]
    for name, version in packages.items():
        sections.append(
            "\n".join(
                (
                    "[[package]]",
                    f'name = "{name}"',
                    f'version = "{version}"',
                    'source = { registry = "https://pypi.org/simple" }',
                    (
                        "wheels = [{ url = "
                        f'"https://example.invalid/{name}.whl", '
                        f'hash = "sha256:{"a" * 64}", size = 1 }}]'
                    ),
                    "",
                )
            )
        )
    return "\n".join(sections).encode()


def _cache_archive(tmp_path: Path) -> tuple[Path, str]:
    cache = tmp_path / "cache"
    (cache / "wheels").mkdir(parents=True)
    (cache / "wheels" / "one.whl").write_bytes(b"wheel")
    archive = tmp_path / "uv-cache.tar"
    digest = oci.create_regular_cache_archive(cache, archive)
    return archive, digest


def _debian_fixture(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "debs"
    bundle.mkdir()
    packages = []
    for name in sorted(
        (
            "ca-certificates",
            "libpython3.12-minimal",
            "python3.12",
            "python3.12-minimal",
            "python3.12-venv",
        )
    ):
        filename = f"{name}_1_amd64.deb"
        contents = name.encode()
        (bundle / filename).write_bytes(contents)
        packages.append(
            {
                "architecture": "amd64",
                "filename": filename,
                "package": name,
                "sha256": _sha(contents),
                "version": "1",
            }
        )
    manifest = {
        "architecture": "amd64",
        "base_image": oci._AUDITED_BASE_IMAGE,
        "installed_file_inventory_sha256": "b" * 64,
        "packages": packages,
        "python_executable": "/usr/bin/python3.12",
        "python_executable_sha256": "c" * 64,
        "repositories": ["https://snapshot.ubuntu.com/ubuntu/20260731T000000Z"],
        "schema_version": oci.DEBIAN_BUNDLE_SCHEMA,
    }
    path = tmp_path / "debian-manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return bundle, path


def test_prepare_build_context_is_offline_noneditable_and_path_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_bytes()
    lock_path = tmp_path / "uv.lock"
    lock_path.write_bytes(lock)
    source = tmp_path / "source.tar"
    _write_git_archive(
        source,
        commit=oci.OFFICIAL_FORAGAX_AUDIT_COMMIT,
        lock=lock,
    )
    source_tree_git_sha1 = oci._source_archive_identity(
        source,
        source_commit=oci.OFFICIAL_FORAGAX_AUDIT_COMMIT,
        dependency_lock=lock,
    )["source_tree_git_sha1"]
    monkeypatch.setattr(
        oci,
        "_AUDITED_SOURCE_TREE_GIT_SHA1",
        source_tree_git_sha1,
    )
    uv = tmp_path / "uv"
    uv.write_bytes(b"fixture uv")
    uv_sha = _sha(uv.read_bytes())
    monkeypatch.setattr(oci, "UV_BINARY_SHA256", uv_sha)
    cache, cache_sha = _cache_archive(tmp_path)
    bundle, debian_manifest = _debian_fixture(tmp_path)
    context = tmp_path / "context"
    prepared = oci.prepare_build_context(
        oci.OciBuildInputs(
            source_archive=source,
            source_archive_sha256=oci._sha256(source),
            dependency_lock=lock_path,
            dependency_lock_sha256=_sha(lock),
            source_commit=oci.OFFICIAL_FORAGAX_AUDIT_COMMIT,
            source_tree_git_sha1=source_tree_git_sha1,
            base_image=oci._AUDITED_BASE_IMAGE,
            uv_binary=uv,
            uv_binary_sha256=uv_sha,
            uv_cache_archive=cache,
            uv_cache_archive_sha256=cache_sha,
            debian_bundle=bundle,
            debian_manifest=debian_manifest,
            output_context=context,
        )
    )

    dockerfile = (context / "Dockerfile").read_text()
    attestation = (context / "build-attestation.json").read_text()
    assert "--network=none" in dockerfile
    assert "UV_OFFLINE=1" in dockerfile
    assert "--offline --frozen --no-dev --group cuda" in dockerfile
    assert "--no-install-project" in dockerfile
    assert "apt-get" not in dockerfile
    assert "editable" not in dockerfile
    assert str(tmp_path) not in dockerfile
    assert str(tmp_path) not in attestation
    expected_created = datetime.fromtimestamp(
        prepared.build_spec["source_commit_timestamp"],
        tz=UTC,
    ).isoformat().replace("+00:00", "Z")
    assert (
        f'org.opencontainers.image.created="{expected_created}"'
        in dockerfile
    )
    assert 'org.opencontainers.image.created="1970-01-01T00:00:00Z"' not in (
        dockerfile
    )
    assert prepared.build_spec["source_commit"] == oci.OFFICIAL_FORAGAX_AUDIT_COMMIT
    assert prepared.build_spec["uv_cache_archive_sha256"] == cache_sha
    build_command = oci.docker_build_command(
        prepared,
        image_tag="alberta-foragax:test",
    )
    assert "--network=none" in build_command
    assert (
        f"SOURCE_DATE_EPOCH={prepared.build_spec['source_commit_timestamp']}"
        in build_command
    )
    assert "--provenance=false" in build_command
    assert "--sbom=false" in build_command


def test_cache_archive_and_hardening_preserve_executable_role(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    executable = root / "executable"
    executable.write_bytes(b"exe")
    executable.chmod(0o755)
    regular = root / "regular"
    regular.write_bytes(b"data")
    regular.chmod(0o644)
    harden_tree(root)
    assert executable.stat().st_mode & 0o777 == 0o555
    assert regular.stat().st_mode & 0o777 == 0o444
    assert root.stat().st_mode & 0o777 == 0o555
    inventory = tree_inventory(root, recorded_root="/opt/test")
    assert inventory["tree_sha256"] == oci._json_sha256(
        {
            "entries": inventory["entries"],
            "hash_scheme": oci.TREE_HASH_SCHEME,
        }
    )


def test_emit_launch_command_is_digest_only_and_exactly_isolated() -> None:
    image_id = "sha256:" + "1" * 64
    command = oci.emit_launch_command(
        image_id=image_id,
        entrypoint="src/continuing_main.py",
        config_path="/opt/continual-foragax-agents/config.json",
        index_expression="0:2",
        gpu=False,
        max_steps=100,
    )
    assert image_id in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--user=65532:65532" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert (
        "--mount=type=tmpfs,destination=/tmp/src,"
        "tmpfs-mode=0555,tmpfs-size=1048576"
    ) in command
    assert not any(argument.startswith("--tmpfs=/tmp:") for argument in command)
    assert "--env=JAX_ENABLE_COMPILATION_CACHE=false" in command
    assert "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1" in command
    assert "--env=NVIDIA_VISIBLE_DEVICES=void" in command
    assert "--env=PYTHONPATH=" in command
    assert "--python-flag=-I" in command
    assert "--python-flag=-B" in command
    assert "--trusted-python-path-mode=isolated-runpy-prepend-v1" in command
    assert "--gpus" not in command
    with pytest.raises(oci.OfficialForagaxOciError, match="image-config"):
        oci.emit_launch_command(
            image_id="repository@sha256:" + "1" * 64,
            entrypoint="src/continuing_main.py",
            config_path="/opt/continual-foragax-agents/config.json",
            index_expression="0",
            gpu=False,
        )


def test_inspected_config_digest_supports_legacy_and_containerd_stores() -> None:
    manifest_digest = "sha256:" + "1" * 64
    config_digest = "sha256:" + "2" * 64
    assert oci._inspected_config_digest({"Id": config_digest}) == config_digest
    inspected = {
        "Descriptor": {
            "annotations": {"config.digest": config_digest},
            "digest": manifest_digest,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "size": 1234,
        },
        "Id": manifest_digest,
    }
    assert oci._inspected_config_digest(inspected) == config_digest
    inspected["Descriptor"]["digest"] = config_digest
    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="descriptor identity",
    ):
        oci._inspected_config_digest(inspected)


def test_native_inventory_rejects_float_alias_for_integer_mode() -> None:
    path = "opt/alberta-runtime/bin/tool"
    rootfs = {
        "entries": [
            {
                "mode": 0o555,
                "path": path,
                "sha256": "a" * 64,
                "size": 1,
                "type": "file",
            }
        ]
    }
    native_entries = [
        {
            "mode": 365.0,
            "path": "bin/tool",
            "sha256": "a" * 64,
            "size": 1,
            "type": "file",
        }
    ]
    native = {
        "entries": native_entries,
        "hash_scheme": oci.TREE_HASH_SCHEME,
        "root": oci.RUNTIME_ROOT.as_posix(),
        "schema_version": oci.NATIVE_INVENTORY_SCHEMA,
        "tree_sha256": oci._json_sha256(
            {
                "entries": native_entries,
                "hash_scheme": oci.TREE_HASH_SCHEME,
            }
        ),
    }

    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="native inventory entry differs from rootfs",
    ):
        oci._verify_native_inventory(rootfs, native)


def test_saved_config_blob_is_extracted_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_bytes = b'{"config":{},"created":"2026-07-17T00:04:15Z"}'
    digest = "sha256:" + _sha(config_bytes)
    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream,
        mode="w:",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        member = tarfile.TarInfo(digest.removeprefix("sha256:") + ".json")
        member.size = len(config_bytes)
        archive.addfile(member, io.BytesIO(config_bytes))

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(stream.getvalue())
            self.stderr = io.BytesIO()

        def kill(self) -> None:
            raise AssertionError("valid image-save stream must not be killed")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        oci.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    original_extractfile = tarfile.TarFile.extractfile
    extraction_count = 0

    def counted_extractfile(
        archive: tarfile.TarFile,
        member: tarfile.TarInfo,
    ) -> object:
        nonlocal extraction_count
        extraction_count += 1
        return original_extractfile(archive, member)

    monkeypatch.setattr(
        tarfile.TarFile,
        "extractfile",
        counted_extractfile,
    )
    config_sha, config = oci._saved_config_identity(
        "fixture",
        config_digest=digest,
        docker=Path("/usr/bin/docker"),
    )
    assert config_sha == digest.removeprefix("sha256:")
    assert config == {"config": {}, "created": "2026-07-17T00:04:15Z"}
    assert extraction_count == 1


def _sqlite_bytes(
    path: Path,
    *,
    touch_same_value: bool,
    seed: int = 7,
    index: int = 7,
) -> bytes:
    connection = sqlite3.connect(path)
    columns = ", ".join(
        f'"{name}" {"INTEGER" if name in {"seed", "id"} else "TEXT"}'
        for name in oci.OFFICIAL_FORAGAX_RESULTS_DB_COLUMNS
    )
    connection.execute(
        f'CREATE TABLE "_metadata_" ({columns})'
    )
    connection.execute(
        'INSERT INTO "_metadata_" (seed, id) VALUES (?, ?)',
        (seed, index),
    )
    connection.commit()
    if touch_same_value:
        connection.execute(
            'UPDATE "_metadata_" SET "seed" = "seed" WHERE "id" = ?',
            (index,),
        )
        connection.commit()
    connection.close()
    return path.read_bytes()


def _v4_archive(path: Path, payloads: list[tuple[str, bytes]]) -> None:
    with path.open("wb") as handle, tarfile.open(
        fileobj=handle,
        mode="w:",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for name, contents in payloads:
            info = tarfile.TarInfo(name)
            info.type = tarfile.REGTYPE
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))


def _workload_identity(
    *,
    seed: int,
    steps: int,
    root: str = "official-results/result",
) -> dict[str, Any]:
    return {
        "backend": {
            "kind": "cpu",
            "launcher_contract": "oci-read-only-stdout-tar-v4",
            "runtime_arguments": [
                "--env=JAX_PLATFORM_NAME=cpu",
                "--env=JAX_PLATFORMS=cpu",
                "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
            ],
        },
        "configuration": {
            "agent": "DQN",
            "config_path": "/opt/continual-foragax-agents/config.json",
            "config_sha256": "2" * 64,
            "entrypoint_family": "continuing",
            "problem": "Foragax",
        },
        "entrypoint": {
            "family": "continuing",
            "path": "src/continuing_main.py",
            "sha256": "6" * 64,
        },
        "invocation": {
            "expected_result_env_steps": steps,
            "index_expression": str(seed),
            "indices": [seed],
            "max_steps_argument": steps,
            "members": [
                {
                    "content_policy": "strict_npz",
                    "path": f"{root}/data/{seed}.npz",
                    "role": "result_npz",
                },
                {
                    "content_policy": "sqlite_foragax_metadata_v1",
                    "path": f"{root}/results.db",
                    "role": "auxiliary",
                },
                {
                    "content_policy": "bounded_utf8_log",
                    "path": "stdout.log",
                    "role": "stdout_log",
                },
                {
                    "content_policy": "bounded_utf8_diagnostic",
                    "path": "stderr.log",
                    "role": "stderr_log",
                },
            ],
        },
        "run": {
            "applied_seed_offset": 0,
            "applied_seed_offset_source": "nested",
            "effective_seed": seed,
            "index": seed,
            "nested_seed_offset": 0,
            "stored_seed": seed,
            "top_level_seed_offset": 0,
        },
        "schema_version": oci.QUALIFICATION_WORKLOAD_SCHEMA,
    }


def test_two_run_qualification_binds_exact_npz_rewards_and_sqlite(
    tmp_path: Path,
) -> None:
    rewards = np.arange(12, dtype=np.float16)
    npz_stream = io.BytesIO()
    np.savez(npz_stream, rewards=rewards)
    npz = npz_stream.getvalue()
    first_db = _sqlite_bytes(tmp_path / "first.db", touch_same_value=False)
    second_db = _sqlite_bytes(tmp_path / "second.db", touch_same_value=True)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    base = [
        ("official-results/result/data/7.npz", npz),
        ("official-results/result/results.db", first_db),
    ]
    _v4_archive(
        first,
        [*base, ("stdout.log", b"time=1\n"), ("stderr.log", b"")],
    )
    _v4_archive(
        second,
        [
            base[0],
            ("official-results/result/results.db", second_db),
            ("stdout.log", b"time=2\n"),
            ("stderr.log", b""),
        ],
    )
    result = oci.qualify_v4_runs(
        first,
        second,
        backend="cpu",
        image_id="sha256:" + "1" * 64,
        runtime_profile_id="fixture",
        effective_seed=7,
        steps=12,
        config_sha256="2" * 64,
        source_archive_sha256="3" * 64,
        workload_identity=_workload_identity(seed=7, steps=12),
        environment_profile_sha256="5" * 64,
    )
    qualification = result["qualification"]
    assert qualification["artifact_sha256"] == _sha(npz)
    assert qualification["rewards_sha256"] == _sha(
        rewards.tobytes(order="C")
    )
    sqlite_evidence = result["evidence"]["sqlite"]
    assert sqlite_evidence["classification"] in {
        "canonical_equal_raw_diff",
        "raw_and_canonical_equal",
    }
    assert result["evidence"]["diagnostic_logs"][0]["first_sha256"] != result[
        "evidence"
    ]["diagnostic_logs"][0]["second_sha256"]

    changed_stream = io.BytesIO()
    np.savez(changed_stream, rewards=rewards + np.float16(1))
    changed = tmp_path / "changed.tar"
    _v4_archive(
        changed,
        [
            ("official-results/result/data/7.npz", changed_stream.getvalue()),
            ("official-results/result/results.db", second_db),
            ("stdout.log", b"time=2\n"),
            ("stderr.log", b""),
        ],
    )
    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="deterministic member bytes differ",
    ):
        oci.qualify_v4_runs(
            first,
            changed,
            backend="cpu",
            image_id="sha256:" + "1" * 64,
            runtime_profile_id="fixture",
            effective_seed=7,
            steps=12,
            config_sha256="2" * 64,
            source_archive_sha256="3" * 64,
            workload_identity=_workload_identity(seed=7, steps=12),
            environment_profile_sha256="5" * 64,
        )


def test_two_run_qualification_rejects_false_seed_and_horizon_bindings(
    tmp_path: Path,
) -> None:
    npz_stream = io.BytesIO()
    np.savez(npz_stream, rewards=np.arange(4, dtype=np.float16))
    npz = npz_stream.getvalue()
    correct_db = _sqlite_bytes(
        tmp_path / "correct.db",
        touch_same_value=False,
    )
    wrong_db = _sqlite_bytes(
        tmp_path / "wrong.db",
        touch_same_value=False,
        seed=8,
        index=8,
    )
    correct = tmp_path / "correct.tar"
    second_correct = tmp_path / "second-correct.tar"
    wrong_seed = tmp_path / "wrong-seed.tar"
    second_wrong_seed = tmp_path / "second-wrong-seed.tar"
    root = "official-results/result"
    correct_members = [
        (f"{root}/data/7.npz", npz),
        (f"{root}/results.db", correct_db),
        ("stdout.log", b"diagnostic\n"),
        ("stderr.log", b""),
    ]
    wrong_seed_members = [
        (f"{root}/data/7.npz", npz),
        (f"{root}/results.db", wrong_db),
        ("stdout.log", b"diagnostic\n"),
        ("stderr.log", b""),
    ]
    _v4_archive(correct, correct_members)
    _v4_archive(second_correct, correct_members)
    _v4_archive(wrong_seed, wrong_seed_members)
    _v4_archive(second_wrong_seed, wrong_seed_members)
    bindings = {
        "backend": "cpu",
        "image_id": "sha256:" + "1" * 64,
        "runtime_profile_id": "fixture",
        "effective_seed": 7,
        "config_sha256": "2" * 64,
        "source_archive_sha256": "3" * 64,
        "environment_profile_sha256": "5" * 64,
    }

    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="one-value-per-step",
    ):
        oci.qualify_v4_runs(
            correct,
            second_correct,
            steps=5,
            workload_identity=_workload_identity(seed=7, steps=5),
            **bindings,
        )
    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="seed/id row differs",
    ):
        oci.qualify_v4_runs(
            wrong_seed,
            second_wrong_seed,
            steps=4,
            workload_identity=_workload_identity(seed=7, steps=4),
            **bindings,
        )


def test_qualification_rejects_reduced_sqlite_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "reduced.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        'CREATE TABLE "_metadata_" ("seed" INTEGER, "id" INTEGER)'
    )
    connection.execute(
        'INSERT INTO "_metadata_" (seed, id) VALUES (7, 7)'
    )
    connection.commit()
    connection.close()

    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="columns differ",
    ):
        oci._canonical_sqlite(
            database_path.read_bytes(),
            expected_seed=7,
        )


def test_two_run_qualification_accepts_actual_nested_e138_result_root(
    tmp_path: Path,
) -> None:
    effective_seed = 2_000_001
    root = (
        "official-results/results/E138-two-biome-large/foragax/"
        "ForagaxTwoBiomeLarge-v1/9/DQN"
    )
    npz_stream = io.BytesIO()
    np.savez(npz_stream, rewards=np.arange(8, dtype=np.float16))
    npz = npz_stream.getvalue()
    first_db = _sqlite_bytes(
        tmp_path / "nested-first.db",
        touch_same_value=False,
        seed=effective_seed,
        index=effective_seed,
    )
    second_db = _sqlite_bytes(
        tmp_path / "nested-second.db",
        touch_same_value=True,
        seed=effective_seed,
        index=effective_seed,
    )
    first = tmp_path / "nested-first.tar"
    second = tmp_path / "nested-second.tar"
    _v4_archive(
        first,
        [
            (f"{root}/data/{effective_seed}.npz", npz),
            (f"{root}/results.db", first_db),
            ("stdout.log", b"first diagnostic\n"),
            ("stderr.log", b""),
        ],
    )
    _v4_archive(
        second,
        [
            (f"{root}/data/{effective_seed}.npz", npz),
            (f"{root}/results.db", second_db),
            ("stdout.log", b"second diagnostic\n"),
            ("stderr.log", b""),
        ],
    )

    result = oci.qualify_v4_runs(
        first,
        second,
        backend="cpu",
        image_id="sha256:" + "1" * 64,
        runtime_profile_id="fixture",
        effective_seed=effective_seed,
        steps=8,
        config_sha256="2" * 64,
        source_archive_sha256="3" * 64,
        workload_identity=_workload_identity(
            seed=effective_seed,
            steps=8,
            root=root,
        ),
        environment_profile_sha256="5" * 64,
    )

    scientific_paths = {
        member["path"] for member in result["evidence"]["member_payloads"]
    }
    assert scientific_paths == {
        f"{root}/data/{effective_seed}.npz",
        f"{root}/results.db",
    }


@pytest.mark.parametrize(
    "result_members",
    (
        (
            "official-results/result/data/8.npz",
            "official-results/result/results.db",
        ),
        (
            "official-results/result/data/7.npz",
            "official-results/result/data/8.npz",
            "official-results/result/results.db",
        ),
        (
            "official-results/first/data/7.npz",
            "official-results/second/data/7.npz",
            "official-results/first/results.db",
        ),
        (
            "official-results/result/data/7.npz",
            "official-results/other/results.db",
        ),
        (
            "official-results/result/data/7.npz",
            "official-results/result/results.db",
            "official-results/other/results.db",
        ),
        (
            "official-results/result/data/7.npz",
            "official-results/result/results.db",
            "official-results/result/unexpected.txt",
        ),
        (
            "official-results/result/data/../data/7.npz",
            "official-results/result/results.db",
        ),
        (
            "official-results/result/data/../../escape/7.npz",
            "official-results/result/results.db",
        ),
    ),
)
def test_two_run_qualification_rejects_ambiguous_or_aliased_result_layout(
    tmp_path: Path,
    result_members: tuple[str, ...],
) -> None:
    npz_stream = io.BytesIO()
    np.savez(npz_stream, rewards=np.arange(4, dtype=np.float16))
    npz = npz_stream.getvalue()
    database = _sqlite_bytes(
        tmp_path / "adversarial.db",
        touch_same_value=False,
    )
    payloads = [
        (path, npz if path.endswith(".npz") else database)
        for path in result_members
    ]
    payloads.extend(
        (("stdout.log", b"diagnostic\n"), ("stderr.log", b""))
    )
    first = tmp_path / "adversarial-first.tar"
    second = tmp_path / "adversarial-second.tar"
    _v4_archive(first, payloads)
    _v4_archive(second, payloads)

    with pytest.raises(oci.OfficialForagaxOciError):
        oci.qualify_v4_runs(
            first,
            second,
            backend="cpu",
            image_id="sha256:" + "1" * 64,
            runtime_profile_id="fixture",
            effective_seed=7,
            steps=4,
            config_sha256="2" * 64,
            source_archive_sha256="3" * 64,
            workload_identity=_workload_identity(seed=7, steps=4),
            environment_profile_sha256="5" * 64,
        )


def test_two_run_qualification_rejects_symlink_and_hardlink_inputs(
    tmp_path: Path,
) -> None:
    npz_stream = io.BytesIO()
    np.savez(npz_stream, rewards=np.arange(4, dtype=np.float16))
    database = _sqlite_bytes(
        tmp_path / "links.db",
        touch_same_value=False,
    )
    members = [
        (
            "official-results/result/data/7.npz",
            npz_stream.getvalue(),
        ),
        ("official-results/result/results.db", database),
        ("stdout.log", b"diagnostic\n"),
        ("stderr.log", b""),
    ]
    original = tmp_path / "original.tar"
    _v4_archive(original, members)
    symlink = tmp_path / "symlink.tar"
    symlink.symlink_to(original)
    bindings = {
        "backend": "cpu",
        "image_id": "sha256:" + "1" * 64,
        "runtime_profile_id": "fixture",
        "effective_seed": 7,
        "steps": 4,
        "config_sha256": "2" * 64,
        "source_archive_sha256": "3" * 64,
        "workload_identity": _workload_identity(seed=7, steps=4),
        "environment_profile_sha256": "5" * 64,
    }

    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="two distinct archive file identities",
    ):
        oci.qualify_v4_runs(original, original, **bindings)
    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="directly named regular file",
    ):
        oci.qualify_v4_runs(symlink, original, **bindings)

    hardlink_target = tmp_path / "hardlink-target.tar"
    _v4_archive(hardlink_target, members)
    hardlink = tmp_path / "hardlink.tar"
    hardlink.hardlink_to(hardlink_target)
    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="file identity is invalid",
    ):
        oci.qualify_v4_runs(hardlink_target, hardlink, **bindings)


def test_qualification_output_creation_is_exclusive_and_no_follow(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification.json"
    oci._write_new_file(output, b"first")
    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="cannot be created exclusively",
    ):
        oci._write_new_file(output, b"replacement")
    assert output.read_bytes() == b"first"

    target = tmp_path / "target.json"
    target.write_bytes(b"target")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(
        oci.OfficialForagaxOciError,
        match="cannot be created exclusively",
    ):
        oci._write_new_file(link, b"replacement")
    assert target.read_bytes() == b"target"


def test_gpu_launch_uses_explicit_devices_and_read_only_driver_bind() -> None:
    contract = oci.DriverLaunchContract(
        driver_host_path="/opt/alberta-driver-595.71.05",
        driver_container_path="/opt/nvidia-driver-595.71.05",
        device_paths=(
            "/dev/nvidia0",
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia-modeset",
        ),
        device_indices=(0,),
        cuda_wheel_library_paths=(
            "/opt/alberta-runtime/lib/python3.12/site-packages/nvidia",
        ),
        driver_user_library_paths=("/opt/nvidia-driver-595.71.05",),
    )
    command = oci.emit_launch_command(
        image_id="sha256:" + "6" * 64,
        entrypoint="src/rtu_ppo.py",
        config_path="/opt/continual-foragax-agents/config.json",
        index_expression="0",
        gpu=True,
        driver=contract,
    )
    assert not any(argument.startswith("--gpus") for argument in command)
    assert "--device=/dev/nvidia0" in command
    assert "--device=/dev/nvidiactl" in command
    assert "--device=/dev/nvidia-uvm" in command
    assert "--env=NVIDIA_VISIBLE_DEVICES=void" in command
    assert any(
        argument
        == (
            "--mount=type=bind,"
            "source=/opt/alberta-driver-595.71.05,"
            "destination=/opt/nvidia-driver-595.71.05,readonly"
        )
        for argument in command
    )
    assert (
        "--env=XLA_FLAGS=--xla_gpu_enable_triton_gemm=false "
        "--xla_gpu_deterministic_ops=true"
    ) in command


class _HostileStr(str):
    """A str subclass whose dunders raise, to prove they never run."""

    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_emit_launch_command_rejects_hostile_entrypoint_before_membership() -> None:
    """A hostile str-subclass entrypoint must not run its __eq__/__hash__ hooks.

    ``emit_launch_command`` gates its entrypoint allowlist with
    ``entrypoint not in {...}`` against an untrusted ``entrypoint: str``
    parameter. Before the fix, a hostile object reached that membership test
    before its type was confirmed, letting a hostile ``__eq__``/``__hash__``
    run.
    """
    hostile = _HostileStr("src/continuing_main.py")
    _HostileStr.calls = 0
    with pytest.raises(oci.OfficialForagaxOciError, match="entrypoint is not allowlisted"):
        oci.emit_launch_command(
            image_id="sha256:" + "1" * 64,
            entrypoint=hostile,  # type: ignore[arg-type]
            config_path="/opt/continual-foragax-agents/config.json",
            index_expression="0",
            gpu=False,
        )
    assert _HostileStr.calls == 0


def test_qualify_v4_runs_rejects_hostile_backend_before_membership(tmp_path: Path) -> None:
    """A hostile str-subclass backend must not run its __eq__/__hash__ hooks.

    ``qualify_v4_runs`` gates its backend allowlist with
    ``backend not in {...}`` as the very first check, before either archive
    is touched. Before the fix, a hostile object reached that membership test
    before its type was confirmed, letting a hostile ``__eq__``/``__hash__``
    run.
    """
    hostile = _HostileStr("cpu")
    _HostileStr.calls = 0
    with pytest.raises(
        oci.OfficialForagaxOciError, match="backend must be exactly cpu or gpu"
    ):
        oci.qualify_v4_runs(
            tmp_path / "first.tar",
            tmp_path / "second.tar",
            backend=hostile,  # type: ignore[arg-type]
            image_id="sha256:" + "1" * 64,
            runtime_profile_id="fixture",
            effective_seed=7,
            steps=12,
            config_sha256="2" * 64,
            source_archive_sha256="3" * 64,
            workload_identity={},
            environment_profile_sha256="5" * 64,
        )
    assert _HostileStr.calls == 0
