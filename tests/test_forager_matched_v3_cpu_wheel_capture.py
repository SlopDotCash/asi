"""Offline tests for the explicitly networked matched-v3 CPU wheel capture boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_cpu_wheel_capture as capture
from alberta_framework.benchmarks import forager_matched_v3_cpu_wheelhouse as wheelhouse

pytestmark = pytest.mark.unit


def _canonical(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _origin_url(filename: str) -> str:
    storage_key = hashlib.sha256(filename.encode("ascii")).hexdigest()
    return (
        "https://files.pythonhosted.org/packages/"
        f"{storage_key[:2]}/{storage_key[2:4]}/{storage_key[4:]}/{filename}"
    )


def _marker_environment() -> dict[str, str]:
    return {
        "implementation_name": "cpython",
        "implementation_version": "3.12.3",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "synthetic",
        "platform_system": "Linux",
        "platform_version": "synthetic",
        "python_full_version": "3.12.3",
        "python_version": "3.12",
        "sys_platform": "linux",
    }


def _manifest_bytes(wheels: dict[str, bytes]) -> tuple[bytes, str]:
    value: dict[str, Any] = {
        "capture": {
            "network_used": True,
            "resolver_argv": ["synthetic-resolver", "--wheel-only"],
            "resolver_binary_sha256": "1" * 64,
            "resolver_name": "synthetic-resolver",
            "resolver_version": "1",
        },
        "claims": copy.deepcopy(wheelhouse.cpu_wheelhouse_contract_descriptor()["claims"]),
        "classification": "networked_solver_output_non_authorizing",
        "root_requirements": ["alpha==1.0"],
        "schema_version": wheelhouse.CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "status": "untrusted_network_capture_candidate_only",
        "target": {
            "abi": "cp312",
            "compatible_tags": ["py3-none-any"],
            "implementation": "CPython",
            "libc": {"family": "glibc", "version": "2.39"},
            "marker_environment": _marker_environment(),
            "oci_platform": "linux/amd64",
            "platform": "linux_x86_64",
            "python_version": "3.12.3",
        },
        "wheels": [
            {
                "filename": filename,
                "origin_url": _origin_url(filename),
                "sha256": _sha(raw),
                "size_bytes": len(raw),
            }
            for filename, raw in sorted(wheels.items())
        ],
    }
    value["manifest_body_sha256"] = _sha(_canonical(value, newline=False))
    raw = _canonical(value)
    return raw, _sha(raw)


class _Response:
    def __init__(
        self,
        raw: bytes,
        *,
        final_url: str,
        status_code: int = 200,
        on_read: Callable[[], None] | None = None,
        ignore_read_bound: bool = False,
    ) -> None:
        self._raw = raw
        self._offset = 0
        self._on_read = on_read
        self._ignore_read_bound = ignore_read_bound
        self.final_url = final_url
        self.status_code = status_code
        self.closed = False

    def read(self, size: int) -> bytes:
        if self._on_read is not None:
            callback, self._on_read = self._on_read, None
            callback()
        if self._offset >= len(self._raw):
            return b""
        stop = (
            len(self._raw) if self._ignore_read_bound else min(len(self._raw), self._offset + size)
        )
        block = self._raw[self._offset : stop]
        self._offset = stop
        return block

    def close(self) -> None:
        self.closed = True


class _Transport:
    def __init__(
        self,
        payloads: dict[str, bytes],
        *,
        response_factory: Callable[[str, bytes], _Response] | None = None,
        on_open: Callable[[str], None] | None = None,
    ) -> None:
        self._payloads = payloads
        self._response_factory = response_factory
        self._on_open = on_open
        self.calls: list[tuple[str, float]] = []
        self.responses: list[_Response] = []

    def open(self, url: str, *, timeout_seconds: float) -> _Response:
        self.calls.append((url, timeout_seconds))
        if self._on_open is not None:
            callback, self._on_open = self._on_open, None
            callback(url)
        raw = self._payloads[url]
        response = (
            self._response_factory(url, raw)
            if self._response_factory is not None
            else _Response(raw, final_url=url)
        )
        self.responses.append(response)
        return response


def _payloads(wheels: dict[str, bytes]) -> dict[str, bytes]:
    return {_origin_url(filename): raw for filename, raw in wheels.items()}


def _capture(
    root: Path,
    wheels: dict[str, bytes],
    transport: _Transport,
) -> capture.PublishedMatchedV3CpuWheelCapture:
    manifest, digest = _manifest_bytes(wheels)
    return capture.capture_matched_v3_cpu_wheels(
        capture_manifest_raw=manifest,
        expected_capture_manifest_sha256=digest,
        publication_root=root,
        transport=transport,
        authorize_network_capture=True,
        request_timeout_seconds=7.0,
    )


def test_descriptor_is_explicitly_networked_and_all_claims_are_false() -> None:
    descriptor = capture.cpu_wheel_capture_contract_descriptor()
    assert descriptor["status"] == "implemented_networked_untrusted_non_authorizing"
    assert descriptor["network"]["implemented_here"] is True
    assert descriptor["network"]["redirects_followed"] == 0
    assert descriptor["network"]["exact_authority"] == "files.pythonhosted.org"
    assert descriptor["network"]["origin_basename_equals_manifest_filename"] is True
    assert descriptor["publication"]["caller_supplied_absolute_root"] is True
    assert descriptor["publication"]["default_root"] is False
    assert descriptor["publication"]["post_commit_descriptor_close_failure"] == (
        "reported_with_committed_content_address_preserved"
    )
    assert descriptor["schemas"]["capture_manifest"] == (
        wheelhouse.CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION
    )
    assert all(value is False for value in descriptor["claims"].values())
    limitations = " ".join(descriptor["limitations"]).lower()
    for phrase in ("tls", "resolver", "index", "network isolation", "content", "execution"):
        assert phrase in limitations


def test_capture_requires_literal_authorization_and_absolute_existing_root(tmp_path: Path) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel"}
    manifest, digest = _manifest_bytes(wheels)
    transport = _Transport(_payloads(wheels))
    root = tmp_path / "publication"
    root.mkdir()

    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="authorization"):
        capture.capture_matched_v3_cpu_wheels(
            capture_manifest_raw=manifest,
            expected_capture_manifest_sha256=digest,
            publication_root=root,
            transport=transport,
            authorize_network_capture=1,  # type: ignore[arg-type]
        )
    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="absolute"):
        capture.capture_matched_v3_cpu_wheels(
            capture_manifest_raw=manifest,
            expected_capture_manifest_sha256=digest,
            publication_root=Path("relative"),
            transport=transport,
            authorize_network_capture=True,
        )
    assert transport.calls == []
    assert list(root.iterdir()) == []


def test_capture_publishes_exact_manifest_and_flat_wheels_by_manifest_sha(tmp_path: Path) -> None:
    wheels = {
        "alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes",
        "bravo-2.0-py3-none-any.whl": b"bravo-wheel-bytes" * 3,
    }
    manifest, digest = _manifest_bytes(wheels)
    transport = _Transport(_payloads(wheels))
    root = tmp_path / "publication"
    root.mkdir()

    published = _capture(root, wheels, transport)

    assert published.directory == root / "sha256" / digest
    assert published.manifest == published.directory / "manifest.v1.json"
    assert published.wheels == published.directory / "wheels"
    assert published.manifest_sha256 == digest
    assert published.wheel_count == 2
    assert published.total_size_bytes == sum(map(len, wheels.values()))
    assert published.manifest.read_bytes() == manifest
    assert {path.name: path.read_bytes() for path in published.wheels.iterdir()} == wheels
    assert [url for url, timeout in transport.calls if timeout == 7.0] == [
        _origin_url(filename) for filename in sorted(wheels)
    ]
    assert all(response.closed for response in transport.responses)
    assert stat.S_IMODE(published.directory.stat().st_mode) == 0o555
    assert stat.S_IMODE(published.wheels.stat().st_mode) == 0o555
    assert stat.S_IMODE(published.manifest.stat().st_mode) == 0o444
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in published.wheels.iterdir())
    parsed = wheelhouse.parse_cpu_wheel_capture_manifest(
        published.manifest.read_bytes(),
        expected_file_sha256=published.manifest_sha256,
    )
    assert all(value is False for value in parsed["claims"].values())


@pytest.mark.parametrize(
    ("returned", "error"),
    [
        (b"too-short", "size"),
        (b"alpha-wheel-bytet", "SHA-256"),
        (b"alpha-wheel-bytes-extra", "size"),
    ],
)
def test_capture_rejects_wrong_size_or_digest_and_removes_partial_tree(
    tmp_path: Path,
    returned: bytes,
    error: str,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    url = next(iter(_payloads(wheels)))
    transport = _Transport({url: returned})
    root = tmp_path / "publication"
    root.mkdir()

    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match=error):
        _capture(root, wheels, transport)

    namespace = root / "sha256"
    assert not namespace.exists() or list(namespace.iterdir()) == []
    assert transport.responses[0].closed


def test_transport_cannot_return_more_than_the_requested_read_bound(tmp_path: Path) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}

    def unbounded(url: str, raw: bytes) -> _Response:
        return _Response(raw + b"extra", final_url=url, ignore_read_bound=True)

    root = tmp_path / "publication"
    root.mkdir()
    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="read bound"):
        _capture(root, wheels, _Transport(_payloads(wheels), response_factory=unbounded))


@pytest.mark.parametrize(("status", "final_url"), [(302, None), (200, "https://other.invalid/x")])
def test_capture_rejects_redirect_status_or_changed_final_url(
    tmp_path: Path,
    status: int,
    final_url: str | None,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}

    def redirected(url: str, raw: bytes) -> _Response:
        return _Response(raw, final_url=url if final_url is None else final_url, status_code=status)

    root = tmp_path / "publication"
    root.mkdir()
    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="redirect|status"):
        _capture(root, wheels, _Transport(_payloads(wheels), response_factory=redirected))


def test_invalid_origin_is_rejected_by_canonical_parser_before_transport(tmp_path: Path) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    raw, _digest = _manifest_bytes(wheels)
    value = json.loads(raw)
    value["wheels"][0]["origin_url"] += "?credential=secret"
    del value["manifest_body_sha256"]
    value["manifest_body_sha256"] = _sha(_canonical(value, newline=False))
    changed = _canonical(value)
    transport = _Transport({})
    root = tmp_path / "publication"
    root.mkdir()

    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="canonical HTTPS"):
        capture.capture_matched_v3_cpu_wheels(
            capture_manifest_raw=changed,
            expected_capture_manifest_sha256=_sha(changed),
            publication_root=root,
            transport=transport,
            authorize_network_capture=True,
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    "origin",
    [
        ("https://127.0.0.1/packages/00/11/" + "2" * 60 + "/alpha-1.0-py3-none-any.whl"),
        (
            "https://files.pythonhosted.org:443/packages/00/11/"
            + "2" * 60
            + "/alpha-1.0-py3-none-any.whl"
        ),
        (
            "https://FILES.pythonhosted.org/packages/00/11/"
            + "2" * 60
            + "/alpha-1.0-py3-none-any.whl"
        ),
        (
            "https://files.pythonhosted.org/packages/00/11/"
            + "2" * 60
            + "/bravo-1.0-py3-none-any.whl"
        ),
        (
            "https://files.pythonhosted.org/packages/00/11/"
            + "2" * 60
            + "/%61lpha-1.0-py3-none-any.whl"
        ),
        "https://éxample.invalid/files/alpha-1.0-py3-none-any.whl",
    ],
)
def test_capture_rejects_ssrf_or_noncanonical_production_origin_before_transport(
    tmp_path: Path,
    origin: str,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    raw, _digest = _manifest_bytes(wheels)
    value = json.loads(raw)
    value["wheels"][0]["origin_url"] = origin
    del value["manifest_body_sha256"]
    value["manifest_body_sha256"] = _sha(_canonical(value, newline=False))
    changed = _canonical(value)
    root = tmp_path / "publication"
    root.mkdir()
    transport = _Transport({})

    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="production origin"):
        capture.capture_matched_v3_cpu_wheels(
            capture_manifest_raw=changed,
            expected_capture_manifest_sha256=_sha(changed),
            publication_root=root,
            transport=transport,
            authorize_network_capture=True,
        )

    assert transport.calls == []
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "failed_entry",
    [
        "staging",
        "wheels",
    ],
)
def test_directory_validation_failure_after_create_removes_owned_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_entry: str,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    original = os.stat
    injected = False

    def fail_after_create(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal injected
        is_target = (failed_entry == "staging" and str(path).startswith(".capture-")) or (
            failed_entry == "wheels" and path == "wheels"
        )
        if is_target and not injected:
            injected = True
            raise OSError("synthetic post-create directory-validation failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fail_after_create)
    with pytest.raises(OSError, match="post-create directory-validation failure"):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    assert injected
    assert list(root.iterdir()) == []


def test_directory_open_failure_without_capability_never_deletes_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    original = os.open
    displaced: Path | None = None
    replacement: Path | None = None
    injected = False

    def substitute_before_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal displaced, replacement, injected
        if not injected and str(path).startswith(".capture-") and flags & getattr(os, "O_DIRECTORY", 0):
            injected = True
            parent_fd = kwargs["dir_fd"]
            parent = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
            created = parent / str(path)
            displaced = parent / "displaced-unopened-staging"
            created.rename(displaced)
            replacement = created
            replacement.mkdir()
            (replacement / "do-not-delete.txt").write_text("replacement", encoding="utf-8")
            raise OSError("synthetic pre-capability directory-open failure")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", substitute_before_open)
    with pytest.raises(OSError, match="pre-capability directory-open failure") as caught:
        _capture(root, wheels, _Transport(_payloads(wheels)))

    assert injected
    assert displaced is not None and displaced.is_dir()
    assert replacement is not None
    assert (replacement / "do-not-delete.txt").read_text(encoding="utf-8") == "replacement"
    assert any("no retained identity" in note for note in caught.value.__notes__)


def test_namespace_fsync_failure_after_create_removes_owned_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    original = os.fsync
    injected = False

    def fail_first_root_fsync(descriptor: int) -> None:
        nonlocal injected
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        if not injected and target == str(root):
            injected = True
            raise OSError("synthetic namespace durability failure")
        original(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_root_fsync)
    with pytest.raises(OSError, match="namespace durability failure"):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    assert injected
    assert list(root.iterdir()) == []


def test_output_validation_failure_after_create_removes_owned_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    original = os.stat
    injected = False

    def fail_first_wheel_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal injected
        if path == "alpha-1.0-py3-none-any.whl" and not injected:
            injected = True
            raise OSError("synthetic post-create output validation failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fail_first_wheel_stat)
    with pytest.raises(OSError, match="post-create output validation failure"):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    assert injected
    assert list(root.iterdir()) == []


def test_output_initial_fstat_failure_recovers_identity_and_removes_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    original = os.fstat
    injected = False

    def fail_first_wheel_fstat(descriptor: int) -> os.stat_result:
        nonlocal injected
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        if not injected and target.endswith("/alpha-1.0-py3-none-any.whl"):
            injected = True
            raise OSError("synthetic initial output fstat failure")
        return original(descriptor)

    monkeypatch.setattr(os, "fstat", fail_first_wheel_fstat)
    with pytest.raises(OSError, match="initial output fstat failure"):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    assert injected
    assert list(root.iterdir()) == []


def test_manifest_finish_failure_removes_owned_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    original = capture._finish_output_file
    injected = False

    def fail_manifest(*args: Any, **kwargs: Any) -> None:
        nonlocal injected
        if kwargs.get("label") == "capture manifest publication file":
            injected = True
            raise OSError("synthetic manifest finish failure")
        original(*args, **kwargs)

    monkeypatch.setattr(capture, "_finish_output_file", fail_manifest)
    with pytest.raises(OSError, match="manifest finish failure"):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    assert injected
    assert list(root.iterdir()) == []


def test_post_commit_descriptor_close_failure_is_not_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    manifest, digest = _manifest_bytes(wheels)
    root = tmp_path / "publication"
    root.mkdir()
    original = os.close
    injected = False

    def close_then_fail(descriptor: int) -> None:
        nonlocal injected
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        original(descriptor)
        if not injected and target.endswith("/wheels"):
            injected = True
            raise OSError("synthetic post-commit descriptor close failure")

    monkeypatch.setattr(os, "close", close_then_fail)
    with pytest.raises(
        capture.ForagerMatchedV3CpuWheelCaptureError,
        match="committed.*descriptor cleanup",
    ):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    assert injected
    published = root / "sha256" / digest
    assert (published / "manifest.v1.json").read_bytes() == manifest
    assert (published / "wheels" / next(iter(wheels))).read_bytes() == next(iter(wheels.values()))


def test_primary_hash_failure_survives_output_descriptor_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    wrong = b"alpha-wheel-bytet"
    url = next(iter(_payloads(wheels)))
    root = tmp_path / "publication"
    root.mkdir()
    original_close = os.close
    injected = False

    def close_then_fail(descriptor: int) -> None:
        nonlocal injected
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        original_close(descriptor)
        if not injected and target.endswith("/wheels/alpha-1.0-py3-none-any.whl"):
            injected = True
            raise OSError("synthetic captured-wheel close failure")

    monkeypatch.setattr(os, "close", close_then_fail)
    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="SHA-256") as caught:
        _capture(root, wheels, _Transport({url: wrong}))

    assert injected
    assert any("captured-wheel close failure" in note for note in caught.value.__notes__)


def test_existing_content_address_is_never_refetched_or_overwritten(tmp_path: Path) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    first = _capture(root, wheels, _Transport(_payloads(wheels)))
    sentinel = first.manifest.read_bytes()
    second_transport = _Transport(_payloads(wheels))

    with pytest.raises(FileExistsError, match="overwrite"):
        _capture(root, wheels, second_transport)

    assert second_transport.calls == []
    assert first.manifest.read_bytes() == sentinel


def test_namespace_substitution_is_rejected_without_touching_replacement(tmp_path: Path) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    displaced = root / "displaced-sha256"

    def substitute(_url: str) -> None:
        namespace = root / "sha256"
        namespace.rename(displaced)
        namespace.mkdir()
        (namespace / "do-not-delete.txt").write_text("replacement", encoding="utf-8")

    transport = _Transport(_payloads(wheels), on_open=substitute)
    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="namespace"):
        _capture(root, wheels, transport)

    assert (root / "sha256" / "do-not-delete.txt").read_text(encoding="utf-8") == "replacement"
    assert displaced.is_dir()
    assert list(displaced.iterdir()) == []


def test_staging_name_substitution_cleanup_is_inode_pinned(tmp_path: Path) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    displaced: Path | None = None
    replacement: Path | None = None

    def substitute() -> None:
        nonlocal displaced, replacement
        namespace = root / "sha256"
        staging = next(namespace.glob(".capture-*"))
        displaced = namespace / "displaced-private-staging"
        staging.rename(displaced)
        replacement = staging
        replacement.mkdir()
        (replacement / "do-not-delete.txt").write_text("replacement", encoding="utf-8")

    def response(url: str, raw: bytes) -> _Response:
        return _Response(raw, final_url=url, on_read=substitute)

    transport = _Transport(_payloads(wheels), response_factory=response)
    with pytest.raises(capture.ForagerMatchedV3CpuWheelCaptureError, match="staging"):
        _capture(root, wheels, transport)

    assert displaced is not None and displaced.is_dir()
    assert replacement is not None
    assert (replacement / "do-not-delete.txt").read_text(encoding="utf-8") == "replacement"


def test_rename_then_error_is_detected_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    _raw, digest = _manifest_bytes(wheels)
    root = tmp_path / "publication"
    root.mkdir()
    original = capture._rename_new_only

    def rename_then_fail(parent_fd: int, source: str, target: str) -> None:
        original(parent_fd, source, target)
        raise OSError("synthetic error after rename")

    monkeypatch.setattr(capture, "_rename_new_only", rename_then_fail)
    with pytest.raises(OSError, match="synthetic error after rename"):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    namespace = root / "sha256"
    assert not (namespace / digest).exists()
    assert not namespace.exists() or list(namespace.iterdir()) == []


def test_post_commit_validation_failure_rolls_back_content_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    _raw, digest = _manifest_bytes(wheels)
    root = tmp_path / "publication"
    root.mkdir()
    original = capture._validate_capture_tree
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        original(*args, **kwargs)
        if calls == 2:
            raise RuntimeError("synthetic post-commit validation failure")

    monkeypatch.setattr(capture, "_validate_capture_tree", fail_second)
    with pytest.raises(RuntimeError, match="synthetic post-commit validation failure"):
        _capture(root, wheels, _Transport(_payloads(wheels)))

    namespace = root / "sha256"
    assert calls == 2
    assert not (namespace / digest).exists()
    assert not namespace.exists() or list(namespace.iterdir()) == []


def test_wheel_files_are_created_exclusively_without_following_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = {"alpha-1.0-py3-none-any.whl": b"alpha-wheel-bytes"}
    root = tmp_path / "publication"
    root.mkdir()
    original_open = os.open
    observed_flags: list[int] = []

    def record_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "alpha-1.0-py3-none-any.whl" and flags & os.O_WRONLY:
            observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", record_open)
    _capture(root, wheels, _Transport(_payloads(wheels)))

    assert len(observed_flags) == 1
    assert observed_flags[0] & os.O_CREAT
    assert observed_flags[0] & os.O_EXCL
    assert observed_flags[0] & getattr(os, "O_NOFOLLOW", 0)
