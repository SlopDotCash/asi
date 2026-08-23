"""Unit coverage for fetch_source: supply-chain safety of the pinned
source archive download — redirect origin check, byte limits, SHA-256
verification, and safe tar extraction."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import external_runtimes.loss_of_plasticity_mnist.fetch_source as fs


class TestDownloadSafety:
    def _fake_archive_path(self) -> MagicMock:
        fake = MagicMock()
        fake.open.return_value.__enter__.return_value = BytesIO()
        return fake

    def test_rejects_redirect_outside_official_origin(self) -> None:
        with (
            patch.object(fs, "_ARCHIVE_PATH", self._fake_archive_path()),
            patch("urllib.request.urlopen") as urlopen,
        ):
            response = MagicMock()
            response.geturl.return_value = "https://evil.com/archive.tar.gz"
            response.read.side_effect = [b"", b""]
            urlopen.return_value.__enter__.return_value = response
            with pytest.raises(ValueError, match="redirected outside"):
                fs._download()

    def test_rejects_archive_over_byte_limit(self) -> None:
        with (
            patch.object(fs, "_ARCHIVE_PATH", self._fake_archive_path()),
            patch("urllib.request.urlopen") as urlopen,
        ):
            response = MagicMock()
            response.geturl.return_value = (
                "https://codeload.github.com/shibhansh/loss-of-plasticity/ok"
            )
            # 单个 17MB chunk 超上限
            response.read.side_effect = [b"x" * 17_000_000, b""]
            urlopen.return_value.__enter__.return_value = response
            with pytest.raises(ValueError, match="exceeds its byte limit"):
                fs._download()

    def test_rejects_sha256_mismatch(self) -> None:
        with (
            patch.object(fs, "_ARCHIVE_PATH", self._fake_archive_path()),
            patch("urllib.request.urlopen") as urlopen,
        ):
            response = MagicMock()
            response.geturl.return_value = (
                "https://codeload.github.com/shibhansh/loss-of-plasticity/ok"
            )
            response.read.side_effect = [b"not-the-pinned-archive", b""]
            urlopen.return_value.__enter__.return_value = response
            with pytest.raises(ValueError, match="differs from the pinned"):
                fs._download()


class TestSafeExtraction:
    def test_rejects_path_traversal_members(self) -> None:
        with patch.object(fs, "tarfile") as tarfile_mock:
            tar = MagicMock()
            member = MagicMock()
            member.name = f"{fs.SOURCE_COMMIT}/../../etc/passwd"
            member.isdir.return_value = False
            member.isreg.return_value = True
            member.size = 100
            tar.getmembers.return_value = [member]
            tarfile_mock.open.return_value.__enter__.return_value = tar
            with pytest.raises(ValueError, match="inadmissible member"):
                fs._extract()

    def test_rejects_absolute_path_members(self) -> None:
        with patch.object(fs, "tarfile") as tarfile_mock:
            tar = MagicMock()
            member = MagicMock()
            member.name = "/etc/passwd"
            member.isdir.return_value = False
            member.isreg.return_value = True
            member.size = 100
            tar.getmembers.return_value = [member]
            tarfile_mock.open.return_value.__enter__.return_value = tar
            with pytest.raises(ValueError, match="inadmissible member"):
                fs._extract()

    def test_rejects_oversized_expansion(self) -> None:
        with (
            patch.object(fs, "tarfile") as tarfile_mock,
            patch.object(fs, "_MAX_EXPANDED_BYTES", 1000),
        ):
            tar = MagicMock()
            member = MagicMock()
            member.name = f"loss-of-plasticity-{fs.SOURCE_COMMIT}/safe/file.txt"
            member.isdir.return_value = False
            member.isreg.return_value = True
            member.size = 5000  # 超上限
            tar.getmembers.return_value = [member]
            tarfile_mock.open.return_value.__enter__.return_value = tar
            with pytest.raises(ValueError, match="expanded byte limit"):
                fs._extract()

    def test_rejects_empty_archive(self) -> None:
        with patch.object(fs, "tarfile") as tarfile_mock:
            tar = MagicMock()
            tar.getmembers.return_value = []
            tarfile_mock.open.return_value.__enter__.return_value = tar
            with pytest.raises(ValueError, match="member count"):
                fs._extract()
