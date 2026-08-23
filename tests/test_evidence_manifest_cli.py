"""Unit coverage for evidence_manifest_cli: fail-closed output-path
protection and main() exit-code wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from alberta_framework.evaluation.evidence_manifest_cli import (
    _resolved_new_output,
    main,
)


class TestResolvedNewOutput:
    def test_rejects_canonical_default_path(self) -> None:
        with patch(
            "alberta_framework.evaluation.evidence_manifest_cli.DEFAULT_OUTPUT",
            Path("/repo/outputs/evidence_manifest.json"),
        ):
            with pytest.raises(FileExistsError, match="canonical artifact path"):
                _resolved_new_output(Path("/repo/outputs/evidence_manifest.json"))

    def test_rejects_existing_output_path(self, tmp_path: Path) -> None:
        existing = tmp_path / "existing.json"
        existing.write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            _resolved_new_output(existing)

    def test_accepts_fresh_path(self, tmp_path: Path) -> None:
        fresh = tmp_path / "fresh.json"
        resolved = _resolved_new_output(fresh)
        assert resolved == fresh.resolve()


class TestMain:
    def test_prints_manifest_and_returns_exit_code(self) -> None:
        manifest_payload = '{"version": 1, "claims": []}'
        with (
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.build_evidence_manifest",
                return_value=object(),
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.evidence_manifest_json",
                return_value=manifest_payload,
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.evidence_manifest_exit_code",
                return_value=0,
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.sys.stdout"
            ) as stdout,
        ):
            assert main([]) == 0
            stdout.write.assert_called_once_with(manifest_payload)

    def test_writes_output_file_when_requested(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        with (
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.build_evidence_manifest",
                return_value=object(),
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.evidence_manifest_json",
                return_value='{"ok": true}',
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.evidence_manifest_exit_code",
                return_value=1,
            ),
        ):
            code = main(["--output", str(out)])
            assert code == 1
            assert out.read_text(encoding="utf-8") == '{"ok": true}'

    def test_returns_2_on_output_error(self) -> None:
        with (
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.build_evidence_manifest",
                return_value=object(),
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.evidence_manifest_exit_code",
                return_value=0,
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli._resolved_new_output",
                side_effect=FileExistsError("refusing to overwrite"),
            ),
            patch(
                "alberta_framework.evaluation.evidence_manifest_cli.sys.stderr"
            ) as stderr,
        ):
            assert main(["--output", "/x/y.json"]) == 2
            stderr.write.assert_called()
