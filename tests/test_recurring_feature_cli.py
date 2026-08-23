"""Unit coverage for recurring_feature_cli: fail-closed output-path
protection and main() mode wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from alberta_framework.evaluation.recurring_feature_cli import (
    _resolved_new_output,
    main,
)


class TestResolvedNewOutput:
    def test_rejects_canonical_default_path(self) -> None:
        with patch(
            "alberta_framework.evaluation.recurring_feature_cli.DEFAULT_OUTPUT",
            Path("/repo/outputs/recurring_feature/evidence.v1.json"),
        ):
            with pytest.raises(FileExistsError, match="canonical artifact path"):
                _resolved_new_output(
                    Path("/repo/outputs/recurring_feature/evidence.v1.json")
                )

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
    def test_verify_mode_prints_and_returns(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "evidence.v1.json"
        artifact_path.write_text("{}", encoding="utf-8")
        with (
            patch(
                "alberta_framework.evaluation.recurring_feature_cli.load_recurring_feature_artifact",
                return_value=object(),
            ),
            patch(
                "alberta_framework.evaluation.recurring_feature_cli.validate_recurring_feature_artifact",
            ) as validate,
        ):
            validate.return_value.accepted = True
            validate.return_value.valid = True
            validate.return_value.errors = []
            code = main(["--verify", str(artifact_path)])
            assert code == 0

    def test_verify_mode_returns_2_on_invalid(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "bad.json"
        artifact_path.write_text("{}", encoding="utf-8")
        with (
            patch(
                "alberta_framework.evaluation.recurring_feature_cli.load_recurring_feature_artifact",
                return_value=object(),
            ),
            patch(
                "alberta_framework.evaluation.recurring_feature_cli.validate_recurring_feature_artifact",
            ) as validate,
        ):
            validate.return_value.accepted = False
            validate.return_value.valid = False
            validate.return_value.errors = ["bad"]
            code = main(["--verify", str(artifact_path)])
            assert code == 2

    def test_returns_2_on_output_error(self) -> None:
        with (
            patch(
                "alberta_framework.evaluation.recurring_feature_cli._resolved_new_output",
                side_effect=FileExistsError("refusing to overwrite"),
            ),
            patch(
                "alberta_framework.evaluation.recurring_feature_cli._emit"
            ) as emit,
        ):
            assert main(["--output", "/x/y.json"]) == 2
            emit.assert_called_once()
            payload = emit.call_args.args[0]
            assert payload["valid"] is False
            assert payload["accepted"] is False
