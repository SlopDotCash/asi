"""Public API, read-only CLI, and distribution checks for the historical lane."""

from __future__ import annotations

import configparser
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import alberta_framework.benchmarks as benchmark_api
import alberta_framework.benchmarks.historical_forager as historical_module
from alberta_framework import forager_cli
from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerRunConfig,
    HistoricalUpdateKernel,
    development_historical_environment_adapter,
    run_historical_forager,
)
from alberta_framework.benchmarks.historical_forager_provenance import (
    HISTORICAL_FORAGER_FAMILY_ID,
    HISTORICAL_FORAGER_PROVENANCE_SHA256,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[1]
provenance_module = importlib.import_module(
    "alberta_framework.benchmarks.historical_forager_provenance"
)
_WHEEL_FILES = tuple(
    path.relative_to(_REPO_ROOT).as_posix()
    for path in sorted((_REPO_ROOT / "alberta_framework").rglob("*"))
    if path.is_file()
    and "__pycache__" not in path.parts
    and (path.suffix == ".py" or path.name == "py.typed")
)
_EXPECTED_SCRIPT_NAMES = {
    "asi-action-conditioned-latent",
    "asi-activation-feature-campaign",
    "asi-activation-feature-ipmnist",
    "asi-adamo-diagnostic",
    "asi-bimu-matched-development",
    "asi-clear-qualification",
    "asi-coom-qualification-smoke",
    "asi-cora-catalog",
    "asi-forager-rerun-preflight",
    "asi-telapa-qualification-smoke",
    "asi-ipmnist-campaign",
    "asi-ipmnist-ceiling",
    "asi-l2er-matched-development",
    "asi-new-directions-audit",
    "asi-nap-ipmnist",
    "asi-native-supervised-catalog",
    "asi-plasticity-diagnostic",
    "asi-jepa-transfer-feasibility",
    "asi-reference-life-scorecard",
    "asi-rule-discovery-summary",
    "alberta-evidence-status",
    "alberta-forager-benchmark",
    "alberta-forager-matched-campaign",
    "alberta-forager-matched-qualification",
    "alberta-forager-matched-sealed-evaluation",
    "alberta-foragax-oci",
    "alberta-foragax-open-screen",
    "alberta-ftl-evidence",
    "alberta-historical-forager",
    "alberta-ia-evidence",
    "alberta-multiagent-evidence",
    "alberta-recurring-feature-evidence",
    "alberta-scale-robust-evidence",
    "alberta-step1-smoke",
    "alberta-step2-smoke",
}
_HARD_LINK_SENSITIVE_SOURCES = (
    "alberta_framework/benchmarks/official_foragax.py",
    "alberta_framework/benchmarks/runtime_profile.py",
    "alberta_framework/benchmarks/forager_results.py",
)
_SDIST_EXTRA_FILES = (
    "CHANGELOG.md",
    "CITATION.cff",
    "FORAGER_BENCHMARK.md",
    "LICENSE",
    "README.md",
    "VENDORING.md",
    "pyproject.toml",
    *(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in sorted((_REPO_ROOT / "docs").rglob("*.md"))
    ),
)


class _TinyHistoricalEnvironment:
    def __init__(self) -> None:
        self._offset = 0

    def start(self) -> int:
        return 0

    def step(self, action: int) -> tuple[float, int, bool, Mapping[str, Any]]:
        reward = (1.0, -0.5, 2.0, 0.25)[self._offset]
        self._offset += 1
        return reward, self._offset, False, {}


def _write_tiny_artifact(output_directory: Path, *, seed: int) -> None:
    def factory(_seed: int, _aperture_size: int) -> _TinyHistoricalEnvironment:
        return _TinyHistoricalEnvironment()

    kernel = HistoricalUpdateKernel[int](
        name="historical_cli_test_kernel",
        start_kernel=lambda _observation: (0, 0),
        update_kernel=lambda state, _reward, _observation: (state + 1, (state + 1) % 4),
        metadata={"purpose": "cli_contract_test"},
    )
    run_historical_forager(
        development_historical_environment_adapter(factory),
        kernel,
        HistoricalForagerRunConfig(
            seed=seed,
            steps=4,
            aperture_size=9,
            output_directory=output_directory,
            allow_unverified_development_adapter=True,
        ),
    )


def _run_historical_cli(
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    original_handlers = list(forager_cli.LOGGER.handlers)
    original_level = forager_cli.LOGGER.level
    original_propagate = forager_cli.LOGGER.propagate
    forager_cli.LOGGER.handlers.clear()
    try:
        try:
            returncode = forager_cli.main(("historical", *arguments))
        except SystemExit as exc:
            returncode = exc.code if isinstance(exc.code, int) else 1
        captured = capsys.readouterr()
    finally:
        forager_cli.LOGGER.handlers.clear()
        forager_cli.LOGGER.handlers.extend(original_handlers)
        forager_cli.LOGGER.setLevel(original_level)
        forager_cli.LOGGER.propagate = original_propagate
    return subprocess.CompletedProcess(arguments, returncode, captured.out, captured.err)


def test_benchmark_package_exports_complete_historical_public_api() -> None:
    expected = set(historical_module.__all__) | set(provenance_module.__all__)

    assert expected <= set(benchmark_api.__all__)
    for name in expected:
        source_module = (
            historical_module if name in historical_module.__all__ else provenance_module
        )
        assert getattr(benchmark_api, name) is getattr(source_module, name)


def test_historical_subcommand_help_is_read_only_and_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_help = forager_cli._parser().format_help()
    historical_parser = forager_cli._historical_parser(prog="alberta-forager-benchmark historical")
    historical_help = historical_parser.format_help()
    command_action = next(
        action
        for action in historical_parser._actions
        if isinstance(action, forager_cli.argparse._SubParsersAction)
    )

    assert "historical --help" in root_help
    assert set(command_action.choices) == {"provenance", "validate", "pair"}
    assert HISTORICAL_FORAGER_FAMILY_ID in historical_help
    assert "explicitly unattested" in historical_help
    assert "never launches a benchmark run" in historical_help

    completed = _run_historical_cli(capsys, "--help")
    assert completed.returncode == 0
    assert "{provenance,validate,pair}" in completed.stdout
    assert completed.stderr == ""


def test_historical_cli_reports_provenance_and_validates_strict_pairing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    wrong_seed = tmp_path / "wrong-seed"
    _write_tiny_artifact(left, seed=17)
    _write_tiny_artifact(right, seed=17)
    _write_tiny_artifact(wrong_seed, seed=18)

    provenance = _run_historical_cli(capsys, "provenance")
    assert provenance.returncode == 0
    provenance_payload = json.loads(provenance.stdout)
    assert provenance_payload["family_id"] == HISTORICAL_FORAGER_FAMILY_ID
    assert provenance_payload["environment_resolution_attested"] is False
    assert provenance_payload["pairable_with_current_foragax"] is False
    assert provenance_payload["provenance_sha256"] == HISTORICAL_FORAGER_PROVENANCE_SHA256
    assert provenance_payload["provenance"]["environment_resolution_attested"] is False

    validated = _run_historical_cli(capsys, "validate", str(left))
    assert validated.returncode == 0
    validated_payload = json.loads(validated.stdout)
    assert validated_payload["artifact"]["status"] == "complete"
    assert validated_payload["artifact"]["family_id"] == HISTORICAL_FORAGER_FAMILY_ID

    paired = _run_historical_cli(capsys, "pair", str(left), str(right))
    assert paired.returncode == 0
    paired_payload = json.loads(paired.stdout)
    assert paired_payload["pairable"] is True
    assert paired_payload["pairing_identity"]["seed"] == 17
    assert paired_payload["pairing_identity"]["family_id"] == HISTORICAL_FORAGER_FAMILY_ID

    rejected = _run_historical_cli(capsys, "pair", str(left), str(wrong_seed))
    assert rejected.returncode == 2
    assert rejected.stdout == ""
    assert "identical provenance, seed, aperture" in rejected.stderr

    missing = _run_historical_cli(capsys, "validate", str(tmp_path / "missing"))
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert "must be a real directory" in missing.stderr


@pytest.mark.package
@pytest.mark.skipif(
    not hasattr(os, "O_TMPFILE"),
    reason="write_new_json publishes through Linux O_TMPFILE and linkat(AT_EMPTY_PATH)",
)
def test_wheel_and_sdist_are_complete_and_hard_link_safe(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        interpreter_bin = str(Path(sys.executable).parent)
        uv = shutil.which("uv", path=interpreter_bin)
    if uv is None:
        pytest.fail("uv is required; install the project's dev dependencies")
    prebuilt_directory = os.environ.get("ALBERTA_PREBUILT_DIST_DIR")
    if prebuilt_directory:
        output_directory = Path(prebuilt_directory)
    else:
        output_directory = tmp_path / "dist"
        build_arguments = [uv, "build", "--no-build-isolation"]
        if os.environ.get("ALBERTA_PACKAGE_BUILD_ALLOW_NETWORK") != "1":
            build_arguments.append("--offline")
        build_arguments.extend(
            (
                "--no-build-logs",
                "--no-create-gitignore",
                "--no-python-downloads",
                "--out-dir",
                str(output_directory),
                str(_REPO_ROOT),
            )
        )
        build_environment = os.environ.copy()
        build_environment["VIRTUAL_ENV"] = sys.prefix
        build_environment["PATH"] = os.pathsep.join(
            (str(Path(sys.executable).parent), build_environment.get("PATH", ""))
        )
        completed = subprocess.run(
            build_arguments,
            cwd=_REPO_ROOT,
            env=build_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr

    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    expected_scripts = project["project"]["scripts"]
    assert set(expected_scripts) == _EXPECTED_SCRIPT_NAMES
    wheel_path = output_directory / f"alberta_framework-{version}-py3-none-any.whl"
    sdist_path = output_directory / f"alberta_framework-{version}.tar.gz"
    assert wheel_path.is_file()
    assert sdist_path.is_file()

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_names = set(wheel.namelist())
        for relative_path in _WHEEL_FILES:
            assert relative_path in wheel_names
            assert wheel.read(relative_path) == (_REPO_ROOT / relative_path).read_bytes()
        entry_points_name = next(
            name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = configparser.ConfigParser(interpolation=None)
        entry_points.optionxform = str
        entry_points.read_string(wheel.read(entry_points_name).decode("utf-8"))
        assert dict(entry_points["console_scripts"]) == expected_scripts

    install_directory = tmp_path / "wheel-install"
    install_environment = os.environ.copy()
    install_environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    installed = subprocess.run(
        (
            uv,
            "pip",
            "install",
            "--offline",
            "--no-python-downloads",
            "--link-mode",
            "hardlink",
            "--target",
            str(install_directory),
            "--no-deps",
            str(wheel_path),
        ),
        cwd=tmp_path,
        env=install_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert installed.returncode == 0, installed.stderr

    for relative_path in _HARD_LINK_SENSITIVE_SOURCES:
        source = install_directory / relative_path
        assert source.stat().st_nlink >= 2

    probe = """
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys

import alberta_framework

install_root = Path(sys.argv[1]).resolve()
package_path = Path(alberta_framework.__file__).resolve()
if install_root not in package_path.parents:
    raise SystemExit(f"source checkout shadowed installed wheel: {package_path}")

distribution = importlib.metadata.distribution("alberta-framework")
distribution_root = Path(distribution.locate_file("")).resolve()
if distribution_root != install_root:
    raise SystemExit(f"wrong distribution loaded: {distribution_root}")

entry_points = sorted(distribution.entry_points, key=lambda item: item.name)
console_scripts = [item for item in entry_points if item.group == "console_scripts"]
for entry_point in console_scripts:
    entry_point.load()

scorecard = next(
    item for item in console_scripts if item.name == "asi-reference-life-scorecard"
)
scorecard_plan = install_root / "reference-life-plan.json"
if scorecard.load()(["plan", "--output", str(scorecard_plan)]) != 0:
    raise SystemExit("installed reference-life scorecard plan command failed")
from alberta_framework.reference_life_checkpoint import _source_identity

scorecard_source = _source_identity()
source_files = scorecard_source["files"]
if not scorecard_plan.is_file():
    raise SystemExit("installed reference-life scorecard did not write its plan")
if "distribution/METADATA" not in source_files or "pyproject.toml" in source_files:
    raise SystemExit("installed reference-life source identity is not distribution-bound")

official_module = "alberta_framework.benchmarks.official_foragax"
if official_module in sys.modules:
    raise SystemExit("entry-point loading eagerly imported the official verifier")

historical = next(
    item for item in console_scripts if item.name == "alberta-historical-forager"
)
original_argv = sys.argv
sys.argv = ["alberta-historical-forager", "provenance"]
try:
    historical_output = io.StringIO()
    historical_errors = io.StringIO()
    with (
        contextlib.redirect_stdout(historical_output),
        contextlib.redirect_stderr(historical_errors),
    ):
        historical_status = historical.load()()
finally:
    sys.argv = original_argv
historical_payload = json.loads(historical_output.getvalue())

if official_module in sys.modules:
    raise SystemExit("historical inspection imported the official verifier")

strict = next(item for item in console_scripts if item.name == "alberta-foragax-oci")
original_argv = sys.argv
sys.argv = ["alberta-foragax-oci", "--help"]
try:
    strict_help = io.StringIO()
    with contextlib.redirect_stdout(strict_help):
        try:
            strict.load()()
        except SystemExit as error:
            if error.code != 0:
                raise
        else:
            raise SystemExit("official OCI help did not terminate through argparse")
finally:
    sys.argv = original_argv
if official_module in sys.modules:
    raise SystemExit("official OCI help imported the source-attested implementation")

try:
    strict.load()()
except Exception as error:
    strict_failure = {"message": str(error), "type": type(error).__name__}
else:
    raise SystemExit("scientific implementation accepted hard-linked source files")

print(
    json.dumps(
        {
            "historical_errors": historical_errors.getvalue(),
            "historical_family": historical_payload["family_id"],
            "historical_status": historical_status,
            "names": [item.name for item in console_scripts],
            "scorecard_source_files": len(source_files),
            "strict_help": strict_help.getvalue(),
            "strict_failure": strict_failure,
        }
    )
)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_directory)
    loaded = subprocess.run(
        (sys.executable, "-c", probe, str(install_directory)),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert loaded.returncode == 0, loaded.stderr
    smoke = json.loads(loaded.stdout)
    assert smoke["names"] == sorted(expected_scripts)
    assert smoke["historical_status"] == 0
    assert smoke["historical_errors"] == ""
    assert smoke["historical_family"] == HISTORICAL_FORAGER_FAMILY_ID
    assert smoke["scorecard_source_files"] > 4
    assert "{prepare,archive-cache,build,inspect,cpu-probe,emit-launch,qualify}" in smoke[
        "strict_help"
    ]
    assert smoke["strict_failure"] == {
        "message": (
            "official harness source "
            "alberta_framework/benchmarks/official_foragax.py must not have external "
            "hard-link aliases"
        ),
        "type": "OfficialForagaxValidationError",
    }

    prefix = f"alberta_framework-{version}"
    with tarfile.open(sdist_path, mode="r:gz") as source_distribution:
        sdist_names = set(source_distribution.getnames())
        for relative_path in (*_WHEEL_FILES, *_SDIST_EXTRA_FILES):
            archived_path = f"{prefix}/{relative_path}"
            assert archived_path in sdist_names
            extracted = source_distribution.extractfile(archived_path)
            assert extracted is not None
            assert extracted.read() == (_REPO_ROOT / relative_path).read_bytes()
