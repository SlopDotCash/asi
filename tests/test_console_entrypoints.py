"""Packaging-safe lazy boundaries for source-attested console commands."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_LAZY_SCRIPTS = {
    "alberta-forager-benchmark": (
        "alberta_framework.console_entrypoints:forager_benchmark_main"
    ),
    "alberta-historical-forager": (
        "alberta_framework.console_entrypoints:historical_forager_main"
    ),
    "alberta-foragax-oci": (
        "alberta_framework.console_entrypoints:official_foragax_oci_main"
    ),
}


def test_source_attested_scripts_use_lazy_entrypoints() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    scripts = project["project"]["scripts"]

    assert {name: scripts[name] for name in _LAZY_SCRIPTS} == _LAZY_SCRIPTS


def test_importing_lazy_entrypoints_does_not_import_scientific_implementations() -> None:
    probe = textwrap.dedent(
        """
        import sys

        import alberta_framework.console_entrypoints

        forbidden = {
            "alberta_framework.forager_cli",
            "alberta_framework.benchmarks.official_foragax",
            "alberta_framework.benchmarks.official_foragax_oci",
        }
        imported = sorted(forbidden.intersection(sys.modules))
        if imported:
            raise SystemExit(f"eager scientific imports: {imported}")
        """
    )

    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("wrapper_name", "module_name", "target_name"),
    (
        ("forager_benchmark_main", "alberta_framework.forager_cli", "main"),
        ("historical_forager_main", "alberta_framework.forager_cli", "historical_main"),
        (
            "official_foragax_oci_main",
            "alberta_framework.benchmarks.official_foragax_oci",
            "main",
        ),
    ),
)
def test_lazy_entrypoints_delegate_to_original_commands(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_name: str,
    module_name: str,
    target_name: str,
) -> None:
    entrypoints = importlib.import_module("alberta_framework.console_entrypoints")
    target_module = ModuleType(module_name)
    calls = 0

    def target() -> int:
        nonlocal calls
        calls += 1
        return 37

    setattr(target_module, target_name, target)
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    monkeypatch.setitem(sys.modules, module_name, target_module)
    monkeypatch.setattr(parent, child_name, target_module, raising=False)

    assert getattr(entrypoints, wrapper_name)() == 37
    assert calls == 1


def test_hardlinked_wheel_oci_help_is_inert_but_execution_fails_closed(
    tmp_path: Path,
) -> None:
    """Installed help must not import source-attested, hard-linked modules."""

    site_packages = tmp_path / "site-packages"
    package = site_packages / "alberta_framework"
    shutil.copytree(_ROOT / "alberta_framework", package)
    harness_sources = (
        package / "benchmarks" / "official_foragax.py",
        package / "benchmarks" / "runtime_profile.py",
        package / "benchmarks" / "forager_results.py",
    )
    for source in harness_sources:
        os.link(source, source.with_name(f".{source.name}.wheel-cache-link"))

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    probe = textwrap.dedent(
        """
        import contextlib
        import io
        import sys

        from alberta_framework.console_entrypoints import official_foragax_oci_main

        cache_root, output = sys.argv[1:]
        help_cases = {
            ("--help",): (
                "{prepare,archive-cache,build,inspect,cpu-probe,emit-launch,qualify}"
            ),
            ("prepare", "--help"): "--source-archive SOURCE_ARCHIVE",
            ("archive-cache", "--help"): "--cache-root CACHE_ROOT",
            ("build", "--help"): "--context CONTEXT",
            ("inspect", "--help"): "--image IMAGE",
            ("cpu-probe", "--help"): "--image-id IMAGE_ID",
            ("emit-launch", "--help"): "--entrypoint ENTRYPOINT",
            ("qualify", "--help"): "--workload-identity WORKLOAD_IDENTITY",
        }
        for arguments, expected_help in help_cases.items():
            sys.argv = ["alberta-foragax-oci", *arguments]
            rendered = io.StringIO()
            try:
                with contextlib.redirect_stdout(rendered):
                    official_foragax_oci_main()
            except SystemExit as exc:
                if exc.code != 0:
                    raise
            else:
                raise SystemExit("help did not exit")
            if expected_help not in rendered.getvalue():
                raise SystemExit(f"missing help text for {arguments!r}")
        print("all OCI help cases passed")

        sys.argv = [
            "alberta-foragax-oci",
            "archive-cache",
            "--cache-root",
            cache_root,
            "--output",
            output,
        ]
        raise SystemExit(official_foragax_oci_main())
        """
    )
    execution = subprocess.run(
        (
            sys.executable,
            "-c",
            probe,
            str(tmp_path / "cache"),
            str(tmp_path / "cache.tar"),
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert execution.returncode != 0
    assert "all OCI help cases passed" in execution.stdout
    assert "must not have external hard-link aliases" in execution.stderr
    assert not (tmp_path / "cache.tar").exists()
