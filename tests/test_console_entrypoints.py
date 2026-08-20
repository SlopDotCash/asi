"""Packaging-safe lazy boundaries for hard-link-sensitive console commands."""

from __future__ import annotations

import importlib
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
    "alberta-forager-benchmark": ("alberta_framework.console_entrypoints:forager_benchmark_main"),
    "alberta-historical-forager": ("alberta_framework.console_entrypoints:historical_forager_main"),
    "alberta-foragax-oci": ("alberta_framework.console_entrypoints:official_foragax_oci_main"),
}
_ASI_SCRIPTS = {
    "asi-nap-ipmnist": ("alberta_framework.benchmarks.nap_ipmnist:main"),
    "asi-native-supervised-catalog": (
        "alberta_framework.benchmarks.native_supervised_suite:main"
    ),
    "asi-plasticity-diagnostic": (
        "alberta_framework.benchmarks.plasticity_diagnostics:main"
    ),
    "asi-activation-feature-ipmnist": (
        "alberta_framework.benchmarks.activation_feature_ipmnist:main"
    ),
    "asi-adamo-diagnostic": "alberta_framework.benchmarks.adamo_diagnostic:main",
    "asi-adamo-matched-development": (
        "alberta_framework.benchmarks.adamo_matched_development:main"
    ),
    "asi-clear-qualification": (
        "alberta_framework.benchmarks.clear_qualification:main"
    ),
    "asi-coom-qualification-smoke": (
        "alberta_framework.benchmarks.coom_qualification:main"
    ),
    "asi-cora-catalog": ("alberta_framework.benchmarks.cora_development:main"),
    "asi-forager-rerun-preflight": (
        "alberta_framework.benchmarks.forager_scientific_rerun_preflight:main"
    ),
    "asi-telapa-qualification-smoke": (
        "alberta_framework.benchmarks.telapa_qualification:main"
    ),
    "asi-reference-life-scorecard": (
        "alberta_framework.benchmarks.reference_life_scorecard:main"
    ),
}


def test_package_import_does_not_require_optional_gymnasium() -> None:
    probe = (
        "import sys; "
        "sys.modules['gymnasium'] = None; "
        "import alberta_framework; "
        "from alberta_framework.streams import GymnasiumStream"
    )
    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_package_exports_pipeline_and_dependency_lazy_gymnasium_api() -> None:
    """Internal import failures must not silently remove advertised APIs."""
    package = importlib.import_module("alberta_framework")
    expected = {
        "AlbertaPipeline",
        "make_alberta_pipeline",
        "GymnasiumStream",
        "make_gymnasium_stream",
    }

    assert expected <= set(package.__all__)
    assert all(hasattr(package, name) for name in expected)


def test_hard_link_sensitive_scripts_use_lazy_entrypoints() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    scripts = project["project"]["scripts"]

    assert {name: scripts[name] for name in _LAZY_SCRIPTS} == _LAZY_SCRIPTS
    assert {name: scripts[name] for name in _ASI_SCRIPTS} == _ASI_SCRIPTS


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
        (
            "forager_benchmark_main",
            "alberta_framework.forager_cli",
            "main",
        ),
        (
            "historical_forager_main",
            "alberta_framework.forager_cli",
            "historical_main",
        ),
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
