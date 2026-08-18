"""Reproducibility contract for the scripts shipped under ``outputs/``.

Reports cite these scripts as the way to reproduce their numbers — for example
``CEILING_ANALYSIS.md`` names ``ceiling_runs.py`` as its runner and
``ceiling_analyze.py`` as its analyzer. A script that resolves its inputs
through one machine's absolute home directory cannot serve that purpose.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUTS = _REPO_ROOT / "outputs"
_ABSOLUTE_HOME = re.compile(r"""["'](/home/|/Users/)""")


def _load(relative: str) -> ModuleType:
    path = _REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:  # pragma: no cover - argparse-driven scripts
        pass
    return module


def test_no_outputs_script_hardcodes_an_absolute_home_directory() -> None:
    """Shipped scripts must not resolve inputs through a developer's home."""
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in sorted(_OUTPUTS.rglob("*.py"))
        if _ABSOLUTE_HOME.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("relative", "attribute"),
    (
        ("outputs/ipmnist_screening/ceiling/ceiling_runs.py", "OUT"),
        ("outputs/ipmnist_screening/ceiling/ceiling_analyze.py", "OUT"),
        ("outputs/ipmnist_screening/ceiling/ceiling_analyze.py", "CONFIRM"),
        ("outputs/rule_discovery/write_real_screen_summary.py", "BASE"),
    ),
)
def test_ceiling_and_screen_scripts_resolve_inside_the_repository(
    relative: str,
    attribute: str,
) -> None:
    """The directories these scripts read must exist in a fresh checkout."""
    resolved = Path(getattr(_load(relative), attribute))

    assert resolved.is_dir()
    assert resolved.is_relative_to(_REPO_ROOT)


def test_write_real_screen_summary_targets_its_own_output_directory() -> None:
    """The summary writer's output path must land in the repository."""
    out = Path(_load("outputs/rule_discovery/write_real_screen_summary.py").OUT)

    assert out.parent.is_dir()
    assert out.parent.is_relative_to(_REPO_ROOT)
    assert out.name == "real_screen_v1.json"
