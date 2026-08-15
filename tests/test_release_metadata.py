"""Release metadata must move as one versioned transaction."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

import alberta_framework

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_MARKDOWN_LINK = re.compile(
    r"!?\[[^\]\n]*\]\((?P<destination><[^>\n]+>|[^)\s]+)(?:\s+['\"].*?['\"])?\)"
)


def _citation_scalar(field: str) -> str:
    matches: list[str] = re.findall(
        rf"(?m)^{re.escape(field)}:\s*[\"']?([^\"'#\s]+)[\"']?\s*$",
        (_ROOT / "CITATION.cff").read_text(encoding="utf-8"),
    )
    if len(matches) != 1:
        raise AssertionError(f"CITATION.cff must contain exactly one scalar {field}")
    return matches[0]


def test_release_version_carriers_and_lockfile_are_synchronized() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = project["project"]["version"]
    assert isinstance(expected, str)
    assert _SEMVER.fullmatch(expected)

    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    root_versions = [
        package["version"]
        for package in lock["package"]
        if package.get("name") == "alberta-framework"
    ]

    assert alberta_framework.__version__ == expected
    assert _citation_scalar("version") == expected
    assert root_versions == [expected]
    release_date = _citation_scalar("date-released")
    assert re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", release_date)
    assert f"## [{expected}] - {release_date}" in (_ROOT / "CHANGELOG.md").read_text(
        encoding="utf-8"
    )


def test_release_repository_and_runtime_floors_are_explicit() -> None:
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert metadata["build-system"]["requires"] == [
        "hatchling==1.32.0",
        "editables==0.6",
    ]
    assert project["urls"] == {
        "Homepage": "https://github.com/elizaOS/asi",
        "Repository": "https://github.com/elizaOS/asi",
        "Issues": "https://github.com/elizaOS/asi/issues",
        "Upstream": "https://github.com/lalalune/alberta",
    }
    assert _citation_scalar("repository-code") == project["urls"]["Repository"]
    dependencies = set(project["dependencies"])
    assert {"jax>=0.7.1", "jaxlib>=0.7.1", "numpy>=1.26"} <= dependencies
    assert {
        "pandas>=2.2",
        "matplotlib>=3.8",
        "scikit-learn>=1.5",
        "joblib>=1.3",
        "tqdm>=4.66",
    }.isdisjoint(dependencies)
    research_dependencies = set(project["optional-dependencies"]["research"])
    assert research_dependencies == {
        "pandas>=2.2",
        "matplotlib>=3.8",
        "scikit-learn>=1.5",
        "joblib>=1.3",
        "tqdm>=4.66",
    }
    dev_dependencies = set(project["optional-dependencies"]["dev"])
    assert {
        "editables==0.6",
        "hatchling==1.32.0",
        "uv==0.9.24",
    } <= dev_dependencies
    assert research_dependencies <= dev_dependencies
    assert set(metadata["build-system"]["requires"]) <= dev_dependencies
    assert project["optional-dependencies"]["gpu"] == ["jax[cuda12]>=0.7.1"]


def test_readme_does_not_claim_the_external_pypi_distribution() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")

    assert "pip install alberta-framework" not in readme
    assert "git clone https://github.com/elizaOS/asi.git" in readme
    assert "existing `alberta-framework` project on PyPI is a different distribution" in readme


def test_agent_guides_are_byte_identical() -> None:
    assert (_ROOT / "AGENTS.md").read_bytes() == (_ROOT / "CLAUDE.md").read_bytes()


def test_local_markdown_links_resolve_inside_the_repository() -> None:
    documents = sorted([*_ROOT.glob("*.md"), *(_ROOT / "docs").rglob("*.md")])
    broken: list[str] = []

    for document in documents:
        source = document.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(source):
            destination = match.group("destination").removeprefix("<").removesuffix(">")
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue

            target = (document.parent / unquote(parsed.path)).resolve()
            line = source.count("\n", 0, match.start()) + 1
            location = f"{document.relative_to(_ROOT)}:{line} -> {destination}"
            if not target.is_relative_to(_ROOT.resolve()):
                broken.append(f"{location} (escapes repository)")
            elif not target.exists():
                broken.append(f"{location} (missing)")

    assert not broken, "Broken local Markdown links:\n" + "\n".join(broken)


def test_top_level_public_exports_are_unique_and_resolvable() -> None:
    exports = alberta_framework.__all__

    assert len(exports) == len(set(exports))
    assert all(hasattr(alberta_framework, name) for name in exports)
