"""Pytest configuration and fixtures for Alberta Framework tests."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import jax.random as jr
import pytest

# Optional dependency markers for graceful skips
pytest_plugins = []

# Check for optional dependencies and create skip markers
_FORAGAX_AVAILABLE = importlib.util.find_spec("continual_foragax") is not None
_BSUITE_AVAILABLE = importlib.util.find_spec("bsuite") is not None
_GYMNASIUM_AVAILABLE = importlib.util.find_spec("gymnasium") is not None

skipif_no_foragax = pytest.mark.skipif(
    not _FORAGAX_AVAILABLE,
    reason="continual-foragax not installed (optional: pip install -e .[forager])",
)
skipif_no_bsuite = pytest.mark.skipif(
    not _BSUITE_AVAILABLE,
    reason="bsuite not installed (optional: pip install -e .[bsuite])",
)
skipif_no_gymnasium = pytest.mark.skipif(
    not _GYMNASIUM_AVAILABLE,
    reason="gymnasium not installed (optional: pip install -e .[gymnasium])",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


# The standalone repository deliberately omits the upstream ``benchmarks/`` and
# ``examples/`` script trees. Tests that depend on those trees remain collectable:
# ``load_script`` below reports a visible skip only when the containing tree is
# absent. A missing script inside a present tree remains a hard failure.
_OPTIONAL_SCRIPT_ROOTS = (PROJECT_ROOT / "benchmarks", PROJECT_ROOT / "examples")


def pytest_terminal_summary(
    terminalreporter: object, exitstatus: int, config: pytest.Config
) -> None:
    """Surface how many 'replication' tests skipped.

    The historical Step 1/Step 2 replication suites gate every test on
    optional upstream JSON artifacts. This fork ships neither the artifacts
    nor their generator trees, so a standalone checkout skips 100% unless
    compatible historical artifacts are supplied. This hook makes that
    visible without treating the optional regression replays as current
    scientific evidence.
    """
    reporter = terminalreporter  # pytest's TerminalReporter (untyped here)
    skipped = reporter.stats.get("skipped", [])  # type: ignore[attr-defined]
    replication_skips = [
        rep
        for rep in skipped
        if getattr(rep, "keywords", None) is not None and "replication" in rep.keywords
    ]
    if replication_skips:
        reporter.write_sep(  # type: ignore[attr-defined]
            "-",
            f"replication suites: {len(replication_skips)} test(s) skipped "
            "because optional historical upstream artifacts are absent "
            "(outputs/step1_canonical/, outputs/step2_canonical/)",
        )


def load_script(path: Path, name: str) -> ModuleType:
    """Load a Python script by filesystem path (works with paths containing spaces).

    Used by tests that exercise example and benchmark scripts whose paths
    cannot be imported with a normal ``import`` statement. Standalone-checkout
    omissions are visible skips; an unexpectedly missing script still fails.
    """
    if not path.is_file():
        resolved_path = path.resolve(strict=False)
        for root in _OPTIONAL_SCRIPT_ROOTS:
            resolved_root = root.resolve(strict=False)
            if resolved_path.is_relative_to(resolved_root) and not root.is_dir():
                relative_path = resolved_path.relative_to(PROJECT_ROOT.resolve())
                pytest.skip(
                    f"{relative_path} unavailable because the standalone checkout "
                    f"omits {root.name}/ (see VENDORING.md)",
                    allow_module_level=True,
                )
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def rng_key():
    """Provide a deterministic JAX random key."""
    return jr.key(42)


@pytest.fixture
def feature_dim():
    """Default feature dimension for tests."""
    return 10


@pytest.fixture
def sample_observation(feature_dim, rng_key):
    """Generate a sample observation vector."""
    return jr.normal(rng_key, (feature_dim,), dtype=jnp.float32)


@pytest.fixture
def sample_target():
    """Generate a sample target value."""
    return jnp.array([1.5], dtype=jnp.float32)
