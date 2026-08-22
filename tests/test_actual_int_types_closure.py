"""Repository-wide closure of the house integer-type gate over numpy C aliases.

Every module that guards a boundary with ``type(value) not in _ACTUAL_INT_TYPES``
must admit the same integer families, otherwise the same value passes one gate and
is rejected by the next. numpy exposes several spellings of one width, and which
of them are distinct type objects is platform-dependent: on 64-bit Windows
``np.intc is np.int32`` is False, so a set that lists only fixed-width names
rejects a valid C ``int`` there while passing on Linux.

The ledger below records the modules that still have the gap because they are
registered evidence sources whose hashes are load-bearing; repairing one
invalidates its pinned artifact and is a maintainer decision, not a drive-by fix.
"""

import importlib
import pkgutil

import numpy as np
import pytest

import alberta_framework

pytestmark = pytest.mark.unit

# Codes for every integer scalar numpy exposes: signed and unsigned char, short,
# C int, C long, long long, and pointer-width. The canonical set in
# core/types.py derives its members from exactly these.
_INTEGER_DTYPE_CODES = "bBhHiIlLqQpP"
_REQUIRED_TYPES = frozenset(np.dtype(code).type for code in _INTEGER_DTYPE_CODES)

# Where the C aliases collapse onto the fixed-width names, a hand-enumerated set is
# accidentally complete, so an unrepaired gate is indistinguishable from a repaired
# one and only the regression direction of the ledger can be checked.
_C_ALIASES_ARE_DISTINCT = np.intc is not np.int32 or np.uintc is not np.uint32

# Registered evidence sources: hash-pinned, so a contributor may not edit them.
_REGISTERED_SOURCE_LEDGER = frozenset(
    {
        "alberta_framework.core.average_reward",
        "alberta_framework.core.ftl_world_model",
        "alberta_framework.core.intelligence_amplification",
        "alberta_framework.core.interaction_features",
        "alberta_framework.core.oak",
        "alberta_framework.core.options",
        "alberta_framework.evaluation.continual_multiagent",
        "alberta_framework.evaluation.ftl_decision_fidelity",
        "alberta_framework.utils.metrics",
    }
)


def _walk_gates() -> tuple[dict[str, frozenset[type]], dict[str, str]]:
    """Import every package module, returning the gates found and the modules that failed."""
    found: dict[str, frozenset[type]] = {}
    unimportable: dict[str, str] = {}
    for info in pkgutil.walk_packages(alberta_framework.__path__, "alberta_framework."):
        try:
            module = importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 - recorded below, not swallowed
            unimportable[info.name] = f"{type(exc).__name__}: {exc}"
            continue
        members = getattr(module, "_ACTUAL_INT_TYPES", None)
        if members is not None:
            found[info.name] = frozenset(members)
    return found, unimportable


_GATES, _UNIMPORTABLE = _walk_gates()


def test_gate_inventory_is_trustworthy() -> None:
    """A module that dropped out of the walk must not look like a gate that passed."""
    assert "alberta_framework.core.types" in _GATES
    assert len(_GATES) >= 80, (
        f"expected the package to define many gates, found {len(_GATES)}; "
        f"{len(_UNIMPORTABLE)} modules did not import: {sorted(_UNIMPORTABLE)[:5]}"
    )
    unchecked = sorted(_REGISTERED_SOURCE_LEDGER - set(_GATES))
    assert not unchecked, (
        "the ledger cannot be checked because these modules did not import or no longer "
        f"define the gate: {[(name, _UNIMPORTABLE.get(name, 'no gate')) for name in unchecked]}"
    )


@pytest.mark.parametrize("module_name", sorted(set(_GATES) - _REGISTERED_SOURCE_LEDGER))
def test_gate_admits_every_integer_family(module_name: str) -> None:
    missing = sorted(
        scalar.__name__ for scalar in _REQUIRED_TYPES if scalar not in _GATES[module_name]
    )
    assert not missing, (
        f"{module_name}._ACTUAL_INT_TYPES rejects {', '.join(missing)}; derive the set from "
        f'frozenset({{int, *(np.dtype(code).type for code in "{_INTEGER_DTYPE_CODES}")}}) '
        "as core/types.py does instead of enumerating fixed-width names"
    )


def test_gate_admits_python_int() -> None:
    for module_name, members in sorted(_GATES.items()):
        assert int in members, f"{module_name}._ACTUAL_INT_TYPES omits the builtin int"


def _incomplete_gates() -> set[str]:
    return {name for name, members in _GATES.items() if not _REQUIRED_TYPES <= members}


def test_no_unregistered_gate_is_incomplete() -> None:
    """The ledger is a record of blocked repairs, not a place to park new drift."""
    unexpected = sorted(_incomplete_gates() - _REGISTERED_SOURCE_LEDGER)
    assert not unexpected, f"new gates regressed and are not registered sources: {unexpected}"


def test_registered_source_ledger_has_no_stale_entry() -> None:
    """A ledger entry must be dropped once its module no longer needs the exemption."""
    if not _C_ALIASES_ARE_DISTINCT:
        pytest.skip("np.intc is np.int32 here, so every enumerated set is already complete")
    resolved = sorted(
        name for name in _REGISTERED_SOURCE_LEDGER & set(_GATES) if name not in _incomplete_gates()
    )
    assert not resolved, (
        "these ledger entries are now closed, so remove them from "
        f"_REGISTERED_SOURCE_LEDGER: {resolved}"
    )


@pytest.mark.parametrize("code", list(_INTEGER_DTYPE_CODES))
def test_scalar_of_each_family_passes_a_representative_gate(code: str) -> None:
    """A value of every family must survive the gate a public constructor uses."""
    from alberta_framework.core.stacked_horde import _require_int32

    scalar = np.dtype(code).type(4)
    assert _require_int32("value", scalar, minimum=0) == 4
