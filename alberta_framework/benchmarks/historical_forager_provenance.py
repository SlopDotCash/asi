"""Fail-closed provenance for the reconstructed NumPy Forager family.

The paper agents declared an unpinned Git dependency on the environment.  The
facts below therefore identify the strongest audited reconstruction; they do
not attest which environment revision was installed for archived paper runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

__all__ = [
    "CURRENT_FORAGAX_055_FAMILY_ID",
    "HISTORICAL_FORAGER_FAMILY_ID",
    "HISTORICAL_FORAGER_PROVENANCE_SCHEMA",
    "HISTORICAL_FORAGER_PROVENANCE_SHA256",
    "HistoricalForagerFamilyMismatchError",
    "HistoricalForagerProvenanceError",
    "assert_historical_family_pairing",
    "historical_forager_provenance",
    "validate_historical_forager_provenance",
]

HISTORICAL_FORAGER_FAMILY_ID: Final = "historical_numpy_forager_d140_reconstructed"
CURRENT_FORAGAX_055_FAMILY_ID: Final = "current_foragax_0_55"
HISTORICAL_FORAGER_PROVENANCE_SCHEMA: Final = "alberta.historical_numpy_forager.provenance.v1"

_HISTORICAL_FORAGER_PROVENANCE: Final[dict[str, Any]] = {
    "schema_version": HISTORICAL_FORAGER_PROVENANCE_SCHEMA,
    "family_id": HISTORICAL_FORAGER_FAMILY_ID,
    "environment_resolution_attested": False,
    "resolution_status": ("strongest_audited_reconstruction_not_attestation_of_archived_run_bits"),
    "agents": {
        "repository_url": "https://github.com/steventango/forager-agents",
        "commit": "696b3a06fbd0dc72407556b039d219e704ec6992",
        "tree": "4936577cba549a3ffb4dec69bff722360c52f8be",
        "commit_timestamp": "2025-08-21T11:03:18-06:00",
        "archive_format": "git archive --format=tar HEAD",
        "archive_sha256": ("a66ee0f7dd565dd64f5959520587e7121e5195d2814ef83504d8fc2b341d4803"),
        "environment_dependency": ("foragerenv @ git+https://github.com/steventango/forager"),
        "environment_dependency_revision_pinned": False,
        "lockfile_present": False,
        "files": {
            "pyproject.toml": ("5c9ea49608eeb34dc990cccc99cc8dd678e97f2fded6e146597508e8ec83c33b"),
            "src/continuing_main.py": (
                "1154447edd05ff06eb2fc66d2bfdc8923fe3db58742720ced88b8cb4dc327494"
            ),
            "src/environments/ForagerTwoBiomeLarge.py": (
                "4bfa26561fb869a0bd86053b0723d4fba684d44d9a17f95fa95623891e6c8d58"
            ),
            "src/problems/ForagerTwoBiomeLarge.py": (
                "bcd4a5ac03723e506e26e445f43e00e0b7454475eefb6802c74a3cea9ea0176c"
            ),
            "experiments/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-9.json": (
                "84a935879caf14b5efd74ecaf73a9f6f51028e697fe099464ce4e6a85f2201a3"
            ),
        },
    },
    "environment": {
        "repository_url": "https://github.com/steventango/forager",
        "commit": "d140bdb3c51c7b6747d0588078ca97a67b55a8e1",
        "tree": "0eb78e64b34cce3222215ebee3b94de2d83d41ce",
        "commit_timestamp": "2025-02-28T17:41:24-07:00",
        "archive_format": "git archive --format=tar HEAD",
        "archive_sha256": ("2b7caf0a83b741404a88dfbb427f34f92e822d25901b4c9a71667d6e24cf14dd"),
        "reconstructed_wheel_filename": "foragerenv-0.0.0-py3-none-any.whl",
        "reconstructed_wheel_sha256": (
            "9fcf134767a73337d36d6dec9c25721da68fd1b9587ea4c4299a3cdc00fc2020"
        ),
        "reconstructed_wheel_is_historical_attestation": False,
        "files": {
            "pyproject.toml": ("8b1fc1be72ca5ca6371cd86ffc5a2a8e0e6d1b00fca0286de199a370c32eba5d"),
            "requirements.txt": (
                "48ff9d8cbb57ab8593d063c7ff285ea07358c3abc0bb260ee6f19edca772cffd"
            ),
            "forager/Env.py": ("b55d1be7a7f5ec9b2a35e29640df86ebb455c240423a3c449047318f6cda5ba0"),
            "forager/ObjectStorage.py": (
                "96f4fa83b8347a92533ec7ee30873e9fb00e22602b7ff54e6ec9df1687e6e5cb"
            ),
            "forager/__init__.py": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "forager/_utils/__init__.py": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "forager/_utils/config.py": (
                "c8e7978df02b9260e55584f63f9645749c4116d7cc0fd7ddae415de927561ed3"
            ),
            "forager/_utils/numba.py": (
                "df093e27f32d0018fb2e8763541b38529cd6230fe1efd4e88d0ea3d106ed47f0"
            ),
            "forager/colors.py": (
                "af7d3774637c24831ead9603d4ed81a3b4b7f2f3e176fbe978354496d2acb799"
            ),
            "forager/config.py": (
                "a834cd369022cb6cf22d048bf6c85fbf4f4eb58cf9027fa4a3e5d86f076ce44c"
            ),
            "forager/exceptions.py": (
                "2bc4db3656c2b3935396b00f1b7ab5179ed34600c1745df560b9ebf22aee8a29"
            ),
            "forager/grid.py": ("baad2996b272b46875f30eddba51cea25289a243db2ad3ba8420694e35f41fd6"),
            "forager/interface.py": (
                "9b27de67d892fa76a023c985e3c47ab970794bdbf565591a5f763aafec633f18"
            ),
            "forager/logger.py": (
                "c71c926fd2ed81a1e83ae3a5260b9bff406673e259417ee25d440809248c3508"
            ),
            "forager/objects.py": (
                "134d12604b4e60b6fd5f6ba529ada16cf6464417a8f7021cd88f1b90b56f644f"
            ),
            "forager/observations.py": (
                "7ecb7240b473d4313552eb3a81839453cefed223613aa25d9f50b6a28a9f4ae6"
            ),
            "forager/py.typed": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
    },
    "audited_compatibility_runtime": {
        "python": "3.12",
        "numpy": "1.26.4",
        "numba": "0.59.1",
        "runtime_is_historical_attestation": False,
    },
}


class HistoricalForagerProvenanceError(ValueError):
    """Raised when reconstructed-family provenance is missing or altered."""


class HistoricalForagerFamilyMismatchError(ValueError):
    """Raised when a comparison attempts to cross environment families."""


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise HistoricalForagerProvenanceError("name must be a string")
    if type(value) is not str:
        raise HistoricalForagerProvenanceError(f"{name} must be a string")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HistoricalForagerProvenanceError(
            "historical Forager provenance must be finite canonical JSON"
        ) from exc


_CANONICAL_PROVENANCE_BYTES: Final = _canonical_json_bytes(_HISTORICAL_FORAGER_PROVENANCE)
HISTORICAL_FORAGER_PROVENANCE_SHA256: Final = hashlib.sha256(
    _CANONICAL_PROVENANCE_BYTES
).hexdigest()


def historical_forager_provenance() -> dict[str, Any]:
    """Return a detached copy of the exact reconstructed-family provenance."""
    value = json.loads(_CANONICAL_PROVENANCE_BYTES)
    if not isinstance(value, dict):  # pragma: no cover - module constant invariant
        raise RuntimeError("canonical historical Forager provenance is not an object")
    return value


def validate_historical_forager_provenance(value: Mapping[str, Any]) -> None:
    """Require byte-equivalent canonical provenance, including explicit false claims."""
    if not isinstance(value, Mapping):
        raise HistoricalForagerProvenanceError("historical Forager provenance must be a mapping")
    if _canonical_json_bytes(value) != _CANONICAL_PROVENANCE_BYTES:
        raise HistoricalForagerProvenanceError(
            "historical Forager provenance differs from the audited reconstruction"
        )


def assert_historical_family_pairing(left_family_id: object, right_family_id: object) -> None:
    """Reject cross-family and unknown-family pairing before metric comparison."""
    host_left = _require_exact_str("left_family_id", left_family_id)
    host_right = _require_exact_str("right_family_id", right_family_id)
    if host_left != HISTORICAL_FORAGER_FAMILY_ID or host_right != HISTORICAL_FORAGER_FAMILY_ID:
        raise HistoricalForagerFamilyMismatchError(
            "historical reconstructed results pair only with "
            f"'{HISTORICAL_FORAGER_FAMILY_ID}'; received "
            f"'{host_left}' and '{host_right}'"
        )
