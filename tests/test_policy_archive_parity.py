"""Parity/robustness coverage for the bounded policy-archive primitive (#1586).

Closes gaps not covered by test_policy_archive.py: latent-width mismatch
rejection, nearest-neighbor replacement position, deterministic tie-breaking,
replacement-path byte-budget enforcement, exact-type entry validation, and
persistent-bytes accounting details (multi-byte identities, latent width).
"""

import pytest

from alberta_framework.core.policy_archive import (
    BoundedPolicyArchive,
    PolicyEntry,
)


def _entry(name: str, latent: tuple[float, ...], score: float, size: int = 4) -> PolicyEntry:
    return PolicyEntry(identity=name, policy_bytes=bytes(size), latent=latent, score=score)


def test_rejects_latent_width_mismatch() -> None:
    archive = BoundedPolicyArchive(
        byte_budget=256, min_latent_distance=0.0
    ).add(_entry("a", (0.0, 0.0), 1.0))
    with pytest.raises(ValueError, match="equal width"):
        archive.add(_entry("b", (1.0,), 2.0))


def test_replacement_keeps_nearest_position_and_order() -> None:
    # Entries at latent 0.0, 1.0, 2.0. New entry at 0.9 replaces the 1.0
    # neighbour in-place, preserving the surrounding order.
    archive = (
        BoundedPolicyArchive(byte_budget=256, min_latent_distance=0.5)
        .add(_entry("a", (0.0,), 1.0))
        .add(_entry("b", (1.0,), 1.0))
        .add(_entry("c", (2.0,), 1.0))
    )
    replaced = archive.add(_entry("d", (0.9,), 5.0))
    assert [e.identity for e in replaced.entries] == ["a", "d", "c"]
    assert replaced.entries[1].latent == (0.9,)


def test_tie_breaks_deterministically_to_first_nearest() -> None:
    # New entry at 0.4: closer to a (0.0, dist 0.4 < 0.5) than b (1.0, dist 0.6).
    # argmin picks a → replacement at index 0.
    archive = (
        BoundedPolicyArchive(byte_budget=256, min_latent_distance=0.5)
        .add(_entry("a", (0.0,), 1.0))
        .add(_entry("b", (1.0,), 1.0))
    )
    replaced = archive.add(_entry("d", (0.4,), 9.0))
    assert [e.identity for e in replaced.entries] == ["d", "b"]


def test_replacement_path_enforces_byte_budget() -> None:
    # Small low-score entry replaced by a bigger high-score neighbour that
    # would exceed the budget → must reject even though it is a "replacement".
    # persistent_bytes = identity + policy + 8*latent + 8(score).
    a = PolicyEntry(identity="a", policy_bytes=bytes(10), latent=(0.0,), score=1.0)  # 1+10+8+8=27
    b = PolicyEntry(identity="b", policy_bytes=bytes(20), latent=(0.1,), score=2.0)  # 1+20+8+8=37
    archive = BoundedPolicyArchive(byte_budget=27, min_latent_distance=0.5).add(a)
    assert archive.persistent_bytes == 27
    with pytest.raises(ValueError, match="byte budget"):
        archive.add(b)


def test_rejects_non_entry_values() -> None:
    archive = BoundedPolicyArchive(byte_budget=256, min_latent_distance=0.0)
    with pytest.raises(ValueError, match="exact PolicyEntry"):
        archive.add(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact tuple"):
        BoundedPolicyArchive(
            byte_budget=256,
            min_latent_distance=0.0,
            entries=[_entry("a", (0.0,), 1.0)],  # list, not tuple
        )


def test_archive_mode_validation() -> None:
    with pytest.raises(ValueError, match="unknown archive mode"):
        BoundedPolicyArchive(byte_budget=256, min_latent_distance=0.0, mode="bogus")


def test_persistent_bytes_accounts_multibyte_identity() -> None:
    # 3-byte UTF-8 identity + 4 policy bytes + 8 latent + 8 score = 23.
    entry = PolicyEntry(identity="策", policy_bytes=bytes(4), latent=(1.0,), score=0.5)
    assert entry.persistent_bytes == 23
    # Latin-1 identity is 1 byte each: 2 + 4 + 8 + 8 = 22.
    assert PolicyEntry(identity="ab", policy_bytes=bytes(4), latent=(1.0,), score=0.5).persistent_bytes == 22


def test_persistent_bytes_scales_with_latent_width() -> None:
    narrow = PolicyEntry(identity="x", policy_bytes=bytes(1), latent=(1.0,), score=0.5)
    wide = PolicyEntry(identity="x", policy_bytes=bytes(1), latent=(1.0, 2.0, 3.0), score=0.5)
    assert wide.persistent_bytes - narrow.persistent_bytes == 16  # 2 extra floats


def test_archive_byte_accounting_matches_entry_sum() -> None:
    archive = (
        BoundedPolicyArchive(byte_budget=256, min_latent_distance=0.5)
        .add(_entry("a", (0.0,), 1.0))
        .add(_entry("b", (1.0,), 2.0))
    )
    expected = sum(e.persistent_bytes for e in archive.entries)
    assert archive.persistent_bytes == expected


def test_zero_distance_requires_score_strictly_greater() -> None:
    # min_latent_distance=0.0 → same-latent distance 0.0 is NOT < 0.0, so
    # identical-latent entries are treated as distinct and appended (the
    # strict `<` boundary). Equal score keeps both; higher score also appends
    # in this mode because the replacement branch requires distance < threshold.
    archive = BoundedPolicyArchive(byte_budget=256, min_latent_distance=0.0).add(
        _entry("a", (0.0,), 1.0)
    )
    same_score = archive.add(_entry("b", (0.0,), 1.0))
    assert [e.identity for e in same_score.entries] == ["a", "b"]
    higher = same_score.add(_entry("c", (0.0,), 2.0))
    assert [e.identity for e in higher.entries] == ["a", "b", "c"]


def test_nearby_replacement_requires_strictly_higher_score() -> None:
    # Distance 0.1 < min 0.5 → nearby. Equal score must NOT replace; strictly
    # higher score replaces.
    archive = BoundedPolicyArchive(byte_budget=256, min_latent_distance=0.5).add(
        _entry("a", (0.0,), 1.0)
    )
    equal = archive.add(_entry("b", (0.1,), 1.0))
    assert [e.identity for e in equal.entries] == ["a"]
    better = archive.add(_entry("c", (0.1,), 2.0))
    assert [e.identity for e in better.entries] == ["c"]


def test_diverse_archive_rejects_when_entry_limit_exceeded() -> None:
    # 4096 max entries; filling is expensive, so verify the guard exists by
    # monkeypatching the module constant to a small value.
    import alberta_framework.core.policy_archive as mod

    original = mod._MAX_ARCHIVE_ENTRIES
    mod._MAX_ARCHIVE_ENTRIES = 2
    try:
        archive = (
            BoundedPolicyArchive(byte_budget=4096, min_latent_distance=2.0)
            .add(_entry("a", (0.0,), 1.0))
            .add(_entry("b", (10.0,), 1.0))
        )
        with pytest.raises(ValueError, match="entry limit"):
            archive.add(_entry("c", (20.0,), 1.0))
    finally:
        mod._MAX_ARCHIVE_ENTRIES = original


def test_latent_distance_uses_euclidean_norm() -> None:
    archive = BoundedPolicyArchive(byte_budget=256, min_latent_distance=1.5).add(
        _entry("a", (0.0, 0.0), 1.0)
    )
    # (1,1) has euclidean distance sqrt(2) ≈ 1.414 < 1.5 → nearby.
    nearby = archive.add(_entry("b", (1.0, 1.0), 0.5))
    assert [e.identity for e in nearby.entries] == ["a"]
    # (2,0) has distance 2.0 > 1.5 → distinct.
    distinct = archive.add(_entry("c", (2.0, 0.0), 0.5))
    assert [e.identity for e in distinct.entries] == ["a", "c"]
