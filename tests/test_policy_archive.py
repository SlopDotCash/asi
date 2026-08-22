import numpy as np
import pytest

from alberta_framework.core.policy_archive import (
    POLICY_ARCHIVE_PROTOCOL,
    BoundedPolicyArchive,
    PolicyEntry,
)


def _entry(name: str, latent: tuple[float, ...], score: float, size: int = 4) -> PolicyEntry:
    return PolicyEntry(identity=name, policy_bytes=bytes(size), latent=latent, score=score)


def test_diverse_archive_respects_exact_byte_budget() -> None:
    archive = BoundedPolicyArchive(byte_budget=58, min_latent_distance=0.5)
    archive = archive.add(_entry("a", (0.0, 0.0), 1.0))
    archive = archive.add(_entry("b", (1.0, 0.0), 2.0))
    assert archive.persistent_bytes == 58
    with pytest.raises(ValueError, match="byte budget"):
        archive.add(_entry("c", (0.0, 1.0), 3.0))


def test_nearby_policy_only_replaces_on_higher_score() -> None:
    archive = BoundedPolicyArchive(byte_budget=32, min_latent_distance=0.5).add(
        _entry("a", (0.0,), 1.0)
    )
    assert archive.add(_entry("low", (0.1,), 0.5)) == archive
    improved = archive.add(_entry("high", (0.1,), 2.0))
    assert [entry.identity for entry in improved.entries] == ["high"]


def test_one_model_and_fixed_snapshot_controls() -> None:
    one = BoundedPolicyArchive(byte_budget=21, min_latent_distance=0.0, mode="one_model")
    one = one.add(_entry("a", (0.0,), 1.0)).add(_entry("b", (1.0,), 0.0))
    assert [entry.identity for entry in one.entries] == ["b"]
    fixed = BoundedPolicyArchive(byte_budget=21, min_latent_distance=0.0, mode="fixed_snapshot")
    fixed = fixed.add(_entry("a", (0.0,), 1.0)).add(_entry("b", (1.0,), 2.0))
    assert [entry.identity for entry in fixed.entries] == ["a"]


def test_archive_rejects_duplicate_identity() -> None:
    archive = BoundedPolicyArchive(byte_budget=42, min_latent_distance=0.0).add(
        _entry("a", (0.0,), 1.0)
    )
    with pytest.raises(ValueError, match="identity already exists"):
        archive.add(_entry("a", (1.0,), 2.0))


def test_archive_preflights_host_dimensions() -> None:
    with pytest.raises(ValueError, match="256 MiB"):
        BoundedPolicyArchive(byte_budget=256 * 1024 * 1024 + 1, min_latent_distance=0.0)
    with pytest.raises(ValueError, match="identity"):
        _entry("x" * 1025, (0.0,), 1.0)
    with pytest.raises(ValueError, match="UTF-8"):
        _entry("\ud800", (0.0,), 1.0)
    entry = _entry("a", (0.0,), 1.0)
    with pytest.raises(ValueError, match="exact tuple"):
        BoundedPolicyArchive(
            byte_budget=256 * 1024 * 1024,
            min_latent_distance=0.0,
            entries=(entry,) * 4097,
        )


@pytest.mark.parametrize("mode", ("diverse_archive", "one_model", "fixed_snapshot"))
def test_constructor_enforces_the_declared_equal_width_invariant(mode: str) -> None:
    """``add`` states the invariant archive-wide, so construction must hold it."""
    with pytest.raises(ValueError, match="equal width"):
        BoundedPolicyArchive(
            byte_budget=4096,
            min_latent_distance=1.0,
            mode=mode,  # type: ignore[arg-type]
            entries=(_entry("a", (0.0,), 0.0), _entry("b", (3.0, 3.0), 9.0)),
        )


def test_mixed_width_entries_cannot_reach_the_distance_computation() -> None:
    """A width-1 latent broadcasts against a width-2 latent to a distance of 0.0.

    Reaching ``add``'s nearest-neighbour search with a mixed-width archive turns a
    shape violation into a wrong result: the candidate below is 3.0 away from every
    entry of its own width, yet the broadcast pairing reports 0.0 and discards it.
    """
    wide = _entry("b", (3.0, 3.0), 9.0)
    assert float(np.linalg.norm(np.asarray((3.0,)) - np.asarray(wide.latent))) == 0.0
    with pytest.raises(ValueError, match="equal width"):
        BoundedPolicyArchive(
            byte_budget=4096,
            min_latent_distance=1.0,
            entries=(_entry("a", (0.0,), 0.0), wide),
        )
    well_formed = BoundedPolicyArchive(
        byte_budget=4096, min_latent_distance=1.0, entries=(_entry("a", (0.0,), 0.0),)
    )
    assert [entry.identity for entry in well_formed.add(_entry("c", (3.0,), 1.0)).entries] == [
        "a",
        "c",
    ]


@pytest.mark.parametrize(("mode", "retained"), (("fixed_snapshot", "a"), ("one_model", "b")))
def test_single_policy_controls_accept_a_wider_entry(mode: str, retained: str) -> None:
    """Neither control compares descriptors, so neither can hold a mixed-width state."""
    archive = BoundedPolicyArchive(
        byte_budget=4096,
        min_latent_distance=0.0,
        mode=mode,  # type: ignore[arg-type]
    ).add(_entry("a", (0.0,), 1.0))
    successor = archive.add(_entry("b", (0.0, 0.0), 2.0))
    assert [entry.identity for entry in successor.entries] == [retained]


def test_width_guard_still_rejects_a_mismatched_candidate() -> None:
    archive = BoundedPolicyArchive(byte_budget=4096, min_latent_distance=0.5).add(
        _entry("a", (0.0, 0.0), 1.0)
    )
    with pytest.raises(ValueError, match="equal width"):
        archive.add(_entry("b", (0.0,), 2.0))


def test_entry_type_is_checked_before_its_latent_width() -> None:
    with pytest.raises(ValueError, match="exact tuple"):
        BoundedPolicyArchive(
            byte_budget=4096,
            min_latent_distance=0.0,
            entries=("a",),  # type: ignore[arg-type]
        )


def test_protocol_is_nonpromoting() -> None:
    assert POLICY_ARCHIVE_PROTOCOL["paper_revision"] == "arXiv:2604.15414v1"
    assert POLICY_ARCHIVE_PROTOCOL["controls"] == ("one_model", "fixed_snapshot")
    assert POLICY_ARCHIVE_PROTOCOL["scientific_promotion_allowed"] is False
