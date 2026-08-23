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


def test_protocol_is_nonpromoting() -> None:
    assert POLICY_ARCHIVE_PROTOCOL["paper_revision"] == "arXiv:2604.15414v1"
    assert POLICY_ARCHIVE_PROTOCOL["controls"] == ("one_model", "fixed_snapshot")
    assert POLICY_ARCHIVE_PROTOCOL["scientific_promotion_allowed"] is False


class TestEqualWidthInvariant:
    """Enforce archive-wide equal latent width and control arm isolation (#2322)."""

    def test_constructor_rejects_mixed_latent_widths(self) -> None:
        narrow = _entry("a", (0.0,), 0.0)
        wide = _entry("b", (3.0, 3.0), 9.0)
        for mode in ("diverse_archive", "one_model", "fixed_snapshot"):
            with pytest.raises(ValueError, match="all latent descriptors must have equal width"):
                BoundedPolicyArchive(
                    byte_budget=4096,
                    min_latent_distance=1.0,
                    mode=mode,
                    entries=(narrow, wide),
                )

    def test_diverse_archive_add_rejects_mismatched_latent_width(self) -> None:
        archive = BoundedPolicyArchive(byte_budget=4096, min_latent_distance=1.0).add(
            _entry("a", (0.0,), 1.0)
        )
        with pytest.raises(ValueError, match="all latent descriptors must have equal width"):
            archive.add(_entry("b", (1.0, 2.0), 2.0))

    def test_controls_accept_different_latent_width(self) -> None:
        narrow = _entry("a", (0.0,), 1.0)
        wide = _entry("b", (1.0, 2.0), 2.0)

        fixed = BoundedPolicyArchive(
            byte_budget=4096, min_latent_distance=0.0, mode="fixed_snapshot"
        ).add(narrow)
        assert fixed.add(wide) is fixed
        assert [entry.identity for entry in fixed.entries] == ["a"]

        one = BoundedPolicyArchive(
            byte_budget=4096, min_latent_distance=0.0, mode="one_model"
        ).add(narrow)
        updated = one.add(wide)
        assert [entry.identity for entry in updated.entries] == ["b"]
        assert updated.entries[0].latent == (1.0, 2.0)
