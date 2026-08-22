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


def test_constructor_enforces_equal_latent_width() -> None:
    narrow = _entry("narrow", (3.0,), 1.0)
    wide = _entry("wide", (3.0, 3.0), 2.0)
    for mode in ("diverse_archive", "one_model", "fixed_snapshot"):
        with pytest.raises(ValueError, match="all latent descriptors must have equal width"):
            BoundedPolicyArchive(
                byte_budget=4096,
                min_latent_distance=1.0,
                mode=mode,  # type: ignore[arg-type]
                entries=(narrow, wide),
            )
    # A same-width tuple of entries still constructs cleanly.
    ok = BoundedPolicyArchive(
        byte_budget=4096,
        min_latent_distance=1.0,
        entries=(narrow, _entry("other", (9.0,), 2.0)),
    )
    assert [entry.identity for entry in ok.entries] == ["narrow", "other"]


def test_diverse_archive_retains_distant_candidate() -> None:
    # Reproduction from the issue: two same-width entries and a candidate that is
    # genuinely 3.0 away from every existing descriptor and clears the distance gate.
    diverse_archive = BoundedPolicyArchive(byte_budget=4096, min_latent_distance=1.0)
    diverse_archive = diverse_archive.add(_entry("a", (0.0,), 1.0))
    diverse_archive = diverse_archive.add(_entry("b", (1.0,), 1.0))
    updated = diverse_archive.add(_entry("far", (4.0,), 0.5))
    assert updated is not diverse_archive
    assert "far" in [entry.identity for entry in updated.entries]
    assert len(updated.entries) == 3


def test_control_arms_accept_wider_entry() -> None:
    wide = _entry("wide", (3.0, 3.0), 2.0)
    fixed = BoundedPolicyArchive(
        byte_budget=4096, min_latent_distance=0.0, mode="fixed_snapshot"
    ).add(_entry("a", (0.0,), 1.0))
    assert fixed.add(wide) is fixed
    one = BoundedPolicyArchive(
        byte_budget=4096, min_latent_distance=0.0, mode="one_model"
    ).add(_entry("a", (0.0,), 1.0))
    replaced = one.add(wide)
    assert tuple(entry.identity for entry in replaced.entries) == ("wide",)
    assert len(replaced.entries) == 1


def test_diverse_archive_still_rejects_mismatched_width() -> None:
    diverse_archive = BoundedPolicyArchive(byte_budget=4096, min_latent_distance=1.0)
    diverse_archive = diverse_archive.add(_entry("a", (0.0,), 1.0))
    diverse_archive = diverse_archive.add(_entry("b", (1.0,), 1.0))
    with pytest.raises(ValueError, match="all latent descriptors must have equal width"):
        diverse_archive.add(_entry("wide", (3.0, 3.0), 2.0))


def test_ill_typed_entry_reports_tuple_message_before_width() -> None:
    good = _entry("a", (0.0,), 1.0)
    with pytest.raises(ValueError, match="exact tuple of PolicyEntry values"):
        BoundedPolicyArchive(
            byte_budget=4096,
            min_latent_distance=0.0,
            entries=(good, "not-an-entry"),  # type: ignore[arg-type]
        )


def test_protocol_is_nonpromoting() -> None:
    assert POLICY_ARCHIVE_PROTOCOL["paper_revision"] == "arXiv:2604.15414v1"
    assert POLICY_ARCHIVE_PROTOCOL["controls"] == ("one_model", "fixed_snapshot")
    assert POLICY_ARCHIVE_PROTOCOL["scientific_promotion_allowed"] is False
