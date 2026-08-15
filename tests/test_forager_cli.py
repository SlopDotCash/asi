"""CLI-level integrity checks for Forager benchmark artifacts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Skip entire module if foragax not available
pytest.importorskip("continual_foragax")

from alberta_framework import forager_cli

pytestmark = pytest.mark.integration


def test_protocol_evaluation_seeds_respect_synthetic_nonzero_start() -> None:
    protocol = SimpleNamespace(evaluation_seed_start=100, evaluation_seeds=3)
    assert forager_cli._protocol_evaluation_seeds(protocol) == (100, 101, 102)


def test_stable_runtime_provenance_records_matching_start_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        forager_cli,
        "_runtime_provenance",
        lambda: {"source_tree_sha256": "a" * 64},
    )

    provenance = forager_cli._stable_runtime_provenance("a" * 64)

    assert provenance["source_tree_sha256"] == "a" * 64
    assert provenance["source_tree_sha256_at_start"] == "a" * 64


def test_stable_runtime_provenance_rejects_mid_run_source_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        forager_cli,
        "_runtime_provenance",
        lambda: {"source_tree_sha256": "b" * 64},
    )

    with pytest.raises(RuntimeError, match="source tree changed during execution"):
        forager_cli._stable_runtime_provenance("a" * 64)


def test_parser_exposes_full_alberta_and_recurrent_variant_controls() -> None:
    args = forager_cli._parser().parse_args(
        [
            "--actor-hidden-sizes",
            "32,16",
            "--critic-hidden-sizes",
            "64",
            "--actor-step-size",
            "0.003",
            "--critic-lambda",
            "0.8",
            "--recurrent-hidden-size",
            "192",
            "--recurrent-scale",
            "0.95",
        ]
    )

    assert args.actor_hidden_sizes == (32, 16)
    assert args.critic_hidden_sizes == (64,)
    assert args.actor_step_size == pytest.approx(0.003)
    assert args.critic_lambda == pytest.approx(0.8)
    assert args.recurrent_hidden_size == 192
    assert args.recurrent_scale == pytest.approx(0.95)


def test_parser_exposes_causal_map_policy() -> None:
    args = forager_cli._parser().parse_args(
        ["--preset", "field_of_view", "--agent", "causal-map"]
    )

    assert args.preset == "field_of_view"
    assert args.agents == ["causal-map"]
