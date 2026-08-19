"""CLI-level integrity checks for Forager benchmark artifacts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from alberta_framework import forager_cli

pytestmark = pytest.mark.integration


def test_protocol_evaluation_seeds_respect_synthetic_nonzero_start() -> None:
    protocol = SimpleNamespace(evaluation_seed_start=100, evaluation_seeds=3)
    assert forager_cli._protocol_evaluation_seeds(protocol) == (100, 101, 102)


def test_protocol_evaluation_seeds_reject_bool_subclass_and_negative() -> None:
    good = SimpleNamespace(evaluation_seed_start=0, evaluation_seeds=3)
    assert forager_cli._protocol_evaluation_seeds(good) == (0, 1, 2)
    for bad_start in (True, False, -1):
        with pytest.raises(ValueError, match="paper protocol declares an invalid evaluation seed interval"):
            forager_cli._protocol_evaluation_seeds(
                SimpleNamespace(evaluation_seed_start=bad_start, evaluation_seeds=3)
            )
    for bad_count in (True, False, 3.0):
        with pytest.raises(ValueError, match="paper protocol declares an invalid evaluation seed interval"):
            forager_cli._protocol_evaluation_seeds(
                SimpleNamespace(evaluation_seed_start=0, evaluation_seeds=bad_count)
            )


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


def test_explicit_int_or_default_keeps_requested_zero() -> None:
    """``0 or default`` is the live hole: argparse 0 must not become missing."""
    assert forager_cli._explicit_int_or_default(None, 100_000) == 100_000
    assert forager_cli._explicit_int_or_default(0, 100_000) == 0
    assert forager_cli._explicit_int_or_default(50_000, 100_000) == 50_000


def test_final_window_zero_is_not_replaced_by_protocol_default() -> None:
    args = forager_cli._parser().parse_args(["--final-window", "0"])
    assert args.final_window == 0
    # The previous main() expression was ``args.final_window or protocol.default``.
    assert (args.final_window or 100_000) == 100_000
    resolved = forager_cli._explicit_int_or_default(args.final_window, 100_000)
    assert resolved == 0


def test_json_safe_rejects_nonfinite_protocol_identities() -> None:
    """Protocol/baseline dumps must not coerce NaN/Inf into JSON null."""
    finite = forager_cli._json_safe({"preset": "toy", "scale": 1.0})
    assert finite == {"preset": "toy", "scale": 1.0}
    encoded = json.dumps(finite, indent=2, sort_keys=True, allow_nan=False)
    assert '"scale": 1.0' in encoded

    for bad in (float("nan"), float("inf"), float("-inf"), np.float64("nan")):
        with pytest.raises(ValueError, match="forager CLI payload is not finite JSON"):
            forager_cli._json_safe({"scale": bad})


def test_protocol_and_baseline_dumps_refuse_nonfinite_json() -> None:
    source = __import__("inspect").getsource(forager_cli.main)
    assert "allow_nan=False" in source
    assert "json.dumps(_json_safe(baseline_payload), indent=2, sort_keys=True)" not in source
    assert "json.dumps(_json_safe(protocol.to_dict()), indent=2, sort_keys=True)" not in source


def test_main_passes_explicit_final_window_to_config() -> None:
    import inspect

    source = inspect.getsource(forager_cli.main)
    assert "_explicit_int_or_default(" in source
    assert "args.final_window, protocol.final_window_steps" in source
    assert "args.final_window or protocol.final_window_steps" not in source
