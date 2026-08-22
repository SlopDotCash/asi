"""CLI action-list last-fit for Forager RNG parity."""

from __future__ import annotations

import argparse

import pytest

from alberta_framework.benchmarks.forager_rng_parity import (
    MAX_ACTIONS,
    MAX_ACTIONS_TEXT,
    _parse_actions_argument,
)

pytestmark = pytest.mark.unit


def test_parse_actions_argument_accepts_last_fit_single_digit_sequence() -> None:
    assert MAX_ACTIONS_TEXT == MAX_ACTIONS * 2 - 1
    text = ",".join("0" for _ in range(MAX_ACTIONS))
    assert len(text) == MAX_ACTIONS_TEXT
    assert _parse_actions_argument(text) == (0,) * MAX_ACTIONS


def test_parse_actions_argument_rejects_oversized_host_text() -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match=rf"actions length must be an integer in \[1, {MAX_ACTIONS_TEXT}\]",
    ):
        _parse_actions_argument("0" * (MAX_ACTIONS_TEXT + 1))
