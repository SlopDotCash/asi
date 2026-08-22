"""Reject oversized Forager CLI comma-separated lists before walk hang."""

from __future__ import annotations

import argparse

import pytest

from alberta_framework.forager_cli import (
    _MAX_FORAGER_CLI_SEQUENCE,
    _parse_floats,
    _parse_ints,
)


def test_forager_cli_sequence_cap_constant() -> None:
    assert _MAX_FORAGER_CLI_SEQUENCE == 4096


def test_forager_cli_accepts_max_unique_int_sequence() -> None:
    payload = ",".join(str(index) for index in range(_MAX_FORAGER_CLI_SEQUENCE))
    assert _parse_ints(payload) == tuple(range(_MAX_FORAGER_CLI_SEQUENCE))


def test_forager_cli_rejects_oversized_int_sequence() -> None:
    payload = ",".join(str(index) for index in range(_MAX_FORAGER_CLI_SEQUENCE + 1))
    with pytest.raises(argparse.ArgumentTypeError, match="sequence length"):
        _parse_ints(payload)


def test_forager_cli_rejects_oversized_float_sequence() -> None:
    payload = ",".join(["0.5"] * (_MAX_FORAGER_CLI_SEQUENCE + 1))
    with pytest.raises(argparse.ArgumentTypeError, match="sequence length"):
        _parse_floats(payload)
