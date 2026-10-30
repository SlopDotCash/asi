"""Ceiling reruns must refuse existing artifacts before training starts."""

from pathlib import Path

import pytest

from alberta_framework.benchmarks import ipmnist_ceiling as ceiling

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["stationary", "carried", "full"])
def test_existing_ceiling_output_is_rejected_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    destination = tmp_path / f"{mode}_sigma0_ndecay099_seed0.npz"
    destination.write_bytes(b"preserved artifact bytes")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("training must not start for an existing output")

    monkeypatch.setattr(ceiling, "run_arm_per_step", forbidden)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        ceiling.main([mode, "--output-dir", str(tmp_path), "--seed", "0"])
    assert destination.read_bytes() == b"preserved artifact bytes"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("mode", ["stationary", "carried", "full"])
def test_different_seed_still_starts_ceiling_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    (tmp_path / f"{mode}_sigma0_ndecay099_seed0.npz").write_bytes(b"seed zero")

    def started(spec_name: str, seed: int, tasks: int, permutation: str) -> None:
        assert (spec_name, seed) == ("sigma0_ndecay099", 1)
        assert (tasks, permutation) == {
            "stationary": (1, "identity"), "carried": (60, "same"), "full": (200, "protocol")
        }[mode]
        raise RuntimeError("training started for new seed")

    monkeypatch.setattr(ceiling, "run_arm_per_step", started)
    with pytest.raises(RuntimeError, match="training started for new seed"):
        ceiling.main([mode, "--output-dir", str(tmp_path), "--seed", "1"])
