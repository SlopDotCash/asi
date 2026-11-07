"""Recompute retained development waves without rerunning historical training."""

from pathlib import Path

import pytest

from alberta_framework._strict_json import load_strict_json_object
from alberta_framework.benchmarks.ipmnist_screening import merge_shards

_OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" / "ipmnist_screening"


@pytest.mark.parametrize(
    "wave",
    (
        "wave_identmap50_nogate",
        "wave_identmap200_nogate",
        "wave_identmap50_nogate_idet003",
        "wave_identmap50_nogate_idet003_s0075",
        "wave_identmap50_nogate_idet003_s005625",
        "wave_identmap50_r_idet002",
        "wave_identmap50_r_idet003_s00421875",
        "wave_identmap50_r_idet004_s00421875",
        "wave_identmap50_r_idet003_s0031640625",
        "wave_identmap50_r_idet003_s002373046875",
        "wave_identmap50_r_idet003_s0031640625_r01",
        "wave_identmap50_r_idet003_s0031640625_r02",
        "wave_identmap50_r_idet003_s0031640625_r02_preset003",
        "gradual64_r2",
    ),
)
def test_historical_wave_replays_from_registered_shards(wave: str) -> None:
    directory = _OUTPUTS / wave
    retained = load_strict_json_object(directory / "summary.json")
    paths = [directory / Path(row["path"]).name for row in retained["shard_manifest"]]

    recomputed = merge_shards(
        paths,
        control_name=retained["control_name"],
        slope_window=retained["slope_window"],
    )

    # Historical invocations used different working directories. Compare
    # the bound shard bytes and identity independently of their path spelling.
    for summary in (retained, recomputed):
        del summary["created_unix"]
        for shard in summary["shard_manifest"]:
            shard["path"] = Path(shard["path"]).name
    assert recomputed == retained
