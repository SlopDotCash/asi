"""Strict import tests for paper-era Forager field-of-view databases."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.forager_results import (
    LEGACY_FOV_FRAMES,
    LegacyFOVSQLiteRunSpec,
    import_legacy_fov_sqlite,
)

pytestmark = pytest.mark.integration


def _write_artifacts(
    directory: Path,
    *,
    seeds: Sequence[int] = (0,),
    seed_offset: int = 0,
    reward_for: Callable[[int, int], float] | None = None,
) -> tuple[Path, Path]:
    config = {
        "agent": "Random",
        "problem": "ForagerTwoBiomeLarge",
        "total_steps": 500_000,
        "episode_cutoff": -1,
        "metaParameters": {
            "environment": {"aperture": 1},
            "experiment": {"seed_offset": seed_offset},
        },
    }
    config_path = directory / "Random.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    database_path = directory / "results.db"
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE metadata("version")')
    cursor.execute(
        """
        CREATE TABLE hyperparameters(
            "config_id",
            "environment.aperture",
            "experiment.seed_offset"
        )
        """
    )
    cursor.execute('CREATE TABLE results("seed", "config_id", "frame", "reward")')
    cursor.execute("INSERT INTO metadata(version) VALUES (?)", ("v2",))
    cursor.execute(
        """
        INSERT INTO hyperparameters(
            config_id,
            "environment.aperture",
            "experiment.seed_offset"
        ) VALUES (?, ?, ?)
        """,
        (8675309, 1, seed_offset),
    )
    reward_function = reward_for or (
        lambda seed, sample: float(seed) + float(sample) / len(LEGACY_FOV_FRAMES)
    )
    cursor.executemany(
        "INSERT INTO results(seed, config_id, frame, reward) VALUES (?, ?, ?, ?)",
        (
            (seed, 8675309, frame, reward_function(seed, sample))
            for seed in seeds
            for sample, frame in enumerate(LEGACY_FOV_FRAMES)
        ),
    )
    connection.commit()
    connection.close()
    return database_path, config_path


def _spec(
    database_path: Path,
    config_path: Path,
    *,
    stored_seed: int = 0,
    expected_stored_seeds: tuple[int, ...] = (0,),
    **changes: object,
) -> LegacyFOVSQLiteRunSpec:
    values: dict[str, object] = {
        "agent": "Random",
        "path": database_path,
        "config_path": config_path,
        "run_index": stored_seed,
        "stored_seed": stored_seed,
        "expected_config_agent": "Random",
        "expected_aperture_size": 1,
        "expected_stored_seeds": expected_stored_seeds,
    }
    values.update(changes)
    return LegacyFOVSQLiteRunSpec(**values)  # type: ignore[arg-type]


def test_imports_preprocessed_curve_without_double_smoothing(tmp_path: Path) -> None:
    database_path, config_path = _write_artifacts(
        tmp_path,
        seeds=(0, 1),
        seed_offset=1_000_000,
    )
    result = import_legacy_fov_sqlite(
        _spec(
            database_path,
            config_path,
            stored_seed=1,
            expected_stored_seeds=(0, 1),
            expected_database_sha256=hashlib.sha256(database_path.read_bytes()).hexdigest(),
            expected_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        )
    )

    stored_curve = 1.0 + np.arange(len(LEGACY_FOV_FRAMES)) / len(LEGACY_FOV_FRAMES)
    expected = float(np.mean(stored_curve[int(0.9 * stored_curve.size) :]))
    assert result.fov_last_10pct_ema_auc == pytest.approx(expected)
    assert result.curve_ewm_reward == pytest.approx(stored_curve)
    assert result.curve_steps == LEGACY_FOV_FRAMES
    assert result.curve_window_reward == ()
    assert math.isnan(result.mean_reward)
    assert math.isnan(result.mean_ewm_reward)
    assert result.seed == 1_000_001
    assert result.agent_metadata["run_index"] == 1
    assert result.agent_metadata["stored_seed"] == 1
    assert result.agent_metadata["effective_seed"] == 1_000_001
    assert result.agent_metadata["result_source"] == "official_fov_sqlite"
    assert result.agent_metadata["runtime"] == "historical_numpy_forager"
    assert result.agent_metadata["sqlite_integrity_validated"] is True
    assert result.agent_metadata["database_sha256"]
    assert result.agent_metadata["config_sha256"]
    assert result.agent_metadata["flattened_hyperparameters_sha256"]
    assert result.environment["runtime"] == "historical_numpy_forager"
    assert result.environment["pairable_with_current_foragax"] is False
    assert result.metric_contract["raw_reward_metrics_available"] is False
    assert result.metric_contract["fov_last_10pct_ema_auc"]["input_is_already_transformed"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cursor: cursor.execute("UPDATE metadata SET version = 'v1'"),
            "PyExpUtils v2 metadata",
        ),
        (
            lambda cursor: cursor.execute(
                """
                UPDATE results
                SET frame = 499800
                WHERE frame = 499900
                """
            ),
            "duplicate result row",
        ),
        (
            lambda cursor: cursor.execute(
                "UPDATE results SET reward = ? WHERE frame = 0",
                (math.inf,),
            ),
            "non-finite/non-numeric reward",
        ),
    ],
)
def test_rejects_malformed_database(
    tmp_path: Path,
    mutation: Callable[[sqlite3.Cursor], object],
    message: str,
) -> None:
    database_path, config_path = _write_artifacts(tmp_path)
    connection = sqlite3.connect(database_path)
    mutation(connection.cursor())
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match=message):
        import_legacy_fov_sqlite(_spec(database_path, config_path))


def test_rejects_config_database_and_seed_mismatches(tmp_path: Path) -> None:
    database_path, config_path = _write_artifacts(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["metaParameters"]["environment"]["aperture"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="aperture"):
        import_legacy_fov_sqlite(_spec(database_path, config_path))

    config["metaParameters"]["environment"]["aperture"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="result rows"):
        import_legacy_fov_sqlite(
            _spec(
                database_path,
                config_path,
                expected_stored_seeds=(0, 1),
            )
        )
    with pytest.raises(ValueError, match="SHA-256"):
        import_legacy_fov_sqlite(
            _spec(
                database_path,
                config_path,
                expected_database_sha256="0" * 64,
            )
        )


def test_rejects_dishonest_labels_and_current_foragax_pairing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be labelled 'Search Oracle'"):
        LegacyFOVSQLiteRunSpec(
            agent="DQN",
            path=tmp_path / "unused.db",
            config_path=tmp_path / "Greedy.json",
            run_index=0,
            stored_seed=0,
            expected_config_agent="Greedy",
            expected_aperture_size=15,
            privileged=True,
        )
    with pytest.raises(ValueError, match="privileged=True"):
        LegacyFOVSQLiteRunSpec(
            agent="Search Oracle",
            path=tmp_path / "unused.db",
            config_path=tmp_path / "Greedy.json",
            run_index=0,
            stored_seed=0,
            expected_config_agent="Greedy",
            expected_aperture_size=15,
        )

    database_path, config_path = _write_artifacts(tmp_path)
    legacy = import_legacy_fov_sqlite(_spec(database_path, config_path))
    with pytest.raises(ValueError, match="curve_steps"):
        dataclasses.replace(
            legacy,
            agent="alberta_horde_ac",
            environment={"runtime": "foragax"},
            agent_metadata={"seed": legacy.seed, "result_source": "in_tree"},
        )
