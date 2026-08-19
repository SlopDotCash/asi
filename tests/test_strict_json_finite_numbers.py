"""Finite-number checks shared by external artifact and shard JSON boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework._strict_json import load_strict_json_object
from alberta_framework.benchmarks import (
    _forager_matched_alberta_worker as worker,
)
from alberta_framework.benchmarks import (
    _forager_matched_container as container,
)
from alberta_framework.benchmarks import (
    _official_foragax_image_helper as image_helper,
)
from alberta_framework.benchmarks import (
    foragax_open_screen,
    forager_matched_campaign,
    forager_matched_evidence,
    forager_matched_executor,
    forager_matched_protocol,
    forager_matched_qualification,
    forager_matched_seal,
    forager_matched_statistics,
    forager_matrix,
    forager_results,
    forager_rng_parity,
    historical_forager,
    ipmnist_campaign_tools,
    official_foragax,
    official_foragax_oci,
    upgd_ipmnist,
    upgd_label_emnist,
)
from alberta_framework.evaluation import evidence_manifest
from alberta_framework.evaluation.upgd_ipmnist_nonpromoting import (
    _decode_strict_json_object,
)

pytestmark = pytest.mark.unit

_OVERFLOW_VARIANTS = (
    '{"value":1e999}',
    '{"nested":{"value":1e999}}',
)

_PATH_LOADERS: tuple[
    tuple[str, Callable[[Path], object], type[BaseException]], ...
] = (
    ("shared", load_strict_json_object, ValueError),
    ("campaign_tools", ipmnist_campaign_tools._json_object, ValueError),
    ("evidence_manifest", evidence_manifest._load_strict_json_object, ValueError),
    ("upgd_ipmnist", upgd_ipmnist._strict_json_object, ValueError),
    ("upgd_label_emnist", upgd_label_emnist._strict_json_object, ValueError),
    ("official_image_helper", image_helper._strict_json, image_helper.ImageHelperError),
)


@pytest.mark.parametrize("raw", _OVERFLOW_VARIANTS, ids=("top_level", "nested"))
@pytest.mark.parametrize(
    "name,loader,error",
    _PATH_LOADERS,
    ids=[case[0] for case in _PATH_LOADERS],
)
def test_strict_path_loaders_reject_float_exponent_overflow(
    tmp_path: Path,
    name: str,
    loader: Callable[[Path], object],
    error: type[BaseException],
    raw: str,
) -> None:
    path = tmp_path / f"{name}.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(error, match="non-finite JSON number"):
        loader(path)


_RAW_LOADERS: tuple[
    tuple[str, Callable[[str], object], type[BaseException]], ...
] = (
    (
        "upgd_nonpromoting",
        lambda raw: _decode_strict_json_object(raw.encode()),
        ValueError,
    ),
    (
        "matched_protocol",
        forager_matched_protocol.decode_strict_json,
        forager_matched_protocol.ForagerMatchedProtocolError,
    ),
    (
        "matched_rng_parity",
        forager_rng_parity.decode_strict_json,
        forager_rng_parity.ForagerRngParityError,
    ),
    (
        "matched_evidence",
        forager_matched_evidence.decode_strict_json,
        forager_matched_evidence.ForagerMatchedEvidenceError,
    ),
    (
        "matched_executor",
        forager_matched_executor.decode_strict_json,
        forager_matched_executor.ForagerMatchedExecutorError,
    ),
    (
        "matched_campaign",
        lambda raw: forager_matched_campaign._decode_canonical(raw.encode(), "test"),
        forager_matched_campaign.ForagerMatchedCampaignError,
    ),
    (
        "matched_seal",
        lambda raw: forager_matched_seal._decode_canonical(raw.encode(), "test"),
        forager_matched_seal.ForagerMatchedSealError,
    ),
    (
        "matched_qualification",
        lambda raw: forager_matched_qualification._decode_json(raw.encode(), "test"),
        forager_matched_qualification.ForagerMatchedQualificationError,
    ),
    (
        "forager_matrix",
        lambda raw: forager_matrix._decode_strict_json(raw, description="test"),
        forager_matrix.ForagerMatrixManifestError,
    ),
    (
        "matched_container",
        lambda raw: container._strict_json(raw.encode()),
        container.ContainerError,
    ),
    (
        "foragax_open_screen",
        lambda raw: foragax_open_screen._load_json_bytes(raw.encode(), "test"),
        foragax_open_screen.ScreenError,
    ),
    (
        "official_foragax",
        lambda raw: official_foragax._strict_json_loads(raw, label="test"),
        official_foragax.OfficialForagaxValidationError,
    ),
    (
        "official_foragax_oci",
        lambda raw: official_foragax_oci._strict_json_bytes(raw.encode(), label="test"),
        official_foragax_oci.OfficialForagaxOciError,
    ),
    (
        "forager_results",
        lambda raw: forager_results._json_without_duplicate_keys(
            raw.encode(), path=Path("test.json")
        ),
        ValueError,
    ),
    (
        "matched_statistics",
        lambda raw: forager_matched_statistics.load_canonical_result(
            raw.encode(), cast(Any, None)
        ),
        forager_matched_statistics.MatchedStatisticsError,
    ),
)


@pytest.mark.parametrize("raw", _OVERFLOW_VARIANTS, ids=("top_level", "nested"))
@pytest.mark.parametrize("name,loader,error", _RAW_LOADERS, ids=[case[0] for case in _RAW_LOADERS])
def test_strict_byte_loaders_reject_float_exponent_overflow(
    name: str,
    loader: Callable[[str], object],
    error: type[BaseException],
    raw: str,
) -> None:
    del name
    with pytest.raises(error, match="non.?finite.*(?:number|constant)"):
        loader(raw)


@pytest.mark.parametrize("raw", _OVERFLOW_VARIANTS, ids=("top_level", "nested"))
def test_historical_artifact_loader_rejects_float_exponent_overflow(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "result.json"
    path.write_text(raw + "\n", encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(
        historical_forager.HistoricalForagerArtifactError,
        match="non-finite JSON number",
    ):
        historical_forager._strict_json_object(path)


@pytest.mark.parametrize(
    "raw",
    (
        '{"schema_version":1e999,"implementation_kind":"x","configuration":{}}',
        '{"schema_version":"x","implementation_kind":"x",'
        '"configuration":{"value":1e999}}',
    ),
    ids=("top_level", "nested"),
)
def test_matched_worker_loader_rejects_float_exponent_overflow(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "configuration.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(worker.MatchedAlbertaWorkerError, match="non-finite configuration"):
        worker._load_configuration(path)


def test_shared_strict_loader_preserves_finite_floats_and_arbitrary_integers(
    tmp_path: Path,
) -> None:
    integer = 10**400
    path = tmp_path / "finite.json"
    path.write_text(
        f'{{"large":1e200,"small":-1e-200,"integer":{integer}}}',
        encoding="utf-8",
    )

    assert load_strict_json_object(path) == {
        "large": 1e200,
        "small": -1e-200,
        "integer": integer,
    }
