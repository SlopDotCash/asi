"""Fast contracts for the production Step 1 stream factory."""

import numpy as np
import pytest

from alberta_framework.steps import Step1KernelConfig, make_step1_stream
from alberta_framework.streams.alberta_plan_step1 import XDistShiftStream

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("noise_std", "noise_in_target"),
    [(0.0, False), (0.25, True), (7.0, True)],
)
def test_xdist_factory_forwards_noise_configuration(
    noise_std: float,
    noise_in_target: bool,
) -> None:
    config = Step1KernelConfig(
        feature_dim=4,
        num_relevant=2,
        stream="xdist_shift",
        noise_std=noise_std,
    )

    stream = make_step1_stream(config)

    assert isinstance(stream, XDistShiftStream)
    assert stream._noise_std == float(np.float32(noise_std))  # noqa: SLF001
    assert stream._noise_in_target is noise_in_target  # noqa: SLF001
