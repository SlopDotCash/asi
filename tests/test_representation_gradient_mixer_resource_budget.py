"""Leftover-identity gates for mixer resource-budget records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerResourceBudget,
)


def _legal_budget() -> RepresentationGradientMixerResourceBudget:
    return RepresentationGradientMixerResourceBudget(
        representation_dim=3,
        persistent_state_scalars=0,
        persistent_state_bytes=0,
        output_float32_scalars=15,
        output_bool_scalars=12,
        output_scalars=27,
        output_nbytes=72,
    )


def test_mixer_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="representation_dim"):
        RepresentationGradientMixerResourceBudget(
            representation_dim=True,
            persistent_state_scalars=0,
            persistent_state_bytes=0,
            output_float32_scalars=15,
            output_bool_scalars=12,
            output_scalars=27,
            output_nbytes=72,
        )
    with pytest.raises(ValueError, match="output_bool_scalars"):
        RepresentationGradientMixerResourceBudget(
            representation_dim=3,
            persistent_state_scalars=0,
            persistent_state_bytes=0,
            output_float32_scalars=15,
            output_bool_scalars=True,
            output_scalars=27,
            output_nbytes=72,
        )
    with pytest.raises(ValueError, match="persistent_state_bytes"):
        RepresentationGradientMixerResourceBudget(
            representation_dim=3,
            persistent_state_scalars=0,
            persistent_state_bytes=float("nan"),
            output_float32_scalars=15,
            output_bool_scalars=12,
            output_scalars=27,
            output_nbytes=72,
        )

    legal = _legal_budget()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"representation_dim": 3' in dumped
    assert '"persistent_state_scalars": 0' in dumped
    assert '"output_bool_scalars": 12' in dumped
    assert '"representation_dim": true' not in dumped
    assert '"output_bool_scalars": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped
