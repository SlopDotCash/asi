"""Hostile values are rejected before conversion, comparison, or formatting hooks."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.behavior_model import selected_action_probabilities
from alberta_framework.core.normalizers import normalizer_from_config
from alberta_framework.core.optimizers import IDBD
from alberta_framework.core.state_builder import state_builder_config_from_config
from alberta_framework.core.upgd import UPGDLearner
from alberta_framework.reference_agent import (
    AgentManifest,
    Decision,
    SpaceSpec,
)
from alberta_framework.utils.timing import Timer


class _EqualityTrapStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("string equality hook must not run")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("string formatting hook must not run")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("string repr hook must not run")


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (
            lambda value: normalizer_from_config(
                {"type": "EMANormalizer", "state_schema": value}
            ),
            "state_schema",
        ),
        (lambda value: IDBD(h_decay_mode=value), "h_decay_mode"),
        (
            lambda value: UPGDLearner.step2_default(2, loss_normalization=value),
            "loss_normalization",
        ),
        (lambda value: UPGDLearner.step2_default(2, readout_mode=value), "readout_mode"),
    ],
)
def test_identity_gates_precede_hostile_equality(factory: Any, field: str) -> None:
    value = _EqualityTrapStr("leftover")
    _EqualityTrapStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        factory(value)
    assert _EqualityTrapStr.calls == 0, field


def test_config_parsers_reject_dict_subclasses_before_mapping_hooks() -> None:
    class _HostileDict(dict[str, Any]):
        def get(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
            raise AssertionError("mapping hook must not run")

        def __iter__(self):  # type: ignore[no-untyped-def]  # pragma: no cover
            raise AssertionError("mapping hook must not run")

    with pytest.raises(ValueError, match="exact dict"):
        normalizer_from_config(_HostileDict(type="EMANormalizer"))
    with pytest.raises(ValueError, match="exact dict"):
        state_builder_config_from_config(_HostileDict(type="IdentityStateBuilder"))


def test_behavior_action_gate_avoids_instance_class_and_array_hooks() -> None:
    class _ClassSpoof:
        calls = 0

        @property
        def __class__(self) -> type:  # pragma: no cover
            type(self).calls += 1
            raise AssertionError("instance __class__ hook must not run")

        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:  # pragma: no cover
            type(self).calls += 1
            raise AssertionError("array hook must not run")

    value = _ClassSpoof()
    with pytest.raises(TypeError, match="trusted array"):
        selected_action_probabilities(jnp.asarray([0.25, 0.75]), value)
    assert _ClassSpoof.calls == 0


def test_reference_records_gate_exact_strings_before_identity_comparison() -> None:
    value = _EqualityTrapStr("leftover")
    _EqualityTrapStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        AgentManifest(
            schema=value,
            implementation_id="a.b",
            state_schema="c.d",
            config_sha256="0" * 64,
            manifest_id="1" * 64,
            observation_spec=None,  # type: ignore[arg-type]
            action_spec=None,  # type: ignore[arg-type]
            capabilities=None,  # type: ignore[arg-type]
            _config_json="{}",
        )
    with pytest.raises(ValueError, match="lowercase identifier"):
        Decision(
            manifest_id="0" * 64,
            lifecycle_id="life",
            decision_index=0,
            decision_id=value,
            observation_id="observation",
            observation=None,  # type: ignore[arg-type]
            proposed_action=None,
            armed=False,
        )
    assert _EqualityTrapStr.calls == 0


def test_reference_space_rejects_leftover_kind_and_shape_identities() -> None:
    with pytest.raises(ValueError, match="exact string"):
        SpaceSpec(
            kind=_EqualityTrapStr("box"),
            shape=(),
            dtype="float32",
            semantic_id="space",
        )
    with pytest.raises(ValueError, match="shape"):
        SpaceSpec(
            kind="box",
            shape=(np.int32(1),),  # type: ignore[arg-type]
            dtype="float32",
            semantic_id="space",
        )


def test_timer_repr_escapes_exact_but_untrusted_text() -> None:
    timer = Timer("quote'\nline", verbose=False)
    rendered = repr(timer)
    assert rendered == 'Timer(name="quote\'\\nline")'
    assert "\n" not in rendered
