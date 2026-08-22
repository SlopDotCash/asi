"""Every numpy integer family reaching the two benchmark exact-type index gates.

``ipmnist_gradual._NUMPY_INTEGER_TYPES`` and
``causal_map_forager._EXACT_NUMPY_INTEGER_TYPES`` are the allowlists behind
``GradualTransitionConfig``, ``transition_alpha``, ``output_interpolation``,
``task_sampling_mask``, ``CausalMapForagerConfig`` and
``CausalMapForagerAgent``.  Both enumerated fixed-width names, so wherever
numpy's C aliases are distinct type objects -- 64-bit Windows, where
``np.intc is np.int32`` is False -- a valid C ``int`` was rejected by these
gates while every other spelling of the same width passed.  Neither constant is
spelled ``_ACTUAL_INT_TYPES``, so no sweep over that name reaches them.

The parametrization is over dtype codes rather than fixed-width names so the
contract is stated on every platform, not only the one that exposed the gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.causal_map_forager import (
    CausalMapForagerAgent,
    CausalMapForagerConfig,
)
from alberta_framework.benchmarks.ipmnist_gradual import (
    GradualTransitionConfig,
    output_interpolation,
    task_sampling_mask,
    transition_alpha,
)

pytestmark = pytest.mark.unit

_INTEGER_DTYPE_CODES = ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")


def test_dtype_codes_enumerate_every_distinct_integer_family() -> None:
    """The parametrization must be an enumeration, not a hand-kept list of names."""
    families = tuple(np.dtype(code).type for code in _INTEGER_DTYPE_CODES)
    assert len(set(families)) == len(families)
    assert set(families) == {np.dtype(code).type for code in "bBhHiIlLqQpP"}


@pytest.mark.parametrize("code", _INTEGER_DTYPE_CODES)
def test_gradual_index_arguments_accept_every_numpy_integer_family(code: str) -> None:
    integer = np.dtype(code).type

    config = GradualTransitionConfig(mode="task_sampling", transition_steps=integer(4))
    assert type(config.transition_steps) is int
    assert config.transition_steps == 4
    assert transition_alpha(integer(2), config) == 0.5

    np.testing.assert_array_equal(
        output_interpolation(integer(0), integer(1), 0.25, n_classes=integer(3)),
        output_interpolation(0, 1, 0.25, n_classes=3),
    )
    np.testing.assert_array_equal(
        task_sampling_mask(seed=0, transition_id=integer(0), count=integer(4), alpha=0.5),
        task_sampling_mask(seed=0, transition_id=0, count=4, alpha=0.5),
    )


@pytest.mark.parametrize("code", _INTEGER_DTYPE_CODES)
def test_causal_map_scalars_accept_every_numpy_integer_family(code: str) -> None:
    integer = np.dtype(code).type

    config = CausalMapForagerConfig(
        initial_retry_delay=integer(1),
        maximum_retry_exponent=integer(0),
        maximum_exact_interval_width=integer(0),
        visit_penalty=integer(0),
    )
    assert type(config.initial_retry_delay) is int
    assert type(config.maximum_retry_exponent) is int
    assert type(config.maximum_exact_interval_width) is int
    assert type(config.visit_penalty) is float

    agent = CausalMapForagerAgent(seed=integer(3))
    assert type(agent.seed) is int
    assert agent.seed == 3


def test_widened_families_do_not_soften_the_exact_type_gates() -> None:
    """Admitting every family must not admit a subclass, a bool, or a float."""

    class ForgedInt(np.int64):
        def __index__(self) -> int:
            raise AssertionError("must not execute")

    with pytest.raises(ValueError, match="integer"):
        GradualTransitionConfig(
            mode="task_sampling",
            transition_steps=ForgedInt(4),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="integer"):
        GradualTransitionConfig(mode="task_sampling", transition_steps=True)
    with pytest.raises(ValueError, match="initial_retry_delay"):
        CausalMapForagerConfig(initial_retry_delay=ForgedInt(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial_retry_delay"):
        CausalMapForagerConfig(initial_retry_delay=np.float64(1.0))  # type: ignore[arg-type]
