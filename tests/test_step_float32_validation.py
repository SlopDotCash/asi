"""Exact binary32 canonicalization shared by the merged Step facades."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from fractions import Fraction
from numbers import Real
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.options import SubtaskSpec
from alberta_framework.steps.step3 import Step3HordeConfig, make_step3_horde_spec
from alberta_framework.steps.step4 import Step4SARSAConfig, make_step4_sarsa_agent
from alberta_framework.steps.step5 import Step5AverageRewardTDConfig
from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import Step7DynaConfig
from alberta_framework.steps.step8 import Step8WorldModelConfig, make_step8_world_model
from alberta_framework.steps.step9 import Step9DreamingConfig, make_step9_components
from alberta_framework.steps.step10 import Step10STOMPConfig

_FacadeValues = Callable[[object], tuple[float, ...]]


def _step3_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step3HordeConfig(
        gammas=(scalar,),
        lamdas=(scalar,),
        step_size=scalar,
        obgd_kappa=scalar,
        sparsity=scalar,
    )
    return (
        config.gammas[0],
        config.lamdas[0],
        config.step_size,
        config.obgd_kappa,
        config.sparsity,
    )


def _step4_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step4SARSAConfig(
        gamma=scalar,
        epsilon_start=scalar,
        epsilon_end=scalar,
        lamda=scalar,
        step_size=scalar,
        meta_step_size=scalar,
        bounder_kappa=scalar,
        sparsity=scalar,
    )
    return (
        config.gamma,
        config.epsilon_start,
        config.epsilon_end,
        config.lamda,
        config.step_size,
        config.meta_step_size,
        config.bounder_kappa,
        config.sparsity,
    )


def _step5_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step5AverageRewardTDConfig(
        step_size=scalar,
        average_reward_step_size=scalar,
        trace_decay=scalar,
    )
    return (
        config.step_size,
        config.average_reward_step_size,
        config.trace_decay,
    )


def _step6_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step6DifferentialSARSAConfig(
        q_step_size=scalar,
        average_reward_step_size=scalar,
        trace_decay=scalar,
        epsilon_start=scalar,
        epsilon_end=scalar,
    )
    return (
        config.q_step_size,
        config.average_reward_step_size,
        config.trace_decay,
        config.epsilon_start,
        config.epsilon_end,
    )


def _step7_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step7DynaConfig(
        planning_importance_ratio_clip=scalar,
        planning_priority_propagation=scalar,
        planning_utility_step_size=scalar,
    )
    return (
        config.planning_importance_ratio_clip,
        config.planning_priority_propagation,
        config.planning_utility_step_size,
    )


def _step8_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step8WorldModelConfig(
        step_size=scalar,
        sparsity=scalar,
        leaky_relu_slope=scalar,
        utility_decay=scalar,
    )
    return (
        config.step_size,
        config.sparsity,
        config.leaky_relu_slope,
        config.utility_decay,
    )


def _step9_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step9DreamingConfig(
        model_step_size=scalar,
        model_sparsity=scalar,
        model_gamma=scalar,
        dreaming_max_model_error=scalar,
        model_error_decay=scalar,
        behavior_model_step_size=scalar,
        dream_surprise_weight=scalar,
        dream_utility_weight=scalar,
    )
    return (
        config.model_step_size,
        config.model_sparsity,
        config.model_gamma,
        config.dreaming_max_model_error,
        config.model_error_decay,
        config.behavior_model_step_size,
        config.dream_surprise_weight,
        config.dream_utility_weight,
    )


def _step10_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step10STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=scalar,
                pseudo_reward_scale=scalar,
            ),
        ),
        base_step_size=scalar,
        base_avg_reward_step_size=scalar,
        base_trace_decay=scalar,
        option_step_size=scalar,
        option_avg_reward_step_size=scalar,
        option_trace_decay=scalar,
        option_gamma=scalar,
        option_model_decay=scalar,
        option_model_step_size=scalar,
        epsilon_base=scalar,
        epsilon_option=scalar,
        option_target_epsilon=scalar,
        option_importance_clip=scalar,
    )
    spec = config.subtask_specs[0]
    return (
        spec.threshold,
        spec.pseudo_reward_scale,
        config.base_step_size,
        config.base_avg_reward_step_size,
        config.base_trace_decay,
        config.option_step_size,
        config.option_avg_reward_step_size,
        config.option_trace_decay,
        config.option_gamma,
        config.option_model_decay,
        config.option_model_step_size,
        config.epsilon_base,
        config.epsilon_option,
        cast(float, config.option_target_epsilon),
        config.option_importance_clip,
    )


@pytest.mark.parametrize(
    "facade_values",
    [
        pytest.param(_step3_values, id="step3"),
        pytest.param(_step4_values, id="step4"),
        pytest.param(_step5_values, id="step5"),
        pytest.param(_step6_values, id="step6"),
        pytest.param(_step7_values, id="step7"),
        pytest.param(_step8_values, id="step8"),
        pytest.param(_step9_values, id="step9"),
        pytest.param(_step10_values, id="step10"),
    ],
)
def test_all_float32_bound_fields_round_exact_fraction_once(
    facade_values: _FacadeValues,
) -> None:
    above_half_midpoint = (
        Fraction(1, 2) + Fraction(1, 2**25) + Fraction(1, 2**70)
    )
    expected = float(np.nextafter(np.float32(0.5), np.float32(1.0)))

    values = facade_values(above_half_midpoint)

    assert values
    assert all(value == expected for value in values)
    json.dumps(values, allow_nan=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            Fraction(1, 1) + Fraction(1, 2**24) - Fraction(1, 2**60),
            1.0,
        ),
        (Fraction(1, 1) + Fraction(1, 2**24), 1.0),
        (
            Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60),
            float(np.nextafter(np.float32(1.0), np.float32(2.0))),
        ),
    ],
    ids=("below", "tie-even", "above"),
)
def test_fraction_midpoint_neighborhood_rounds_once(
    value: Fraction,
    expected: float,
) -> None:
    assert Step5AverageRewardTDConfig(step_size=value).step_size == expected


def test_fraction_odd_lower_significand_tie_rounds_up() -> None:
    odd_lower_tie = Fraction(1, 1) + Fraction(3, 2**24)
    expected = float(np.nextafter(np.float32(1.0), np.float32(2.0), dtype=np.float32))
    expected = float(np.nextafter(np.float32(expected), np.float32(2.0)))

    assert Step8WorldModelConfig(step_size=odd_lower_tie).step_size == expected


def test_subnormal_tie_and_above_round_with_even_zero() -> None:
    half_min_subnormal = Fraction(1, 2**150)
    just_above = half_min_subnormal + Fraction(1, 2**200)
    min_subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0)))

    assert Step8WorldModelConfig(step_size=half_min_subnormal).step_size == 0.0
    assert Step8WorldModelConfig(step_size=just_above).step_size == min_subnormal


def test_positive_field_rejects_value_that_rounds_to_zero() -> None:
    with pytest.raises(ValueError, match="option_importance_clip must be positive"):
        Step10STOMPConfig(option_importance_clip=Fraction(1, 2**150))


def test_nonnegative_field_rejects_negative_value_that_underflows() -> None:
    with pytest.raises(ValueError, match="model_step_size must be non-negative"):
        Step9DreamingConfig(model_step_size=Fraction(-1, 2**1200))


def test_unit_interval_rejects_exact_value_above_endpoint() -> None:
    with pytest.raises(ValueError, match=r"gamma must be in \[0, 1\]"):
        Step4SARSAConfig(gamma=Fraction(1, 1) + Fraction(1, 2**200))


@pytest.mark.parametrize(
    "value",
    [Fraction(1, 2**150), Fraction(1, 2**149)],
    ids=("underflows-to-zero", "minimum-subnormal"),
)
def test_gvf_probabilities_reject_nonzero_subnormal_inputs(value: Fraction) -> None:
    with pytest.raises(ValueError, match="zero or a normal float32"):
        Step3HordeConfig(gammas=(value,), lamdas=(0.0,))
    with pytest.raises(ValueError, match="zero or a normal float32"):
        Step4SARSAConfig(gamma=value)
    with pytest.raises(ValueError, match="zero or a normal float32"):
        Step4SARSAConfig(lamda=value)


def test_gvf_probabilities_accept_minimum_normal_and_reach_core() -> None:
    minimum_normal = Fraction(1, 2**126)
    expected = float.fromhex("0x1.0p-126")
    step3 = Step3HordeConfig(gammas=(minimum_normal,), lamdas=(minimum_normal,))
    step4 = Step4SARSAConfig(gamma=minimum_normal, lamda=minimum_normal)

    assert step3.gammas == (expected,)
    assert step3.lamdas == (expected,)
    assert step4.gamma == expected
    assert step4.lamda == expected
    make_step3_horde_spec(step3)
    make_step4_sarsa_agent(step4)


@pytest.mark.parametrize(
    "build_config",
    [
        pytest.param(
            lambda value: Step3HordeConfig(gammas=(value,), lamdas=(0.0,)),
            id="step3-gamma",
        ),
        pytest.param(
            lambda value: Step3HordeConfig(gammas=(0.0,), lamdas=(value,)),
            id="step3-lamda",
        ),
        pytest.param(
            lambda value: Step4SARSAConfig(gamma=value),
            id="step4-gamma",
        ),
        pytest.param(
            lambda value: Step4SARSAConfig(lamda=value),
            id="step4-lamda",
        ),
    ],
)
def test_gvf_probabilities_reject_exact_subnormal_that_rounds_to_normal(
    build_config: Callable[[Any], object],
) -> None:
    value = Fraction(1, 2**126) - Fraction(1, 2**200)
    assert float(np.float32(value)) == float.fromhex("0x1.0p-126")

    with pytest.raises(ValueError, match="zero or a normal float32"):
        build_config(value)


def test_zero_host_with_subnormal_ratio_is_rejected_before_core() -> None:
    class SubnormalRatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2**149)

    value = SubnormalRatioFloat(0.0)

    with pytest.raises(ValueError, match="zero or a normal float32"):
        Step3HordeConfig(gammas=(value,), lamdas=(0.0,))
    with pytest.raises(ValueError, match="zero or a normal float32"):
        Step4SARSAConfig(gamma=value)


def test_host_comparisons_cannot_hide_invalid_narrowed_domains() -> None:
    class NegativeRatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 1)

    class AboveUnitRatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (2, 1)

    negative = NegativeRatioFloat(0.5)
    above_unit = AboveUnitRatioFloat(0.5)

    with pytest.raises(ValueError, match="step_size must be non-negative"):
        Step4SARSAConfig(step_size=negative)
    with pytest.raises(ValueError, match="step_size must be non-negative"):
        Step5AverageRewardTDConfig(step_size=negative)
    with pytest.raises(ValueError, match="option_importance_clip must be positive"):
        Step10STOMPConfig(option_importance_clip=negative)
    with pytest.raises(ValueError, match=r"gammas must be in \[0, 1\]"):
        Step3HordeConfig(gammas=(above_unit,), lamdas=(0.0,))
    with pytest.raises(ValueError, match=r"trace_decay must be in \[0, 1\]"):
        Step5AverageRewardTDConfig(trace_decay=above_unit)


@pytest.mark.parametrize(
    "build_config",
    [
        pytest.param(lambda value: Step3HordeConfig(sparsity=value), id="step3"),
        pytest.param(lambda value: Step4SARSAConfig(epsilon_start=value), id="step4"),
        pytest.param(
            lambda value: Step5AverageRewardTDConfig(trace_decay=value),
            id="step5",
        ),
        pytest.param(
            lambda value: Step6DifferentialSARSAConfig(epsilon_start=value),
            id="step6",
        ),
        pytest.param(
            lambda value: Step7DynaConfig(planning_utility_step_size=value),
            id="step7",
        ),
        pytest.param(lambda value: Step8WorldModelConfig(sparsity=value), id="step8"),
        pytest.param(lambda value: Step9DreamingConfig(model_gamma=value), id="step9"),
        pytest.param(lambda value: Step10STOMPConfig(epsilon_base=value), id="step10"),
    ],
)
@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param((-1, 2**200), id="negative-rounds-to-negative-zero"),
        pytest.param((2**200 + 1, 2**200), id="above-one-rounds-to-one"),
    ],
)
def test_exact_ratio_cannot_hide_outside_closed_unit_domain(
    build_config: Callable[[Any], object],
    ratio: tuple[int, int],
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        build_config(HiddenBoundaryFloat(0.5))


@pytest.mark.parametrize(
    "build_config",
    [
        pytest.param(lambda value: Step3HordeConfig(step_size=value), id="step3"),
        pytest.param(lambda value: Step4SARSAConfig(step_size=value), id="step4"),
        pytest.param(
            lambda value: Step5AverageRewardTDConfig(step_size=value),
            id="step5",
        ),
        pytest.param(
            lambda value: Step6DifferentialSARSAConfig(q_step_size=value),
            id="step6",
        ),
        pytest.param(
            lambda value: Step7DynaConfig(planning_priority_propagation=value),
            id="step7",
        ),
        pytest.param(lambda value: Step8WorldModelConfig(step_size=value), id="step8"),
        pytest.param(
            lambda value: Step9DreamingConfig(dreaming_max_model_error=value),
            id="step9",
        ),
        pytest.param(lambda value: Step10STOMPConfig(base_step_size=value), id="step10"),
    ],
)
def test_exact_ratio_cannot_hide_negative_underflow_from_nonnegative_domain(
    build_config: Callable[[Any], object],
) -> None:
    class HiddenNegativeFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match="must be non-negative"):
        build_config(HiddenNegativeFloat(0.5))


@pytest.mark.parametrize(
    "build_config",
    [
        pytest.param(
            lambda value: Step3HordeConfig(gammas=(value,), lamdas=(0.0,)),
            id="step3-gamma",
        ),
        pytest.param(
            lambda value: Step3HordeConfig(gammas=(0.0,), lamdas=(value,)),
            id="step3-lamda",
        ),
        pytest.param(lambda value: Step4SARSAConfig(gamma=value), id="step4-gamma"),
        pytest.param(lambda value: Step4SARSAConfig(lamda=value), id="step4-lamda"),
    ],
)
def test_exact_ratio_cannot_hide_nonzero_gvf_underflow(
    build_config: Callable[[Any], object],
) -> None:
    class HiddenTinyFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2**150)

    with pytest.raises(ValueError, match="zero or a normal float32"):
        build_config(HiddenTinyFloat(0.5))


@pytest.mark.parametrize(
    ("config_type", "field"),
    [
        pytest.param(Step8WorldModelConfig, "utility_decay", id="step8"),
        pytest.param(Step9DreamingConfig, "model_error_decay", id="step9"),
    ],
)
def test_exact_ratio_cannot_hide_negative_underflow_from_half_open_domain(
    config_type: type[Step8WorldModelConfig] | type[Step9DreamingConfig],
    field: str,
) -> None:
    class HiddenNegativeFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match=rf"{field} must be in \[0, 1\)"):
        config_type(**{field: cast(Any, HiddenNegativeFloat(0.5))})


def test_exact_ratio_is_read_once_during_validation() -> None:
    class StatefulRatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            if type(self).calls == 1:
                return (1, 2)
            return (2, 1)

    value = StatefulRatioFloat(0.5)
    config = Step8WorldModelConfig(step_size=value)

    assert StatefulRatioFloat.calls == 1
    assert config.step_size == 0.5


@pytest.mark.parametrize(
    "build_config",
    [
        pytest.param(lambda value: Step3HordeConfig(step_size=value), id="step3"),
        pytest.param(lambda value: Step4SARSAConfig(step_size=value), id="step4"),
        pytest.param(
            lambda value: Step5AverageRewardTDConfig(step_size=value),
            id="step5",
        ),
        pytest.param(
            lambda value: Step6DifferentialSARSAConfig(q_step_size=value),
            id="step6",
        ),
        pytest.param(
            lambda value: Step7DynaConfig(planning_priority_propagation=value),
            id="step7",
        ),
        pytest.param(lambda value: Step8WorldModelConfig(step_size=value), id="step8"),
        pytest.param(
            lambda value: Step9DreamingConfig(dreaming_max_model_error=value),
            id="step9",
        ),
        pytest.param(lambda value: Step10STOMPConfig(base_step_size=value), id="step10"),
    ],
)
def test_class_property_cannot_spoof_actual_real_type(
    build_config: Callable[[Any], object],
) -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

        def __lt__(self, other: object) -> bool:
            return 0.5 < cast(Any, other)

        def __le__(self, other: object) -> bool:
            return 0.5 <= cast(Any, other)

        def __gt__(self, other: object) -> bool:
            return 0.5 > cast(Any, other)

        def __ge__(self, other: object) -> bool:
            return 0.5 >= cast(Any, other)

    value = ClassSpoof()
    assert isinstance(value, Real)
    assert not issubclass(type(value), Real)

    with pytest.raises(ValueError, match="must be a real number"):
        build_config(value)


@pytest.mark.parametrize(
    ("config_type", "field"),
    [
        pytest.param(Step8WorldModelConfig, "utility_decay", id="step8"),
        pytest.param(Step9DreamingConfig, "model_error_decay", id="step9"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(Fraction(1, 1) - Fraction(1, 2**25), id="exact-tie"),
        pytest.param(Fraction(1, 1) - Fraction(1, 2**200), id="exact-rational"),
        pytest.param(math.nextafter(1.0, 0.0), id="builtin-float"),
    ],
)
def test_half_open_interval_rejects_value_that_rounds_to_one(
    config_type: type[Step8WorldModelConfig] | type[Step9DreamingConfig],
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be in \[0, 1\)"):
        config_type(**{field: cast(Any, value)})


def test_half_open_interval_accepts_value_that_rounds_below_one() -> None:
    value = Fraction(1, 1) - Fraction(1, 2**25) - Fraction(1, 2**200)
    expected = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
    step8 = Step8WorldModelConfig(utility_decay=value)
    step9 = Step9DreamingConfig(model_error_decay=value)

    assert step8.utility_decay == expected
    assert step9.model_error_decay == expected
    make_step8_world_model(step8)
    make_step9_components(step9)


@pytest.mark.parametrize(
    ("offset", "accepted"),
    [
        (-Fraction(1, 2**80), True),
        (Fraction(0, 1), False),
        (Fraction(1, 2**80), False),
    ],
    ids=("below", "tie", "above"),
)
def test_half_open_decay_rejects_values_that_round_to_one(
    offset: Fraction,
    accepted: bool,
) -> None:
    value = Fraction(1, 1) - Fraction(1, 2**25) + offset

    if not accepted:
        with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
            Step8WorldModelConfig(utility_decay=value)
        with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
            Step9DreamingConfig(model_error_decay=value)
        return

    step8 = Step8WorldModelConfig(utility_decay=value)
    step9 = Step9DreamingConfig(model_error_decay=value)
    expected = float(np.nextafter(np.float32(1.0), np.float32(0.0)))

    assert step8.utility_decay == expected
    assert step9.model_error_decay == expected
    make_step8_world_model(step8)
    make_step9_components(step9)


def test_finite_overflow_boundary_accepts_below_and_rejects_tie() -> None:
    overflow_midpoint = (2**25 - 1) * 2**103
    maximum = float(np.finfo(np.float32).max)

    config = Step7DynaConfig(
        planning_importance_ratio_clip=overflow_midpoint - 1,
    )
    assert config.planning_importance_ratio_clip == maximum
    with pytest.raises(ValueError, match="must be finite"):
        Step7DynaConfig(planning_importance_ratio_clip=overflow_midpoint)


def test_large_integral_midpoint_uses_binary32_ties_to_even() -> None:
    lower = 2**64
    tie = lower + 2**40
    above = tie + 1
    expected_upper = float(np.nextafter(np.float32(lower), np.float32(np.inf)))

    assert Step9DreamingConfig(dreaming_max_model_error=tie).dreaming_max_model_error == float(
        lower
    )
    assert (
        Step9DreamingConfig(dreaming_max_model_error=above).dreaming_max_model_error
        == expected_upper
    )


@pytest.mark.parametrize(
    "field",
    ["step_size", "average_reward_step_size"],
)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            2**64 + 2**40 + 1,
            float.fromhex("0x1.000002p+64"),
            id="above-large-midpoint",
        ),
        pytest.param(
            (2**25 - 1) * 2**103 - 1,
            float.fromhex("0x1.fffffep+127"),
            id="below-overflow-midpoint",
        ),
    ],
)
def test_step5_builtin_int_storage_matches_validated_float32_sink(
    field: str,
    value: int,
    expected: float,
) -> None:
    config = Step5AverageRewardTDConfig(**{field: cast(Any, value)})
    stored = cast(float, getattr(config, field))

    assert float(jnp.asarray(stored, dtype=jnp.float32)) == expected
    assert type(stored) is float
    assert stored == expected


def test_step5_preserves_only_sink_exact_builtin_int_payloads() -> None:
    exact = 2**64
    config = Step5AverageRewardTDConfig(
        step_size=cast(Any, exact),
        average_reward_step_size=cast(Any, 0),
        trace_decay=cast(Any, 1),
    )
    payload = config.to_dict()

    assert type(config.step_size) is int
    assert config.step_size == exact
    assert type(config.average_reward_step_size) is int
    assert type(config.trace_decay) is int
    assert float(jnp.asarray(config.step_size, dtype=jnp.float32)) == float(exact)
    json.dumps(payload, allow_nan=False)


def test_step5_builtin_signed_zero_storage_is_preserved() -> None:
    config = Step5AverageRewardTDConfig(
        step_size=-0.0,
        average_reward_step_size=-0.0,
        trace_decay=-0.0,
    )
    payload = config.to_dict()

    for field in ("step_size", "average_reward_step_size", "trace_decay"):
        assert math.copysign(1.0, cast(float, getattr(config, field))) == -1.0
        assert math.copysign(1.0, cast(float, payload[field])) == -1.0


def test_numpy_integral_midpoint_uses_exact_integer_ratio() -> None:
    lower = 2**62
    tie = np.int64(lower + 2**38)
    above = np.int64(int(tie) + 1)
    expected_upper = float(np.nextafter(np.float32(lower), np.float32(np.inf)))

    assert Step9DreamingConfig(dreaming_max_model_error=tie).dreaming_max_model_error == float(
        lower
    )
    assert (
        Step9DreamingConfig(dreaming_max_model_error=above).dreaming_max_model_error
        == expected_upper
    )


def test_numpy_float64_midpoint_neighborhood_rounds_once() -> None:
    tie = np.float64(float.fromhex("0x1.000001p0"))
    above = np.nextafter(tie, np.float64(np.inf))
    expected_upper = float(np.nextafter(np.float32(1.0), np.float32(2.0)))

    tie_config = Step8WorldModelConfig(step_size=tie)
    above_config = Step8WorldModelConfig(step_size=above)

    assert type(tie_config.step_size) is float
    assert tie_config.step_size == 1.0
    assert above_config.step_size == expected_upper


def test_float_subclass_cannot_disagree_with_its_exact_ratio() -> None:
    class RatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (3, 4)

    value = RatioFloat(0.5)

    assert Step3HordeConfig(step_size=value).step_size == 0.75
    assert Step8WorldModelConfig(step_size=value).step_size == 0.75


def test_float_subclass_cannot_spoof_builtin_identity() -> None:
    class RatioFloat(float):
        def __getattribute__(self, name: str) -> object:
            if name == "__class__":
                return float
            return super().__getattribute__(name)

        def as_integer_ratio(self) -> tuple[int, int]:
            return (3, 4)

    value = RatioFloat(0.5)

    assert Step3HordeConfig(step_size=value).step_size == 0.75


def test_nonfinite_float_subclass_cannot_bypass_exact_ratio_storage() -> None:
    class RatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 1)

    config = Step9DreamingConfig(dream_surprise_weight=RatioFloat(float("nan")))

    assert config.dream_surprise_weight == 1.0
    json.dumps(config.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "value",
    [np.float16(0.1), np.float32(0.1), np.float64(0.1), np.int32(1), np.int64(1)],
)
def test_numpy_real_and_integral_scalars_canonicalize_for_json(value: object) -> None:
    config = Step3HordeConfig(step_size=cast(Any, value))

    assert type(config.step_size) is float
    json.dumps(config.to_dict(), allow_nan=False)


@pytest.mark.parametrize("value", [jnp.float32(0.5), jnp.int32(1)])
def test_jax_array_scalars_remain_outside_the_real_facade(value: object) -> None:
    with pytest.raises(ValueError, match="step_size must be a real number"):
        Step3HordeConfig(step_size=cast(Any, value))


@pytest.mark.parametrize(
    ("config_and_payload", "field"),
    [
        pytest.param(
            lambda: (
                Step3HordeConfig(step_size=0.1),
                Step3HordeConfig(step_size=0.1).to_dict(),
            ),
            "step_size",
            id="step3",
        ),
        pytest.param(
            lambda: (
                Step4SARSAConfig(step_size=0.1),
                Step4SARSAConfig(step_size=0.1).to_dict(),
            ),
            "step_size",
            id="step4",
        ),
        pytest.param(
            lambda: (
                Step5AverageRewardTDConfig(step_size=0.1),
                Step5AverageRewardTDConfig(step_size=0.1).to_dict(),
            ),
            "step_size",
            id="step5",
        ),
        pytest.param(
            lambda: (
                Step6DifferentialSARSAConfig(q_step_size=0.1),
                Step6DifferentialSARSAConfig(q_step_size=0.1).to_dict(),
            ),
            "q_step_size",
            id="step6",
        ),
        pytest.param(
            lambda: (
                Step7DynaConfig(planning_priority_propagation=0.1),
                Step7DynaConfig(planning_priority_propagation=0.1).to_dict(),
            ),
            "planning_priority_propagation",
            id="step7",
        ),
        pytest.param(
            lambda: (
                Step8WorldModelConfig(step_size=0.1),
                Step8WorldModelConfig(step_size=0.1).to_dict(),
            ),
            "step_size",
            id="step8",
        ),
        pytest.param(
            lambda: (
                Step9DreamingConfig(model_step_size=0.1),
                Step9DreamingConfig(model_step_size=0.1).to_dict(),
            ),
            "model_step_size",
            id="step9",
        ),
        pytest.param(
            lambda: (
                Step10STOMPConfig(base_step_size=0.1),
                Step10STOMPConfig(base_step_size=0.1).to_config(),
            ),
            "base_step_size",
            id="step10",
        ),
    ],
)
def test_builtin_float_serialization_value_is_preserved(
    config_and_payload: Callable[[], tuple[object, dict[str, Any]]],
    field: str,
) -> None:
    _, payload = config_and_payload()

    selected = payload[field]
    assert type(selected) is float
    assert selected == 0.1
    json.dumps(payload, allow_nan=False)


def test_signed_zero_survives_validation_and_serialization() -> None:
    config = Step3HordeConfig(gammas=(-0.0,), lamdas=(0.0,))
    payload = config.to_dict()

    assert math.copysign(1.0, config.gammas[0]) == -1.0
    assert math.copysign(1.0, cast(list[float], payload["gammas"])[0]) == -1.0
