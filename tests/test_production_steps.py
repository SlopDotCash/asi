# mypy: disable-error-code="untyped-decorator,unused-ignore"
"""Production-facing Step 1/2 kernel tests."""

import json
import tomllib
from fractions import Fraction
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.cli import step1_smoke_main, step2_smoke_main
from alberta_framework.steps import (
    Step1KernelConfig,
    Step2HybridConfig,
    Step2KernelConfig,
    Step2MemoryConfig,
    Step2StrictDigitReadoutConfig,
    Step2TemporalContextConfig,
    make_step1_learner,
    make_step1_optimizer,
    make_step1_stream,
    make_step2_hybrid_learner,
    make_step2_learner,
    make_step2_memory_learner,
    make_step2_stream,
    make_step2_strict_digit_readout_learner,
    make_step2_temporal_context,
    make_step2_temporal_learner,
    run_step1_smoke,
    run_step2_smoke,
)

# The module is ~40 seconds serial on the supported development environment.
pytestmark = pytest.mark.slow

_INT32_MAX = int(np.iinfo(np.int32).max)


def test_step1_kernel_factory_and_smoke_are_finite() -> None:
    config = Step1KernelConfig(optimizer="autostep", normalizer="ema")
    learner = make_step1_learner(config)
    stream = make_step1_stream(config)
    state = learner.init(stream.feature_dim)

    prediction = learner.predict(state, jnp.zeros(stream.feature_dim))
    assert prediction.shape == (1,)

    result = run_step1_smoke(config, steps=16, final_window=4)
    assert result.finite
    assert result.metrics_shape == (16, 4)
    assert result.final_window_mse >= 0.0
    assert result.to_dict()["config"] == config.to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", True),
        ("steps", 4.0),
        ("final_window", True),
        ("final_window", 2.0),
    ],
)
def test_step1_smoke_rejects_non_integer_counts(field: str, value: object) -> None:
    kwargs: dict[str, object] = {"steps": 4, "final_window": 2, field: value}

    with pytest.raises(ValueError, match=field):
        run_step1_smoke(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["steps", "final_window"])
def test_step1_smoke_rejects_objects_that_only_spoof_integer_class(field: str) -> None:
    class IntegerClassSpoof:
        @property
        def __class__(self) -> type[int]:
            return int

        def __int__(self) -> int:
            return 2

    value = IntegerClassSpoof()
    assert isinstance(value, int)
    assert not issubclass(type(value), int)
    kwargs: dict[str, object] = {"steps": 4, "final_window": 2, field: value}

    with pytest.raises(ValueError, match=field):
        run_step1_smoke(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "optimizer",
    [
        "lms",
        "idbd",
        "autostep",
        "autostep_gtd",
        "adagain",
        "adam",
        "rmsprop",
        "nadaline",
    ],
)
def test_step1_kernel_all_public_optimizers_smoke(optimizer: str) -> None:
    config = Step1KernelConfig(
        optimizer=optimizer,  # type: ignore[arg-type]
        normalizer="ema",
        feature_dim=8,
        num_relevant=3,
        noise_std=0.1,
    )
    result = run_step1_smoke(config, steps=12, final_window=3)
    assert result.finite
    assert result.metrics_shape == (12, 4)


def test_step1_kernel_rejects_unpublished_auto_alias() -> None:
    with pytest.raises(ValueError, match="optimizer"):
        Step1KernelConfig(optimizer="auto")  # type: ignore[arg-type]


def test_step1_kernel_rejects_misspelled_adagain_alias() -> None:
    with pytest.raises(ValueError, match="optimizer"):
        Step1KernelConfig(optimizer="adagiven")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("optimizer", "LMS", "lms"),
        ("normalizer", "EMA", "ema"),
        ("stream", "ALBERTA", "alberta"),
    ],
)
def test_step1_kernel_canonicalizes_supported_enum_spellings(
    field: str,
    value: object,
    expected: str,
) -> None:
    config = Step1KernelConfig(**{field: value})

    assert getattr(config, field) == expected
    make_step1_learner(config)
    make_step1_stream(config)


_INVALID_STEP1_FIELDS: tuple[tuple[str, Any], ...] = (
    ("feature_dim", 0),
    ("feature_dim", -1),
    ("feature_dim", True),
    ("feature_dim", False),
    ("feature_dim", "20"),
    ("feature_dim", 20.5),
    ("feature_dim", float("nan")),
    ("feature_dim", float("inf")),
    ("feature_dim", None),
    ("num_relevant", 0),
    ("num_relevant", -1),
    ("num_relevant", True),
    ("num_relevant", False),
    ("num_relevant", "5"),
    ("num_relevant", 5.5),
    ("num_relevant", float("nan")),
    ("num_relevant", float("inf")),
    ("num_relevant", None),
    ("optimizer", "unknown_opt"),
    ("optimizer", 123),
    ("optimizer", None),
    ("normalizer", "unknown_norm"),
    ("normalizer", 123),
    ("normalizer", None),
    ("stream", "unknown_stream"),
    ("stream", 123),
    ("stream", None),
    ("step_size", float("nan")),
    ("step_size", float("inf")),
    ("step_size", float("-inf")),
    ("step_size", True),
    ("step_size", False),
    ("step_size", -1.0),
    ("step_size", "0.01"),
    ("step_size", None),
    ("meta_step_size", float("nan")),
    ("meta_step_size", float("inf")),
    ("meta_step_size", True),
    ("meta_step_size", False),
    ("meta_step_size", -0.01),
    ("drift_rate_w", float("nan")),
    ("drift_rate_w", float("inf")),
    ("drift_rate_w", True),
    ("drift_rate_w", False),
    ("drift_rate_w", -0.001),
    ("drift_rate_b", float("nan")),
    ("drift_rate_b", float("inf")),
    ("drift_rate_b", True),
    ("drift_rate_b", False),
    ("drift_rate_b", -0.001),
    ("noise_std", float("nan")),
    ("noise_std", float("inf")),
    ("noise_std", True),
    ("noise_std", False),
    ("noise_std", -1.0),
    ("feature_std", float("nan")),
    ("feature_std", float("inf")),
    ("feature_std", True),
    ("feature_std", False),
    ("feature_std", 0.0),
    ("feature_std", -1.0),
    ("feature_std", "1.0"),
    ("feature_std", None),
    ("ema_decay", float("nan")),
    ("ema_decay", float("inf")),
    ("ema_decay", True),
    ("ema_decay", False),
    ("ema_decay", -0.1),
    ("ema_decay", 1.1),
    ("ema_decay", "0.99"),
    ("streaming_batch_momentum", float("nan")),
    ("streaming_batch_momentum", float("inf")),
    ("streaming_batch_momentum", True),
    ("streaming_batch_momentum", False),
    ("streaming_batch_momentum", -0.1),
    ("streaming_batch_momentum", 1.1),
    ("streaming_batch_momentum", "0.99"),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP1_FIELDS)
def test_step1_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Step1KernelConfig(**{field: value})


def test_step1_num_relevant_exceeding_feature_dim_raises() -> None:
    with pytest.raises(ValueError, match="num_relevant"):
        Step1KernelConfig(feature_dim=4, num_relevant=5)


def test_step1_fields_preserve_legal_endpoints() -> None:
    config = Step1KernelConfig(
        feature_dim=1,
        num_relevant=1,
        optimizer="lms",
        normalizer="none",
        stream="alberta",
        step_size=0.0,
        meta_step_size=0.0,
        drift_rate_w=0.0,
        drift_rate_b=0.0,
        noise_std=0.0,
        feature_std=1e-12,
        ema_decay=0.0,
        streaming_batch_momentum=0.0,
    )
    make_step1_learner(config)
    stream = make_step1_stream(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step1KernelConfig.from_dict(payload)
    assert restored.feature_dim == 1
    assert restored.num_relevant == 1
    assert restored.step_size == 0.0
    assert restored.feature_std == 1e-12
    assert restored.ema_decay == 0.0
    assert restored.streaming_batch_momentum == 0.0
    assert stream.feature_dim == 1

    upper = Step1KernelConfig(
        feature_dim=10,
        num_relevant=10,
        ema_decay=1.0,
        streaming_batch_momentum=1.0,
    )
    make_step1_learner(upper)
    assert upper.ema_decay == 1.0
    assert upper.streaming_batch_momentum == 1.0


def test_step1_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.05)
    config = Step1KernelConfig(
        feature_dim=np.int64(10),
        num_relevant=np.int64(3),
        step_size=value,
        meta_step_size=value,
        drift_rate_w=value,
        drift_rate_b=value,
        noise_std=value,
        feature_std=np.float64(1.0),
        ema_decay=value,
        streaming_batch_momentum=value,
    )
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert config.feature_dim == 10
    assert config.num_relevant == 3
    assert config.step_size == 0.05
    assert type(payload["feature_dim"]) is int
    assert type(payload["num_relevant"]) is int
    assert type(payload["step_size"]) is float
    assert type(payload["meta_step_size"]) is float
    assert type(payload["drift_rate_w"]) is float
    assert type(payload["drift_rate_b"]) is float
    assert type(payload["noise_std"]) is float
    assert type(payload["feature_std"]) is float
    assert type(payload["ema_decay"]) is float
    assert type(payload["streaming_batch_momentum"]) is float


@pytest.mark.parametrize(
    "field",
    [
        "step_size",
        "meta_step_size",
        "drift_rate_w",
        "drift_rate_b",
        "noise_std",
        "feature_std",
    ],
)
def test_step1_float32_consumed_fields_reject_finite_wide_overflow(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        Step1KernelConfig(**{field: 1e100})


def test_step1_positive_float32_field_rejects_positive_to_zero_collapse() -> None:
    with pytest.raises(ValueError, match="feature_std"):
        Step1KernelConfig(feature_std=1e-50)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_size", Fraction(-1, 10**1000)),
        ("meta_step_size", Fraction(-1, 10**1000)),
        ("drift_rate_w", Fraction(-1, 10**1000)),
        ("drift_rate_b", Fraction(-1, 10**1000)),
        ("noise_std", Fraction(-1, 10**1000)),
        ("feature_std", Fraction(1, 10**1000)),
        ("ema_decay", Fraction(-1, 10**1000)),
        ("ema_decay", Fraction(10**1000 + 1, 10**1000)),
        ("streaming_batch_momentum", Fraction(-1, 10**1000)),
        (
            "streaming_batch_momentum",
            Fraction(10**1000 + 1, 10**1000),
        ),
    ],
)
def test_step1_fields_check_exact_domains_before_float_conversion(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        Step1KernelConfig(**{field: value})


def test_step1_fields_wrap_real_conversion_overflow_as_config_error() -> None:
    with pytest.raises(ValueError, match="step_size"):
        Step1KernelConfig(step_size=Fraction(10**1000, 1))


def test_step1_field_uses_direct_float32_narrowing_at_overflow_boundary() -> None:
    overflow_midpoint = np.ldexp(
        np.longdouble(2) - np.ldexp(np.longdouble(1), -24),
        127,
    )
    largest_finite_input = np.nextafter(
        overflow_midpoint,
        np.longdouble("-inf"),
    )

    config = Step1KernelConfig(optimizer="lms", step_size=largest_finite_input)
    optimizer = make_step1_optimizer(config)

    assert bool(np.isfinite(np.asarray(config.step_size, dtype=np.float32)))
    assert optimizer.to_config()["step_size"] == config.step_size


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-Fraction(1, 2**80), 1.0),
        (Fraction(0), 1.0),
        (
            Fraction(1, 2**80),
            float(np.nextafter(np.float32(1.0), np.float32(np.inf))),
        ),
    ],
)
def test_step1_fraction_midpoint_rounds_once_to_nearest_even(
    offset: Fraction,
    expected: float,
) -> None:
    midpoint = Fraction(1) + Fraction(1, 2**24)
    config = Step1KernelConfig(optimizer="lms", step_size=midpoint + offset)

    assert config.step_size == expected


def test_step1_fraction_float32_overflow_midpoint_is_exact() -> None:
    maximum = Fraction((2**24 - 1) * 2**104)
    overflow_midpoint = maximum + 2**103

    just_below = Step1KernelConfig(optimizer="lms", step_size=overflow_midpoint - 1)
    assert just_below.step_size == float(np.finfo(np.float32).max)
    with pytest.raises(ValueError, match="step_size"):
        Step1KernelConfig(optimizer="lms", step_size=overflow_midpoint)


def test_step2_kernel_factory_and_smoke_are_finite() -> None:
    config = Step2KernelConfig(feature_dim=4, n_heads=2, hidden_sizes=(8,))
    learner = make_step2_learner(config)
    state = learner.init(config.feature_dim, jr.key(0))

    prediction = learner.predict(state, jnp.zeros(config.feature_dim))
    assert prediction.shape == (2,)

    result = run_step2_smoke(config, steps=16, final_window=4)
    assert result.finite
    assert result.metrics_shape == (16, 4)
    assert result.final_window_mse >= 0.0
    assert result.learner_config["loss_normalization"] == "target_structure"
    assert result.to_dict()["config"] == config.to_dict()


def test_step2_strict_digit_readout_factory_exposes_promoted_branch() -> None:
    config = Step2StrictDigitReadoutConfig(n_heads=3, hidden_sizes=(8, 8))
    learner = make_step2_strict_digit_readout_learner(config)
    state = learner.init(feature_dim=4, key=jr.key(0))

    prediction = learner.predict(state, jnp.zeros(4))
    learner_config = learner.to_config()

    assert prediction.shape == (3,)
    assert learner_config["loss_normalization"] == "target_structure"
    assert learner_config["readout_mode"] == "two_timescale_simplex"
    assert learner_config["readout_fast_head_bounder_mode"] == "separate"
    assert config.to_dict()["hidden_sizes"] == [8, 8]


def test_step2_memory_factory_updates_fixed_budget_memory() -> None:
    config = Step2MemoryConfig(feature_dim=4, n_classes=3, slots_per_class=2)
    learner = make_step2_memory_learner(config)
    state = learner.init()

    prediction = learner.predict(state, jnp.zeros(config.feature_dim))
    assert prediction.shape == (3,)

    target = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    result = learner.update(state, jnp.ones(config.feature_dim), target)
    assert int(result.state.step_count) == 1
    assert int(jnp.sum(result.state.counts > 0.0)) == 1
    assert learner.config.to_config()["slots_per_class"] == 2
    assert config.to_dict()["n_classes"] == 3


_INVALID_STEP2_KERNEL_FIELDS: tuple[tuple[str, Any], ...] = (
    ("feature_dim", 0),
    ("feature_dim", -1),
    ("feature_dim", True),
    ("feature_dim", False),
    ("feature_dim", "8"),
    ("feature_dim", 8.5),
    ("feature_dim", float("nan")),
    ("feature_dim", float("inf")),
    ("feature_dim", None),
    ("feature_dim", _INT32_MAX + 1),
    ("n_heads", 0),
    ("n_heads", -1),
    ("n_heads", True),
    ("n_heads", False),
    ("n_heads", "3"),
    ("n_heads", None),
    ("n_heads", _INT32_MAX + 1),
    ("hidden_sizes", [32]),
    ("hidden_sizes", (0,)),
    ("hidden_sizes", (-1,)),
    ("hidden_sizes", (True,)),
    ("hidden_sizes", (32.5,)),
    ("hidden_sizes", (_INT32_MAX + 1,)),
    ("stream", "unknown_stream"),
    ("readout_mode", "unknown_mode"),
    ("step_size", float("nan")),
    ("step_size", float("inf")),
    ("step_size", float("-inf")),
    ("step_size", True),
    ("step_size", False),
    ("step_size", -1.0),
    ("step_size", "0.03"),
    ("step_size", None),
    ("loss_normalization", "invalid_norm"),
    ("context_length", 0),
    ("context_length", -1),
    ("context_length", True),
    ("context_length", False),
    ("context_length", "128"),
    ("context_length", None),
    ("context_length", _INT32_MAX + 1),
    ("noise_std", float("nan")),
    ("noise_std", float("inf")),
    ("noise_std", True),
    ("noise_std", False),
    ("noise_std", -1.0),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP2_KERNEL_FIELDS)
def test_step2_kernel_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Step2KernelConfig(**{field: value})


_INVALID_STEP2_MEMORY_FIELDS: tuple[tuple[str, Any], ...] = (
    ("feature_dim", 0),
    ("feature_dim", -1),
    ("feature_dim", True),
    ("feature_dim", False),
    ("feature_dim", "784"),
    ("feature_dim", None),
    ("feature_dim", _INT32_MAX + 1),
    ("n_classes", 1),
    ("n_classes", 0),
    ("n_classes", -1),
    ("n_classes", True),
    ("n_classes", False),
    ("n_classes", "10"),
    ("n_classes", None),
    ("n_classes", _INT32_MAX + 1),
    ("slots_per_class", 0),
    ("slots_per_class", -1),
    ("slots_per_class", True),
    ("slots_per_class", False),
    ("slots_per_class", "20"),
    ("slots_per_class", None),
    ("slots_per_class", _INT32_MAX + 1),
    ("update_rate", float("nan")),
    ("update_rate", float("inf")),
    ("update_rate", True),
    ("update_rate", False),
    ("update_rate", 0.0),
    ("update_rate", -0.1),
    ("update_rate", 1.1),
    ("novelty_threshold", float("nan")),
    ("novelty_threshold", float("inf")),
    ("novelty_threshold", True),
    ("novelty_threshold", False),
    ("novelty_threshold", -0.01),
    ("bandwidth", float("nan")),
    ("bandwidth", float("inf")),
    ("bandwidth", True),
    ("bandwidth", False),
    ("bandwidth", 0.0),
    ("bandwidth", -0.01),
    ("bandwidth", None),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_STEP2_MEMORY_FIELDS)
def test_step2_memory_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Step2MemoryConfig(**{field: value})


def test_step2_fields_preserve_legal_endpoints() -> None:
    k_cfg = Step2KernelConfig(
        feature_dim=1,
        n_heads=1,
        hidden_sizes=(),
        stream="frequency",
        step_size=0.0,
        context_length=1,
        noise_std=0.0,
    )
    make_step2_learner(k_cfg)
    payload = k_cfg.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step2KernelConfig.from_dict(payload)
    assert restored.feature_dim == 1
    assert restored.n_heads == 1
    assert restored.step_size == 0.0
    assert restored.noise_std == 0.0

    m_cfg = Step2MemoryConfig(
        feature_dim=1,
        n_classes=2,
        slots_per_class=1,
        update_rate=float(np.float32(1e-6)),
        novelty_threshold=0.0,
        bandwidth=float(np.float32(1e-12)),
    )
    make_step2_memory_learner(m_cfg)
    m_payload = m_cfg.to_dict()
    json.dumps(m_payload, allow_nan=False)
    m_restored = Step2MemoryConfig.from_dict(m_payload)
    assert m_restored.update_rate == float(np.float32(1e-6))
    assert m_restored.novelty_threshold == 0.0
    assert m_restored.bandwidth == float(np.float32(1e-12))

    m_upper = Step2MemoryConfig(
        update_rate=1.0,
    )
    assert m_upper.update_rate == 1.0


def test_step2_rejects_float32_overflow_and_underflow() -> None:
    with pytest.raises(ValueError, match="step_size"):
        Step2KernelConfig(step_size=1e100)
    with pytest.raises(ValueError, match="bandwidth"):
        Step2MemoryConfig(bandwidth=1e100)
    with pytest.raises(ValueError, match="bandwidth"):
        Step2MemoryConfig(bandwidth=1e-50)


def test_step2_fields_canonicalize_nonbuiltin_numbers() -> None:
    value = np.float64(0.03)
    k_cfg = Step2KernelConfig(
        feature_dim=np.int64(8),
        n_heads=np.int64(3),
        hidden_sizes=(np.int64(16),),
        step_size=value,
        context_length=np.int64(64),
        noise_std=value,
    )
    payload = k_cfg.to_dict()
    json.dumps(payload, allow_nan=False)
    assert k_cfg.feature_dim == 8
    assert k_cfg.n_heads == 3
    assert k_cfg.hidden_sizes == (16,)
    assert k_cfg.step_size == float(np.float32(0.03))
    assert type(payload["feature_dim"]) is int
    assert type(payload["n_heads"]) is int
    assert type(payload["hidden_sizes"][0]) is int
    assert type(payload["step_size"]) is float
    assert type(payload["context_length"]) is int
    assert type(payload["noise_std"]) is float


@pytest.mark.parametrize(
    ("config_type", "field", "value"),
    [
        (Step2KernelConfig, "step_size", Fraction(-1, 10**400)),
        (Step2KernelConfig, "noise_std", Fraction(-1, 10**400)),
        (Step2StrictDigitReadoutConfig, "step_size", Fraction(-1, 10**400)),
        (Step2MemoryConfig, "update_rate", Fraction(1, 10**400)),
        (Step2MemoryConfig, "update_rate", Fraction(10**400 + 1, 10**400)),
        (Step2MemoryConfig, "novelty_threshold", Fraction(-1, 10**400)),
        (Step2MemoryConfig, "bandwidth", Fraction(1, 10**400)),
    ],
)
def test_step2_configs_enforce_exact_scientific_domains(
    config_type: type[Any],
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        config_type(**{field: value})


def test_step2_configs_use_direct_float32_narrowing_without_double_rounding() -> None:
    overflow_midpoint = np.ldexp(
        np.longdouble(2) - np.ldexp(np.longdouble(1), -24),
        127,
    )
    largest_finite_input = np.nextafter(
        overflow_midpoint,
        np.longdouble("-inf"),
    )

    kernel = Step2KernelConfig(step_size=largest_finite_input)
    strict = Step2StrictDigitReadoutConfig(step_size=largest_finite_input)
    memory = Step2MemoryConfig(novelty_threshold=largest_finite_input)

    for value in (kernel.step_size, strict.step_size, memory.novelty_threshold):
        assert type(value) is float
        assert bool(np.isfinite(np.asarray(value, dtype=np.float32)))


@pytest.mark.parametrize(
    ("config_type", "field"),
    [
        (Step2KernelConfig, "step_size"),
        (Step2KernelConfig, "noise_std"),
        (Step2StrictDigitReadoutConfig, "step_size"),
        (Step2MemoryConfig, "update_rate"),
        (Step2MemoryConfig, "novelty_threshold"),
        (Step2MemoryConfig, "bandwidth"),
    ],
)
@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-Fraction(1, 2**61), 0.5),
        (Fraction(0), 0.5),
        (
            Fraction(1, 2**61),
            float(np.nextafter(np.float32(0.5), np.float32(1.0))),
        ),
    ],
    ids=("below", "tie-to-even", "above"),
)
def test_step2_configs_round_fraction_midpoints_once(
    config_type: type[Any],
    field: str,
    offset: Fraction,
    expected: float,
) -> None:
    midpoint = Fraction(1, 2) + Fraction(1, 2**25)
    config = config_type(**{field: midpoint + offset})
    assert getattr(config, field) == expected


@pytest.mark.parametrize(
    ("config_type", "field"),
    [
        (Step2KernelConfig, "step_size"),
        (Step2KernelConfig, "noise_std"),
        (Step2StrictDigitReadoutConfig, "step_size"),
        (Step2MemoryConfig, "novelty_threshold"),
        (Step2MemoryConfig, "bandwidth"),
    ],
)
def test_step2_configs_apply_exact_float32_overflow_midpoint(
    config_type: type[Any],
    field: str,
) -> None:
    float32_max = (2**24 - 1) * 2**104
    overflow_midpoint = Fraction(float32_max + 2**103)

    config = config_type(**{field: overflow_midpoint - 1})
    assert getattr(config, field) == float(np.finfo(np.float32).max)
    with pytest.raises(ValueError, match=field):
        config_type(**{field: overflow_midpoint})


@pytest.mark.parametrize(
    ("config_type", "field", "allows_zero"),
    [
        (Step2KernelConfig, "step_size", True),
        (Step2KernelConfig, "noise_std", True),
        (Step2StrictDigitReadoutConfig, "step_size", True),
        (Step2MemoryConfig, "update_rate", False),
        (Step2MemoryConfig, "novelty_threshold", True),
        (Step2MemoryConfig, "bandwidth", False),
    ],
)
def test_step2_configs_apply_exact_subnormal_midpoint(
    config_type: type[Any],
    field: str,
    allows_zero: bool,
) -> None:
    subnormal_midpoint = Fraction(1, 2**150)
    if allows_zero:
        config = config_type(**{field: subnormal_midpoint})
        assert getattr(config, field) == 0.0
    else:
        with pytest.raises(ValueError, match=field):
            config_type(**{field: subnormal_midpoint})

    above = config_type(**{field: subnormal_midpoint + Fraction(1, 2**200)})
    assert getattr(above, field) == float(
        np.nextafter(np.float32(0.0), np.float32(1.0))
    )


@pytest.mark.parametrize(
    ("config_type", "field", "allows_zero"),
    [
        (Step2KernelConfig, "step_size", True),
        (Step2KernelConfig, "noise_std", True),
        (Step2StrictDigitReadoutConfig, "step_size", True),
        (Step2MemoryConfig, "update_rate", False),
        (Step2MemoryConfig, "novelty_threshold", True),
        (Step2MemoryConfig, "bandwidth", False),
    ],
)
def test_step2_configs_preserve_signed_zero_domain_policy(
    config_type: type[Any],
    field: str,
    allows_zero: bool,
) -> None:
    if allows_zero:
        config = config_type(**{field: -0.0})
        assert np.signbit(getattr(config, field))
    else:
        with pytest.raises(ValueError, match=field):
            config_type(**{field: -0.0})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_heads", 0),
        ("n_heads", -1),
        ("n_heads", True),
        ("n_heads", False),
        ("n_heads", "10"),
        ("n_heads", 10.5),
        ("n_heads", None),
        ("n_heads", _INT32_MAX + 1),
        ("hidden_sizes", [8]),
        ("hidden_sizes", (0,)),
        ("hidden_sizes", (-1,)),
        ("hidden_sizes", (False,)),
        ("hidden_sizes", (8.5,)),
        ("hidden_sizes", (_INT32_MAX + 1,)),
        ("step_size", float("nan")),
        ("step_size", float("inf")),
        ("step_size", float("-inf")),
        ("step_size", True),
        ("step_size", False),
        ("step_size", -1.0),
        ("step_size", 3.5e38),
        ("step_size", "0.018"),
        ("step_size", None),
        ("step_size", jnp.asarray(0.25)),
    ],
)
def test_step2_strict_digit_config_rejects_malformed_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        Step2StrictDigitReadoutConfig(**{field: value})


@pytest.mark.parametrize(
    ("config_type", "field"),
    [
        (Step2KernelConfig, "step_size"),
        (Step2KernelConfig, "noise_std"),
        (Step2StrictDigitReadoutConfig, "step_size"),
        (Step2MemoryConfig, "update_rate"),
        (Step2MemoryConfig, "novelty_threshold"),
        (Step2MemoryConfig, "bandwidth"),
    ],
)
def test_step2_real_conversion_overflow_is_a_value_error(
    config_type: type[Any],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        config_type(**{field: Fraction(10**1000, 1)})


def test_step2_integer_fields_accept_int32_maximum() -> None:
    kernel = Step2KernelConfig(
        feature_dim=_INT32_MAX,
        n_heads=_INT32_MAX,
        hidden_sizes=(_INT32_MAX,),
        stream="frequency",
        context_length=_INT32_MAX,
    )
    strict = Step2StrictDigitReadoutConfig(
        n_heads=_INT32_MAX,
        hidden_sizes=(_INT32_MAX,),
    )
    memory = Step2MemoryConfig(
        feature_dim=_INT32_MAX,
        n_classes=_INT32_MAX,
        slots_per_class=_INT32_MAX,
    )

    assert kernel.feature_dim == _INT32_MAX
    assert kernel.n_heads == _INT32_MAX
    assert kernel.hidden_sizes == (_INT32_MAX,)
    assert kernel.context_length == _INT32_MAX
    assert strict.n_heads == _INT32_MAX
    assert strict.hidden_sizes == (_INT32_MAX,)
    assert memory.feature_dim == _INT32_MAX
    assert memory.n_classes == _INT32_MAX
    assert memory.slots_per_class == _INT32_MAX


def test_step2_context_length_int32_maximum_reaches_stream_step() -> None:
    stream = make_step2_stream(Step2KernelConfig(context_length=_INT32_MAX))
    state = stream.init(jr.key(0))

    timestep, next_state = stream.step(state, jnp.asarray(0, dtype=jnp.int32))

    assert timestep.observation.shape == (8,)
    assert int(next_state.step_count) == 1


def test_step2_strict_digit_config_canonical_json_roundtrip() -> None:
    config = Step2StrictDigitReadoutConfig(
        n_heads=np.int64(3),
        hidden_sizes=(np.int64(8), np.int64(16)),
        step_size=np.float64(0.018),
    )

    payload = json.loads(json.dumps(config.to_dict(), allow_nan=False))
    restored = Step2StrictDigitReadoutConfig.from_dict(payload)

    assert type(config.n_heads) is int
    assert all(type(size) is int for size in config.hidden_sizes)
    assert type(config.step_size) is float
    assert payload == {
        "n_heads": 3,
        "hidden_sizes": [8, 16],
        "step_size": float(np.float32(0.018)),
    }
    assert restored == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream", "POLYNOMIAL"),
        ("readout_mode", "LINEAR_MSE"),
        ("loss_normalization", "TARGET_STRUCTURE"),
    ],
)
def test_step2_kernel_rejects_case_mismatched_names(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        Step2KernelConfig(**{field: value})


def test_step2_kernel_rejects_polynomial_dimensions_without_a_triple() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        Step2KernelConfig(feature_dim=2, stream="polynomial")


def test_step2_builtin_float_defaults_remain_serialization_compatible() -> None:
    assert Step2KernelConfig().to_dict() == {
        "feature_dim": 8,
        "n_heads": 3,
        "hidden_sizes": [32],
        "stream": "polynomial",
        "readout_mode": "linear_mse",
        "step_size": 0.03,
        "loss_normalization": "target_structure",
        "context_length": 128,
        "noise_std": 0.05,
    }
    assert Step2StrictDigitReadoutConfig().to_dict()["step_size"] == 0.018
    assert Step2MemoryConfig().to_dict() == {
        "feature_dim": 784,
        "n_classes": 10,
        "slots_per_class": 20,
        "update_rate": 0.3,
        "novelty_threshold": 0.08,
        "bandwidth": 0.01,
    }


@pytest.mark.parametrize(
    "config",
    [
        Step2KernelConfig(),
        Step2StrictDigitReadoutConfig(),
        Step2MemoryConfig(),
    ],
)
def test_step2_config_from_dict_requires_exact_keys(config: Any) -> None:
    payload = config.to_dict()
    missing = dict(payload)
    missing.pop(next(iter(missing)))
    extra = {**payload, "unexpected": 1}

    for malformed in (missing, extra):
        with pytest.raises(ValueError, match=type(config).__name__):
            type(config).from_dict(malformed)


def test_step2_hybrid_factory_updates_upgd_and_memory() -> None:
    config = Step2HybridConfig(
        feature_dim=4,
        n_heads=3,
        hidden_sizes=(8,),
        upgd_head_repetition_multiplier=2.0,
        upgd_head_repetition_warmup_steps=4,
        target_trace_blend_scale=0.2,
    )
    learner = make_step2_hybrid_learner(config)
    state = learner.init(jr.key(0))

    prediction = learner.predict(state, jnp.zeros(config.feature_dim))
    assert prediction.shape == (3,)

    target = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    result = learner.update(state, jnp.ones(config.feature_dim), target)
    assert int(result.state.upgd_state.step_count) == 1
    assert int(result.state.memory_state.step_count) == 1
    assert int(jnp.sum(result.state.memory_state.counts > 0.0)) == 1
    assert learner.config.to_dict()["slots_per_class"] == config.slots_per_class
    upgd_config = learner.upgd.to_config()
    assert upgd_config["head_repetition_multiplier"] == 2.0
    assert upgd_config["head_repetition_warmup_steps"] == 4
    assert learner.config.target_trace_blend_scale == 0.2


def test_step2_hybrid_default_is_promoted_trace_variant() -> None:
    config = Step2HybridConfig()
    learner = make_step2_hybrid_learner(config)

    assert config.initial_memory_logit == 0.0
    assert config.target_trace_blend_scale == 0.8
    assert config.target_trace_pressure_threshold == 0.5
    assert learner.config.initial_memory_logit == 0.0
    assert learner.config.target_trace_blend_scale == 0.8
    assert learner.config.target_trace_pressure_threshold == 0.5


def test_step2_temporal_context_factory_matches_learner_input() -> None:
    config = Step2TemporalContextConfig(feature_dim=4, n_heads=2, hidden_sizes=(8,))
    featurizer = make_step2_temporal_context(config)
    learner = make_step2_temporal_learner(config)
    context_state = featurizer.init()
    context_state, features = featurizer.step(
        context_state,
        jnp.ones(config.feature_dim),
    )
    learner_state = learner.init(features.shape[0], jr.key(0))

    prediction = learner.predict(learner_state, features)

    assert prediction.shape == (2,)
    assert featurizer.config.include_phase_products
    assert not featurizer.config.include_ema
    assert not featurizer.config.include_delta
    assert config.to_dict()["periods"] == list(config.periods)


def test_step_facade_configs_json_roundtrip() -> None:
    configs = [
        Step1KernelConfig(),
        Step2KernelConfig(),
        Step2StrictDigitReadoutConfig(),
        Step2MemoryConfig(),
        Step2HybridConfig(),
        Step2TemporalContextConfig(),
    ]

    for config in configs:
        payload = json.loads(json.dumps(config.to_dict()))
        rebuilt = type(config).from_dict(payload)
        assert rebuilt == config


def test_cli_smoke_entrypoints_return_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert step1_smoke_main(["--steps", "8", "--final-window", "2"]) == 0
    assert '"finite": true' in capsys.readouterr().out

    assert step2_smoke_main(["--steps", "8", "--final-window", "2"]) == 0
    assert '"finite": true' in capsys.readouterr().out


def test_documented_cli_scripts_are_packaged() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject.read_text())
    scripts = payload["project"]["scripts"]

    assert scripts["alberta-step1-smoke"] == "alberta_framework.cli:step1_smoke_main"
    assert scripts["alberta-step2-smoke"] == "alberta_framework.cli:step2_smoke_main"
