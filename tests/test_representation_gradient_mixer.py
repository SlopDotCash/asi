"""Contract tests for the stateless two-source representation-gradient mixer."""

from __future__ import annotations

import dataclasses
import json
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.representation_gradient_mixer import (
    GradientMixMode,
    RepresentationGradientMixerConfig,
    RepresentationGradientMixResult,
    mix_representation_gradients,
)

pytestmark = pytest.mark.unit


def _assert_tree_allclose(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert str(first_tree) == str(second_tree)
    assert len(first_leaves) == len(second_leaves)
    for left, right in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-6, atol=1e-7)


def test_config_roundtrip_is_exact_strict_and_json_compatible() -> None:
    config = RepresentationGradientMixerConfig(
        representation_dim=7,
        mode="world_only",
        behavior_weight=0.25,
        grounded_world_weight=2.0,
        behavior_normalization="unit_l2",
        grounded_world_normalization="none",
        normalization_epsilon=1e-7,
        behavior_clip_norm=3.0,
        grounded_world_clip_norm=4.0,
        final_clip_norm=5.0,
    )
    payload = config.to_config()
    assert RepresentationGradientMixerConfig.from_config(payload) == config
    assert RepresentationGradientMixerConfig.from_config(
        cast(dict[str, object], json.loads(json.dumps(payload)))
    ) == config
    assert payload == {
        "schema": "alberta.representation-gradient-mixer.config.v1",
        "type": "RepresentationGradientMixerConfig",
        "representation_dim": 7,
        "mode": "world_only",
        "behavior_weight": 0.25,
        "grounded_world_weight": 2.0,
        "behavior_normalization": "unit_l2",
        "grounded_world_normalization": "none",
        "normalization_epsilon": 1e-7,
        "behavior_clip_norm": 3.0,
        "grounded_world_clip_norm": 4.0,
        "final_clip_norm": 5.0,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.mode = "full"  # type: ignore[misc]

    for malformed in (
        {**payload, "unknown": True},
        {key: value for key, value in payload.items() if key != "mode"},
        {**payload, "schema": "wrong"},
        {**payload, "type": "wrong"},
    ):
        with pytest.raises(ValueError):
            RepresentationGradientMixerConfig.from_config(malformed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"representation_dim": True},
        {"representation_dim": 0},
        {"representation_dim": 2, "mode": "adaptive"},
        {"representation_dim": 2, "behavior_weight": -1.0},
        {"representation_dim": 2, "grounded_world_weight": float("inf")},
        {"representation_dim": 2, "behavior_normalization": "batch"},
        {"representation_dim": 2, "grounded_world_normalization": "unit"},
        {"representation_dim": 2, "normalization_epsilon": 0.0},
        {"representation_dim": 2, "behavior_clip_norm": -1.0},
        {"representation_dim": 2, "grounded_world_clip_norm": float("nan")},
        {"representation_dim": 2, "final_clip_norm": 0.0},
    ],
)
def test_config_rejects_nonfinite_negative_or_unknown_rules(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        RepresentationGradientMixerConfig(**kwargs)  # type: ignore[arg-type]


class _SpoofedFloat:
    """Mimics ``float`` via ``__class__`` to defeat ``isinstance`` checks."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return 0.5


@pytest.mark.parametrize(
    "field",
    ["behavior_weight", "grounded_world_weight", "normalization_epsilon"],
)
def test_config_rejects_class_spoofed_float_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RepresentationGradientMixerConfig(
            representation_dim=2,
            **{field: _SpoofedFloat()},  # type: ignore[arg-type]
        )


def test_full_mode_matches_disclosed_weighted_algebra() -> None:
    config = RepresentationGradientMixerConfig(
        representation_dim=3,
        behavior_weight=0.25,
        grounded_world_weight=2.0,
    )
    behavior = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    world = jnp.asarray([-1.0, 4.0, 1.0], dtype=jnp.float32)
    result = mix_representation_gradients(config, behavior, world)
    expected_behavior = 0.25 * behavior
    expected_world = 2.0 * world
    expected = expected_behavior + expected_world

    np.testing.assert_allclose(result.gradient, expected)
    assert bool(result.valid)
    assert bool(result.applied)
    assert not bool(result.rejected)
    assert not bool(result.zero_output)
    diagnostics = result.diagnostics
    np.testing.assert_allclose(diagnostics.behavior_raw_norm, jnp.linalg.norm(behavior))
    np.testing.assert_allclose(
        diagnostics.grounded_world_raw_norm,
        jnp.linalg.norm(world),
    )
    np.testing.assert_allclose(
        diagnostics.behavior_used_norm,
        jnp.linalg.norm(expected_behavior),
    )
    np.testing.assert_allclose(
        diagnostics.grounded_world_used_norm,
        jnp.linalg.norm(expected_world),
    )
    np.testing.assert_allclose(
        diagnostics.dot_product,
        jnp.vdot(expected_behavior, expected_world),
    )
    np.testing.assert_allclose(
        diagnostics.cosine_similarity,
        jnp.vdot(expected_behavior, expected_world)
        / (jnp.linalg.norm(expected_behavior) * jnp.linalg.norm(expected_world)),
    )
    np.testing.assert_allclose(diagnostics.behavior_weight, 0.25)
    np.testing.assert_allclose(diagnostics.grounded_world_weight, 2.0)
    np.testing.assert_allclose(diagnostics.behavior_effective_weight, 0.25)
    np.testing.assert_allclose(diagnostics.grounded_world_effective_weight, 2.0)
    assert bool(diagnostics.behavior_active)
    assert bool(diagnostics.grounded_world_active)


def test_per_source_unit_normalization_weights_and_zero_are_exact() -> None:
    config = RepresentationGradientMixerConfig(
        representation_dim=2,
        behavior_weight=2.0,
        grounded_world_weight=3.0,
        behavior_normalization="unit_l2",
        grounded_world_normalization="unit_l2",
        normalization_epsilon=1e-6,
    )
    result = mix_representation_gradients(
        config,
        jnp.asarray([3.0, 4.0]),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    np.testing.assert_allclose(result.gradient, jnp.asarray([1.2, 1.6]), rtol=1e-6)
    np.testing.assert_allclose(result.diagnostics.behavior_used_norm, 2.0, rtol=1e-6)
    np.testing.assert_array_equal(result.diagnostics.grounded_world_used_norm, 0.0)
    assert bool(result.applied)
    assert not bool(result.zero_output)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(result))


def test_per_source_and_final_global_norm_clips_are_applied_in_order() -> None:
    config = RepresentationGradientMixerConfig(
        representation_dim=2,
        behavior_clip_norm=2.0,
        grounded_world_clip_norm=1.0,
        final_clip_norm=1.0,
    )
    result = mix_representation_gradients(
        config,
        jnp.asarray([3.0, 4.0]),
        jnp.asarray([0.0, 4.0]),
    )
    unclipped = jnp.asarray([1.2, 2.6])
    expected = unclipped / jnp.linalg.norm(unclipped)
    np.testing.assert_allclose(result.gradient, expected, rtol=1e-6)
    np.testing.assert_allclose(result.diagnostics.behavior_raw_norm, 5.0)
    np.testing.assert_allclose(result.diagnostics.grounded_world_raw_norm, 4.0)
    np.testing.assert_allclose(result.diagnostics.behavior_used_norm, 2.0, rtol=1e-6)
    np.testing.assert_allclose(
        result.diagnostics.grounded_world_used_norm,
        1.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.diagnostics.unclipped_mixed_norm,
        jnp.linalg.norm(unclipped),
        rtol=1e-6,
    )
    np.testing.assert_allclose(result.diagnostics.final_mixed_norm, 1.0, rtol=1e-6)


def test_conflicting_active_sources_report_negative_dot_and_cosine() -> None:
    config = RepresentationGradientMixerConfig(representation_dim=2)
    result = mix_representation_gradients(
        config,
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([-2.0, 0.0]),
    )
    np.testing.assert_allclose(result.diagnostics.dot_product, -2.0)
    np.testing.assert_allclose(result.diagnostics.cosine_similarity, -1.0)
    assert bool(result.diagnostics.conflict)

    tiny = mix_representation_gradients(
        config,
        jnp.asarray([1e-30, 0.0]),
        jnp.asarray([-1e-30, 0.0]),
    )
    np.testing.assert_allclose(tiny.diagnostics.cosine_similarity, -1.0)
    assert bool(tiny.diagnostics.conflict)


@pytest.mark.parametrize(
    ("mode", "expected", "behavior_active", "world_active", "applied"),
    [
        ("full", [4.0, 6.0], True, True, True),
        ("behavior_only", [1.0, 2.0], True, False, True),
        ("world_only", [3.0, 4.0], False, True, True),
        ("discard", [0.0, 0.0], False, False, False),
    ],
)
def test_modes_keep_the_same_two_inputs_but_apply_only_the_declared_sources(
    mode: GradientMixMode,
    expected: list[float],
    behavior_active: bool,
    world_active: bool,
    applied: bool,
) -> None:
    config = RepresentationGradientMixerConfig(
        representation_dim=2,
        mode=mode,
    )
    result = mix_representation_gradients(
        config,
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([3.0, 4.0]),
    )
    np.testing.assert_array_equal(result.gradient, jnp.asarray(expected))
    assert bool(result.valid)
    assert bool(result.applied) is applied
    assert not bool(result.rejected)
    assert bool(result.zero_output) is (mode == "discard")
    assert bool(result.diagnostics.behavior_active) is behavior_active
    assert bool(result.diagnostics.grounded_world_active) is world_active


@pytest.mark.parametrize(
    ("behavior", "world", "behavior_valid", "world_valid"),
    [
        ([1.0, 2.0], [3.0, 4.0], False, True),
        ([float("nan"), 2.0], [3.0, 4.0], True, True),
        ([1.0, 2.0], [float("inf"), 4.0], True, True),
    ],
)
def test_any_invalid_actively_weighted_source_atomically_rejects_to_exact_zero(
    behavior: list[float],
    world: list[float],
    behavior_valid: bool,
    world_valid: bool,
) -> None:
    result = mix_representation_gradients(
        RepresentationGradientMixerConfig(representation_dim=2),
        jnp.asarray(behavior),
        jnp.asarray(world),
        behavior_valid=jnp.asarray(behavior_valid),
        grounded_world_valid=jnp.asarray(world_valid),
    )
    np.testing.assert_array_equal(result.gradient, jnp.zeros((2,), dtype=jnp.float32))
    assert not bool(result.valid)
    assert not bool(result.applied)
    assert bool(result.rejected)
    assert bool(result.zero_output)
    assert bool(result.diagnostics.rejected)
    assert np.isfinite(np.asarray(result.diagnostics.final_mixed_norm)).all()


def test_masked_or_zero_weight_invalid_source_is_recorded_without_poisoning_output() -> None:
    invalid_world = jnp.asarray([jnp.nan, jnp.inf], dtype=jnp.float32)
    behavior = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
    for config in (
        RepresentationGradientMixerConfig(
            representation_dim=2,
            mode="behavior_only",
            grounded_world_weight=7.0,
        ),
        RepresentationGradientMixerConfig(
            representation_dim=2,
            mode="full",
            grounded_world_weight=0.0,
        ),
    ):
        result = mix_representation_gradients(
            config,
            behavior,
            invalid_world,
            grounded_world_valid=jnp.asarray(False),
        )
        np.testing.assert_array_equal(result.gradient, behavior)
        assert bool(result.valid)
        assert bool(result.applied)
        assert not bool(result.rejected)
        assert not bool(result.diagnostics.grounded_world_valid)
        assert not bool(result.diagnostics.grounded_world_active)
        assert np.isinf(np.asarray(result.diagnostics.grounded_world_raw_norm)).all()


def test_wrong_gradient_shape_or_dynamic_validity_predicate_fails_closed() -> None:
    config = RepresentationGradientMixerConfig(representation_dim=2)
    with pytest.raises(ValueError, match="shape"):
        mix_representation_gradients(config, jnp.ones((3,)), jnp.ones((2,)))
    with pytest.raises(ValueError, match="scalar boolean"):
        mix_representation_gradients(
            config,
            jnp.ones((2,)),
            jnp.ones((2,)),
            behavior_valid=jnp.asarray([True]),
        )
    with pytest.raises(TypeError, match="scalar boolean"):
        mix_representation_gradients(
            config,
            jnp.ones((2,)),
            jnp.ones((2,)),
            grounded_world_valid=jnp.asarray(1),
        )


def test_zero_and_extreme_finite_gradients_are_nan_free_and_scale_safe() -> None:
    normalized = RepresentationGradientMixerConfig(
        representation_dim=2,
        behavior_normalization="unit_l2",
        grounded_world_normalization="unit_l2",
        behavior_clip_norm=1.0,
        grounded_world_clip_norm=1.0,
        final_clip_norm=1.0,
    )
    zero = mix_representation_gradients(
        normalized,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(zero.gradient, jnp.zeros((2,), dtype=jnp.float32))
    assert bool(zero.valid)
    assert bool(zero.applied)
    assert bool(zero.zero_output)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(zero))

    huge = jnp.asarray([3e30, 4e30], dtype=jnp.float32)
    scale_safe = mix_representation_gradients(
        RepresentationGradientMixerConfig(
            representation_dim=2,
            mode="behavior_only",
            behavior_normalization="unit_l2",
            behavior_clip_norm=0.5,
            final_clip_norm=0.25,
        ),
        huge,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    np.testing.assert_allclose(scale_safe.gradient, jnp.asarray([0.15, 0.2]), rtol=1e-5)
    np.testing.assert_allclose(scale_safe.diagnostics.behavior_raw_norm, 5e30, rtol=1e-5)
    np.testing.assert_allclose(scale_safe.diagnostics.final_mixed_norm, 0.25, rtol=1e-5)
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(scale_safe)
    )

    near_float32_limit = mix_representation_gradients(
        RepresentationGradientMixerConfig(
            representation_dim=2,
            mode="behavior_only",
            behavior_normalization="unit_l2",
        ),
        jnp.asarray([3e38, 3e38], dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        near_float32_limit.gradient,
        jnp.asarray([2**-0.5, 2**-0.5]),
        rtol=1e-5,
    )
    assert np.isfinite(np.asarray(near_float32_limit.diagnostics.behavior_raw_norm))


def test_jit_matches_eager_and_dynamic_false_predicate_rejects() -> None:
    config = RepresentationGradientMixerConfig(
        representation_dim=3,
        behavior_weight=0.5,
        grounded_world_weight=1.5,
        behavior_normalization="unit_l2",
        grounded_world_clip_norm=2.0,
        final_clip_norm=1.0,
    )

    def run(
        behavior: Array,
        world: Array,
        behavior_valid: Array,
        world_valid: Array,
    ) -> RepresentationGradientMixResult:
        return mix_representation_gradients(
            config,
            behavior,
            world,
            behavior_valid=behavior_valid,
            grounded_world_valid=world_valid,
        )

    behavior = jnp.asarray([3.0, 4.0, 0.0], dtype=jnp.float32)
    world = jnp.asarray([-1.0, 2.0, 3.0], dtype=jnp.float32)
    eager = run(behavior, world, jnp.asarray(True), jnp.asarray(True))
    compiled = jax.jit(run)(behavior, world, jnp.asarray(True), jnp.asarray(True))
    _assert_tree_allclose(eager, compiled)

    rejected = jax.jit(run)(behavior, world, jnp.asarray(True), jnp.asarray(False))
    np.testing.assert_array_equal(
        rejected.gradient,
        jnp.zeros((3,), dtype=jnp.float32),
    )
    assert bool(rejected.rejected)
    assert not bool(rejected.valid)
