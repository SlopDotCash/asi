"""Complete aggregate preflight for Step 1 baseline optimizer state."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.baseline_optimizers import NADALINE, AdaGain, Adam, RMSprop

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_ONE_BANK_FITS = _INT32_MAX // 4
_WORKING_SET_OVERFLOW = 50_000_000


class _IntSubclass(int):
    pass


class _IndexOnly:
    def __index__(self) -> int:
        return 4


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_adam_one_moment_bank_fits_while_persistent_aggregate_does_not() -> None:
    one_bank_bytes = 4 * _ONE_BANK_FITS
    persistent_bytes = 4 * (2 * _ONE_BANK_FITS + 7)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="persistent state byte count"):
        Adam().init(_ONE_BANK_FITS)


def test_adagain_one_gain_bank_fits_while_persistent_aggregate_does_not() -> None:
    one_bank_bytes = 4 * _ONE_BANK_FITS
    persistent_bytes = 4 * (2 * _ONE_BANK_FITS + 4)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="persistent state byte count"):
        AdaGain().init(_ONE_BANK_FITS)


def test_adam_rejects_simultaneous_update_working_set() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (2 * _WORKING_SET_OVERFLOW + 7)
    contributor_update_bytes = 4 * (9 * _WORKING_SET_OVERFLOW + 8)
    update_bytes = 4 * (16 * _WORKING_SET_OVERFLOW + 14)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert contributor_update_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        Adam().init(_WORKING_SET_OVERFLOW)


def test_adagain_rejects_simultaneous_update_working_set() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (2 * _WORKING_SET_OVERFLOW + 4)
    contributor_update_bytes = 4 * (9 * _WORKING_SET_OVERFLOW + 8)
    update_bytes = 4 * (13 * _WORKING_SET_OVERFLOW + 8)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert contributor_update_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        AdaGain().init(_WORKING_SET_OVERFLOW)


def test_legal_adam_and_adagain_update_identity_is_unchanged() -> None:
    observation = jnp.ones(4, dtype=jnp.float32)
    error = jnp.asarray(0.5, dtype=jnp.float32)

    adam = Adam(step_size=0.01)
    adam_state = adam.init(4)
    assert adam_state.m.shape == (4,)
    assert adam_state.v.shape == (4,)
    assert 4 * (2 * 4 + 7) <= _INT32_MAX
    adam_result = adam.update(adam_state, error, observation)
    assert bool(adam_result.update_applied)
    assert adam_result.weight_delta.shape == (4,)
    assert adam_result.new_state.m.shape == (4,)
    assert float(adam_result.new_state.t) == pytest.approx(1.0)

    adagain = AdaGain()
    adagain_state = adagain.init(4)
    assert adagain_state.step_sizes.shape == (4,)
    assert adagain_state.gradient_trace.shape == (4,)
    adagain_result = adagain.update(adagain_state, error, observation)
    assert bool(adagain_result.update_applied)
    assert adagain_result.weight_delta.shape == (4,)
    assert adagain_result.new_state.gradient_trace.shape == (4,)


@pytest.mark.parametrize(
    ("factory", "vectors", "scalars", "label"),
    [
        (Adam, 16, 14, "Adam"),
        (AdaGain, 13, 8, "AdaGain"),
        (RMSprop, 9, 9, "RMSprop"),
        (NADALINE, 11, 12, "NADALINE"),
    ],
)
def test_every_linear_optimizer_rejects_first_overflowing_update_width(
    factory: type[Adam] | type[AdaGain] | type[RMSprop] | type[NADALINE],
    vectors: int,
    scalars: int,
    label: str,
) -> None:
    float32_limit = _INT32_MAX // 4
    first_overflowing = (float32_limit - scalars) // vectors + 1
    assert 4 * (vectors * (first_overflowing - 1) + scalars) <= _INT32_MAX
    assert 4 * (vectors * first_overflowing + scalars) > _INT32_MAX
    with pytest.raises(ValueError, match=rf"{label} update working set byte count"):
        factory().init(first_overflowing)


@pytest.mark.parametrize("factory", [Adam, AdaGain, RMSprop, NADALINE])
@pytest.mark.parametrize(
    "feature_dim",
    [True, 4.0, _IntSubclass(4), _IndexOnly(), jnp.asarray(4, dtype=jnp.int32)],
)
def test_linear_optimizer_feature_dim_rejects_hostile_integer_surrogates(
    factory: type[Adam] | type[AdaGain] | type[RMSprop] | type[NADALINE],
    feature_dim: object,
) -> None:
    with pytest.raises(ValueError, match="feature_dim must be an integer"):
        factory().init(feature_dim)  # type: ignore[arg-type]


@pytest.mark.parametrize("factory", [Adam, RMSprop])
@pytest.mark.parametrize(
    "shape",
    [
        [2, 2],
        (2, True),
        (2, 2.0),
        (2, _IntSubclass(2)),
        (2, _IndexOnly()),
        (2, jnp.asarray(2, dtype=jnp.int32)),
    ],
)
def test_parameter_optimizer_shape_rejects_hostile_schema(
    factory: type[Adam] | type[RMSprop], shape: object
) -> None:
    with pytest.raises(ValueError, match="shape"):
        factory().init_for_shape(shape)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "vectors", "scalars", "label"),
    [
        (Adam, 15, 8, "Adam"),
        (RMSprop, 8, 6, "RMSprop"),
    ],
)
def test_parameter_optimizer_rejects_derived_shape_working_set_before_allocation(
    factory: type[Adam] | type[RMSprop], vectors: int, scalars: int, label: str
) -> None:
    float32_limit = _INT32_MAX // 4
    first_overflowing = (float32_limit - scalars) // vectors + 1
    with pytest.raises(ValueError, match=rf"{label} update working set byte count"):
        factory().init_for_shape((first_overflowing,))


def test_actual_numpy_integer_dimensions_remain_supported() -> None:
    assert Adam().init(np.int64(4)).m.shape == (4,)
    assert RMSprop().init_for_shape((np.uint16(2), np.int32(2))).v.shape == (2, 2)
