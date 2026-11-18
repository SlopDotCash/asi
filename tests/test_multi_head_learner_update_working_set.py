"""#1383-complete update working-set preflight for the multi-head MLP learner."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.baseline_optimizers import Adam, RMSprop
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    _direct_state_bytes,
    _direct_state_scalars,
    _multi_head_update_working_set_bytes,
    _preflight_multi_head_update_working_set,
    _validate_direct_state_resources,
    measure_multi_head_mlp_state_nbytes,
)
from alberta_framework.core.normalizers import (
    EMANormalizer,
    EMANormalizerState,
    StreamingBatchNormalizer,
    WelfordNormalizer,
)
from alberta_framework.core.optimizers import (
    IDBD,
    LMS,
    Autostep,
    ObGD,
    ObGDBounding,
    Optimizer,
)
from alberta_framework.core.types import LMSState

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_HIDDEN = 30_000_000
_LAST_FIT_HIDDEN = 25_565_279
_FIRST_OVERFLOW_HIDDEN = 25_565_280
_OVERFLOW_FEATURE_DIM = 90_000_000
_LAST_FIT_FEATURE_DIM = 89_478_479
_FIRST_OVERFLOW_FEATURE_DIM = 89_478_480


def _unit_persist_bytes(hidden: int) -> int:
    return _direct_state_bytes(1, (hidden,), 1)


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _unit_persist_bytes(_OVERFLOW_HIDDEN)
    working_set_bytes = _multi_head_update_working_set_bytes(
        1, (_OVERFLOW_HIDDEN,), 1
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 840_000_036
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_HIDDEN <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        MultiHeadMLPLearner(n_heads=1, hidden_sizes=(_OVERFLOW_HIDDEN,))


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for hidden in range(_LAST_FIT_HIDDEN, _FIRST_OVERFLOW_HIDDEN + 2):
        persist_bytes = _unit_persist_bytes(hidden)
        working_set_bytes = _multi_head_update_working_set_bytes(1, (hidden,), 1)
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * hidden <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = hidden
        elif first_overflow is None:
            first_overflow = hidden
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_HIDDEN
    learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(last_fit,))
    assert learner._hidden_sizes == (last_fit,)
    with pytest.raises(ValueError, match="update working set byte count"):
        MultiHeadMLPLearner(n_heads=1, hidden_sizes=(first_overflow,))


def test_init_rejects_linear_feature_dim_working_set_before_allocation() -> None:
    persist_bytes = _direct_state_bytes(1, (), _OVERFLOW_FEATURE_DIM)
    working_set_bytes = _multi_head_update_working_set_bytes(
        1, (), _OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 720_000_028
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=())
    with pytest.raises(ValueError, match="update working set byte count"):
        learner.init(_OVERFLOW_FEATURE_DIM, jr.key(0))


def test_init_last_fit_and_first_overflow_feature_dims_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(
        _LAST_FIT_FEATURE_DIM, _FIRST_OVERFLOW_FEATURE_DIM + 2
    ):
        persist_bytes = _direct_state_bytes(1, (), feature_dim)
        working_set_bytes = _multi_head_update_working_set_bytes(
            1, (), feature_dim
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * feature_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = feature_dim
        elif first_overflow is None:
            first_overflow = feature_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_FEATURE_DIM
    _preflight_multi_head_update_working_set(1, (), last_fit)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_multi_head_update_working_set(1, (), first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_multi_head_update_working_set(1, (_OVERFLOW_HIDDEN,), 1)


def test_persist_bound_still_fires_before_working_set() -> None:
    with pytest.raises(ValueError, match="direct_state_bytes"):
        MultiHeadMLPLearner(n_heads=134_217_728, hidden_sizes=())


def test_legal_small_multi_head_learner_still_constructs() -> None:
    persist_bytes = _direct_state_bytes(2, (4,), 3)
    assert persist_bytes == 260
    learner = MultiHeadMLPLearner(
        n_heads=2, hidden_sizes=(4,), sparsity=0.0, use_layer_norm=False
    )
    state = learner.init(feature_dim=3, key=jr.key(0))
    learner.update(
        state,
        jnp.zeros((3,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )


def test_named_persist_matches_measured_state_bytes() -> None:
    # The named persist count must cover every JAX leaf init allocates,
    # including the per-(weight, bias) LMS step-size scalars: two per trunk
    # layer and two per head. Otherwise the int32 persist/working-set
    # preflight accepts states whose true bytes exceed the bound.
    for n_heads, hidden_sizes, feature_dim in (
        (2, (4,), 3),
        (1, (), 3),
        (3, (8, 4), 5),
    ):
        learner = MultiHeadMLPLearner(
            n_heads=n_heads,
            hidden_sizes=hidden_sizes,
            sparsity=0.0,
            use_layer_norm=False,
        )
        state = learner.init(feature_dim=feature_dim, key=jr.key(0))
        assert measure_multi_head_mlp_state_nbytes(state) == _direct_state_bytes(
            n_heads, hidden_sizes, feature_dim
        )


@pytest.mark.parametrize(
    ("optimizer", "head_optimizer", "normalizer"),
    [
        (None, None, None),
        (Autostep(), None, None),
        (IDBD(), None, None),
        (None, Autostep(), None),
        (Autostep(), IDBD(), None),
        (None, None, EMANormalizer()),
        (None, None, WelfordNormalizer()),
        (None, None, StreamingBatchNormalizer()),
        (Autostep(), None, EMANormalizer()),
    ],
)
def test_named_persist_matches_measured_state_for_configured_optimizer(
    optimizer: Any,
    head_optimizer: Any,
    normalizer: Any,
) -> None:
    """The named persist count must cover the configured optimizer and normalizer.

    Charging one scalar per optimizer state is only true for LMS: Autostep
    keeps three per-element arrays and two scalars per parameter, IDBD two
    per-element arrays and one scalar, and a feature normalizer adds two or
    three feature-width arrays of its own. Before the fix the named count
    undercounted every non-default configuration, so the int32 persist and
    working-set preflights admitted states larger than the bound.
    """
    n_heads, hidden_sizes, feature_dim = 2, (4, 5), 6
    learner = MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        optimizer=optimizer,
        head_optimizer=head_optimizer,
        normalizer=normalizer,
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(feature_dim=feature_dim, key=jr.key(0))
    assert measure_multi_head_mlp_state_nbytes(state) == _direct_state_bytes(
        n_heads,
        hidden_sizes,
        feature_dim,
        optimizer=optimizer,
        head_optimizer=head_optimizer,
        normalizer=normalizer,
    )


def test_preflight_counts_configured_optimizer_and_normalizer() -> None:
    """A configuration the LMS-only formula admitted must now be rejected.

    At feature_dim 40,000,000 the LMS working set is 960,000,129 bytes, but
    an Autostep trunk names 2,400,000,189 and overflows signed int32.
    """
    feature_dim = 40_000_000
    assert _multi_head_update_working_set_bytes(1, (), feature_dim) <= _INT32_MAX
    assert (
        _multi_head_update_working_set_bytes(1, (), feature_dim, optimizer=Autostep())
        > _INT32_MAX
    )
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_multi_head_update_working_set(
            1, (), feature_dim, optimizer=Autostep()
        )

    learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), optimizer=Autostep())
    with pytest.raises(ValueError, match="update working set byte count"):
        learner.init(feature_dim, jr.key(0))


def test_direct_state_bound_counts_configured_optimizer_and_normalizer() -> None:
    """The persist bound itself must count the configured optimizer state.

    At feature_dim 110,000,000 the LMS persist count is 880,000,028 bytes
    while an Autostep trunk names 2,200,000,048. IDBD and an EMA normalizer
    fit at 134,217,725 and overflow at 134,217,726, where LMS still names
    1,073,741,836.
    """
    _validate_direct_state_resources(1, (), 110_000_000)
    assert _direct_state_bytes(1, (), 110_000_000) <= _INT32_MAX
    with pytest.raises(ValueError, match="direct_state_bytes"):
        _validate_direct_state_resources(1, (), 110_000_000, optimizer=Autostep())

    for configured in (
        {"optimizer": IDBD()},
        {"normalizer": EMANormalizer()},
    ):
        _validate_direct_state_resources(1, (), 134_217_725, **configured)
        assert _direct_state_bytes(1, (), 134_217_726) <= _INT32_MAX
        with pytest.raises(ValueError, match="direct_state_bytes"):
            _validate_direct_state_resources(1, (), 134_217_726, **configured)


def test_bounder_is_not_an_optimizer_and_owns_no_persistent_state() -> None:
    """A bounder is a separate ``bounder=`` argument, not optimizer state.

    ``Bounder`` is an abstract base with only ``bound()`` and ``to_config()``.
    It has no ``init_for_shape`` and no ``supported_for_mlp``, it takes a
    ``kappa`` rather than an optimizer, and its concrete subclasses keep only
    Python floats, so nothing of it reaches the JAX state the persist count
    measures.
    """
    bounder = ObGDBounding()
    assert not hasattr(bounder, "init_for_shape")
    assert not hasattr(bounder, "supported_for_mlp")

    plain = MultiHeadMLPLearner(
        n_heads=1, hidden_sizes=(4,), sparsity=0.0, use_layer_norm=False
    )
    bounded = MultiHeadMLPLearner(
        n_heads=1,
        hidden_sizes=(4,),
        bounder=ObGDBounding(),
        sparsity=0.0,
        use_layer_norm=False,
    )
    plain_state = plain.init(feature_dim=3, key=jr.key(0))
    bounded_state = bounded.init(feature_dim=3, key=jr.key(0))
    assert measure_multi_head_mlp_state_nbytes(bounded_state) == (
        measure_multi_head_mlp_state_nbytes(plain_state)
    )
    assert _direct_state_bytes(1, (4,), 3) == 212

    # A bounder has no shape-generic state, so the counting helper refuses it
    # rather than charging a guessed shape.
    with pytest.raises(ValueError, match="no shape-generic state"):
        _direct_state_bytes(1, (), 3, optimizer=ObGDBounding())


def test_optimizer_without_a_known_shape_generic_state_fails_closed() -> None:
    """An optimizer whose state this module cannot size must be refused.

    ``ObGD`` keeps its own state but does not override ``init_for_shape``, so
    it is rejected at construction by the capability gate. The counting
    helper refuses it too, so a future caller cannot reach the persist count
    with an optimizer it would otherwise charge a guessed shape.
    """
    with pytest.raises(ValueError, match="optimizer.*shape-generic"):
        _direct_state_bytes(1, (), 3, optimizer=ObGD())

    with pytest.raises(ValueError, match="does not support the MLP"):
        MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), optimizer=ObGD())


def test_every_shape_generic_optimizer_is_counted_from_its_real_state() -> None:
    """The count follows the state each optimizer builds, not a class table.

    ``init_for_shape`` exists on LMS, IDBD, Autostep, Adam, RMSprop and
    AdamO, and the probe decomposition reproduces each one's fields. Adam and
    RMSprop live in ``baseline_optimizers`` and were missed by an
    implementation that enumerated only ``optimizers.py`` classes.
    """
    cases = [
        (LMS(step_size=0.1), 0, 1),
        (IDBD(), 2, 1),
        (Autostep(), 3, 2),
        (Adam(step_size=0.1), 2, 5),
        (RMSprop(), 1, 3),
    ]
    for optimizer, per_element, fixed in cases:
        for size in (1, 7, 64):
            # One linear head of ``size`` weights plus a bias, two copies of
            # params and traces, the optimizer state, and the three counters.
            expected = 2 * (size + 1) + 3 + per_element * size + fixed
            assert (
                _direct_state_scalars(1, (), size, optimizer=optimizer) == expected
            )

    for optimizer in (Adam(step_size=0.2, beta1=0.0, beta2=0.0), RMSprop()):
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(4,),
            optimizer=LMS(step_size=0.1),
            head_optimizer=optimizer,
            sparsity=0.0,
            use_layer_norm=False,
        )
        state = learner.init(feature_dim=3, key=jr.key(0))
        assert measure_multi_head_mlp_state_nbytes(state) == _direct_state_bytes(
            2, (4,), 3, optimizer=LMS(step_size=0.1), head_optimizer=optimizer
        )


def test_state_that_is_not_affine_in_width_fails_closed() -> None:
    """A state builder that is not affine cannot be extrapolated.

    The probe decomposition fits a line through three widths and refuses a
    builder that does not lie on it, so a future optimizer with a width
    dependent state shape cannot be silently miscounted.
    """

    class RaggedOptimizer(Optimizer[LMSState]):
        def init_for_shape(self, shape: tuple[int, ...]) -> Any:
            width = shape[0]
            return jnp.zeros(width + (width % 2), dtype=jnp.float32)

        def update_from_gradient(self, state: Any, gradient: Any, error: Any = None) -> Any:
            return gradient, state

    with pytest.raises(ValueError, match="not affine in the parameter width"):
        _direct_state_bytes(1, (), 3, optimizer=RaggedOptimizer())


def test_normalizer_that_keeps_its_parent_state_keeps_its_parent_count() -> None:
    """A normalizer subclass that returns the parent state is counted exactly.

    The count is derived from the state ``init`` returns, so a subclass that
    only changes configuration keeps the parent's field count instead of
    falling into an unknown branch.
    """

    class RenamedEMA(EMANormalizer):
        def init(self, feature_dim: int) -> EMANormalizerState:
            return super().init(feature_dim)

    assert _direct_state_bytes(1, (), 3, normalizer=RenamedEMA()) == (
        _direct_state_bytes(1, (), 3, normalizer=EMANormalizer())
    )
    learner = MultiHeadMLPLearner(
        n_heads=1, hidden_sizes=(), normalizer=RenamedEMA(), sparsity=0.0
    )
    state = learner.init(feature_dim=3, key=jr.key(0))
    assert measure_multi_head_mlp_state_nbytes(state) == _direct_state_bytes(
        1, (), 3, normalizer=RenamedEMA()
    )


def test_inherited_shape_generic_hooks_keep_their_parent_count() -> None:
    """A subclass that only changes scalars keeps its parent state shape.

    The dispatch follows the ``init_for_shape`` implementation the class
    actually uses, which is the same structural rule ``supported_for_mlp``
    applies. An LMS subclass therefore keeps the LMS count instead of
    falling into an unknown branch.
    """

    class ScaledLMS(LMS):
        pass

    class ScaledAutostep(Autostep):
        pass

    assert _direct_state_bytes(1, (4,), 3, optimizer=ScaledLMS(step_size=0.5)) == (
        _direct_state_bytes(1, (4,), 3, optimizer=LMS(step_size=0.5))
    )
    assert _direct_state_bytes(
        2, (4, 5), 6, optimizer=ScaledAutostep()
    ) == _direct_state_bytes(2, (4, 5), 6, optimizer=Autostep())

    learner = MultiHeadMLPLearner(
        n_heads=2,
        hidden_sizes=(4,),
        optimizer=ScaledLMS(step_size=0.5),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(feature_dim=3, key=jr.key(0))
    assert measure_multi_head_mlp_state_nbytes(state) == _direct_state_bytes(
        2, (4,), 3, optimizer=LMS(step_size=0.5)
    )

