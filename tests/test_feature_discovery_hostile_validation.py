"""Hostile validation for streams/feature_discovery sink gates."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.feature_discovery import (
    InteractionFeatureDiscoveryStream,
    NonlinearFeatureDiscoveryStream,
)

_INT32_MAX = 2**31 - 1


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat hook must not leak via !r")


class _HostileInt(int):
    calls = 0

    @property
    def __class__(self) -> type[int]:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__class__ must not be called")

    def __int__(self) -> int:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__int__ must not be called")

    def __index__(self) -> int:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


def test_hostile_int_traps_every_documented_coercion_hook() -> None:
    """Guard against a second definition shadowing the stronger traps.

    A duplicate ``_HostileInt`` used to shadow this class while trapping only
    ``__int__`` and ``__repr__``, so the ``__class__`` and ``__index__`` traps
    never ran -- and ``__index__`` is precisely the hook the house integer
    gates reach for.
    """
    for hook in ("__class__", "__int__", "__index__", "__repr__"):
        assert hook in vars(_HostileInt), hook

    _HostileInt.calls = 0
    with pytest.raises(AssertionError, match=r"HostileInt\.__int__"):
        int(_HostileInt(4))
    assert _HostileInt.calls == 1


def test_rejects_string_subclass_for_feature_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_feature_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        NonlinearFeatureDiscoveryStream(feature_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_rejects_bool_and_hostile_int_for_feature_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=np.bool_(True))  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer"):
        NonlinearFeatureDiscoveryStream(feature_dim=_HostileInt(4))
    assert _HostileInt.calls == 0


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (NonlinearFeatureDiscoveryStream, "feature_dim"),
        (NonlinearFeatureDiscoveryStream, "n_tasks"),
        (NonlinearFeatureDiscoveryStream, "n_latents"),
        (NonlinearFeatureDiscoveryStream, "n_contexts"),
        (NonlinearFeatureDiscoveryStream, "context_length"),
        (NonlinearFeatureDiscoveryStream, "active_latents_per_context"),
        (InteractionFeatureDiscoveryStream, "feature_dim"),
        (InteractionFeatureDiscoveryStream, "n_tasks"),
        (InteractionFeatureDiscoveryStream, "n_contexts"),
        (InteractionFeatureDiscoveryStream, "context_length"),
        (InteractionFeatureDiscoveryStream, "active_pairs_per_context"),
    ],
)
def test_every_integer_field_rejects_subclass_without_hooks(
    factory: type[NonlinearFeatureDiscoveryStream] | type[InteractionFeatureDiscoveryStream],
    field: str,
) -> None:
    _HostileInt.calls = 0
    kwargs: dict[str, object] = {field: _HostileInt(4)}
    if field != "feature_dim":
        kwargs["feature_dim"] = 4
    with pytest.raises(ValueError, match=field):
        factory(**kwargs)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must narrow to a finite float32") as exc:
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=_HostileFloat(1.0))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)


def test_rejects_plain_string_for_feature_std() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std="1.0")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_positive_real() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=_StringSubclass("1.0"))  # type: ignore[arg-type]


def test_rejects_nonpositive_feature_std_without_repr() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=0.0)
    with pytest.raises(ValueError, match="must be a real number") as exc:
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=_StringSubclass("bad"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc.value)


def test_rejects_include_squares_non_bool_without_repr() -> None:
    with pytest.raises(TypeError, match="must be a boolean") as exc:
        InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=_StringSubclass("true"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil = _EvilStr("true")
    with pytest.raises(TypeError, match="must be a boolean"):
        InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=evil)  # type: ignore[arg-type]


def test_valid_configs_still_pass() -> None:
    s = NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=1.0, linear_scale=0.05)
    assert s.feature_dim == 4
    s2 = InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=True)
    assert s2.include_squares is True
    s3 = InteractionFeatureDiscoveryStream(feature_dim=4, include_squares=np.bool_(False))
    assert s3.include_squares is False


def test_numpy_scalars_pass() -> None:
    s = NonlinearFeatureDiscoveryStream(feature_dim=np.int32(4), feature_std=np.float32(1.0))
    assert s.feature_dim == 4
    assert s.feature_std == pytest.approx(1.0)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (3, 4)

    with pytest.raises(ValueError, match="must narrow to a finite float32"):
        NonlinearFeatureDiscoveryStream(feature_dim=4, feature_std=RatioFloat(0.5))  # type: ignore[arg-type]


def test_derived_nonlinear_resources_fail_before_allocation() -> None:
    with pytest.raises(ValueError, match="context weight count"):
        NonlinearFeatureDiscoveryStream(
            feature_dim=2,
            n_tasks=50_000,
            n_latents=50_000,
            n_contexts=2,
        )
    with pytest.raises(ValueError, match="latent weight bytes"):
        NonlinearFeatureDiscoveryStream(
            feature_dim=20_000,
            n_tasks=1,
            n_latents=50_000,
            n_contexts=1,
        )


@pytest.mark.parametrize(
    ("feature_dim", "include_squares"),
    [(65_536, True), (65_537, False)],
)
def test_interaction_pair_count_fails_before_python_pair_construction(
    feature_dim: int, include_squares: bool
) -> None:
    with pytest.raises(ValueError, match="pair count"):
        InteractionFeatureDiscoveryStream(
            feature_dim=feature_dim,
            include_squares=include_squares,
        )


def test_interaction_derived_context_resources_fail_before_allocation() -> None:
    with pytest.raises(ValueError, match="context weight count"):
        InteractionFeatureDiscoveryStream(
            feature_dim=100,
            n_tasks=50_000,
            n_contexts=50_000,
        )


def test_collect_rejects_hostile_duck_stream_without_attribute_hooks() -> None:
    calls = 0

    class HostileStream:
        @property
        def __class__(self) -> type[NonlinearFeatureDiscoveryStream]:  # pragma: no cover
            nonlocal calls
            calls += 1
            raise AssertionError("class hook")

        def __getattribute__(self, name: str) -> object:  # pragma: no cover
            if name != "__class__":
                nonlocal calls
                calls += 1
                raise AssertionError("attribute hook")
            return super().__getattribute__(name)

    from alberta_framework.streams.feature_discovery import collect_feature_discovery_stream

    with pytest.raises(TypeError, match="exact feature-discovery stream"):
        collect_feature_discovery_stream(HostileStream(), 1, jr.key(0))
    assert calls == 0


def test_collect_preflights_output_resource_total() -> None:
    from alberta_framework.streams.feature_discovery import collect_feature_discovery_stream

    stream = NonlinearFeatureDiscoveryStream(feature_dim=4, n_tasks=1)
    with pytest.raises(ValueError, match="output (count|bytes)"):
        collect_feature_discovery_stream(stream, _INT32_MAX, jr.key(0))


# =============================================================================
# collect_feature_discovery_stream scan-length ceiling (hang guard)
# =============================================================================
#
# ``collect_feature_discovery_stream`` hands ``num_steps`` straight to
# ``jnp.arange``/``jax.lax.scan`` bounded only by ``_INT32_MAX``. A caller
# supplying a long-but-narrow stream (small row width) can pass the int32
# overflow preflight above while still forcing JAX to trace/compile a scan of
# hundreds of millions of steps, hanging the process well before any step
# executes -- the same hang class fixed this session for other scan-driven
# array/loop runners in ``steps/step9.py``, ``streams/gauntlet.py``,
# ``core/sarsa.py``, ``core/average_reward.py``, and
# ``core/horde_actor_critic.py``.


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(int(xs.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={xs.shape[0]}")

    monkeypatch.setattr("alberta_framework.streams.feature_discovery.jax.lax.scan", spy)
    return seen


def test_feature_discovery_loop_ceiling_is_documented() -> None:
    from alberta_framework.streams.feature_discovery import (
        _FEATURE_DISCOVERY_LOOP_MAX_STEPS,
    )

    assert _FEATURE_DISCOVERY_LOOP_MAX_STEPS == 10_000


def test_collect_rejects_narrow_overflow_length_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-but-narrow request that would clear the int32 resource
    preflight is still rejected before ``jax.lax.scan`` ever runs."""
    from alberta_framework.streams.feature_discovery import (
        _FEATURE_DISCOVERY_LOOP_MAX_STEPS,
        collect_feature_discovery_stream,
    )

    seen = _spy_scan(monkeypatch)
    stream = NonlinearFeatureDiscoveryStream(feature_dim=1, n_tasks=1, n_latents=1)
    with pytest.raises(
        ValueError, match=rf"num_steps must be <= {_FEATURE_DISCOVERY_LOOP_MAX_STEPS}"
    ):
        collect_feature_discovery_stream(
            stream, _FEATURE_DISCOVERY_LOOP_MAX_STEPS + 1, jr.key(0)
        )
    assert seen == []


def test_collect_rejects_origin_hang_class_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A far larger narrow request -- the actual hang class, still well
    under the int32 resource preflight -- is also rejected before scan."""
    from alberta_framework.streams.feature_discovery import collect_feature_discovery_stream

    seen = _spy_scan(monkeypatch)
    stream = NonlinearFeatureDiscoveryStream(feature_dim=1, n_tasks=1, n_latents=1)
    with pytest.raises(ValueError, match="num_steps must be <="):
        collect_feature_discovery_stream(stream, 50_000_000, jr.key(0))
    assert seen == []


def test_collect_ceiling_length_still_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from alberta_framework.streams.feature_discovery import (
        _FEATURE_DISCOVERY_LOOP_MAX_STEPS,
        collect_feature_discovery_stream,
    )

    stream = NonlinearFeatureDiscoveryStream(feature_dim=2, n_tasks=1, n_latents=1)
    observations, targets = collect_feature_discovery_stream(
        stream, _FEATURE_DISCOVERY_LOOP_MAX_STEPS, jr.key(0)
    )
    assert observations.shape[0] == _FEATURE_DISCOVERY_LOOP_MAX_STEPS
    assert targets.shape[0] == _FEATURE_DISCOVERY_LOOP_MAX_STEPS


@pytest.mark.parametrize(
    "factory", [NonlinearFeatureDiscoveryStream, InteractionFeatureDiscoveryStream]
)
def test_init_requires_scalar_typed_threefry_key(
    factory: type[NonlinearFeatureDiscoveryStream] | type[InteractionFeatureDiscoveryStream],
) -> None:
    stream = factory(feature_dim=4)
    for bad in (jnp.asarray([0, 1], dtype=jnp.uint32), jnp.asarray(0, dtype=jnp.int32)):
        with pytest.raises(TypeError, match="Threefry"):
            stream.init(bad)


@pytest.mark.parametrize(
    "factory", [NonlinearFeatureDiscoveryStream, InteractionFeatureDiscoveryStream]
)
def test_eager_and_jit_transactions_match_and_validate_static_contracts(
    factory: type[NonlinearFeatureDiscoveryStream] | type[InteractionFeatureDiscoveryStream],
) -> None:
    stream = factory(feature_dim=4, n_tasks=2, n_contexts=2)
    state = stream.init(jr.key(7))
    idx = jnp.asarray(0, dtype=jnp.int32)
    eager_timestep, eager_state = stream.step(state, idx)
    jit_timestep, jit_state = jax.jit(stream.step)(state, idx)
    np.testing.assert_array_equal(eager_timestep.observation, jit_timestep.observation)
    np.testing.assert_array_equal(eager_timestep.target, jit_timestep.target)
    np.testing.assert_array_equal(eager_state.step_count, jit_state.step_count)

    with pytest.raises(ValueError, match="idx"):
        stream.step(state, jnp.asarray(0.0, dtype=jnp.float32))


def test_step_rejects_wrong_state_array_before_jax_computation() -> None:
    stream = NonlinearFeatureDiscoveryStream(feature_dim=4, n_latents=3)
    state = stream.init(jr.key(0)).replace(
        latent_weights=jnp.zeros((3, 4), dtype=jnp.int32)
    )
    with pytest.raises(ValueError, match="latent_weights"):
        stream.step(state, jnp.asarray(0, dtype=jnp.int32))
