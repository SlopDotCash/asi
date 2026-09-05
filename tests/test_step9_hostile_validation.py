"""Hostile validation for Step 9 guarded-dreaming facade."""

from fractions import Fraction
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.steps.step9 import (
    _STEP9_SEQUENCE_MAX_STEPS,
    Step9DreamingConfig,
    _require_step9_trusted_array,
    init_step9_state,
    make_step9_components,
    run_step9_scan,
    run_step9_smoke,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("HostileInt.__int__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.as_integer_ratio must not be called")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileFloat.__repr__ must not be called")


class _HostileTypeName(type):
    calls = 0

    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            _HostileTypeName.calls += 1
            raise AssertionError("metaclass __name__ hook must not be called")
        return super().__getattribute__(name)


class _HostileHiddenSizes(metaclass=_HostileTypeName):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("container iteration hook must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("container repr hook must not be called")


def test_rejects_string_subclass_for_observation_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step9DreamingConfig(observation_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_observation_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step9DreamingConfig(observation_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step9DreamingConfig(observation_dim=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step9DreamingConfig(observation_dim=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step9DreamingConfig(model_step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_plain_string_for_model_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step9DreamingConfig(model_gamma="0.99")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_model_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step9DreamingConfig(model_gamma=_StringSubclass("0.99"))  # type: ignore[arg-type]


def test_rejects_out_of_range_model_gamma_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step9DreamingConfig(model_gamma=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_hidden_sizes_non_tuple_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a tuple of integers") as exc:
        Step9DreamingConfig(model_hidden_sizes=[64])  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)


def test_rejects_hostile_hidden_sizes_without_metadata_hooks() -> None:
    _HostileTypeName.calls = 0
    _HostileHiddenSizes.calls = 0
    with pytest.raises(ValueError, match="must be a tuple of integers"):
        Step9DreamingConfig(model_hidden_sizes=_HostileHiddenSizes())  # type: ignore[arg-type]
    assert _HostileTypeName.calls == 0
    assert _HostileHiddenSizes.calls == 0


def test_rejects_model_error_decay_out_of_range_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\)") as exc:
        Step9DreamingConfig(model_error_decay=1.0)
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step9DreamingConfig(observation_dim=4, model_gamma=0.99)
    assert cfg.observation_dim == 4
    assert cfg.model_gamma == pytest.approx(0.99)
    cfg2 = Step9DreamingConfig(model_hidden_sizes=(32, 16), observation_dim=4)
    assert cfg2.model_hidden_sizes == (32, 16)


def test_numpy_scalars_pass() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=cast(Any, np.int32(4)),
        model_step_size=cast(Any, np.float32(0.05)),
        model_gamma=cast(Any, np.float64(0.9)),
    )
    assert cfg.observation_dim == 4
    cfg2 = Step9DreamingConfig(model_step_size=cast(Any, Fraction(1, 20)))
    assert cfg2.model_step_size == pytest.approx(0.05)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step9DreamingConfig(model_step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0


def test_from_dict_requires_exact_complete_and_nested_schema_without_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # pragma: no cover - must not run
            type(self).calls += 1
            raise AssertionError("mapping hook must not run")

    with pytest.raises(ValueError, match="exact dictionary"):
        Step9DreamingConfig.from_dict(cast(Any, HostileDict()))
    assert HostileDict.calls == 0

    payload = Step9DreamingConfig().to_dict()
    malformed = dict(payload)
    malformed["extra"] = 1
    with pytest.raises(ValueError, match="schema"):
        Step9DreamingConfig.from_dict(malformed)
    payload["control"] = HostileDict()
    with pytest.raises(ValueError, match="exact dictionary"):
        Step9DreamingConfig.from_dict(payload)
    assert HostileDict.calls == 0


def test_hidden_size_validation_is_bounded_before_iteration() -> None:
    with pytest.raises(ValueError, match="at most 4096"):
        Step9DreamingConfig(model_hidden_sizes=(1,) * 4_097)
    payload = Step9DreamingConfig().to_dict()
    payload["model_hidden_sizes"] = [1] * 4_097
    with pytest.raises(ValueError, match="at most 4096"):
        Step9DreamingConfig.from_dict(payload)


def test_derived_allocation_and_runtime_work_fail_at_config_boundary() -> None:
    with pytest.raises(ValueError, match="observation buffer bytes"):
        Step9DreamingConfig(buffer_capacity=536_870_912, observation_dim=1)
    with pytest.raises(ValueError, match="dream work per real step"):
        Step9DreamingConfig(
            planning_budget=65,
            dream_candidate_count=64,
            dream_rollout_horizon=1,
        )


def test_runtime_entry_points_require_exact_config_without_truthiness_hooks() -> None:
    calls = 0

    class HostileConfig:
        def __bool__(self) -> bool:  # pragma: no cover - must not run
            nonlocal calls
            calls += 1
            raise AssertionError("truthiness hook must not run")

    value = HostileConfig()
    with pytest.raises(TypeError, match="exact Step9DreamingConfig"):
        make_step9_components(cast(Any, value))
    with pytest.raises(TypeError, match="exact Step9DreamingConfig"):
        run_step9_smoke(cast(Any, value), steps=1)
    assert calls == 0


def test_smoke_preflights_complete_output_shape_before_allocation() -> None:
    with pytest.raises(ValueError, match="observation row count"):
        run_step9_smoke(steps=2**31 - 1)


# =============================================================================
# run_step9_scan sequence-length ceiling (hang guard)
# =============================================================================
#
# ``run_step9_scan`` hands ``rewards``/``next_observations`` straight to
# ``jax.lax.scan`` with no bound on the leading (step) axis. A hostile or
# mistaken caller supplying a huge array forces JAX to trace/compile a scan
# of that length, hanging the process well before any step executes -- the
# same hang class fixed this session for other scan-driven array loops in
# ``core/sarsa.py``, ``core/average_reward.py``, and
# ``core/horde_actor_critic.py``.


def _step9_scan_components(
    steps: int,
) -> tuple[Step9DreamingConfig, Any, Any, Any, Any, jax.Array, jax.Array]:
    cfg = Step9DreamingConfig(
        observation_dim=3,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=0,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent,
        model,
        buffer,
        key=jr.key(0),
        initial_observation=jnp.zeros((3,), dtype=jnp.float32),
    )
    rewards = jnp.zeros((steps,), dtype=jnp.float32)
    next_observations = jnp.zeros((steps, 3), dtype=jnp.float32)
    return cfg, agent, model, buffer, state, rewards, next_observations


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={first.shape[0]}")

    monkeypatch.setattr("alberta_framework.steps.step9.jax.lax.scan", spy)
    return seen


def test_step9_sequence_ceiling_is_documented() -> None:
    assert _STEP9_SEQUENCE_MAX_STEPS == 10_000


def test_step9_ceiling_length_array_still_passes_the_trusted_array_gate() -> None:
    rewards = jnp.zeros((_STEP9_SEQUENCE_MAX_STEPS,), dtype=jnp.float32)
    checked = _require_step9_trusted_array(
        "rewards", rewards, shape=(_STEP9_SEQUENCE_MAX_STEPS,), dtype=jnp.float32
    )
    assert checked.shape == (_STEP9_SEQUENCE_MAX_STEPS,)


def test_run_step9_scan_rejects_overflow_length_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    cfg, agent, model, buffer, state, rewards, next_observations = _step9_scan_components(
        _STEP9_SEQUENCE_MAX_STEPS + 1
    )
    with pytest.raises(ValueError, match=r"rewards length must be an integer in \[1, 10000\]"):
        run_step9_scan(cfg, agent, model, buffer, state, rewards, next_observations)
    assert seen == []


def test_run_step9_scan_rejects_origin_hang_class_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A far larger sequence length -- the actual hang class -- is also
    rejected before ``jax.lax.scan`` is ever called."""
    seen = _spy_scan(monkeypatch)
    cfg, agent, model, buffer, state, rewards, next_observations = _step9_scan_components(
        200_000
    )
    with pytest.raises(ValueError, match="rewards length must be an integer in"):
        run_step9_scan(cfg, agent, model, buffer, state, rewards, next_observations)
    assert seen == []


def test_run_step9_scan_rejects_mismatched_next_observations_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    cfg, agent, model, buffer, state, rewards, _next_observations = _step9_scan_components(5)
    mismatched = jnp.zeros((6, 3), dtype=jnp.float32)
    with pytest.raises(ValueError, match="next_observations must have shape"):
        run_step9_scan(cfg, agent, model, buffer, state, rewards, mismatched)
    assert seen == []


def test_run_step9_scan_rejects_non_config_type_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    _cfg, agent, model, buffer, state, rewards, next_observations = _step9_scan_components(5)
    with pytest.raises(TypeError, match="actual Step9DreamingConfig"):
        run_step9_scan(
            cast(Any, object()), agent, model, buffer, state, rewards, next_observations
        )
    assert seen == []


def test_run_step9_scan_rejects_wrong_dtype_rewards_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    cfg, agent, model, buffer, state, _rewards, next_observations = _step9_scan_components(5)
    int_rewards = jnp.zeros((5,), dtype=jnp.int32)
    with pytest.raises(TypeError, match="rewards must have dtype"):
        run_step9_scan(cfg, agent, model, buffer, state, int_rewards, next_observations)
    assert seen == []


class _HostileArray:
    """Duck-types ``.shape``/``.dtype`` but is not a trusted array type."""

    calls = 0

    @property
    def shape(self) -> tuple[int, ...]:
        type(self).calls += 1
        raise AssertionError("shape hook must not run")

    @property
    def dtype(self) -> np.dtype[Any]:
        type(self).calls += 1
        raise AssertionError("dtype hook must not run")


def test_run_step9_scan_rejects_hostile_rewards_without_touching_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    cfg, agent, model, buffer, state, _rewards, next_observations = _step9_scan_components(5)
    _HostileArray.calls = 0
    with pytest.raises(TypeError, match="rewards must be a trusted array"):
        run_step9_scan(
            cfg, agent, model, buffer, state, cast(Any, _HostileArray()), next_observations
        )
    assert _HostileArray.calls == 0
    assert seen == []


def test_run_step9_scan_accepts_a_small_in_bounds_sequence() -> None:
    cfg, agent, model, buffer, state, rewards, next_observations = _step9_scan_components(4)
    result = run_step9_scan(cfg, agent, model, buffer, state, rewards, next_observations)
    assert result.real_td_errors.shape == (4,)
