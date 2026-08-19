"""Protocol step/seed ceilings for public learning-loop scans.

Documented last-fit is README / ``__init__`` / loop-docstring ``num_steps=10_000``
and batched ``30`` seeds. Origin handed ``10**12`` and ``2**31-1`` to
``jnp.arange`` with no reject — that is the hang/OOM class, not an INT32 leftover.
"""

from __future__ import annotations

from typing import Any, cast

import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.learners import (
    _LEARNING_LOOP_MAX_SEED_STEPS,
    _LEARNING_LOOP_MAX_SEEDS,
    _LEARNING_LOOP_MAX_STEPS,
    LinearLearner,
    _require_learning_loop_keys,
    _require_learning_loop_seed_steps,
    _require_learning_loop_steps,
    run_learning_loop,
    run_learning_loop_batched,
    run_mlp_learning_loop,
    run_mlp_learning_loop_batched,
    run_td_learning_loop,
    run_true_online_td_loop,
)
from alberta_framework.streams.synthetic import RandomWalkStream


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __int__(self) -> int:  # pragma: no cover
        raise AssertionError("int hook executed")


class _HostileKeys:
    calls = 0

    @property
    def shape(self) -> Any:
        type(self).calls += 1
        raise AssertionError("shape hook executed")

    @property
    def ndim(self) -> Any:
        type(self).calls += 1
        raise AssertionError("ndim hook executed")


def _spy_arange(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []

    def spy(*args: object, **kwargs: object) -> Any:
        seen.append((args, kwargs))
        raise AssertionError(f"jnp.arange must not run: {args} {kwargs}")

    monkeypatch.setattr("alberta_framework.core.learners.jnp.arange", spy)
    return seen


def test_documented_protocol_ceilings_match_public_examples() -> None:
    assert _LEARNING_LOOP_MAX_STEPS == 10_000
    assert _LEARNING_LOOP_MAX_SEEDS == 30
    assert _LEARNING_LOOP_MAX_SEED_STEPS == 30 * 10_000


def test_last_fit_protocol_step_count_is_accepted() -> None:
    assert _require_learning_loop_steps("num_steps", _LEARNING_LOOP_MAX_STEPS) == 10_000


def test_first_overflow_protocol_step_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    stream = RandomWalkStream(feature_dim=5)
    learner = LinearLearner()
    with pytest.raises(ValueError, match=r"num_steps must be an integer in \[1, 10000\]"):
        run_learning_loop(learner, stream, _LEARNING_LOOP_MAX_STEPS + 1, jr.key(0))
    assert seen == []


@pytest.mark.parametrize(
    "fn",
    [
        run_learning_loop,
        run_mlp_learning_loop,
        run_td_learning_loop,
        run_true_online_td_loop,
    ],
)
def test_trillion_steps_rejected_before_arange(
    fn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        fn(cast(Any, object()), cast(Any, object()), 10**12, cast(Any, object()))
    assert seen == []


def test_int32_max_steps_rejected_before_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_learning_loop(
            cast(Any, object()),
            cast(Any, object()),
            2**31 - 1,
            cast(Any, object()),
        )
    assert seen == []


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 10_000.0])
def test_rejects_non_exact_or_non_positive_step_counts(value: object) -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_learning_loop_steps("num_steps", value)


def test_rejects_numpy_and_subclass_step_counts_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_learning_loop_steps("num_steps", np.int64(10))
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_learning_loop_steps("num_steps", _HostileInt(10))


def test_batched_last_fit_seed_count_is_accepted() -> None:
    keys = jr.split(jr.key(0), _LEARNING_LOOP_MAX_SEEDS)
    assert _require_learning_loop_keys(keys) == 30
    _require_learning_loop_seed_steps(_LEARNING_LOOP_MAX_STEPS, 30)


def test_batched_first_overflow_seed_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_arange(monkeypatch)
    keys = jr.split(jr.key(0), _LEARNING_LOOP_MAX_SEEDS + 1)
    stream = RandomWalkStream(feature_dim=5)
    learner = LinearLearner()
    with pytest.raises(ValueError, match="seed count"):
        run_learning_loop_batched(learner, stream, 1, keys)
    assert seen == []


def test_batched_trillion_steps_rejected_before_keys_or_arange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_learning_loop_batched(
            cast(Any, object()),
            cast(Any, object()),
            10**12,
            cast(Any, object()),
        )
    assert seen == []


def test_mlp_batched_trillion_steps_rejected_before_arange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_mlp_learning_loop_batched(
            cast(Any, object()),
            cast(Any, object()),
            10**12,
            cast(Any, object()),
        )
    assert seen == []


def test_batched_keys_exact_gate_does_not_read_hostile_shape() -> None:
    _HostileKeys.calls = 0
    with pytest.raises(TypeError, match="keys must be a JAX array"):
        _require_learning_loop_keys(_HostileKeys())
    assert _HostileKeys.calls == 0


def test_seed_step_product_rejects_combined_overflow() -> None:
    with pytest.raises(ValueError, match="seed-steps exceed the documented protocol budget"):
        _require_learning_loop_seed_steps(10_000, 31)
